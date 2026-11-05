import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from locodellm.__main__ import _compute_case_stats, main
from locodellm.bench import BenchPromptTest, ExpectedResult, PromptTest
from locodellm.bench.python_harness import PythonHarness, extract_python_calls
from locodellm.generate.repository import GitRepository
from locodellm.utils import extract_code, strip_thinking


class ScriptedSession:
    """Replays model turns while recording prompts and generation options."""

    def __init__(self, responses):
        self.responses = responses
        self.turns = []
        self.children = []
        self.text = ""
        self.generated_token_count = 0

    def new_session(self):
        """Creates an independent conversation with the same scripted turns."""
        child = ScriptedSession(self.responses)
        self.children.append(child)
        return child

    def generate(self, prompt, **options):
        """Emits the next scripted model response."""
        self.text = self.responses[len(self.turns)]
        self.turns.append((prompt, options))
        self.generated_token_count = 5


def create_repository(directory):
    """Creates a real repository with committed files for cleanup and diff tests."""
    directory.mkdir(exist_ok=True)
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(directory)], check=True, capture_output=True
    )
    for filename, content in {
        ".gitignore": "ignored.txt\n",
        "tracked.txt": "original\n",
        "removed.txt": "remove me\n",
        "tool_script.py": "Do not overwrite.",
        "tool_function_calls.bin": "Do not overwrite.",
    }.items():
        (directory / filename).write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(directory), "add", "."], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(directory),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            "Initial files",
        ],
        check=True,
        capture_output=True,
    )
    return GitRepository(str(directory))


class TestPythonHarness(unittest.TestCase):
    def test_generate_json_repository_and_independent_prompts(self):
        """Cleans each prompt and saves applicable patches including new binary files."""
        from locodellm.generate.generate_from_model import _session_cache

        model_id = "test/scripted-json-cli"
        key = (model_id, None, None, (), "{}")
        self.assertNotIn(key, _session_cache)
        session = ScriptedSession(
            [
                (
                    "<python>from pathlib import Path\n"
                    "assert Path('tracked.txt').read_text() == 'original\\n'\n"
                    "assert not Path('initial_untracked').exists()\n"
                    "assert not Path('initial_directory').exists()\n"
                    "assert not Path('value.txt').exists()\n"
                    "assert Path('ignored.txt').read_text() == 'keep'\n"
                    "Path('tracked.txt').write_text('modified\\n')\n"
                    "Path('removed.txt').unlink()\n"
                    "Path('binary.bin').write_bytes(b'\\x00\\xffpayload')\n"
                    "Path('non_utf8.txt').write_bytes(b'\\xff\\n')\n"
                    "path = Path('value.txt')\n"
                    "path.write_text(str(int(path.read_text()) + 1) if path.exists() else '1')\n"
                    "</python>"
                ),
                (
                    "<python>from pathlib import Path\nimport subprocess\n"
                    "subprocess.run(['git', 'add', 'tracked.txt'], check=True)\n"
                    "print(Path('value.txt').read_text())</python>"
                ),
                "Updated the repository.",
            ]
        )
        _session_cache[key] = session
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaisesRegex(RuntimeError, "Git rev-parse"):
                    GitRepository(str(root))
                repository = root / "repo"
                target = create_repository(repository)
                (repository / "tracked.txt").write_text("initial dirty content")
                target.git("add", "tracked.txt")
                (repository / "initial_untracked").write_text("discard")
                (repository / "initial_directory").mkdir()
                (repository / "initial_directory" / "file").write_text("discard")
                (repository / "ignored.txt").write_text("keep")
                source = root / "tasks.json"
                prompts = ["Make the first change.", "Make the second change."]
                source.write_text(
                    json.dumps({"repo": "repo", "prompts": prompts}), encoding="utf-8"
                )
                output = root / "events.json"
                with (
                    contextlib.redirect_stdout(io.StringIO()) as stdout,
                    contextlib.redirect_stderr(io.StringIO()) as stderr,
                ):
                    main(
                        [
                            "generate",
                            model_id,
                            "--input",
                            str(source),
                            "--max-tool-calls",
                            "2",
                            "--output",
                            str(output),
                            "--verbose",
                            "1",
                        ]
                    )
                document = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(document["repo"], str(repository))
                self.assertEqual(document["base_revision"], target.revision)
                self.assertEqual((repository / "value.txt").read_text(), "1")
                self.assertEqual((repository / "tracked.txt").read_text(), "modified\n")
                self.assertFalse((repository / "removed.txt").exists())
                for filename in ("tool_script.py", "tool_function_calls.bin"):
                    self.assertEqual(
                        (repository / filename).read_text(encoding="utf-8"), "Do not overwrite."
                    )
                self.assertEqual(stdout.getvalue(), "Updated the repository.\n" * 2)
                for phase in (
                    "[generate] prompt 1/2",
                    "[generate] prompt 2/2",
                    "[generate] resetting repository",
                    "[harness] executing script 1/2",
                    "[harness] script 2/2 finished: exit=0, timed_out=False",
                    "[generate] collecting repository diff",
                    "[generate] saving",
                ):
                    self.assertIn(phase, stderr.getvalue())
                results = document["results"]
                self.assertEqual(results[0]["diff"], results[1]["diff"])
                self.assertEqual([entry["prompt"] for entry in results], prompts)
                self.assertEqual([len(child.turns) for child in session.children], [3, 3])
                for index, (entry, child) in enumerate(zip(results, session.children)):
                    self.assertEqual(entry["tool_call_count"], 2)
                    self.assertIsNone(entry["error"])
                    self.assertIn(str(repository), child.turns[0][0])
                    self.assertTrue(child.turns[0][0].endswith(prompts[index]))
                    self.assertNotIn(prompts[1 - index], child.turns[0][0])
                    self.assertIn("Remaining Python tool calls: 0", child.turns[-1][0])
                    self.assertEqual(entry["tool_calls"][1]["stdout"], "1\n")
                    self.assertIn("-original\n+modified\n", entry["diff"])
                    self.assertIn("deleted file mode", entry["diff"])
                    self.assertIn("new file mode", entry["diff"])
                    self.assertIn("GIT binary patch", entry["diff"])
                    self.assertNotIn("ignored.txt", entry["diff"])
                    self.assertNotIn("initial dirty content", entry["diff"])
                    for call in entry["tool_calls"]:
                        self.assertEqual(call["cwd"], str(repository))
                        self.assertNotEqual(Path(call["command"][-1]).parent, repository)
                        self.assertFalse(Path(call["command"][-1]).exists())
                self.assertEqual(
                    target.git("diff", "--cached", "--name-only").strip(), "tracked.txt"
                )
                target.clean()
                subprocess.run(
                    ["git", "-C", str(repository), "apply", "--binary", "-"],
                    input=results[0]["diff"].encode("utf-8", errors="surrogateescape"),
                    check=True,
                    capture_output=True,
                )
                self.assertEqual((repository / "tracked.txt").read_text(), "modified\n")
                self.assertFalse((repository / "removed.txt").exists())
                self.assertEqual((repository / "value.txt").read_text(), "1")
                self.assertEqual((repository / "binary.bin").read_bytes(), b"\x00\xffpayload")
                self.assertEqual((repository / "non_utf8.txt").read_bytes(), b"\xff\n")
        finally:
            del _session_cache[key]

    def test_generate_json_exports_failure_before_stopping(self):
        """Preserves changes and failure events without starting later prompts."""
        from locodellm.generate.generate_from_model import _session_cache

        model_id = "test/scripted-json-failure"
        key = (model_id, None, None, (), "{}")
        self.assertNotIn(key, _session_cache)
        session = ScriptedSession(
            [
                (
                    "<python>from pathlib import Path\n"
                    "Path('changed.txt').write_text('kept')</python>"
                ),
                "<python>raise AssertionError('must not execute')</python>",
            ]
        )
        _session_cache[key] = session
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                repository = root / "repo"
                create_repository(repository)
                source = root / "tasks.json"
                source.write_text(
                    json.dumps({"repo": str(repository), "prompts": ["First.", "Second."]}),
                    encoding="utf-8",
                )
                output = root / "events.json"
                with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                    main(
                        [
                            "generate",
                            model_id,
                            "--input",
                            str(source),
                            "--max-tool-calls",
                            "1",
                            "-o",
                            str(output),
                        ]
                    )
                document = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(len(document["results"]), 1)
                self.assertIn("budget exhausted", document["results"][0]["error"])
                self.assertEqual(document["results"][0]["tool_call_count"], 1)
                self.assertIn("+kept", document["results"][0]["diff"])
                self.assertEqual(len(session.children), 1)
                self.assertEqual((repository / "changed.txt").read_text(), "kept")
        finally:
            del _session_cache[key]

    def test_repository_validation_before_cleanup(self):
        """Protects input files, output files, and repository parents from cleanup."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repo"
            target = create_repository(repository)
            (repository / "tracked.txt").write_text("must remain dirty")
            for source, output in [
                (repository / "tasks.json", root / "events.json"),
                (root / "tasks.json", repository / "events.json"),
            ]:
                source.write_text(
                    json.dumps({"repo": str(repository), "prompts": ["Prompt."]}),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "outside the target repository"):
                    main(["generate", "unused/model", "--input", str(source), "-o", str(output)])
                self.assertEqual((repository / "tracked.txt").read_text(), "must remain dirty")
            subdirectory = repository / "subdirectory"
            subdirectory.mkdir()
            with self.assertRaisesRegex(ValueError, "repository root"):
                GitRepository(str(subdirectory))
            target.git("checkout", "-b", "other")
            with self.assertRaisesRegex(RuntimeError, "branch changed"):
                target.clean()
            self.assertEqual((repository / "tracked.txt").read_text(), "must remain dirty")

    def test_repository_diff_without_changes(self):
        """Returns an empty patch without changing the real Git index."""
        with tempfile.TemporaryDirectory() as directory:
            target = create_repository(Path(directory))
            index = target.path / ".git" / "index"
            before = index.read_bytes()
            self.assertEqual(target.diff(), "")
            self.assertEqual(index.read_bytes(), before)

    def test_missing_working_directory(self):
        """Rejects a missing working directory before starting the model."""
        with tempfile.TemporaryDirectory() as directory:
            session = ScriptedSession([])
            with self.assertRaisesRegex(ValueError, "Working directory does not exist"):
                PythonHarness().run(session, "Problem.", 100, directory=f"{directory}/missing")
            self.assertEqual(session.turns, [])

    def test_generate_cli_events_and_configurable_budget(self):
        """Exports real file operations, all turns, and exhausted-budget errors."""
        from locodellm.generate.generate_from_model import _session_cache

        model_id = "test/scripted-harness-cli"
        key = (model_id, None, None, (), "{}")
        self.assertNotIn(key, _session_cache)
        write = "<python>from pathlib import Path\nPath('value.txt').write_text('42')</python>"
        read = "<python>from pathlib import Path\nprint(Path('value.txt').read_text())</python>"
        for harness, budget, responses, failed in [
            (True, 100, [write] + [read] * 99 + ["The value is 42."], False),
            (
                True,
                2,
                [write, read, "<python>raise AssertionError('not executed')</python>"],
                True,
            ),
            (False, 10, ["Plain answer."], False),
        ]:
            with self.subTest(harness=harness, budget=budget):
                session = ScriptedSession(responses)
                _session_cache[key] = session
                try:
                    with tempfile.TemporaryDirectory() as directory:
                        output = os.path.join(directory, "events.json")
                        args = ["generate", model_id, "Read and write a file.", "-o", output]
                        if harness:
                            args += ["--harness", "python", "--max-tool-calls", str(budget)]
                        stdout = io.StringIO()
                        with contextlib.redirect_stdout(stdout):
                            if failed:
                                with self.assertRaisesRegex(RuntimeError, "2 executions"):
                                    main(args)
                            else:
                                main(args)
                        with open(output, encoding="utf-8") as stream:
                            document = json.load(stream)
                finally:
                    del _session_cache[key]
                turns = session.children[0].turns
                self.assertEqual(document["initial_prompt"], turns[0][0])
                self.assertEqual(
                    [
                        event["content"]
                        for event in document["events"]
                        if event["type"] == "prompt"
                    ],
                    [turn[0] for turn in turns[1:]],
                )
                self.assertEqual(
                    [
                        event["content"]
                        for event in document["events"]
                        if event["type"] == "model_response"
                    ],
                    responses,
                )
                self.assertEqual(document["token_count"], 5 * len(responses))
                if harness:
                    self.assertIn(f"up to {budget} scripts", turns[0][0])
                    self.assertNotIn("final function", turns[0][0])
                    self.assertEqual(document["tool_call_count"], budget)
                    self.assertEqual(document["tool_calls"][1]["stdout"], "42\n")
                    self.assertEqual(
                        len(
                            [
                                event
                                for event in document["events"]
                                if event["type"] == "process_exit"
                            ]
                        ),
                        budget,
                    )
                    self.assertIn("Remaining Python tool calls: 0", turns[-1][0])
                    self.assertEqual(turns[0][1]["max_new_tokens"], 1024)
                else:
                    self.assertEqual(document["tool_calls"], [])
                    self.assertEqual(turns[0][1]["max_length"], 200)
                if failed:
                    self.assertIn("budget exhausted", document["error"])
                    self.assertEqual(stdout.getvalue(), "")
                else:
                    self.assertIsNone(document["error"])
                    self.assertEqual(document["text"], responses[-1])
                    self.assertEqual(stdout.getvalue().strip(), responses[-1])

    def test_invalid_budget(self):
        """Rejects nonpositive and noninteger script budgets."""
        for budget in (0, -1, 1.5, True):
            with (
                self.subTest(budget=budget),
                self.assertRaisesRegex(ValueError, "positive integer"),
            ):
                PythonHarness(max_tool_calls=budget)

    def test_tool_feedback_and_benchmark_exports(self):
        """Checks real script execution, feedback, scoring, and exported counts."""
        session = ScriptedSession(
            [
                (
                    "<think><python>raise AssertionError('not a tool call')</python></think>\n"
                    "<python>def checked():\n    return 6 * 7\nprint(checked())</python>"
                ),
                (
                    "I should test the error path.\n"
                    "```python\nraise AssertionError('not the final answer')\n```\n</think>\n"
                    "<python>import sys\nprint('diagnostic', file=sys.stderr)\n"
                    "sys.exit(3)</python>"
                ),
                (
                    "<think>The tool confirmed the value.</think>\n"
                    "```python\ndef answer():\n    return 42\n```"
                ),
            ]
        )
        test = PromptTest("Returns the answer.", [ExpectedResult((), 42), ExpectedResult((), 42)])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "results.json")
            result = BenchPromptTest([test]).run(
                session, harness=PythonHarness(), json_output=path, temperature=0.1
            )
            with open(path, encoding="utf-8") as stream:
                exported = json.load(stream)

        self.assertEqual(result.passed, 1)
        case = result.results[0]
        self.assertEqual(case.tool_call_count, 2)
        self.assertEqual(case.token_count, 15)
        self.assertEqual(case.tool_calls[0].stdout, "42\n")
        self.assertEqual(case.function_call_count, 1)
        self.assertEqual(case.prompt_count, 3)
        self.assertEqual(case.tool_calls[1].returncode, 3)
        self.assertIn("diagnostic", case.tool_calls[1].stderr)
        turns = session.children[0].turns
        self.assertIn("stdout:\n42\n", turns[1][0])
        self.assertIn("Remaining Python tool calls: 9", turns[1][0])
        self.assertIn("exit code: 3", turns[2][0])
        self.assertEqual(turns[0][1], {"max_new_tokens": 200, "temperature": 0.1})
        self.assertEqual(exported[0]["tool_call_count"], 2)
        self.assertEqual(len(exported[0]["tool_calls"]), 2)
        self.assertEqual(list(exported[0])[:2], ["initial_prompt", "events"])
        self.assertEqual(exported[0]["initial_prompt"], turns[0][0])
        self.assertEqual(exported[0]["initial_generation_options"], turns[0][1])
        interactions = exported[0]["events"]
        self.assertEqual(
            [event["type"] for event in interactions],
            [
                "model_response",
                "command",
                "stdout",
                "stderr",
                "process_exit",
                "prompt",
                "model_response",
                "command",
                "stdout",
                "stderr",
                "process_exit",
                "prompt",
                "model_response",
                "evaluation",
            ],
        )
        self.assertEqual(interactions[0]["content"], session.responses[0])
        self.assertEqual(
            interactions[1]["code"], "def checked():\n    return 6 * 7\nprint(checked())"
        )
        self.assertEqual(interactions[1]["argv"], case.tool_calls[0].command)
        self.assertEqual(interactions[1]["argv"][:2], [sys.executable, "-u"])
        self.assertEqual(interactions[1]["cwd"], os.path.dirname(interactions[1]["argv"][-1]))
        self.assertEqual(interactions[2]["content"], "42\n")
        self.assertEqual(interactions[3]["content"], "")
        self.assertEqual(interactions[4]["returncode"], 0)
        self.assertEqual(interactions[5]["content"], turns[1][0])
        self.assertEqual(interactions[9]["content"], "diagnostic\n")
        self.assertEqual(interactions[10]["returncode"], 3)
        self.assertEqual(interactions[12]["content"], session.responses[-1])
        self.assertEqual(interactions[-1]["results"], exported[0]["results"])
        frame = result.to_dataframe()
        self.assertEqual(frame["tool_call_count"].tolist(), [2, 2])
        stats = _compute_case_stats(frame)
        self.assertEqual(stats["tool_call_count"].tolist(), [2])
        self.assertEqual(stats["function_call_count"].tolist(), [1])
        self.assertEqual(stats["prompt_count"].tolist(), [3])
        self.assertEqual(exported[0]["function_call_count"], 1)
        self.assertEqual(exported[0]["prompt_count"], 3)

    def test_budget_boundary_and_reset(self):
        """Allows a final answer after ten calls and resets each problem."""
        session = ScriptedSession(
            ["<python>def checked():\n    return 'test'\nprint(checked())</python>"] * 10
            + ["```python\ndef answer():\n    return 42\n```"]
        )
        tests = [
            PromptTest("Problem one.", [ExpectedResult((), 42)]),
            PromptTest("Problem two.", [ExpectedResult((), 42)]),
        ]
        result = BenchPromptTest(tests).run(session, harness=PythonHarness())
        self.assertEqual(result.passed, 2)
        self.assertEqual([case.tool_call_count for case in result.results], [10, 10])
        self.assertEqual([len(child.turns) for child in session.children], [11, 11])
        self.assertIn("Remaining Python tool calls: 0", session.children[0].turns[-1][0])
        self.assertEqual(sum(case.tool_call_count for case in result.results), 20)

    def test_eleventh_call_is_not_executed(self):
        """Rejects an eleventh request without executing its script."""
        with tempfile.TemporaryDirectory() as directory:
            marker = os.path.join(directory, "should-not-exist")
            forbidden = f"<python>open({marker!r}, 'w').close()</python>"
            session = ScriptedSession(["<python>pass</python>"] * 10 + [forbidden])
            result = BenchPromptTest([PromptTest("Problem.", [ExpectedResult((), 42)])]).run(
                session, harness=PythonHarness()
            )
            self.assertFalse(os.path.exists(marker))
        self.assertEqual(result.failed, 1)
        self.assertEqual(result.results[0].tool_call_count, 10)
        self.assertIn("budget exhausted", result.to_json()[0]["harness_error"])
        self.assertFalse(result.results[0].run_status.compiled)

    def test_tagged_definitions_execute_and_receive_feedback(self):
        """Executes definitions and returns even empty output before the final answer."""
        definitions = (
            "<think>Now return the function.</think>\n<python>\n"
            "from math import factorial\n"
            "def answer():\n"
            "    return factorial(5)\n</python>"
        )
        session = ScriptedSession(
            [
                (
                    "<python>def answer():\n    return 120\n"
                    "assert answer() == 120\nprint(answer())</python>"
                ),
                definitions,
                (
                    "```python\nfrom math import factorial\n"
                    "def answer():\n    return factorial(5)\n```"
                ),
            ]
        )
        result = BenchPromptTest([PromptTest("Problem.", [ExpectedResult((), 120)])]).run(
            session, harness=PythonHarness()
        )
        self.assertEqual(result.passed, 1)
        case = result.results[0]
        self.assertEqual(case.tool_call_count, 2)
        self.assertEqual(case.tool_calls[0].stdout, "120\n")
        self.assertEqual(case.tool_calls[1].stdout, "")
        self.assertEqual(case.tool_calls[1].returncode, 0)
        self.assertEqual(case.function_call_count, 2)
        turns = session.children[0].turns
        self.assertEqual(len(turns), 3)
        self.assertIn("stdout:\n\nstderr:\n", turns[2][0])
        self.assertIn("exit code: 0", turns[2][0])
        self.assertIn("WITHOUT <python>", turns[2][0])
        responses = [
            event for event in result.to_json()[0]["events"] if event["type"] == "model_response"
        ]
        self.assertEqual(responses[1]["content"], definitions)

    def test_definition_only_calls_do_not_bypass_the_tool_budget(self):
        """Executes ten definition-only scripts instead of treating them as answers."""
        answer = 'def answer():\n    return "<think>literal</think>"'
        session = ScriptedSession([f"<python>{answer}</python>"] * 11)
        result = PythonHarness().run(session, "Problem.", max_length=100)
        self.assertEqual(len(result.tool_calls), 10)
        self.assertTrue(all(call.returncode == 0 for call in result.tool_calls))
        self.assertIn("budget exhausted", result.error)
        self.assertEqual(len(session.turns), 11)
        self.assertEqual(result.token_count, 55)
        self.assertEqual(extract_code(f"<python>{answer}</python>"), answer)

    def test_mixed_tool_call_executes_before_another_model_response(self):
        """Returns real execution output instead of accepting an accompanying answer."""
        final = "```python\ndef answer():\n    return 42\n```"
        session = ScriptedSession(
            [
                "Here is my check:\n<python>def answer():\n"
                "    return 6 * 7\nprint(answer())</python>\n" + final,
                final,
            ]
        )
        result = PythonHarness().run(session, "Problem.", max_length=100)
        self.assertIsNone(result.error)
        self.assertEqual(result.text, final)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].stdout, "42\n")
        self.assertEqual(len(session.turns), 2)
        self.assertIn("stdout:\n42\n", session.turns[1][0])
        self.assertEqual(result.token_count, 10)

    def test_publication_does_not_require_tool_execution(self):
        """Distinguishes published code from tool requests solely by message format."""
        final = '```python\ndef answer():\n    return "<python>literal</python>"\n```'
        session = ScriptedSession([final])
        result = PythonHarness().run(session, "Problem.", max_length=100)
        self.assertIsNone(result.error)
        self.assertEqual(result.text, final)
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(len(session.turns), 1)
        definitions = "<python>def answer():\n    return 42</python>"
        unchecked = PythonHarness().run(
            ScriptedSession([definitions, final]), "Problem.", max_length=100
        )
        self.assertIsNone(unchecked.error)
        self.assertEqual(unchecked.text, final)
        self.assertEqual(len(unchecked.tool_calls), 1)
        self.assertEqual(unchecked.tool_calls[0].function_call_count, 0)

    def test_multiple_calls_return_all_outputs_before_the_final_answer(self):
        """Executes grouped scripts in order and returns failures with other outputs."""
        response = (
            "<python>from pathlib import Path\n"
            "Path('value.txt').write_text('21')\nprint('created')</python>\n"
            "<python>raise ValueError('bad check')</python>\n"
            "<python>from pathlib import Path\n"
            "def answer():\n    return 2 * int(Path('value.txt').read_text())\n"
            "print(answer())</python>\n"
            "```python\ndef answer():\n    return 999\n```"
        )
        session = ScriptedSession([response, "```python\ndef answer():\n    return 42\n```"])
        result = BenchPromptTest(
            [PromptTest("Problem.", [ExpectedResult((), 42), ExpectedResult((), 42)])]
        ).run(session, harness=PythonHarness())
        self.assertEqual(result.passed, 1)
        case = result.results[0]
        self.assertEqual(case.tool_call_count, 3)
        self.assertEqual(case.function_call_count, 1)
        self.assertEqual(case.prompt_count, 2)
        self.assertEqual(case.token_count, 10)
        self.assertEqual([call.returncode for call in case.tool_calls], [0, 1, 0])
        self.assertEqual(case.multi_tool_turn_count, 1)
        self.assertEqual(case.max_tool_calls_per_turn, 3)
        feedback = session.children[0].turns[1][0]
        self.assertIn("stdout:\ncreated\n", feedback)
        self.assertIn("ValueError: bad check", feedback)
        self.assertIn("stdout:\n42\n", feedback)
        self.assertIn("Remaining Python tool calls: 7", feedback)
        self.assertLess(feedback.index("result 1/3"), feedback.index("result 2/3"))
        self.assertLess(feedback.index("result 2/3"), feedback.index("result 3/3"))
        entry = result.to_json()[0]
        self.assertEqual(entry["events"][0]["content"], response)
        self.assertEqual(entry["events"][0]["requested_tool_call_count"], 3)
        self.assertEqual(entry["multi_tool_turn_count"], 1)
        self.assertEqual(entry["max_tool_calls_per_turn"], 3)
        stats = _compute_case_stats(result.to_dataframe())
        self.assertEqual(stats["multi_tool_turn_count"].tolist(), [1])
        self.assertEqual(stats["max_tool_calls_per_turn"].tolist(), [3])

    def test_batch_budget_boundary_and_overflow(self):
        """Counts each grouped script and rejects an over-budget batch atomically."""
        session = ScriptedSession(
            ["<python>pass</python>" * 10, "```python\ndef answer():\n    return 42\n```"]
        )
        accepted = PythonHarness().run(session, "Problem.", max_length=100)
        self.assertIsNone(accepted.error)
        self.assertEqual(len(accepted.tool_calls), 10)
        self.assertEqual(len(session.turns), 2)
        self.assertIn("Remaining Python tool calls: 0", session.turns[1][0])
        with tempfile.TemporaryDirectory() as directory:
            marker = os.path.join(directory, "should-not-exist")
            script = f"<python>open({marker!r}, 'w').close()</python>"
            rejected = PythonHarness().run(
                ScriptedSession(["<python>pass</python>"] * 9 + [script * 2]),
                "Problem.",
                max_length=100,
            )
            self.assertFalse(os.path.exists(marker))
        self.assertEqual(len(rejected.tool_calls), 9)
        self.assertIn("budget exhausted", rejected.error)
        self.assertIn("requested 2 with 1 remaining", rejected.error)

    def test_native_python_request_executes_before_final_answer(self):
        """Executes Nemotron requests and preserves literal thinking tags in code."""
        request = (
            "<tool_call>\n<function=stateful_python_code_exec>\n<parameter=code>\n"
            "def checked():\n    return 42\n"
            "print('<think>literal</think>', checked())\n"
            "</parameter>\n</function>\n</tool_call>"
        )
        final = "```python\ndef answer():\n    return 42\n```"
        for name in ["stateful_python_code_exec", "python"]:
            with self.subTest(tool=name):
                response = (
                    "<think>Prepare the check.</think>\n"
                    + request.replace("stateful_python_code_exec", name)
                    + "\n"
                    + final
                )
                session = ScriptedSession([response, final])
                result = BenchPromptTest([PromptTest("Problem.", [ExpectedResult((), 42)])]).run(
                    session, harness=PythonHarness()
                )
                case = result.results[0]
                self.assertEqual(result.passed, 1)
                self.assertEqual(case.tool_call_count, 1)
                self.assertEqual(case.prompt_count, 2)
                self.assertEqual(case.tool_calls[0].stdout, "<think>literal</think> 42\n")
                self.assertIn(
                    "stdout:\n<think>literal</think> 42\n", session.children[0].turns[1][0]
                )
                self.assertEqual(case.interactions[1]["content"], response)
                self.assertEqual(case.interactions[1]["requested_tool_call_count"], 1)

    def test_mixed_native_and_python_requests_preserve_order(self):
        """Combines native and Python-tagged requests without executing fenced examples."""
        native = (
            "<tool_call><function=stateful_python_code_exec><parameter=code>"
            "from pathlib import Path\nPath('value.txt').write_text('42')\nprint('created')"
            "</parameter></function></tool_call>"
        )
        response = (
            f"```text\n{native}\n```\n{native}\n"
            "<python>from pathlib import Path\nprint(Path('value.txt').read_text())</python>"
        )
        session = ScriptedSession([response, "```python\ndef answer():\n    return 42\n```"])
        result = BenchPromptTest([PromptTest("Problem.", [ExpectedResult((), 42)])]).run(
            session, harness=PythonHarness()
        )
        case = result.results[0]
        self.assertEqual(result.passed, 1)
        self.assertEqual(case.tool_call_count, 2)
        self.assertEqual([call.stdout for call in case.tool_calls], ["created\n", "42\n"])
        self.assertEqual(case.multi_tool_turn_count, 1)
        self.assertEqual(case.max_tool_calls_per_turn, 2)
        self.assertEqual(len(extract_python_calls(response)), 2)
        feedback = session.children[0].turns[1][0]
        self.assertIn("Remaining Python tool calls: 8", feedback)
        self.assertLess(feedback.index("created"), feedback.index("42"))

    def test_abbreviated_native_request(self):
        """Executes complete code-parameter replies without guessing missing closing tags."""
        request = (
            "<parameter=code>\ndef answer():\n    return 42\n"
            "print('<think>literal</think>', answer())\n"
            "</parameter>\n</function>\n</tool_call>"
        )
        final = "```python\ndef answer():\n    return 42\n```"
        response = "<think>Check the solution.</think>\n" + request
        session = ScriptedSession([response, final])
        result = PythonHarness().run(session, "Problem.", max_length=100)
        self.assertIsNone(result.error)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].stdout, "<think>literal</think> 42\n")
        self.assertIn("<think>literal</think> 42", session.turns[1][0])
        self.assertEqual(result.interactions[1]["content"], response)
        for text in [
            f"```text\n{request}\n```",
            f"<think>{request}</think>",
            request.removesuffix("</tool_call>"),
            request.replace("</function>", ""),
            "Example:\n" + request,
            request.replace("<parameter=code>", "<parameter=other>"),
            request.replace("<parameter=code>", "<tool_call><function=unknown><parameter=code>"),
        ]:
            with self.subTest(text=text):
                self.assertEqual(extract_python_calls(text), [])

    def test_final_python_blocks_exclude_output_and_explanations(self):
        """Scores published Python without compiling output or other-language fences."""
        response = (
            "<think>Prepare the answer.</think>\n"
            "```python\ndef answer():\n    return helper()\n```\n"
            "Explanation:\n```text\nThis is not Python.\n```\n"
            "```py\ndef helper():\n    return 42\n```\n"
            "Output:\n```\n42 is the result.\n```"
        )
        for harness in [None, PythonHarness()]:
            with self.subTest(harness=harness):
                session = ScriptedSession([response])
                result = BenchPromptTest([PromptTest("Problem.", [ExpectedResult((), 42)])]).run(
                    session, harness=harness
                )
                self.assertEqual(result.passed, 1)
                self.assertEqual(result.results[0].tool_call_count, 0)
                self.assertEqual(result.results[0].interactions[1]["content"], response)
        self.assertEqual(
            extract_code("```\ndef answer():\n    return 42\n```"), "def answer():\n    return 42"
        )
        self.assertEqual(extract_code("```text\nprint('not code')\n```"), "")

    def test_published_assertions_are_not_removed(self):
        """Keeps incorrect self-tests as real evaluation failures."""
        response = (
            "```python\ndef answer():\n    return 42\nassert answer() == 0\n```\n"
            "Output:\n```\n0\n```"
        )
        result = BenchPromptTest([PromptTest("Problem.", [ExpectedResult((), 42)])]).run(
            ScriptedSession([response])
        )
        self.assertEqual(result.passed, 0)
        self.assertIsInstance(result.results[0].run_status.run_error, AssertionError)

    def test_native_request_budget_and_unsupported_tools(self):
        """Rejects oversized native batches and unknown tools before executing scripts."""
        native = (
            "<tool_call><function=stateful_python_code_exec>"
            "<parameter=code>raise AssertionError('must not execute')"
            "</parameter></function></tool_call>"
        )
        rejected = PythonHarness().run(ScriptedSession([native * 11]), "Problem.", 100)
        self.assertEqual(rejected.tool_calls, [])
        self.assertIn("requested 11 with 10 remaining", rejected.error)
        for request in [
            native.replace("stateful_python_code_exec", "unknown_tool"),
            native.removesuffix("</tool_call>"),
        ]:
            with self.subTest(request=request):
                result = PythonHarness().run(ScriptedSession([request]), "Problem.", 100)
                self.assertEqual(result.tool_calls, [])
                self.assertIn("native Python tool call", result.error)

    def test_invalid_tagged_script_receives_tool_feedback(self):
        """Keeps invalid Python scripts on the tool execution error path."""
        session = ScriptedSession(
            [
                "<python>def answer(:</python>",
                "<python>def answer():\n    return 42\nassert answer() == 42</python>",
                "```python\ndef answer():\n    return 42\n```",
            ]
        )
        result = PythonHarness().run(session, "Problem.", max_length=100)
        self.assertIsNone(result.error)
        self.assertEqual(len(result.tool_calls), 2)
        self.assertIn("SyntaxError", result.tool_calls[0].stderr)
        self.assertIn("SyntaxError", session.turns[1][0])

    def test_timeout_feedback_and_working_directory(self):
        """Counts timed-out executions and preserves files between calls."""
        session = ScriptedSession(
            [
                (
                    "<python>from pathlib import Path\n"
                    "def prepare():\n    Path('value.txt').write_text('42')\nprepare()\n"
                    "import time\ntime.sleep(30)</python>"
                ),
                (
                    "<python>from pathlib import Path\n"
                    "def read_value():\n    return Path('value.txt').read_text()\n"
                    "print(read_value())"
                    "</python>"
                ),
                "def answer():\n    return 42",
            ]
        )
        result = PythonHarness(timeout=0.5).run(session, "Problem.", max_length=100)
        self.assertEqual(len(result.tool_calls), 2)
        self.assertTrue(result.tool_calls[0].timed_out)
        self.assertEqual(result.tool_calls[0].function_call_count, 1)
        self.assertIsNone(result.tool_calls[0].returncode)
        self.assertIn("timed out: True", session.turns[1][0])
        self.assertEqual(
            next(event for event in result.interactions if event["type"] == "process_exit"),
            {"type": "process_exit", "returncode": None, "timed_out": True},
        )
        self.assertEqual(result.tool_calls[1].stdout, "42\n")
        self.assertIsNone(result.error)

    def test_no_tool_and_invalid_responses(self):
        """Reports malformed and empty turns without executing incomplete scripts."""
        for response, error in [
            ("<python>print(42)", "Malformed"),
            ("<think><python>print(42)</python>", "empty answer"),
            ("", "empty answer"),
        ]:
            with self.subTest(response=response):
                result = PythonHarness().run(
                    ScriptedSession([response]), "Problem.", max_length=100
                )
                self.assertEqual(len(result.tool_calls), 0)
                self.assertIn(error, result.error)

    def test_script_error_and_output_truncation(self):
        """Captures script failures and bounds the output sent to the model."""
        with tempfile.TemporaryDirectory() as directory:
            harness = PythonHarness()
            failed = harness.execute("raise ValueError('bad input')", directory)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("ValueError: bad input", failed.stderr)
            large = harness.execute("print('x' * 20000)", directory)
        self.assertTrue(large.stdout.endswith("[output truncated]"))
        self.assertLess(len(large.stdout), 16_100)

    def test_function_counter_records_calls_not_definitions(self):
        """Counts actual script-defined calls, including recursion and failures."""
        with tempfile.TemporaryDirectory() as directory:
            harness = PythonHarness()
            definitions = harness.execute("def answer():\n    return 42", directory)
            recursive = harness.execute(
                "def factorial(n):\n"
                "    return n * factorial(n - 1) if n else 1\n"
                "print(factorial(5))",
                directory,
            )
            failed = harness.execute(
                "def fail():\n    raise ValueError('failure')\nfail()", directory
            )
        self.assertEqual(definitions.function_call_count, 0)
        self.assertEqual(recursive.function_call_count, 6)
        self.assertEqual(recursive.stdout, "120\n")
        self.assertEqual(failed.function_call_count, 1)
        self.assertNotEqual(failed.returncode, 0)

    def test_without_harness_preserves_single_generation(self):
        """Keeps the original generation path and reports zero tool calls."""
        session = ScriptedSession(
            [
                (
                    "<think>```python\nraise AssertionError('thinking must not execute')\n```"
                    "</think>\n```python\ndef answer():\n    return 42\n```"
                )
            ]
        )
        result = BenchPromptTest([PromptTest("Problem.", [ExpectedResult((), 42)])]).run(session)
        self.assertEqual(result.passed, 1)
        self.assertEqual(result.results[0].tool_call_count, 0)
        self.assertEqual(session.children[0].turns, [("Problem.", {"max_length": 200})])
        entry = result.to_json()[0]
        self.assertEqual(entry["initial_prompt"], "Problem.")
        interactions = entry["events"]
        self.assertEqual(
            [event["type"] for event in interactions], ["model_response", "evaluation"]
        )
        self.assertEqual(interactions[0]["content"], session.responses[0])

    def test_thinking_boundaries(self):
        """Ignores reasoning and preserves literal tags inside fenced final code."""
        answer = 'def answer():\n    return "<think>literal</think>"'
        fenced = f"```python\n{answer}\n```"
        for prefix in [
            "",
            "<think>Plan.</think>\n",
            "Plan already started in the prompt.</think>\n",
            "<think>First.</think>\n<think>Second.</think>\n",
        ]:
            with self.subTest(prefix=prefix):
                self.assertEqual(extract_code(prefix + fenced), answer)
                self.assertEqual(strip_thinking(prefix + fenced), fenced)
        self.assertEqual(extract_code("<think>```python\nraise AssertionError()\n```"), "")

    def test_failed_evaluation_is_logged_without_feedback(self):
        """Records errors without revealing test outcomes to the model."""
        for response, error in [
            ("```python\ndef answer(:\n```", "SyntaxError"),
            ("```python\ndef answer():\n    raise ValueError('failure')\n```", "ValueError"),
        ]:
            with self.subTest(error=error):
                session = ScriptedSession(
                    [
                        "<python>def checked():\n    return 'check'\nprint(checked())</python>",
                        response,
                    ]
                )
                result = BenchPromptTest(
                    [PromptTest("Problem.", [ExpectedResult((), "hidden-value")])]
                ).run(session, harness=PythonHarness())
                entry = result.to_json()[0]
                json.dumps(entry)
                self.assertEqual(result.failed, 1)
                self.assertEqual(len(session.children[0].turns), 2)
                self.assertTrue(
                    all("hidden-value" not in turn[0] for turn in session.children[0].turns)
                )
                self.assertEqual(entry["events"][-2]["content"], response)
                evaluation = entry["events"][-1]
                self.assertEqual(evaluation["type"], "evaluation")
                self.assertIn(error, evaluation["compile_error"] or evaluation["run_error"])


if __name__ == "__main__":
    unittest.main()
