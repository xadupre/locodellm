import json
import os
import sys
import tempfile
import unittest

from locodellm.__main__ import _compute_case_stats
from locodellm.bench import BenchPromptTest, ExpectedResult, PromptTest
from locodellm.bench.python_harness import PythonHarness
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


class TestPythonHarness(unittest.TestCase):
    def test_tool_feedback_and_benchmark_exports(self):
        """Checks real script execution, feedback, scoring, and exported counts."""
        session = ScriptedSession(
            [
                (
                    "<think><python>raise AssertionError('not a tool call')</python></think>\n"
                    "<python>print(6 * 7)</python>"
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
        self.assertEqual(interactions[1]["code"], "print(6 * 7)")
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

    def test_budget_boundary_and_reset(self):
        """Allows a final answer after ten calls and resets each problem."""
        session = ScriptedSession(
            ["<python>print('test')</python>"] * 10
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

    def test_timeout_feedback_and_working_directory(self):
        """Counts timed-out executions and preserves files between calls."""
        session = ScriptedSession(
            [
                (
                    "<python>from pathlib import Path\n"
                    "Path('value.txt').write_text('42')\n"
                    "import time\ntime.sleep(30)</python>"
                ),
                (
                    "<python>from pathlib import Path\nprint(Path('value.txt').read_text())"
                    "</python>"
                ),
                "def answer():\n    return 42",
            ]
        )
        result = PythonHarness(timeout=0.5).run(session, "Problem.", max_length=100)
        self.assertEqual(len(result.tool_calls), 2)
        self.assertTrue(result.tool_calls[0].timed_out)
        self.assertIsNone(result.tool_calls[0].returncode)
        self.assertIn("timed out: True", session.turns[1][0])
        self.assertEqual(
            next(event for event in result.interactions if event["type"] == "process_exit"),
            {"type": "process_exit", "returncode": None, "timed_out": True},
        )
        self.assertEqual(result.tool_calls[1].stdout, "42\n")
        self.assertIsNone(result.error)

    def test_no_tool_and_invalid_responses(self):
        """Handles optional tool use and reports malformed or empty turns."""
        for response, error in [
            ("def answer():\n    return 42", None),
            ("<python>print(42)", "Malformed"),
            ("<think><python>print(42)</python>", "empty answer"),
            ("", "empty answer"),
        ]:
            with self.subTest(response=response):
                result = PythonHarness().run(
                    ScriptedSession([response]), "Problem.", max_length=100
                )
                self.assertEqual(len(result.tool_calls), 0)
                if error is None:
                    self.assertIsNone(result.error)
                    self.assertEqual(result.text, response)
                else:
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
                session = ScriptedSession([response])
                result = BenchPromptTest(
                    [PromptTest("Problem.", [ExpectedResult((), "hidden-value")])]
                ).run(session, harness=PythonHarness())
                entry = result.to_json()[0]
                json.dumps(entry)
                self.assertEqual(result.failed, 1)
                self.assertEqual(len(session.children[0].turns), 1)
                self.assertNotIn("hidden-value", session.children[0].turns[0][0])
                self.assertEqual(entry["events"][0]["content"], response)
                evaluation = entry["events"][-1]
                self.assertEqual(evaluation["type"], "evaluation")
                self.assertIn(error, evaluation["compile_error"] or evaluation["run_error"])


if __name__ == "__main__":
    unittest.main()
