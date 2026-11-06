import contextlib
import io
import unittest
from unittest.mock import patch

from locodellm import __version__
from locodellm.__main__ import _load_generate_input, main
from locodellm.ext_test_case import ExtTestCase, requires_onnxruntime_genai


class TestMain(ExtTestCase):
    def test_version(self):
        """Checks that the version subcommand prints the version."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["version"])
        self.assertEqual(buf.getvalue().strip(), __version__)

    def test_benchmarks(self):
        """Checks that built-in and LM-Eval benchmarks are listed."""
        buf = io.StringIO()
        with (
            patch("locodellm.__main__._get_lm_eval_benchmarks", return_value=[]),
            contextlib.redirect_stdout(buf),
        ):
            main(["benchmarks"])
        output = buf.getvalue()
        self.assertIn("basic", output)
        self.assertIn("gsm8k", output)
        self.assertIn(
            "https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/", output
        )

    def test_models(self):
        """Checks that the models subcommand lists available mock models."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["models"])
        output = buf.getvalue()
        self.assertIn("mock/generate", output)
        self.assertIn("mock/tiny", output)

    def test_no_command_exits(self):
        """Checks that no subcommand prints help and exits."""
        with self.assertRaises(SystemExit) as ctx:
            main([])
        self.assertEqual(ctx.exception.code, 1)


class TestMainGenerate(ExtTestCase):
    """Tests for the generate subcommand."""

    @requires_onnxruntime_genai()
    def test_onnx_generation_max_time(self):
        """Stops ONNX decoding before the next token without dropping prompt history."""
        import tempfile
        from pathlib import Path

        from locodellm.generate.generate_from_model import get_session
        from locodellm.test_models import create_tiny_model

        with tempfile.TemporaryDirectory() as directory:
            model = create_tiny_model(str(Path(directory) / "model"))
            session = get_session(model).new_session()
            session.generate("<s>", max_new_tokens=100, max_time=1e-9)
            self.assertEqual(session.generated_token_count, 0)
            self.assertEqual(session.text, "")
            self.assertEqual(session.tokens.size, len(session.tokenizer.encode("<s>")))
            for value in (0, -1, float("nan"), float("inf")):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "max_time"):
                    session.generate("<s>", max_time=value)

    @requires_onnxruntime_genai()
    def test_generate_onnx_temperature(self):
        """Runs greedy and sampled ONNX decoding with locally generated model files."""
        import json
        import tempfile
        from pathlib import Path

        from locodellm.test_models import create_tiny_model

        with tempfile.TemporaryDirectory() as directory:
            model = create_tiny_model(str(Path(directory) / "model"))
            output = Path(directory) / "events.json"
            for temperature in (0.0, 0.7):
                with (
                    self.subTest(temperature=temperature),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    main(
                        [
                            "generate",
                            model,
                            "<s>",
                            "--max-length",
                            "8",
                            "--temperature",
                            str(temperature),
                            "--output",
                            str(output),
                        ]
                    )
                document = json.loads(output.read_text())
                options = document["initial_generation_options"]
                self.assertEqual(options["do_sample"], temperature > 0)
                if temperature > 0:
                    self.assertEqual(options["temperature"], temperature)
                else:
                    self.assertNotIn("temperature", options)
                self.assertIsNone(document["error"])

    @requires_onnxruntime_genai()
    def test_generate_onnx_live_log(self):
        """Streams ONNX model output into a retained transcript without verbose."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "live.log"
            with (
                contextlib.redirect_stdout(io.StringIO()) as stdout,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                main(
                    [
                        "generate",
                        "mock/generate",
                        'write a python function which returns "hello"',
                        "--chat-template",
                        "chatml",
                        "--live-log",
                        str(path),
                    ]
                )
            text = path.read_text()
            self.assertIn("def hello", stdout.getvalue())
            self.assertIn("def hello", text)
            self.assertIn("[model response complete]", text)

    def test_invalid_json_input(self):
        """Rejects malformed prompt lists and missing repositories before loading."""
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tasks.json"
            for document in [
                [],
                {"prompts": ["Prompt."]},
                {"repo": directory, "prompts": ["Prompt."], "unknown": True},
                {"repo": directory, "prompts": []},
                {"repo": directory, "prompts": "Prompt."},
                {"repo": directory, "prompts": [" "]},
                {"repo": directory, "prompts": [1]},
                {"repo": None, "prompts": ["Prompt."]},
                {"repo": "", "prompts": ["Prompt."]},
                {"repo": str(source), "prompts": ["Prompt."]},
            ]:
                with self.subTest(document=document):
                    source.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        main(["generate", "unused/model", "--input", str(source)])
            source.write_text("{", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                main(["generate", "unused/model", "--input", str(source)])
            source.unlink()
            with self.assertRaises(FileNotFoundError):
                main(["generate", "unused/model", "--input", str(source)])

    def test_generate_requires_exactly_one_prompt_source(self):
        """Rejects missing prompts and conflicting positional and JSON inputs."""
        for options in [[], ["Prompt.", "--input", "tasks.json"]]:
            with self.subTest(options=options):
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    main(["generate", "unused/model", *options])
                self.assertEqual(error.exception.code, 2)

    def test_jsonl_input_extracts_only_prompts(self):
        """Loads UTF-8 prompts in order without exposing reference answers or metadata."""
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tasks.jsonl"
            source.write_text(
                "\n".join(
                    json.dumps(record, ensure_ascii=False)
                    for record in [
                        {"prompt": "Vérifie l'export.", "expected_answer": "SECRET ANSWER"},
                        {"prompt": "Second task.", "repository": "https://example.com/repo"},
                    ]
                )
                + "\n\n",
                encoding="utf-8",
            )
            prompts, repository = _load_generate_input(str(source), ".")
            self.assertEqual(prompts, ["Vérifie l'export.", "Second task."])
            self.assertEqual(repository, str(Path.cwd()))

    def test_invalid_jsonl_input(self):
        """Rejects missing repositories, empty inputs, and malformed prompt records."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tasks.jsonl"
            source.write_text('{"prompt": "Task."}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires 'repository'"):
                main(["generate", "unused/model", "--input", str(source)])
            for content, message in [
                (" \n", "nonempty list"),
                ('{"prompt": "Task."}\n{\n', "at line 2"),
                ('{"prompt": "Task."}\n{"prompt": 1}\n', "JSONL line 2"),
                ('{"expected_answer": "Answer."}\n', "JSONL line 1"),
                ('{"prompt": " "}\n', "JSONL line 1"),
                ('["Task."]\n', "JSONL line 1"),
            ]:
                with self.subTest(content=content):
                    source.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        main(
                            [
                                "generate",
                                "unused/model",
                                "--input",
                                str(source),
                                "--repo",
                                directory,
                            ]
                        )
            source.write_text('{"prompt": "Task."}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Repository directory does not exist"):
                _load_generate_input(str(source), str(Path(directory) / "missing"))
            source = source.with_suffix(".json")
            source.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "--repo is only supported with JSONL"):
                _load_generate_input(str(source), directory)
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                main(["generate", "unused/model", "Task.", "--repo", directory])
            self.assertEqual(error.exception.code, 2)

    def test_invalid_jsonl_clone_metadata(self):
        """Rejects incompatible repositories and invalid revisions before cloning."""
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tasks.jsonl"
            record = {"prompt": "Task.", "repository": directory}
            for records, message in [
                ([{**record, "repository": " "}], "requires 'repository'"),
                ([{**record, "verification": []}], "invalid 'verification'"),
                ([{**record, "verification": {"revision": "main"}}], "full commit SHA"),
                ([{**record, "verification": {"revision": 1}}], "full commit SHA"),
                ([record, {**record, "repository": directory + "/other"}], "same repository"),
                ([record, {**record, "verification": {"revision": "a" * 40}}], "same repository"),
            ]:
                with self.subTest(records=records):
                    source.write_text(
                        "\n".join(json.dumps(value) for value in records), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        _load_generate_input(str(source), clone_directory=directory)
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                main(["generate", "unused/model", "Task.", "--clone-dir", directory])
            self.assertEqual(error.exception.code, 2)

    def test_invalid_temperature(self):
        """Rejects negative and nonfinite temperatures before loading a model."""
        for temperature in ["-1", "nan", "inf", "-inf"]:
            with (
                self.subTest(temperature=temperature),
                contextlib.redirect_stderr(io.StringIO()) as stderr,
                self.assertRaises(SystemExit) as error,
            ):
                main(["generate", "unused/model", "Prompt.", f"--temperature={temperature}"])
            self.assertEqual(error.exception.code, 2)
            self.assertIn("--temperature must be finite and nonnegative", stderr.getvalue())

    def test_invalid_harness_options(self):
        """Rejects invalid budgets and lengths before loading a model."""
        for command, positional in [("generate", ["Prompt."]), ("bench", ["basic"])]:
            for options in [
                ["--harness", "python", "--max-tool-calls", "0"],
                ["--max-tool-calls", "100"],
                ["--max-length", "0"],
                ["--backend", "pytorch", "--provider", "CPUExecutionProvider"],
                ["--backend", "pytorch", "--precision", "int4"],
                ["--device", "cpu"],
            ]:
                with self.subTest(command=command, options=options):
                    with (
                        contextlib.redirect_stderr(io.StringIO()),
                        self.assertRaises(SystemExit) as error,
                    ):
                        main([command, "unused/model", *positional, *options])
                    self.assertEqual(error.exception.code, 2)

    @requires_onnxruntime_genai()
    def test_generate_mock_model(self):
        """Checks that generate subcommand produces output."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(
                [
                    "generate",
                    "mock/generate",
                    'write a python function which returns "hello"',
                    "--chat-template",
                    "chatml",
                ]
            )
        output = buf.getvalue()
        self.assertIn("def hello", output)


class TestMainBench(ExtTestCase):
    """Tests for the bench subcommand."""

    def test_harness_rejects_lm_eval(self):
        """Rejects the Python harness for external LM-Eval tasks."""
        with self.assertRaisesRegex(ValueError, "single built-in benchmark"):
            main(["bench", "unused/model", "gsm8k", "--harness", "python"])

    @requires_onnxruntime_genai()
    def test_bench_python_harness(self):
        """Checks harness CLI output and the matching JSON and Excel exports."""
        import json
        import os
        import tempfile

        import pandas

        from locodellm.test_models import create_mock_generate_model

        with tempfile.TemporaryDirectory() as directory:
            model_path = create_mock_generate_model(os.path.join(directory, "model"))
            output_path = os.path.join(directory, "results.csv")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                main(
                    [
                        "bench",
                        model_path,
                        "basic",
                        "--harness",
                        "python",
                        "--max-tool-calls",
                        "100",
                        "--max-length",
                        "200",
                        "--chat-template",
                        "chatml",
                        "--output",
                        output_path,
                    ]
                )
            raw = pandas.read_csv(output_path)
            with pandas.ExcelFile(os.path.join(directory, "results.xlsx")) as workbook:
                stats = pandas.read_excel(workbook, sheet_name="aggregated")
            with open(os.path.join(directory, "results.json"), encoding="utf-8") as stream:
                detailed = json.load(stream)
        self.assertIn("total_tool_calls", buf.getvalue())
        self.assertIn("tool_call_count", buf.getvalue())
        self.assertEqual(raw["tool_call_count"].tolist(), [0] * len(raw))
        self.assertEqual(stats["tool_call_count"].tolist(), [0] * 10)
        self.assertEqual([case["tool_call_count"] for case in detailed], [0] * 10)
        self.assertEqual([case["tool_calls"] for case in detailed], [[] for _ in range(10)])
        self.assertTrue(all("up to 100 scripts" in case["initial_prompt"] for case in detailed))

    @requires_onnxruntime_genai()
    def test_bench_mock_model(self):
        """Checks that bench subcommand produces a markdown table."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["bench", "mock/generate", "basic", "--chat-template", "chatml"])
        output = buf.getvalue()
        self.assertIn("| prompt", output)
        self.assertIn("compiled", output)
        self.assertIn("passed", output)

    @requires_onnxruntime_genai()
    def test_bench_output_csv(self):
        """Checks that CSV output is accompanied by JSON and Excel files."""
        import os
        import tempfile

        tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(tmpdir, "results.csv")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(
                [
                    "bench",
                    "mock/generate",
                    "basic",
                    "--chat-template",
                    "chatml",
                    "--output",
                    csv_path,
                ]
            )
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(os.path.join(tmpdir, "results.json")))
        self.assertTrue(os.path.exists(os.path.join(tmpdir, "results.xlsx")))
        with open(csv_path) as f:
            content = f.read()
        self.assertIn("prompt", content)
        self.assertIn("compiled", content)

    @requires_onnxruntime_genai()
    def test_bench_output_xlsx(self):
        """Checks that Excel output is accompanied by detailed JSON."""
        import json
        import os
        import tempfile

        import pandas

        with tempfile.TemporaryDirectory() as tmpdir:
            xlsx_path = os.path.join(tmpdir, "results.xlsx")
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "bench",
                        "mock/generate",
                        "basic",
                        "--chat-template",
                        "chatml",
                        "--output",
                        xlsx_path,
                    ]
                )

            workbook = pandas.ExcelFile(xlsx_path)
            self.assertEqual(workbook.sheet_names, ["aggregated", "raw_data"])
            aggregated = pandas.read_excel(workbook, sheet_name="aggregated")
            raw_data = pandas.read_excel(workbook, sheet_name="raw_data")
            json_path = os.path.join(tmpdir, "results.json")
            self.assertTrue(os.path.exists(json_path))
            with open(json_path, encoding="utf-8") as f:
                detailed = json.load(f)

        self.assertIn("score", aggregated.columns)
        self.assertIn("input_index", raw_data.columns)
        self.assertEqual(len(aggregated), 10)
        self.assertEqual(len(raw_data), 34)
        self.assertEqual(len(detailed), 10)
        self.assertIn("generated_code", detailed[0])
        self.assertIn("results", detailed[0])

    @requires_onnxruntime_genai()
    def test_bench_output_json(self):
        """Checks that JSON output is accompanied by an Excel workbook."""
        import os
        import tempfile

        import pandas

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "results.json")
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "bench",
                        "mock/generate",
                        "basic",
                        "--chat-template",
                        "chatml",
                        "--output",
                        json_path,
                    ]
                )

            self.assertTrue(os.path.exists(json_path))
            workbook = pandas.ExcelFile(os.path.join(tmpdir, "results.xlsx"))

        self.assertEqual(workbook.sheet_names, ["aggregated", "raw_data"])

    @requires_onnxruntime_genai()
    def test_bench_output_uppercase_xlsx(self):
        """Checks that the requested path is preserved for uppercase suffixes."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            xlsx_path = os.path.join(tmpdir, "results.XLSX")
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "bench",
                        "mock/generate",
                        "basic",
                        "--chat-template",
                        "chatml",
                        "--output",
                        xlsx_path,
                    ]
                )

            self.assertTrue(os.path.exists(xlsx_path))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "results.json")))


if __name__ == "__main__":
    unittest.main()
