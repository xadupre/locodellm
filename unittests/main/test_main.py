import contextlib
import io
import unittest
from unittest.mock import patch

from locodellm import __version__
from locodellm.__main__ import main
from locodellm.ext_test_case import ExtTestCase, skipif_no_genai


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

    @skipif_no_genai()
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

    @skipif_no_genai()
    def test_bench_mock_model(self):
        """Checks that bench subcommand produces a markdown table."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["bench", "mock/generate", "basic", "--chat-template", "chatml"])
        output = buf.getvalue()
        self.assertIn("| prompt", output)
        self.assertIn("compiled", output)
        self.assertIn("passed", output)

    @skipif_no_genai()
    def test_bench_output_csv(self):
        """Checks that bench subcommand can write a CSV file."""
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
        with open(csv_path) as f:
            content = f.read()
        self.assertIn("prompt", content)
        self.assertIn("compiled", content)

    @skipif_no_genai()
    def test_bench_output_xlsx(self):
        """Checks that Excel output contains aggregated data before raw data."""
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

        self.assertIn("score", aggregated.columns)
        self.assertIn("input_index", raw_data.columns)
        self.assertEqual(len(aggregated), 10)
        self.assertEqual(len(raw_data), 34)


if __name__ == "__main__":
    unittest.main()
