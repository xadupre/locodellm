import os
import unittest

from locodellm.bench import BenchPromptTest, ExpectedResult, PromptTest
from locodellm.ext_test_case import ExtTestCase, skipif_no_genai
from locodellm.test_models import create_mock_generate_model


class TestBenchPromptTest(ExtTestCase):
    """Tests for BenchPromptTest using the mock ONNX model."""

    @classmethod
    def setUpClass(cls):
        """Creates the mock model directory once for all tests."""
        folder = cls.get_dump_folder("bench_prompt_test")
        cls.model_path = create_mock_generate_model(os.path.join(folder, "mock-llm"))

    @skipif_no_genai()
    def test_passing_test(self):
        """Checks that a correct expected result passes."""
        from locodellm.session import create_session

        session = create_session(self.model_path, chat_template="chatml")
        tests = [
            PromptTest(
                prompt='write a python function which returns "hello"',
                expected=[ExpectedResult(args=(), expected="hello")],
            )
        ]
        bench = BenchPromptTest(tests, max_length=200)
        result = bench.run(session)
        self.assertEqual(result.total, 1)
        self.assertEqual(result.passed, 1)
        self.assertEqual(result.failed, 0)
        self.assertTrue(result.results[0].all_passed)
        self.assertTrue(result.results[0].run_status.compiled)
        self.assertTrue(result.results[0].run_status.ran)

    @skipif_no_genai()
    def test_failing_test(self):
        """Checks that an incorrect expected result fails."""
        from locodellm.session import create_session

        session = create_session(self.model_path, chat_template="chatml")
        tests = [
            PromptTest(
                prompt='write a python function which returns "hello"',
                expected=[ExpectedResult(args=(), expected="wrong")],
            )
        ]
        bench = BenchPromptTest(tests, max_length=200)
        result = bench.run(session)
        self.assertEqual(result.total, 1)
        self.assertEqual(result.passed, 0)
        self.assertEqual(result.failed, 1)
        self.assertFalse(result.results[0].all_passed)

    @skipif_no_genai()
    def test_multiple_tests(self):
        """Checks running multiple prompt tests in one bench."""
        from locodellm.session import create_session

        session = create_session(self.model_path, chat_template="chatml")
        tests = [
            PromptTest(
                prompt='write a python function which returns "hello"',
                expected=[ExpectedResult(args=(), expected="hello")],
            ),
            PromptTest(
                prompt='write a python function which returns "hello"',
                expected=[ExpectedResult(args=(), expected="not hello")],
            ),
        ]
        bench = BenchPromptTest(tests, max_length=200)
        result = bench.run(session)
        self.assertEqual(result.total, 2)
        self.assertEqual(result.passed, 1)
        self.assertEqual(result.failed, 1)

    @skipif_no_genai()
    def test_session_is_restarted_between_tests(self):
        """Checks that each test gets a fresh session."""
        from locodellm.session import create_session

        session = create_session(self.model_path, chat_template="chatml")
        tests = [
            PromptTest(
                prompt='write a python function which returns "hello"',
                expected=[ExpectedResult(args=(), expected="hello")],
            ),
            PromptTest(
                prompt='write a python function which returns "hello"',
                expected=[ExpectedResult(args=(), expected="hello")],
            ),
        ]
        bench = BenchPromptTest(tests, max_length=200)
        result = bench.run(session)
        # Both should produce identical code since session is restarted
        self.assertEqual(result.results[0].generated_code, result.results[1].generated_code)
        self.assertEqual(result.passed, 2)

    @skipif_no_genai()
    def test_generated_code_is_captured(self):
        """Checks that the generated code is stored in the result."""
        from locodellm.session import create_session

        session = create_session(self.model_path, chat_template="chatml")
        tests = [
            PromptTest(
                prompt='write a python function which returns "hello"',
                expected=[ExpectedResult(args=(), expected="hello")],
            )
        ]
        bench = BenchPromptTest(tests, max_length=200)
        result = bench.run(session)
        self.assertIn("def hello", result.results[0].generated_code)

    @skipif_no_genai()
    def test_to_dataframe(self):
        """Checks that to_dataframe returns a correct DataFrame."""
        import pandas

        from locodellm.session import create_session

        session = create_session(self.model_path, chat_template="chatml")
        tests = [
            PromptTest(
                prompt='write a python function which returns "hello"',
                expected=[
                    ExpectedResult(args=(), expected="hello"),
                    ExpectedResult(args=(), expected="wrong"),
                ],
            )
        ]
        bench = BenchPromptTest(tests, max_length=200)
        result = bench.run(session)
        df = result.to_dataframe()
        self.assertIsInstance(df, pandas.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn("prompt", df.columns)
        self.assertIn("passed", df.columns)
        self.assertTrue(df.iloc[0]["passed"])
        self.assertFalse(df.iloc[1]["passed"])


if __name__ == "__main__":
    unittest.main()
