import unittest

from locodellm.bench import ExpectedResult, PromptTest


class TestPromptTestSerialization(unittest.TestCase):
    """Tests for PromptTest and ExpectedResult JSON serialization."""

    def _make_prompt_test(self):
        """Creates a sample PromptTest instance."""
        return PromptTest(
            prompt="write a function that adds two numbers",
            expected=[
                ExpectedResult(args=(1, 2), expected=3),
                ExpectedResult(args=(0, 0), expected=0),
                ExpectedResult(args=(-1, 1), expected=0),
            ],
        )

    def test_expected_result_to_dict(self):
        """Checks that to_dict produces the expected dictionary."""
        er = ExpectedResult(args=(1, 2), expected=3)
        self.assertEqual(er.to_dict(), {"args": [1, 2], "expected": 3})

    def test_expected_result_from_dict(self):
        """Checks that from_dict reconstructs the instance."""
        data = {"args": [4, 5], "expected": 9}
        er = ExpectedResult.from_dict(data)
        self.assertEqual(er.args, (4, 5))
        self.assertEqual(er.expected, 9)

    def test_expected_result_roundtrip(self):
        """Checks that to_dict/from_dict is a lossless roundtrip."""
        er = ExpectedResult(args=(10, 20, 30), expected="hello")
        self.assertEqual(ExpectedResult.from_dict(er.to_dict()), er)

    def test_to_json_returns_string(self):
        """Checks that to_json returns a JSON string."""
        pt = self._make_prompt_test()
        result = pt.to_json()
        self.assertIsInstance(result, str)
        self.assertIn('"prompt"', result)
        self.assertIn('"expected"', result)

    def test_from_json_reconstructs_instance(self):
        """Checks that from_json reconstructs the PromptTest."""
        pt = self._make_prompt_test()
        restored = PromptTest.from_json(pt.to_json())
        self.assertEqual(restored.prompt, pt.prompt)
        self.assertEqual(len(restored.expected), 3)
        self.assertEqual(restored.expected[0].args, (1, 2))
        self.assertEqual(restored.expected[0].expected, 3)

    def test_roundtrip(self):
        """Checks that to_json/from_json is a lossless roundtrip."""
        pt = self._make_prompt_test()
        self.assertEqual(PromptTest.from_json(pt.to_json()), pt)

    def test_from_json_empty_expected(self):
        """Checks that from_json handles missing expected field."""
        restored = PromptTest.from_json('{"prompt": "do something"}')
        self.assertEqual(restored.prompt, "do something")
        self.assertEqual(restored.expected, [])

    def test_from_json_with_none_expected(self):
        """Checks that from_json handles None as expected value."""
        pt = PromptTest(
            prompt="return nothing", expected=[ExpectedResult(args=(), expected=None)]
        )
        restored = PromptTest.from_json(pt.to_json())
        self.assertEqual(restored, pt)

    def test_from_json_with_nested_structures(self):
        """Checks that from_json handles nested lists and dicts."""
        pt = PromptTest(
            prompt="return a dict", expected=[ExpectedResult(args=([1, 2],), expected={"sum": 3})]
        )
        restored = PromptTest.from_json(pt.to_json())
        self.assertEqual(restored.expected[0].expected, {"sum": 3})
        self.assertEqual(restored.expected[0].args, ([1, 2],))


if __name__ == "__main__":
    unittest.main()
