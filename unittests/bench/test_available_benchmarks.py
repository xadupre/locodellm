import unittest

from locodellm.bench import BenchPromptTest, PromptTest, get_available_benchmarks, load_benchmark


class TestAvailableBenchmarks(unittest.TestCase):
    """Tests for get_available_benchmarks and load_benchmark."""

    def test_get_available_benchmarks_returns_dict(self):
        """Checks that get_available_benchmarks returns a non-empty dict."""
        benchmarks = get_available_benchmarks()
        self.assertIsInstance(benchmarks, dict)
        self.assertGreater(len(benchmarks), 0)

    def test_basic_benchmark_is_listed(self):
        """Checks that the 'basic' benchmark is available."""
        benchmarks = get_available_benchmarks()
        self.assertIn("basic", benchmarks)
        self.assertIsInstance(benchmarks["basic"], str)
        self.assertGreater(len(benchmarks["basic"]), 0)

    def test_load_benchmark_basic(self):
        """Checks that load_benchmark('basic') returns a BenchPromptTest."""
        bench = load_benchmark("basic")
        self.assertIsInstance(bench, BenchPromptTest)
        self.assertEqual(len(bench.tests), 10)
        self.assertGreater(len(bench.description), 0)
        for t in bench.tests:
            self.assertIsInstance(t, PromptTest)
            self.assertGreater(len(t.expected), 0)

    def test_load_benchmark_unknown_raises(self):
        """Checks that load_benchmark raises KeyError for unknown names."""
        with self.assertRaises(KeyError):
            load_benchmark("nonexistent")


if __name__ == "__main__":
    unittest.main()
