import unittest

from locodellm.bench.run_code import UNDEFINED, run_function


class TestRunFunction(unittest.TestCase):
    """Tests for run_function and RunStatus."""

    def test_compile_error(self):
        """Checks that a syntax error is captured."""
        status = run_function("def foo(:\n    pass")
        self.assertFalse(status.compiled)
        self.assertIsInstance(status.compile_error, SyntaxError)
        self.assertFalse(status.ran)
        self.assertFalse(status.success)

    def test_simple_function_no_args(self):
        """Checks that a no-arg function runs successfully."""
        status = run_function("def greet():\n    return 'hello'")
        self.assertTrue(status.compiled)
        self.assertIsNone(status.compile_error)
        self.assertTrue(status.ran)
        self.assertIsNone(status.run_error)
        self.assertEqual(status.result, "hello")
        self.assertTrue(status.success)

    def test_function_with_positional_args(self):
        """Checks that positional args receive UNDEFINED."""
        status = run_function("def add(a, b):\n    return (a, b)")
        self.assertTrue(status.success)
        self.assertEqual(status.result, (UNDEFINED, UNDEFINED))

    def test_function_with_defaults(self):
        """Checks that default values are used when present."""
        source = "def inc(x, step=1):\n    return step"
        status = run_function(source)
        self.assertTrue(status.success)
        self.assertEqual(status.result, 1)

    def test_function_with_kwargs_only(self):
        """Checks keyword-only arguments get UNDEFINED."""
        source = "def f(*, key):\n    return key"
        status = run_function(source)
        self.assertTrue(status.success)
        self.assertIs(status.result, UNDEFINED)

    def test_function_with_var_args(self):
        """Checks that *args and **kwargs don't cause failures."""
        source = "def f(*args, **kwargs):\n    return len(args)"
        status = run_function(source)
        self.assertTrue(status.success)
        self.assertEqual(status.result, 0)

    def test_runtime_error(self):
        """Checks that a runtime exception is captured."""
        source = "def f():\n    raise ValueError('boom')"
        status = run_function(source)
        self.assertTrue(status.compiled)
        self.assertFalse(status.ran)
        self.assertIsInstance(status.run_error, ValueError)
        self.assertFalse(status.success)

    def test_no_function_in_source(self):
        """Checks the error when source defines no function."""
        status = run_function("x = 42")
        self.assertTrue(status.compiled)
        self.assertFalse(status.ran)
        self.assertIsInstance(status.run_error, RuntimeError)

    def test_undefined_operations(self):
        """Checks that UNDEFINED supports common operations without raising."""
        source = "def f(x):\n    y = x + 1\n    z = x * 2\n    return z\n"
        status = run_function(source)
        self.assertTrue(status.success)
        self.assertIs(status.result, UNDEFINED)

    def test_multiple_functions_picks_first(self):
        """Checks that the first function defined is the one executed."""
        source = "def first():\n    return 'first'\n\ndef second():\n    return 'second'\n"
        status = run_function(source)
        self.assertTrue(status.success)
        self.assertEqual(status.result, "first")


if __name__ == "__main__":
    unittest.main()
