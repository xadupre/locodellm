import os
import tempfile
import unittest

from locodellm.ext_test_case import ExtTestCase, skipif_no_genai
from locodellm.general.generate_from_model import (
    _model_path_cache,
    _session_cache,
    generate_from_model,
)


class TestGenerateFromModel(ExtTestCase):
    """Tests for generate_from_model."""

    def setUp(self):
        """Clears caches between tests."""
        _session_cache.clear()
        _model_path_cache.clear()
        self._tmpdir = tempfile.mkdtemp()

    @skipif_no_genai()
    def test_mock_generate_model(self):
        """Checks that mock/generate produces expected output."""
        from locodellm import extract_code

        session = generate_from_model(
            "mock/generate",
            prompt='write a python function which returns "hello"',
            chat_template="chatml",
            cache_dir=self._tmpdir,
            max_length=200,
        )
        code = extract_code(session.text)
        self.assertIn("def hello", code)

    @skipif_no_genai()
    def test_model_is_cached(self):
        """Checks that repeated calls reuse the cached model."""
        generate_from_model(
            "mock/generate",
            prompt='write a python function which returns "hello"',
            chat_template="chatml",
            cache_dir=self._tmpdir,
            max_length=200,
        )
        # Second call should hit the cache.
        model_path = _model_path_cache.get(("mock/generate", None))
        self.assertIsNotNone(model_path)
        self.assertTrue(os.path.exists(model_path))

        # Session cache should also be populated.
        self.assertIn(("mock/generate", None, None), _session_cache)

    @skipif_no_genai()
    def test_conversion_not_repeated(self):
        """Checks that the ONNX model is not regenerated on second call."""
        generate_from_model(
            "mock/generate",
            prompt='write a python function which returns "hello"',
            chat_template="chatml",
            cache_dir=self._tmpdir,
            max_length=200,
        )
        model_path = _model_path_cache[("mock/generate", None)]
        onnx_file = os.path.join(model_path, "model.onnx")
        mtime = os.path.getmtime(onnx_file)

        # Clear session cache but keep path cache to simulate a new process
        # that still has the files on disk.
        _session_cache.clear()
        _model_path_cache.clear()

        generate_from_model(
            "mock/generate",
            prompt='write a python function which returns "hello"',
            chat_template="chatml",
            cache_dir=self._tmpdir,
            max_length=200,
        )
        self.assertEqual(os.path.getmtime(onnx_file), mtime)

    @skipif_no_genai()
    def test_verbose_output(self):
        """Checks that verbose mode does not crash."""
        session = generate_from_model(
            "mock/generate",
            prompt='write a python function which returns "hello"',
            chat_template="chatml",
            cache_dir=self._tmpdir,
            max_length=200,
            verbose=1,
        )
        self.assertGreater(len(session.text), 0)

    def test_unknown_model_raises(self):
        """Checks that an unknown model id raises KeyError."""
        with self.assertRaises(KeyError):
            generate_from_model("mock/nonexistent", prompt="hello", cache_dir=self._tmpdir)


if __name__ == "__main__":
    unittest.main()
