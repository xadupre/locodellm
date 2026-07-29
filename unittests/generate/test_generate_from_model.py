import os
import tempfile
import unittest

from locodellm.ext_test_case import ExtTestCase, skipif_no_genai
from locodellm.generate.generate_from_model import (
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

    def _inject_fake_modelbuilder(self, calls):
        """Replaces modelbuilder.builder in sys.modules with a fake."""
        import sys
        import types

        def fake_create_model(**kwargs):
            calls.append(kwargs)

        fake_parent = types.ModuleType("modelbuilder")
        fake_builder = types.ModuleType("modelbuilder.builder")
        fake_builder.create_model = fake_create_model
        fake_parent.builder = fake_builder

        originals = {
            "modelbuilder": sys.modules.get("modelbuilder"),
            "modelbuilder.builder": sys.modules.get("modelbuilder.builder"),
        }
        sys.modules["modelbuilder"] = fake_parent
        sys.modules["modelbuilder.builder"] = fake_builder
        return originals

    def _restore_modelbuilder(self, originals):
        """Restores the original modelbuilder modules."""
        import sys

        for key, value in originals.items():
            if value is not None:
                sys.modules[key] = value
            else:
                sys.modules.pop(key, None)

    def test_convert_model_success(self):
        """Checks that _convert_model passes correct args to create_model."""
        from locodellm.generate.generate_from_model import _convert_model

        calls = []
        originals = self._inject_fake_modelbuilder(calls)
        try:
            _convert_model(
                "Qwen/Qwen2.5-Coder-0.5B-Instruct", "/tmp/out", precision="fp16", verbose=1
            )
        finally:
            self._restore_modelbuilder(originals)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model_name"], "Qwen/Qwen2.5-Coder-0.5B-Instruct")
        self.assertEqual(calls[0]["output_dir"], "/tmp/out")
        self.assertEqual(calls[0]["precision"], "fp16")

    def test_convert_model_default_precision(self):
        """Checks that _convert_model defaults to fp32 precision."""
        from locodellm.generate.generate_from_model import _convert_model

        calls = []
        originals = self._inject_fake_modelbuilder(calls)
        try:
            _convert_model("some/model", "/tmp/out")
        finally:
            self._restore_modelbuilder(originals)

        self.assertEqual(calls[0]["precision"], "fp32")

    def test_download_and_convert(self):
        """Checks _download_and_convert delegates to _convert_model."""
        from locodellm.generate.generate_from_model import _download_and_convert

        calls = []
        originals = self._inject_fake_modelbuilder(calls)
        model_dir = os.path.join(self._tmpdir, "model_out")
        try:
            _download_and_convert(
                "owner/repo", model_dir, self._tmpdir, "model", precision="int4", verbose=1
            )
        finally:
            self._restore_modelbuilder(originals)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model_name"], "owner/repo")
        self.assertEqual(calls[0]["output_dir"], model_dir)
        self.assertEqual(calls[0]["precision"], "int4")

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model_name"], "owner/repo")
        self.assertEqual(calls[0]["output_dir"], model_dir)
        self.assertEqual(calls[0]["precision"], "int4")


if __name__ == "__main__":
    unittest.main()
