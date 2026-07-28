import os
import shutil
import tempfile
import unittest

from locodellm.test_models import create_tiny_model


class TestTinyModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Creates a temporary tiny-llm model directory once for all tests."""
        cls._tmpdir = tempfile.mkdtemp(prefix="tiny_llm_")
        cls.model_path = create_tiny_model(os.path.join(cls._tmpdir, "tiny-llm"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_tiny_model_files_exist(self):
        """Checks that the tiny model directory contains the expected files."""
        for name in (
            "model.onnx",
            "genai_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            path = os.path.join(self.model_path, name)
            self.assertTrue(os.path.isfile(path), f"missing {name}")


if __name__ == "__main__":
    unittest.main()
