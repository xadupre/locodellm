import os
import unittest

from locodellm.ext_test_case import ExtTestCase
from locodellm.test_models import create_tiny_model


class TestTinyModel(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        """Creates a tiny-llm model directory once for all tests."""
        folder = cls.get_dump_folder("tiny_llm_model")
        cls.model_path = create_tiny_model(os.path.join(folder, "tiny-llm"))

    def test_tiny_model_files_exist(self):
        """Checks that the tiny model directory contains the expected files."""
        for name in (
            "model.onnx",
            "genai_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            path = os.path.join(self.model_path, name)
            self.assertExists(path)


if __name__ == "__main__":
    unittest.main()
