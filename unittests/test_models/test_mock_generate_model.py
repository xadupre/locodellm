import os
import unittest

from locodellm.ext_test_case import ExtTestCase, skipif_no_genai
from locodellm.test_models import create_mock_generate_model


class TestMockGenerateModel(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        """Creates the mock model directory once for all tests."""
        folder = cls.get_dump_folder("mock_generate_model")
        cls.model_path = create_mock_generate_model(os.path.join(folder, "mock-llm"))

    def test_mock_model_files_exist(self):
        """Checks that the mock model directory contains the expected files."""
        for name in (
            "model.onnx",
            "genai_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            path = os.path.join(self.model_path, name)
            self.assertExists(path)

    @skipif_no_genai()
    def test_first_prompt(self):
        """Checks that the first prompt produces the expected code."""
        from locodellm import extract_code
        from locodellm.session import create_session

        session = create_session(self.model_path, chat_template="chatml")
        session.generate('write a python function which returns "hello"', max_length=200)
        code = extract_code(session.text)
        self.assertEqual(code, 'def hello():\n    return "hello"')

    @skipif_no_genai()
    def test_second_prompt(self):
        """Checks that the second prompt produces the expected code."""
        from locodellm import extract_code
        from locodellm.session import create_session

        session = create_session(self.model_path, chat_template="chatml")
        session.generate('write a python function which returns "hello"', max_length=200)
        session.generate("change hello into bonjour", max_length=500)
        code = extract_code(session.text)
        self.assertEqual(code, 'def bonjour():\n    return "bonjour"')

    @skipif_no_genai()
    def test_token_counts(self):
        """Checks the expected token counts for both turns."""
        from locodellm.session import create_session

        session = create_session(self.model_path, chat_template="chatml")
        session.generate('write a python function which returns "hello"', max_length=200)
        self.assertEqual(session.tokens.size, 56)

        session.generate("change hello into bonjour", max_length=500)
        self.assertEqual(session.tokens.size, 114)


if __name__ == "__main__":
    unittest.main()
