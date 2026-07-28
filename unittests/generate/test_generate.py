import os
import unittest

from locodellm.ext_test_case import ExtTestCase
from locodellm.generate import generate
from locodellm.session import SessionState, create_session
from locodellm.test_models import create_tiny_model


class TestGenerate(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        """Creates a tiny-llm model directory once for all tests."""
        folder = cls.get_dump_folder("tiny_llm_generate")
        cls.model_path = create_tiny_model(os.path.join(folder, "tiny-llm"))

    def test_create_session(self):
        """Checks that create_session returns a SessionState."""
        session = create_session(self.model_path)
        self.assertIsInstance(session, SessionState)

    def test_create_session_with_providers(self):
        """Checks that passing an explicit provider list works."""
        session = create_session(self.model_path, providers=["CPUExecutionProvider"])
        self.assertIsInstance(session, SessionState)

    def test_create_session_with_loaded_model(self):
        """Checks that passing an already-loaded Model object works."""
        import onnxruntime_genai as og

        model = og.Model(self.model_path)
        session = create_session(model)
        self.assertIsInstance(session, SessionState)

    def test_generate_method(self):
        """Checks that session.generate returns text."""
        session = create_session(self.model_path)
        session.generate("<s>", max_length=10)
        self.assertIsInstance(session.text, str)
        self.assertGreater(session.tokens.size, 0)

    def test_generate_method_with_search_options(self):
        """Checks that extra search options are accepted."""
        session = create_session(self.model_path)
        session.generate("<s>", max_length=10, temperature=0.5, top_k=5)
        self.assertIsInstance(session, SessionState)

    def test_generate_method_multi_turn(self):
        """Checks that calling generate twice continues the conversation."""
        session = create_session(self.model_path)
        session.generate("<s>", max_length=10)
        first_token_count = session.tokens.size

        session.generate("<s>", max_length=30)
        self.assertGreater(session.tokens.size, first_token_count)

    def test_generate_two_consecutive_multi_token_prompts(self):
        """Two consecutive prompts each containing multiple tokens."""
        session = create_session(self.model_path)
        session.generate("<s> 3 4 5", max_length=15)
        first_text = session.text
        first_token_count = session.tokens.size
        self.assertGreater(first_token_count, 4, "First prompt should produce >4 tokens")

        session.generate("6 7 8 9", max_length=30)
        self.assertGreater(session.tokens.size, first_token_count)
        self.assertTrue(
            session.text.startswith(first_text),
            f"Second turn text should start with first turn text: "
            f"{session.text!r} does not start with {first_text!r}",
        )

    def test_generate_convenience_function(self):
        """Checks the backward-compatible generate() function still works."""
        session = generate(prompt="<s>", model=self.model_path, max_length=10)
        self.assertIsInstance(session, SessionState)
        self.assertGreater(session.tokens.size, 0)

        session = generate(prompt="<s>", model=session, max_length=30)
        self.assertGreater(session.tokens.size, 1)


import unittest

if __name__ == "__main__":
    unittest.main()
