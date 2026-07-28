import os
import shutil
import tempfile
import unittest

from locodellm.generate import generate
from locodellm.session import SessionState
from locodellm.test_models import create_tiny_model


class TestGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Creates a temporary tiny-llm model directory once for all tests."""
        cls._tmpdir = tempfile.mkdtemp(prefix="tiny_llm_")
        cls.model_path = create_tiny_model(os.path.join(cls._tmpdir, "tiny-llm"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_generate_returns_session_state(self):
        """Checks that generate returns a SessionState with text."""
        session = generate(prompt="<s>", model=self.model_path, max_length=10)
        self.assertIsInstance(session, SessionState)
        self.assertIsInstance(session.text, str)
        self.assertGreater(session.tokens.size, 0)

    def test_generate_with_providers(self):
        """Checks that passing an explicit provider list works."""
        session = generate(
            prompt="<s>", model=self.model_path, providers=["CPUExecutionProvider"], max_length=10
        )
        self.assertIsInstance(session, SessionState)

    def test_generate_with_search_options(self):
        """Checks that extra search options are accepted."""
        session = generate(
            prompt="<s>", model=self.model_path, max_length=10, temperature=0.5, top_k=5
        )
        self.assertIsInstance(session, SessionState)

    def test_generate_with_loaded_model(self):
        """Checks that passing an already-loaded Model object works."""
        import onnxruntime_genai as og

        model = og.Model(self.model_path)
        session = generate(prompt="<s>", model=model, max_length=10)
        self.assertIsInstance(session, SessionState)

    def test_generate_multi_turn(self):
        """Checks that a SessionState can be reused for a second prompt."""
        session = generate(prompt="<s>", model=self.model_path, max_length=10)
        first_token_count = session.tokens.size

        session = generate(prompt="<s>", model=session, max_length=30)
        self.assertGreater(session.tokens.size, first_token_count)

    def test_generate_two_consecutive_multi_token_prompts(self):
        """Two consecutive prompts each containing multiple tokens."""
        # First prompt: "<s> 3 4 5" encodes to at least 4 tokens
        session = generate(prompt="<s> 3 4 5", model=self.model_path, max_length=15)
        self.assertIsInstance(session, SessionState)
        first_text = session.text
        first_token_count = session.tokens.size
        self.assertGreater(first_token_count, 4, "First prompt should produce >4 tokens")

        # Second prompt: "6 7 8 9" also multiple tokens, continues the session
        session = generate(prompt="6 7 8 9", model=session, max_length=30)
        self.assertIsInstance(session, SessionState)
        self.assertGreater(session.tokens.size, first_token_count)
        # The full text should contain the first generation
        self.assertTrue(
            session.text.startswith(first_text),
            f"Second turn text should start with first turn text: "
            f"{session.text!r} does not start with {first_text!r}",
        )


if __name__ == "__main__":
    unittest.main()
