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
        session = generate(
            prompt="<s>",
            model=self.model_path,
            max_length=10,
        )
        self.assertIsInstance(session, SessionState)
        self.assertIsInstance(session.text, str)
        self.assertGreater(session.tokens.size, 0)

    def test_generate_with_providers(self):
        """Checks that passing an explicit provider list works."""
        session = generate(
            prompt="<s>",
            model=self.model_path,
            providers=["CPUExecutionProvider"],
            max_length=10,
        )
        self.assertIsInstance(session, SessionState)

    def test_generate_with_search_options(self):
        """Checks that extra search options are accepted."""
        session = generate(
            prompt="<s>",
            model=self.model_path,
            max_length=10,
            temperature=0.5,
            top_k=5,
        )
        self.assertIsInstance(session, SessionState)

    def test_generate_with_loaded_model(self):
        """Checks that passing an already-loaded Model object works."""
        import onnxruntime_genai as og

        model = og.Model(self.model_path)
        session = generate(
            prompt="<s>",
            model=model,
            max_length=10,
        )
        self.assertIsInstance(session, SessionState)

    def test_generate_multi_turn(self):
        """Checks that a SessionState can be reused for a second prompt."""
        session = generate(
            prompt="<s>",
            model=self.model_path,
            max_length=10,
        )
        first_token_count = session.tokens.size

        session = generate(
            prompt="<s>",
            model=session,
            max_length=30,
        )
        self.assertGreater(session.tokens.size, first_token_count)


if __name__ == "__main__":
    unittest.main()
