import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from locodellm.__main__ import main


class _LM:
    def __init__(self):
        pass


lm_eval = types.ModuleType("lm_eval")
lm_eval_api = types.ModuleType("lm_eval.api")
lm_eval_model = types.ModuleType("lm_eval.api.model")
lm_eval_model.LM = _LM


class _Tokenizer:
    def encode(self, text):
        return text.split()


class _Session:
    def __init__(self, text="answer STOP ignored"):
        self.tokenizer = _Tokenizer()
        self.text = text
        self.calls = []

    def new_session(self):
        return self

    def _wrap_prompt(self, prompt):
        return prompt

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self


class TestOnnxRuntimeGenAILM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = patch.dict(
            sys.modules,
            {"lm_eval": lm_eval, "lm_eval.api": lm_eval_api, "lm_eval.api.model": lm_eval_model},
        )
        cls.modules.start()
        from locodellm.lm_eval import OnnxRuntimeGenAILM

        cls.model_type = OnnxRuntimeGenAILM

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("locodellm.lm_eval", None)
        cls.modules.stop()

    def test_generate_until(self):
        """Checks generation length, search options, and stop sequences."""
        session = _Session()
        with patch("locodellm.lm_eval.get_session", return_value=session):
            model = self.model_type("model", max_length=20)

        request = SimpleNamespace(
            args=("two token", {"max_gen_toks": 5, "until": ["STOP"], "temperature": 0.5})
        )
        self.assertEqual(model.generate_until([request]), ["answer "])
        self.assertEqual(session.calls, [("two token", {"max_length": 7, "temperature": 0.5})])

    def test_unsupported_likelihood(self):
        """Checks that likelihood tasks report the backend limitation."""
        with patch("locodellm.lm_eval.get_session", return_value=_Session()):
            model = self.model_type("model")
        with self.assertRaises(NotImplementedError):
            model.loglikelihood([])
        with self.assertRaises(NotImplementedError):
            model.loglikelihood_rolling([])

    def test_prompt_exceeds_max_length(self):
        """Checks that an oversized prompt produces a clear error."""
        with patch("locodellm.lm_eval.get_session", return_value=_Session()):
            model = self.model_type("model", max_length=2)
        request = SimpleNamespace(args=("two tokens", {"max_gen_toks": 1}))
        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            model.generate_until([request])


class TestLmEvalCommand(unittest.TestCase):
    def test_arguments(self):
        """Checks that the LM-Eval command parses model and task options."""
        with patch("locodellm.__main__._cmd_lm_eval") as command:
            main(["lm-eval", "model", "gsm8k", "squad", "--limit", "10"])
        args = command.call_args.args[0]
        self.assertEqual(args.model, "model")
        self.assertEqual(args.tasks, ["gsm8k", "squad"])
        self.assertEqual(args.limit, 10)


if __name__ == "__main__":
    unittest.main()
