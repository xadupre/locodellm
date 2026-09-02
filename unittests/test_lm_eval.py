import json
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from locodellm.__main__ import _cmd_bench, _cmd_lm_eval, main


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


class _Config:
    def __init__(self, model):
        self.model = model
        self.providers = []
        self.provider_options = []
        self.overlays = []

    def clear_providers(self):
        self.providers.clear()

    def append_provider(self, provider):
        self.providers.append(provider)

    def set_provider_option(self, provider, name, value):
        self.provider_options.append((provider, name, value))

    def overlay(self, value):
        self.overlays.append(json.loads(value))


class _RejectingConfig(_Config):
    def overlay(self, value):
        raise RuntimeError("Expected a string but saw a number")


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
        with patch("locodellm.lm_eval.get_session", return_value=session) as get_session:
            model = self.model_type(
                "model",
                provider="CUDAExecutionProvider",
                provider_options={"device_id": "1"},
                session_options={"intra_op_num_threads": 4},
                max_length=20,
            )
        self.assertEqual(get_session.call_args.kwargs["provider_options"], {"device_id": "1"})
        self.assertEqual(
            get_session.call_args.kwargs["session_options"], {"intra_op_num_threads": 4}
        )

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


class TestSessionProviderOptions(unittest.TestCase):
    def test_provider_options(self):
        """Checks that provider options are applied to ONNX Runtime GenAI."""
        from locodellm.session.session_state import create_session

        runtime = SimpleNamespace(
            Config=_Config, Model=lambda config: config, Tokenizer=lambda model: object()
        )
        with patch.dict(sys.modules, {"onnxruntime_genai": runtime}):
            session = create_session(
                "model",
                providers=["CUDAExecutionProvider"],
                provider_options={"device_id": "1"},
                session_options={"intra_op_num_threads": 4},
            )
        self.assertEqual(session.model.providers, ["CUDAExecutionProvider"])
        self.assertEqual(
            session.model.provider_options, [("CUDAExecutionProvider", "device_id", "1")]
        )
        self.assertEqual(
            session.model.overlays,
            [{"model": {"decoder": {"session_options": {"intra_op_num_threads": 4}}}}],
        )

    def test_session_options_without_provider(self):
        """Checks that session options do not require an execution provider."""
        from locodellm.session.session_state import create_session

        runtime = SimpleNamespace(
            Config=_Config, Model=lambda config: config, Tokenizer=lambda model: object()
        )
        with patch.dict(sys.modules, {"onnxruntime_genai": runtime}):
            session = create_session("model", session_options={"inter_op_num_threads": 2})
        self.assertEqual(
            session.model.overlays,
            [{"model": {"decoder": {"session_options": {"inter_op_num_threads": 2}}}}],
        )

    def test_invalid_session_option_value_has_actionable_error(self):
        """Checks that session option type errors explain how to quote string values."""
        from locodellm.session.session_state import create_session

        runtime = SimpleNamespace(
            Config=_RejectingConfig, Model=lambda config: config, Tokenizer=lambda model: object()
        )
        with (
            patch.dict(sys.modules, {"onnxruntime_genai": runtime}),
            self.assertRaisesRegex(
                ValueError, r"""--session-option 'session\.enable_moe_expert_statistics="1"'"""
            ),
        ):
            create_session("model", session_options={"session.enable_moe_expert_statistics": 1})


class TestLmEvalCommand(unittest.TestCase):
    def test_arguments(self):
        """Checks that bench retains every LM-Eval model and task option."""
        with patch("locodellm.__main__._cmd_bench") as command:
            main(
                [
                    "bench",
                    "model",
                    "gsm8k",
                    "squad",
                    "--precision",
                    "int4",
                    "--provider",
                    "CUDAExecutionProvider",
                    "--provider-option",
                    "device_id=1",
                    "--session-option",
                    "intra_op_num_threads=4",
                    "--chat-template",
                    "chatml",
                    "--max-length",
                    "1024",
                    "--num-fewshot",
                    "2",
                    "--limit",
                    "10",
                    "--verbose",
                    "1",
                ]
            )
        args = command.call_args.args[0]
        self.assertEqual(args.model, "model")
        self.assertEqual(args.benchmark, ["gsm8k", "squad"])
        self.assertEqual(args.precision, "int4")
        self.assertEqual(args.limit, 10)
        self.assertEqual(args.provider_option, [("device_id", "1")])
        self.assertEqual(args.session_option, [("intra_op_num_threads", 4)])
        self.assertEqual(args.chat_template, "chatml")
        self.assertEqual(args.max_length, 1024)
        self.assertEqual(args.num_fewshot, 2)
        self.assertEqual(args.verbose, 1)

    def test_provider_options_forwarded(self):
        """Checks that CLI provider options reach the LM-Eval adapter."""
        run_lm_eval = Mock(return_value=None)
        adapter = SimpleNamespace(run_lm_eval=run_lm_eval)
        utils = SimpleNamespace(make_table=Mock())
        args = SimpleNamespace(
            model="model",
            benchmark=["gsm8k"],
            precision=None,
            provider="CUDAExecutionProvider",
            provider_option=[("device_id", "1")],
            session_option=[("intra_op_num_threads", 4)],
            chat_template=None,
            max_length=100,
            num_fewshot=None,
            limit=10,
            verbose=0,
        )
        with patch.dict(sys.modules, {"locodellm.lm_eval": adapter, "lm_eval.utils": utils}):
            _cmd_lm_eval(args)
        self.assertEqual(run_lm_eval.call_args.kwargs["provider_options"], {"device_id": "1"})
        self.assertEqual(
            run_lm_eval.call_args.kwargs["session_options"], {"intra_op_num_threads": 4}
        )

    def test_lm_eval_runs_through_bench(self):
        """Checks that a non-built-in benchmark runs with LM-Eval."""
        args = SimpleNamespace(benchmark=["gsm8k"])
        with (
            patch("locodellm.bench.get_available_benchmarks", return_value={"basic": "built-in"}),
            patch("locodellm.__main__._cmd_lm_eval") as command,
        ):
            _cmd_bench(args)
        command.assert_called_once_with(args)


if __name__ == "__main__":
    unittest.main()
