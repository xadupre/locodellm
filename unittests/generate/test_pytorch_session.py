import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest

import numpy

from locodellm.__main__ import main
from locodellm.generate.generate_from_model import _model_path_cache, _session_cache, get_session


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TestPyTorchSession(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Creates a real, deterministic Transformers checkpoint without downloads."""
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

        cls.directory = tempfile.TemporaryDirectory()
        cls.model_path = cls.directory.name
        vocabulary = {
            "answer": 0,
            "[UNK]": 1,
            "[PAD]": 2,
            "[EOS]": 3,
            "<|user|>": 4,
            "<|assistant|>": 5,
            "<|end|>": 6,
        }
        tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            eos_token="[EOS]",
            additional_special_tokens=["<|user|>", "<|assistant|>", "<|end|>"],
        )
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ '<|' + message['role'] + '|>\\n' + message['content'] + '<|end|>\\n' }}"
            "{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
        )
        tokenizer.save_pretrained(cls.model_path)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(vocabulary),
                n_positions=1024,
                n_embd=8,
                n_layer=1,
                n_head=1,
                eos_token_id=3,
                pad_token_id=2,
                bos_token_id=None,
            )
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        model.save_pretrained(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        _session_cache.clear()
        cls.directory.cleanup()

    def setUp(self):
        _session_cache.clear()

    def test_native_loading_generation_and_history(self):
        """Checks native chat formatting, token limits, caching and fresh conversations."""
        session = get_session(self.model_path, backend="pytorch", device="cpu")
        self.assertIs(session, get_session(self.model_path, backend="pytorch", device="cpu"))
        self.assertFalse(session.model.training)
        self.assertEqual(session._wrap_prompt("hello"), "<|user|>\nhello<|end|>\n<|assistant|>\n")
        prompt_length = session.get_prompt_length("hello")
        session.generate("hello", max_length=prompt_length + 3)
        self.assertEqual(session.generated_token_count, 3)
        self.assertEqual(session.text, "answer answer answer")
        self.assertEqual(len(session.tokens), prompt_length + 3)
        second_prompt = session._wrap_prompt("again")
        self.assertEqual(second_prompt.count("<|user|>"), 2)
        self.assertIn("answer answer answer", second_prompt)
        second_length = session.get_prompt_length("again")
        session.generate("again", max_new_tokens=2)
        self.assertEqual(len(session.tokens), second_length + 2)
        self.assertEqual(session.generated_token_count, 2)
        fresh = session.new_session()
        self.assertIs(fresh.model, session.model)
        self.assertEqual(fresh.messages, [])
        self.assertEqual(fresh.tokens.size, 0)
        self.assertEqual(fresh.get_prompt_length("hello"), prompt_length)

    def test_chatml_and_loading_precision(self):
        """Checks explicit ChatML and separates cached sessions with different settings."""
        import torch

        native = get_session(self.model_path, backend="pytorch")
        chatml = get_session(
            self.model_path, backend="pytorch", precision="fp32", chat_template="chatml"
        )
        self.assertIsNot(native, chatml)
        self.assertEqual(chatml.model.dtype, torch.float32)
        chatml.generate("hello", max_new_tokens=2)
        history = chatml.tokens.copy()
        chatml.generate("again", max_new_tokens=1)
        numpy.testing.assert_array_equal(chatml.tokens[: len(history)], history)
        self.assertEqual(chatml.generated_token_count, 1)

    def test_generation_errors(self):
        """Rejects invalid per-turn limits and prompts exceeding the total limit."""
        session = get_session(self.model_path, backend="pytorch")
        with self.assertRaisesRegex(ValueError, "must be positive"):
            session.generate("hello", max_new_tokens=0)
        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            session.generate("hello", max_length=session.get_prompt_length("hello"))
        self.assertEqual(session.messages, [])

    def test_eos_and_untemplated_history(self):
        """Strips terminal EOS tokens and preserves history without a chat template."""
        session = get_session(self.model_path, backend="pytorch")
        session.generate("hello", max_new_tokens=3, eos_token_id=[0])
        self.assertEqual(session.text, "")
        self.assertEqual(session.generated_token_count, 0)
        session.tokenizer.chat_template = None
        fresh = session.new_session()
        fresh.generate("hello", max_new_tokens=2)
        history = fresh.tokens.copy()
        fresh.generate("again", max_new_tokens=1)
        numpy.testing.assert_array_equal(fresh.tokens[: len(history)], history)
        self.assertEqual(fresh.text, "answer")

    def test_automatic_device_placement(self):
        """Loads and generates with Accelerate's automatic device mapping."""
        session = get_session(self.model_path, backend="pytorch", device="auto")
        session.generate("hello", max_new_tokens=1)
        self.assertEqual(session.text, "answer")

    def test_bench_cli_and_exports(self):
        """Runs the actual benchmark CLI and exports with real local PyTorch weights."""
        import pandas

        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "bench.csv")
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                main(
                    [
                        "bench",
                        self.model_path,
                        "basic",
                        "--backend",
                        "pytorch",
                        "--device",
                        "cpu",
                        "--harness",
                        "python",
                        "--max-length",
                        "2",
                        "--output",
                        output,
                    ]
                )
            data = pandas.read_csv(output)
            with open(os.path.join(directory, "bench.json"), encoding="utf-8") as stream:
                trace = json.load(stream)
            with pandas.ExcelFile(os.path.join(directory, "bench.xlsx")) as workbook:
                self.assertEqual(workbook.sheet_names, ["aggregated", "raw_data"])
            self.assertEqual(set(data["backend"]), {"pytorch"})
            self.assertEqual(set(data["device"]), {"cpu"})
            self.assertEqual(len(trace), 10)
            self.assertGreater(trace[0]["token_count"], 0)
            self.assertIn("Statistics", stdout.getvalue())

    def test_generate_cli_and_harness_exports(self):
        """Generates and exports real native-model answers with and without the harness."""
        session = get_session(self.model_path, backend="pytorch", device="cpu")
        for harness in (False, True):
            with self.subTest(harness=harness), tempfile.TemporaryDirectory() as directory:
                output = os.path.join(directory, "events.json")
                limit = 2 if harness else session.get_prompt_length("hello") + 2
                args = [
                    "generate",
                    self.model_path,
                    "hello",
                    "--backend",
                    "pytorch",
                    "--device",
                    "cpu",
                    "--max-length",
                    str(limit),
                    "--output",
                    output,
                ]
                if harness:
                    args += ["--harness", "python"]
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    main(args)
                with open(output, encoding="utf-8") as stream:
                    document = json.load(stream)
                self.assertEqual(stdout.getvalue().strip(), "answer answer")
                self.assertEqual(document["text"], "answer answer")
                self.assertEqual(document["token_count"], 2)
                self.assertIsNone(document["error"])
                self.assertEqual(document["tool_calls"], [])
                option = "max_new_tokens" if harness else "max_length"
                self.assertEqual(document["initial_generation_options"], {option: limit})

    def test_python_harness(self):
        """Accepts direct answers while preserving per-turn generation limits."""
        from locodellm.bench.python_harness import PythonHarness

        session = get_session(self.model_path, backend="pytorch")
        result = PythonHarness().run(session, "hello", max_length=2)
        self.assertEqual(result.text, "answer answer")
        self.assertEqual(result.token_count, 2)
        self.assertEqual(result.tool_calls, [])
        self.assertIsNone(result.error)

    def test_qwen35_text_generation(self):
        """Loads a tiny multimodal Qwen3.5 checkpoint and generates text offline."""
        import torch
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        from transformers import Qwen3_5TextConfig, Qwen3_5VisionConfig

        text = Qwen3_5TextConfig(
            vocab_size=7,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            layer_types=["linear_attention", "full_attention"],
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            eos_token_id=3,
            pad_token_id=2,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 0],
            },
        )
        vision = Qwen3_5VisionConfig(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            out_hidden_size=16,
            num_position_embeddings=16,
        )
        model = Qwen3_5ForConditionalGeneration(
            Qwen3_5Config(text_config=text, vision_config=vision)
        )
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            session = get_session(self.model_path, backend="pytorch")
            session.tokenizer.save_pretrained(directory)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            qwen = get_session(
                directory,
                backend="pytorch",
                device=device,
                precision="bf16" if device == "cuda" else "fp32",
            )
            self.assertIsInstance(qwen.model, Qwen3_5ForConditionalGeneration)
            qwen.generate("hello", max_new_tokens=2, eos_token_id=None)
            self.assertEqual(qwen.generated_token_count, 2)


class TestBackendValidation(unittest.TestCase):
    def test_rejects_incompatible_cli_options(self):
        """Rejects mixed backend options before loading models or optional dependencies."""
        cases = [
            (["--device", "cpu"], "--device requires"),
            (["--backend", "pytorch", "--provider", "CPUExecutionProvider"], "ONNX-only"),
            (["--backend", "pytorch", "--provider-option", "device_id=0"], "ONNX-only"),
            (["--backend", "pytorch", "--session-option", "x=1"], "ONNX-only"),
            (["--backend", "pytorch", "--precision", "int4"], "pre-quantized"),
        ]
        for options, message in cases:
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit) as error:
                    main(["bench", "unused/model", "basic", *options])
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, err.getvalue())

    def test_rejects_incompatible_session_options(self):
        """Rejects invalid programmatic backend options without attempting a download."""
        cases = [
            ({"backend": "other"}, "Unknown backend"),
            ({"device": "cpu"}, "--device requires"),
            ({"backend": "pytorch", "provider": "CPUExecutionProvider"}, "ONNX-only"),
            ({"backend": "pytorch", "precision": "int4"}, "pre-quantized"),
        ]
        paths = dict(_model_path_cache)
        for options, message in cases:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                get_session("unused/model", **options)
        self.assertEqual(_model_path_cache, paths)


if __name__ == "__main__":
    unittest.main()
