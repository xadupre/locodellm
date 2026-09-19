import io
import os
import subprocess
import sys
import tempfile
import unittest

from locodellm.ext_test_case import ExtTestCase, requires_onnxruntime_genai
from locodellm.generate.chat import chat


class RecordingOutput(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flushed = []

    def flush(self):
        self.flushed.append(self.getvalue())


class Tokenizer:
    def __init__(self):
        self.streams = []

    def encode(self, text):
        return list(text.encode("utf-8"))

    def create_stream(self):
        import codecs

        class Decoder:
            def __init__(self):
                self.decoder = codecs.getincrementaldecoder("utf-8")()

            def decode(self, token):
                return self.decoder.decode(bytes([token]))

        stream = Decoder()
        self.streams.append(stream)
        return stream


class Generator:
    def __init__(self, output):
        self.output = output
        self.history = []
        self.contexts = []
        self.appended = []
        self.rewinds = []
        self.pending = []

    def append_tokens(self, tokens):
        self.appended.append(tokens)
        self.history.extend(tokens)
        self.contexts.append(bytes(self.history).decode("utf-8"))
        self.pending = list("café".encode())

    def rewind_to(self, length):
        self.rewinds.append(length)
        self.history = self.history[:length]
        self.pending = []

    def is_done(self):
        return not self.pending

    def generate_next_token(self):
        # Output must be flushed before requesting the next token.
        assert self.output.flushed[-1] == self.output.getvalue()
        self.token = self.pending.pop(0)
        self.history.append(self.token)

    def get_next_tokens(self):
        return [self.token]

    def get_sequence(self, index):
        assert index == 0
        return self.history


class TestChat(ExtTestCase):
    def run_chat(self, prompts, max_length=1000, chat_template=None):
        output = RecordingOutput()
        generator = Generator(output)
        tokenizer = Tokenizer()
        chat(
            generator,
            tokenizer,
            max_length,
            chat_template=chat_template,
            input_stream=io.StringIO(prompts),
            output_stream=output,
        )
        return generator, tokenizer, output

    def test_many_turns_streaming_and_clear(self):
        generator, tokenizer, output = self.run_chat(
            "one\n \ntwo\nthree\n/clear\nfour\n/quit\nignored\n"
        )
        self.assertEqual(generator.contexts, ["one", "onecafétwo", "onecafétwocaféthree", "four"])
        self.assertEqual(
            [bytes(t).decode() for t in generator.appended], ["one", "two", "three", "four"]
        )
        self.assertEqual(generator.rewinds, [0])
        self.assertEqual(len(tokenizer.streams), 2)
        self.assertEqual(output.getvalue().count("Assistant: café"), 4)
        self.assertTrue(any(text.endswith("Assistant: c") for text in output.flushed))
        self.assertTrue(any(text.endswith("Assistant: ca") for text in output.flushed))
        self.assertNotIn("\ufffd", output.getvalue())

    def test_chatml_and_reset(self):
        generator, _, _ = self.run_chat("one\ntwo\n/clear\none\n", chat_template="chatml")
        prompts = [bytes(tokens).decode() for tokens in generator.appended]
        first = "<|im_start|>user\none<|im_end|>\n<|im_start|>assistant\n"
        self.assertEqual(prompts[0], first)
        self.assertEqual(
            prompts[1], "<|im_end|>\n<|im_start|>user\ntwo<|im_end|>\n<|im_start|>assistant\n"
        )
        self.assertEqual(prompts[2], first)

    def test_limit_preserves_history_and_allows_clear(self):
        generator, _, output = self.run_chat("one\ntwo\n/clear\ntwo\n", max_length=10)
        self.assertEqual(generator.contexts, ["one", "two"])
        self.assertIn("Context limit reached", output.getvalue())
        self.assertEqual(output.getvalue().count("Assistant: café"), 2)

    def test_chatml_retained_eos(self):
        class EosGenerator(Generator):
            def append_tokens(self, tokens):
                super().append_tokens(tokens)
                self.pending.extend(b"<|im_end|>")

        output = RecordingOutput()
        generator = EosGenerator(output)
        chat(
            generator,
            Tokenizer(),
            1000,
            chat_template="chatml",
            input_stream=io.StringIO("one\ntwo\n"),
            output_stream=output,
        )
        self.assertEqual(
            bytes(generator.appended[1]).decode(),
            "\n<|im_start|>user\ntwo<|im_end|>\n<|im_start|>assistant\n",
        )

    def test_oversized_prompt_and_empty_input(self):
        generator, _, output = self.run_chat("12345\n\n/clear\n/clear\n", max_length=5)
        self.assertEqual(generator.appended, [])
        self.assertEqual(generator.rewinds, [0, 0])
        self.assertIn("Context limit reached", output.getvalue())
        generator, _, _ = self.run_chat("")
        self.assertEqual(generator.appended, [])

    def test_invalid_options(self):
        with self.assertRaisesRegex(ValueError, "Unknown chat template"):
            self.run_chat("", chat_template="unknown")
        with self.assertRaisesRegex(ValueError, "must be positive"):
            self.run_chat("", max_length=0)

    def test_cli_help_and_validation(self):
        result = subprocess.run(
            [sys.executable, "-m", "locodellm", "chat", "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("/clear", result.stdout)
        result = subprocess.run(
            [sys.executable, "-m", "locodellm", "chat", "unused", "--max-length", "0"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--max-length must be positive", result.stderr)


@requires_onnxruntime_genai()
class TestChatGenAI(ExtTestCase):
    def test_real_generator_cache_and_cli(self):
        """Checks cached continuation after EOS and clearing with real GenAI."""
        import numpy
        import onnx
        import onnxruntime_genai

        from locodellm.session import create_session
        from locodellm.test_models import create_tiny_model

        with tempfile.TemporaryDirectory() as folder:
            model_path = create_tiny_model(folder)
            path = os.path.join(model_path, "model.onnx")
            model = onnx.load(path)
            embedding = numpy.eye(32, 64, dtype=numpy.float32)
            projection = numpy.zeros((64, 32), dtype=numpy.float32)
            projection[:, 2] = 1
            for prompt in range(3, 7):
                projection[prompt, prompt + 4] = 2
            model.graph.initializer[0].CopyFrom(
                onnx.numpy_helper.from_array(embedding, name="embed_weight")
            )
            model.graph.initializer[1].CopyFrom(
                onnx.numpy_helper.from_array(projection, name="proj_weight")
            )
            onnx.save(model, path)
            session = create_session(model_path)
            params = onnxruntime_genai.GeneratorParams(session.model)
            params.set_search_options(max_length=128)
            generator = onnxruntime_genai.Generator(session.model, params)
            output = io.StringIO()
            chat(
                generator,
                session.tokenizer,
                128,
                input_stream=io.StringIO("3\n4\n5\n"),
                output_stream=output,
            )
            self.assertEqual(
                [token for token in generator.get_sequence(0) if token != 2], [3, 7, 4, 8, 5, 9]
            )
            for answer in ("7", "8", "9"):
                self.assertIn(f"Assistant: {answer}", output.getvalue())
            chat(
                generator,
                session.tokenizer,
                128,
                input_stream=io.StringIO("/clear\n6\n"),
                output_stream=output,
            )
            self.assertEqual(
                [token for token in generator.get_sequence(0) if token != 2], [6, 10]
            )
            result = subprocess.run(
                [sys.executable, "-m", "locodellm", "chat", model_path, "--max-length", "128"],
                input="3\n4\n/clear\n5\n/quit\n6\n",
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            self.assertEqual(result.stdout.count("Assistant: "), 3)
            self.assertIn("History cleared.", result.stdout)


if __name__ == "__main__":
    unittest.main()
