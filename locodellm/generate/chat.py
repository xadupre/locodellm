"""Streams an interactive conversation using a persistent GenAI generator."""

import sys
from typing import Any, TextIO


def chat(
    generator: Any,
    tokenizer: Any,
    max_length: int,
    chat_template: str | None = None,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> None:
    """Runs a chat loop, retaining the generator's in-memory KV cache.

    Args:
        generator: A fresh ``onnxruntime_genai.Generator`` with its search
            ``max_length`` set to *max_length*.
        tokenizer: The model's ``onnxruntime_genai.Tokenizer``.
        max_length: Maximum total conversation tokens, including answers.
        chat_template: ``"chatml"`` or ``None`` for unformatted prompts.
        input_stream: Input lines, defaulting to standard input.
        output_stream: Streamed output, defaulting to standard output.
    """
    if chat_template not in (None, "chatml"):
        raise ValueError(f"Unknown chat template: {chat_template!r}")
    if max_length < 1:
        raise ValueError("max_length must be positive")
    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stdout if output_stream is None else output_stream
    decoder = tokenizer.create_stream()
    token_count = 0
    print("Enter a prompt, /clear to clear history, or /quit to exit.", file=output_stream)

    while True:
        print("\nYou: ", end="", file=output_stream, flush=True)
        line = input_stream.readline()
        if not line:
            print(file=output_stream)
            return
        prompt = line.rstrip("\r\n")
        command = prompt.strip()
        if command == "/quit":
            return
        if command == "/clear":
            generator.rewind_to(0)
            decoder = tokenizer.create_stream()
            token_count = 0
            print("History cleared.", file=output_stream)
            continue
        if not command:
            continue

        if chat_template == "chatml":
            prefix = ""
            if token_count:
                end_ids = tokenizer.encode("<|im_end|>")
                history = generator.get_sequence(0)
                # GenAI versions differ in whether the sequence retains EOS.
                prefix = (
                    "\n" if list(history[-len(end_ids) :]) == list(end_ids) else "<|im_end|>\n"
                )
            prompt = f"{prefix}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        prompt_ids = tokenizer.encode(prompt)
        if token_count + len(prompt_ids) >= max_length:
            print(
                "Context limit reached. Use /clear or enter a shorter prompt.", file=output_stream
            )
            continue

        generator.append_tokens(prompt_ids)
        print("Assistant: ", end="", file=output_stream, flush=True)
        while not generator.is_done():
            generator.generate_next_token()
            token = int(generator.get_next_tokens()[0])
            print(decoder.decode(token), end="", file=output_stream, flush=True)
        token_count = len(generator.get_sequence(0))
        print(file=output_stream, flush=True)
