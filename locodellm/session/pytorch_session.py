"""Runs generation directly with PyTorch and Hugging Face Transformers."""

from __future__ import annotations

import sys
import time
from typing import Any

import numpy

from locodellm.session.session_state import SessionState


class GenerationProgress:
    """Reports token progress without buffering generated text or starting threads."""

    def __init__(self, max_new_tokens: int) -> None:
        self.max_new_tokens = max_new_tokens
        self.token_count = 0
        self.skip_prompt = True
        self.started_at = time.monotonic()
        self.last_report = self.started_at

    def put(self, tokens: Any) -> None:
        """Reports the prefill phase and periodically reports generated token counts."""
        if self.skip_prompt:
            self.skip_prompt = False
            print("[generate] waiting for first token", file=sys.stderr, flush=True)
            return
        self.token_count += tokens.numel()
        now = time.monotonic()
        if self.token_count == 1 or now - self.last_report >= 1:
            print(
                f"[generate] generated {self.token_count}/{self.max_new_tokens} tokens "
                f"({now - self.started_at:.1f}s)",
                file=sys.stderr,
                flush=True,
            )
            self.last_report = now

    def end(self) -> None:
        """Reports generation completion."""
        print(
            f"[generate] generation finished: {self.token_count} tokens "
            f"({time.monotonic() - self.started_at:.1f}s)",
            file=sys.stderr,
            flush=True,
        )


class PyTorchSessionState(SessionState):
    """Preserves the benchmark session interface for a Transformers model."""

    def __init__(
        self, model: Any, tokenizer: Any, verbose: int = 0, chat_template: str | None = None
    ) -> None:
        super().__init__(model, tokenizer, verbose=verbose, chat_template=chat_template)
        self.messages: list[dict[str, str]] = []

    def new_session(self) -> PyTorchSessionState:
        """Returns an empty conversation sharing the loaded model and tokenizer."""
        return PyTorchSessionState(
            self.model, self.tokenizer, verbose=self.verbose, chat_template=self._chat_template
        )

    def _wrap_prompt(self, prompt: str) -> str:
        """Applies the tokenizer's native template unless ChatML is requested."""
        if self._chat_template is not None or not self.tokenizer.chat_template:
            return super()._wrap_prompt(prompt)
        return self.tokenizer.apply_chat_template(
            [*self.messages, {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _encode_context(self, prompt: str) -> numpy.ndarray:
        """Encodes the complete conversation without duplicating chat history."""
        native_template = self._chat_template is None and bool(self.tokenizer.chat_template)
        prompt_ids = self.tokenizer.encode(
            self._wrap_prompt(prompt),
            add_special_tokens=not (native_template or self._chat_template),
        )
        context = numpy.asarray(prompt_ids, dtype=numpy.int32)
        if not native_template and self.tokens.size:
            context = numpy.concatenate([self.tokens, context])
        return context

    def get_prompt_length(self, prompt: str) -> int:
        """Returns the exact input length, including the native chat template."""
        return len(self._encode_context(prompt))

    def generate(
        self,
        prompt: str,
        max_length: int = 200,
        *,
        max_new_tokens: int | None = None,
        **search_options: Any,
    ) -> PyTorchSessionState:
        """Generates a continuation with total-context or per-turn token limits."""
        import torch

        if max_new_tokens is not None and max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")
        context = self._encode_context(prompt)
        if max_new_tokens is None:
            max_new_tokens = max_length - len(context)
        if max_new_tokens < 1:
            raise ValueError(
                f"The prompt has {len(context)} tokens and exceeds max_length={max_length}."
            )

        if self.verbose:
            print(
                f"[generate] context: {len(context)} tokens, max_new_tokens={max_new_tokens}",
                file=sys.stderr,
                flush=True,
            )
        device = self.model.get_input_embeddings().weight.device
        input_ids = torch.tensor([context.tolist()], dtype=torch.long, device=device)
        options = {"do_sample": False, **search_options}
        if self.tokenizer.pad_token_id is not None:
            options.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        if self.verbose and "streamer" not in options:
            options["streamer"] = GenerationProgress(max_new_tokens)
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=False,
                **options,
            )
        new_tokens = output[0, len(context) :].tolist()
        eos = search_options.get("eos_token_id", self.model.generation_config.eos_token_id)
        eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
        while new_tokens and new_tokens[-1] in eos_ids:
            new_tokens.pop()
        self.tokens = numpy.concatenate([context, numpy.asarray(new_tokens, dtype=numpy.int32)])
        self.text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        self.generated_token_count = len(new_tokens)
        self.messages.extend(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": self.text}]
        )
        self._turn_count += 1
        return self


def create_pytorch_session(
    model_id: str,
    precision: str | None = None,
    device: str | None = None,
    cache_dir: str | None = None,
    chat_template: str | None = None,
    verbose: int = 0,
) -> PyTorchSessionState:
    """Loads native Transformers weights, including supported quantized checkpoints.

    ``precision`` selects the loading dtype, not a new quantization. Quantized
    checkpoints retain their own quantization configuration and require the
    corresponding Transformers backend and compatible hardware.
    """
    dtypes = {
        None: "auto",
        "auto": "auto",
        "fp32": "float32",
        "fp16": "float16",
        "bf16": "bfloat16",
    }
    if precision not in dtypes:
        raise ValueError(
            "The PyTorch backend supports --precision auto, fp32, fp16, or bf16. "
            "For quantized weights, pass a pre-quantized model id without --precision int4."
        )
    if model_id.startswith("mock/"):
        raise ValueError("mock/ models are ONNX-only; use a Transformers model with PyTorch.")
    if verbose:
        print("[create_pytorch_session] importing Transformers", file=sys.stderr, flush=True)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    from transformers import AutoTokenizer

    if verbose:
        print(
            f"[create_pytorch_session] loading configuration for {model_id!r}",
            file=sys.stderr,
            flush=True,
        )
    config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
    model_class = (
        AutoModelForImageTextToText if hasattr(config, "vision_config") else AutoModelForCausalLM
    )
    if verbose:
        print(
            f"[create_pytorch_session] loading weights with {model_class.__name__} "
            f"on {device or 'cpu'}",
            file=sys.stderr,
            flush=True,
        )
    model = model_class.from_pretrained(
        model_id,
        config=config,
        cache_dir=cache_dir,
        dtype=dtypes[precision],
        device_map=device or "cpu",
    )
    model.eval()
    if verbose:
        print("[create_pytorch_session] loading tokenizer", file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    return PyTorchSessionState(model, tokenizer, verbose=verbose, chat_template=chat_template)
