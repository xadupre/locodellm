"""Runs generation directly with PyTorch and Hugging Face Transformers."""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING, Any

import numpy

from locodellm.session.session_state import SessionState

if TYPE_CHECKING:
    from locodellm.generate.live_log import LiveLog


class GenerationProgress:
    """Reports token progress and a bounded response prefix without starting threads."""

    def __init__(
        self,
        max_new_tokens: int,
        tokenizer: Any,
        *,
        verbose: bool = True,
        live_log: LiveLog | None = None,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.tokenizer = tokenizer
        self.preview_tokens: list[int] = []
        self.recent_tokens: list[int] = []
        self.preview_reported = False
        self.token_count = 0
        self.skip_prompt = True
        self.started_at = time.monotonic()
        self.last_report = self.started_at
        self.verbose = verbose
        self.log_streamer = live_log.model_streamer(tokenizer) if live_log is not None else None

    def put(self, tokens: Any) -> None:
        """Reports the prefill phase and periodically reports generated token counts."""
        if self.log_streamer is not None:
            self.log_streamer.put(tokens)
        if not self.verbose:
            return
        if self.skip_prompt:
            self.skip_prompt = False
            print("[generate] waiting for first token", file=sys.stderr, flush=True)
            return
        self.token_count += tokens.numel()
        token_ids = tokens.reshape(-1).tolist()
        self.recent_tokens = [*self.recent_tokens, *token_ids][-128:]
        if not self.preview_reported:
            self.preview_tokens.extend(token_ids)
            preview = self.tokenizer.decode(self.preview_tokens, skip_special_tokens=False)
            if len(preview) >= 30:
                print(f"[generate] response start: {preview[:30]!r}", file=sys.stderr, flush=True)
                self.preview_reported = True
                self.preview_tokens.clear()
        now = time.monotonic()
        if self.token_count == 1 or now - self.last_report >= 1:
            recent_text = self.tokenizer.decode(self.recent_tokens, skip_special_tokens=False)[
                -30:
            ]
            print(
                f"[generate] generated {self.token_count}/{self.max_new_tokens} tokens "
                f"({now - self.started_at:.1f}s) text={recent_text!r}",
                file=sys.stderr,
                flush=True,
            )
            self.last_report = now

    def end(self) -> None:
        """Reports generation completion."""
        if self.log_streamer is not None:
            self.log_streamer.end()
        if not self.verbose:
            return
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
        self.tools: list[dict[str, Any]] = []
        self.chat_template_kwargs: dict[str, Any] = {}

    def configure_tools(self, tools: list[dict[str, Any]]) -> bool:
        """Enables declared tools for native Qwen3-Coder-compatible templates."""
        self.tools = []
        if self._chat_template is not None or not self.tokenizer.chat_template:
            return False
        template = self.tokenizer.get_chat_template(tools=tools)
        if "<tool_call>" not in template or "<function=" not in template:
            return False
        self.tools = tools
        return True

    def add_tool_result(self, name: str, content: str) -> None:
        """Appends a native tool response after the assistant's execution request."""
        if not self.tools:
            raise ValueError("Native tools have not been configured for this session.")
        self.messages.append({"role": "tool", "name": name, "content": content})

    def new_session(self) -> PyTorchSessionState:
        """Returns an empty conversation sharing the loaded model and tokenizer."""
        session = PyTorchSessionState(
            self.model, self.tokenizer, verbose=self.verbose, chat_template=self._chat_template
        )
        session.chat_template_kwargs = dict(self.chat_template_kwargs)
        return session

    def _wrap_prompt(self, prompt: str) -> str:
        """Applies the tokenizer's native template unless ChatML is requested."""
        if self._chat_template is not None or not self.tokenizer.chat_template:
            return super()._wrap_prompt(prompt)
        options = dict(self.chat_template_kwargs)
        if self.tools:
            options["tools"] = self.tools
        return self.tokenizer.apply_chat_template(
            [*self.messages, {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **options,
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
        if "eos_token_id" not in options:
            generation_config = options.get("generation_config", self.model.generation_config)
            configured_eos = generation_config.eos_token_id
            stop_ids = (
                list(configured_eos)
                if isinstance(configured_eos, (list, tuple))
                else ([] if configured_eos is None else [configured_eos])
            )
            tokenizer_eos = self.tokenizer.eos_token_id
            if tokenizer_eos is not None and tokenizer_eos not in stop_ids:
                stop_ids.append(tokenizer_eos)
            if stop_ids:
                options["eos_token_id"] = stop_ids
        if self.tokenizer.pad_token_id is not None:
            options.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        if self.live_log is not None and "streamer" in options:
            raise ValueError("A live log cannot be combined with a custom generation streamer.")
        if (self.verbose or self.live_log is not None) and "streamer" not in options:
            options["streamer"] = GenerationProgress(
                max_new_tokens, self.tokenizer, verbose=bool(self.verbose), live_log=self.live_log
            )
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=False,
                **options,
            )
        new_tokens = output[0, len(context) :].tolist()
        eos = options.get("eos_token_id", self.model.generation_config.eos_token_id)
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
