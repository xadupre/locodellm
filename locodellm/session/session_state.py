"""Holds the state of a generation session."""

from __future__ import annotations

from typing import Any

import numpy as np


class SessionState:
    """Holds the state of a generation session.

    Use :func:`create_session` to build an instance from a model path or
    an already-loaded ``onnxruntime_genai.Model``.

    Attributes:
        model: The loaded ``onnxruntime_genai.Model``.
        tokenizer: The tokenizer bound to *model*.
        tokens: All token ids accumulated so far (prompt + generated).
        text: The text generated during the last :meth:`generate` call.
        verbose: Verbosity level (0 = silent, 1+ = print progress).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        verbose: int = 0,
        model_path: str | None = None,
        chat_template: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokens: np.ndarray = np.array([], dtype=np.int32)
        self.text: str = ""
        self.verbose = verbose
        self._model_path = model_path
        self._eos_ids: set[int] | None = None
        self._chat_template = chat_template
        self._turn_count = 0

    def _get_eos_token_ids(self) -> set[int]:
        """Return the set of EOS token ids from the model config."""
        if self._eos_ids is not None:
            return self._eos_ids

        import json
        import os

        if self._model_path is None:
            self._eos_ids = set()
            return self._eos_ids

        genai_config = os.path.join(self._model_path, "genai_config.json")
        if not os.path.exists(genai_config):
            self._eos_ids = set()
            return self._eos_ids

        with open(genai_config) as f:
            config = json.load(f)

        eos = config.get("model", {}).get("decoder", {}).get("eos_token_id", None)
        if eos is None:
            self._eos_ids = set()
        elif isinstance(eos, int):
            self._eos_ids = {eos}
        elif isinstance(eos, list):
            self._eos_ids = set(eos)
        else:
            self._eos_ids = set()
        return self._eos_ids

    def _wrap_prompt(self, prompt: str) -> str:
        """Wrap *prompt* with the chat template if one is configured.

        For ChatML (used by Qwen, many instruct models):
        - First turn: ``<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n``
        - Subsequent turns:
          ``<|im_end|>\\n<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n``
        """
        if self._chat_template is None:
            return prompt
        if self._chat_template == "chatml":
            if self._turn_count == 0:
                return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                return (
                    f"<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                )
        raise ValueError(f"Unknown chat template: {self._chat_template!r}")

    def new_session(self) -> SessionState:
        """Starts a new session, resetting the conversation history.

        The model and tokenizer are preserved but all accumulated tokens,
        generated text, and turn count are cleared.

        Returns:
            A new :class:`SessionState` sharing the same model and settings.
        """
        return SessionState(
            model=self.model,
            tokenizer=self.tokenizer,
            verbose=self.verbose,
            model_path=self._model_path,
            chat_template=self._chat_template,
        )

    def generate(self, prompt: str, max_length: int = 200, **search_options: Any) -> SessionState:
        """Generate text from *prompt*, appending to the conversation history.

        The full token history (previous turns + *prompt*) is replayed so
        the model sees the complete context.

        Args:
            prompt: The text prompt to send to the model.
            max_length: Maximum number of tokens (including all tokens
                accumulated across turns).
            **search_options: Extra search options forwarded to
                ``GeneratorParams.set_search_options`` (e.g. ``temperature``,
                ``top_k``, ``top_p``).

        Returns:
            ``self``, with :attr:`tokens` and :attr:`text` updated.
        """
        import onnxruntime_genai as og

        if self.verbose:
            print(f"[generate] encoding prompt ({len(prompt)} chars)")

        wrapped = self._wrap_prompt(prompt)
        prompt_ids = self.tokenizer.encode(wrapped)

        if self.tokens.size > 0:
            context = np.concatenate([self.tokens, prompt_ids])
        else:
            context = np.asarray(prompt_ids, dtype=np.int32)

        if self.verbose:
            print(
                f"[generate] context: {len(context)} tokens "
                f"(history={self.tokens.size}, prompt={len(prompt_ids)})"
            )

        params = og.GeneratorParams(self.model)
        params.set_search_options(max_length=max_length, **search_options)

        if self.verbose:
            print(f"[generate] starting generation (max_length={max_length})")

        generator = og.Generator(self.model, params)
        generator.append_tokens(context)

        new_tokens: list[int] = []
        while not generator.is_done():
            generator.generate_next_token()
            token = generator.get_next_tokens()
            new_tokens.extend(token.tolist())

        # Strip trailing EOS tokens so the next turn doesn't start with them.
        # When the model finishes via EOS (not max_length), these tokens would
        # cause immediate EOS generation on the next turn.
        eos_ids = self._get_eos_token_ids()
        while new_tokens and new_tokens[-1] in eos_ids:
            new_tokens.pop()

        self.tokens = np.concatenate([context, np.array(new_tokens, dtype=np.int32)])
        self.text = self.tokenizer.decode(new_tokens)
        self._turn_count += 1

        if self.verbose:
            print(
                f"[generate] done: {len(new_tokens)} new tokens, "
                f"{self.tokens.size} total tokens"
            )

        return self


def create_session(
    model: str | Any,
    providers: list[str] | None = None,
    verbose: int = 0,
    chat_template: str | None = None,
) -> SessionState:
    """Create a :class:`SessionState` from a model path or loaded model.

    Args:
        model: A path (``str``) to the model directory or an
            already-loaded ``onnxruntime_genai.Model`` instance.
        providers: Ordered list of execution providers, e.g.
            ``["CUDAExecutionProvider", "CPUExecutionProvider"]``.
            When *None*, onnxruntime-genai picks its default provider.
            Ignored when *model* is not a path.
        verbose: Verbosity level (0 = silent, 1+ = print progress
            messages during model loading and generation).
        chat_template: Chat template to apply around prompts.
            Use ``"chatml"`` for Qwen and other ChatML-based instruct models.
            When *None*, prompts are sent as-is.

    Returns:
        A new :class:`SessionState` ready for :meth:`SessionState.generate`.
    """
    import onnxruntime_genai as og

    if isinstance(model, str):
        if verbose:
            print(f"[create_session] loading model from {model!r}")
        if providers is not None:
            config = og.Config(model)
            config.clear_providers()
            for provider in providers:
                if verbose:
                    print(f"[create_session] adding provider {provider}")
                config.append_provider(provider)
            loaded = og.Model(config)
        else:
            loaded = og.Model(model)
        if verbose:
            print("[create_session] model loaded, creating tokenizer")
        return SessionState(
            loaded,
            og.Tokenizer(loaded),
            verbose=verbose,
            model_path=model,
            chat_template=chat_template,
        )

    if verbose:
        print("[create_session] using pre-loaded model, creating tokenizer")
    return SessionState(model, og.Tokenizer(model), verbose=verbose, chat_template=chat_template)
