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

    def __init__(self, model: Any, tokenizer: Any, verbose: int = 0) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokens: np.ndarray = np.array([], dtype=np.int32)
        self.text: str = ""
        self.verbose = verbose

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
        prompt_ids = self.tokenizer.encode(prompt)

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

        self.tokens = np.concatenate([context, np.array(new_tokens, dtype=np.int32)])
        self.text = self.tokenizer.decode(new_tokens)

        if self.verbose:
            print(
                f"[generate] done: {len(new_tokens)} new tokens, "
                f"{self.tokens.size} total tokens"
            )

        return self


def create_session(
    model: str | Any, providers: list[str] | None = None, verbose: int = 0
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
        return SessionState(loaded, og.Tokenizer(loaded), verbose=verbose)

    if verbose:
        print("[create_session] using pre-loaded model, creating tokenizer")
    return SessionState(model, og.Tokenizer(model), verbose=verbose)
