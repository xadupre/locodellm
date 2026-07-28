"""Calls a local LLM with a prompt and returns the answer."""

from __future__ import annotations

from typing import Any

import numpy as np

from locodellm.session import SessionState


def generate(
    prompt: str,
    model: str | SessionState | Any,
    providers: list[str] | None = None,
    max_length: int = 200,
    **search_options: Any,
) -> SessionState:
    """Calls a local LLM and returns a :class:`SessionState`.

    The returned session keeps the full token history so that a follow-up
    call can continue the conversation::

        session = generate("Hello", "path/to/model")
        session = generate("Tell me more", session)

    Args:
        prompt: The text prompt to send to the model.
        model: A path (``str``) to the model directory, an already-loaded
            ``onnxruntime_genai.Model`` instance, or a :class:`SessionState`
            returned by a previous ``generate`` call (to continue the
            conversation).
        providers: Ordered list of execution providers, e.g.
            ``["CUDAExecutionProvider", "CPUExecutionProvider"]``.
            When *None*, onnxruntime-genai picks its default provider.
            Ignored when *model* is not a path.
        max_length: Maximum number of tokens to generate (including all
            tokens accumulated across turns).
        **search_options: Extra search options forwarded to
            ``GeneratorParams.set_search_options`` (e.g. ``temperature``,
            ``top_k``, ``top_p``).

    Returns:
        A :class:`SessionState` containing the generated text and full
        token history.
    """
    import onnxruntime_genai as og

    if isinstance(model, SessionState):
        session = model
    elif isinstance(model, str):
        if providers is not None:
            config = og.Config(model)
            config.clear_providers()
            for provider in providers:
                config.append_provider(provider)
            loaded = og.Model(config)
        else:
            loaded = og.Model(model)
        session = SessionState(loaded, og.Tokenizer(loaded))
    else:
        session = SessionState(model, og.Tokenizer(model))

    prompt_ids = session.tokenizer.encode(prompt)

    # Build the full context: previous tokens + new prompt tokens.
    if session.tokens.size > 0:
        context = np.concatenate([session.tokens, prompt_ids])
    else:
        context = np.asarray(prompt_ids, dtype=np.int32)

    params = og.GeneratorParams(session.model)
    params.set_search_options(max_length=max_length, **search_options)

    generator = og.Generator(session.model, params)
    generator.append_tokens(context)

    new_tokens: list[int] = []
    while not generator.is_done():
        generator.generate_next_token()
        token = generator.get_next_tokens()
        new_tokens.extend(token.tolist())

    session.tokens = np.concatenate([context, np.array(new_tokens, dtype=np.int32)])
    session.text = session.tokenizer.decode(new_tokens)
    return session
