"""Calls a local LLM with a prompt and returns the answer."""

from __future__ import annotations

from typing import Any

from locodellm.session import SessionState, create_session


def generate(
    prompt: str,
    model: str | SessionState | Any,
    providers: list[str] | None = None,
    max_length: int = 200,
    **search_options: Any,
) -> SessionState:
    """Calls a local LLM and returns a :class:`SessionState`.

    This is a convenience wrapper around :func:`~locodellm.session.create_session`
    and :meth:`SessionState.generate`::

        session = create_session("path/to/model")
        session.generate("Hello")

    Args:
        prompt: The text prompt to send to the model.
        model: A path (``str``) to the model directory, an already-loaded
            ``onnxruntime_genai.Model`` instance, or a :class:`SessionState`
            returned by a previous call (to continue the conversation).
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
    if isinstance(model, SessionState):
        session = model
    else:
        session = create_session(model, providers=providers)

    return session.generate(prompt, max_length=max_length, **search_options)
