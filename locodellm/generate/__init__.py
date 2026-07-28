"""Generation utilities for calling local LLMs via onnxruntime-genai.

.. deprecated::
    This module is kept only for backward compatibility.
    Use :func:`locodellm.session.create_session` and
    :meth:`~locodellm.session.SessionState.generate` directly.
"""

from locodellm.session import SessionState, create_session

__all__ = ["SessionState", "create_session"]
