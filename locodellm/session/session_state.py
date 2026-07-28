"""Holds the state of a generation session."""

from __future__ import annotations

from typing import Any

import numpy as np


class SessionState:
    """Holds the state of a generation session.

    Attributes:
        model: The loaded ``onnxruntime_genai.Model``.
        tokenizer: The tokenizer bound to *model*.
        tokens: All token ids accumulated so far (prompt + generated).
        text: The text generated during the last ``generate`` call.
    """

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokens: np.ndarray = np.array([], dtype=np.int32)
        self.text: str = ""
