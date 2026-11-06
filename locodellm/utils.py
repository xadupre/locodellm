"""Utility functions for locodellm."""

from __future__ import annotations

import re

_FENCED_CODE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_PYTHON_CODE_RE = re.compile(r"\s*<python>\s*(.*?)\s*</python>\s*", re.DOTALL)
_THINKING_MARKER_RE = re.compile(
    r"```.*?```|<python>.*?</python>|<bash>.*?</bash>|<tool_call>.*?</tool_call>"
    r"|<parameter=code>.*?</parameter>|</?think>",
    re.DOTALL,
)


def strip_thinking(text: str) -> str:
    """Returns the answer after thinking, without modifying the original response.

    Handles completions whose opening ``<think>`` was already in the prompt.
    Fenced and tool-tagged code is opaque so literal thinking tags are preserved.
    An explicitly unclosed thinking section contains no final answer.
    """
    answer_start = 0
    thinking = False
    for marker in _THINKING_MARKER_RE.finditer(text):
        if marker[0] == "<think>":
            thinking = True
        elif marker[0] == "</think>":
            thinking = False
            answer_start = marker.end()
    return "" if thinking else text[answer_start:].strip()


def extract_code(text: str) -> str:
    """Extracts code from Markdown or Python tags after the thinking preamble.

    Prefers blocks labelled ``python`` or ``py``, concatenated with blank lines.
    Uses unlabelled blocks only when no Python-labelled blocks exist, and ignores
    blocks labelled with other languages. A complete ``<python>...</python>``
    wrapper is also accepted. Without fences, returns *text* stripped of
    leading/trailing whitespace.

    Args:
        text: Raw model output potentially wrapped in markdown fences.

    Returns:
        The extracted code as a string.
    """
    text = strip_thinking(text)
    tagged = _PYTHON_CODE_RE.fullmatch(text)
    if tagged is not None:
        return tagged.group(1).strip()
    blocks = _FENCED_CODE_RE.findall(text)
    if blocks:
        python_blocks = [
            code for language, code in blocks if language.strip().lower() in {"python", "py"}
        ]
        selected = python_blocks or [code for language, code in blocks if not language.strip()]
        return "\n\n".join(code.strip() for code in selected)
    return text.strip()
