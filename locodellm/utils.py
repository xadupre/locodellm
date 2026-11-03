"""Utility functions for locodellm."""

from __future__ import annotations

import re

_FENCED_CODE_RE = re.compile(r"```(?:\w*)\n(.*?)```", re.DOTALL)
_THINKING_MARKER_RE = re.compile(r"```.*?```|</?think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Returns the answer after thinking, without modifying the original response.

    Handles completions whose opening ``<think>`` was already in the prompt.
    Fenced code is opaque so literal thinking tags inside answers are preserved.
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
    """Extracts code from markdown-fenced output after the thinking preamble.

    If *text* contains one or more fenced code blocks
    (`````python\\n...````` or `````\\n...`````), returns their content
    concatenated with blank lines.  Otherwise returns *text* stripped
    of leading/trailing whitespace.

    Args:
        text: Raw model output potentially wrapped in markdown fences.

    Returns:
        The extracted code as a string.
    """
    text = strip_thinking(text)
    blocks = _FENCED_CODE_RE.findall(text)
    if blocks:
        return "\n\n".join(block.strip() for block in blocks)
    return text.strip()
