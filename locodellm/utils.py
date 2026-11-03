"""Utility functions for locodellm."""

from __future__ import annotations

import re

_FENCED_CODE_RE = re.compile(r"```(?:\w*)\n(.*?)```", re.DOTALL)
_PYTHON_CODE_RE = re.compile(r"\s*<python>\s*(.*?)\s*</python>\s*", re.DOTALL)
_THINKING_MARKER_RE = re.compile(
    r"```.*?```|<python>.*?</python>|<tool_call>.*?</tool_call>|</?think>", re.DOTALL
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

    If *text* contains one or more fenced code blocks
    (`````python\\n...````` or `````\\n...`````), returns their content
    concatenated with blank lines. A complete ``<python>...</python>`` wrapper
    is also accepted. Otherwise returns *text* stripped
    of leading/trailing whitespace.

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
        return "\n\n".join(block.strip() for block in blocks)
    return text.strip()
