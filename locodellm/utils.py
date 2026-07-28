"""Utility functions for locodellm."""

from __future__ import annotations

import re

_FENCED_CODE_RE = re.compile(r"```(?:\w*)\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Extract code from markdown-fenced output.

    If *text* contains one or more fenced code blocks
    (`````python\\n...````` or `````\\n...`````), returns their content
    concatenated with blank lines.  Otherwise returns *text* stripped
    of leading/trailing whitespace.

    Args:
        text: Raw model output potentially wrapped in markdown fences.

    Returns:
        The extracted code as a string.
    """
    blocks = _FENCED_CODE_RE.findall(text)
    if blocks:
        return "\n\n".join(block.strip() for block in blocks)
    return text.strip()
