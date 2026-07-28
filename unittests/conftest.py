"""Pytest configuration for the locodellm unit tests.

Enables the ``UNITTEST_GOING`` flag so documentation examples and other
scripts pick their cheaper, offline-friendly code paths while being tested.
"""

import os

os.environ.setdefault("UNITTEST_GOING", "1")
