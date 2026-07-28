"""Extended test case base class for locodellm unit tests."""

from __future__ import annotations

import os
import shutil
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Any, Callable, Optional, Sequence, Tuple, Union


def is_windows() -> bool:
    """Returns ``True`` when running on Windows."""
    import sys

    return sys.platform == "win32"


def is_apple() -> bool:
    """Returns ``True`` when running on macOS."""
    import sys

    return sys.platform == "darwin"


def is_unittest_going() -> bool:
    """Returns ``True`` when the code runs inside the unit tests.

    The flag is enabled by setting the environment variable
    ``UNITTEST_GOING`` to ``"1"``.  It lets documentation examples and other
    scripts pick cheaper, offline-friendly code paths while being tested.
    """
    return os.environ.get("UNITTEST_GOING") == "1"


def has_onnxruntime_genai() -> bool:
    """Returns ``True`` when ``onnxruntime-genai`` is importable."""
    try:
        import onnxruntime_genai  # noqa: F401

        return True
    except ImportError:
        return False


def skipif_no_genai(msg: str = "onnxruntime-genai not installed") -> Callable:
    """Skips the test when ``onnxruntime-genai`` is not available."""
    if not has_onnxruntime_genai():
        return unittest.skip(msg)
    return lambda x: x


def ignore_warnings(warns: Sequence[type[Warning]]) -> Callable:
    """Decorator that silences the listed warning categories inside a test."""

    def wrapper(fct):
        def call_f(self):
            with warnings.catch_warnings():
                for w in warns:
                    warnings.simplefilter("ignore", w)
                return fct(self)

        call_f.__name__ = fct.__name__
        call_f.__doc__ = fct.__doc__
        return call_f

    return wrapper


class ExtTestCase(unittest.TestCase):
    """Base test class for all locodellm tests.

    Provides extra assertion helpers and utility methods.
    """

    _warns: list[tuple[str, int, Warning]] = []

    def shortDescription(self) -> None:
        return None

    def assertExists(self, name: str) -> None:
        """Asserts that a file or directory exists."""
        if not os.path.exists(name):
            raise AssertionError(f"File or folder {name!r} does not exist.")

    def assertNotEmpty(self, value: Any, msg: Optional[Union[Callable, str]] = None) -> None:
        """Asserts that *value* is not None and not empty."""
        if value is None or (isinstance(value, (list, dict, tuple, set, str)) and not value):
            if callable(msg):
                msg = msg()
            raise AssertionError(msg or f"value is empty: {value!r}.")

    def assertStartsWith(self, prefix: str, full: str) -> None:
        """Asserts that *full* starts with *prefix*."""
        if not full.startswith(prefix):
            raise AssertionError(f"prefix={prefix!r} does not start string {full!r}.")

    def capture(self, fct: Callable) -> Tuple[Any, str, str]:
        """Runs *fct* and captures stdout and stderr.

        Returns:
            A tuple ``(result, stdout_text, stderr_text)``.
        """
        sout = StringIO()
        serr = StringIO()
        with redirect_stdout(sout), redirect_stderr(serr):
            try:
                res = fct()
            except Exception as e:
                raise AssertionError(
                    f"function {fct} failed, stdout="
                    f"\n{sout.getvalue()}\n---\nstderr=\n{serr.getvalue()}"
                ) from e
        return res, sout.getvalue(), serr.getvalue()

    @classmethod
    def tearDownClass(cls) -> None:
        for name, line, w in cls._warns:
            warnings.warn(f"\n{name}:{line}: {type(w)}\n  {w!s}", stacklevel=0)

    @classmethod
    def get_dump_folder(cls, name: str, folder: Optional[str] = None, clean: bool = False) -> str:
        """Returns a path to a test dump folder, creating it if needed.

        Args:
            name: Subfolder name inside *folder*.
            folder: Parent folder (defaults to ``"dump_test"``).
            clean: When ``True``, delete the subfolder first.

        Returns:
            Absolute path to the created subfolder.
        """
        if folder is None:
            folder = "dump_test"
        if not os.path.exists(folder):
            os.mkdir(folder)
        res = os.path.join(folder, name)
        if clean and os.path.exists(res):
            shutil.rmtree(res)
        if not os.path.exists(res):
            os.mkdir(res)
        return res
