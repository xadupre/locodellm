import io
import contextlib
import unittest

from locodellm.__main__ import main
from locodellm import __version__
from locodellm.ext_test_case import ExtTestCase


class TestMain(ExtTestCase):
    def test_version(self):
        """Checks that the version subcommand prints the version."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["version"])
        self.assertEqual(buf.getvalue().strip(), __version__)


if __name__ == "__main__":
    unittest.main()
