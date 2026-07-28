import unittest
from locodellm.__main__ import main
from locodellm import __version__


class TestMain(unittest.TestCase):
    def test_version(self):
        """Checks that the version subcommand prints the version."""
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["version"])
        self.assertEqual(buf.getvalue().strip(), __version__)


if __name__ == "__main__":
    unittest.main()
