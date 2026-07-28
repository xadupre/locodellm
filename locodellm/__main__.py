"""Entry point for ``python -m locodellm``.

Supported subcommands
---------------------
version
    Prints the package version.

    Usage::

        python -m locodellm version
"""

from __future__ import annotations

import argparse
import sys


def _cmd_version(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Prints the package version."""
    from locodellm import __version__

    print(__version__)


def main(args: list[str] | None = None) -> None:
    """Parses the command line and dispatches to the appropriate subcommand."""
    parser = argparse.ArgumentParser(prog="python -m locodellm", description="locodellm CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("version", help="Print the package version.")

    parsed = parser.parse_args(args)

    if parsed.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {"version": _cmd_version}
    dispatch[parsed.command](parsed)


if __name__ == "__main__":
    main()
