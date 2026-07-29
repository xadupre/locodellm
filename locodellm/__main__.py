"""Entry point for ``python -m locodellm``.

Supported subcommands
---------------------
version
    Prints the package version.

    Usage::

        python -m locodellm version

benchmarks
    Lists the available built-in benchmarks.

    Usage::

        python -m locodellm benchmarks

models
    Lists the available mock ONNX test models.

    Usage::

        python -m locodellm models

generate
    Generates text from a prompt using a model.

    Usage::

        python -m locodellm generate mock/generate \
            'write a python function which returns "hello"' --chat-template chatml
        python -m locodellm generate ./Qwen2.5-Coder-0.5B-onnx \
            'write a python function which returns "hello"' --chat-template chatml
"""

from __future__ import annotations

import argparse
import sys


def _cmd_version(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Prints the package version."""
    from locodellm import __version__

    print(__version__)


def _cmd_benchmarks(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Prints the list of available benchmarks."""
    from locodellm.bench import get_available_benchmarks

    benchmarks = get_available_benchmarks()
    width = max(len(name) for name in benchmarks)
    for name, description in benchmarks.items():
        print(f"{name:<{width}}  {description}")


def _cmd_models(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Prints the available mock ONNX test models."""
    from locodellm.test_models import get_available_models

    models = get_available_models()
    width = max(len(model_id) for model_id in models)
    for model_id, description in models.items():
        print(f"{model_id:<{width}}  {description}")


def _cmd_generate(args: argparse.Namespace) -> None:
    """Generates text from a prompt using a model."""
    from locodellm.generate.generate_from_model import generate_from_model

    session = generate_from_model(
        model_id=args.model,
        prompt=args.prompt,
        precision=args.precision,
        provider=args.provider,
        max_length=args.max_length,
        chat_template=args.chat_template,
        verbose=args.verbose,
    )
    print(session.text)


def main(args: list[str] | None = None) -> None:
    """Parses the command line and dispatches to the appropriate subcommand."""
    parser = argparse.ArgumentParser(prog="python -m locodellm", description="locodellm CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("version", help="Print the package version.")
    sub.add_parser("benchmarks", help="List available benchmarks.")
    sub.add_parser("models", help="List available mock ONNX test models.")

    gen_parser = sub.add_parser(
        "generate",
        help="Generate text from a prompt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m locodellm generate mock/generate \\\n"
            "      'write a python function which returns \"hello\"'"
            " --chat-template chatml\n"
            "  python -m locodellm generate ./Qwen2.5-Coder-0.5B-onnx \\\n"
            "      'write hello' --chat-template chatml\n"
            "\n"
            "note:\n"
            "  HuggingFace models are automatically downloaded and converted\n"
            "  to ONNX format using modelbuilder. You can also convert\n"
            "  manually with:\n"
            "    python -m modelbuilder.builder \\\n"
            "      -m Qwen/Qwen2.5-Coder-0.5B-Instruct"
            " -o ./Qwen2.5-Coder-0.5B-onnx -p fp32 -e cpu"
        ),
    )
    gen_parser.add_argument(
        "model",
        help=(
            "Model id or path. Use mock/generate for the mock model, "
            "or a HuggingFace id like Qwen/Qwen2.5-Coder-0.5B-Instruct."
        ),
    )
    gen_parser.add_argument("prompt", help="The prompt to send to the model.")
    gen_parser.add_argument(
        "--precision", default=None, help="Precision qualifier (e.g. fp16, int4)."
    )
    gen_parser.add_argument(
        "--provider", default=None, help="Execution provider (e.g. CUDAExecutionProvider)."
    )
    gen_parser.add_argument(
        "--max-length", type=int, default=200, help="Maximum token length (default: 200)."
    )
    gen_parser.add_argument("--chat-template", default=None, help="Chat template (e.g. chatml).")
    gen_parser.add_argument(
        "--verbose", "-v", type=int, default=0, help="Verbosity level (default: 0)."
    )

    parsed = parser.parse_args(args)

    if parsed.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "version": _cmd_version,
        "benchmarks": _cmd_benchmarks,
        "models": _cmd_models,
        "generate": _cmd_generate,
    }
    dispatch[parsed.command](parsed)


if __name__ == "__main__":
    main()
