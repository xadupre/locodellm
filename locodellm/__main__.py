"""Entry point for ``python -m locodellm``.

Supported subcommands
---------------------
version
    Prints the package version.

    Usage::

        python -m locodellm version

benchmarks
    Lists the available built-in and LM-Eval benchmarks.

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

bench
    Runs a benchmark against a model and outputs results as a markdown table.

    Usage::

        python -m locodellm bench mock/generate basic --chat-template chatml
        python -m locodellm bench mock/generate basic --chat-template chatml -o results.csv

"""

from __future__ import annotations

import argparse
import sys


def _parse_provider_option(value: str) -> tuple[str, str]:
    """Parses an ONNX Runtime provider option written as NAME=VALUE."""
    name, separator, option_value = value.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("provider options must use NAME=VALUE")
    return name, option_value


def _parse_session_option(value: str) -> tuple[str, object]:
    """Parses an ONNX Runtime session option written as NAME=JSON_VALUE."""
    import json

    name, separator, option_value = value.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("session options must use NAME=JSON_VALUE")
    return name, json.loads(option_value)


def _get_lm_eval_benchmarks() -> list[str]:
    """Returns the available LM-Eval benchmark names when LM-Eval is installed."""
    import importlib.util

    if importlib.util.find_spec("lm_eval") is None:
        return []

    from lm_eval.tasks import TaskManager

    return sorted(TaskManager().all_tasks)


def _cmd_version(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Prints the package version."""
    from locodellm import __version__

    print(__version__)


def _cmd_benchmarks(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Prints the list of available benchmarks."""
    from locodellm.bench import get_available_benchmarks

    benchmarks = get_available_benchmarks()
    benchmarks.setdefault("gsm8k", "LM Evaluation Harness benchmark.")
    for name in _get_lm_eval_benchmarks():
        benchmarks.setdefault(name, "LM Evaluation Harness benchmark.")
    width = max(len(name) for name in benchmarks)
    for name, description in benchmarks.items():
        print(f"{name:<{width}}  {description}")
    print(
        "\nFull LM-Eval benchmark list: "
        "https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/"
    )


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


def _compute_case_stats(df):  # noqa: ANN001, ANN202
    """Computes per-case statistics from the benchmark results DataFrame."""
    import pandas

    grouped = df.groupby("prompt", sort=False)
    rows = []
    for prompt, group in grouped:
        total_inputs = len(group)
        compiled = bool(group["compiled"].iloc[0])
        ran = bool(group["ran"].iloc[0])
        passed_count = int(group["passed"].sum())
        duration = float(group["duration"].iloc[0])
        token_count = int(group["token_count"].iloc[0])
        tokens_per_second = float(group["tokens_per_second"].iloc[0])
        rows.append(
            {
                "prompt": prompt,
                "duration": duration,
                "token_count": token_count,
                "tokens_per_second": tokens_per_second,
                "compiled": compiled,
                "ran": ran,
                "inputs": total_inputs,
                "passed": passed_count,
                "failed": total_inputs - passed_count,
                "score": passed_count / total_inputs if total_inputs > 0 else 0.0,
            }
        )
    return pandas.DataFrame(rows)


def _cmd_builtin_bench(args: argparse.Namespace) -> None:
    """Runs a benchmark against a model and outputs results."""
    from locodellm.bench import load_benchmark
    from locodellm.generate.generate_from_model import get_session

    inner_verbose = max(args.verbose - 1, 0)

    session = get_session(
        model_id=args.model,
        precision=args.precision,
        provider=args.provider,
        provider_options=dict(args.provider_option),
        session_options=dict(args.session_option),
        chat_template=args.chat_template,
        verbose=inner_verbose,
    )

    # Determine JSON output path.
    json_output = None
    if args.output and args.output.endswith(".json"):
        json_output = args.output

    benchmark = load_benchmark(args.benchmark[0])
    benchmark.max_length = 200 if args.max_length is None else args.max_length
    result = benchmark.run(session, verbose=args.verbose, json_output=json_output)
    df = result.to_dataframe()

    # Add model metadata columns.
    df.insert(0, "model_id", args.model)
    df.insert(1, "precision", args.precision or "")
    df.insert(2, "provider", args.provider or "")

    # Compute per-case statistics.
    stats = _compute_case_stats(df)

    # Print as markdown table to stdout.
    columns = [
        "prompt",
        "duration",
        "token_count",
        "tokens_per_second",
        "compiled",
        "ran",
        "input_index",
        "passed",
    ]
    df_display = df[columns]
    print(df_display.to_markdown(index=False))
    print("\n\nStatistics\n")
    print(stats.to_markdown(index=False))

    # Compute aggregated summary.
    import pandas

    total_cases = len(stats)
    total_inputs = int(stats["inputs"].sum())
    total_passed = int(stats["passed"].sum())
    total_failed = int(stats["failed"].sum())
    cases_compiled = int(stats["compiled"].sum())
    cases_ran = int(stats["ran"].sum())
    avg_duration = float(stats["duration"].mean())
    avg_tokens_per_second = float(stats["tokens_per_second"].mean())
    avg_score = float(stats["score"].mean())
    summary = pandas.DataFrame(
        [
            {"metric": "total_cases", "value": total_cases},
            {"metric": "total_inputs", "value": total_inputs},
            {"metric": "total_passed", "value": total_passed},
            {"metric": "total_failed", "value": total_failed},
            {"metric": "cases_compiled", "value": cases_compiled},
            {"metric": "cases_ran", "value": cases_ran},
            {"metric": "avg_duration", "value": avg_duration},
            {"metric": "avg_tokens_per_second", "value": avg_tokens_per_second},
            {"metric": "avg_score", "value": avg_score},
        ]
    )

    print("\n\nSummary\n")
    print(summary.to_markdown(index=False))

    # Export to CSV/Excel if requested (JSON is written incrementally above).
    if args.output and not args.output.endswith(".json"):
        output = args.output
        if output.endswith(".csv"):
            df.to_csv(output, index=False)
        elif output.endswith(".xlsx"):
            with pandas.ExcelWriter(output) as writer:
                stats.to_excel(writer, sheet_name="aggregated", index=False)
                df.to_excel(writer, sheet_name="raw_data", index=False)
        else:
            df.to_csv(output, index=False)
        if args.verbose:
            print(f"\n[bench] results saved to {output}")


def _cmd_lm_eval(args: argparse.Namespace) -> None:
    """Runs LM Evaluation Harness against a model."""
    from lm_eval.utils import make_table

    from locodellm.lm_eval import run_lm_eval

    results = run_lm_eval(
        model=args.model,
        tasks=args.benchmark,
        precision=args.precision,
        provider=args.provider,
        provider_options=dict(args.provider_option),
        session_options=dict(args.session_option),
        chat_template=args.chat_template,
        max_length=2048 if args.max_length is None else args.max_length,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        verbose=args.verbose,
    )
    if results is not None:
        print(make_table(results))


def _cmd_bench(args: argparse.Namespace) -> None:
    """Runs a built-in or LM-Eval benchmark against a model."""
    from locodellm.bench import get_available_benchmarks

    if len(args.benchmark) == 1 and args.benchmark[0] in get_available_benchmarks():
        _cmd_builtin_bench(args)
    else:
        _cmd_lm_eval(args)


def main(args: list[str] | None = None) -> None:
    """Parses the command line and dispatches to the appropriate subcommand."""
    parser = argparse.ArgumentParser(prog="python -m locodellm", description="locodellm CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("version", help="Print the package version.")
    sub.add_parser("benchmarks", help="List available built-in and LM-Eval benchmarks.")
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

    bench_parser = sub.add_parser(
        "bench",
        help="Run a benchmark against a model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m locodellm bench mock/generate basic --chat-template chatml\n"
            "  python -m locodellm bench Qwen/Qwen2.5-Coder-0.5B-Instruct basic \\\n"
            "      --chat-template chatml --output results.csv\n"
            "  python -m locodellm bench mock/generate basic \\\n"
            "      --chat-template chatml --output results.xlsx\n"
            "  python -m locodellm bench ./model gsm8k --limit 10"
        ),
    )
    bench_parser.add_argument(
        "model",
        help=(
            "Model id or path. Use mock/generate for the mock model, "
            "or a HuggingFace id like Qwen/Qwen2.5-Coder-0.5B-Instruct."
        ),
    )
    bench_parser.add_argument(
        "benchmark",
        nargs="+",
        help="Benchmark name(s) (use 'python -m locodellm benchmarks' to list).",
    )
    bench_parser.add_argument(
        "--precision", default=None, help="Precision qualifier (e.g. fp16, int4)."
    )
    bench_parser.add_argument(
        "--provider", default=None, help="Execution provider (e.g. CUDAExecutionProvider)."
    )
    bench_parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum token length (default: 200 built-in, 2048 LM-Eval).",
    )
    bench_parser.add_argument(
        "--chat-template", default=None, help="Chat template (e.g. chatml)."
    )
    bench_parser.add_argument(
        "--output", "-o", default=None, help="Output file path (.csv, .xlsx, or .json)."
    )
    bench_parser.add_argument(
        "--provider-option",
        action="append",
        default=[],
        type=_parse_provider_option,
        metavar="NAME=VALUE",
        help="ONNX Runtime option for the selected provider; may be repeated.",
    )
    bench_parser.add_argument(
        "--session-option",
        action="append",
        default=[],
        type=_parse_session_option,
        metavar="NAME=JSON_VALUE",
        help=(
            "ONNX Runtime session option parsed as JSON; quote string values, for example "
            "'session.enable_moe_expert_statistics=\"1\"'. May be repeated."
        ),
    )
    bench_parser.add_argument("--num-fewshot", type=int, default=None)
    bench_parser.add_argument("--limit", type=float, default=None)
    bench_parser.add_argument(
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
        "bench": _cmd_bench,
    }
    dispatch[parsed.command](parsed)


if __name__ == "__main__":
    main()
