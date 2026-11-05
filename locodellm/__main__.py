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

chat
    Streams an interactive conversation while retaining the model's KV cache.

    Usage::

        python -m locodellm chat mock/generate --chat-template chatml

bench
    Runs a benchmark against a model and outputs results as a markdown table.

    Usage::

        python -m locodellm bench mock/generate basic --chat-template chatml
        python -m locodellm bench mock/generate basic --chat-template chatml -o results.csv

"""

from __future__ import annotations

import argparse
import os
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


def _load_generate_input(filename: str) -> tuple[list[str], str]:
    """Loads independent prompts and resolves their repository directory."""
    import json
    from pathlib import Path

    source = Path(filename).resolve()
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {"prompts", "repo"}:
        raise ValueError("JSON input must contain exactly 'prompts' and 'repo'.")
    prompts = document["prompts"]
    if (
        not isinstance(prompts, list)
        or not prompts
        or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts)
    ):
        raise ValueError("'prompts' must be a nonempty list of nonempty strings.")
    repository = document["repo"]
    if not isinstance(repository, str) or not repository.strip():
        raise ValueError("'repo' must be a nonempty local directory path.")
    path = Path(repository).expanduser()
    if not path.is_absolute():
        path = source.parent / path
    path = path.resolve()
    if not path.is_dir():
        raise ValueError(f"Repository directory does not exist: {path}")
    return prompts, str(path)


def _cmd_generate(args: argparse.Namespace) -> None:
    """Generates answers for a prompt or a JSON list of independent prompts."""
    if args.verbose:
        print(
            f"[generate] starting: backend={args.backend}, model={args.model!r}",
            file=sys.stderr,
            flush=True,
        )
    import json
    from dataclasses import asdict
    from pathlib import Path

    from locodellm.bench.python_harness import HarnessResult, PythonHarness, generate_with_trace
    from locodellm.generate.generate_from_model import get_session
    from locodellm.generate.repository import GitRepository

    if args.verbose:
        print("[generate] reading prompts and validating repository", file=sys.stderr, flush=True)
    prompts, repository = (
        _load_generate_input(args.input) if args.input is not None else ([args.prompt], None)
    )
    target = GitRepository(repository) if repository is not None else None
    if target is not None:
        for filename in (args.input, args.output):
            if filename is not None and Path(filename).resolve().is_relative_to(target.path):
                raise ValueError(
                    "Input and output JSON files must be outside the target repository."
                )
    if args.verbose:
        print("[generate] loading model", file=sys.stderr, flush=True)
    loaded = get_session(
        model_id=args.model,
        precision=args.precision,
        provider=args.provider,
        chat_template=args.chat_template,
        verbose=args.verbose,
        backend=args.backend,
        device=args.device,
    )
    if args.verbose:
        print("[generate] model ready", file=sys.stderr, flush=True)
    max_length = (
        args.max_length if args.max_length is not None else (1024 if args.harness else 200)
    )
    documents = []
    for prompt_index, prompt in enumerate(prompts, 1):
        if args.verbose:
            print(f"[generate] prompt {prompt_index}/{len(prompts)}", file=sys.stderr, flush=True)
        if target is not None:
            if args.verbose:
                print("[generate] resetting repository", file=sys.stderr, flush=True)
            target.clean()
        session = loaded.new_session()
        session.verbose = args.verbose
        if args.harness:
            result = PythonHarness(
                max_tool_calls=args.max_tool_calls,
                verbose=args.verbose,
                bash_only=args.harness == "bash",
            ).run(
                session,
                prompt,
                max_length=max_length,
                final_answer_instructions="return your final answer without a tool call.",
                directory=repository,
            )
        else:
            result = HarnessResult()
            generate_with_trace(session, prompt, result.interactions, max_length=max_length)
            result.text = session.text
            result.token_count = session.generated_token_count
        initial = result.interactions[0]
        document = {
            "initial_prompt": initial["content"],
            "events": result.interactions[1:],
            "initial_generation_options": initial["generation_options"],
            "prompt": prompt,
            "text": result.text,
            "token_count": result.token_count,
            "tool_call_count": len(result.tool_calls),
            "tool_calls": [asdict(call) for call in result.tool_calls],
            "error": result.error,
        }
        if target is not None:
            if args.verbose:
                print("[generate] collecting repository diff", file=sys.stderr, flush=True)
            document["diff"] = target.diff()
        documents.append(document)
        if args.output:
            if args.verbose:
                print(f"[generate] saving {args.output!r}", file=sys.stderr, flush=True)
            exported = (
                {"repo": repository, "base_revision": target.revision, "results": documents}
                if target is not None
                else document
            )
            Path(args.output).write_text(
                json.dumps(exported, indent=2, ensure_ascii=True), encoding="utf-8"
            )
        if result.error:
            raise RuntimeError(result.error)
        print(result.text, flush=True)


def _cmd_chat(args: argparse.Namespace) -> None:
    """Loads a model once and starts a cached, streaming conversation."""
    import onnxruntime_genai

    from locodellm.generate.chat import chat
    from locodellm.generate.generate_from_model import get_session

    session = get_session(
        model_id=args.model,
        precision=args.precision,
        provider=args.provider,
        verbose=args.verbose,
    )
    params = onnxruntime_genai.GeneratorParams(session.model)
    params.set_search_options(max_length=args.max_length)
    generator = onnxruntime_genai.Generator(session.model, params)
    chat(generator, session.tokenizer, args.max_length, chat_template=args.chat_template)


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
        tool_call_count = int(group["tool_call_count"].iloc[0])
        rows.append(
            {
                "prompt": prompt,
                "duration": duration,
                "token_count": token_count,
                "tokens_per_second": tokens_per_second,
                "tool_call_count": tool_call_count,
                "function_call_count": int(group["function_call_count"].iloc[0]),
                "prompt_count": int(group["prompt_count"].iloc[0]),
                "multi_tool_turn_count": int(group["multi_tool_turn_count"].iloc[0]),
                "max_tool_calls_per_turn": int(group["max_tool_calls_per_turn"].iloc[0]),
                "harness_error": group["harness_error"].iloc[0],
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
    from locodellm.bench.python_harness import PythonHarness
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
        backend=args.backend,
        device=args.device,
    )

    output_root, output_suffix = os.path.splitext(args.output) if args.output else (None, "")
    output_suffix = output_suffix.lower()
    json_output = None
    xlsx_output = None
    if output_root:
        json_output = args.output if output_suffix == ".json" else f"{output_root}.json"
        xlsx_output = args.output if output_suffix == ".xlsx" else f"{output_root}.xlsx"

    benchmark = load_benchmark(args.benchmark[0])
    benchmark.max_length = (
        (1024 if args.harness else 200) if args.max_length is None else args.max_length
    )
    result = benchmark.run(
        session,
        verbose=args.verbose,
        json_output=json_output,
        harness=(
            PythonHarness(max_tool_calls=args.max_tool_calls, bash_only=args.harness == "bash")
            if args.harness
            else None
        ),
    )
    df = result.to_dataframe()

    # Add model metadata columns.
    df.insert(0, "model_id", args.model)
    df.insert(1, "precision", args.precision or "")
    df.insert(2, "provider", args.provider or "")
    df.insert(3, "backend", args.backend)
    df.insert(4, "device", args.device or ("cpu" if args.backend == "pytorch" else ""))

    # Compute per-case statistics.
    stats = _compute_case_stats(df)

    # Print as markdown table to stdout.
    columns = [
        "prompt",
        "duration",
        "token_count",
        "tokens_per_second",
        "tool_call_count",
        "function_call_count",
        "prompt_count",
        "multi_tool_turn_count",
        "max_tool_calls_per_turn",
        "harness_error",
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
            {
                "metric": "success_rate",
                "value": total_passed / total_inputs if total_inputs else 0.0,
            },
            {"metric": "avg_generated_tokens", "value": float(stats["token_count"].mean())},
            {"metric": "avg_prompts", "value": float(stats["prompt_count"].mean())},
            {"metric": "total_function_calls", "value": int(stats["function_call_count"].sum())},
            {
                "metric": "multi_tool_turn_count",
                "value": int(stats["multi_tool_turn_count"].sum()),
            },
            {
                "metric": "max_tool_calls_per_turn",
                "value": int(stats["max_tool_calls_per_turn"].max()),
            },
            {
                "metric": "total_tool_calls",
                "value": sum(item.tool_call_count for item in result.results),
            },
        ]
    )

    print("\n\nSummary\n")
    print(summary.to_markdown(index=False))

    # JSON is written incrementally above; always pair it with an Excel summary.
    if xlsx_output:
        with (
            open(xlsx_output, "wb") as stream,
            pandas.ExcelWriter(stream, engine="openpyxl") as writer,
        ):
            stats.to_excel(writer, sheet_name="aggregated", index=False)
            df.to_excel(writer, sheet_name="raw_data", index=False)
        outputs = [xlsx_output]
        if json_output:
            outputs.insert(0, json_output)
        if output_suffix not in {".json", ".xlsx"}:
            df.to_csv(args.output, index=False)
            outputs.append(args.output)
        if args.verbose:
            print(f"\n[bench] results saved to {', '.join(outputs)}")


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
        backend=args.backend,
        device=args.device,
    )
    if results is not None:
        print(make_table(results))


def _cmd_bench(args: argparse.Namespace) -> None:
    """Runs a built-in or LM-Eval benchmark against a model."""
    from locodellm.bench import get_available_benchmarks

    if len(args.benchmark) == 1 and args.benchmark[0] in get_available_benchmarks():
        _cmd_builtin_bench(args)
    else:
        if args.harness:
            raise ValueError("--harness is only supported for a single built-in benchmark.")
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
    gen_parser.add_argument("prompt", nargs="?", help="The prompt to send to the model.")
    gen_parser.add_argument(
        "--input",
        default=None,
        metavar="JSON",
        help="JSON prompts and local repo path; enables the Python/Bash harness.",
    )
    gen_parser.add_argument(
        "--precision",
        default=None,
        help="ONNX conversion precision, or PyTorch loading dtype (auto, fp32, fp16, bf16).",
    )
    gen_parser.add_argument(
        "--backend",
        choices=["onnx", "pytorch"],
        default="onnx",
        help="Generation backend (default: onnx). PyTorch loads Transformers weights directly.",
    )
    gen_parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device (cpu, cuda, cuda:0, auto). Defaults to cpu.",
    )
    gen_parser.add_argument(
        "--provider", default=None, help="Execution provider (e.g. CUDAExecutionProvider)."
    )
    gen_parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum token length (default: 200); with --harness, new tokens per turn (1024).",
    )
    gen_parser.add_argument(
        "--harness",
        choices=["python", "bash"],
        default=None,
        help="Enables Python/Bash script execution, or Bash only with --harness bash.",
    )
    gen_parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=10,
        help="Maximum Python and Bash script executions with --harness (default: 10).",
    )
    gen_parser.add_argument(
        "--output", "-o", default=None, help="Writes the conversation and tool events as JSON."
    )
    gen_parser.add_argument("--chat-template", default=None, help="Chat template (e.g. chatml).")
    gen_parser.add_argument(
        "--verbose", "-v", type=int, default=0, help="Verbosity level (default: 0)."
    )

    chat_parser = sub.add_parser(
        "chat",
        help="Stream an interactive conversation.",
        description="Streams answers with cached history. Use /clear to reset, /quit to exit.",
    )
    chat_parser.add_argument("model", help="Model id or path, as for generate.")
    chat_parser.add_argument("--precision", default=None, help="Precision qualifier.")
    chat_parser.add_argument("--provider", default=None, help="Execution provider.")
    chat_parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum total conversation tokens, including answers (default: 2048).",
    )
    chat_parser.add_argument("--chat-template", choices=["chatml"], default=None)
    chat_parser.add_argument("--verbose", "-v", type=int, default=0)

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
            "\n"
            "  python -m locodellm bench Qwen/Qwen3.5-4B basic \\\n"
            "      --backend pytorch --device cuda --max-length 1024"
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
        "--precision",
        default=None,
        help="ONNX conversion precision, or PyTorch loading dtype (auto, fp32, fp16, bf16).",
    )
    bench_parser.add_argument(
        "--backend",
        choices=["onnx", "pytorch"],
        default="onnx",
        help="Generation backend (default: onnx). PyTorch loads Transformers weights directly.",
    )
    bench_parser.add_argument(
        "--device",
        default=None,
        help=(
            "PyTorch device (cpu, cuda, cuda:0, auto). "
            "Defaults to cpu; auto distributes weights."
        ),
    )
    bench_parser.add_argument(
        "--provider", default=None, help="Execution provider (e.g. CUDAExecutionProvider)."
    )
    bench_parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help=(
            "Maximum token length (default: 200 built-in, 2048 LM-Eval); "
            "with --harness, new tokens per turn (default: 1024)."
        ),
    )
    bench_parser.add_argument(
        "--harness",
        choices=["python", "bash"],
        default=None,
        help="Enables Python/Bash scripts per problem, or Bash only with --harness bash.",
    )
    bench_parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=10,
        help="Maximum Python and Bash scripts per problem with --harness (default: 10).",
    )
    bench_parser.add_argument(
        "--chat-template",
        default=None,
        help="Chat template (e.g. chatml). PyTorch defaults to the tokenizer's native template.",
    )
    bench_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Built-in benchmark output path; writes matching .json and .xlsx files "
            "(.csv is optional)."
        ),
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

    if parsed.command == "chat" and parsed.max_length < 1:
        chat_parser.error("--max-length must be positive")
    if parsed.command == "generate":
        if (parsed.prompt is None) == (parsed.input is None):
            gen_parser.error("provide either PROMPT or --input JSON, but not both")
        if parsed.input is not None:
            parsed.harness = parsed.harness or "python"
    if parsed.command in {"generate", "bench"}:
        command_parser = gen_parser if parsed.command == "generate" else bench_parser
        if parsed.max_length is not None and parsed.max_length < 1:
            command_parser.error("--max-length must be positive")
        if parsed.max_tool_calls < 1:
            command_parser.error("--max-tool-calls must be positive")
        if not parsed.harness and parsed.max_tool_calls != 10:
            command_parser.error("--max-tool-calls requires --harness python or bash")
        if parsed.backend == "pytorch":
            onnx_options = bool(parsed.provider)
            if parsed.command == "bench":
                onnx_options = onnx_options or bool(
                    parsed.provider_option or parsed.session_option
                )
            if onnx_options:
                command_parser.error(
                    "--provider, --provider-option and --session-option are ONNX-only; "
                    "use --device with --backend pytorch"
                )
            if parsed.precision not in {None, "auto", "fp32", "fp16", "bf16"}:
                command_parser.error(
                    "PyTorch --precision must be auto, fp32, fp16, or bf16; "
                    "load pre-quantized weights by model id"
                )
        elif parsed.device is not None:
            command_parser.error("--device requires --backend pytorch")

    dispatch = {
        "version": _cmd_version,
        "benchmarks": _cmd_benchmarks,
        "models": _cmd_models,
        "generate": _cmd_generate,
        "chat": _cmd_chat,
        "bench": _cmd_bench,
    }
    dispatch[parsed.command](parsed)


if __name__ == "__main__":
    main()
