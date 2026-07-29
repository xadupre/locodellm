"""
Run a benchmark on a local LLM
===============================

This example loads a model and runs the ``basic`` benchmark against it.
Each prompt in the benchmark asks the model to write a Python function.
The generated code is compiled, executed with test inputs, and compared
to the expected results.

The model, precision, and execution provider can be changed from the
command line::

    python docs/examples/plot_bench.py \\
        --model Qwen/Qwen2.5-Coder-0.5B-Instruct --precision fp32 --provider cpu

The equivalent CLI command is::

    python -m locodellm bench Qwen/Qwen2.5-Coder-0.5B-Instruct basic \\
        --chat-template chatml --precision fp32 --verbose 1
"""

# %%
# Configuration
# -------------
#
# Default values can be overridden with ``--model``, ``--precision``, and
# ``--provider`` when running the script directly.
#
# Under ``UNITTEST_GOING=1`` (used during CI and documentation builds),
# the mock model is used instead of a real HuggingFace model.

import argparse
import os
import sys

UNITTEST_GOING = os.environ.get("UNITTEST_GOING") == "1"

if "__file__" in dir():
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

_defaults = dict(
    model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    precision="fp32",
    provider="cpu",
    verbose=1,
    chat_template="chatml",
)

if "__file__" in dir():
    _parser = argparse.ArgumentParser(description="Run a benchmark on a local LLM.")
    _parser.add_argument("--model", default=_defaults["model"], help="HuggingFace model id.")
    _parser.add_argument(
        "--precision",
        default=_defaults["precision"],
        help="Conversion precision (fp32, fp16, int4).",
    )
    _parser.add_argument(
        "--provider", default=_defaults["provider"], help="Execution provider (cpu, cuda)."
    )
    _parser.add_argument(
        "--verbose", type=int, default=_defaults["verbose"], help="Verbosity level (0=silent)."
    )
    _parser.add_argument(
        "--chat-template",
        default=_defaults["chat_template"],
        help="Chat template (chatml, or empty for none).",
    )
    _args = _parser.parse_args()
    MODEL_ID = _args.model
    PRECISION = _args.precision
    PROVIDER = _args.provider
    VERBOSE = _args.verbose
    CHAT_TEMPLATE = _args.chat_template or None
else:
    MODEL_ID = _defaults["model"]
    PRECISION = _defaults["precision"]
    PROVIDER = _defaults["provider"]
    VERBOSE = _defaults["verbose"]
    CHAT_TEMPLATE = _defaults["chat_template"]

print(
    f"MODEL_ID={MODEL_ID}, PRECISION={PRECISION}, PROVIDER={PROVIDER}, "
    f"VERBOSE={VERBOSE}, CHAT_TEMPLATE={CHAT_TEMPLATE}"
)

# %%
# Load the model
# ---------------
#
# We use :func:`~locodellm.generate.generate_from_model.get_session` to
# download, convert, and load the model.  Under ``UNITTEST_GOING=1``, the
# mock model ``mock/generate`` is used instead.

from locodellm.generate.generate_from_model import get_session  # noqa: E402

if UNITTEST_GOING:
    MODEL_ID = "mock/generate"

session = get_session(
    model_id=MODEL_ID,
    precision=PRECISION,
    chat_template=CHAT_TEMPLATE,
    verbose=max(VERBOSE - 1, 0),
)

print(f"Model loaded: {MODEL_ID}")

# %%
# Run the benchmark
# ------------------
#
# We load the ``basic`` benchmark and run it against the session.
# The benchmark contains 10 Python function prompts with growing
# difficulty.

from locodellm.bench import load_benchmark  # noqa: E402

benchmark = load_benchmark("basic")
print(f"Benchmark: {benchmark.description}")
print(f"Number of prompts: {len(benchmark.tests)}")

result = benchmark.run(session, verbose=VERBOSE)

# %%
# Results table
# -------------
#
# Each row shows one input set for a prompt: whether the generated code
# compiled, ran, and produced the expected result.

df = result.to_dataframe()
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
print(df[columns].to_markdown(index=False))

# %%
# Per-case statistics
# --------------------
#
# One row per prompt showing how many input sets passed.

stats = df.groupby("prompt", sort=False)
rows = []
for prompt, group in stats:
    total = len(group)
    passed = int(group["passed"].sum())
    rows.append(
        {
            "prompt": prompt[:60] + "..." if len(prompt) > 60 else prompt,
            "duration": round(float(group["duration"].iloc[0]), 3),
            "tokens_per_second": round(float(group["tokens_per_second"].iloc[0]), 1),
            "compiled": bool(group["compiled"].iloc[0]),
            "ran": bool(group["ran"].iloc[0]),
            "passed": f"{passed}/{total}",
            "score": round(passed / total, 2) if total > 0 else 0.0,
        }
    )

import pandas  # noqa: E402

stats_df = pandas.DataFrame(rows)
print(stats_df.to_markdown(index=False))

# %%
# Summary
# -------
#
# Overall benchmark score.

print(f"Total prompts: {result.total}")
print(f"Passed (all inputs correct): {result.passed}/{result.total}")
print(f"Failed: {result.failed}/{result.total}")
print(f"Overall score: {result.passed / result.total:.0%}")
