"""Benchmarking utilities for generated code."""

# ruff: noqa: RUF022

from locodellm.bench.bench_prompt_test import BenchPromptTest
from locodellm.bench.bench_result import BenchResult, PromptTestResult
from locodellm.bench.prompt_test import (
    ExpectedResult,
    PromptTest,
    dump_prompt_tests,
    load_prompt_tests,
)
from locodellm.bench.run_code import UNDEFINED, RunStatus, run_function

__all__ = [
    "BenchPromptTest",
    "BenchResult",
    "ExpectedResult",
    "PromptTest",
    "PromptTestResult",
    "RunStatus",
    "UNDEFINED",
    "dump_prompt_tests",
    "get_available_benchmarks",
    "load_prompt_tests",
    "run_function",
]


_BENCHMARKS: dict[str, tuple[str, str]] = {
    "basic": (
        "locodellm.bench.basic_benchmark",
        (
            "10 Python function prompts with growing difficulty, from returning"
            " a constant string to computing an edit distance."
        ),
    )
}


def get_available_benchmarks() -> dict[str, str]:
    """Returns a dictionary of available benchmark names and descriptions.

    Returns:
        A mapping from benchmark name to a short description.
    """
    return {name: desc for name, (_, desc) in _BENCHMARKS.items()}


def load_benchmark(name: str) -> BenchPromptTest:
    """Loads a built-in benchmark by name.

    Args:
        name: The benchmark name as returned by
            :func:`get_available_benchmarks`.

    Returns:
        A :class:`BenchPromptTest` ready to run.

    Raises:
        KeyError: If *name* is not a known benchmark.
    """
    import importlib

    if name not in _BENCHMARKS:
        raise KeyError(f"Unknown benchmark {name!r}. Available: {list(_BENCHMARKS.keys())}")
    module_path, description = _BENCHMARKS[name]
    module = importlib.import_module(module_path)
    return BenchPromptTest(tests=module.BASIC_BENCHMARK, description=description)
