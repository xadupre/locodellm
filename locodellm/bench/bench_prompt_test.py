"""Benchmark runner for evaluating generated code against expected results."""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from locodellm.bench.bench_result import BenchResult, PromptTestResult
from locodellm.bench.prompt_test import ExpectedResult, PromptTest
from locodellm.bench.run_code import run_function


class BenchPromptTest:
    """Benchmarks a model by running prompt tests and comparing results.

    Attributes:
        tests: The list of :class:`PromptTest` to evaluate.
        description: A short description of the benchmark.
        max_length: Maximum token length for generation.
    """

    def __init__(
        self, tests: list[PromptTest], description: str = "", max_length: int = 200
    ) -> None:
        self.tests = tests
        self.description = description
        self.max_length = max_length

    def run(
        self,
        session: Any,
        verbose: int = 0,
        json_output: str | None = None,
        **search_options: Any,
    ) -> BenchResult:
        """Runs all prompt tests against the given session.

        For each :class:`PromptTest`, the session is restarted, the prompt
        is submitted, the generated code is extracted and executed with the
        specified arguments, and the results are compared to expected values.

        Args:
            session: A :class:`~locodellm.session.SessionState` instance.
            verbose: Verbosity level. When >= 1, displays a progress bar.
            json_output: If set, writes results incrementally to this JSON
                file path after each prompt test completes.
            **search_options: Extra options forwarded to
                :meth:`~locodellm.session.SessionState.generate`.

        Returns:
            A :class:`BenchResult` with outcomes for all tests.
        """
        from locodellm.utils import extract_code

        bench_result = BenchResult()
        total = len(self.tests)

        for idx, prompt_test in enumerate(self.tests):
            if verbose >= 1:
                done = idx + 1
                bar_len = 30
                filled = int(bar_len * done / total)
                bar = "█" * filled + "░" * (bar_len - filled)
                sys.stderr.write(f"\r[{bar}] {done}/{total}")
                sys.stderr.flush()

            current = session.new_session()
            t0 = time.perf_counter()
            current.generate(prompt_test.prompt, max_length=self.max_length, **search_options)
            duration = time.perf_counter() - t0

            generated_code = extract_code(current.text)
            run_status = run_function(generated_code)

            results: list[tuple[ExpectedResult, Any, bool]] = []

            if run_status.compiled and run_status._function is not None:
                func = run_status._function
                for expected_result in prompt_test.expected:
                    try:
                        actual = func(*expected_result.args)
                        passed = actual == expected_result.expected
                    except Exception:
                        actual = None
                        passed = False
                    results.append((expected_result, actual, passed))
            else:
                for expected_result in prompt_test.expected:
                    results.append((expected_result, None, False))

            bench_result.results.append(
                PromptTestResult(
                    prompt_test=prompt_test,
                    generated_code=generated_code,
                    run_status=run_status,
                    results=results,
                    duration=duration,
                    token_count=current.generated_token_count,
                )
            )

            if json_output:
                with open(json_output, "w") as f:
                    json.dump(bench_result.to_json(), f, indent=2)

        if verbose >= 1:
            sys.stderr.write("\n")
            sys.stderr.flush()

        return bench_result
