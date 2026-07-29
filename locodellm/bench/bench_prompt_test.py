"""Benchmark runner for evaluating generated code against expected results."""

from __future__ import annotations

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

    def run(self, session: Any, **search_options: Any) -> BenchResult:
        """Runs all prompt tests against the given session.

        For each :class:`PromptTest`, the session is restarted, the prompt
        is submitted, the generated code is extracted and executed with the
        specified arguments, and the results are compared to expected values.

        Args:
            session: A :class:`~locodellm.session.SessionState` instance.
            **search_options: Extra options forwarded to
                :meth:`~locodellm.session.SessionState.generate`.

        Returns:
            A :class:`BenchResult` with outcomes for all tests.
        """
        from locodellm.utils import extract_code

        bench_result = BenchResult()

        for prompt_test in self.tests:
            current = session.new_session()
            current.generate(prompt_test.prompt, max_length=self.max_length, **search_options)

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
                )
            )

        return bench_result
