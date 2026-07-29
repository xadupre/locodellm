"""Benchmark runner for evaluating generated code against expected results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from locodellm.bench.prompt_test import ExpectedResult, PromptTest
from locodellm.bench.run_code import RunStatus, run_function


@dataclass
class PromptTestResult:
    """Result of running a single :class:`PromptTest`.

    Attributes:
        prompt_test: The original prompt test that was evaluated.
        generated_code: The code extracted from the model output.
        run_status: The :class:`RunStatus` from compiling/running the code
            with undefined arguments.
        results: A list of tuples ``(expected, actual, passed)`` for each
            :class:`ExpectedResult` entry.
    """

    prompt_test: PromptTest
    generated_code: str
    run_status: RunStatus
    results: list[tuple[ExpectedResult, Any, bool]] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """Returns *True* if every expected result matched."""
        return all(passed for _, _, passed in self.results)


@dataclass
class BenchResult:
    """Aggregated results from a :class:`BenchPromptTest` run.

    Attributes:
        results: Individual results for each prompt test.
    """

    results: list[PromptTestResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Returns the total number of prompt tests evaluated."""
        return len(self.results)

    @property
    def passed(self) -> int:
        """Returns the number of prompt tests where all assertions passed."""
        return sum(1 for r in self.results if r.all_passed)

    @property
    def failed(self) -> int:
        """Returns the number of prompt tests with at least one failure."""
        return self.total - self.passed


class BenchPromptTest:
    """Benchmarks a model by running prompt tests and comparing results.

    Attributes:
        tests: The list of :class:`PromptTest` to evaluate.
        max_length: Maximum token length for generation.
    """

    def __init__(self, tests: list[PromptTest], max_length: int = 200) -> None:
        self.tests = tests
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
