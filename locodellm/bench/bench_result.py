"""Result classes for benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from locodellm.bench.prompt_test import ExpectedResult, PromptTest
from locodellm.bench.run_code import RunStatus


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
