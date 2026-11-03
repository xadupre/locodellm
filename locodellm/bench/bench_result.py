"""Result classes for benchmark runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas

from locodellm.bench.prompt_test import ExpectedResult, PromptTest
from locodellm.bench.python_harness import PythonToolResult
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
        duration: Time in seconds spent generating the answer, including
            tool execution when using a harness.
        tool_calls: Python tool executions, excluding final benchmark scoring.
        interactions: Ordered prompts, unmodified model responses, and tool events.
        harness_error: Failure to obtain a final answer through the harness.
    """

    prompt_test: PromptTest
    generated_code: str
    run_status: RunStatus
    results: list[tuple[ExpectedResult, Any, bool]] = field(default_factory=list)
    duration: float = 0.0
    token_count: int = 0
    tool_calls: list[PythonToolResult] = field(default_factory=list)
    interactions: list[dict[str, Any]] = field(default_factory=list)
    harness_error: str | None = None

    @property
    def tool_call_count(self) -> int:
        """Returns the number of Python tool executions for this problem."""
        return len(self.tool_calls)

    @property
    def tokens_per_second(self) -> float:
        """Returns the token generation speed."""
        if self.duration > 0:
            return self.token_count / self.duration
        return 0.0

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

    def to_dataframe(self) -> pandas.DataFrame:
        """Exports the results as a pandas DataFrame.

        Each row represents one expected result assertion. Columns are:

        - ``prompt``: the prompt text
        - ``duration``: time in seconds spent generating the answer
        - ``compiled``: whether the generated code compiled
        - ``ran``: whether the generated code ran without error
        - ``input_index``: index of the input set
        - ``passed``: whether expected matched actual

        Returns:
            A :class:`pandas.DataFrame` with one row per assertion.
        """
        rows: list[dict[str, Any]] = []
        for result in self.results:
            base = {
                "prompt": result.prompt_test.prompt,
                "duration": result.duration,
                "token_count": result.token_count,
                "tokens_per_second": result.tokens_per_second,
                "tool_call_count": result.tool_call_count,
                "harness_error": result.harness_error,
                "compiled": result.run_status.compiled,
                "ran": result.run_status.ran,
            }
            if result.results:
                for i, (_, _, passed) in enumerate(result.results):
                    rows.append({**base, "input_index": i, "passed": passed})
            else:
                rows.append({**base, "input_index": None, "passed": None})
        return pandas.DataFrame(rows)

    def to_json(self) -> list[dict[str, Any]]:
        """Exports the results as a JSON-serializable list.

        Each entry contains the prompt, the generated code, and the
        detailed results for each input set. The interaction trace preserves
        raw model responses, including any emitted reasoning text, followed
        by the final evaluation, which is not sent to the model.

        Returns:
            A list of dictionaries, one per prompt test.
        """
        entries: list[dict[str, Any]] = []
        for result in self.results:
            entry: dict[str, Any] = {
                "prompt": result.prompt_test.prompt,
                "duration": result.duration,
                "token_count": result.token_count,
                "tokens_per_second": result.tokens_per_second,
                "tool_call_count": result.tool_call_count,
                "tool_calls": [asdict(call) for call in result.tool_calls],
                "harness_error": result.harness_error,
                "compiled": result.run_status.compiled,
                "compile_error": (
                    f"{type(result.run_status.compile_error).__name__}: "
                    f"{result.run_status.compile_error}"
                    if result.run_status.compile_error is not None
                    else None
                ),
                "ran": result.run_status.ran,
                "run_error": (
                    f"{type(result.run_status.run_error).__name__}: "
                    f"{result.run_status.run_error}"
                    if result.run_status.run_error is not None
                    else None
                ),
                "generated_code": result.generated_code,
                "results": [],
            }
            for expected_result, actual, passed in result.results:
                entry["results"].append(
                    {
                        "args": list(expected_result.args),
                        "expected": expected_result.expected,
                        "actual": actual,
                        "passed": passed,
                    }
                )
            entry["interactions"] = [
                *result.interactions,
                {
                    "type": "evaluation",
                    "compiled": entry["compiled"],
                    "compile_error": entry["compile_error"],
                    "ran": entry["ran"],
                    "run_error": entry["run_error"],
                    "results": entry["results"],
                },
            ]
            entries.append(entry)
        return entries
