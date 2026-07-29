"""Defines test cases for evaluating generated code from prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExpectedResult:
    """A single expected input/output pair for a generated function.

    Attributes:
        args: Positional arguments to pass to the function.
        expected: The expected return value.
    """

    args: tuple[Any, ...]
    expected: Any

    def to_dict(self) -> dict[str, Any]:
        """Converts the instance to a JSON-serializable dictionary."""
        return {"args": list(self.args), "expected": self.expected}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpectedResult:
        """Creates an instance from a dictionary."""
        return cls(args=tuple(data["args"]), expected=data["expected"])


@dataclass
class PromptTest:
    """A test case pairing a prompt with expected input/output pairs.

    Attributes:
        prompt: The natural-language prompt sent to the model to generate
            a function.
        expected: A list of :class:`ExpectedResult` instances describing
            how the generated function should behave.
    """

    prompt: str
    expected: list[ExpectedResult] = field(default_factory=list)

    def to_json(self) -> str:
        """Serializes the instance to a JSON string."""
        data = {"prompt": self.prompt, "expected": [e.to_dict() for e in self.expected]}
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, text: str) -> PromptTest:
        """Deserializes a :class:`PromptTest` from a JSON string."""
        data = json.loads(text)
        return cls(
            prompt=data["prompt"],
            expected=[ExpectedResult.from_dict(e) for e in data.get("expected", [])],
        )


def dump_prompt_tests(tests: list[PromptTest], path: str) -> None:
    """Writes a list of :class:`PromptTest` to a JSON Lines file.

    Each line in the output file is a self-contained JSON object
    representing one :class:`PromptTest`.

    Args:
        tests: The list of prompt tests to serialize.
        path: File path to write to.
    """
    with open(path, "w", encoding="utf-8") as f:
        for test in tests:
            row = {"prompt": test.prompt, "expected": [e.to_dict() for e in test.expected]}
            f.write(json.dumps(row) + "\n")


def load_prompt_tests(path: str) -> list[PromptTest]:
    """Reads a list of :class:`PromptTest` from a JSON Lines file.

    Args:
        path: File path to read from.

    Returns:
        The deserialized list of prompt tests.
    """
    tests: list[PromptTest] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            tests.append(
                PromptTest(
                    prompt=data["prompt"],
                    expected=[ExpectedResult.from_dict(e) for e in data.get("expected", [])],
                )
            )
    return tests
