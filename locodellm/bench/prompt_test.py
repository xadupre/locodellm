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
