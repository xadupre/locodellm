"""Runs a model-driven conversation with a bounded Python execution tool."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_PYTHON_CALL = re.compile(r"\s*<python>\s*(.*?)\s*</python>\s*", re.DOTALL)
_INSTRUCTIONS = """You may use a Python tool up to 10 times to solve this problem.
To call it, reply ONLY with <python> followed by a complete Python script and
</python>. Print the values you want to inspect. You will receive stdout,
stderr, the exit code, and whether execution timed out.
Each script runs in a fresh Python process in the same temporary directory.
When finished, return your final function in a Python markdown code block,
without a tool call. Tool calls are optional. The hidden benchmark tests are
not available to the tool.

Problem:
"""


@dataclass
class PythonToolResult:
    """Records one Python tool execution, including failed executions."""

    code: str
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool


@dataclass
class HarnessResult:
    """Records the final answer and tool usage for one problem."""

    text: str = ""
    token_count: int = 0
    tool_calls: list[PythonToolResult] = field(default_factory=list)
    interactions: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def generate_with_trace(
    session: Any, prompt: str, interactions: list[dict[str, Any]], **search_options: Any
) -> None:
    """Records the submitted prompt and the complete generated response."""
    interactions.append(
        {"type": "prompt", "content": prompt, "generation_options": search_options}
    )
    session.generate(prompt, **search_options)
    interactions.append(
        {
            "type": "response",
            "content": session.text,
            "generated_token_count": session.generated_token_count,
        }
    )


class PythonHarness:
    """Provides at most ten Python executions per problem.

    Scripts run with the current user's permissions; this is not a sandbox.
    """

    max_tool_calls = 10

    def __init__(self, timeout: float = 10.0) -> None:
        if timeout <= 0:
            raise ValueError("The Python tool timeout must be positive.")
        self.timeout = timeout

    def execute(self, code: str, directory: str) -> PythonToolResult:
        """Executes a script and captures bounded output and execution status."""
        script = Path(directory) / "tool_script.py"
        script.write_text(code, encoding="utf-8")
        timed_out = False
        returncode = None
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            try:
                completed = subprocess.run(
                    [sys.executable, "-u", str(script)],
                    cwd=directory,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=self.timeout,
                    check=False,
                )
                returncode = completed.returncode
            except subprocess.TimeoutExpired:
                timed_out = True

            outputs = []
            for stream in (stdout, stderr):
                stream.seek(0)
                data = stream.read(16_001)
                text = data[:16_000].decode("utf-8", errors="replace")
                if len(data) > 16_000:
                    text += "\n[output truncated]"
                outputs.append(text)

        return PythonToolResult(code, outputs[0], outputs[1], returncode, timed_out)

    def run(
        self, session: Any, prompt: str, max_length: int, **search_options: Any
    ) -> HarnessResult:
        """Runs tool calls until the model answers or exhausts its budget.

        ``max_length`` is the maximum number of new tokens per model turn.
        """
        result = HarnessResult()
        message = _INSTRUCTIONS + prompt
        with tempfile.TemporaryDirectory(prefix="locodellm-python-") as directory:
            for _ in range(self.max_tool_calls + 1):
                generate_with_trace(
                    session,
                    message,
                    result.interactions,
                    max_new_tokens=max_length,
                    **search_options,
                )
                result.token_count += session.generated_token_count
                response = session.text.strip()
                call = _PYTHON_CALL.fullmatch(response)
                if call is None:
                    if not response:
                        result.error = "The model returned an empty answer."
                    elif response.startswith("<python") or response.endswith("</python>"):
                        result.error = (
                            "Malformed Python tool call; expected <python>...</python>."
                        )
                    else:
                        result.text = response
                    return result
                if len(result.tool_calls) == self.max_tool_calls:
                    result.error = "Python tool budget exhausted (10 executions per problem)."
                    return result

                code = call.group(1)
                result.interactions.append(
                    {"type": "tool_call", "name": "python", "code": code, "timeout": self.timeout}
                )
                execution = self.execute(code, directory)
                result.tool_calls.append(execution)
                result.interactions.append(
                    {"type": "tool_result", "name": "python", **asdict(execution)}
                )
                remaining = self.max_tool_calls - len(result.tool_calls)
                message = (
                    f"Python tool result:\n"
                    f"stdout:\n{execution.stdout}\n"
                    f"stderr:\n{execution.stderr}\n"
                    f"exit code: {execution.returncode}\n"
                    f"timed out: {execution.timed_out}\n"
                    f"Remaining Python tool calls: {remaining}.\n"
                    "Return your final function or make another allowed tool call."
                )
        return result
