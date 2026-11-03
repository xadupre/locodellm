"""Runs a model-driven conversation with a bounded Python execution tool."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from locodellm.utils import strip_thinking

_PYTHON_CALL = re.compile(r"```.*?```|<python>\s*(.*?)\s*</python>", re.DOTALL)
_INSTRUCTIONS = """You may use the Python tool to check your solution before answering.
You may execute up to 10 scripts to solve this problem.
To call it, reply ONLY with <python> followed by a complete Python script and
</python>. Print the values you want to inspect. You will receive stdout,
stderr, the exit code, and whether execution timed out.
Each script runs in a fresh Python process in the same temporary directory.
When finished, return your final function in a Python markdown code block,
without a tool call. The hidden benchmark tests are
not available to the tool.
Every <python>...</python> block is an execution request, never a final answer.
A function definition alone does not call the function or print anything.
Include calls and print or assert statements to check your solution.

Problem:
"""
# Shared memory preserves completed function calls even when the script times out.
_TOOL_RUNNER = """
import mmap
import runpy
import struct
import sys

counter_path, script = sys.argv[1:]
sys.argv = [script]
with open(counter_path, "r+b") as counter_file:
    with mmap.mmap(counter_file.fileno(), 8) as counter:
        def record_call(frame, event, argument):
            name = frame.f_code.co_name
            if (
                event == "call"
                and frame.f_code.co_filename == script
                and (not name.startswith("<") or name == "<lambda>")
            ):
                count = struct.unpack_from("Q", counter)[0]
                struct.pack_into("Q", counter, 0, count + 1)

        sys.setprofile(record_call)
        try:
            runpy.run_path(script, run_name="__main__")
        finally:
            sys.setprofile(None)
"""


@dataclass
class PythonToolResult:
    """Records one Python tool execution, including failed executions."""

    code: str
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool
    command: list[str]
    cwd: str
    function_call_count: int = 0


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
            "type": "model_response",
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
        counter = Path(directory) / "tool_function_calls.bin"
        counter.write_bytes(b"\0" * 8)
        command = [sys.executable, "-u", "-c", _TOOL_RUNNER, str(counter), str(script)]
        timed_out = False
        returncode = None
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            try:
                completed = subprocess.run(
                    command,
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

        return PythonToolResult(
            code,
            outputs[0],
            outputs[1],
            returncode,
            timed_out,
            command,
            directory,
            int.from_bytes(counter.read_bytes(), sys.byteorder),
        )

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
                response = strip_thinking(session.text)
                calls = [
                    match.group(1)
                    for match in _PYTHON_CALL.finditer(response)
                    if match.group(1) is not None
                ]
                if len(calls) > 1:
                    result.error = "Multiple Python tool calls; expected one script per turn."
                    return result
                if not calls:
                    if not response:
                        result.error = "The model returned an empty answer."
                    elif response.startswith("<python") or response.endswith("</python>"):
                        result.error = (
                            "Malformed Python tool call; expected <python>...</python>."
                        )
                    else:
                        result.text = response
                    return result
                code = calls[0]
                if len(result.tool_calls) == self.max_tool_calls:
                    result.error = "Python tool budget exhausted (10 executions per problem)."
                    return result
                execution = self.execute(code, directory)
                result.tool_calls.append(execution)
                result.interactions.extend(
                    [
                        {
                            "type": "command",
                            "argv": execution.command,
                            "cwd": execution.cwd,
                            "code": code,
                            "timeout": self.timeout,
                        },
                        {"type": "stdout", "content": execution.stdout},
                        {"type": "stderr", "content": execution.stderr},
                        {
                            "type": "process_exit",
                            "returncode": execution.returncode,
                            "timed_out": execution.timed_out,
                        },
                    ]
                )
                remaining = self.max_tool_calls - len(result.tool_calls)
                message = (
                    f"Python tool result:\n"
                    f"stdout:\n{execution.stdout}\n"
                    f"stderr:\n{execution.stderr}\n"
                    f"exit code: {execution.returncode}\n"
                    f"timed out: {execution.timed_out}\n"
                    f"Function calls executed: {execution.function_call_count}.\n"
                    f"Remaining Python tool calls: {remaining}.\n"
                    "The preceding <python> block was executed, not accepted as a final "
                    "answer. An empty stdout means the script printed nothing; defining "
                    "a function does not call it.\n"
                    "To finish, return your final function in a ```python markdown "
                    "code block, WITHOUT <python> or </python> tags. Otherwise make "
                    "another allowed tool call with your own checks."
                )
        result.error = "The model did not return a final answer within the conversation budget."
        return result
