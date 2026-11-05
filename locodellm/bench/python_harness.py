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

_PYTHON_CALL = re.compile(
    r"```.*?```|<python>\s*(.*?)\s*</python>"
    r"|<tool_call>\s*<function=(?:stateful_python_code_exec|python)>\s*"
    r"<parameter=code>\s*(.*?)\s*</parameter>\s*</function>\s*</tool_call>"
    r"|\A<parameter=code>\s*(.*?)\s*</parameter>\s*</function>\s*</tool_call>\Z",
    re.DOTALL,
)
_INSTRUCTIONS = """You may use the Python tool to check your solution before answering.
You may execute up to {max_tool_calls} scripts to solve this problem.
To call it, reply with one or more <python>...</python> blocks, each containing
a complete Python script. Blocks execute in order, and their results are returned
together before your next response. Each block consumes one execution.
Print the values you want to inspect. You will receive stdout,
stderr, the exit code, and whether execution timed out.
{working_directory_instructions}
When finished, {final_answer_instructions}
The hidden benchmark tests are
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


def extract_python_calls(text: str) -> list[str]:
    """Extracts explicit Python requests outside thinking and Markdown fences."""
    return [
        code
        for match in _PYTHON_CALL.finditer(strip_thinking(text))
        for code in match.groups()
        if code is not None
    ]


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
    """Provides a configurable budget of Python executions per problem.

    Scripts run with the current user's permissions; this is not a sandbox.
    """

    max_tool_calls = 10

    def __init__(self, timeout: float = 10.0, max_tool_calls: int = 10) -> None:
        if timeout <= 0:
            raise ValueError("The Python tool timeout must be positive.")
        if isinstance(max_tool_calls, bool) or not isinstance(max_tool_calls, int):
            raise ValueError("The Python tool budget must be a positive integer.")
        if max_tool_calls <= 0:
            raise ValueError("The Python tool budget must be a positive integer.")
        self.timeout = timeout
        self.max_tool_calls = max_tool_calls

    def execute(
        self, code: str, directory: str, *, script_directory: str | None = None
    ) -> PythonToolResult:
        """Executes a script and captures bounded output and execution status."""
        script = Path(script_directory or directory) / "tool_script.py"
        script.write_text(code, encoding="utf-8")
        counter = script.parent / "tool_function_calls.bin"
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
        self,
        session: Any,
        prompt: str,
        max_length: int,
        final_answer_instructions: str = (
            "return your final function in a Python markdown code block, without a tool call."
        ),
        directory: str | None = None,
        **search_options: Any,
    ) -> HarnessResult:
        """Runs tool calls until the model answers or exhausts its budget.

        ``max_length`` is the maximum number of new tokens per model turn.
        ``directory`` selects a persistent working directory instead of a
        temporary one. Internal scripts and counters remain in temporary storage.
        """
        if directory is not None:
            path = Path(directory).resolve()
            if not path.is_dir():
                raise ValueError(f"Working directory does not exist: {path}")
            directory = str(path)
        result = HarnessResult()
        message = (
            _INSTRUCTIONS.format(
                max_tool_calls=self.max_tool_calls,
                final_answer_instructions=final_answer_instructions,
                working_directory_instructions=(
                    "Each script runs in a fresh Python process in the same temporary directory."
                    if directory is None
                    else (
                        "Each script runs in a fresh Python process in the repository "
                        f"working directory {directory!r}. Relative file paths refer to this "
                        "directory. File modifications persist after execution."
                    )
                ),
            )
            + prompt
        )
        with tempfile.TemporaryDirectory(prefix="locodellm-python-") as script_directory:
            working_directory = directory or script_directory
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
                calls = extract_python_calls(response)
                result.interactions[-1]["requested_tool_call_count"] = len(calls)
                if not calls:
                    if not response:
                        result.error = "The model returned an empty answer."
                    elif response.startswith("<python") or response.endswith("</python>"):
                        result.error = (
                            "Malformed Python tool call; expected <python>...</python>."
                        )
                    elif response.startswith("<tool_call") or response.endswith("</tool_call>"):
                        result.error = "Unsupported or malformed native Python tool call."
                    else:
                        result.text = response
                    return result
                remaining = self.max_tool_calls - len(result.tool_calls)
                if len(calls) > remaining:
                    result.error = (
                        f"Python tool budget exhausted ({self.max_tool_calls} "
                        "executions per problem; "
                        f"requested {len(calls)} with {remaining} remaining)."
                    )
                    return result
                feedback = []
                for index, code in enumerate(calls, 1):
                    execution = self.execute(
                        code, working_directory, script_directory=script_directory
                    )
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
                    heading = (
                        "Python tool result"
                        if len(calls) == 1
                        else f"Python tool result {index}/{len(calls)}"
                    )
                    feedback.append(
                        f"{heading}:\n"
                        f"stdout:\n{execution.stdout}\n"
                        f"stderr:\n{execution.stderr}\n"
                        f"exit code: {execution.returncode}\n"
                        f"timed out: {execution.timed_out}\n"
                        f"Function calls executed: {execution.function_call_count}."
                    )
                remaining = self.max_tool_calls - len(result.tool_calls)
                message = "\n\n".join(feedback) + (
                    "\n"
                    f"Remaining Python tool calls: {remaining}.\n"
                    "The preceding Python tool requests were executed, not accepted as a final "
                    "answer. An empty stdout means the script printed nothing; defining "
                    "a function does not call it.\n"
                    f"To finish, {final_answer_instructions}\n"
                    "Answer WITHOUT <python> or </python> tags. Otherwise make "
                    "another allowed tool call with your own checks."
                )
        result.error = "The model did not return a final answer within the conversation budget."
        return result
