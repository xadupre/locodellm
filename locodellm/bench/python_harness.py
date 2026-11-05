"""Runs a model-driven conversation with bounded Python and Bash execution tools."""

from __future__ import annotations

import os
import re
import shutil
import signal
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
_SCRIPT_CALL = re.compile(_PYTHON_CALL.pattern + r"|<bash>\s*(.*?)\s*</bash>", re.DOTALL)
_TOOL_MARKER = re.compile(
    r"```.*?```|(?P<python></?python\b)|(?P<bash></?bash\b)|(?P<native></?tool_call\b)", re.DOTALL
)
_INSTRUCTIONS = """{tool_instructions}
You may execute up to {max_tool_calls} scripts to solve this problem.
Blocks execute in order, and their results are returned
together before your next response. Each block consumes one execution.
Print the values you want to inspect. You will receive stdout,
stderr, the exit code, and whether execution timed out.
{working_directory_instructions}
When finished, {final_answer_instructions}
The hidden benchmark tests are
not available to the tool.
{tool_rules}
Bash uses pipefail; shell variables do not persist between scripts.

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
    """Records one Python or Bash execution, including failed executions."""

    code: str
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool
    command: list[str]
    cwd: str
    function_call_count: int = 0
    tool: str = "python"


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
    return [code for tool, code in extract_script_calls(text) if tool == "python"]


def extract_script_calls(text: str) -> list[tuple[str, str]]:
    """Extracts ordered Python and Bash requests outside thinking and Markdown."""
    return [
        ("bash" if index == 3 else "python", code)
        for match in _SCRIPT_CALL.finditer(strip_thinking(text))
        for index, code in enumerate(match.groups())
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
    if getattr(session, "verbose", 0):
        print(f"[generate] response preview: {session.text[:30]!r}", file=sys.stderr, flush=True)
    interactions.append(
        {
            "type": "model_response",
            "content": session.text,
            "generated_token_count": session.generated_token_count,
        }
    )


class PythonHarness:
    """Provides a shared configurable budget of Python and Bash executions.

    Scripts run with the current user's permissions; this is not a sandbox.
    """

    max_tool_calls = 10

    def __init__(
        self,
        timeout: float = 10.0,
        max_tool_calls: int = 10,
        verbose: int = 0,
        bash_only: bool = False,
    ) -> None:
        if timeout <= 0:
            raise ValueError("The Python tool timeout must be positive.")
        if isinstance(max_tool_calls, bool) or not isinstance(max_tool_calls, int):
            raise ValueError("The Python tool budget must be a positive integer.")
        if max_tool_calls <= 0:
            raise ValueError("The Python tool budget must be a positive integer.")
        self.timeout = timeout
        self.max_tool_calls = max_tool_calls
        self.verbose = verbose
        self.bash_only = bash_only

    def execute(
        self,
        code: str,
        directory: str,
        *,
        script_directory: str | None = None,
        tool: str = "python",
    ) -> PythonToolResult:
        """Executes a script and captures bounded output and execution status."""
        if tool not in {"python", "bash"}:
            raise ValueError(f"Unsupported execution tool: {tool!r}")
        if self.bash_only and tool != "bash":
            raise ValueError("Python tool requests are disabled in Bash-only mode.")
        script = Path(script_directory or directory) / (
            "tool_script.sh" if tool == "bash" else "tool_script.py"
        )
        script.write_text(code, encoding="utf-8")
        counter = script.parent / "tool_function_calls.bin"
        counter.write_bytes(b"\0" * 8)
        command = [sys.executable, "-u", "-c", _TOOL_RUNNER, str(counter), str(script)]
        if tool == "bash":
            bash = shutil.which("bash")
            if bash is None:
                raise FileNotFoundError("Bash execution requires the 'bash' executable on PATH.")
            command = [bash, "--noprofile", "--norc", "-o", "pipefail", str(script)]
        timed_out = False
        returncode = None
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            with subprocess.Popen(
                command,
                cwd=directory,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=os.name == "posix",
            ) as process:
                try:
                    returncode = process.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()

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
            tool,
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
                tool_instructions=(
                    "You may use only the Bash tool to solve the task before answering.\n"
                    "To call it, reply with one or more <bash>...</bash> blocks "
                    "containing Bash commands.\nPython tool requests are disabled."
                    if self.bash_only
                    else (
                        "You may use Python and Bash tools to solve the task before answering.\n"
                        "Reply with one or more <python>...</python> blocks containing "
                        "Python scripts, or <bash>...</bash> blocks containing Bash commands.\n"
                        "Python and Bash share the same execution budget and working directory."
                    )
                ),
                tool_rules=(
                    "Every <bash>...</bash> block is an execution request, never a final answer."
                    if self.bash_only
                    else (
                        "Every <python>...</python> or <bash>...</bash> block is an execution "
                        "request, never a final answer.\n"
                        "A function definition alone does not call it or print anything.\n"
                        "Include calls and print or assert statements to check your solution."
                    )
                ),
                max_tool_calls=self.max_tool_calls,
                final_answer_instructions=final_answer_instructions,
                working_directory_instructions=(
                    "Each script runs in a fresh process in the same temporary directory."
                    if directory is None
                    else (
                        "Each script runs in a fresh process in the repository "
                        f"working directory {directory!r}. Relative file paths refer to this "
                        "directory. File modifications persist after execution."
                    )
                ),
            )
            + prompt
        )
        with tempfile.TemporaryDirectory(prefix="locodellm-python-") as script_directory:
            working_directory = directory or script_directory
            for turn in range(1, self.max_tool_calls + 2):
                if self.verbose:
                    print(
                        f"[harness] model turn {turn}; "
                        f"{self.max_tool_calls - len(result.tool_calls)} scripts remaining",
                        file=sys.stderr,
                        flush=True,
                    )
                generate_with_trace(
                    session,
                    message,
                    result.interactions,
                    max_new_tokens=max_length,
                    **search_options,
                )
                if self.verbose:
                    print(
                        f"[harness] model turn {turn} finished: "
                        f"{session.generated_token_count} tokens",
                        file=sys.stderr,
                        flush=True,
                    )
                result.token_count += session.generated_token_count
                response = strip_thinking(session.text)
                calls = extract_script_calls(response)
                budget_label = (
                    "Bash"
                    if self.bash_only
                    else (
                        "Python/Bash"
                        if any(tool == "bash" for tool, _ in calls)
                        or any(call.tool == "bash" for call in result.tool_calls)
                        else "Python"
                    )
                )
                result.interactions[-1]["requested_tool_call_count"] = len(calls)
                if self.bash_only and any(tool != "bash" for tool, _ in calls):
                    result.error = "Python tool requests are disabled in Bash-only mode."
                    return result
                if not calls:
                    marker = next(
                        (
                            match
                            for match in _TOOL_MARKER.finditer(response)
                            if match.lastgroup is not None
                        ),
                        None,
                    )
                    if not response:
                        result.error = "The model returned an empty answer."
                    elif marker is not None and marker.lastgroup == "python":
                        result.error = (
                            "Malformed Python tool call; expected <python>...</python>."
                        )
                    elif marker is not None and marker.lastgroup == "native":
                        result.error = "Unsupported or malformed native Python tool call."
                    elif marker is not None and marker.lastgroup == "bash":
                        result.error = "Malformed Bash tool call; expected <bash>...</bash>."
                    else:
                        result.text = response
                    return result
                remaining = self.max_tool_calls - len(result.tool_calls)
                if len(calls) > remaining:
                    result.error = (
                        f"{budget_label} tool budget exhausted ({self.max_tool_calls} "
                        "executions per problem; "
                        f"requested {len(calls)} with {remaining} remaining)."
                    )
                    return result
                feedback = []
                for index, (tool, code) in enumerate(calls, 1):
                    script_index = len(result.tool_calls) + 1
                    if self.verbose:
                        print(
                            f"[harness] executing script {script_index}/{self.max_tool_calls} "
                            f"(tool={tool}, timeout={self.timeout}s)",
                            file=sys.stderr,
                            flush=True,
                        )
                    execution = self.execute(
                        code, working_directory, script_directory=script_directory, tool=tool
                    )
                    result.tool_calls.append(execution)
                    if self.verbose:
                        print(
                            f"[harness] script {script_index}/{self.max_tool_calls} finished: "
                            f"exit={execution.returncode}, timed_out={execution.timed_out}",
                            file=sys.stderr,
                            flush=True,
                        )
                    result.interactions.extend(
                        [
                            {
                                "type": "command",
                                "tool": tool,
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
                    label = "Bash" if tool == "bash" else "Python"
                    heading = f"{label} tool result"
                    if len(calls) > 1:
                        heading += f" {index}/{len(calls)}"
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
                    f"Remaining {budget_label} tool calls: {remaining}.\n"
                    "The preceding tool requests were executed, not accepted as a final "
                    "answer. An empty stdout means the script printed nothing; defining "
                    "a function does not call it.\n"
                    f"To finish, {final_answer_instructions}\n"
                    "Answer WITHOUT <python> or </python> tags. Otherwise make "
                    "another allowed tool call with your own checks. "
                    "Do not include <bash> or </bash> tags in the final answer. "
                    + (
                        "Only Bash tool requests are allowed."
                        if self.bash_only
                        else "Python and Bash share the remaining execution budget."
                    )
                )
        result.error = "The model did not return a final answer within the conversation budget."
        return result
