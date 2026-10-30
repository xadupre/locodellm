"""Computes verification rewards and behavioral diagnostics."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass

PENALTIES = {"error": -10.0, "high": -5.0, "normal": -1.0, "low": -0.2}
UNIT_REWARDS = {
    "passed": 1.0,
    "assertion": -0.3,
    "timeout": -0.3,
    "runtime": -0.6,
    "interpreter": -1.0,
    "other": -1.0,
}
LOGGER = logging.getLogger(__name__)
# The paper specifies severity weights, but does not publish its rule mapping.
RULE_SEVERITY = {
    "invalid-syntax": "error",
    "E999": "error",
    "F821": "error",
    "F822": "error",
    "F823": "error",
    "F631": "high",
    "F632": "high",
    "F701": "high",
    "F702": "high",
    "F704": "high",
    "F706": "high",
    "F707": "high",
    "F722": "high",
    "F401": "low",
    "F841": "low",
}

WORKER = """
import json
import pathlib
import resource
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
result_path = pathlib.Path(sys.argv[2])
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

def report_exception(kind, value, traceback):
    if issubclass(kind, (SyntaxError, NameError)):
        category = "interpreter"
    elif issubclass(kind, AssertionError):
        category = "assertion"
    elif issubclass(kind, Exception):
        category = "runtime"
    else:
        category = "other"
    result_path.write_text(json.dumps({"outcome": category, "error_type": kind.__name__}))

sys.excepthook = report_exception
namespace = {}
exec(compile(payload["setup"], "<setup>", "exec"), namespace)
exec(compile(payload["source"], "<completion>", "exec"), namespace)
for test in payload["tests"]:
    exec(compile(test, "<test>", "exec"), namespace)
result_path.write_text(json.dumps({"outcome": "passed", "error_type": None}))
"""


@dataclass
class Execution:
    """Records the execution category and original exception type."""

    outcome: str
    error_type: str | None = None


@dataclass
class RuffResult:
    """Records raw and length-normalized Ruff penalties."""

    counts: dict[str, int]
    raw: float
    normalized: float
    failed: bool = False


@dataclass
class Reward:
    """Records all reward components and their diagnostics."""

    total: float
    unit: float | None
    ruff: RuffResult | None
    short: float
    execution: Execution | None
    lines: int
    chars: int

    def to_dict(self) -> dict:
        """Returns JSON-serializable diagnostics."""
        return asdict(self)


def extract_code(completion: str) -> str:
    """Extracts fenced Python code, excluding a closed thinking preamble."""
    completion = completion.rsplit("</think>", 1)[-1]
    blocks = re.findall(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", completion, re.DOTALL)
    return "\n\n".join(block.strip() for block in blocks) if blocks else completion.strip()


def build_prompt(task: dict) -> str:
    """Builds the paper's prompt using exactly one example test."""
    if not task.get("test_list"):
        raise ValueError("An MBPP task must contain at least one test.")
    return (
        "Please generate a python function for my problem. "
        "Do not provide any explanations, write only python code.\n\n"
        f"Here is my problem:\n\n{task['text']}\n\n"
        f"Test case example:\n\n{task['test_list'][0]}"
    )


def execute_tests(source: str, tests: list[str], *, allow_execution: bool, timeout=3.0, setup=""):
    """Runs tests in a disposable subprocess, not a security sandbox."""
    if not allow_execution:
        raise PermissionError("Generated code execution requires explicit opt-in.")
    if os.name != "posix":
        raise OSError("The execution harness requires POSIX process groups and resource limits.")
    if not tests:
        raise ValueError("Verification requires at least one test.")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive.")
    with tempfile.TemporaryDirectory(prefix="slm-grpo-") as directory:
        payload_path = Path(directory, "payload.json")
        result_path = Path(directory, "result.json")
        payload_path.write_text(json.dumps({"source": source, "tests": tests, "setup": setup}))
        process = subprocess.Popen(
            [sys.executable, "-I", "-c", WORKER, str(payload_path), str(result_path)],
            cwd=directory,
            env={"PATH": os.defpath},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Also cleans up descendants left behind by a completed candidate.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if timed_out:
            return Execution("timeout", "TimeoutExpired")
        if not result_path.exists():
            return Execution("other", f"ProcessExit({process.returncode})")
        result = json.loads(result_path.read_text())
        if result["outcome"] == "passed" and process.returncode != 0:
            return Execution("other", f"ProcessExit({process.returncode})")
        return Execution(**result)


def severity(issue: dict) -> str:
    """Maps Ruff rules, then diagnostic tokens, then the normal fallback."""
    code = issue.get("code")
    if code in RULE_SEVERITY:
        return RULE_SEVERITY[code]
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]*", issue.get("message", ""))
    for token in tokens:
        if token in RULE_SEVERITY:
            return RULE_SEVERITY[token]
    return "normal"


def ruff_reward(source: str) -> RuffResult:
    """Runs Ruff independently of the checkout's lint configuration."""
    counts = dict.fromkeys(PENALTIES, 0)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--select",
            "E4,E7,E9,F",
            "--stdin-filename",
            "tmp.py",
            "--output-format=json",
            "-",
        ],
        input=source,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    try:
        issues = json.loads(result.stdout) if result.returncode in (0, 1) else None
    except json.JSONDecodeError:
        issues = None
    failed = not isinstance(issues, list) or any(not isinstance(issue, dict) for issue in issues)
    if failed:
        LOGGER.warning("Ruff verification failed: %s", result.stderr.strip() or result.stdout)
        raw = -0.5
    else:
        for issue in issues:
            counts[severity(issue)] += 1
        raw = sum(PENALTIES[level] * count for level, count in counts.items())
    normalized = max(-1.0, min(0.0, raw / max(1, len(source.splitlines()))))
    return RuffResult(counts, raw, normalized, failed)


def score(
    source: str, tests: list[str], mode: str, *, allow_execution=False, timeout=3.0, setup=""
) -> Reward:
    """Computes unit, Ruff, normalized Ruff, or pass-gated combined rewards."""
    if mode not in ("unit", "ruff", "ruff-normalized", "combined"):
        raise ValueError(f"Unknown reward mode: {mode!r}")
    execution = (
        execute_tests(
            source, tests, allow_execution=allow_execution, timeout=timeout, setup=setup
        )
        if mode in ("unit", "combined")
        else None
    )
    unit = UNIT_REWARDS[execution.outcome] if execution is not None else None
    lint = ruff_reward(source) if mode != "unit" else None
    lines = len(source.splitlines())
    short = -0.5 if mode == "combined" and (len(source) < 30 or lines < 3) else 0.0
    if mode == "unit":
        total = unit
    elif mode == "ruff":
        total = lint.raw
    elif mode == "ruff-normalized":
        total = lint.normalized
    else:
        total = unit + (0.3 if unit == 1.0 else 0.1) * lint.normalized + short
    return Reward(total, unit, lint, short, execution, lines, len(source))
