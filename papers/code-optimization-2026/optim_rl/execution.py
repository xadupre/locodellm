"""Measures real stdin/stdout programs on an explicitly opted-in local backend."""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .environment import summarize, test_plan

WORKER = r"""
import json
import os
import resource
import sys
import time
from pathlib import Path

directory = Path(sys.argv[1])
payload = json.loads((directory / "payload.json").read_text())
if payload["cpu"] is not None:
    os.sched_setaffinity(0, {payload["cpu"]})
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
resource.setrlimit(resource.RLIMIT_AS, (1073741824, 1073741824))
resource.setrlimit(resource.RLIMIT_CPU, (payload["cpu_seconds"], payload["cpu_seconds"]))
code = compile(payload["source"], "<candidate>", "exec")
sys.stdin = (directory / "input.txt").open()
sys.stdout = (directory / "stdout.txt").open("w")
os.dup2(sys.stdin.fileno(), 0)
os.dup2(sys.stdout.fileno(), 1)
(directory / "ready").touch()
started = time.perf_counter()
exit_code = 0
try:
    exec(code, {"__name__": "__main__"})
except SystemExit as error:
    exit_code = error.code
sys.stdout.flush()
elapsed = time.perf_counter() - started
(directory / "result.json").write_text(json.dumps({"seconds": elapsed, "exit_code": exit_code}))
"""


def extract_code(text):
    """Extracts fenced Python after a closed reasoning preamble."""
    text = text.rsplit("</think>", 1)[-1]
    blocks = re.findall(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", text, re.DOTALL)
    return "\n\n".join(block.strip() for block in blocks) if blocks else text.strip()


def execute(source, case, *, timeout, cpu=None):
    """Times execution excluding interpreter startup, with tokenwise output comparison."""
    with tempfile.TemporaryDirectory(prefix="code-optim-") as directory:
        path = Path(directory)
        (path / "payload.json").write_text(
            json.dumps({"source": source, "cpu": cpu, "cpu_seconds": math.ceil(timeout) + 1})
        )
        (path / "input.txt").write_text(case["input"])
        with (path / "stderr.txt").open("w") as stderr:
            process = subprocess.Popen(
                [sys.executable, "-I", "-c", WORKER, directory],
                cwd=directory,
                env={"PATH": os.defpath, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr,
                start_new_session=True,
            )
            timed_out = False
            try:
                startup_deadline = time.monotonic() + 10
                while not (path / "ready").exists() and process.poll() is None:
                    if time.monotonic() >= startup_deadline:
                        raise RuntimeError("Verification worker did not start within 10 seconds.")
                    time.sleep(0.001)
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if timed_out:
            return {"status": "timeout", "seconds": timeout}
        result_path = path / "result.json"
        if process.returncode != 0 or not result_path.exists():
            return {
                "status": "failed",
                "seconds": timeout,
                "error": (path / "stderr.txt").read_text(errors="replace")[-2000:]
                or f"Candidate exited without a result: {process.returncode}",
            }
        result = json.loads(result_path.read_text())
        if result["seconds"] >= timeout:
            return {"status": "timeout", "seconds": timeout}
        try:
            actual = (path / "stdout.txt").read_text()
        except UnicodeDecodeError:
            return {
                "status": "failed",
                "seconds": max(1e-9, result["seconds"]),
                "error": "Candidate output is not valid UTF-8.",
            }
        status = (
            "passed"
            if (result["exit_code"] in (0, None) and actual.split() == case["output"].split())
            else "failed"
        )
        return {
            "status": status,
            "seconds": max(1e-9, result["seconds"]),
            "error": None if status == "passed" else "Output mismatch or nonzero exit.",
        }


def measure(source, case, *, timeout, repeats, cpu):
    """Returns median timings only if every independent repetition passes."""
    records = [execute(source, case, timeout=timeout, cpu=cpu) for _ in range(repeats)]
    failed = next((record for record in records if record["status"] == "failed"), None)
    timed_out = next((record for record in records if record["status"] == "timeout"), None)
    if failed is not None:
        return failed
    if timed_out is not None:
        return timed_out
    return {"status": "passed", "seconds": statistics.median(row["seconds"] for row in records)}


def calibrate(task, *, timeout, repeats, cpu):
    """Remeasures every reference on the same worker CPU as generated candidates."""
    matrix = []
    for reference_index, source in enumerate(task["references"]):
        for case in task["correctness_tests"]:
            record = measure(source, case, timeout=timeout, repeats=repeats, cpu=cpu)
            if record["status"] != "passed":
                raise ValueError(f"Reference {reference_index} failed correctness: {record}")
        durations = []
        for case in task["optimization_tests"]:
            record = measure(source, case, timeout=timeout, repeats=repeats, cpu=cpu)
            if record["status"] != "passed":
                raise ValueError(f"Reference {reference_index} failed optimization: {record}")
            durations.append(record["seconds"])
        matrix.append(durations)
    return matrix


def verify(source, task, references, env, *, repeats, cpu, optimization_stream=False):
    """Executes selected tests and computes correctness-gated reference-ranked rewards."""
    correctness = [
        measure(source, case, timeout=10.0, repeats=repeats, cpu=cpu)
        for case in task["correctness_tests"]
    ]
    keep, deadlines = test_plan(task, references, env)
    optimization = [
        measure(
            source,
            task["optimization_tests"][index],
            timeout=deadlines[index],
            repeats=repeats,
            cpu=cpu,
        )
        for index in keep
    ]
    return summarize(
        task, references, env, correctness, optimization, optimization_stream=optimization_stream
    )
