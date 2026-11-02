"""Times generated function calls in fresh, resource-limited Python workers."""

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
import types
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
calls = json.loads((directory / "input.txt").read_text())
sys.stdout = (directory / "log.txt").open("w")
os.dup2(sys.stdout.fileno(), 1)
candidate = types.ModuleType("__solution__")
sys.modules["__solution__"] = candidate
exec(code, candidate.__dict__)
function = candidate.__dict__[payload["function"]]
(directory / "ready").touch()
started = time.perf_counter()
outputs = [
    json.dumps(function(*call["args"], **call.get("kwargs", {})), allow_nan=False)
    for call in calls
]
sys.stdout.flush()
elapsed = time.perf_counter() - started
(directory / "stdout.txt").write_text("[" + ",".join(outputs) + "]")
result = directory / "result.tmp"
result.write_text(json.dumps({"seconds": elapsed}))
result.replace(directory / "result.json")
"""


def extract_code(text):
    """Extracts fenced Python after a closed reasoning preamble."""
    text = text.rsplit("</think>", 1)[-1]
    blocks = re.findall(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", text, re.DOTALL)
    return "\n\n".join(block.strip() for block in blocks) if blocks else text.strip()


def normalized_output(text, comparison):
    """Normalizes output tokens or JSON without changing JSON string contents."""
    if comparison == "tokens":
        return text.split()
    if comparison == "json":
        return json.dumps(json.loads(text), sort_keys=True, allow_nan=False)
    raise ValueError(f"Unknown output comparison mode: {comparison}")


def execute(source, case, *, function, timeout, cpu=None, comparison="json"):
    """Times calls and return-value encoding, excluding module setup and worker shutdown."""
    expected = normalized_output(case["output"], comparison)
    with tempfile.TemporaryDirectory(prefix="code-optim-") as directory:
        path = Path(directory)
        (path / "payload.json").write_text(
            json.dumps(
                {
                    "source": source,
                    "function": function,
                    "cpu": cpu,
                    "cpu_seconds": math.ceil(timeout) + 1,
                }
            )
        )
        (path / "input.txt").write_text(case["input"])
        result_path = path / "result.json"
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
                        timed_out = True
                        break
                    time.sleep(0.001)
                deadline = time.monotonic() + timeout
                while not timed_out and not result_path.exists() and process.poll() is None:
                    if time.monotonic() >= deadline:
                        timed_out = True
                        break
                    time.sleep(0.001)
            finally:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if timed_out and not result_path.exists():
            return {"status": "timeout", "seconds": timeout}
        if not result_path.exists():
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
        try:
            matches = normalized_output(actual, comparison) == expected
        except (ValueError, RecursionError) as error:
            return {
                "status": "failed",
                "seconds": max(1e-9, result["seconds"]),
                "error": f"Candidate output is not valid finite JSON: {error}",
            }
        status = "passed" if matches else "failed"
        return {
            "status": status,
            "seconds": max(1e-9, result["seconds"]),
            "error": None if status == "passed" else "Returned values do not match.",
        }


def measure(source, case, *, function, timeout, repeats, cpu, comparison="json"):
    """Returns median timings only if every independent repetition passes."""
    records = [
        execute(source, case, function=function, timeout=timeout, cpu=cpu, comparison=comparison)
        for _ in range(repeats)
    ]
    failed = next((record for record in records if record["status"] == "failed"), None)
    timed_out = next((record for record in records if record["status"] == "timeout"), None)
    if failed is not None:
        return failed
    if timed_out is not None:
        return timed_out
    return {"status": "passed", "seconds": statistics.median(row["seconds"] for row in records)}


def reference_measurements(source, task, *, timeout, repeats, cpu):
    """Verifies one proposed reference without treating rejection as an exception."""
    return {
        kind: [
            measure(
                source,
                case,
                function=task["function"],
                timeout=timeout,
                repeats=repeats,
                cpu=cpu,
                comparison=task["output_comparison"],
            )
            for case in task[kind]
        ]
        for kind in ("correctness_tests", "optimization_tests")
    }


def calibrate(task, *, timeout, repeats, cpu):
    """Remeasures every reference on the same worker CPU as generated candidates."""
    matrix = []
    for reference_index, source in enumerate(task["references"]):
        records = reference_measurements(source, task, timeout=timeout, repeats=repeats, cpu=cpu)
        for kind, measurements in records.items():
            for record in measurements:
                if record["status"] != "passed":
                    raise ValueError(f"Reference {reference_index} failed {kind}: {record}")
        matrix.append([record["seconds"] for record in records["optimization_tests"]])
    return matrix


def verify(source, task, references, env, *, repeats, cpu, optimization_stream=False):
    """Executes selected tests and computes correctness-gated reference-ranked rewards."""
    comparison = task["output_comparison"]
    correctness = [
        measure(
            source,
            case,
            function=task["function"],
            timeout=10.0,
            repeats=repeats,
            cpu=cpu,
            comparison=comparison,
        )
        for case in task["correctness_tests"]
    ]
    keep, deadlines = test_plan(task, references, env)
    optimization = [
        measure(
            source,
            task["optimization_tests"][index],
            function=task["function"],
            timeout=deadlines[index],
            repeats=repeats,
            cpu=cpu,
            comparison=comparison,
        )
        for index in keep
    ]
    return summarize(
        task, references, env, correctness, optimization, optimization_stream=optimization_stream
    )
