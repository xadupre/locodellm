"""Runs the original evaluator in bounded disposable processes."""

import contextlib
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


def load_evaluator(path):
    """Loads trusted evaluator code with the existing standalone cost interface."""
    path = Path(path).resolve(strict=True)
    name = "_while_cost_" + hashlib.sha256(path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Evaluator must be an importable Python source file.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "response_cost", None)) or not callable(
        getattr(module, "reference_from_question", None)
    ):
        raise ValueError("Evaluator must define response_cost and reference_from_question.")
    if "report_cost" not in inspect.signature(module.response_cost).parameters:
        raise ValueError("Evaluator response_cost must accept the report_cost keyword.")
    return module


def grade_worker(evaluator_path, request_path, result_path, progress_path):
    """Calls the original evaluator on the raw response and supplied case metadata."""
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    request = json.loads(Path(request_path).read_text())
    evaluator = load_evaluator(evaluator_path)
    reference = evaluator.reference_from_question(request["task"])

    def report_cost(cost):
        """Publishes completed penalties atomically before executing submitted code."""
        if type(cost) is not int or cost < 0:
            raise ValueError("Reported cost must be a nonnegative integer.")
        temporary = Path(progress_path).with_suffix(".tmp")
        temporary.write_text(json.dumps({"cost": cost}, allow_nan=False))
        temporary.replace(progress_path)

    with (
        open(os.devnull, "w") as sink,
        contextlib.redirect_stdout(sink),
        contextlib.redirect_stderr(sink),
    ):
        cost = evaluator.response_cost(request["completion"], *reference, report_cost=report_cost)
    if type(cost) is not int or cost < 0:
        raise ValueError("Evaluator cost must be a nonnegative integer.")
    temporary = Path(result_path).with_suffix(".tmp")
    temporary.write_text(json.dumps({"cost": cost}, allow_nan=False))
    temporary.replace(result_path)


def grade_completion(completion, task, evaluator, timeout):
    """Adds one on execution timeout while retaining all previously recorded penalties."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Grading timeout must be finite and positive.")
    if not isinstance(completion, str):
        raise TypeError("Completion must be a raw string.")
    evaluator = Path(evaluator).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="while-grade-") as directory:
        root = Path(directory)
        request = root / "request.json"
        result = root / "result.json"
        progress = root / "progress.json"
        request.write_text(json.dumps({"completion": completion, "task": task}, allow_nan=False))
        with (root / "worker.log").open("w+") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    str(Path(__file__).resolve()),
                    str(evaluator),
                    str(request),
                    str(result),
                    str(progress),
                ],
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
                env={"PATH": os.defpath, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            )
            status = "graded"
            error = None
            expired = False
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                expired = True
            finally:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if expired and not result.is_file():
                if progress.is_file():
                    cost = read_cost(progress) + 1
                    return {
                        "status": "graded",
                        "cost": cost,
                        "reward": -float(cost),
                        "success": False,
                        "timed_out": True,
                        "timeout_penalty": 1,
                    }
                status = "timeout"
                error = (
                    f"Evaluator exceeded {timeout} seconds before reporting execution progress."
                )
            if status == "graded" and not expired and process.returncode != 0:
                status = "worker_error"
                log.seek(0)
                error = f"Evaluator exited with {process.returncode}: {log.read(8192)}"
            if status == "graded" and not result.is_file():
                status, error = "worker_error", "Evaluator exited without publishing a cost."
        if error is not None:
            return {
                "status": status,
                "cost": None,
                "reward": None,
                "success": False,
                "error": error,
            }
        cost = read_cost(result)
        return {
            "status": "graded",
            "cost": cost,
            "reward": -float(cost),
            "success": cost == 0,
            "timed_out": False,
            "timeout_penalty": 0,
        }


def read_cost(path):
    """Validates a completed result or an atomic execution-progress record."""
    cost = json.loads(Path(path).read_text())["cost"]
    if type(cost) is not int or cost < 0:
        raise ValueError("Evaluator result must contain a nonnegative integer cost.")
    return cost


if __name__ == "__main__":
    grade_worker(*sys.argv[1:])
