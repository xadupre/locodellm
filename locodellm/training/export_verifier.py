"""Replays final export examples independently of a solving conversation."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import tempfile

from locodellm.bench.python_harness import PythonHarness
from locodellm.bench.rewards import export_failure_reward
from locodellm.generate.repository import GitRepository
from locodellm.utils import strip_thinking


@dataclass
class ExportReward:
    """Records the factual reward and independent replay evidence."""

    total: float
    criteria: dict[str, bool]
    replays: list[dict]

    def to_dict(self):
        """Returns serializable verification diagnostics."""
        return asdict(self)


def extract_example(answer: str) -> str:
    """Extracts a runnable final script, never tool requests or model claims."""
    blocks = re.findall(r"```(?:python|py)[ \t]*\r?\n(.*?)```", strip_thinking(answer), re.DOTALL)
    code = "\n\n".join(block.strip() for block in blocks)
    if not code.strip():
        return ""
    try:
        ast.parse(code)
    except SyntaxError:
        return ""
    return code


def replay_example(code: str, repository: GitRepository, *, timeout: float) -> dict:
    """Executes instrumentation in a fresh process, with bounded diagnostic output."""
    worker = Path(__file__).with_name("export_worker.py")
    with tempfile.TemporaryDirectory(prefix="locodellm-verify-") as directory:
        source = Path(directory, "example.py")
        result = Path(directory, "verification.json")
        source.write_text(code, encoding="utf-8")
        launch = (
            "import runpy, sys\n"
            f"sys.path.insert(0, {str(repository.path)!r})\n"
            f"sys.argv = [{str(worker)!r}, {str(source)!r}, {str(result)!r}]\n"
            f"runpy.run_path({str(worker)!r}, run_name='__main__')\n"
        )
        execution = PythonHarness(timeout=timeout).execute(
            launch, str(repository.path), script_directory=directory
        )
        if not result.exists() and not execution.timed_out:
            raise RuntimeError(f"The export verifier failed to initialize:\n{execution.stderr}")
        return {
            "observations": json.loads(result.read_text()) if result.exists() else [],
            "timed_out": execution.timed_out,
            "returncode": execution.returncode,
            "stdout": execution.stdout,
            "stderr": execution.stderr,
            "repository": str(repository.path),
            "revision": repository.revision,
            "source_modified": bool(repository.diff()),
        }


def verify_export_answer(
    answer: str,
    source_repository: GitRepository,
    clone_directory: Path,
    *,
    allow_execution: bool,
    timeout: float = 60,
    allowed_converters: set[str] | None = None,
) -> ExportReward:
    """Scores the best observed example using two independent clean replays.

    Runs arbitrary generated Python, not a sandbox. The caller must provide an
    isolated environment. No claimed boolean or expected answer is used as evidence.
    """
    if not allow_execution:
        raise PermissionError("Export verification requires --allow-code-execution.")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("The verification timeout must be finite and positive.")
    code = extract_example(answer)
    empty = {
        "reference_valid": False,
        "export_call_valid": False,
        "failure_valid": False,
        "reproduction_valid": False,
        "answer_extractable": bool(code),
    }
    if not code:
        return ExportReward(export_failure_reward(**empty), empty, [])
    replays = []
    for _ in range(2):
        repository = GitRepository.clone(
            str(source_repository.path), str(clone_directory), source_repository.revision
        )
        replays.append(replay_example(code, repository, timeout=timeout))
    best = ExportReward(export_failure_reward(**empty), empty, replays)
    first, second = replays
    if first["source_modified"] or second["source_modified"]:
        return best
    for index, observed in enumerate(first["observations"]):
        relevant = allowed_converters is None or bool(
            allowed_converters.intersection(observed["converters"])
        )
        reproduced = (
            not first["timed_out"]
            and not second["timed_out"]
            and index < len(second["observations"])
            and observed == second["observations"][index]
            and observed["failure_valid"]
            and bool(observed["converters"])
            and relevant
        )
        criteria = {
            **empty,
            "reference_valid": observed["reference_valid"],
            "export_call_valid": observed["export_call_valid"],
            "failure_valid": observed["failure_valid"] and relevant,
            "reproduction_valid": bool(reproduced),
        }
        total = export_failure_reward(**criteria)
        if total > best.total:
            best = ExportReward(total, criteria, replays)
    return best
