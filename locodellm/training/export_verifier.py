"""Replays final export examples independently of a solving conversation."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import shlex
import tempfile

from locodellm.bench.python_harness import PythonHarness
from locodellm.bench.rewards import export_failure_reward
from locodellm.generate.repository import GitRepository
from locodellm.utils import strip_thinking


def load_tasks(filename: str, limit: int | None = None) -> list[dict]:
    """Loads pinned JSONL discovery tasks without copying references into prompts."""
    tasks = []
    identities = set()
    identifiers = set()
    with Path(filename).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} must be an object.")
            if any(
                not isinstance(record.get(key), str) or not record[key].strip()
                for key in ("id", "prompt", "repository", "function")
            ):
                raise ValueError(
                    f"JSONL line {line_number} requires id, prompt, repository and function."
                )
            verification = record.get("verification")
            if (
                not isinstance(verification, dict)
                or not isinstance(verification.get("revision"), str)
                or re.fullmatch(r"[0-9a-fA-F]{40}", verification["revision"]) is None
            ):
                raise ValueError(f"JSONL line {line_number} requires a full verification SHA.")
            if record["id"] in identifiers:
                raise ValueError(f"Duplicate task id: {record['id']}")
            identifiers.add(record["id"])
            identities.add((record["repository"], verification["revision"]))
            tasks.append(record)
    if not tasks or len(identities) != 1:
        raise ValueError("Tasks must be nonempty and share one repository and revision.")
    if limit is not None and limit < 1:
        raise ValueError("The task limit must be positive.")
    return tasks if limit is None else tasks[:limit]


def allowed_converters(task: dict) -> set[str]:
    """Returns the target converter and recorded same-family helpers."""
    names = {task["function"]}
    for example in task.get("expected_answer", {}).get("examples", []):
        names.update(
            name
            for name in example.get("actual_converters", [])
            if name.startswith(task["function"] + "_")
        )
    return names


@dataclass
class ExportReward:
    """Records the factual reward and independent replay evidence."""

    total: float
    criteria: dict[str, bool]
    replays: list[dict]
    base_reward: float = 0.0
    progress_reward: float = 0.0

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


def example_progress(code: str) -> tuple[bool, bool, bool]:
    """Identifies module examples, comparison tests and export calls in Python."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, False, False
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            aliases.update(
                (name.asname or name.name, f"{node.module}.{name.name}") for name in node.names
            )
        elif isinstance(node, ast.Import):
            aliases.update((name.asname or name.name, name.name) for name in node.names)

    def name(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f"{name(node.value)}.{node.attr}"
        return ""

    calls = [name(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    example = any(
        isinstance(node, ast.ClassDef)
        and any(name(base) in {"torch.nn.Module", "nn.Module", "Module"} for base in node.bases)
        and any(
            isinstance(member, ast.FunctionDef) and member.name == "forward"
            for member in node.body
        )
        for node in ast.walk(tree)
    ) or any(
        call.startswith("torch.nn.") and call.rsplit(".", 1)[-1][:1].isupper() for call in calls
    )
    export = any(
        call.rsplit(".", 1)[-1] in {"to_onnx", "_to_onnx", "_call_exporter"} for call in calls
    )
    comparison = any(
        call.rsplit(".", 1)[-1]
        in {
            "assert_allclose",
            "assert_array_equal",
            "assert_equal",
            "assert_close",
            "assertEqualArray",
            "assert_conversion_with_ort_on_cpu",
        }
        for call in calls
    ) or any(
        isinstance(node, ast.Assert) and isinstance(node.test, (ast.Call, ast.Compare))
        for node in ast.walk(tree)
    )
    runtime = any(
        call.rsplit(".", 1)[-1]
        in {
            "InferenceSession",
            "ExtendedReferenceEvaluator",
            "ReferenceEvaluator",
            "assert_conversion_with_ort_on_cpu",
        }
        for call in calls
    )
    return example, export and comparison and runtime, export


def added_python_examples(diff: str) -> dict[str, str] | None:
    """Extracts added Python artifacts, rejecting changes to existing source files."""
    if not diff:
        return {}
    files = {}
    for patch in re.split(r"(?=^diff --git )", diff, flags=re.MULTILINE):
        if not patch:
            continue
        match = re.search(
            r"^new file mode 100(?:644|755)\n.*?^--- /dev/null\n^\+\+\+ b/([^\n]+\.py)\n",
            patch,
            re.MULTILINE | re.DOTALL,
        )
        if not patch.startswith("diff --git ") or match is None:
            return None
        body = patch[match.end() :]
        files[match[1]] = "\n".join(
            line[1:] for line in body.splitlines() if line.startswith("+")
        )
    return files or None


def tool_python_examples(tool_calls: list[dict]) -> list[str]:
    """Extracts Python tool bodies and literal Bash Python commands, never shell code."""
    examples = []
    for call in tool_calls:
        code = call.get("code", "")
        if call.get("tool", "python") == "python":
            examples.append(code)
            continue
        # Heredoc bodies are Python only when passed directly to the interpreter.
        examples.extend(
            match[3]
            for match in re.finditer(
                r"(?:^|\n|&&)\s*python(?:3(?:\.\d+)?)?\s+-\s*<<\s*"
                r"(['\"]?)([A-Za-z_]\w*)\1[ \t]*\n(.*?)^\2[ \t]*(?:\n|$)",
                code,
                re.MULTILINE | re.DOTALL,
            )
        )
        try:
            tokens = shlex.split(code)
        except ValueError:
            continue
        for index, token in enumerate(tokens[:-2]):
            if (
                re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(token).name)
                and tokens[index + 1] == "-c"
            ):
                examples.append(tokens[index + 2])
    return [example.strip() for example in examples]


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
    tool_calls: list[dict] | None = None,
    generated_diff: str = "",
    base_eligible: bool = True,
) -> ExportReward:
    """Adds verified progress to the best original score across saved Python evidence.

    Runs arbitrary generated Python, not a sandbox. The caller must provide an
    isolated environment. No claimed boolean or expected answer is used as evidence.
    """
    if not allow_execution:
        raise PermissionError("Export verification requires --allow-code-execution.")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("The verification timeout must be finite and positive.")
    code = extract_example(answer)
    base_eligible = base_eligible and not generated_diff
    empty = {
        "reference_valid": False,
        "export_call_valid": False,
        "failure_valid": False,
        "reproduction_valid": False,
        "answer_extractable": bool(code) and base_eligible,
        "example_created": False,
        "test_created": False,
        "test_passed": False,
        "failure_confirmed": False,
    }
    files = added_python_examples(generated_diff)
    if files is None:
        return ExportReward(0.0, dict.fromkeys(empty, False), [])
    candidates = [code, *tool_python_examples(tool_calls or []), *files.values()]
    replays = []
    best_criteria = empty.copy()
    base_reward = export_failure_reward(**empty) if base_eligible else 0.0
    progress = dict.fromkeys(
        ("example_created", "test_created", "test_passed", "failure_confirmed"), False
    )
    passing_cases, failing_cases = set(), set()
    for candidate in dict.fromkeys(candidates):
        if not candidate:
            continue
        example, test, export = example_progress(candidate)
        progress["example_created"] |= example
        progress["test_created"] |= test
        if not export:
            continue
        pair = []
        for _ in range(2):
            repository = GitRepository.clone(
                str(source_repository.path), str(clone_directory), source_repository.revision
            )
            replay = replay_example(candidate, repository, timeout=timeout)
            pair.append(replay)
            replays.append(replay)
        first, second = pair
        if first["source_modified"] or second["source_modified"]:
            continue
        for index, observed in enumerate(first["observations"]):
            base_relevant = allowed_converters is None or bool(
                allowed_converters.intersection(observed["converters"])
            )
            relevant = bool(observed["converters"]) and base_relevant
            reproduced = (
                not first["timed_out"]
                and not second["timed_out"]
                and index < len(second["observations"])
                and observed == second["observations"][index]
                and relevant
            )
            criteria = {
                **empty,
                "reference_valid": observed["reference_valid"],
                "export_call_valid": observed["export_call_valid"],
                "failure_valid": observed["failure_valid"] and base_relevant,
                "reproduction_valid": reproduced and observed["failure_valid"],
            }
            if base_eligible and candidate == code:
                total = export_failure_reward(**criteria)
                if total > base_reward:
                    base_reward, best_criteria = total, criteria
            progress["example_created"] |= observed["reference_valid"]
            passed = (
                test
                and reproduced
                and observed["comparison_valid"]
                and all(
                    replay["returncode"] == 0
                    or any(item["failure_observed"] for item in replay["observations"])
                    for replay in pair
                )
            )
            progress["test_passed"] |= passed
            # A case is identified by its converters, input values and PyTorch outputs.
            case = (
                tuple(observed["converters"]),
                observed.get("inputs"),
                observed.get("reference_outputs"),
            )
            if passed:
                passing_cases.add(case)
            if reproduced and observed["failure_observed"]:
                failing_cases.add(case)
        # The failure bonus requires a second case distinct from a passing test case.
        progress["failure_confirmed"] = bool(passing_cases) and bool(
            failing_cases - passing_cases
        )
        if all(progress.values()) and (base_reward == 1.0 or not base_eligible):
            break
    progress_reward = 0.25 * sum(progress.values())
    return ExportReward(
        base_reward + progress_reward,
        {**best_criteria, **progress},
        replays,
        base_reward,
        progress_reward,
    )
