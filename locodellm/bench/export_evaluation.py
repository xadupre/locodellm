"""Evaluates saved ATen generations and prints an evidence-based final report."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile

from locodellm.generate.repository import GitRepository
from locodellm.training.export_verifier import (
    added_python_examples,
    allowed_converters,
    load_tasks,
    verify_export_answer,
)


def diagnosis(result: dict) -> str:
    """Returns an observed failure reason, never a claim from the model."""
    if result["error"]:
        return f"Generation failed: {result['error']}"
    if result["source_modified"]:
        return "Repository modified during generation"
    if any(replay["timed_out"] for replay in result["replays"]):
        return "Verification timed out"
    if result["criteria"]["reproduction_valid"]:
        return "Converter failure confirmed by two independent replays"
    if result["criteria"]["failure_valid"]:
        return "Failure observed, not independently reproduced"
    if result["criteria"]["failure_confirmed"]:
        return "Converter failure confirmed outside the original export protocol"
    if result["criteria"]["test_passed"]:
        return "Passing export comparison test confirmed"
    for replay in result["replays"]:
        if replay["returncode"] and replay["stderr"].strip():
            return replay["stderr"].strip().splitlines()[-1]
    if result["criteria"]["test_created"]:
        return "Comparison test created; no passing baseline confirmed"
    if result["criteria"]["example_created"]:
        return "PyTorch example created; no comparison test confirmed"
    if not result["criteria"]["answer_extractable"]:
        return "No extractable runnable Python"
    return "No converter failure confirmed"


def evaluate(
    dataset: Path,
    results: Path,
    *,
    allow_execution: bool,
    timeout: float = 60,
    output: Path | None = None,
) -> dict:
    """Scores saved generations with the training reward, without loading a model."""
    if not allow_execution:
        raise PermissionError(
            "Evaluation requires --allow-code-execution in an isolated environment."
        )
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("--verification-timeout must be finite and positive.")
    if output is not None and output.resolve() in {dataset.resolve(), results.resolve()}:
        raise ValueError("The evaluation output must differ from the dataset and results.")
    tasks = load_tasks(str(dataset))
    by_prompt = {task["prompt"]: task for task in tasks}
    if len(by_prompt) != len(tasks):
        raise ValueError("Dataset prompts must be unique to identify saved generations.")
    raw = results.read_bytes()
    document = json.loads(raw)
    if (
        not isinstance(document, dict)
        or not isinstance(document.get("repo"), str)
        or not document["repo"].strip()
        or document.get("base_revision") != tasks[0]["verification"]["revision"]
        or not isinstance(document.get("results"), list)
        or not document["results"]
    ):
        raise ValueError("Expected nonempty generate results at the dataset's recorded revision.")
    seen = set()
    for row in document["results"]:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("prompt"), str)
            or row["prompt"] not in by_prompt
            or not isinstance(row.get("text"), str)
        ):
            raise ValueError("Each result must match a dataset prompt and contain text.")
        if row["prompt"] in seen:
            raise ValueError("Duplicate generation result for a dataset prompt.")
        seen.add(row["prompt"])
        if (
            type(row.get("tool_call_count")) is not int
            or row["tool_call_count"] < 0
            or (row.get("error") is not None and not isinstance(row["error"], str))
            or not isinstance(row.get("diff", ""), str)
            or not isinstance(row.get("tool_calls", []), list)
            or any(
                not isinstance(call, dict)
                or not isinstance(call.get("code"), str)
                or call.get("tool", "python") not in ("python", "bash")
                for call in row.get("tool_calls", [])
            )
        ):
            raise ValueError("Invalid generation tool count, error or diff.")
    source = GitRepository(document["repo"])
    if source.revision != document["base_revision"]:
        raise ValueError("The generation repository no longer matches its recorded revision.")
    if output is not None and output.resolve().is_relative_to(source.path):
        raise ValueError("The evaluation output must be outside the generation repository.")
    evaluated = []
    for row in document["results"]:
        task = by_prompt[row["prompt"]]
        with tempfile.TemporaryDirectory(prefix="locodellm-evaluation-") as directory:
            reward = verify_export_answer(
                row["text"],
                source,
                Path(directory),
                allow_execution=allow_execution,
                timeout=timeout,
                allowed_converters=allowed_converters(task),
                tool_calls=row.get("tool_calls", []),
                generated_diff=row.get("diff", ""),
                base_eligible=not bool(row.get("error")),
            )
        result = {
            "task_id": task["id"],
            "function": task["function"],
            "tool_call_count": row["tool_call_count"],
            "error": row.get("error"),
            "source_modified": added_python_examples(row.get("diff", "")) is None,
            **reward.to_dict(),
        }
        result["diagnosis"] = diagnosis(result)
        evaluated.append(result)
    report = {
        "generation_file": str(results.resolve()),
        "generation_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset": str(dataset.resolve()),
        "revision": source.revision,
        "results": evaluated,
        "summary": {
            "dataset_tasks": len(tasks),
            "evaluated_tasks": len(evaluated),
            "missing_tasks": len(tasks) - len(evaluated),
            "mean_reward": sum(result["total"] for result in evaluated) / len(evaluated),
            "mean_base_reward": sum(result["base_reward"] for result in evaluated)
            / len(evaluated),
            "mean_progress_reward": sum(result["progress_reward"] for result in evaluated)
            / len(evaluated),
            "max_reward": 2.0,
            "criteria_validated": {
                name: sum(result["criteria"][name] for result in evaluated)
                for name in evaluated[0]["criteria"]
            },
            "generation_tool_calls": sum(result["tool_call_count"] for result in evaluated),
            "tasks_without_tools": sum(result["tool_call_count"] == 0 for result in evaluated),
            "generation_errors": sum(bool(result["error"]) for result in evaluated),
            "independent_replays": sum(len(result["replays"]) for result in evaluated),
        },
    }
    if results.read_bytes() != raw:
        raise RuntimeError("Generation results changed during evaluation; report not saved.")
    if output is not None:
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
    return report


def print_report(report: dict) -> None:
    """Prints factual scores, coverage and per-problem diagnostics to stdout."""
    summary = report["summary"]
    count = summary["evaluated_tasks"]
    criteria = summary["criteria_validated"]
    print("\nATen export evaluation")
    print(f"Evaluated: {count}/{summary['dataset_tasks']} tasks")
    print(f"Mean reward: {summary['mean_reward']:.4f} / 2")
    print(
        f"Original reward: {summary['mean_base_reward']:.4f} / 1; "
        f"progress bonuses: {summary['mean_progress_reward']:.4f} / 1"
    )
    for name, label in (
        ("example_created", "PyTorch examples created"),
        ("test_created", "Export comparison tests created"),
        ("test_passed", "Passing comparison tests"),
        ("failure_confirmed", "Confirmed failure examples"),
    ):
        print(f"{label}: {criteria[name]}/{count}")
    print(f"Confirmed converter failures: {criteria['failure_valid']}/{count}")
    print(f"Independently reproduced failures: {criteria['reproduction_valid']}/{count}")
    print(f"Valid PyTorch references: {criteria['reference_valid']}/{count}")
    print(f"Valid to_onnx calls: {criteria['export_call_valid']}/{count}")
    print(f"Extractable final Python scripts: {criteria['answer_extractable']}/{count}")
    print(
        f"Generation tool calls: {summary['generation_tool_calls']}; "
        f"tasks without tools: {summary['tasks_without_tools']}/{count}"
    )
    print(
        f"Generation errors: {summary['generation_errors']}; "
        f"independent replays: {summary['independent_replays']}"
    )
    if summary["missing_tasks"]:
        print(f"WARNING: {summary['missing_tasks']} missing tasks are not included in the mean.")
    print("\nFunction | Reward | Original | Progress | Tools | Diagnosis")
    for result in report["results"]:
        print(
            f"{result['function']} | {result['total']:.2f} | "
            f"{result['base_reward']:.2f} | {result['progress_reward']:.2f} | "
            f"{result['tool_call_count']} | {result['diagnosis']}"
        )
    print("\nPartial reward is not a bug-discovery success rate.", flush=True)


def run(args) -> None:
    """Evaluates the saved batch and prints its final report."""
    report = evaluate(
        Path(args.input),
        Path(args.results),
        allow_execution=args.allow_code_execution,
        timeout=args.verification_timeout,
        output=Path(args.output) if args.output else None,
    )
    print_report(report)
    if args.output:
        print(f"Report: {Path(args.output).resolve()}", flush=True)
