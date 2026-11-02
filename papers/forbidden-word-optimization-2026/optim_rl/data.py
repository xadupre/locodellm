"""Prepares function-speed tasks from local forbidden-word-avoidance JSONL files."""

import hashlib
import json
import keyword
from pathlib import Path

from .execution import extract_code, reference_measurements


def batch_case(cases, count, offset=0):
    """Builds independent argument batches with known return values."""
    selected = [cases[(index + offset) % len(cases)] for index in range(count)]
    return {
        "input": json.dumps(
            [{"args": case["args"], "kwargs": case.get("kwargs", {})} for case in selected],
            allow_nan=False,
        )
        + "\n",
        "output": json.dumps(
            [case["expected"] for case in selected], sort_keys=True, allow_nan=False
        )
        + "\n",
    }


def load_source_tasks(path, *, batch_sizes=(64, 256), limit=None):
    """Validates local function contracts without importing code or accessing models."""
    if not batch_sizes or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in batch_sizes
    ):
        raise ValueError("Performance batch sizes must be positive integers.")
    if len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("Performance batch sizes must be distinct.")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive.")
    path = Path(path)
    content = path.read_bytes()
    text = content.decode()
    rows = (
        json.loads(text)
        if text.lstrip().startswith("[")
        else [json.loads(line) for line in text.splitlines() if line.strip()]
    )
    if not isinstance(rows, list) or not rows:
        raise ValueError("Source dataset must contain a nonempty JSON list or JSONL records.")
    if limit is not None:
        rows = rows[:limit]
    source_digest = hashlib.sha256(content).hexdigest()
    tasks, ids = [], set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError("Each source task must be an object.")
        task_id = row.get("id")
        if not isinstance(task_id, (str, int)) or isinstance(task_id, bool):
            raise ValueError("Every source task needs a string/integer id.")
        if "prefix_index" in row:
            prefix = row["prefix_index"]
            if not isinstance(prefix, int) or isinstance(prefix, bool) or prefix < 0:
                raise ValueError("prefix_index must be a nonnegative integer.")
            task_id = (
                f"{row.get('source_dataset', 'forbidden-word-avoidance')}/{task_id}/{prefix}"
            )
        if task_id in ids:
            raise ValueError(f"Duplicate source task identity: {task_id}")
        ids.add(task_id)
        function = row.get("function")
        if (
            not isinstance(function, str)
            or not function.isidentifier()
            or keyword.iskeyword(function)
        ):
            raise ValueError(f"Invalid function name for task {task_id}.")
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"Missing prompt for task {task_id}.")
        cases = row.get("cases")
        if (
            not isinstance(cases, list)
            or not cases
            or any(
                not isinstance(case, dict)
                or not isinstance(case.get("args"), list)
                or not isinstance(case.get("kwargs", {}), dict)
                or "expected" not in case
                for case in cases
            )
        ):
            raise ValueError(f"Task {task_id} needs args/expected cases and optional kwargs.")
        tasks.append(
            {
                "task_id": task_id,
                "problem": row["prompt"],
                "function": function,
                "output_comparison": "json",
                "correctness_tests": [batch_case([case], 1) for case in cases],
                "optimization_tests": [
                    batch_case(cases, size, offset) for offset, size in enumerate(batch_sizes)
                ],
                "source": {
                    "path": str(path.resolve()),
                    "sha256": source_digest,
                    "record_index": index,
                    "id": row["id"],
                    "record_sha256": hashlib.sha256(
                        json.dumps(row, sort_keys=True, allow_nan=False).encode()
                    ).hexdigest(),
                    "batch_sizes": list(batch_sizes),
                },
            }
        )
    return tasks


def prepare_references(rank, args, tasks, devices, cpu, model, tokenizer):
    """Samples distinct verified references and leaves every rejection in the attempt log."""
    from .policy import prompt_tokens, sample
    from .runner import calibration_record, write_record

    output = Path(args.output)
    with (
        (output / f"prepare-rank{rank}.jsonl").open("w") as prepared,
        (output / f"reference-attempts-rank{rank}.jsonl").open("w") as attempts,
        (output / f"calibration-rank{rank}.jsonl").open("w") as calibration,
    ):
        for task_index in range(rank, len(tasks), len(devices)):
            task = tasks[task_index]
            prompt = prompt_tokens(tokenizer, task, devices[rank])
            references, matrix, seen = [], [], set()
            for attempt in range(args.reference_attempts):
                tokens, text = sample(model, tokenizer, prompt, args)
                source = extract_code(text)
                duplicate = source in seen
                seen.add(source)
                measurements = (
                    None
                    if duplicate
                    else reference_measurements(
                        source, task, timeout=args.timeout, repeats=args.repeats, cpu=cpu
                    )
                )
                accepted = measurements is not None and all(
                    record["status"] == "passed"
                    for records in measurements.values()
                    for record in records
                )
                write_record(
                    attempts,
                    {
                        "task_id": task["task_id"],
                        "attempt": attempt,
                        "completion": text,
                        "generated_tokens": tokens.shape[1] - prompt["input_ids"].shape[1],
                        "duplicate": duplicate,
                        "accepted": accepted,
                        "measurements": measurements,
                    },
                )
                if accepted:
                    references.append(source)
                    matrix.append(
                        [record["seconds"] for record in measurements["optimization_tests"]]
                    )
                if len(references) == args.reference_count:
                    break
            if len(references) < args.reference_count:
                raise RuntimeError(
                    f"Task {task['task_id']}: obtained {len(references)} verified distinct "
                    f"references, need {args.reference_count}. See {attempts.name}."
                )
            task = {**task, "references": references}
            write_record(prepared, task)
            write_record(
                calibration, calibration_record(f"r{rank}-t{task_index}", task, matrix, cpu)
            )
            print(
                f"[prepare rank={rank}] {task['task_id']}: "
                f"{len(references)} references verified",
                flush=True,
            )
