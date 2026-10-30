"""Runs synchronous distributed GRPO, generation, replay and reference calibration."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path

from .devices import discover_devices, verification_cpus
from .environment import Environment, pass_at_k, summarize, test_plan, validate_matrix
from .execution import calibrate, extract_code, verify


def load_tasks(path, limit=None):
    """Validates explicit local stdin/stdout tasks without downloading any corpus."""
    tasks = json.loads(Path(path).read_text())
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Dataset must be a nonempty JSON list.")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive.")
        tasks = tasks[:limit]
    ids = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise TypeError("Each dataset task must be a JSON object.")
        task_id = task.get("task_id")
        if not isinstance(task_id, (str, int)) or task_id in ids:
            raise ValueError("Every task needs a unique string/integer task_id.")
        ids.add(task_id)
        if not isinstance(task.get("problem"), str) or not task["problem"].strip():
            raise ValueError("Every task needs a nonempty problem statement.")
        for field in ("correctness_tests", "optimization_tests"):
            tests = task.get(field)
            if (
                not isinstance(tests, list)
                or not tests
                or any(
                    not isinstance(test, dict)
                    or not isinstance(test.get("input"), str)
                    or not isinstance(test.get("output"), str)
                    for test in tests
                )
            ):
                raise ValueError(f"{field} must contain input/output string pairs.")
        references = task.get("references")
        if (
            not isinstance(references, list)
            or len(references) < 2
            or any(not isinstance(source, str) or not source.strip() for source in references)
        ):
            raise ValueError("Each task needs at least two reference program strings.")
    return tasks


def write_record(stream, record):
    """Persists one finite JSON record and flushes progress immediately."""
    stream.write(json.dumps(record, allow_nan=False) + "\n")
    stream.flush()


def task_digest(task):
    """Identifies the complete problem, private tests and reference programs."""
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()


def implementation_digest():
    """Identifies the actual study source, including uncommitted changes."""
    directory = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in [directory.parent / "run.py", *sorted(directory.glob("*.py"))]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def calibration_record(key, task, matrix, cpu):
    """Binds measured durations to immutable task content and a timing CPU."""
    return {
        "key": key,
        "task_id": task["task_id"],
        "cpu": cpu,
        "task_sha256": task_digest(task),
        "durations": matrix,
    }


def validate_calibration(tasks, records):
    """Rejects mismatched tasks, duplicate reference keys and invalid timing matrices."""
    by_id = {task["task_id"]: task for task in tasks}
    keys = set()
    covered = set()
    for row in records:
        if row["key"] in keys:
            raise ValueError(f"Duplicate calibration key: {row['key']}")
        keys.add(row["key"])
        if row["task_id"] not in by_id:
            continue
        task = by_id[row["task_id"]]
        if row.get("task_sha256") != task_digest(task):
            raise ValueError(f"Calibration does not match task content: {row['task_id']}")
        validate_matrix(row["durations"], len(task["optimization_tests"]))
        covered.add(row["task_id"])
    if covered != set(by_id):
        raise ValueError("Calibration does not cover every requested task.")


def save_adapter(model, tokenizer, path, step):
    """Publishes weights and progress only after a complete adapter save."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with tempfile.TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as temporary:
        model.save_pretrained(temporary)
        tokenizer.save_pretrained(temporary)
        Path(temporary, "checkpoint.json").write_text(json.dumps({"optimizer_updates": step}))
        Path(temporary).rename(path)


def scheduled_task(tasks, seed, index):
    """Visits a deterministic shuffled epoch independently of rank scheduling."""
    order = list(range(len(tasks)))
    random.Random(seed + index // len(tasks)).shuffle(order)
    return tasks[order[index % len(tasks)]]


def model_group(model, tokenizer, task, args, device):
    """Generates a same-prompt group before any optimizer update."""
    import torch

    from .objective import log_probs
    from .policy import prompt_tokens, sample

    prompt = prompt_tokens(tokenizer, task, device)
    prompt_length = prompt["input_ids"].shape[1]
    group = []
    for _ in range(args.group_size):
        tokens, text = sample(model, tokenizer, prompt, args)
        with torch.no_grad():
            old = log_probs(model, tokens, prompt_length, args.temperature)
        group.append((tokens, text, old, prompt_length))
    return group


def training(rank, args, tasks, devices, cpu, model, tokenizer, env):
    """Applies the fixed-horizon global sum with DDP-correct gradient scaling."""
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from tqdm import tqdm

    from .objective import advantages, clipped_loss, log_probs

    output = Path(args.output)
    world = len(devices)
    policy = (
        DistributedDataParallel(model, device_ids=[int(devices[rank].split(":")[1])])
        if (world > 1)
        else model
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    updates = 0
    checkpoint_time = time.monotonic()
    with (
        (output / f"train-rank{rank}.jsonl").open("w") as stream,
        (output / f"calibration-rank{rank}.jsonl").open("w") as calibration,
        tqdm(
            total=args.steps, desc="[train]", file=sys.stdout, disable=rank != 0, unit="iteration"
        ) as progress,
    ):
        for iteration in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            local_loss = local_reward = useful = 0.0
            for prompt_index in range(args.prompts_per_step):
                index = (
                    iteration * world * args.prompts_per_step
                    + rank * args.prompts_per_step
                    + prompt_index
                )
                task = scheduled_task(tasks, args.seed, index)
                group = model_group(model, tokenizer, task, args, devices[rank])
                references = calibrate(task, timeout=10.0, repeats=args.repeats, cpu=cpu)
                key = f"r{rank}-i{iteration}-p{prompt_index}"
                write_record(calibration, calibration_record(key, task, references, cpu))
                optimization_stream = random.Random(args.seed + index).random() < 0.5
                records = []
                for member, (_, text, _, _) in enumerate(group):
                    record = verify(
                        extract_code(text),
                        task,
                        references,
                        env,
                        repeats=args.repeats,
                        cpu=cpu,
                        optimization_stream=optimization_stream,
                    )
                    records.append(record)
                    write_record(
                        stream,
                        {
                            "iteration": iteration,
                            "task_id": task["task_id"],
                            "sample": member,
                            "rank": rank,
                            "device": devices[rank],
                            "reference_key": key,
                            "completion": text,
                            "optimization_stream": optimization_stream,
                            **record,
                        },
                    )
                rewards = torch.tensor(
                    [record["reward"] for record in records],
                    device=devices[rank],
                    dtype=torch.float32,
                )
                lengths = torch.tensor(
                    [tokens.shape[1] - length for tokens, _, _, length in group],
                    device=devices[rank],
                    dtype=torch.float32,
                )
                centered = advantages(rewards, lengths)
                useful += float(not (rewards == rewards[0]).all().item())
                local_reward += rewards.mean().item() / args.prompts_per_step
                for member, (tokens, _, old, prompt_length) in enumerate(group):
                    last = (
                        prompt_index == args.prompts_per_step - 1
                        and member == args.group_size - 1
                    )
                    synchronization = (
                        policy.no_sync() if world > 1 and not last else (contextlib.nullcontext())
                    )
                    with synchronization:
                        current = log_probs(policy, tokens, prompt_length, args.temperature)
                        loss = clipped_loss(
                            current, old, centered[member], horizon=args.horizon, clip=args.clip
                        )
                        # DDP averages gradients; equation (51) requires their global sum.
                        (loss * world).backward()
                    local_loss += loss.detach().item()
            totals = torch.tensor([useful, local_loss, local_reward], device=devices[rank])
            if world > 1:
                dist.all_reduce(totals)
            grad_norm = 0.0
            if totals[0].item() > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                ).item()
                optimizer.step()
                updates += 1
            if rank == 0:
                progress.set_postfix(
                    updates=updates,
                    loss=f"{totals[1].item():.5f}",
                    reward=f"{totals[2].item() / world:.3f}",
                    grad=f"{grad_norm:.4f}",
                    useful=int(totals[0].item()),
                )
                progress.update(1)
            due = (
                rank == 0
                and updates > 0
                and (
                    (args.checkpoint_steps > 0 and updates % args.checkpoint_steps == 0)
                    or time.monotonic() - checkpoint_time >= args.checkpoint_minutes * 60
                )
            )
            flag = torch.tensor(int(due), device=devices[rank])
            if world > 1:
                dist.broadcast(flag, src=0)
            checkpoint = output / f"checkpoint-{updates}"
            if flag.item() and not checkpoint.exists():
                if rank == 0:
                    save_adapter(model, tokenizer, checkpoint, updates)
                    progress.write(f"[train] Saved {checkpoint}")
                checkpoint_time = time.monotonic()
            if world > 1:
                dist.barrier()
        if rank == 0:
            save_adapter(model, tokenizer, output / "adapter", updates)
        digest = hashlib.sha256()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                digest.update(name.encode())
                digest.update(
                    parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                )
        (output / f"training-rank{rank}.json").write_text(
            json.dumps(
                {
                    "iterations": args.steps,
                    "optimizer_updates": updates,
                    "adapter_sha256": digest.hexdigest(),
                    "device": devices[rank],
                },
                indent=2,
            )
            + "\n"
        )


def evaluation(rank, args, tasks, devices, cpu, model, tokenizer, env):
    """Distributes individual samples across every idle GPU, even for small datasets."""
    from tqdm import tqdm
    from transformers import set_seed

    from .policy import prompt_tokens, sample

    env = replace(env, family="qp", timeout=10.0, tolerance=1.0)
    cache = {}
    work = [
        (task_index, member) for task_index in range(len(tasks)) for member in range(args.samples)
    ][rank :: len(devices)]
    with (
        Path(args.output, f"evaluate-rank{rank}.jsonl").open("w") as stream,
        Path(args.output, f"calibration-rank{rank}.jsonl").open("w") as calibration,
        tqdm(
            work, desc=f"[evaluate cuda-worker {rank}]", file=sys.stdout, disable=rank != 0
        ) as progress,
    ):
        for task_index, member in progress:
            task = tasks[task_index]
            set_seed(args.seed + task_index * args.samples + member)
            prompt = prompt_tokens(tokenizer, task, devices[rank])
            tokens, text = sample(model, tokenizer, prompt, args)
            if task_index not in cache:
                cache[task_index] = calibrate(task, timeout=10.0, repeats=args.repeats, cpu=cpu)
                write_record(
                    calibration,
                    calibration_record(f"r{rank}-t{task_index}", task, cache[task_index], cpu),
                )
            record = verify(
                extract_code(text), task, cache[task_index], env, repeats=args.repeats, cpu=cpu
            )
            write_record(
                stream,
                {
                    "task_index": task_index,
                    "task_id": task["task_id"],
                    "sample": member,
                    "device": devices[rank],
                    "rank": rank,
                    "reference_key": f"r{rank}-t{task_index}",
                    "completion": text,
                    "generated_tokens": tokens.shape[1] - prompt["input_ids"].shape[1],
                    **record,
                },
            )


def worker(rank, args, tasks, devices, cpus, env):
    """Loads one replica per selected GPU and tears down distributed resources."""
    import torch
    import torch.distributed as dist
    from transformers import set_seed

    from .policy import load_policy

    torch.set_num_threads(1)
    set_seed(args.seed)
    distributed = args.command == "train" and len(devices) > 1
    if devices[rank].startswith("cuda"):
        torch.cuda.set_device(devices[rank])
    if distributed:
        dist.init_process_group(
            "nccl",
            init_method=Path(args.output, ".rendezvous").resolve().as_uri(),
            world_size=len(devices),
            rank=rank,
            device_id=torch.device(devices[rank]),
            timeout=timedelta(minutes=10),
        )
    try:
        model, tokenizer = load_policy(
            args.model, args.adapter, devices[rank], train=args.command == "train"
        )
        Path(args.output, f"model-rank{rank}.json").write_text(
            json.dumps(
                {
                    "model": args.model,
                    "model_revision": getattr(model.config, "_commit_hash", None),
                    "device": devices[rank],
                    "cpu": cpus[rank],
                    "torch_version": torch.__version__,
                    "dtype": str(next(model.parameters()).dtype),
                },
                indent=2,
            )
            + "\n"
        )
        set_seed(args.seed + rank)
        if args.command == "train":
            training(rank, args, tasks, devices, cpus[rank], model, tokenizer, env)
        else:
            evaluation(rank, args, tasks, devices, cpus[rank], model, tokenizer, env)
    finally:
        if distributed:
            dist.destroy_process_group()


def evaluation_metrics(records, ks):
    """Aggregates strict p100/p50/p30/p10 pass@k using leaderboard percentiles."""
    groups = {}
    for record in records:
        groups.setdefault(record["task_id"], []).append(record)
    if not groups:
        raise ValueError("No evaluation records.")
    result = {"tasks": len(groups), "samples": len(records)}
    for threshold in (100, 50, 30, 10):
        for k in ks:
            scores = []
            for group in groups.values():
                passed = sum(
                    row["strict_correct"] and (threshold == 100 or row["qp"] <= threshold / 100)
                    for row in group
                )
                scores.append(pass_at_k(len(group), passed, k))
            result[f"p{threshold}/pass@{k}"] = statistics.mean(scores)
    return result


def replay(tasks, calibration, records, env):
    """Reuses full evaluation traces without regenerating or executing code."""
    by_id = {task["task_id"]: task for task in tasks}
    references = {row["key"]: row for row in calibration}
    result = []
    for row in records:
        task = by_id[row["task_id"]]
        if row["optimization_indices"] != list(range(len(task["optimization_tests"]))):
            raise ValueError("Replay requires complete, unfiltered evaluation traces.")
        if env.timeout > 10:
            raise ValueError("Cannot replay a deadline beyond the recorded 10-second cap.")
        reference = references[row["reference_key"]]
        if reference["task_id"] != task["task_id"]:
            raise ValueError("Evaluation record points to another task's references.")
        matrix = reference["durations"]
        keep, deadlines = test_plan(task, matrix, env)
        optimization = []
        for index in keep:
            recorded = row["optimization"][index]
            optimization.append(
                {"status": "timeout", "seconds": deadlines[index]}
                if recorded["seconds"] >= deadlines[index] and recorded["status"] != "failed"
                else recorded
            )
        result.append({**row, **summarize(task, matrix, env, row["correctness"], optimization)})
    return result


def read_jsonl(path):
    """Loads recorded artifacts, allowing I/O and malformed-JSON failures to propagate."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def parser():
    """Defines explicit model, environment, training and artifact options."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "command", choices=["train", "evaluate", "calibrate", "simulate", "replay"]
    )
    result.add_argument(
        "--dataset", required=True, help="Local JSON tasks with verified references."
    )
    result.add_argument("--output", required=True, help="New, absent output directory.")
    result.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    result.add_argument("--adapter")
    result.add_argument(
        "--device", default="cuda", help="cuda: all idle visible GPUs; cpu; cuda:N."
    )
    result.add_argument("--limit", type=int)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument(
        "--environment",
        default="qar",
        choices=[
            "qar",
            "qp",
            "absolute-filter",
            "relative-filter",
            "length-filter",
            "timeout",
            "relative-timeout",
            "ranked-timeout",
        ],
    )
    result.add_argument("--threshold", type=float, default=0.3)
    result.add_argument("--timeout", type=float, default=10.0)
    result.add_argument("--tolerance", type=float, default=1.0)
    result.add_argument("--cutoff", type=float, default=2.0)
    result.add_argument(
        "--reward",
        default="collapsed",
        choices=["correctness", "optimization", "collapsed", "two-gate", "blend", "multitask"],
    )
    result.add_argument(
        "--quality-map",
        default="binary",
        choices=["binary", "linear", "bucket", "range-third", "range-half"],
    )
    result.add_argument("--blend", type=float, default=0.5)
    result.add_argument("--steps", type=int, default=10000)
    result.add_argument("--group-size", type=int, default=16)
    result.add_argument(
        "--prompts-per-step", type=int, default=1, help="Groups per GPU per iteration."
    )
    result.add_argument("--learning-rate", type=float, default=1e-7)
    result.add_argument("--temperature", type=float, default=1.0)
    result.add_argument("--max-tokens", type=int, default=32768)
    result.add_argument("--horizon", type=int, default=32768)
    result.add_argument("--clip", type=float, default=0.2)
    result.add_argument("--checkpoint-steps", type=int, default=50)
    result.add_argument("--checkpoint-minutes", type=float, default=10)
    result.add_argument("--samples", type=int, default=20)
    result.add_argument("--pass-k", type=int, nargs="+", default=[1, 10])
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--calibration", help="calibration.jsonl from calibrate/evaluate.")
    result.add_argument("--records", help="evaluate.jsonl to replay.")
    result.add_argument("--allow-code-execution", action="store_true")
    return result


def main(argv=None):
    """Validates inputs, executes the selected command, and publishes completed metrics."""
    cli = parser()
    args = cli.parse_args(argv)
    env = Environment(
        family=args.environment,
        threshold=args.threshold,
        timeout=args.timeout,
        tolerance=args.tolerance,
        cutoff=args.cutoff,
        reward=args.reward,
        quality_map=args.quality_map,
        blend=args.blend,
    )
    for name in ("steps", "prompts_per_step", "samples", "repeats", "max_tokens", "horizon"):
        if getattr(args, name) < 1:
            cli.error(f"--{name.replace('_', '-')} must be positive.")
    if args.group_size < 2 or args.max_tokens > args.horizon:
        cli.error("group-size must be >= 2 and max-tokens must be <= horizon.")
    for name in ("learning_rate", "temperature", "checkpoint_minutes"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            cli.error(f"{name} must be finite and positive.")
    if not 0 < args.clip < 1 or args.checkpoint_steps < 0:
        cli.error("clip must be in (0, 1); checkpoint-steps must be nonnegative.")
    if any(not 1 <= k <= args.samples for k in args.pass_k):
        cli.error("Every --pass-k must be between 1 and --samples.")
    live = args.command in ("train", "evaluate", "calibrate")
    if live and not args.allow_code_execution:
        cli.error(
            "Execution requires --allow-code-execution in a disposable, offline VM/container."
        )
    if live and (os.name != "posix" or not hasattr(os, "sched_getaffinity")):
        cli.error(
            "The local timing backend requires Linux CPU affinity and POSIX process groups."
        )
    if args.command in ("simulate", "replay") and not args.calibration:
        cli.error("--calibration is required for offline commands.")
    if args.command == "replay" and not args.records:
        cli.error("--records is required for replay.")
    tasks = load_tasks(args.dataset, args.limit)
    calibration_records = read_jsonl(args.calibration) if args.calibration else []
    if args.command in ("simulate", "replay"):
        validate_calibration(tasks, calibration_records)
    devices = discover_devices(args.device) if args.command in ("train", "evaluate") else ["cpu"]
    cpus = verification_cpus(len(devices)) if live else []
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args),
        "environment_config": asdict(env),
        "devices": devices,
        "verification_cpus": cpus,
        "platform": platform.platform(),
        "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        "implementation_sha256": implementation_digest(),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(
        f"[{args.command}] devices={devices} verification_cpus={cpus} output={output}", flush=True
    )
    if args.command in ("train", "evaluate"):
        import torch.multiprocessing

        if len(devices) > 1:
            torch.multiprocessing.spawn(
                worker, args=(args, tasks, devices, cpus, env), nprocs=len(devices), join=True
            )
        else:
            worker(0, args, tasks, devices, cpus, env)
        if args.command == "train":
            ranks = [
                json.loads(path.read_text())
                for path in sorted(output.glob("training-rank*.json"))
            ]
            if len(ranks) != len(devices) or len({row["adapter_sha256"] for row in ranks}) != 1:
                raise RuntimeError(
                    "Training replicas did not finish with identical adapter weights."
                )
            (output / "training-metrics.json").write_text(
                json.dumps(
                    {
                        "iterations": args.steps,
                        "optimizer_updates": ranks[0]["optimizer_updates"],
                        "devices": devices,
                        "adapter_sha256": ranks[0]["adapter_sha256"],
                    },
                    indent=2,
                )
                + "\n"
            )
        if args.command == "evaluate":
            records = [
                row
                for path in sorted(output.glob("evaluate-rank*.jsonl"))
                for row in read_jsonl(path)
            ]
            records.sort(key=lambda row: (row["task_index"], row["sample"]))
            if len(records) != len(tasks) * args.samples:
                raise RuntimeError("Evaluation did not produce every requested sample.")
            with (output / "evaluate.jsonl").open("w") as stream:
                for row in records:
                    write_record(stream, row)
            metrics = evaluation_metrics(records, args.pass_k)
            (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        with (output / "calibration.jsonl").open("w") as stream:
            for path in sorted(output.glob("calibration-rank*.jsonl")):
                for row in read_jsonl(path):
                    write_record(stream, row)
    elif args.command == "calibrate":
        with (output / "calibration.jsonl").open("w") as stream:
            for index, task in enumerate(tasks):
                write_record(
                    stream,
                    calibration_record(
                        f"t{index}",
                        task,
                        calibrate(task, timeout=10.0, repeats=args.repeats, cpu=cpus[0]),
                        cpus[0],
                    ),
                )
    elif args.command == "simulate":
        from .simulation import simulate

        metrics = simulate(tasks, calibration_records, env, seed=args.seed)
        (output / "simulation.json").write_text(json.dumps(metrics, indent=2) + "\n")
    else:
        records = replay(tasks, calibration_records, read_jsonl(args.records), env)
        with (output / "replay.jsonl").open("w") as stream:
            for row in records:
                write_record(stream, row)
    print(f"[{args.command}] Complete: {output}", flush=True)
