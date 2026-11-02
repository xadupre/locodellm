"""Trains the stabilized GRPO objective against the existing negative-cost reward."""

import argparse
import contextlib
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import queue
import random
import statistics
import sys
import tempfile
import time

from .devices import discover_devices
from .grading import grade_completion, load_evaluator


def load_tasks(path, evaluator, limit=None):
    """Validates local JSON Lines tasks without changing prompts or grader references."""
    tasks = read_jsonl(path)
    if not tasks or (limit is not None and limit < 1):
        raise ValueError("Dataset must be nonempty and limit must be positive.")
    identities = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("Each task must be a JSON object.")
        if not isinstance(task.get("id"), str) or not task["id"].strip():
            raise ValueError("Each task needs a nonempty string id.")
        if not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
            raise ValueError("Each task needs a nonempty prompt.")
        function, cases = evaluator.reference_from_question(task)
        if function is None or not cases:
            raise ValueError("Each forbidden-word task requires its original function and cases.")
        identity = json.dumps(
            [task.get("source_dataset"), task["id"], task.get("prefix_index")],
            sort_keys=True,
            allow_nan=False,
        )
        if identity in identities:
            raise ValueError(f"Duplicate task identity: {identity}")
        identities.add(identity)
        task["task_id"] = identity
    return tasks[:limit] if limit is not None else tasks


def scored_response(text, task, args):
    """Binds an unchanged evaluator verdict to its raw completion and task."""
    return {
        "task_id": task["task_id"],
        "task_sha256": artifact_digest(task),
        "evaluator_sha256": args.evaluator_sha256,
        "completion": text,
        **grade_completion(text, task, args.evaluator, args.timeout),
    }


def require_score(record):
    """Stops a run on grading infrastructure failures instead of inventing a reward."""
    if record["status"] != "graded":
        raise RuntimeError(f"Grading failed for {record['task_id']}: {record['error']}")


def write_record(stream, record):
    """Persists one finite JSON record and flushes progress immediately."""
    stream.write(json.dumps(record, allow_nan=False) + "\n")
    stream.flush()


def artifact_digest(record):
    """Identifies immutable task content."""
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def implementation_digest():
    """Identifies the actual study source, including uncommitted changes."""
    directory = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in [directory.parent / "run.py", *sorted(directory.glob("*.py"))]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


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


def training(rank, args, tasks, devices, model, tokenizer):
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
                records = []
                for member, (_, text, _, _) in enumerate(group):
                    record = scored_response(text, task, args)
                    records.append(record)
                    write_record(
                        stream,
                        {
                            "iteration": iteration,
                            "sample": member,
                            "rank": rank,
                            "device": devices[rank],
                            **record,
                        },
                    )
                    require_score(record)
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


def evaluation(rank, args, tasks, devices, model, tokenizer, report=None):
    """Distributes individual samples across every idle GPU, even for small datasets."""
    from transformers import set_seed

    from .policy import prompt_tokens, sample

    work = [
        (task_index, member) for task_index in range(len(tasks)) for member in range(args.samples)
    ][rank :: len(devices)]
    with Path(args.output, f"evaluate-rank{rank}.jsonl").open("w") as stream:
        for task_index, member in work:
            task = tasks[task_index]
            set_seed(args.seed + task_index * args.samples + member)
            prompt = prompt_tokens(tokenizer, task, devices[rank])
            tokens, text = sample(model, tokenizer, prompt, args)
            record = scored_response(text, task, args)
            write_record(
                stream,
                {
                    "task_index": task_index,
                    "sample": member,
                    "device": devices[rank],
                    "rank": rank,
                    "generated_tokens": tokens.shape[1] - prompt["input_ids"].shape[1],
                    **record,
                },
            )
            require_score(record)
            if report is not None:
                report(task_index)


def worker(rank, args, tasks, devices, report=None):
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
                    "torch_version": torch.__version__,
                    "dtype": str(next(model.parameters()).dtype),
                },
                indent=2,
            )
            + "\n"
        )
        set_seed(args.seed + rank)
        if args.command == "train":
            training(rank, args, tasks, devices, model, tokenizer)
        else:
            evaluation(rank, args, tasks, devices, model, tokenizer, report)
    finally:
        if distributed:
            dist.destroy_process_group()


def run_evaluation(args, tasks, devices):
    """Aggregates all workers into one parent-owned response and problem counter."""
    import torch.multiprocessing
    from tqdm import tqdm

    counts = [0] * len(tasks)
    completed = 0
    with tqdm(
        total=len(tasks) * args.samples,
        desc="[evaluate]",
        unit="response",
        postfix={"problems": f"0/{len(tasks)}"},
        file=sys.stdout,
    ) as progress:

        def report(task_index):
            """Counts a response only after its verified record has been flushed."""
            nonlocal completed
            counts[task_index] += 1
            if counts[task_index] == args.samples:
                completed += 1
            progress.set_postfix(problems=f"{completed}/{len(tasks)}", refresh=False)
            progress.update()

        if len(devices) == 1:
            worker(0, args, tasks, devices, report)
        else:
            channel = torch.multiprocessing.get_context("spawn").Queue()
            processes = None
            try:
                processes = torch.multiprocessing.spawn(
                    worker,
                    args=(args, tasks, devices, channel.put),
                    nprocs=len(devices),
                    join=False,
                )
                while not processes.join(timeout=0):
                    try:
                        report(channel.get(timeout=0.1))
                    except queue.Empty:
                        continue
                while True:
                    try:
                        report(channel.get_nowait())
                    except queue.Empty:
                        break
            finally:
                if processes is not None:
                    for process in processes.processes:
                        if process.is_alive():
                            process.terminate()
                    for process in processes.processes:
                        process.join()
                channel.close()
                channel.join_thread()
        if progress.n != progress.total:
            raise RuntimeError("Evaluation workers did not report every completed response.")


def read_jsonl(path):
    """Loads recorded artifacts, allowing I/O and malformed-JSON failures to propagate."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def evaluation_metrics(records, ks):
    """Aggregates unmodified costs and unbiased zero-cost pass-at-k estimates."""
    groups = {}
    for row in records:
        require_score(row)
        if type(row["cost"]) is not int or row["cost"] < 0 or row["reward"] != -row["cost"]:
            raise ValueError("Evaluation requires nonnegative integer costs and reward = -cost.")
        groups.setdefault(row["task_id"], []).append(row)
    if not groups or any(not 1 <= k <= len(group) for group in groups.values() for k in ks):
        raise ValueError("Evaluation needs records and valid pass-at-k sample counts.")
    metrics = {
        "tasks": len(groups),
        "samples": len(records),
        "mean_cost": statistics.mean(row["cost"] for row in records),
        "mean_reward": statistics.mean(row["reward"] for row in records),
        "zero_cost_samples": sum(row["cost"] == 0 for row in records),
    }
    for k in ks:
        scores = []
        for group in groups.values():
            n = len(group)
            failed = sum(row["cost"] != 0 for row in group)
            scores.append(1.0 if failed < k else 1 - math.comb(failed, k) / math.comb(n, k))
        metrics[f"zero_cost/pass@{k}"] = statistics.mean(scores)
    return metrics


def parser():
    """Defines independent training and evaluation with the bundled cost.py."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=["train", "evaluate"])
    result.add_argument("--dataset", required=True, help="Local JSONL train.json or test.json.")
    result.add_argument(
        "--evaluator",
        default=str(Path(__file__).with_name("cost.py")),
        help="Optional trusted cost module override; defaults to the bundled cost.py.",
    )
    result.add_argument("--output", required=True, help="New, absent output directory.")
    result.add_argument("--model", default="Qwen/Qwen3.5-4B")
    result.add_argument("--adapter", help="Local LoRA adapter directory or Hugging Face ID.")
    result.add_argument("--device", default="cuda", help="All idle visible GPUs; cpu; cuda:N.")
    result.add_argument("--limit", type=int)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--timeout", type=float, default=5.0)
    result.add_argument("--steps", type=int, default=10000)
    result.add_argument("--group-size", type=int, default=5)
    result.add_argument("--prompts-per-step", type=int, default=1)
    result.add_argument("--learning-rate", type=float, default=1e-7)
    result.add_argument("--temperature", type=float, default=1.0)
    result.add_argument("--max-tokens", type=int, default=32768)
    result.add_argument("--horizon", type=int, default=32768)
    result.add_argument("--clip", type=float, default=0.2)
    result.add_argument("--checkpoint-steps", type=int, default=0)
    result.add_argument("--checkpoint-minutes", type=float, default=20)
    result.add_argument("--samples", type=int, default=20)
    result.add_argument("--pass-k", type=int, nargs="+", default=[1])
    result.add_argument("--allow-code-execution", action="store_true")
    result.add_argument(
        "--dry-run", action="store_true", help="Validates inputs without a model."
    )
    return result


def validate_adapter(cli, args):
    """Rejects incomplete local adapters before dataset, device or model work."""
    if args.adapter is None:
        return
    if not args.adapter.strip():
        cli.error("--adapter must not be empty; omit it to evaluate the base model.")
    adapter = Path(args.adapter).expanduser()
    local = (
        adapter.exists()
        or adapter.is_absolute()
        or args.adapter.startswith((".", "~"))
        or len(adapter.parts) > 2
    )
    if local:
        if not (adapter / "adapter_config.json").is_file() or not any(
            (adapter / name).is_file()
            for name in ("adapter_model.safetensors", "adapter_model.bin")
        ):
            cli.error(
                f"Local adapter requires adapter_config.json and adapter weights: {adapter}"
            )
        args.adapter = str(adapter.resolve())


def main(argv=None):
    """Validates provenance, snapshots the existing evaluator and publishes completed runs."""
    cli = parser()
    args = cli.parse_args(argv)
    for name in ("steps", "prompts_per_step", "samples", "max_tokens", "horizon"):
        if getattr(args, name) < 1:
            cli.error(f"--{name.replace('_', '-')} must be positive.")
    if args.group_size < 2 or args.max_tokens > args.horizon:
        cli.error("group-size must be >= 2 and max-tokens must be <= horizon.")
    for name in ("learning_rate", "temperature", "checkpoint_minutes", "timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            cli.error(f"--{name.replace('_', '-')} must be finite and positive.")
    if not 0 < args.clip < 1 or args.checkpoint_steps < 0:
        cli.error("clip must be in (0, 1); checkpoint-steps must be nonnegative.")
    if any(not 1 <= k <= args.samples for k in args.pass_k):
        cli.error("Every --pass-k must be between 1 and --samples.")
    if not args.dry_run and not args.allow_code_execution:
        cli.error(
            "Execution requires --allow-code-execution in a disposable offline VM/container."
        )
    if not args.dry_run and os.name != "posix":
        cli.error("Grading requires POSIX resource limits and process groups.")
    validate_adapter(cli, args)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(
            f"--output already exists: {output}; previous runs are never overwritten."
        )
    evaluator_path = Path(args.evaluator).expanduser().resolve(strict=True)
    evaluator_source = evaluator_path.read_bytes()
    evaluator = load_evaluator(evaluator_path)
    tasks = load_tasks(args.dataset, evaluator, args.limit)
    args.evaluator_sha256 = hashlib.sha256(evaluator_source).hexdigest()
    config = {
        **vars(args),
        "reward": "-response_cost(raw_completion, *reference_from_question(task))",
        "evaluator_source": str(evaluator_path),
        "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        "implementation_sha256": implementation_digest(),
        "tasks": len(tasks),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    devices = discover_devices(args.device)
    output.mkdir(parents=True, exist_ok=False)
    args.evaluator = str((output / "evaluator.py").resolve())
    Path(args.evaluator).write_bytes(evaluator_source)
    config.update(devices=devices, evaluator_snapshot=args.evaluator)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"[{args.command}] devices={devices} output={output}", flush=True)
    import torch.multiprocessing

    if args.command == "evaluate":
        run_evaluation(args, tasks, devices)
        records = [
            row
            for path in sorted(output.glob("evaluate-rank*.jsonl"))
            for row in read_jsonl(path)
        ]
        records.sort(key=lambda row: (row["task_index"], row["sample"]))
        if len(records) != len(tasks) * args.samples:
            raise RuntimeError("Evaluation did not produce every requested response.")
        with (output / "evaluate.jsonl").open("w") as stream:
            for record in records:
                write_record(stream, record)
        (output / "metrics.json").write_text(
            json.dumps(evaluation_metrics(records, args.pass_k), indent=2) + "\n"
        )
    else:
        if len(devices) > 1:
            torch.multiprocessing.spawn(
                worker, args=(args, tasks, devices), nprocs=len(devices), join=True
            )
        else:
            worker(0, args, tasks, devices)
        ranks = [
            json.loads(path.read_text()) for path in sorted(output.glob("training-rank*.json"))
        ]
        if len(ranks) != len(devices) or len({row["adapter_sha256"] for row in ranks}) != 1:
            raise RuntimeError("Training replicas did not finish with identical adapter weights.")
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
