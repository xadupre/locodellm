"""Trains LoRA policies and evaluates MBPP code-generation diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, closing
import csv
import json
import multiprocessing
import os
from pathlib import Path
import random
import sys
import subprocess
import tempfile
import time

from .rewards import PENALTIES, build_prompt, extract_code, score

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
EVALUATION_STATE = None


def load_tasks(dataset: str, split: str, limit: int | None):
    """Loads an MBPP split or a local JSON list without exposing reference code."""
    path = Path(dataset)
    if path.is_file():
        tasks = json.loads(path.read_text())
    else:
        from datasets import load_dataset

        tasks = list(load_dataset(dataset, "full", split=split))
    if limit is not None:
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError(f"No tasks in {dataset!r}, split {split!r}.")
    for task in tasks:
        build_prompt(task)
    return tasks


def load_policy(model_name, adapter, *, train, device):
    """Loads a frozen base with trainable or saved LoRA adapters."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model

    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device).index or 0)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = (
        torch.bfloat16
        if device.startswith("cuda") and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=train)
    elif train:
        model = get_peft_model(
            model,
            LoraConfig(
                r=32,
                lora_alpha=64,
                lora_dropout=0.0,
                target_modules=TARGET_MODULES,
                task_type="CAUSAL_LM",
            ),
        )
    # Evaluation mode disables base-model dropout without disabling LoRA gradients.
    model.eval()
    return model, tokenizer


def encode_prompt(tokenizer, task, device):
    """Applies the instruct template with Qwen thinking disabled."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": build_prompt(task)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)


def sample(model, tokenizer, prompt, *, temperature, max_tokens):
    """Samples one completion without top-k or top-p truncation."""
    import torch

    with torch.no_grad():
        tokens = model.generate(
            **prompt,
            do_sample=True,
            temperature=temperature,
            top_k=0,
            top_p=1.0,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_length = prompt["input_ids"].shape[1]
    completion = tokens[0, prompt_length:]
    return tokens, tokenizer.decode(completion, skip_special_tokens=True), len(completion)


def verify(task, text, args):
    """Scores both standard and challenge tests without changing the prompt."""
    return score(
        extract_code(text),
        task["test_list"] + task.get("challenge_test_list", []),
        args.reward,
        allow_execution=args.allow_code_execution,
        timeout=args.timeout,
        setup=task.get("test_setup_code", ""),
    )


def save_adapter(model, tokenizer, path: Path, *, checkpoint_state=None):
    """Publishes an adapter snapshot only after all of its files are saved."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing adapter: {path}")
    with tempfile.TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as directory:
        model.save_pretrained(directory)
        tokenizer.save_pretrained(directory)
        if checkpoint_state is not None:
            Path(directory, "checkpoint.json").write_text(
                json.dumps(checkpoint_state, indent=2) + "\n"
            )
        Path(directory).rename(path)


def train(model, tokenizer, tasks, args, output):
    """Optimizes one group per update, saving adapters and per-completion diagnostics."""
    import torch
    from tqdm import tqdm

    from .optimization import completion_log_probs, group_advantages, policy_loss

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    step = 0
    optimizer_step = 0
    last_checkpoint = time.monotonic()
    with (
        (output / "train.jsonl").open("w") as stream,
        tqdm(
            total=args.epochs * len(tasks) * args.updates_per_group,
            desc="[train]",
            unit="update",
            file=sys.stdout,
        ) as progress,
    ):
        for epoch in range(args.epochs):
            random.shuffle(tasks)
            for task_index, task in enumerate(tasks, start=1):
                counters = {
                    "epoch": f"{epoch + 1}/{args.epochs}",
                    "task": f"{task_index}/{len(tasks)}",
                }
                progress.set_postfix(
                    {**counters, "stage": "sampling", "sample": f"0/{args.group_size}"}
                )
                prompt = encode_prompt(tokenizer, task, args.device)
                prompt_length = prompt["input_ids"].shape[1]
                group = []
                rewards = []
                for member in range(args.group_size):
                    tokens, text, length = sample(
                        model,
                        tokenizer,
                        prompt,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    reward = verify(task, text, args)
                    outcome = reward.execution.outcome if reward.execution else "ruff-only"
                    progress.set_postfix(
                        {
                            **counters,
                            "sample": f"{member + 1}/{args.group_size}",
                            "tokens": length,
                            "reward": f"{reward.total:.4f}",
                            "outcome": outcome,
                        }
                    )
                    with torch.no_grad():
                        old = completion_log_probs(
                            model, tokens, prompt_length, temperature=args.temperature
                        )
                    group.append((tokens, old))
                    rewards.append(reward.total)
                    stream.write(
                        json.dumps(
                            {
                                "step": step,
                                "epoch": epoch,
                                "task_id": task.get("task_id"),
                                "member": member,
                                "generated_tokens": length,
                                "completion": text,
                                **reward.to_dict(),
                            }
                        )
                        + "\n"
                    )
                advantages = group_advantages(
                    torch.tensor(rewards, device=args.device), normalize=args.normalize_advantages
                )
                for update in range(args.updates_per_group):
                    progress.set_postfix(
                        {
                            **counters,
                            "stage": "optimizing",
                            "update": f"{update + 1}/{args.updates_per_group}",
                        }
                    )
                    optimizer.zero_grad()
                    update_loss = 0.0
                    for index, (tokens, old) in enumerate(group):
                        current = completion_log_probs(
                            model, tokens, prompt_length, temperature=args.temperature
                        )
                        loss = policy_loss(
                            current,
                            old,
                            torch.ones_like(current, dtype=torch.bool),
                            advantages[index : index + 1],
                            algorithm=args.algorithm,
                        )
                        (loss / args.group_size).backward()
                        update_loss += loss.detach().item() / args.group_size
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                    optimizer_step += 1
                    progress.set_postfix(
                        {
                            **counters,
                            "update": f"{update + 1}/{args.updates_per_group}",
                            "loss": f"{update_loss:.6f}",
                            "grad_norm": f"{grad_norm.item():.4f}",
                            "mean_reward": f"{sum(rewards) / len(rewards):.4f}",
                        },
                        refresh=False,
                    )
                    progress.update(1)
                    steps_due = (
                        args.checkpoint_steps is not None
                        and optimizer_step % args.checkpoint_steps == 0
                    )
                    time_due = time.monotonic() - last_checkpoint >= args.checkpoint_minutes * 60
                    if steps_due or time_due:
                        stream.flush()
                        checkpoint = output / f"checkpoint-{optimizer_step}"
                        save_adapter(
                            model,
                            tokenizer,
                            checkpoint,
                            checkpoint_state={
                                "optimizer_step": optimizer_step,
                                "epoch": epoch + 1,
                                "task_index": task_index,
                                "task_id": task.get("task_id"),
                                "update_within_group": update + 1,
                            },
                        )
                        last_checkpoint = time.monotonic()
                        progress.write(f"[train] Saved checkpoint: {checkpoint}", file=sys.stdout)
                step += 1
                stream.flush()
    print(f"[train] Saving LoRA adapter and tokenizer to {output / 'adapter'}", flush=True)
    save_adapter(model, tokenizer, output / "adapter")
    print(f"[train] Complete. Results: {output / 'train.jsonl'}", flush=True)


def evaluate_task(model, tokenizer, task, args):
    """Generates and verifies one program, returning its serializable diagnostics."""
    prompt = encode_prompt(tokenizer, task, args.device)
    _, text, length = sample(
        model, tokenizer, prompt, temperature=args.temperature, max_tokens=args.max_tokens
    )
    reward = score(
        extract_code(text),
        task["test_list"] + task.get("challenge_test_list", []),
        "combined",
        allow_execution=args.allow_code_execution,
        timeout=args.timeout,
        setup=task.get("test_setup_code", ""),
    )
    return {
        "task_id": task.get("task_id"),
        "device": args.device,
        "completion": text,
        "generated_tokens": length,
        **reward.to_dict(),
    }


def initialize_evaluation_worker(args, device):
    """Loads one model replica on the GPU assigned to this spawned worker."""
    import torch

    global EVALUATION_STATE
    torch.set_num_threads(1)
    args.device = device
    model, tokenizer = load_policy(args.model, args.adapter, train=False, device=device)
    EVALUATION_STATE = (model, tokenizer, args)


def evaluate_worker(task_index, task):
    """Evaluates one task with a seed independent of worker scheduling."""
    from transformers import set_seed

    if EVALUATION_STATE is None:
        raise RuntimeError("The evaluation worker was not initialized.")
    model, tokenizer, args = EVALUATION_STATE
    set_seed(args.seed + task_index)
    return evaluate_task(model, tokenizer, task, args)


def busy_cuda_uuids():
    """Queries GPUs used by other compute processes without reserving them."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Unable to query GPU availability: {result.stderr.strip()}")
    return {
        uuid.strip()
        for uuid, pid in csv.reader(result.stdout.splitlines())
        if int(pid.strip()) != os.getpid()
    }


def evaluation_devices(device, task_count, gpu_uuids, busy_uuids):
    """Selects idle visible GPUs, parallelizing only above 50 problems."""
    if device != "cuda":
        return [device]
    busy = {uuid.removeprefix("GPU-") for uuid in busy_uuids}
    idle = [
        f"cuda:{index}"
        for index, uuid in enumerate(gpu_uuids)
        if str(uuid).removeprefix("GPU-") not in busy
    ]
    if not idle:
        raise RuntimeError(
            "No idle CUDA GPU is available. Use --device cpu or explicitly select "
            "--device cuda:N to share an occupied GPU."
        )
    return idle[:task_count] if task_count > 50 else idle[:1]


def evaluation_results(model, tokenizer, tasks, args, devices):
    """Yields indexed results serially or from one persistent process per GPU."""
    if len(devices) == 1:
        for index, task in enumerate(tasks):
            yield index, evaluate_task(model, tokenizer, task, args)
        return
    with ExitStack() as stack:
        executors = [
            stack.enter_context(
                ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=initialize_evaluation_worker,
                    initargs=(args, device),
                )
            )
            for device in devices
        ]
        futures = {
            executors[index % len(executors)].submit(evaluate_worker, index, task): index
            for index, task in enumerate(tasks)
        }
        try:
            for future in as_completed(futures):
                yield futures[future], future.result()
        finally:
            for future in futures:
                future.cancel()


def evaluate(model, tokenizer, tasks, args, output, *, devices=None):
    """Aggregates evaluation results with one progress bar across all devices."""
    from tqdm import tqdm

    devices = devices or [args.device]
    failures = Counter()
    severities = Counter(dict.fromkeys(PENALTIES, 0))
    passed = tokens = lines = ruff_failures = 0
    pending = {}
    next_index = 0
    with (
        (output / "evaluate.jsonl").open("w") as stream,
        closing(evaluation_results(model, tokenizer, tasks, args, devices)) as results,
        tqdm(total=len(tasks), desc="[evaluate]", unit="task", file=sys.stdout) as progress,
    ):
        for completed, (task_index, record) in enumerate(results, start=1):
            execution = record["execution"]
            passed += execution["outcome"] == "passed"
            if execution["outcome"] != "passed":
                failures[execution["error_type"]] += 1
            severities.update(record["ruff"]["counts"])
            ruff_failures += record["ruff"]["failed"]
            tokens += record["generated_tokens"]
            lines += record["lines"]
            pending[task_index] = record
            while next_index in pending:
                stream.write(json.dumps(pending.pop(next_index)) + "\n")
                next_index += 1
            stream.flush()
            progress.set_postfix(
                {
                    "device": record["device"],
                    "tokens": record["generated_tokens"],
                    "outcome": execution["outcome"],
                    "pass@1": f"{passed / completed:.4f}",
                },
                refresh=False,
            )
            progress.update(1)
    count = len(tasks)
    summary = {
        "tasks": count,
        "pass@1": passed / count,
        "error_types": dict(failures),
        "mean_generated_tokens": tokens / count,
        "mean_lines": lines / count,
        "mean_ruff_findings": {key: value / count for key, value in severities.items()},
        "ruff_failures": ruff_failures,
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[evaluate] Complete. Results: {output}", flush=True)


def main(argv=None):
    """Parses the training or evaluation command and saves its configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate"])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter")
    parser.add_argument("--dataset", default="google-research-datasets/mbpp")
    parser.add_argument("--split", help="Defaults to train for training, test for evaluation.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--algorithm", choices=["grpo", "gspo"], default="grpo")
    parser.add_argument(
        "--reward", choices=["unit", "ruff", "ruff-normalized", "combined"], default="combined"
    )
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--temperature", type=float, help="Defaults to 0.9 train / 0.6 evaluation."
    )
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--device",
        help="cpu, cuda (idle visible GPUs for evaluation; parallel above 50 tasks), "
        "or cuda:N (explicitly select one GPU, even if occupied). "
        "Defaults to CUDA when available, otherwise CPU.",
    )
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--updates-per-group", type=int, default=1)
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        help="Also saves an adapter checkpoint every N completed optimizer updates.",
    )
    parser.add_argument(
        "--checkpoint-minutes",
        type=float,
        default=10.0,
        help="Saves an adapter checkpoint after this many minutes, at the next completed "
        "optimizer update (default: 10).",
    )
    parser.add_argument("--normalize-advantages", action="store_true")
    parser.add_argument(
        "--allow-code-execution",
        action="store_true",
        help="Allows arbitrary generated Python. Use only in a disposable, offline container.",
    )
    args = parser.parse_args(argv)
    import math

    for name in ("epochs", "max_tokens", "updates_per_group"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.group_size < 2:
        parser.error("--group-size must be at least two.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    if args.checkpoint_steps is not None and args.checkpoint_steps < 1:
        parser.error("--checkpoint-steps must be positive.")
    for name in ("learning_rate", "timeout", "temperature", "checkpoint_minutes"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive.")
    if not args.allow_code_execution and (
        args.command == "evaluate" or args.reward in ("unit", "combined")
    ):
        parser.error("This command requires --allow-code-execution in an isolated environment.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be empty or absent; existing runs are never overwritten.")

    import torch
    from transformers import set_seed

    set_seed(args.seed)
    args.split = args.split or ("train" if args.command == "train" else "test")
    args.temperature = args.temperature or (0.9 if args.command == "train" else 0.6)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable; use --device cpu.")
    print(
        f"[{args.command}] model={args.model} device={args.device} "
        f"algorithm={args.algorithm} reward={args.reward} seed={args.seed} output={args.output}",
        flush=True,
    )
    print(f"[{args.command}] Loading dataset={args.dataset} split={args.split}", flush=True)
    tasks = load_tasks(args.dataset, args.split, args.limit)
    devices = [args.device]
    if args.command == "evaluate":
        if args.device == "cuda":
            uuids = [
                torch.cuda.get_device_properties(index).uuid
                for index in range(torch.cuda.device_count())
            ]
            devices = evaluation_devices(args.device, len(tasks), uuids, busy_cuda_uuids())
            if len(devices) == 1:
                args.device = devices[0]
        args.evaluation_devices = devices
    if len(devices) > 1:
        print(
            f"[evaluate] Loaded {len(tasks)} tasks. Using {len(devices)} GPU workers: "
            f"{', '.join(devices)}",
            flush=True,
        )
        model = tokenizer = None
    else:
        print(
            f"[{args.command}] Loaded {len(tasks)} tasks. Loading model and tokenizer.",
            flush=True,
        )
        model, tokenizer = load_policy(
            args.model, args.adapter, train=args.command == "train", device=args.device
        )
        print(f"[{args.command}] Model ready on {args.device}.", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    configuration = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    (args.output / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    if args.command == "train":
        train(model, tokenizer, tasks, args, args.output)
    else:
        evaluate(model, tokenizer, tasks, args, args.output, devices=devices)
