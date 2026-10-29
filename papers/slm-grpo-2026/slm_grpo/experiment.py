"""Trains LoRA policies and evaluates MBPP code-generation diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from .rewards import PENALTIES, build_prompt, extract_code, score

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]


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


def train(model, tokenizer, tasks, args, output):
    """Optimizes one group per update, saving adapters and per-completion diagnostics."""
    import torch

    from .optimization import completion_log_probs, group_advantages, policy_loss

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    step = 0
    with (output / "train.jsonl").open("w") as stream:
        for epoch in range(args.epochs):
            random.shuffle(tasks)
            for task in tasks:
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
                for _ in range(args.updates_per_group):
                    optimizer.zero_grad()
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
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                step += 1
                stream.flush()
    model.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "adapter")


def evaluate(model, tokenizer, tasks, args, output):
    """Reports single-sample pass@1, exception types, lint severities, and lengths."""
    failures = Counter()
    severities = Counter(dict.fromkeys(PENALTIES, 0))
    passed = tokens = lines = ruff_failures = 0
    with (output / "evaluate.jsonl").open("w") as stream:
        for task in tasks:
            prompt = encode_prompt(tokenizer, task, args.device)
            _, text, length = sample(
                model, tokenizer, prompt, temperature=args.temperature, max_tokens=args.max_tokens
            )
            # Evaluation always measures BOTH correctness and static analysis.
            reward = score(
                extract_code(text),
                task["test_list"] + task.get("challenge_test_list", []),
                "combined",
                allow_execution=args.allow_code_execution,
                timeout=args.timeout,
                setup=task.get("test_setup_code", ""),
            )
            passed += reward.execution.outcome == "passed"
            if reward.execution.outcome != "passed":
                failures[reward.execution.error_type] += 1
            severities.update(reward.ruff.counts)
            ruff_failures += reward.ruff.failed
            tokens += length
            lines += reward.lines
            stream.write(
                json.dumps(
                    {
                        "task_id": task.get("task_id"),
                        "completion": text,
                        "generated_tokens": length,
                        **reward.to_dict(),
                    }
                )
                + "\n"
            )
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
    print(json.dumps(summary, indent=2))


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
    parser.add_argument("--device", help="Defaults to CUDA when available, otherwise CPU.")
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--updates-per-group", type=int, default=1)
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
    for name in ("learning_rate", "timeout", "temperature"):
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
    tasks = load_tasks(args.dataset, args.split, args.limit)
    model, tokenizer = load_policy(
        args.model, args.adapter, train=args.command == "train", device=args.device
    )
    args.output.mkdir(parents=True, exist_ok=True)
    configuration = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    (args.output / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    if args.command == "train":
        train(model, tokenizer, tasks, args, args.output)
    else:
        evaluate(model, tokenizer, tasks, args, args.output)
