"""Trains tool-using export-discovery policies with the SLM-GRPO protocol."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

import torch

from locodellm.bench.python_harness import PythonHarness
from locodellm.generate.repository import GitRepository
from locodellm.session.pytorch_session import PyTorchSessionState
from locodellm.training.export_verifier import (
    added_python_examples,
    allowed_converters,
    load_tasks,
    verify_export_answer,
)
from locodellm.training.optimization import completion_log_probs, group_advantages, policy_loss
from locodellm.training.policy import load_policy, save_adapter


@dataclass
class Turn:
    """Stores exact sampled tokens and frozen old-policy probabilities for one turn."""

    tokens: torch.Tensor
    prompt_length: int
    old_log_probs: torch.Tensor


class TrainingSession(PyTorchSessionState):
    """Captures assistant actions without treating subsequent tool results as actions."""

    def __init__(self, model, tokenizer, *, temperature):
        super().__init__(model, tokenizer)
        self.temperature = temperature
        self.turns: list[Turn] = []
        self.chat_template_kwargs = {"enable_thinking": False}

    def generate(self, prompt, *args, **kwargs):
        """Records the original tokenization, including sampled end-of-turn tokens."""
        prompt_length = self.get_prompt_length(prompt)
        super().generate(prompt, *args, **kwargs)
        device = self.model.get_input_embeddings().weight.device
        tokens = torch.tensor(
            [[*self.tokens[:prompt_length].tolist(), *self.last_generated_tokens]],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            old = completion_log_probs(
                self.model, tokens, prompt_length, temperature=self.temperature
            )
        self.turns.append(Turn(tokens.cpu(), prompt_length, old.detach().cpu()))
        return self


def trajectory_backward(model, turns: list[Turn], advantage, *, temperature, group_size):
    """Backpropagates the token-average GRPO loss over an entire trajectory.

    Each turn conditions on its own full conversation, but only that turn's
    assistant tokens contribute to the objective. Long and short trajectories
    have equal weight within a group, not equal weight per turn.
    """
    count = sum(turn.old_log_probs.numel() for turn in turns)
    if not count:
        raise ValueError("Cannot optimize an empty trajectory.")
    device = model.get_input_embeddings().weight.device
    total = 0.0
    for turn in turns:
        current = completion_log_probs(
            model, turn.tokens.to(device), turn.prompt_length, temperature=temperature
        )
        loss = policy_loss(
            current,
            turn.old_log_probs.to(device),
            torch.ones_like(current, dtype=torch.bool),
            advantage.reshape(1),
        )
        weighted = loss * (current.numel() / count) / group_size
        weighted.backward()
        total += weighted.detach().item()
    return total


def sample_trajectory(model, tokenizer, task, source, args):
    """Samples a multi-turn rollout and verifies final, tool and added-test evidence."""
    repository = GitRepository.clone(
        str(source.path), str(args.output / "repositories"), source.revision
    )
    session = TrainingSession(model, tokenizer, temperature=args.temperature)
    result = PythonHarness(
        timeout=args.tool_timeout,
        max_tool_calls=args.max_tool_calls,
        bash_only=args.harness == "bash",
        verbose=args.verbose,
    ).run(
        session,
        task["prompt"],
        max_length=args.max_new_tokens,
        directory=str(repository.path),
        final_answer_instructions=(
            "return a complete runnable export example in a Python markdown code block, "
            "with observations outside the block and without a tool call. Use explicit "
            "input_names in to_onnx so its ONNX inputs can be verified independently."
        ),
        max_duration=args.max_duration,
        final_answer_after=args.final_answer_after,
        do_sample=True,
        temperature=args.temperature,
        top_k=0,
        top_p=1.0,
    )
    if not session.turns:
        raise RuntimeError("The rollout produced no sampled assistant actions.")
    diff = repository.diff()
    source_modified = added_python_examples(diff) is None
    reward = verify_export_answer(
        result.text,
        source,
        args.output / "repositories",
        allow_execution=args.allow_code_execution,
        timeout=args.verification_timeout,
        allowed_converters=allowed_converters(task),
        tool_calls=[asdict(call) for call in result.tool_calls],
        generated_diff=diff,
        base_eligible=not bool(result.error),
    )
    return session.turns, {
        "task_id": task["id"],
        "text": result.text,
        "events": result.interactions,
        "error": result.error,
        "timed_out": result.timed_out,
        "protocol_error": result.protocol_error,
        "empty_answer": result.empty_answer,
        "source_modified": source_modified,
        "repository": str(repository.path),
        "revision": repository.revision,
        "turns": [
            {
                "input_ids": turn.tokens[0].tolist(),
                "prompt_length": turn.prompt_length,
                "old_log_probs": turn.old_log_probs[0].tolist(),
            }
            for turn in session.turns
        ],
        **reward.to_dict(),
    }


def train(model, tokenizer, tasks, source, args):
    """Optimizes grouped trajectory rewards with the SLM-GRPO hyperparameters."""
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    optimizer_step = 0
    last_checkpoint = time.monotonic()
    with (args.output / "train.jsonl").open("w", encoding="utf-8") as stream:
        for epoch in range(args.epochs):
            order = list(tasks)
            random.shuffle(order)
            for task in order:
                group = []
                rewards = []
                for member in range(args.group_size):
                    turns, record = sample_trajectory(model, tokenizer, task, source, args)
                    group.append(turns)
                    rewards.append(record["total"])
                    stream.write(
                        json.dumps(
                            {
                                "epoch": epoch + 1,
                                "optimizer_step": optimizer_step,
                                "member": member,
                                **record,
                            }
                        )
                        + "\n"
                    )
                    stream.flush()
                    print(
                        f"[train] epoch={epoch + 1}/{args.epochs} task={task['id']} "
                        f"sample={member + 1}/{args.group_size} turns={len(turns)} "
                        f"reward={record['total']:.4f}",
                        flush=True,
                    )
                device = model.get_input_embeddings().weight.device
                advantages = group_advantages(
                    torch.tensor(rewards, device=device), normalize=args.normalize_advantages
                )
                for _ in range(args.updates_per_group):
                    optimizer.zero_grad()
                    loss = sum(
                        trajectory_backward(
                            model,
                            turns,
                            advantage,
                            temperature=args.temperature,
                            group_size=args.group_size,
                        )
                        for turns, advantage in zip(group, advantages)
                    )
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                    optimizer_step += 1
                    print(
                        f"[train] update={optimizer_step} loss={loss:.6f} "
                        f"grad_norm={norm.item():.4f} "
                        f"mean_reward={sum(rewards) / len(rewards):.4f}",
                        flush=True,
                    )
                    if (
                        args.checkpoint_steps and optimizer_step % args.checkpoint_steps == 0
                    ) or time.monotonic() - last_checkpoint >= args.checkpoint_minutes * 60:
                        save_adapter(
                            model,
                            tokenizer,
                            args.output / f"checkpoint-{optimizer_step}",
                            checkpoint_state={
                                "optimizer_step": optimizer_step,
                                "epoch": epoch + 1,
                            },
                        )
                        last_checkpoint = time.monotonic()
    save_adapter(model, tokenizer, args.output / "adapter")


def run(args: argparse.Namespace):
    """Validates a training run before loading models or cloning repositories."""
    for name in ("epochs", "max_new_tokens", "max_tool_calls", "updates_per_group"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in ("limit", "checkpoint_steps"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.group_size < 2:
        raise ValueError("--group-size must be at least two.")
    for name in (
        "learning_rate",
        "temperature",
        "tool_timeout",
        "verification_timeout",
        "max_duration",
        "final_answer_after",
        "checkpoint_minutes",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive.")
    if args.final_answer_after >= args.max_duration:
        raise ValueError("--final-answer-after must be less than --max-duration.")
    if not args.allow_code_execution:
        raise PermissionError(
            "Training requires --allow-code-execution in an isolated environment."
        )
    args.output = args.output.resolve()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError("--output must be an empty or absent directory.")
    tasks = load_tasks(args.input, args.limit)
    from transformers import GenerationConfig, set_seed

    set_seed(args.seed)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu.")
    revision = tasks[0]["verification"]["revision"]
    if args.repo:
        source = GitRepository(args.repo)
        if source.revision != revision or source.diff():
            raise ValueError("--repo must be clean and at the dataset's recorded revision.")
        if args.output.is_relative_to(source.path):
            raise ValueError("--output must be outside --repo.")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        source = GitRepository.clone(
            tasks[0]["repository"], str(args.output / "repositories"), revision
        )
    args.output.mkdir(parents=True, exist_ok=True)
    configuration = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    configuration.update(
        source_repository=str(source.path),
        revision=source.revision,
        reward="export_failure_reward",
        reference_source_changes=[
            task["verification"].get("relevant_preexisting_source_changes", []) for task in tasks
        ],
    )
    (args.output / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    model, tokenizer = load_policy(args.model, args.adapter, train=True, device=args.device)
    previous = model.generation_config
    model.generation_config = GenerationConfig(
        bos_token_id=previous.bos_token_id,
        eos_token_id=previous.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    train(model, tokenizer, tasks, source, args)
