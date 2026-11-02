"""Implements the paper's token- and sequence-level clipped objectives."""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, *, normalize=False, epsilon=1e-4):
    """Centers rewards within each prompt's group, optionally scaling by its SD."""
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("A reward group must be one-dimensional with at least two samples.")
    if not torch.isfinite(rewards).all():
        raise ValueError("Rewards must be finite.")
    advantages = rewards - rewards.mean()
    if normalize:
        advantages = advantages / (rewards.std(correction=0) + epsilon)
    return advantages.detach()


def policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    mask: torch.Tensor,
    advantages: torch.Tensor,
    *,
    algorithm="grpo",
    clip=0.2,
):
    """Returns the negative clipped surrogate from equations (3) and (5)."""
    if algorithm not in ("grpo", "gspo"):
        raise ValueError(f"Unknown policy algorithm: {algorithm!r}")
    if not 0 < clip < 1:
        raise ValueError("clip must be between zero and one.")
    if (
        log_probs.ndim != 2
        or log_probs.shape != old_log_probs.shape
        or mask.shape != log_probs.shape
        or advantages.shape != (log_probs.shape[0],)
    ):
        raise ValueError("Expected matching [group, tokens] tensors and [group] advantages.")
    if mask.dtype != torch.bool or not mask.any(dim=1).all():
        raise ValueError("Each completion must contain at least one unmasked token.")
    delta = (log_probs - old_log_probs.detach()).masked_fill(~mask, 0)
    if algorithm == "gspo":
        # Equation (4) uses a SUM, unlike length-normalized GSPO in current TRL.
        ratio = delta.sum(dim=1).exp()
        advantage = advantages.detach()
        return -torch.minimum(
            ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage
        ).mean()
    ratio = delta.exp()
    advantage = advantages.detach().unsqueeze(1)
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    return -((surrogate * mask).sum(dim=1) / mask.sum(dim=1)).mean()


def completion_log_probs(model, tokens, prompt_length, *, temperature=0.9):
    """Computes completion-token log probabilities under the sampling policy."""
    if temperature <= 0 or not 0 < prompt_length < tokens.shape[1]:
        raise ValueError("Expected a positive temperature and a nonempty prompt and completion.")
    logits = model(input_ids=tokens, use_cache=False).logits
    logits = logits[:, prompt_length - 1 : -1].float() / temperature
    targets = tokens[:, prompt_length:]
    return logits.log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
