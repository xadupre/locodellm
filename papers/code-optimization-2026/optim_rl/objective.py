"""Implements the stabilized token-weighted, fixed-horizon GRPO objective."""

import torch


def advantages(rewards, lengths):
    """Centers returns with token weights, without standard-deviation normalization (50)."""
    if (
        rewards.ndim != 1
        or rewards.shape != lengths.shape
        or len(rewards) < 2
        or not torch.isfinite(rewards).all()
        or not torch.isfinite(lengths).all()
        or not (lengths > 0).all()
    ):
        raise ValueError("Expected at least two finite rewards and positive token lengths.")
    if (rewards == rewards[0]).all():
        return torch.zeros_like(rewards)
    baseline = (rewards * lengths).sum() / lengths.sum()
    return (rewards - baseline).detach()


def clipped_loss(current, old, advantage, *, horizon, clip=0.2):
    """Returns the negative sum of clipped token surrogates divided by fixed N (51)."""
    if (
        horizon < 1
        or not 0 < clip < 1
        or current.shape != old.shape
        or current.numel() < 1
        or not torch.isfinite(current).all()
        or not torch.isfinite(old).all()
        or not torch.isfinite(advantage).all()
    ):
        raise ValueError("Invalid log probabilities, advantage, horizon or clipping width.")
    ratio = (current - old.detach()).exp()
    advantage = advantage.detach()
    return (
        -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).sum()
        / horizon
    )


def log_probs(model, tokens, prompt_length, temperature):
    """Computes log probabilities for generated tokens under the sampling temperature."""
    logits = model(input_ids=tokens, use_cache=False).logits
    logits = logits[:, prompt_length - 1 : -1].float() / temperature
    return logits.log_softmax(-1).gather(-1, tokens[:, prompt_length:].unsqueeze(-1)).squeeze(-1)
