"""Reuses the packaged SLM-GRPO token- and sequence-level objectives."""

from locodellm.training.optimization import (  # noqa: F401
    completion_log_probs,
    group_advantages,
    policy_loss,
)
