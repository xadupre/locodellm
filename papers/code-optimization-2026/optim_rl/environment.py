"""Implements the paper's reference rankings, environments and reward compositions."""

from __future__ import annotations

import math
import statistics
from bisect import bisect_left
from dataclasses import dataclass


@dataclass(frozen=True)
class Environment:
    """Specifies one environment from appendix D.2 and reward from appendix D.4."""

    family: str = "qar"
    threshold: float = 0.3
    timeout: float = 10.0
    tolerance: float = 1.0
    cutoff: float = 2.0
    reward: str = "collapsed"
    quality_map: str = "linear"
    blend: float = 0.5

    def __post_init__(self):
        """Rejects unsupported environments and nonfinite hyperparameters."""
        if self.family not in (
            "qar",
            "qp",
            "absolute-filter",
            "relative-filter",
            "length-filter",
            "timeout",
            "relative-timeout",
            "ranked-timeout",
        ):
            raise ValueError(f"Unknown environment: {self.family}")
        if self.reward not in (
            "correctness",
            "optimization",
            "collapsed",
            "two-gate",
            "blend",
            "multitask",
        ):
            raise ValueError(f"Unknown reward: {self.reward}")
        if self.quality_map not in ("binary", "linear", "bucket", "range-third", "range-half"):
            raise ValueError(f"Unknown quality map: {self.quality_map}")
        for name in ("threshold", "tolerance", "blend"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1].")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be finite and positive.")
        if not math.isfinite(self.cutoff) or self.cutoff <= 0:
            raise ValueError("cutoff must be finite and positive.")


def validate_matrix(matrix, columns):
    """Checks a nonempty rectangular, positive finite human-duration matrix."""
    if not isinstance(matrix, list) or len(matrix) < 2:
        raise ValueError("At least two verified reference solutions are required.")
    if columns < 1:
        raise ValueError("At least one optimization test is required.")
    for row in matrix:
        if len(row) != columns or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for value in row
        ):
            raise ValueError("Reference durations must be positive, finite and rectangular.")


def percentile(value, references):
    """Returns the inserted participant's percentile, using strict-less tie ranks."""
    if not references:
        raise ValueError("A percentile requires references.")
    return sum(other < value for other in references) / len(references)


def rankings(durations, references):
    """Computes QAR and QP by reranking all participants after insertion (28-30)."""
    validate_matrix(references, len(durations))
    if any(not math.isfinite(value) or value <= 0 for value in durations):
        raise ValueError("Candidate durations must be finite and positive.")
    participants = [*references, durations]
    means = mean_ranks(participants)
    return means[-1], percentile(means[-1], means[:-1])


def mean_ranks(participants):
    """Averages strict-less ranks without quadratic reference-pool comparisons."""
    columns = [sorted(column) for column in zip(*participants)]
    return [
        statistics.mean(
            bisect_left(column, value) / (len(participants) - 1)
            for value, column in zip(row, columns)
        )
        for row in participants
    ]


def quantile(values, fraction):
    """Returns the linearly interpolated empirical quantile."""
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def test_plan(task, references, env):
    """Selects optimization tests and deadlines before execution."""
    tests = task["optimization_tests"]
    validate_matrix(references, len(tests))
    means = [statistics.mean(column) for column in zip(*references)]
    keep = list(range(len(tests)))
    if env.family == "absolute-filter":
        keep = [index for index in keep if means[index] < env.cutoff]
    elif env.family == "relative-filter":
        boundary = quantile(means, 1 - env.threshold)
        keep = [index for index in keep if means[index] <= boundary]
    elif env.family == "length-filter":
        keep = [
            index
            for index in keep
            if len(tests[index]["input"]) + len(tests[index]["output"]) < env.cutoff
        ]
    deadlines = dict.fromkeys(keep, env.timeout)
    if env.family == "relative-timeout":
        deadlines = {
            index: max(0.001, quantile([row[index] for row in references], env.threshold))
            for index in keep
        }
    elif env.family == "ranked-timeout":
        scores = mean_ranks(references)
        best = sorted(range(len(references)), key=scores.__getitem__)
        best = best[: max(1, math.ceil(env.threshold * len(best)))]
        deadlines = {
            index: max(0.001, max(references[row][index] for row in best)) for index in keep
        }
    return keep, deadlines


def quality_credit(quality, gate, mapping):
    """Maps lower-is-better quality to the paper's unsigned credit (44-45)."""
    if mapping == "binary":
        return float(gate)
    if mapping == "bucket":
        return 1.0 if quality == 0 else max(0.0, 0.9 - 0.1 * math.floor(10 * (quality - 1e-6)))
    low = {"linear": 0.0, "range-third": 1 / 3, "range-half": 0.5}[mapping]
    return low + (1 - low) * (1 - quality)


def compose(correct, strict, gate, quality, env, *, optimization_stream=False):
    """Combines signed correctness and speed signals using table 41."""
    credit = quality_credit(quality, gate, env.quality_map)
    signed = 2 * credit - 1
    if env.reward == "correctness":
        return 2 * int(correct) - 1
    if env.reward == "optimization":
        return signed
    if env.reward == "collapsed":
        return 2 * int(strict) * credit - 1
    if env.reward == "two-gate":
        return credit if strict else -1.0
    if env.reward == "blend":
        return env.blend * (2 * int(correct) - 1) + (1 - env.blend) * signed
    return signed if optimization_stream else 2 * int(correct) - 1


def summarize(task, references, env, correctness, optimization, *, optimization_stream=False):
    """Uses the same recorded-verdict path for live execution and offline replay."""
    keep, _ = test_plan(task, references, env)
    if len(correctness) != len(task["correctness_tests"]) or len(optimization) != len(keep):
        raise ValueError("Execution records do not match the requested tests.")
    correct = all(record["status"] == "passed" for record in correctness)
    strict = correct and all(record["status"] in ("passed", "timeout") for record in optimization)
    timeout_fraction = sum(record["status"] == "timeout" for record in optimization) / max(
        1, len(optimization)
    )
    qar = qp = None
    if keep:
        qar, qp = rankings(
            [record["seconds"] for record in optimization],
            [[row[index] for index in keep] for row in references],
        )
    ranked = env.family in ("qar", "qp")
    quality = (qar if env.family == "qar" else qp) if ranked and keep else timeout_fraction
    gate = timeout_fraction <= env.tolerance and (
        not ranked or not keep or quality <= env.threshold
    )
    return {
        "correct": correct,
        "strict_correct": strict,
        "gate": gate,
        "quality": quality,
        "qar": qar,
        "qp": qp,
        "timeout_fraction": timeout_fraction,
        "optimization_indices": keep,
        "reward": compose(
            correct, strict, gate, quality, env, optimization_stream=optimization_stream
        ),
        "correctness": correctness,
        "optimization": optimization,
    }


def pass_at_k(total, passed, k):
    """Computes the unbiased pass@k estimator in equation (2)."""
    if not 0 <= passed <= total or not 1 <= k <= total:
        raise ValueError("Expected 0 <= passed <= total and 1 <= k <= total.")
    if total - passed < k:
        return 1.0
    return 1 - math.prod((total - passed - index) / (total - index) for index in range(k))
