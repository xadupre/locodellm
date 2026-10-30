"""Replays measured human programs through the same environment without GPU training."""

import math
import random
import statistics

from .environment import mean_ranks, summarize, test_plan


def diagnostics(curve, window=5):
    """Computes appendix E.2 density, clipped steepness, variance and rank correlation."""
    import numpy
    from scipy.optimize import least_squares
    from scipy.stats import spearmanr

    if len(curve) < 2 or any(not 0 <= value <= 1 for value in curve):
        raise ValueError("A simulator curve needs at least two probabilities.")
    window = min(window, len(curve))
    x = numpy.linspace(0, 1, len(curve))
    y = numpy.asarray(curve)
    smooth = numpy.asarray(
        [
            statistics.mean(curve[index : index + window])
            for index in range(len(curve) - window + 1)
        ]
    )
    fits = [
        least_squares(
            lambda parameters: numpy.clip(parameters[0] * x + parameters[1], 0, 1) - y, [slope, 0]
        )
        for slope in (0.0, 1.0, 4.0)
    ]
    fit = min(fits, key=lambda fitted: float(numpy.square(fitted.fun).sum()))
    slope = float(fit.x[0])

    def correlation(values):
        if len(values) < 2 or len(set(values)) < 2:
            return None
        return float(spearmanr(range(len(values)), values).statistic)

    return {
        "auc": float(y.mean()),
        "smoothed_auc": float(smooth.mean()),
        "steepness": slope / (1 + abs(slope)),
        "variance": float(numpy.square(y[window - 1 :] - smooth).mean()),
        "diagonal_deviation": float(numpy.abs(y - x).mean()),
        "spearman": correlation(curve),
        "smoothed_spearman": correlation(smooth.tolist()),
    }


def simulate(tasks, calibration, env, *, seed=42, bands=20, draws=100):
    """Samples progressively stronger measured references, without predicting policy learning."""
    if bands < 2 or draws < 1:
        raise ValueError("Expected at least two quality bands and positive draw count.")
    by_id = {}
    for row in calibration:
        by_id.setdefault(row["task_id"], row["durations"])
    rng = random.Random(seed)
    prepared = []
    for task in tasks:
        matrix = by_id[task["task_id"]]
        keep, deadlines = test_plan(task, matrix, env)
        scores = mean_ranks(matrix)
        order = sorted(range(len(matrix)), key=scores.__getitem__)
        prepared.append((task, matrix, keep, deadlines, order))
    curves = []
    rewards = []
    for band in range(bands):
        ability = band / (bands - 1)
        activated = []
        returns = []
        for task, matrix, keep, deadlines, order in prepared:
            center = (1 - ability) * (len(order) - 1)
            low = max(0, math.floor(center - 0.1 * len(order)))
            high = min(len(order), math.ceil(center + 0.1 * len(order)) + 1)
            for _ in range(draws):
                row = matrix[rng.choice(order[low:high])]
                optimization = [
                    {
                        "status": "timeout" if row[index] >= deadlines[index] else "passed",
                        "seconds": min(row[index], deadlines[index]),
                    }
                    for index in keep
                ]
                record = summarize(
                    task,
                    matrix,
                    env,
                    [{"status": "passed", "seconds": 1.0} for _ in task["correctness_tests"]],
                    optimization,
                    optimization_stream=rng.random() < 0.5,
                )
                activated.append(int(record["gate"] and record["strict_correct"]))
                returns.append(record["reward"])
        curves.append(statistics.mean(activated))
        rewards.append(statistics.mean(returns))
    return {
        "activation_curve": curves,
        "mean_reward_curve": rewards,
        "diagnostics": diagnostics(curves),
        "bands": bands,
        "draws_per_task_band": draws,
        "limitations": "Correct-reference quality-band replay, not a model-learning prediction.",
    }
