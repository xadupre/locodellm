"""Scores independently verified export-failure examples."""


def export_failure_reward(
    *,
    reference_valid: bool,
    export_call_valid: bool,
    failure_valid: bool,
    reproduction_valid: bool,
    answer_extractable: bool = False,
) -> float:
    """Returns an additive reward to maximize, between zero and one.

    Each verified criterion contributes 0.25, independently of the others.
    An extractable final code example receives a minimum of 0.05, not an
    additional bonus on top of the verified criteria.
    The caller supplies evaluation results, not claims extracted from model text.
    This function does not extract examples or execute their verification.

    Args:
        reference_valid: The reference PyTorch module runs on the concrete inputs.
        export_call_valid: The to_onnx call respects the experiment's constraints.
        failure_valid: A genuine export, ONNX execution or numerical failure is
            observed, rather than invalid inputs or an incorrect API call.
        reproduction_valid: Independent replay reproduces the observation at the
            recorded revision, with the conversion route and failure stage identified.
        answer_extractable: The final answer contains a clearly extractable code
            example, even if it has not been verified. Rejected tool requests alone
            do not satisfy this condition.

    Returns:
        The greater of the four contributions' sum and the extraction credit.

    Raises:
        TypeError: A criterion is not a boolean.
    """
    criteria = {
        "reference_valid": reference_valid,
        "export_call_valid": export_call_valid,
        "failure_valid": failure_valid,
        "reproduction_valid": reproduction_valid,
    }
    for name, value in {**criteria, "answer_extractable": answer_extractable}.items():
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean, got {type(value).__name__}.")
    return max(0.25 * sum(criteria.values()), 0.05 if answer_extractable else 0.0)
