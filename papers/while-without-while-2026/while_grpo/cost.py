"""Computes the original cost of a raw Python response.

This executes submitted code in-process without a timeout or security sandbox.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
import inspect
import json
import sys


def _execution_errors() -> tuple[type[BaseException], ...]:
    # Includes pytest's BaseException subclass only when the response imports pytest.
    pytest = sys.modules.get("pytest")
    if pytest is not None:
        return Exception, SystemExit, pytest.fail.Exception
    return Exception, SystemExit


def _has_work(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.Assert, ast.Raise, ast.Delete)):
            return True
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            return True
        if isinstance(node, ast.Return) and node.value is not None:
            if not isinstance(node.value, ast.Constant) or node.value.value is not None:
                return True
        if isinstance(node, ast.Expr) and not isinstance(node.value, ast.Constant):
            return True
    return False


def reference_from_question(
    question: Mapping,
) -> tuple[str | None, list[tuple[Sequence[object], object]]]:
    """Reads optional grading fields and rejects incomplete references."""
    if not isinstance(question, Mapping):
        raise ValueError("A grading reference must be an object.")
    if "function" not in question and "cases" not in question:
        return None, []
    function_name = question.get("function")
    cases = question.get("cases")
    if (
        not isinstance(function_name, str)
        or not function_name.isidentifier()
        or not isinstance(cases, list)
        or not cases
    ):
        raise ValueError("A grading reference requires a function name and nonempty cases.")
    for case in cases:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("args"), list)
            or "expected" not in case
        ):
            raise ValueError("Each grading case requires args and expected.")
    return function_name, [(case["args"], case["expected"]) for case in cases]


def response_cost(
    code: str,
    function_name: str | None = None,
    cases: Sequence[tuple[Sequence[object], object]] = (),
    *,
    report_cost: Callable[[int], None] | None = None,
) -> int:
    """Sums lexical, syntax, work and executable-test penalties for one response.

    Optional function/cases add the original expected-output checks. Without
    references, no target correctness is claimed. No catalog, prompt or model
    dependency is needed. Markdown fences remain part of the raw response.
    The optional callback reports accumulated penalties before submitted code runs.
    """
    if function_name is not None and (
        not isinstance(function_name, str) or not function_name.isidentifier()
    ):
        raise ValueError("The target function must be a Python identifier.")
    if function_name is None and cases:
        raise ValueError("Expected-output cases require a function name.")
    cost = int("while" in code)
    try:
        tree = ast.parse(code, filename="<response>")
        program = compile(tree, "<response>", "exec")
    except (SyntaxError, ValueError):
        return cost + 1

    tests = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]
    cost += int(not _has_work(tree)) + int(not tests)
    namespace: dict[str, object] = {"__name__": "__main__"}
    if report_cost is not None:
        report_cost(cost)
    try:
        exec(program, namespace)
    except _execution_errors():
        return cost + 1

    execution_failed = False
    for name in tests:
        if report_cost is not None:
            report_cost(cost + int(execution_failed))
        try:
            test = namespace.get(name)
            if (
                not callable(test)
                or inspect.iscoroutinefunction(test)
                or inspect.isgeneratorfunction(test)
                or inspect.isasyncgenfunction(test)
            ):
                execution_failed = True
                continue
            test()
        except _execution_errors():
            execution_failed = True

    if function_name is None:
        return cost + int(execution_failed)
    function = namespace.get(function_name)
    if not callable(function):
        return cost + 1
    cost += int(execution_failed)
    for arguments, expected in cases:
        if report_cost is not None:
            report_cost(cost)
        try:
            actual = function(*arguments)
            matches = json.loads(json.dumps(actual, allow_nan=False)) == expected
        except _execution_errors():
            cost += 1
        else:
            cost += int(not matches)
    return cost
