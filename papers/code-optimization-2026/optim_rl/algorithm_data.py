"""Builds a small, verified batch-algorithm corpus from pinned upstream doctests."""

from __future__ import annotations

import argparse
import ast
import copy
import doctest
import hashlib
import itertools
import json
import math
import platform
import subprocess
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .execution import execute

REPOSITORY = "https://github.com/TheAlgorithms/Python"
REVISION = "6e03a48579ca212a08b36be00378f887dc41a003"
STDLIB = {
    "__future__",
    "abc",
    "bisect",
    "cmath",
    "collections",
    "copy",
    "dataclasses",
    "decimal",
    "enum",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "re",
    "statistics",
    "string",
    "typing",
}


def encode(value):
    """Returns the exact, finite JSON representation used by the task contract."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def json_value(value):
    """Checks that literals survive the JSON interface without losing dictionary keys."""
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and json_value(item) for key, item in value.items())
    return False


def literal_cases(function):
    """Extracts distinct direct calls with literal inputs and literal expected outputs."""
    cases = {}
    for example in doctest.DocTestParser().get_examples(ast.get_docstring(function) or ""):
        if example.exc_msg or example.options:
            continue
        try:
            body = ast.parse(example.source).body
            if len(body) != 1 or not isinstance(body[0], ast.Expr):
                continue
            call = body[0].value
            if (
                not isinstance(call, ast.Call)
                or not isinstance(call.func, ast.Name)
                or call.func.id != function.name
                or any(keyword.arg is None for keyword in call.keywords)
            ):
                continue
            arguments = {
                "args": [ast.literal_eval(arg) for arg in call.args],
                "kwargs": {
                    keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords
                },
            }
            expected = ast.literal_eval(example.want.strip())
        except (ValueError, SyntaxError):
            continue
        if expected is not None and json_value(arguments) and json_value(expected):
            key = encode(arguments)
            if key in cases and encode(cases[key][1]) != encode(expected):
                raise ValueError(f"Conflicting upstream doctests for {function.name}: {key}")
            cases[key] = (json.loads(key), json.loads(encode(expected)))
    return list(cases.values())[:8]


def prose(docstring):
    """Removes executable examples while preserving the upstream problem explanation."""
    return "\n".join(
        part.strip()
        for part in doctest.DocTestParser().parse(docstring or "")
        if isinstance(part, str) and part.strip()
    )


class WithoutDocstrings(ast.NodeTransformer):
    """Removes documentation from executable references, not from problem statements."""

    def visit_FunctionDef(self, node):
        """Strips the docstring from a function and its nested definitions."""
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)
        return node

    visit_ClassDef = visit_FunctionDef


def reference_body(tree, function):
    """Extracts transitive definitions and pure standard-library imports without importing."""
    bindings = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            bindings[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bindings[node.target.id] = node
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".")[0]] = ast.copy_location(
                    ast.Import(names=[alias]), node
                )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bindings[alias.asname or alias.name] = node
    needed = {}
    pending = [function.name]
    while pending:
        name = pending.pop()
        if name in needed or name not in bindings:
            continue
        node = bindings[name]
        needed[name] = node
        for child in ast.walk(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                modules = (
                    [alias.name for alias in child.names]
                    if isinstance(child, ast.Import)
                    else [child.module or ""]
                )
                if getattr(child, "level", 0) or any(
                    module.split(".")[0] not in STDLIB for module in modules
                ):
                    raise ValueError("Non-whitelisted import in reference dependencies.")
            if isinstance(child, (ast.Global, ast.Nonlocal)):
                raise ValueError("Stateful reference.")
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                if child.func.id in {
                    "open",
                    "input",
                    "print",
                    "exec",
                    "eval",
                    "compile",
                    "__import__",
                }:
                    raise ValueError("I/O or dynamic execution in reference.")
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                pending.append(child.id)
    nodes = sorted({id(node): node for node in needed.values()}.values(), key=lambda n: n.lineno)
    module = ast.Module(body=copy.deepcopy(nodes), type_ignores=[])
    WithoutDocstrings().visit(module)
    return "from __future__ import annotations\n" + ast.unparse(module) + "\n"


def batch_case(cases, count, offset=0):
    """Builds a batch with known outputs, rotating calls to vary order and multiplicity."""
    batch = [cases[(index + offset) % len(cases)] for index in range(count)]
    return {
        "input": encode([call for call, _ in batch]) + "\n",
        "output": encode([expected for _, expected in batch]) + "\n",
    }


def references(body, symbol, attribution):
    """Wraps one upstream algorithm in uncached and memoized batch implementations."""
    header = "".join(f"# {line}\n" for line in attribution.splitlines())
    prefix = header + body + "\nimport json as __dataset_json\nimport sys as __dataset_sys\n"
    suffix = (
        "print(__dataset_json.dumps(__dataset_results, sort_keys=True, "
        "separators=(',', ':'), ensure_ascii=True, allow_nan=False))\n"
    )
    uncached = (
        "__dataset_results = []\n"
        "for __dataset_call in __dataset_json.load(__dataset_sys.stdin):\n"
        f"    __dataset_results.append({symbol}("
        "*__dataset_call['args'], **__dataset_call['kwargs']))\n"
    )
    cached = (
        "__dataset_results = []\n"
        "__dataset_cache = {}\n"
        "for __dataset_call in __dataset_json.load(__dataset_sys.stdin):\n"
        "    __dataset_key = __dataset_json.dumps(__dataset_call, sort_keys=True)\n"
        "    if __dataset_key not in __dataset_cache:\n"
        f"        __dataset_cache[__dataset_key] = {symbol}("
        "*__dataset_call['args'], **__dataset_call['kwargs'])\n"
        "    __dataset_results.append(__dataset_cache[__dataset_key])\n"
    )
    return [prefix + uncached + suffix, prefix + cached + suffix]


def candidates(path, root, license_text):
    """Creates JSON batch tasks from sufficiently documented, directly testable functions."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    relative = path.relative_to(root).as_posix()
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]
    for function in sorted(functions, key=lambda node: node.name):
        cases = literal_cases(function)
        description = prose(ast.get_docstring(function))
        if len(cases) < 4 or len(description) < 40:
            continue
        try:
            body = reference_body(tree, function)
        except ValueError:
            continue
        origin = f"{REPOSITORY}/blob/{REVISION}/{relative}"
        attribution = f"Adapted from {origin}, function {function.name}.\n\n{license_text}"
        tests = [batch_case(cases, 64, 1), batch_case(cases, 256, 3)]
        if any(len(encode(test).encode()) > 262144 for test in tests):
            continue
        module_description = prose(ast.get_docstring(tree))
        problem = (
            f"Implement {function.name.replace('_', ' ')} for batches of independent calls.\n\n"
            f"{module_description}\n\n{description}\n\n"
            f"Function signature: {function.name}({ast.unparse(function.args)}).\n"
            "Read a JSON array from stdin. Each item has 'args' (positional argument array) "
            "and 'kwargs' (keyword argument object). Evaluate each call independently and "
            "preserve input order. Inputs satisfy the function's documented preconditions. "
            "Repeated identical calls may reuse a result; do not mutate a cached result. "
            "Return values use JSON arrays for Python tuples. "
            "Write the array of results as one JSON line, using "
            "json.dumps(results, sort_keys=True, separators=(',', ':'), ensure_ascii=True, "
            "allow_nan=False). No diagnostics or other output.\n\n"
            f"Example input: {batch_case(cases, 1)['input'].strip()}\n"
            f"Example output: {batch_case(cases, 1)['output'].strip()}"
        )
        yield {
            "task_id": f"thealgorithms/{relative[:-3]}/{function.name}",
            "problem": problem,
            "correctness_tests": [
                batch_case(cases, len(cases)),
                batch_case(cases, len(cases) + 1, 1),
            ],
            "optimization_tests": tests,
            "references": references(body, function.name, attribution),
            "source": {
                "repository": REPOSITORY,
                "revision": REVISION,
                "path": relative,
                "symbol": function.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "family": relative.split("/")[0],
                "license": "MIT",
                "doctest_cases": len(cases),
            },
        }


def validate_task(task, timeout=10.0):
    """Runs both complete references against all known-answer correctness and load tests."""
    for index, reference in enumerate(task["references"]):
        for kind in ("correctness_tests", "optimization_tests"):
            for case in task[kind]:
                result = execute(reference, case, timeout=timeout)
                if result["status"] != "passed":
                    return f"reference {index}, {kind}: {result}"
    return None


def prepare_file(path, root, license_text):
    """Selects one statically eligible problem per file, independently of machine timing."""
    try:
        available = list(candidates(path, root, license_text))
    except (SyntaxError, ValueError) as error:
        return None, [f"{path.relative_to(root)}: {error}"]
    return (available[0] if available else None), []


def split_tasks(tasks, size):
    """Keeps entire upstream source categories on one side of the train/test boundary."""
    families = defaultdict(list)
    for task in tasks:
        families[task["source"]["family"]].append(task)
    subsets = {0: ()}
    for family in sorted(families, key=lambda name: hashlib.sha256(name.encode()).hexdigest()):
        count = len(families[family])
        for total, chosen in list(subsets.items()):
            if total + count <= size:
                subsets.setdefault(total + count, (*chosen, family))
    if size not in subsets or len(tasks) != size * 2:
        raise ValueError("Cannot create equally sized, category-disjoint splits.")
    train_families = set(subsets[size])
    train = [task for task in tasks if task["source"]["family"] in train_families]
    test = [task for task in tasks if task["source"]["family"] not in train_families]
    return train, test


def build(source, output, workers):
    """Builds and atomically publishes 100 training and 100 evaluation problems."""
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Expected upstream revision {REVISION}, got {revision}.")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if dirty:
        raise ValueError("Upstream tracked files must be unmodified.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    license_text = (source / "LICENSE.md").read_text()
    tracked = subprocess.check_output(
        ["git", "-C", str(source), "ls-files", "*.py"], text=True
    ).splitlines()
    families = defaultdict(list)
    for relative in tracked:
        if relative.startswith((".github/", "scripts/", "project_euler/")):
            continue
        families[relative.split("/")[0]].append(source / relative)
    for paths in families.values():
        paths.sort(
            key=lambda path: hashlib.sha256(str(path.relative_to(source)).encode()).hexdigest()
        )
    ordered = [
        path
        for row in itertools.zip_longest(*(families[key] for key in sorted(families)))
        for path in row
        if path is not None
    ]
    tasks, exclusions, symbols = [], [], set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(ordered), workers):
            batch = ordered[start : start + workers]
            results = pool.map(lambda path: prepare_file(path, source, license_text), batch)
            for task, rejected in results:
                exclusions.extend(rejected)
                if task is None:
                    continue
                symbol = task["source"]["symbol"].replace("_", "").lower()
                if symbol in symbols:
                    exclusions.append(f"{task['task_id']}: duplicate normalized symbol.")
                    continue
                symbols.add(symbol)
                tasks.append(task)
                print(f"[dataset] Selected {len(tasks)}/200: {task['task_id']}", flush=True)
                if len(tasks) == 200:
                    break
            if len(tasks) == 200:
                break
        if len(tasks) != 200:
            raise ValueError(
                f"Only {len(tasks)} eligible problems; need 200. Exclusions: {exclusions}"
            )
        for index, (task, error) in enumerate(zip(tasks, pool.map(validate_task, tasks)), 1):
            if error is not None:
                raise ValueError(f"Selected task {task['task_id']} failed validation: {error}")
            print(f"[dataset] Verified {index}/200: {task['task_id']}", flush=True)
    train, test = split_tasks(tasks, 100)
    files = {
        "optimization-train.json": json.dumps(train, indent=2, ensure_ascii=True) + "\n",
        "optimization-test.json": json.dumps(test, indent=2, ensure_ascii=True) + "\n",
    }
    manifest = {
        "repository": REPOSITORY,
        "revision": revision,
        "python_version": platform.python_version(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "construction": "Literal upstream doctest oracles; independent-call JSON batch tasks.",
        "selection": "Static eligibility and source ordering before execution; failures abort.",
        "references": [
            "upstream algorithm with a loop",
            "same algorithm with per-batch memoization",
        ],
        "optimization_batch_sizes": [64, 256],
        "validation": "Both references passed every correctness and optimization test.",
        "validation_timeout_seconds": 10.0,
        "split_unit": "source category; one problem per file; distinct normalized symbols",
        "train": {
            "problems": len(train),
            "families": dict(Counter(t["source"]["family"] for t in train)),
        },
        "test": {
            "problems": len(test),
            "families": dict(Counter(t["source"]["family"] for t in test)),
        },
        "sha256": {
            name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()
        },
        "limitations": [
            "Not DMC-Optim or a reproduction of the paper's corpus.",
            "Load increases by batching known cases, not by scaling individual algorithm inputs.",
            "References share algorithm code; they are not independent correctness oracles.",
            "No duration-spread qualification or semantic/pretraining decontamination.",
        ],
        "exclusions": exclusions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        directory = Path(temporary)
        for name, text in files.items():
            (directory / name).write_text(text)
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (directory / "NOTICE.txt").write_text(
            f"Source: {REPOSITORY}\nRevision: {revision}\n\n{license_text}"
        )
        directory.rename(output)
    print(f"[dataset] Published 100 train + 100 test problems in {output}", flush=True)


def main():
    """Parses explicit source and execution consent before constructing the dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Pinned upstream Git checkout."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--allow-code-execution", action="store_true")
    args = parser.parse_args()
    if not args.allow_code_execution:
        parser.error(
            "Reference execution requires --allow-code-execution; this is not a sandbox."
        )
    if args.workers < 1:
        parser.error("--workers must be positive.")
    build(args.source.resolve(), args.output.resolve(), args.workers)


if __name__ == "__main__":
    main()
