"""Checks the independent function-speed study with real workers and offline LoRA."""

import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

PACKAGE = "papers.forbidden-word-optimization-2026.optim_rl"
data = importlib.import_module(f"{PACKAGE}.data")
environment = importlib.import_module(f"{PACKAGE}.environment")
execution = importlib.import_module(f"{PACKAGE}.execution")
runner = importlib.import_module(f"{PACKAGE}.runner")

FAST = "def solve(n):\n    return n + 1"
SLOW = "def solve(n):\n    import time\n    time.sleep(0.002)\n    return n + 1"


def source_task():
    """Returns the external dataset's function/cases/prompt contract."""
    return {
        "id": "increment",
        "function": "solve",
        "prompt": "PUBLIC: Write solve(n), returning n + 1, with a top-level test function.",
        "cases": [{"args": [1], "expected": 2}, {"args": [-1], "expected": 0}],
    }


def prepared_task(tmp_path):
    """Builds known-answer batches and two executable, differently paced references."""
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source_task()) + "\n")
    (task,) = data.load_source_tasks(path, batch_sizes=[2, 4])
    return {**task, "references": [FAST, SLOW]}


def test_source_jsonl_batches_and_prefix_identities(tmp_path):
    path = tmp_path / "train.json"
    original = source_task()
    rows = [
        {**original, "source_dataset": "forbidden-word-avoidance", "prefix_index": prefix}
        for prefix in (0, 1)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    tasks = data.load_source_tasks(path, batch_sizes=[3, 5])
    assert len({task["task_id"] for task in tasks}) == 2
    for task in tasks:
        assert task["problem"] == original["prompt"]
        assert [len(json.loads(test["input"])) for test in task["optimization_tests"]] == [3, 5]
        assert json.loads(task["correctness_tests"][0]["output"]) == [2]
        assert json.loads(task["optimization_tests"][1]["output"]) == [0, 2, 0, 2, 0]
    path.write_text(json.dumps([original]))
    assert len(data.load_source_tasks(path)) == 1


@pytest.mark.parametrize(
    "change", ["duplicate", "bad-function", "missing-expected", "bad-args", "empty-prompt", "nan"]
)
def test_source_validation_fails_explicitly(tmp_path, change):
    row = source_task()
    if change == "bad-function":
        row["function"] = "not valid()"
    elif change == "missing-expected":
        del row["cases"][0]["expected"]
    elif change == "bad-args":
        row["cases"][0]["args"] = "not a list"
    elif change == "empty-prompt":
        row["prompt"] = ""
    elif change == "nan":
        row["cases"][0]["expected"] = math.nan
    path = tmp_path / "source.json"
    path.write_text(json.dumps([row, row] if change == "duplicate" else [row]))
    with pytest.raises(ValueError):
        data.load_source_tasks(path)
    with pytest.raises(ValueError, match="batch sizes"):
        data.load_source_tasks(path, batch_sizes=[0])


@pytest.mark.parametrize(
    "source,status",
    [
        (FAST, "passed"),
        ("def solve(n):\n    return n", "failed"),
        ("def broken(:", "failed"),
        ("def solve(n):\n    raise ValueError('bad return')", "failed"),
        ("def solve(n):\n    while True:\n        pass", "timeout"),
        (FAST + "\nif __name__ == '__main__':\n    raise RuntimeError('script only')", "passed"),
        (
            (
                "class Record:\n    pass\n"
                "def solve(n):\n    import pickle\n"
                "    assert isinstance(pickle.loads(pickle.dumps(Record())), Record)\n"
                "    return n + 1"
            ),
            "passed",
        ),
    ],
)
def test_real_function_verification(tmp_path, source, status):
    task = prepared_task(tmp_path)
    result = execution.execute(
        source, task["correctness_tests"][0], function=task["function"], timeout=0.1
    )
    assert result["status"] == status, result
    assert 0 < result["seconds"] <= 0.1


def test_function_setup_and_output_are_not_part_of_the_batch_result():
    source = (
        "import time\n"
        "time.sleep(0.05)\n"
        "def solve(n):\n"
        "    print('debug output')\n"
        "    return n + 1"
    )
    case = data.batch_case(source_task()["cases"], 2)
    result = execution.execute(source, case, function="solve", timeout=0.001)
    assert result["status"] == "passed", result
    assert result["seconds"] < 0.001


def test_json_returns_and_independent_mutable_arguments():
    case = data.batch_case([{"args": [[1]], "expected": [1, 2]}], 3)
    source = "def solve(values):\n    values.append(2)\n    return values"
    assert execution.execute(source, case, function="solve", timeout=1)["status"] == "passed"
    case = data.batch_case([{"args": [], "expected": "Hello world"}], 1)
    source = "def solve():\n    return 'Hello  world'"
    assert execution.execute(source, case, function="solve", timeout=1)["status"] == "failed"
    case = data.batch_case([{"args": [1], "expected": [2]}, {"args": [-1], "expected": [0]}], 2)
    source = "result = []\ndef solve(n):\n    result[:] = [n + 1]\n    return result"
    assert execution.execute(source, case, function="solve", timeout=1)["status"] == "passed"


def test_calibrated_speed_rewards_and_trace_identity(tmp_path):
    task = prepared_task(tmp_path)
    task["references"] = [SLOW, SLOW.replace("0.002", "0.004")]
    matrix = execution.calibrate(task, timeout=1, repeats=1, cpu=None)
    assert all(fast < slow for fast, slow in zip(*matrix, strict=True))
    env = environment.Environment(family="qp", quality_map="binary")
    fast = execution.verify(FAST, task, matrix, env, repeats=1, cpu=None)
    slow = execution.verify(SLOW.replace("0.002", "0.01"), task, matrix, env, repeats=1, cpu=None)
    assert fast["strict_correct"] and slow["strict_correct"]
    assert fast["reward"] == 1 and slow["reward"] == -1
    wrong = execution.verify(
        "def solve(n):\n    return 99", task, matrix, env, repeats=1, cpu=None
    )
    assert not wrong["strict_correct"] and wrong["reward"] == -1
    reference = runner.calibration_record("measured", task, matrix, None)
    row = {
        "task_id": task["task_id"],
        "reference_key": reference["key"],
        **runner.response_identity(task, reference),
        **fast,
    }
    assert runner.replay([task], [reference], [row], env) == [row]
    changed = {**task, "problem": "Different public specification"}
    updated = runner.calibration_record("measured", changed, matrix, None)
    with pytest.raises(ValueError, match="task content"):
        runner.replay([changed], [updated], [row], env)


def test_stabilized_objective_matches_pr39_equations():
    torch = pytest.importorskip("torch")
    objective = importlib.import_module(f"{PACKAGE}.objective")
    rewards = torch.tensor([1.0, -1.0])
    lengths = torch.tensor([1.0, 3.0])
    assert objective.advantages(rewards, lengths).tolist() == [1.5, -0.5]
    current = torch.tensor([math.log(1.5), math.log(0.5)], requires_grad=True)
    loss = objective.clipped_loss(current, torch.zeros(2), torch.tensor(1.0), horizon=4)
    assert loss.item() == pytest.approx(-1.7 / 4)
    loss.backward()
    assert current.grad.tolist() == pytest.approx([0, -0.125])


def make_tiny_policy(path):
    """Creates a real offline decoder with complete function sources as output tokens."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    vocabulary = {"[UNK]": 0, "[EOS]": 1, "[PAD]": 2, FAST: 3, SLOW: 4}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]"
    )
    tokenizer.save_pretrained(path)
    torch.manual_seed(42)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=5,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=512,
            eos_token_id=1,
            pad_token_id=2,
        )
    )
    model.save_pretrained(path)


def test_cpu_preparation_training_evaluation_and_replay(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    model = tmp_path / "model"
    make_tiny_policy(model)
    dataset = tmp_path / "source.json"
    dataset.write_text(json.dumps(source_task()) + "\n")
    common = [
        "--model",
        str(model),
        "--device",
        "cpu",
        "--max-tokens",
        "1",
        "--horizon",
        "8",
        "--repeats",
        "1",
        "--samples",
        "2",
        "--pass-k",
        "1",
        "--allow-code-execution",
    ]
    prepare = tmp_path / "prepare"
    runner.main(
        [
            "prepare",
            *common,
            "--dataset",
            str(dataset),
            "--output",
            str(prepare),
            "--batch-sizes",
            "2",
            "4",
            "--reference-attempts",
            "40",
            "--timeout",
            "1",
        ]
    )
    tasks = runner.load_tasks(prepare / "tasks.json")
    assert len(tasks) == 1 and len(set(tasks[0]["references"])) == 2
    attempts = runner.read_jsonl(prepare / "reference-attempts-rank0.jsonl")
    assert sum(row["accepted"] for row in attempts) == 2
    runner.validate_calibration(tasks, runner.read_jsonl(prepare / "calibration.jsonl"))
    train = tmp_path / "train"
    runner.main(
        [
            "train",
            *common,
            "--dataset",
            str(prepare / "tasks.json"),
            "--output",
            str(train),
            "--steps",
            "1",
            "--checkpoint-steps",
            "1",
            "--learning-rate",
            "0.001",
            "--reward",
            "two-gate",
            "--quality-map",
            "linear",
        ]
    )
    metrics = json.loads((train / "training-metrics.json").read_text())
    assert metrics["optimizer_updates"] == 1
    assert (train / "checkpoint-1" / "adapter_config.json").is_file()
    weights = load_file(train / "adapter" / "adapter_model.safetensors")
    assert any(
        "lora_B" in key and tensor.abs().max().item() > 0 for key, tensor in weights.items()
    )
    evaluate = tmp_path / "evaluate"
    runner.main(
        [
            "evaluate",
            *common,
            "--dataset",
            str(prepare / "tasks.json"),
            "--output",
            str(evaluate),
            "--adapter",
            str(train / "adapter"),
        ]
    )
    assert json.loads((evaluate / "metrics.json").read_text())["samples"] == 2
    replay = tmp_path / "replay"
    runner.main(
        [
            "replay",
            *common,
            "--dataset",
            str(prepare / "tasks.json"),
            "--output",
            str(replay),
            "--calibration",
            str(evaluate / "calibration.jsonl"),
            "--records",
            str(evaluate / "evaluate.jsonl"),
        ]
    )
    assert len(runner.read_jsonl(replay / "replay.jsonl")) == 2
    with pytest.raises(FileExistsError):
        runner.main(
            ["train", *common, "--dataset", str(prepare / "tasks.json"), "--output", str(train)]
        )


def test_prepare_failure_does_not_publish_success(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    torch.set_num_threads(1)
    model = tmp_path / "model"
    make_tiny_policy(model)
    dataset = tmp_path / "source.json"
    task = source_task()
    task["cases"][0]["expected"] = 1000
    dataset.write_text(json.dumps(task) + "\n")
    output = tmp_path / "rejected"
    with pytest.raises(RuntimeError, match="verified distinct references"):
        runner.main(
            [
                "prepare",
                "--dataset",
                str(dataset),
                "--output",
                str(output),
                "--model",
                str(model),
                "--device",
                "cpu",
                "--batch-sizes",
                "2",
                "--max-tokens",
                "1",
                "--reference-attempts",
                "2",
                "--repeats",
                "1",
                "--allow-code-execution",
            ]
        )
    assert not (output / "tasks.json").exists()
    attempts = runner.read_jsonl(output / "reference-attempts-rank0.jsonl")
    assert len(attempts) == 2 and not any(row["accepted"] for row in attempts)


def test_prompt_only_contains_public_specification(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from transformers import AutoTokenizer

    policy = importlib.import_module(f"{PACKAGE}.policy")
    model = tmp_path / "model"
    make_tiny_policy(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    task = prepared_task(tmp_path)
    prompt = policy.prompt_tokens(tokenizer, task, "cpu")
    expected = tokenizer(task["problem"], return_tensors="pt", add_special_tokens=False)
    assert prompt["input_ids"].tolist() == expected["input_ids"].tolist()


def test_cli_help_and_execution_opt_in(tmp_path):
    with pytest.raises(SystemExit):
        runner.main(
            [
                "prepare",
                "--dataset",
                str(tmp_path / "missing.json"),
                "--output",
                str(tmp_path / "output"),
            ]
        )
    assert not (tmp_path / "output").exists()
    script = Path(runner.__file__).resolve().parents[1] / "run.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert result.returncode == 0 and "prepare" in result.stdout
