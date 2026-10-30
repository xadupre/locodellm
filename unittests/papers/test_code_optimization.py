"""Checks real execution, paper equations, reference replay and offline LoRA training."""

import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

PACKAGE = "papers.code-optimization-2026.optim_rl"
environment = importlib.import_module(f"{PACKAGE}.environment")
execution = importlib.import_module(f"{PACKAGE}.execution")
runner = importlib.import_module(f"{PACKAGE}.runner")
devices = importlib.import_module(f"{PACKAGE}.devices")
simulation = importlib.import_module(f"{PACKAGE}.simulation")
algorithm_data = importlib.import_module(f"{PACKAGE}.algorithm_data")


def test_algorithm_dataset_extraction_and_real_references(tmp_path):
    source = tmp_path / "maths"
    source.mkdir()
    path = source / "total.py"
    path.write_text('''"""Computes a shifted sum for independent lists of integers."""
BASE = 3

def helper(values):
    return sum(values)

def total(values, extra=0):
    """Computes the sum of values plus three and the optional extra argument.

    >>> total([])
    3
    >>> total([1])
    4
    >>> total([2, 4])
    9
    >>> total([0], extra=2)
    5
    """
    return helper(values) + BASE + extra

raise RuntimeError("The whole module must never be imported.")
''')
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    (sample,) = algorithm_data.candidates(path, tmp_path, "Test fixture.")
    assert sample["source"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "def total" not in sample["problem"]
    assert sample["source"]["doctest_cases"] == 4
    assert algorithm_data.validate_task(sample) is None
    assert len(json.loads(sample["optimization_tests"][-1]["input"])) == 256
    path.write_text(path.read_text().replace(">>> total([])\n    3", ">>> total([])\n    999"))
    selected, rejected = algorithm_data.prepare_file(path, tmp_path, "Test fixture.")
    assert selected["task_id"] == sample["task_id"]
    assert not rejected
    assert "reference 0, correctness_tests" in algorithm_data.validate_task(selected)


def test_algorithm_dataset_category_split_and_impossible_partition():
    tasks = [
        {"task_id": f"{family}-{index}", "source": {"family": family}}
        for family, count in (("maths", 2), ("strings", 2), ("sorts", 4))
        for index in range(count)
    ]
    train, test = algorithm_data.split_tasks(tasks, 4)
    assert len(train) == len(test) == 4
    assert {t["source"]["family"] for t in train}.isdisjoint(t["source"]["family"] for t in test)
    with pytest.raises(ValueError, match="category-disjoint"):
        algorithm_data.split_tasks(tasks[:6], 3)


def test_committed_algorithm_dataset_contract_and_reference_calibration():
    directory = Path(runner.__file__).resolve().parents[1] / "data" / "thealgorithms-python"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert hashlib.sha256(Path(algorithm_data.__file__).read_bytes()).hexdigest() == (
        manifest["generator_sha256"]
    )
    splits = []
    for split in ("train", "test"):
        path = directory / f"optimization-{split}.json"
        tasks = runner.load_tasks(path)
        assert len(tasks) == 100
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (manifest["sha256"][path.name])
        splits.append(tasks)
    train, test = splits
    assert len({task["task_id"] for task in train + test}) == 200
    for field in ("path", "symbol", "family"):
        assert {t["source"][field] for t in train}.isdisjoint(t["source"][field] for t in test)
    matrix = execution.calibrate(train[0], timeout=3, repeats=1, cpu=None)
    assert len(matrix) == 2 and all(len(row) == 2 for row in matrix)
    assert algorithm_data.validate_task(test[-1]) is None


def task():
    """Returns an explicit stdin/stdout task with private optimization tests."""
    return {
        "task_id": "sum-positive",
        "problem": "Given n, prints the sum of integers from 1 through n.",
        "correctness_tests": [{"input": "1\n", "output": "1\n"}],
        "optimization_tests": [{"input": "3\n", "output": "6\n"}],
        "references": [
            "n = int(input())\nprint(n * (n + 1) // 2)",
            "n = int(input())\nprint(sum(range(n + 1)))",
        ],
    }


def test_inserted_rankings_and_exact_ties():
    references = [[1, 4], [2, 2], [4, 1]]
    assert environment.rankings([0.5, 0.5], references) == (0, 0)
    assert environment.rankings([5, 5], references) == (1, 1)
    assert environment.rankings([2, 2], references) == pytest.approx((1 / 3, 0))
    assert environment.rankings([3, 3], references) == pytest.approx((2 / 3, 1))
    with pytest.raises(ValueError, match="rectangular"):
        environment.rankings([1, 2], [[1], [2]])
    with pytest.raises(ValueError, match="finite"):
        environment.rankings([math.nan], [[1], [2]])


@pytest.mark.parametrize(
    "reward,correct,strict,gate,quality,expected",
    [
        ("correctness", False, False, True, 0, -1),
        ("optimization", False, False, True, 0, 1),
        ("collapsed", False, False, True, 0, -1),
        ("collapsed", True, True, True, 0.25, 0.5),
        ("collapsed", True, False, True, 0, -1),
        ("two-gate", True, True, False, 1, 0),
        ("two-gate", False, False, True, 0, -1),
        ("blend", False, False, True, 0, 0),
    ],
)
def test_signed_reward_compositions(reward, correct, strict, gate, quality, expected):
    env = environment.Environment(reward=reward, quality_map="linear")
    assert environment.compose(correct, strict, gate, quality, env) == pytest.approx(expected)


def test_quality_maps_and_multitask():
    credit = environment.quality_credit
    assert credit(0, True, "bucket") == 1
    assert credit(0.1, True, "bucket") == pytest.approx(0.9)
    assert credit(0.10001, True, "bucket") == pytest.approx(0.8)
    assert credit(1, True, "bucket") == 0
    assert credit(1, True, "range-third") == pytest.approx(1 / 3)
    assert credit(1, True, "range-half") == 0.5
    assert credit(0, False, "binary") == 0
    env = environment.Environment(reward="multitask")
    assert environment.compose(False, False, True, 0, env) == -1
    assert environment.compose(False, False, True, 0, env, optimization_stream=True) == 1


def test_environment_plans_and_timeout_gating():
    sample = task()
    sample["optimization_tests"] *= 3
    matrix = [[0.1, 0.5, 2], [0.2, 1, 4]]
    env = environment.Environment(family="absolute-filter", cutoff=1)
    assert environment.test_plan(sample, matrix, env)[0] == [0, 1]
    env = replace(env, family="length-filter", cutoff=1)
    assert environment.test_plan(sample, matrix, env)[0] == []
    env = replace(env, family="relative-filter", threshold=0.5)
    assert environment.test_plan(sample, matrix, env)[0] == [0, 1]
    env = replace(env, family="relative-timeout", threshold=0.5)
    assert environment.test_plan(sample, matrix, env)[1] == pytest.approx(
        {0: 0.15, 1: 0.75, 2: 3}
    )
    env = replace(env, family="ranked-timeout", threshold=0.5)
    assert environment.test_plan(sample, matrix, env)[1] == {0: 0.1, 1: 0.5, 2: 2}
    sample = task()
    env = environment.Environment(family="timeout", tolerance=0, quality_map="binary")
    correct = [{"status": "passed", "seconds": 0.1}]
    timed_out = [{"status": "timeout", "seconds": 10}]
    record = environment.summarize(sample, [[1], [2]], env, correct, timed_out)
    assert record["strict_correct"] and not record["gate"] and record["reward"] == -1
    failed = [{"status": "failed", "seconds": 10}]
    record = environment.summarize(sample, [[1], [2]], env, correct, failed)
    assert not record["strict_correct"]
    assert runner.evaluation_metrics([{**record, "task_id": "a"}], [1])["p100/pass@1"] == 0
    record = environment.summarize(sample, [[1], [2]], env, correct, timed_out)
    assert runner.evaluation_metrics([{**record, "task_id": "a"}], [1])["p100/pass@1"] == 1


def test_token_weighted_advantages_and_fixed_horizon():
    torch = pytest.importorskip("torch")
    objective = importlib.import_module(f"{PACKAGE}.objective")
    rewards = torch.tensor([1.0, -1.0], requires_grad=True)
    lengths = torch.tensor([1.0, 3.0])
    centered = objective.advantages(rewards, lengths)
    assert centered.tolist() == [1.5, -0.5]
    assert (centered * lengths).sum().item() == 0
    assert not centered.requires_grad
    assert objective.advantages(torch.ones(2), lengths).tolist() == [0, 0]
    current = torch.tensor([math.log(1.5), math.log(0.5)], requires_grad=True)
    old = torch.zeros(2, requires_grad=True)
    loss = objective.clipped_loss(current, old, torch.tensor(1.0), horizon=4)
    assert loss.item() == pytest.approx(-1.7 / 4)
    loss.backward()
    assert current.grad.tolist() == pytest.approx([0, -0.125])
    assert old.grad is None
    with pytest.raises(ValueError, match="positive"):
        objective.advantages(rewards, torch.tensor([1.0, 0.0]))


def test_eight_gpu_selection_and_prompt_scheduling():
    visible = [f"GPU-{index}" for index in range(8)]
    assert devices.select_devices("cuda", visible, set()) == [f"cuda:{i}" for i in range(8)]
    assert devices.select_devices("cuda", visible, {"GPU-0", "7"}) == [
        f"cuda:{i}" for i in range(1, 7)
    ]
    assert devices.select_devices("cuda:0", visible, set(visible)) == ["cuda:0"]
    with pytest.raises(RuntimeError, match="No idle"):
        devices.select_devices("cuda", visible, set(visible))
    tasks = [{"task_id": i} for i in range(16)]
    scheduled = [runner.scheduled_task(tasks, 42, i)["task_id"] for i in range(32)]
    assert set(scheduled[:16]) == set(scheduled[16:]) == set(range(16))
    assert len(devices.verification_cpus(1)) == 1


@pytest.mark.parametrize(
    "source,status",
    [
        ("print(1)", "passed"),
        ("import os\nos.write(1, b'1\\n')", "passed"),
        ("import os\nos.write(1, b'\\xff')", "failed"),
        ("print(2)", "failed"),
        ("def broken(:", "failed"),
        ("raise RuntimeError('candidate failed')", "failed"),
        ("print(1)\nraise SystemExit(0)", "passed"),
        ("print(1)\nraise SystemExit(2)", "failed"),
        ("while True:\n    pass", "timeout"),
    ],
)
def test_real_stdin_stdout_execution(source, status):
    result = execution.execute(source, task()["correctness_tests"][0], timeout=0.1)
    assert result["status"] == status
    assert 0 < result["seconds"] <= 0.1


def test_calibration_and_hidden_optimization_failures():
    sample = task()
    matrix = execution.calibrate(sample, timeout=1, repeats=1, cpu=None)
    assert len(matrix) == 2 and all(row[0] > 0 for row in matrix)
    env = environment.Environment(quality_map="binary")
    result = execution.verify("print(1)", sample, matrix, env, repeats=1, cpu=None)
    assert result["correct"] and not result["strict_correct"] and result["reward"] == -1
    sample["references"][0] = "print(2)"
    with pytest.raises(ValueError, match="failed correctness"):
        execution.calibrate(sample, timeout=1, repeats=1, cpu=None)


def test_replay_simulator_and_pass_at_k():
    sample = task()
    matrix = [[1], [2], [3], [4]]
    correct = [{"status": "passed", "seconds": 0.1}]
    optimization = [{"status": "passed", "seconds": 1.5}]
    env = environment.Environment(family="qp", quality_map="binary")
    row = {
        "task_id": sample["task_id"],
        "reference_key": "ref",
        **environment.summarize(sample, matrix, env, correct, optimization),
    }
    calibration = [{"key": "ref", "task_id": sample["task_id"], "durations": matrix}]
    assert runner.replay([sample], calibration, [row], env)[0] == row
    record = runner.replay(
        [sample], calibration, [row], replace(env, family="timeout", timeout=1, tolerance=0)
    )[0]
    assert record["strict_correct"] and not record["gate"]
    with pytest.raises(ValueError, match="recorded"):
        runner.replay([sample], calibration, [row], replace(env, timeout=20))
    assert environment.pass_at_k(20, 2, 10) == pytest.approx(
        1 - math.comb(18, 10) / math.comb(20, 10)
    )
    assert environment.pass_at_k(2, 2, 1) == 1
    pytest.importorskip("scipy")
    result = simulation.simulate([sample], calibration, env, bands=5, draws=10)
    assert result["activation_curve"][-1] > result["activation_curve"][0]
    assert result["diagnostics"]["steepness"] > 0
    assert simulation.diagnostics([1, 1, 1])["spearman"] is None


def test_calibration_identity_and_duplicate_keys():
    sample = task()
    record = runner.calibration_record("measured", sample, [[0.1], [0.2]], 0)
    runner.validate_calibration([sample], [record])
    with pytest.raises(ValueError, match="Duplicate"):
        runner.validate_calibration([sample], [record, record])
    changed = {**sample, "problem": "Different problem"}
    with pytest.raises(ValueError, match="task content"):
        runner.validate_calibration([changed], [record])
    with pytest.raises(ValueError, match="cover"):
        runner.validate_calibration([sample], [])


def make_tiny_policy(path):
    """Creates a real offline Qwen3 checkpoint emitting runnable one-token programs."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    vocabulary = {"[UNK]": 0, "[EOS]": 1, "[PAD]": 2, "print(1)": 3, "print(2)": 4}
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
    return tokenizer


@pytest.mark.parametrize(
    "checkpoint_options", [("--checkpoint-steps", "1"), ("--checkpoint-minutes", "0.000001")]
)
def test_cpu_cli_training_evaluation_and_offline_replay(tmp_path, checkpoint_options):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    model = tmp_path / "model"
    tokenizer = make_tiny_policy(model)
    policy = importlib.import_module(f"{PACKAGE}.policy")
    sample = task()
    sample["optimization_tests"] = list(sample["correctness_tests"])
    prompt = policy.prompt_tokens(tokenizer, {**sample, "problem": "PUBLIC"}, "cpu")
    assert prompt["input_ids"].numel() > 0
    dataset = tmp_path / "tasks.json"
    dataset.write_text(json.dumps([sample]))
    common = [
        "--dataset",
        str(dataset),
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
    train = tmp_path / "train"
    runner.main(
        [
            "train",
            *common,
            "--output",
            str(train),
            "--steps",
            "1",
            "--group-size",
            "16",
            "--reward",
            "two-gate",
            "--quality-map",
            "linear",
            *checkpoint_options,
        ]
    )
    metrics = json.loads((train / "training-metrics.json").read_text())
    assert metrics["optimizer_updates"] == 1
    assert (train / "checkpoint-1" / "adapter_config.json").exists()
    adapter = load_file(train / "adapter" / "adapter_model.safetensors")
    assert any(
        "lora_B" in key and tensor.abs().max().item() > 0 for key, tensor in adapter.items()
    )
    baseline = tmp_path / "baseline"
    runner.main(["evaluate", *common, "--output", str(baseline)])
    assert json.loads((baseline / "metrics.json").read_text())["samples"] == 2
    evaluate = tmp_path / "evaluate"
    runner.main(
        ["evaluate", *common, "--output", str(evaluate), "--adapter", str(train / "adapter")]
    )
    assert json.loads((evaluate / "metrics.json").read_text())["samples"] == 2
    replay = tmp_path / "replay"
    runner.main(
        [
            "replay",
            *common,
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
        runner.main(["evaluate", *common, "--output", str(evaluate)])
    cli = Path(__file__).resolve().parents[2] / "papers" / "code-optimization-2026" / "run.py"
    result = subprocess.run(
        [sys.executable, str(cli), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert result.returncode == 0 and "simulate" in result.stdout


def test_invalid_inputs_and_execution_opt_in(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps([task(), task()]))
    with pytest.raises(ValueError, match="unique"):
        runner.load_tasks(dataset)
    with pytest.raises(SystemExit):
        runner.main(["calibrate", "--dataset", str(dataset), "--output", str(tmp_path / "run")])
    assert not (tmp_path / "run").exists()
    with pytest.raises(ValueError, match="finite"):
        environment.Environment(timeout=math.nan)


@pytest.mark.parametrize("kind", ["relative-missing", "absolute-missing", "incomplete", "empty"])
def test_local_adapter_rejected_before_dataset_loading(tmp_path, capsys, kind):
    adapter = tmp_path / "adapter"
    if kind == "incomplete":
        adapter.mkdir()
    value = str(adapter)
    if kind == "relative-missing":
        value = f"missing-{tmp_path.name}/run/adapter"
    elif kind == "empty":
        value = ""
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as error:
        runner.main(
            [
                "evaluate",
                "--dataset",
                str(tmp_path / "missing-dataset.json"),
                "--output",
                str(output),
                "--adapter",
                value,
                "--allow-code-execution",
            ]
        )
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "adapter" in message
    assert "missing-dataset" not in message
    assert not output.exists()


def test_remote_adapter_id_is_not_treated_as_a_missing_local_path(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(FileNotFoundError, match="missing-dataset"):
        runner.main(
            [
                "evaluate",
                "--dataset",
                str(tmp_path / "missing-dataset.json"),
                "--output",
                str(output),
                "--adapter",
                "organization/adapter",
                "--allow-code-execution",
            ]
        )
    assert not output.exists()


def test_prompt_keeps_private_verification_material_hidden():
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    policy = importlib.import_module(f"{PACKAGE}.policy")

    class CapturingTokenizer:
        chat_template = None

        def __call__(self, text, **kwargs):
            self.text = text
            torch = pytest.importorskip("torch")
            from transformers import BatchEncoding

            return BatchEncoding({"input_ids": torch.tensor([[1]])})

    tokenizer = CapturingTokenizer()
    private = {
        "problem": "PUBLIC PROBLEM",
        "correctness_tests": [{"input": "PRIVATE INPUT", "output": "PRIVATE OUTPUT"}],
        "optimization_tests": [{"input": "HIDDEN LARGE INPUT", "output": "HIDDEN ANSWER"}],
        "references": ["PRIVATE REFERENCE"],
    }
    policy.prompt_tokens(tokenizer, private, "cpu")
    assert "PUBLIC PROBLEM" in tokenizer.text
    for secret in (
        "PRIVATE INPUT",
        "PRIVATE OUTPUT",
        "HIDDEN LARGE INPUT",
        "HIDDEN ANSWER",
        "PRIVATE REFERENCE",
    ):
        assert secret not in tokenizer.text


def test_all_equal_rewards_skip_optimizer_update(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    model = tmp_path / "model"
    make_tiny_policy(model)
    sample = task()
    sample["correctness_tests"] = [{"input": "", "output": "1000"}]
    sample["optimization_tests"] = list(sample["correctness_tests"])
    sample["references"] = ["print(1000)", "value = 1000\nprint(value)"]
    dataset = tmp_path / "tasks.json"
    dataset.write_text(json.dumps([sample]))
    output = tmp_path / "constant"
    runner.main(
        [
            "train",
            "--dataset",
            str(dataset),
            "--model",
            str(model),
            "--device",
            "cpu",
            "--output",
            str(output),
            "--steps",
            "1",
            "--group-size",
            "2",
            "--max-tokens",
            "1",
            "--repeats",
            "1",
            "--allow-code-execution",
        ]
    )
    assert json.loads((output / "training-metrics.json").read_text())["optimizer_updates"] == 0
    assert (output / "adapter" / "adapter_model.safetensors").exists()
    assert not list(output.glob("checkpoint-*"))
