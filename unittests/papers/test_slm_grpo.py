"""Checks the standalone RLVR paper implementation without model downloads."""

import importlib
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

PACKAGE = "papers.slm-grpo-2026.slm_grpo"
rewards = importlib.import_module(f"{PACKAGE}.rewards")
experiment = importlib.import_module(f"{PACKAGE}.experiment")


def test_prompt_and_extraction():
    task = {
        "text": "Returns the square.",
        "test_list": ["assert square(2) == 4", "assert square(3) == 9"],
        "challenge_test_list": ["assert square(-4) == 16"],
        "code": "REFERENCE MUST NOT LEAK",
    }
    prompt = rewards.build_prompt(task)
    assert task["test_list"][0] in prompt
    assert task["test_list"][1] not in prompt
    assert task["challenge_test_list"][0] not in prompt
    assert task["code"] not in prompt
    assert (
        rewards.extract_code(
            "<think>```python\nwrong\n```</think>\n```python\ndef square(x):\n    return x*x\n```"
        )
        == "def square(x):\n    return x*x"
    )
    assert rewards.extract_code("```python\nimport math\n```\n```py\nx = math.pi\n```") == (
        "import math\n\nx = math.pi"
    )
    assert rewards.extract_code("  def f():\n    pass\n") == "def f():\n    pass"


@pytest.mark.parametrize(
    "source,tests,outcome,error_type",
    [
        ("def f(x):\n    return x + 1", ["assert f(1) == 2"], "passed", None),
        ("def f(x):\n    return x", ["assert f(1) == 2"], "assertion", "AssertionError"),
        ("def f(:", ["assert f(1) == 2"], "interpreter", "SyntaxError"),
        ("def f(x):\n    return missing", ["assert f(1) == 2"], "interpreter", "NameError"),
        ("def f(x):\n    return 1 / 0", ["f(1)"], "runtime", "ZeroDivisionError"),
        ("raise ValueError('bad')", ["assert True"], "runtime", "ValueError"),
        ("import sys\nsys.exit(0)", ["assert False"], "other", "ProcessExit(0)"),
    ],
)
def test_execution_categories(source, tests, outcome, error_type):
    result = rewards.execute_tests(source, tests, allow_execution=True)
    assert (result.outcome, result.error_type) == (outcome, error_type)
    assert (
        rewards.UNIT_REWARDS[outcome]
        == {"passed": 1, "assertion": -0.3, "interpreter": -1, "runtime": -0.6, "other": -1}[
            outcome
        ]
    )


def test_execution_opt_in_timeout_and_setup():
    with pytest.raises(PermissionError):
        rewards.execute_tests("pass", ["assert True"], allow_execution=False)
    with pytest.raises(ValueError, match="at least one test"):
        rewards.execute_tests("pass", [], allow_execution=True)
    with pytest.raises(ValueError, match="finite and positive"):
        rewards.execute_tests("pass", ["assert True"], allow_execution=True, timeout=-1)
    result = rewards.execute_tests(
        "def f():\n    return math.sqrt(4)",
        ["assert f() == 2"],
        setup="import math",
        allow_execution=True,
    )
    assert result.outcome == "passed"
    result = rewards.execute_tests(
        "while True:\n    pass", ["assert True"], allow_execution=True, timeout=0.2
    )
    assert result.outcome == "timeout"
    assert rewards.UNIT_REWARDS[result.outcome] == -0.3


def test_ruff_rewards_are_isolated_from_repository_config():
    source = "import os\n\n\ndef solution(value):\n    return value + 1\n"
    result = rewards.ruff_reward(source)
    assert result.counts == {"error": 0, "high": 0, "normal": 0, "low": 1}
    assert result.raw == -0.2
    assert result.normalized == pytest.approx(-0.2 / 5)
    assert not result.failed
    assert rewards.ruff_reward("result = missing\n").normalized == -1
    assert rewards.ruff_reward("def broken(:").counts["error"] > 0
    assert rewards.ruff_reward("").normalized == 0
    # E501 is enabled by this repository, but not by Ruff's paper/default ruleset.
    assert rewards.ruff_reward('text = "' + "x" * 150 + '"\n').raw == 0
    assert rewards.severity({"code": "NEW", "message": "unmapped diagnostic"}) == "normal"
    assert rewards.severity({"code": None, "message": "diagnostic F821"}) == "error"


def test_reward_modes_and_combined_gate():
    source = "import os\n\n\ndef solution(value):\n    return value + 1\n"
    tests = ["assert solution(1) == 2"]
    unit = rewards.score(source, tests, "unit", allow_execution=True)
    assert unit.total == 1
    assert unit.ruff is None
    raw = rewards.score(source, tests, "ruff")
    assert raw.execution is None
    assert raw.total == -0.2
    assert rewards.score(source, tests, "ruff-normalized").total == pytest.approx(-0.04)
    passed = rewards.score(source, tests, "combined", allow_execution=True)
    assert passed.total == pytest.approx(1 - 0.3 * 0.04)
    assert passed.short == 0
    failed = rewards.score(source, ["assert solution(1) == 7"], "combined", allow_execution=True)
    assert failed.total == pytest.approx(-0.3 - 0.1 * 0.04)
    short = rewards.score("pass", ["assert True"], "combined", allow_execution=True)
    assert short.short == -0.5
    assert short.total == 0.5
    assert short.to_dict()["execution"]["outcome"] == "passed"
    with pytest.raises(ValueError, match="Unknown reward"):
        rewards.score(source, tests, "unknown")


def test_challenge_tests_and_local_dataset(tmp_path):
    task = {
        "text": "Adds one.",
        "test_list": ["assert solution(1) == 2"],
        "challenge_test_list": ["assert solution(2) == 3"],
    }
    dataset = tmp_path / "tasks.json"
    dataset.write_text(json.dumps([task]))
    tasks = experiment.load_tasks(str(dataset), "train", None)
    assert tasks == [task]
    from argparse import Namespace

    result = experiment.verify(
        task,
        "def solution(x):\n    return 2",
        Namespace(reward="unit", allow_code_execution=True, timeout=3.0),
    )
    assert result.execution.outcome == "assertion"


def test_group_advantages_and_clipped_objectives():
    torch = pytest.importorskip("torch")
    optimization = importlib.import_module(f"{PACKAGE}.optimization")
    advantages = optimization.group_advantages(torch.tensor([1.0, -0.3, -0.6, -1.0]))
    assert advantages.mean().item() == pytest.approx(0, abs=1e-7)
    assert torch.equal(
        optimization.group_advantages(torch.ones(8), normalize=True), torch.zeros(8)
    )
    normalized = optimization.group_advantages(torch.tensor([0.0, 2.0]), normalize=True)
    assert normalized.tolist() == pytest.approx([-1 / 1.0001, 1 / 1.0001])
    mask = torch.tensor([[True, True, False]])
    current = torch.tensor([[math.log(1.05), math.log(1.05), 100.0]], requires_grad=True)
    old = torch.zeros_like(current, requires_grad=True)
    advantage = torch.ones(1, requires_grad=True)
    for algorithm, expected in [("grpo", -1.05), ("gspo", -(1.05**2))]:
        loss = optimization.policy_loss(current, old, mask, advantage, algorithm=algorithm)
        assert loss.item() == pytest.approx(expected)
        loss.backward()
        assert current.grad[0, 2] == 0
        assert old.grad is None
        assert advantage.grad is None
        current.grad.zero_()
    for value, adv, expected in [(1.5, 1.0, -1.2), (0.5, -1.0, 0.8)]:
        loss = optimization.policy_loss(
            torch.tensor([[math.log(value)]]),
            torch.zeros(1, 1),
            torch.ones(1, 1, dtype=torch.bool),
            torch.tensor([adv]),
        )
        assert loss.item() == pytest.approx(expected)
    with pytest.raises(ValueError, match="at least two"):
        optimization.group_advantages(torch.ones(1))
    with pytest.raises(ValueError, match="unmasked"):
        optimization.policy_loss(
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.zeros(1, 1, dtype=torch.bool),
            torch.ones(1),
        )


def make_tiny_policy(path):
    """Creates a genuine tiny Qwen3 checkpoint and local tokenizer for offline tests."""
    pytest.importorskip("peft")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    vocabulary = {
        "[UNK]": 0,
        "[EOS]": 1,
        "[PAD]": 2,
        "pass": 3,
        "def": 4,
        "return": 5,
        "value": 6,
        "1": 7,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]"
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    tokenizer.save_pretrained(path)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=len(vocabulary),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=256,
            eos_token_id=1,
            pad_token_id=2,
        )
    )
    model.save_pretrained(path)
    return model, tokenizer


@pytest.mark.parametrize("algorithm", ["grpo", "gspo"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_offline_lora_training_and_evaluation(tmp_path, algorithm, device, capsys):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable.")
    torch.set_num_threads(1)
    model_path = tmp_path / "model"
    make_tiny_policy(model_path)
    task_path = tmp_path / "tasks.json"
    task_path.write_text(
        json.dumps(
            [
                {
                    "task_id": 1,
                    "text": "Returns one.",
                    "test_list": ["assert solution() == 1"],
                    "challenge_test_list": ["assert solution() != 0"],
                }
            ]
        )
    )
    common = [
        "--model",
        str(model_path),
        "--dataset",
        str(task_path),
        "--device",
        device,
        "--max-tokens",
        "4",
    ]
    train_path = tmp_path / "train"
    experiment.main(
        [
            "train",
            *common,
            "--output",
            str(train_path),
            "--algorithm",
            algorithm,
            "--reward",
            "ruff",
            "--epochs",
            "1",
            "--group-size",
            "2",
        ]
    )
    stdout = capsys.readouterr().out
    assert f"device={device} algorithm={algorithm} reward=ruff" in stdout
    assert "[train] Loaded 1 tasks." in stdout
    assert "[train] epoch=1/1 task=1/1 sample=1/2" in stdout
    assert "sample=2/2" in stdout
    assert "step=1/1 update=1/1 loss=" in stdout
    assert "grad_norm=" in stdout
    assert "mean_reward=" in stdout
    assert f"Results: {train_path / 'train.jsonl'}" in stdout
    assert (train_path / "adapter" / "adapter_config.json").exists()
    records = [json.loads(line) for line in (train_path / "train.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert all(record["execution"] is None for record in records)
    eval_path = tmp_path / "evaluate"
    experiment.main(
        [
            "evaluate",
            *common,
            "--adapter",
            str(train_path / "adapter"),
            "--output",
            str(eval_path),
            "--allow-code-execution",
        ]
    )
    stdout = capsys.readouterr().out
    assert "[evaluate]:" in stdout
    assert "100%" in stdout
    assert "1/1" in stdout
    assert "tokens=" in stdout
    assert "outcome=" in stdout
    assert "pass@1=" in stdout
    assert "[evaluate] task=" not in stdout
    assert f"[evaluate] Complete. Results: {eval_path}" in stdout
    metrics = json.loads((eval_path / "metrics.json").read_text())
    assert metrics["tasks"] == 1
    assert metrics["pass@1"] in (0.0, 1.0)
    assert metrics["mean_generated_tokens"] <= 4
    assert metrics["ruff_failures"] == 0
    assert json.loads((eval_path / "config.json").read_text())["split"] == "test"


def test_optimizer_updates_only_adapter_parameters(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    optimization = importlib.import_module(f"{PACKAGE}.optimization")
    torch.set_num_threads(1)
    model_path = tmp_path / "model"
    make_tiny_policy(model_path)
    model, _ = experiment.load_policy(str(model_path), None, train=True, device="cpu")
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=0.01
    )
    tokens = torch.tensor([[3, 4, 5, 6], [3, 4, 7, 5]])
    log_probs = optimization.completion_log_probs(model, tokens, 2)
    loss = optimization.policy_loss(
        log_probs,
        log_probs.detach(),
        torch.ones_like(log_probs, dtype=torch.bool),
        torch.tensor([1.0, -1.0]),
    )
    loss.backward()
    optimizer.step()
    changed = [
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(before[name], parameter)
    ]
    assert changed
    assert all("lora_" in name for name in changed)
    assert all(
        "lora_" in name for name, parameter in model.named_parameters() if parameter.requires_grad
    )


def test_cli_rejects_unsafe_or_invalid_runs(tmp_path):
    with pytest.raises(SystemExit) as exc:
        experiment.main(["evaluate", "--output", str(tmp_path / "run")])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        experiment.main(
            ["train", "--reward", "ruff", "--group-size", "1", "--output", str(tmp_path / "run")]
        )
    assert exc.value.code == 2
    assert not Path(tmp_path / "run").exists()


def test_cli_rejects_unavailable_cuda_before_loading(tmp_path, capsys):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("This check requires an environment without CUDA.")
    with pytest.raises(SystemExit) as exc:
        experiment.main(
            [
                "train",
                "--reward",
                "ruff",
                "--device",
                "cuda",
                "--dataset",
                str(tmp_path / "missing-dataset"),
                "--output",
                str(tmp_path / "run"),
            ]
        )
    assert exc.value.code == 2
    assert "CUDA was requested but is unavailable; use --device cpu." in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_cuda_cli_repairs_cudnn_loader_path(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable.")
    if importlib.util.find_spec("nvidia") is None:
        pytest.skip("cuDNN is not installed as an NVIDIA Python package.")
    if importlib.util.find_spec("nvidia.cudnn") is None:
        pytest.skip("cuDNN is not installed as an NVIDIA Python package.")
    idle_uuids = idle_test_gpu_uuids()
    model_path = tmp_path / "model"
    make_tiny_policy(model_path)
    tasks = tmp_path / "tasks.json"
    tasks.write_text(
        json.dumps([{"text": "Returns one.", "test_list": ["assert solution() == 1"]}])
    )
    output = tmp_path / "evaluate"
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = str(tmp_path / "missing-cudnn")
    environment["CUDA_VISIBLE_DEVICES"] = idle_uuids[0]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "papers" / "slm-grpo-2026" / "run.py"),
            "evaluate",
            "--model",
            str(model_path),
            "--dataset",
            str(tasks),
            "--device",
            "cuda",
            "--max-tokens",
            "4",
            "--output",
            str(output),
            "--allow-code-execution",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("[runtime] Using cuDNN libraries from") == 1
    assert "[evaluate] Complete." in result.stdout
    assert json.loads((output / "metrics.json").read_text())["tasks"] == 1


@pytest.mark.parametrize(
    "device,tasks,uuids,busy,expected",
    [
        ("cuda", 50, ["a", "b"], [], ["cuda:0"]),
        ("cuda", 51, ["a", "b"], [], ["cuda:0", "cuda:1"]),
        ("cuda", 51, ["a"], [], ["cuda:0"]),
        ("cuda", 51, ["a", "b", "c"], ["GPU-a"], ["cuda:1", "cuda:2"]),
        ("cuda", 50, ["a", "b", "c"], ["GPU-a"], ["cuda:1"]),
        ("cuda", 51, ["c", "a", "b"], ["GPU-a"], ["cuda:0", "cuda:2"]),
        ("cuda:1", 51, ["a", "b"], ["GPU-b"], ["cuda:1"]),
        ("cpu", 51, ["a", "b"], [], ["cpu"]),
    ],
)
def test_evaluation_device_selection(device, tasks, uuids, busy, expected):
    assert experiment.evaluation_devices(device, tasks, uuids, busy) == expected


def test_evaluation_rejects_all_gpus_occupied():
    with pytest.raises(RuntimeError, match="No idle CUDA GPU"):
        experiment.evaluation_devices("cuda", 51, ["a", "b"], ["GPU-a", "GPU-b"])


def idle_test_gpu_uuids():
    """Finds visible GPUs unused even by this test process for subprocess checks."""
    torch = pytest.importorskip("torch")
    uuids = [
        "GPU-" + str(torch.cuda.get_device_properties(index).uuid).removeprefix("GPU-")
        for index in range(torch.cuda.device_count())
    ]
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    busy = set(result.stdout.splitlines())
    idle = [uuid for uuid in uuids if uuid not in busy]
    if not idle:
        pytest.skip("No idle GPU is available for this subprocess check.")
    return idle


def test_multi_gpu_evaluation_combines_results(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    if torch.cuda.device_count() < 2:
        pytest.skip("This check requires at least two GPUs.")
    idle_uuids = idle_test_gpu_uuids()
    if len(idle_uuids) < 2:
        pytest.skip("This check requires at least two idle GPUs.")
    model_path = tmp_path / "model"
    make_tiny_policy(model_path)
    tasks = tmp_path / "tasks.json"
    tasks.write_text(
        json.dumps(
            [
                {
                    "task_id": index,
                    "text": "Returns one.",
                    "test_list": ["assert solution() == 1"],
                }
                for index in range(51)
            ]
        )
    )
    output = tmp_path / "evaluate"
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(idle_uuids[:2])
    result = subprocess.run(
        [
            sys.executable,
            str(root / "papers" / "slm-grpo-2026" / "run.py"),
            "evaluate",
            "--model",
            str(model_path),
            "--dataset",
            str(tasks),
            "--device",
            "cuda",
            "--max-tokens",
            "1",
            "--output",
            str(output),
            "--allow-code-execution",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Using 2 GPU workers: cuda:0, cuda:1" in result.stdout
    assert "51/51" in result.stdout
    records = [json.loads(line) for line in (output / "evaluate.jsonl").read_text().splitlines()]
    assert [record["task_id"] for record in records] == list(range(51))
    assert [record["device"] for record in records] == [
        f"cuda:{index % 2}" for index in range(51)
    ]
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["tasks"] == 51
    assert (
        metrics["pass@1"]
        == sum(record["execution"]["outcome"] == "passed" for record in records) / 51
    )
    assert metrics["error_types"] == dict(
        Counter(
            record["execution"]["error_type"]
            for record in records
            if record["execution"]["outcome"] != "passed"
        )
    )
    assert (
        metrics["mean_generated_tokens"]
        == sum(record["generated_tokens"] for record in records) / 51
    )
    for level, mean in metrics["mean_ruff_findings"].items():
        assert mean == sum(record["ruff"]["counts"][level] for record in records) / 51
    assert json.loads((output / "config.json").read_text())["evaluation_devices"] == [
        "cuda:0",
        "cuda:1",
    ]


def test_parallel_worker_errors_do_not_write_success_metrics(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from argparse import Namespace

    model_path = tmp_path / "model"
    make_tiny_policy(model_path)
    output = tmp_path / "evaluate"
    output.mkdir()
    tasks = [
        {
            "text": "Returns one.",
            "test_list": ["assert solution() == 1"],
            "challenge_test_list": None,
        },
        {"text": "Returns one.", "test_list": ["assert solution() == 1"]},
    ]
    args = Namespace(
        model=str(model_path),
        adapter=None,
        device="cpu",
        seed=42,
        temperature=0.6,
        max_tokens=1,
        allow_code_execution=True,
        timeout=3.0,
    )
    with pytest.raises(TypeError, match="NoneType"):
        experiment.evaluate(None, None, tasks, args, output, devices=["cpu", "cpu"])
    assert not (output / "metrics.json").exists()
