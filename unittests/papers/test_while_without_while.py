"""Checks real grading processes, unchanged external costs and offline GRPO training."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

PACKAGE = "papers.while-without-while-2026.while_grpo"
runner = importlib.import_module(f"{PACKAGE}.runner")
grading = importlib.import_module(f"{PACKAGE}.grading")
GOOD = "def solve(value):\n    return value + 1\ndef test_solve():\n    assert solve(1) == 2\n"
BAD = GOOD.replace("value + 1", "value - 1")


def task():
    """Returns a complete question with private references."""
    return {
        "id": "increment",
        "prompt": "PUBLIC",
        "function": "solve",
        "cases": [{"args": [1], "expected": 2}, {"args": [2], "expected": 3}],
    }


@pytest.fixture
def evaluator(tmp_path):
    """Creates an explicit known-cost protocol fixture, not a copy of the real scorer."""
    path = tmp_path / "cost.py"
    path.write_text(
        "import os\nimport time\n"
        "def reference_from_question(question):\n"
        "    cases = [(c['args'], c['expected']) for c in question['cases']]\n"
        "    return question['function'], cases\n"
        "def response_cost(code, function, cases):\n"
        "    assert function == 'solve'\n"
        "    assert cases == [([1], 2), ([2], 3)]\n"
        "    if code == 'WAIT': time.sleep(60)\n"
        "    if code == 'CRASH': raise RuntimeError('evaluator failure')\n"
        "    if code == 'EXIT': os._exit(0)\n"
        "    if code == 'BOOLEAN': return True\n"
        f"    return 0 if code == {GOOD!r} else 2\n"
    )
    return path


@pytest.fixture
def original_evaluator():
    """Uses the explicitly supplied existing scorer without copying or changing it."""
    path = os.environ.get("FORBIDDEN_WORD_EVALUATOR")
    if not path:
        pytest.skip("Set FORBIDDEN_WORD_EVALUATOR to run original-cost parity checks.")
    return Path(path).resolve(strict=True)


def test_raw_completion_and_references_reach_existing_protocol(evaluator):
    for source, cost in [(GOOD, 0), (BAD, 2), ("```python\n" + GOOD + "```", 2)]:
        result = grading.grade_completion(source, task(), evaluator, 5)
        assert result == {
            "status": "graded",
            "cost": cost,
            "reward": -float(cost),
            "success": cost == 0,
        }


@pytest.mark.parametrize(
    "source,status,message",
    [
        ("WAIT", "timeout", "exceeded"),
        ("CRASH", "worker_error", "evaluator failure"),
        ("EXIT", "worker_error", "without publishing"),
        ("BOOLEAN", "worker_error", "nonnegative integer"),
    ],
)
def test_worker_failures_never_become_numeric_rewards(evaluator, source, status, message):
    result = grading.grade_completion(source, task(), evaluator, 0.2 if source == "WAIT" else 5)
    assert result["status"] == status
    assert result["cost"] is result["reward"] is None
    assert result["success"] is False
    assert message in result["error"]
    with pytest.raises(RuntimeError, match="Grading failed"):
        runner.require_score({"task_id": "test", **result})


@pytest.mark.parametrize(
    "source,cost",
    [
        (GOOD, 0),
        (BAD, 3),
        ("# while while\n" + GOOD, 1),
        ("# meanwhile\n" + GOOD, 1),
        ("# WHILE\n" + GOOD, 0),
        ("marker = 'while'\n" + GOOD, 1),
        ("```python\n" + GOOD + "```", 1),
        (GOOD.split("def test_solve")[0], 1),
        (GOOD.replace("assert solve(1) == 2", "assert False"), 1),
    ],
)
def test_original_cost_is_unchanged_in_real_subprocess(original_evaluator, source, cost):
    original = grading.load_evaluator(original_evaluator)
    assert original.response_cost(source, *original.reference_from_question(task())) == cost
    result = grading.grade_completion(source, task(), original_evaluator, 5)
    assert result["status"] == "graded"
    assert result["cost"] == cost
    assert result["reward"] == -float(cost)


def test_original_json_equality_and_pytest_failure_are_preserved(original_evaluator):
    question = {**task(), "cases": [{"args": [1], "expected": 1}]}
    source = "def solve(value):\n    return True\ndef test_solve():\n    assert True\n"
    assert grading.grade_completion(source, question, original_evaluator, 5)["cost"] == 0
    pytest.importorskip("pytest")
    source = GOOD + "import pytest\ndef test_failure():\n    pytest.fail('expected')\n"
    result = grading.grade_completion(source, task(), original_evaluator, 5)
    assert result["status"] == "graded"
    assert result["cost"] == 1


def test_original_infinite_response_is_an_explicit_timeout(original_evaluator):
    source = "while True:\n    pass\n"
    result = grading.grade_completion(source, task(), original_evaluator, 0.2)
    assert result["status"] == "timeout"
    assert result["cost"] is result["reward"] is None


def test_loader_preserves_prompts_and_distinguishes_prefix_variants(tmp_path, evaluator):
    questions = [
        {**task(), "source_dataset": "forbidden-word-avoidance", "prefix_index": prefix}
        for prefix in (0, 1)
    ]
    path = tmp_path / "train.json"
    path.write_text("".join(json.dumps(row) + "\n" for row in questions))
    scorer = grading.load_evaluator(evaluator)
    loaded = runner.load_tasks(path, scorer)
    assert len({row["task_id"] for row in loaded}) == 2
    for actual, expected in zip(loaded, questions):
        assert {k: v for k, v in actual.items() if k != "task_id"} == expected
    path.write_text(json.dumps(task()) + "\n" + json.dumps(task()) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        runner.load_tasks(path, scorer, limit=1)


def test_dry_run_validates_without_model_or_output(tmp_path, evaluator, capsys):
    dataset = tmp_path / "train.json"
    dataset.write_text(json.dumps(task()) + "\n")
    output = tmp_path / "run"
    common = [
        "train",
        "--dataset",
        str(dataset),
        "--evaluator",
        str(evaluator),
        "--output",
        str(output),
        "--model",
        str(tmp_path / "nonexistent-model"),
    ]
    runner.main([*common, "--dry-run"])
    config = json.loads(capsys.readouterr().out)
    assert config["tasks"] == 1
    assert config["evaluator_sha256"] == hashlib.sha256(evaluator.read_bytes()).hexdigest()
    assert not output.exists()
    with pytest.raises(SystemExit):
        runner.main(common)
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "adapter_config.json").write_text("{}")
    with pytest.raises(SystemExit):
        runner.main([*common, "--dry-run", "--adapter", str(incomplete)])
    for flag, value in [("--timeout", "nan"), ("--max-tokens", "0"), ("--group-size", "1")]:
        with pytest.raises(SystemExit):
            runner.main([*common, "--dry-run", flag, value])


def test_evaluation_metrics_use_zero_cost_not_only_correctness():
    records = [
        {"task_id": "one", "status": "graded", "cost": cost, "reward": -float(cost)}
        for cost in (0, 1, 2)
    ]
    metrics = runner.evaluation_metrics(records, [1, 2])
    assert metrics["mean_cost"] == 1
    assert metrics["zero_cost/pass@1"] == pytest.approx(1 / 3)
    assert metrics["zero_cost/pass@2"] == pytest.approx(2 / 3)
    records[0]["status"] = "timeout"
    records[0]["error"] = "timeout"
    with pytest.raises(RuntimeError, match="timeout"):
        runner.evaluation_metrics(records, [1])


def test_stabilized_objective_keeps_token_baseline_and_fixed_horizon():
    torch = pytest.importorskip("torch")
    objective = importlib.import_module(f"{PACKAGE}.objective")
    centered = objective.advantages(
        torch.tensor([0.0, -1.0, -3.0]), torch.tensor([2.0, 5.0, 1.0])
    )
    torch.testing.assert_close(centered, torch.tensor([1.0, 0.0, -2.0]))
    current = torch.full((2,), 2.0).log().requires_grad_()
    old = torch.zeros(2, requires_grad=True)
    loss = objective.clipped_loss(current, old, torch.tensor(1.0), horizon=8)
    torch.testing.assert_close(loss, torch.tensor(-0.3))
    loss.backward()
    assert old.grad is None


def test_equal_cost_groups_skip_optimizer_updates(tmp_path, evaluator):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    evaluator.write_text(
        evaluator.read_text() + "\ndef response_cost(code, function, cases):\n    return 2\n"
    )
    model = tmp_path / "model"
    make_tiny_policy(model)
    dataset = tmp_path / "train.json"
    dataset.write_text(json.dumps(task()) + "\n")
    output = tmp_path / "train"
    runner.main(
        [
            "train",
            "--dataset",
            str(dataset),
            "--evaluator",
            str(evaluator),
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
            "--horizon",
            "8",
            "--checkpoint-steps",
            "1",
            "--allow-code-execution",
        ]
    )
    assert json.loads((output / "training-metrics.json").read_text())["optimizer_updates"] == 0
    assert not list(output.glob("checkpoint-*"))
    weights = load_file(output / "adapter" / "adapter_model.safetensors")
    assert all(tensor.count_nonzero() == 0 for key, tensor in weights.items() if "lora_B" in key)


def make_tiny_policy(path, model_type="qwen3"):
    """Creates a genuine local policy with complete Python sources as single tokens."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    vocabulary = {"[UNK]": 0, "[EOS]": 1, "[PAD]": 2, GOOD: 3, BAD: 4}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]"
    )
    tokenizer.save_pretrained(path)
    settings = {
        "vocab_size": 5,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "max_position_embeddings": 512,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "attention_dropout": 0.4,
    }
    torch.manual_seed(42)
    if model_type == "qwen3":
        model = Qwen3ForCausalLM(Qwen3Config(**settings))
    else:
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

        model = Qwen3_5ForConditionalGeneration(
            Qwen3_5Config(
                text_config={
                    **settings,
                    "num_hidden_layers": 2,
                    "layer_types": ["full_attention", "full_attention"],
                    "rope_parameters": {
                        "rope_type": "default",
                        "rope_theta": 10000.0,
                        "partial_rotary_factor": 1.0,
                        "mrope_section": [1, 1, 2],
                    },
                },
                vision_config={
                    "depth": 1,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_heads": 2,
                    "out_hidden_size": 16,
                    "patch_size": 2,
                    "temporal_patch_size": 2,
                    "spatial_merge_size": 1,
                    "num_position_embeddings": 4,
                },
                image_token_id=10,
                video_token_id=11,
                vision_start_token_id=12,
                vision_end_token_id=13,
            )
        )
    model.save_pretrained(path)
    return tokenizer


@pytest.mark.parametrize("grader_name", ["evaluator", "original_evaluator"])
def test_offline_training_checkpoint_and_base_adapter_evaluation(tmp_path, request, grader_name):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from safetensors.torch import load_file

    evaluator = request.getfixturevalue(grader_name)
    torch.set_num_threads(1)
    model = tmp_path / "model"
    make_tiny_policy(model)
    dataset = tmp_path / "train.json"
    dataset.write_text(json.dumps(task()) + "\n")
    common = [
        "--dataset",
        str(dataset),
        "--evaluator",
        str(evaluator),
        "--model",
        str(model),
        "--device",
        "cpu",
        "--max-tokens",
        "1",
        "--horizon",
        "8",
        "--samples",
        "2",
        "--pass-k",
        "1",
        "--allow-code-execution",
    ]
    train = tmp_path / "train"
    runner.main(
        ["train", *common, "--output", str(train), "--steps", "1", "--checkpoint-steps", "1"]
    )
    metrics = json.loads((train / "training-metrics.json").read_text())
    assert metrics["optimizer_updates"] == 1
    records = runner.read_jsonl(train / "train-rank0.jsonl")
    assert len(records) == 5
    assert all(row["reward"] == -float(row["cost"]) for row in records)
    assert len({row["cost"] for row in records}) > 1
    assert (train / "evaluator.py").read_bytes() == evaluator.read_bytes()
    assert (train / "checkpoint-1" / "adapter_config.json").is_file()
    adapter = load_file(train / "adapter" / "adapter_model.safetensors")
    assert any(
        "lora_B" in key and tensor.abs().max().item() > 0 for key, tensor in adapter.items()
    )
    for name, options in [("base", []), ("adapted", ["--adapter", str(train / "adapter")])]:
        output = tmp_path / name
        runner.main(["evaluate", *common, "--output", str(output), *options])
        assert json.loads((output / "metrics.json").read_text())["samples"] == 2
        assert len(runner.read_jsonl(output / "evaluate.jsonl")) == 2
        with pytest.raises(FileExistsError):
            runner.main(["evaluate", *common, "--output", str(output), *options])


@pytest.mark.parametrize("model_type", ["qwen3", "qwen3_5"])
def test_policy_keeps_raw_prompt_and_recomputes_sample_probabilities(tmp_path, model_type):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from types import SimpleNamespace

    torch.set_num_threads(1)
    policy = importlib.import_module(f"{PACKAGE}.policy")
    objective = importlib.import_module(f"{PACKAGE}.objective")
    make_tiny_policy(tmp_path / "model", model_type)
    model, tokenizer = policy.load_policy(str(tmp_path / "model"), None, "cpu", train=True)
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        "{% if enable_thinking %}THINK{% endif %}"
    )
    prompt = policy.prompt_tokens(tokenizer, {**task(), "prompt": "[UNK]"}, "cpu")
    assert (
        prompt["input_ids"].tolist()
        == tokenizer("[UNK]", return_tensors="pt", add_special_tokens=False)["input_ids"].tolist()
    )
    assert model.training and model.is_gradient_checkpointing
    config = getattr(model.config, "text_config", model.config)
    assert config.attention_dropout == 0.0
    model.generation_config.repetition_penalty = 2.0

    class ScoredPolicy(torch.nn.Module):
        """Records the real generation scores without changing sampling."""

        def __init__(self, base):
            super().__init__()
            self.base = base
            self.generated = None

        def generate(self, **kwargs):
            self.generated = self.base.generate(
                **kwargs, return_dict_in_generate=True, output_scores=True
            )
            return self.generated.sequences

    scored = ScoredPolicy(model)
    tokens, text = policy.sample(
        scored, tokenizer, prompt, SimpleNamespace(temperature=1.0, max_tokens=1)
    )
    assert model.training
    assert text == tokenizer.decode(
        tokens[0, prompt["input_ids"].shape[1] :], skip_special_tokens=True
    )
    old = objective.log_probs(model, tokens, prompt["input_ids"].shape[1], 1.0)
    sampled_log_probability = scored.generated.scores[0].log_softmax(-1)[0, tokens[0, -1]]
    torch.testing.assert_close(old[0, 0], sampled_log_probability, rtol=1e-5, atol=1e-6)
    current = objective.log_probs(model, tokens, prompt["input_ids"].shape[1], 1.0)
    torch.testing.assert_close(current, old, rtol=0, atol=0)
    objective.clipped_loss(current, old, torch.tensor(1.0), horizon=8).backward()
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.parameters())


def test_original_dataset_dry_run(tmp_path, original_evaluator):
    directory = os.environ.get("FORBIDDEN_WORD_DATASET")
    if not directory:
        pytest.skip("Set FORBIDDEN_WORD_DATASET to validate the supplied external data.")
    evaluator = grading.load_evaluator(original_evaluator)
    root = Path(directory)
    for name, count in [
        ("train.json", 100),
        ("test.json", 100),
        ("prefixed-mixed-cost-training/train.json", 1664),
    ]:
        tasks = runner.load_tasks(root / name, evaluator)
        assert len(tasks) == count
        assert len({row["task_id"] for row in tasks}) == count


def test_cli_help_needs_no_model_or_evaluator():
    cli = Path(__file__).resolve().parents[2] / "papers" / "while-without-while-2026" / "run.py"
    result = subprocess.run(
        [sys.executable, str(cli), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=False,
    )
    assert result.returncode == 0
    assert "--evaluator" in result.stdout
