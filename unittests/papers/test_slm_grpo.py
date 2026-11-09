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


def make_export_repository(path):
    """Creates an explicit lightweight converter with real ONNX runtime behavior."""
    from locodellm.generate.repository import GitRepository

    path.mkdir(parents=True)
    (path / ".gitignore").write_text("__pycache__/\n")
    converter = path / "yobx" / "torch" / "interpreter"
    converter.mkdir(parents=True)
    (path / "yobx" / "__init__.py").write_text("")
    (path / "yobx" / "torch" / "__init__.py").write_text("")
    (converter / "__init__.py").write_text("""import onnx

def aten_fixture(module, inputs, input_names):
    if module.mode == "export":
        raise ValueError("A reproducible converter failure.")
    if module.mode == "upstream":
        return None
    names = input_names or ["x"]
    shape = list(inputs[0].shape)
    output_shape = [99] if module.mode == "metadata" else shape
    value = onnx.helper.make_tensor("bias", onnx.TensorProto.FLOAT, [1],
                                    [1.0 if module.mode == "comparison" else 0.0])
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Add", [names[0], "bias"], ["output"])],
        "fixture",
        [onnx.helper.make_tensor_value_info(names[0], onnx.TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, output_shape)],
        [value])
    return onnx.helper.make_model_gen_version(
        graph, opset_imports=[onnx.helper.make_opsetid("", 18)])

def to_onnx(module, inputs, *, target_opset=18, optimize=True, input_names=None):
    if module.mode == "upstream":
        raise ValueError("Upstream constraint error, not a converter failure.")
    return aten_fixture(module, inputs, input_names)
""")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "Creates an explicit test converter",
        ],
        check=True,
    )
    return GitRepository(str(path))


def export_answer(mode="comparison", *, invalid_reference=False, invalid_api=False):
    """Returns a complete, controlled export example for verifier tests."""
    return (
        "```python\n"
        "import torch\n"
        "from yobx.torch.interpreter import to_onnx\n"
        "class Model(torch.nn.Module):\n"
        f"    mode = {mode!r}\n"
        "    def forward(self, x):\n"
        + (
            "        raise ValueError('invalid reference')\n"
            if invalid_reference
            else "        return x\n"
        )
        + "to_onnx(Model(), (torch.ones(2),), target_opset=18, "
        f"optimize={invalid_api!r}, input_names=['x'])\n"
        "```"
    )


@pytest.mark.parametrize(
    "mode,invalid_reference,invalid_api,expected",
    [
        ("comparison", False, False, (1.0, True, "comparison")),
        ("export", False, False, (1.0, True, "export")),
        ("metadata", False, False, (1.0, True, "metadata")),
        ("passed", False, False, (0.5, False, "comparison")),
        ("upstream", False, False, (0.5, False, "export")),
        ("comparison", True, False, (0.05, False, "pytorch")),
        ("comparison", False, True, (0.25, False, "export_call")),
        ("unrelated", False, False, (0.5, False, "comparison")),
        ("modified", False, False, (0.05, False, "comparison")),
    ],
)
def test_factual_export_replays(tmp_path, mode, invalid_reference, invalid_api, expected):
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    from locodellm.training.export_verifier import verify_export_answer

    source = make_export_repository(tmp_path / "source")
    answer = export_answer(
        "comparison" if mode in ("unrelated", "modified") else mode,
        invalid_reference=invalid_reference,
        invalid_api=invalid_api,
    )
    if mode == "modified":
        answer = answer.replace(
            "import torch\n",
            "import torch\n"
            "from pathlib import Path\n"
            "Path('forbidden_source.py').write_text('pass')\n",
        )
    result = verify_export_answer(
        answer,
        source,
        tmp_path / "clones",
        allow_execution=True,
        timeout=60,
        allowed_converters={"aten_other" if mode == "unrelated" else "aten_fixture"},
    )
    assert result.total == expected[0]
    assert result.criteria["failure_valid"] == expected[1]
    assert result.criteria["reproduction_valid"] == expected[1]
    assert result.replays[0]["observations"][0]["stage"] == expected[2]
    assert result.replays[0]["repository"] != result.replays[1]["repository"]
    assert all(replay["revision"] == source.revision for replay in result.replays)
    assert not source.diff()


def test_export_extraction_opt_in_and_timeout(tmp_path):
    from locodellm.training.export_verifier import extract_example, verify_export_answer

    assert not extract_example("<bash>echo not a final example</bash>")
    assert not extract_example("```python\ndef broken(:\n```")
    assert extract_example("<think>```python\nwrong\n```</think>\n```py\npass\n```") == "pass"
    source = make_export_repository(tmp_path / "source")
    with pytest.raises(PermissionError, match="allow-code-execution"):
        verify_export_answer("", source, tmp_path / "clones", allow_execution=False)
    result = verify_export_answer("", source, tmp_path / "clones", allow_execution=True)
    assert result.total == 0
    assert not result.replays
    result = verify_export_answer(
        export_answer(), source, tmp_path / "clones", allow_execution=True, timeout=0.01
    )
    assert result.total == 0.05
    assert all(replay["timed_out"] for replay in result.replays)


@pytest.fixture
def saved_export_batch(tmp_path):
    """Creates saved generations with a failure, missing code and rejected rollouts."""
    source = make_export_repository(tmp_path / "source")
    tasks = [
        {
            "id": f"task-{index}",
            "function": "aten_fixture",
            "prompt": f"Investigate export case {index}.",
            "repository": str(source.path),
            "verification": {"revision": source.revision},
        }
        for index in range(5)
    ]
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text("".join(json.dumps(task) + "\n" for task in tasks))
    rows = [
        {
            "prompt": task["prompt"],
            "text": export_answer() if index != 1 else "All tests passed, trust me.",
            "tool_call_count": 0 if index == 1 else 1,
            "error": "Final answer refused" if index == 2 else None,
            "diff": "Source changed" if index == 3 else "",
        }
        for index, task in enumerate(tasks[:4])
    ]
    results = tmp_path / "events.json"
    results.write_text(
        json.dumps({"repo": str(source.path), "base_revision": source.revision, "results": rows})
    )
    return source, dataset, results


def test_export_evaluation_cli_report(saved_export_batch, tmp_path, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    from locodellm.__main__ import main

    source, dataset, results = saved_export_batch
    output = tmp_path / "evaluation.json"
    main(
        [
            "evaluate",
            "--input",
            str(dataset),
            "--results",
            str(results),
            "--output",
            str(output),
            "--allow-code-execution",
        ]
    )
    report = json.loads(output.read_text())
    assert report["summary"]["mean_reward"] == 0.25
    assert report["summary"]["missing_tasks"] == 1
    assert report["summary"]["generation_tool_calls"] == 3
    assert report["summary"]["independent_replays"] == 2
    assert [row["total"] for row in report["results"]] == [1, 0, 0, 0]
    assert report["results"][0]["criteria"]["reproduction_valid"]
    assert not report["results"][1]["criteria"]["answer_extractable"]
    assert all(not row["replays"] for row in report["results"][1:])
    stdout = capsys.readouterr().out
    assert "Mean reward: 0.2500 / 1" in stdout
    assert "Evaluated: 4/5 tasks" in stdout
    assert "Confirmed converter failures: 1/4" in stdout
    assert "WARNING: 1 missing tasks" in stdout
    assert "No extractable runnable Python" in stdout
    assert "Final answer refused" in stdout
    assert "Repository modified during generation" in stdout
    assert f"Report: {output}" in stdout
    assert not source.diff()


@pytest.mark.parametrize("mode", ["saved", "stdout", "disabled"])
def test_generate_automatically_evaluates_aten(saved_export_batch, mode, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    from locodellm.__main__ import main
    from locodellm.generate.generate_from_model import _session_cache
    from unittests.bench.test_python_harness import ScriptedSession

    class BatchSession(ScriptedSession):
        def new_session(self):
            """Creates a model failure followed by a complete export reproducer."""
            child = ScriptedSession([""] if not self.children else [export_answer()])
            self.children.append(child)
            return child

    source, dataset, results = saved_export_batch
    tasks = [json.loads(line) for line in dataset.read_text().splitlines()][:2]
    dataset.write_text("".join(json.dumps(task) + "\n" for task in tasks))
    model_id = f"test/automatic-aten-evaluation-{mode}"
    key = (model_id, None, None, (), "{}")
    assert key not in _session_cache
    _session_cache[key] = BatchSession([])
    output = results.with_suffix(".evaluation.json")
    options = [
        "generate",
        model_id,
        "--input",
        str(dataset),
        "--repo",
        str(source.path),
        "--harness",
        "bash",
    ]
    if mode == "saved":
        options.extend(["--output", str(results)])
    elif mode == "disabled":
        options.extend(["--no-evaluate", "--output", str(results)])
    try:
        main(options)
    finally:
        del _session_cache[key]
    stdout = capsys.readouterr().out
    if mode == "disabled":
        assert "ATen export evaluation" not in stdout
        assert not output.exists()
    else:
        assert "ATen export evaluation" in stdout
        assert "Mean reward: 0.5000 / 1" in stdout
        assert "Evaluated: 2/2 tasks" in stdout
        assert "Confirmed converter failures: 1/2" in stdout
        assert "The model returned an empty answer" in stdout
        if mode == "saved":
            import hashlib

            report = json.loads(output.read_text())
            assert report["generation_sha256"] == hashlib.sha256(results.read_bytes()).hexdigest()
            assert report["summary"]["mean_reward"] == 0.5
            assert f"Report: {output}" in stdout
        else:
            assert not output.exists()
            assert "Report:" not in stdout
    assert not source.diff()


def test_generate_aten_validation_before_loading(saved_export_batch, capsys):
    from locodellm.__main__ import main

    source, dataset, _ = saved_export_batch
    tasks = [json.loads(line) for line in dataset.read_text().splitlines()]
    for task in tasks:
        task["verification"]["revision"] = "0" * 40
    dataset.write_text("".join(json.dumps(task) + "\n" for task in tasks))
    common = ["generate", "unused/model", "--input", str(dataset), "--repo", str(source.path)]
    with pytest.raises(ValueError, match="recorded revision"):
        main(common)
    for timeout in ("0", "nan", "inf"):
        with pytest.raises(SystemExit) as error:
            main([*common, "--verification-timeout", timeout])
        assert error.value.code == 2
    assert "loading model" not in capsys.readouterr().err


def test_export_evaluation_execution_opt_in_and_paths(saved_export_batch, tmp_path):
    from locodellm.bench.export_evaluation import evaluate

    source, dataset, results = saved_export_batch
    with pytest.raises(PermissionError, match="allow-code-execution"):
        evaluate(dataset, results, allow_execution=False)
    for timeout in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and positive"):
            evaluate(dataset, results, allow_execution=True, timeout=timeout)
    for output in (dataset, results, source.path / "evaluation.json"):
        with pytest.raises(ValueError, match="output must"):
            evaluate(dataset, results, allow_execution=True, output=output)
    assert not (tmp_path / "evaluation.json").exists()


@pytest.mark.parametrize("problem", ["revision", "unknown", "duplicate", "empty", "count"])
def test_export_evaluation_rejects_mismatched_results(saved_export_batch, problem):
    from locodellm.bench.export_evaluation import evaluate

    _, dataset, results = saved_export_batch
    document = json.loads(results.read_text())
    if problem == "revision":
        document["base_revision"] = "0" * 40
    elif problem == "unknown":
        document["results"][0]["prompt"] = "A prompt from another run."
    elif problem == "duplicate":
        document["results"].append(document["results"][0])
    elif problem == "empty":
        document["results"] = []
    else:
        document["results"][0]["tool_call_count"] = -1
    results.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        evaluate(dataset, results, allow_execution=True)


def test_multiturn_token_weighted_gradient(tmp_path):
    torch = pytest.importorskip("torch")
    from locodellm.training.grpo import Turn, trajectory_backward
    from locodellm.training.optimization import policy_loss, trajectory_log_probs

    torch.set_num_threads(1)
    model, _ = make_tiny_policy(tmp_path / "model")
    model.eval()
    tokens = [(torch.tensor([[3, 4, 5]]), 1), (torch.tensor([[3, 4, 6, 7, 3, 5]]), 2)]
    current = trajectory_log_probs(model, tokens, temperature=0.7)
    # Unequal turn lengths and ratios distinguish token averaging from turn averaging.
    shifts = torch.tensor([[0.01, -0.02, 0.03, -0.04, 0.05, -0.06]])
    old = current.detach() - shifts
    expected_loss = (
        policy_loss(
            current, old, torch.ones_like(current, dtype=torch.bool), torch.tensor([0.75])
        )
        / 2
    )
    expected_loss.backward()
    expected_gradients = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    assert any(value.abs().sum() > 0 for value in expected_gradients.values())
    model.zero_grad()
    turns = [Turn(tokens[0][0], 1, old[:, :2]), Turn(tokens[1][0], 2, old[:, 2:])]
    actual_loss = trajectory_backward(
        model, turns, torch.tensor(0.75), temperature=0.7, group_size=2
    )
    assert actual_loss == pytest.approx(expected_loss.item())
    for name, parameter in model.named_parameters():
        if name in expected_gradients:
            torch.testing.assert_close(parameter.grad, expected_gradients[name])


def test_training_session_retains_actions_and_eos(tmp_path):
    torch = pytest.importorskip("torch")
    from locodellm.training.grpo import TrainingSession

    torch.set_num_threads(1)
    model, tokenizer = make_tiny_policy(tmp_path / "model")
    model.eval()
    session = TrainingSession(model, tokenizer, temperature=0.9)
    session.generate("Returns one.", max_new_tokens=1, do_sample=False, forced_eos_token_id=1)
    assert session.text == ""
    assert session.turns[0].tokens[0, -1].item() == tokenizer.eos_token_id
    assert session.turns[0].old_log_probs.shape == (1, 1)
    session.generate(
        "Tool result: 12345", max_new_tokens=2, do_sample=False, forced_eos_token_id=1
    )
    assert len(session.turns) == 2
    second = session.turns[1]
    assert second.prompt_length > session.turns[0].prompt_length
    assert second.old_log_probs.numel() == second.tokens.shape[1] - second.prompt_length
    assert not second.old_log_probs.requires_grad
    assert second.tokens[0, second.prompt_length :].tolist() == session.last_generated_tokens


def test_training_session_with_real_tool_feedback(tmp_path):
    torch = pytest.importorskip("torch")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from locodellm.bench.python_harness import PythonHarness
    from locodellm.training.grpo import TrainingSession

    torch.set_num_threads(1)
    model, _ = make_tiny_policy(tmp_path / "model")
    vocabulary = {"[UNK]": 0, "[EOS]": 1, "[PAD]": 2}
    vocabulary.update({chr(value): index + 3 for index, value in enumerate(range(9, 127))})
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]"
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    model.resize_token_embeddings(len(vocabulary))
    model.eval()
    actions = ["<bash>printf 7</bash>", "No failure confirmed."]

    class GuidedSession(TrainingSession):
        def generate(self, prompt, *args, **kwargs):
            start = self.get_prompt_length(prompt)
            tokens = [*tokenizer.encode(actions[len(self.turns)]), tokenizer.eos_token_id]
            kwargs["prefix_allowed_tokens_fn"] = lambda batch, ids: [tokens[len(ids) - start]]
            return super().generate(prompt, *args, **kwargs)

    session = GuidedSession(model, tokenizer, temperature=0.9)
    result = PythonHarness(max_tool_calls=1, bash_only=True).run(
        session, "Find an export failure.", max_length=128, do_sample=False
    )
    assert result.error is None
    assert result.text == actions[1]
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].stdout == "7"
    assert len(session.turns) == 2
    assert "7" in session.messages[2]["content"]
    for turn, action in zip(session.turns, actions):
        assert turn.tokens[0, turn.prompt_length :].tolist() == [
            *tokenizer.encode(action),
            tokenizer.eos_token_id,
        ]
        assert turn.old_log_probs.numel() == len(tokenizer.encode(action)) + 1


def test_aten_training_cli_offline(tmp_path, capsys):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from locodellm.__main__ import main
    from locodellm.training.grpo import load_tasks

    torch.set_num_threads(1)
    source = make_export_repository(tmp_path / "source")
    model_path = tmp_path / "model"
    make_tiny_policy(model_path)
    dataset = tmp_path / "tasks.jsonl"
    task = {
        "id": "test-export",
        "function": "aten_fixture",
        "prompt": "Find a module export failure.",
        "repository": str(source.path),
        "verification": {"revision": source.revision},
        "expected_answer": {"secret": "REFERENCE MUST NOT LEAK"},
    }
    dataset.write_text(json.dumps(task) + "\n")
    assert load_tasks(str(dataset)) == [task]
    output = tmp_path / "train"
    common = [
        "train",
        str(model_path),
        "--input",
        str(dataset),
        "--output",
        str(output),
        "--repo",
        str(source.path),
        "--device",
        "cpu",
        "--epochs",
        "1",
        "--group-size",
        "2",
        "--max-new-tokens",
        "2",
        "--max-tool-calls",
        "1",
        "--updates-per-group",
        "2",
        "--checkpoint-steps",
        "1",
    ]
    with pytest.raises(PermissionError, match="allow-code-execution"):
        main(common)
    assert not output.exists()
    main([*common, "--allow-code-execution"])
    assert (output / "adapter" / "adapter_model.safetensors").exists()
    assert (output / "checkpoint-1" / "adapter_config.json").exists()
    assert (output / "checkpoint-2" / "adapter_config.json").exists()
    records = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert len({record["repository"] for record in records}) == 2
    assert all(record["turns"] for record in records)
    assert "REFERENCE MUST NOT LEAK" not in (output / "train.jsonl").read_text()
    assert json.loads((output / "config.json").read_text())["reward"] == "export_failure_reward"
    assert "update=2" in capsys.readouterr().out
    with pytest.raises(ValueError, match="empty or absent"):
        main([*common, "--allow-code-execution"])
    task["verification"]["revision"] = "wrong"
    dataset.write_text(json.dumps(task) + "\n")
    with pytest.raises(ValueError, match="full verification SHA"):
        load_tasks(str(dataset))


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
            "--updates-per-group",
            "2",
            "--checkpoint-minutes",
            "0.000001",
        ]
    )
    stdout = capsys.readouterr().out
    assert f"device={device} algorithm={algorithm} reward=ruff" in stdout
    assert "[train] Loaded 1 tasks." in stdout
    assert "[train]:" in stdout
    assert "100%" in stdout
    assert "epoch=1/1" in stdout
    assert "task=1/1" in stdout
    assert "sample=1/2" in stdout
    assert "sample=2/2" in stdout
    assert "2/2" in stdout
    assert "update=2/2" in stdout
    assert "loss=" in stdout
    assert "grad_norm=" in stdout
    assert "mean_reward=" in stdout
    assert "[train] epoch=" not in stdout
    assert f"Results: {train_path / 'train.jsonl'}" in stdout
    assert (train_path / "adapter" / "adapter_config.json").exists()
    assert sorted(path.name for path in train_path.glob("checkpoint-*")) == [
        "checkpoint-1",
        "checkpoint-2",
    ]
    for update in (1, 2):
        checkpoint = train_path / f"checkpoint-{update}"
        assert (checkpoint / "adapter_model.safetensors").exists()
        assert (checkpoint / "tokenizer_config.json").exists()
        assert json.loads((checkpoint / "checkpoint.json").read_text()) == {
            "optimizer_step": update,
            "epoch": 1,
            "task_index": 1,
            "task_id": 1,
            "update_within_group": update,
        }
        assert f"[train] Saved checkpoint: {checkpoint}" in stdout
    records = [json.loads(line) for line in (train_path / "train.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert all(record["execution"] is None for record in records)
    eval_path = tmp_path / "evaluate"
    experiment.main(
        [
            "evaluate",
            *common,
            "--adapter",
            str(train_path / "checkpoint-2"),
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


def test_step_checkpoint_boundary(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    torch.set_num_threads(1)
    model_path = tmp_path / "model"
    make_tiny_policy(model_path)
    tasks = tmp_path / "tasks.json"
    tasks.write_text(
        json.dumps([{"text": "Returns one.", "test_list": ["assert solution() == 1"]}])
    )
    output = tmp_path / "train"
    experiment.main(
        [
            "train",
            "--model",
            str(model_path),
            "--dataset",
            str(tasks),
            "--device",
            "cpu",
            "--reward",
            "ruff",
            "--epochs",
            "1",
            "--group-size",
            "2",
            "--max-tokens",
            "1",
            "--updates-per-group",
            "51",
            "--checkpoint-steps",
            "50",
            "--output",
            str(output),
        ]
    )
    assert [path.name for path in output.glob("checkpoint-*")] == ["checkpoint-50"]
    state = json.loads((output / "checkpoint-50" / "checkpoint.json").read_text())
    assert state["optimizer_step"] == 50
    assert state["update_within_group"] == 50
    assert (output / "adapter" / "adapter_model.safetensors").exists()


def test_incomplete_checkpoint_is_not_published(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    model_path = tmp_path / "model"
    make_tiny_policy(model_path)
    model, tokenizer = experiment.load_policy(str(model_path), None, train=True, device="cpu")
    previous = tmp_path / "checkpoint-1"
    experiment.save_adapter(model, tokenizer, previous, checkpoint_state={"optimizer_step": 1})
    original = {file.name: file.read_bytes() for file in previous.iterdir()}

    class FailingTokenizer:
        def save_pretrained(self, path):
            raise OSError("Tokenizer write failed.")

    failed = tmp_path / "checkpoint-2"
    with pytest.raises(OSError, match="Tokenizer write failed"):
        experiment.save_adapter(
            model, FailingTokenizer(), failed, checkpoint_state={"optimizer_step": 2}
        )
    assert not failed.exists()
    assert not list(tmp_path.glob(".checkpoint-*"))
    assert {file.name: file.read_bytes() for file in previous.iterdir()} == original
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        experiment.save_adapter(model, tokenizer, previous)


@pytest.mark.parametrize(
    "option,value",
    [
        ("--checkpoint-minutes", "0"),
        ("--checkpoint-minutes", "nan"),
        ("--checkpoint-minutes", "inf"),
        ("--checkpoint-steps", "0"),
    ],
)
def test_checkpoint_options_reject_invalid_values(tmp_path, option, value):
    with pytest.raises(SystemExit) as exc:
        experiment.main(
            ["train", "--reward", "ruff", "--output", str(tmp_path / "train"), option, value]
        )
    assert exc.value.code == 2
    assert not (tmp_path / "train").exists()


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
