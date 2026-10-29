# RLVR for small code language models (2026)

An independent implementation of **Improving Small Language Models for Code
Generation with Reinforcement Learning from Verification Feedback**, E. Kotelnikov.

Paper: [arXiv:2605.30478](https://arxiv.org/abs/2605.30478)
([PDF](https://arxiv.org/pdf/2605.30478)).

## Paper summary

The paper studies reinforcement learning with verifiable rewards (RLVR) for Python
code generation on MBPP, using Qwen3-0.6B and Llama-3.2-1B-Instruct with frozen base
weights and LoRA adapters. It compares token-level GRPO with sequence-level GSPO,
and unit-test, Ruff-only, and combined rewards. Unit-test outcomes provide a graded
correctness signal; Ruff findings supply severity-weighted penalties.

The combined reward normalizes lint penalties by program length, clips them, and
weights them more strongly for passing solutions. A short-output penalty discourages
empty completions. The authors report up to **13 percentage points** improvement in
MBPP pass@1, but no universal winner between GRPO and GSPO. Ruff-only optimization
can favor short, incorrect programs. EvalPlus evaluation also shows that combined
rewards do not consistently beat unit-test-only rewards. Generation length,
execution errors, and lint severity therefore matter alongside pass@1.

## What is implemented

This directory is standalone: it does not change the ONNX inference API.

| Component | Implementation |
|---|---|
| Prompts | MBPP description and **only the first** example test; no reference solution |
| Verification | All standard and challenge tests, plus MBPP setup code |
| Unit reward | Passed `+1`; assertion failure or timeout `-0.3`; runtime error `-0.6`; interpreter/other error `-1` |
| Ruff reward | Severity penalties: error `-10`, high `-5`, normal `-1`, low `-0.2`; raw and line-normalized variants |
| Combined reward | `unit + weight * clip(raw / max(1, lines), -1, 0) + short`; weight `0.3` if passed, otherwise `0.1` |
| Short penalty | `-0.5` if fewer than 30 characters **or** fewer than 3 lines |
| Advantages | Per-prompt centered rewards; optional population-SD normalization |
| Optimization | Clipped token-level GRPO and sequence-level GSPO; frozen old log probabilities; no reference model (`beta=0`) |
| Training | LoRA rank 32, alpha 64; all seven attention/MLP projections; AdamW |
| Evaluation | One sample per task, pass@1, original exception types, mean tokens/lines, severity counts, Ruff failures |
| Artifacts | Saved adapters/tokenizer, configuration JSON, per-completion JSONL diagnostics, evaluation metrics |

Defaults follow section 4.3: eight completions per prompt, one prompt per update,
one accumulation step, learning rate `1e-5`, five epochs, temperature `0.9`,
768 new tokens, clipping `0.2`, seed `42`. Evaluation uses temperature `0.6`.
The code uses PyTorch, Transformers and PEFT directly rather than relying on
version-sensitive TRL objective defaults.

**Reproducibility caveats:**

- The paper does not publish the exact Ruff rule-to-severity mapping.
  `slm_grpo/rewards.py` makes an explicit approximation: undefined names and invalid
  syntax are errors; selected invalid control-flow/comparison rules are high;
  unused imports/variables are low; other findings default to normal, with a
  rule-token fallback from diagnostic messages. This is not an exact reproduction.
- Ruff runs with an isolated default ruleset (`E4,E7,E9,F`), never this repository's
  broader lint configuration. The optional requirements pin the paper's Ruff
  version. A failed Ruff invocation/invalid JSON is logged and gives raw `-0.5`
  before normalization. Missing OS executables and timeouts propagate explicitly.
- Equation (4) in **this paper** defines GSPO as `exp(sum(new_logp - old_logp))`.
  This implementation follows that equation, **not** the length-normalized
  geometric-mean ratio used in canonical GSPO and current TRL. It is intentionally
  labeled a paper-specific variant.
- Group advantages are centered by default; `--normalize-advantages` enables the
  paper's optional normalization. Sampling log probabilities include temperature
  scaling; top-k and top-p filtering are disabled. The model stays in evaluation
  mode during optimization to disable dropout while preserving adapter gradients.
- The paper leaves optimizer details, extraction edge cases and execution timeout
  underspecified. Here AdamW has zero weight decay, gradients are clipped to norm 1,
  fenced Python blocks are joined after a closed thinking preamble, Qwen thinking
  is disabled, and the per-candidate timeout defaults to 3 seconds.
- Unsloth acceleration, vLLM/EvalPlus, the authors' full multi-seed experiments,
  plots and statistical significance tests are **not** reproduced here.

## Execution safety

**Generated Python is arbitrary code.** The verification subprocess is **not a
security sandbox**: isolation mode, a temporary working directory, a reduced
environment, basic POSIX resource limits, a wall-clock timeout, and process-group
cleanup only contain ordinary accidental failures. They do not prevent filesystem
or network access, reward tampering, or adversarial process/resource abuse.

Run training/evaluation **inside a disposable container or VM**, as an unprivileged
user, with no secrets or sensitive mounts, restricted resources, and no network
during verification. Download models/datasets beforehand. All commands executing
generated code require `--allow-code-execution`; Ruff-only training does not.
Execution requires POSIX (Linux/macOS). Do not use this harness to evaluate
untrusted submissions on your host.

## Installation

From the repository root, create a separate environment so the experiment's
dependencies, especially its older Ruff pin, do not replace development tools:

```bash
python -m venv papers/slm-grpo-2026/.venv
source papers/slm-grpo-2026/.venv/bin/activate
python -m pip install -r papers/slm-grpo-2026/requirements.txt
```

No editable project installation is needed. A CUDA GPU is recommended for the real
models; CPU is useful for tiny offline tests. Llama requires access to the gated
`meta-llama/Llama-3.2-1B-Instruct` checkpoint.

CUDA requires a working PyTorch/CUDA/cuDNN installation. If it reports
`CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`, check that `LD_LIBRARY_PATH` includes
`nvidia/cudnn/lib` from the **active Python environment**, not another installation.

## Training

From the repository root, inside the isolated environment described above, run:

```bash
python papers/slm-grpo-2026/run.py train \
  --model Qwen/Qwen3-0.6B \
  --device cuda \
  --algorithm grpo --reward combined \
  --group-size 8 --epochs 5 --learning-rate 1e-5 \
  --temperature 0.9 --max-tokens 768 --seed 42 \
  --output papers/slm-grpo-2026/runs/qwen-grpo-combined \
  --allow-code-execution
```

This command explicitly selects the paper's training hyperparameters. For CPU
execution, replace **`--device cuda` with `--device cpu`**; the same training code
and output layout apply, but full training is much slower. `--device cuda:1`
selects another GPU. If omitted, the device defaults to CUDA when available,
otherwise CPU. Explicitly requesting unavailable CUDA fails rather than silently
switching to CPU.

| Option | Default | Other choices |
|---|---|---|
| `--model` | `Qwen/Qwen3-0.6B` | `meta-llama/Llama-3.2-1B-Instruct` (gated), or a compatible local Transformers checkpoint |
| `--device` | CUDA if available, otherwise CPU | `cuda`, `cuda:N`, `cpu` |
| `--algorithm` | `grpo` | `gspo` (paper-specific variant described above) |
| `--reward` | `combined` | `unit`, `ruff`, `ruff-normalized` |
| `--output` | Required; no implicit directory | Any empty or absent run directory |

Use `--algorithm gspo` for the paper's sequence objective. Use `--reward unit`,
`--reward ruff`, or `--reward ruff-normalized` for the ablations. To test the other
architecture, set `--model meta-llama/Llama-3.2-1B-Instruct`.
`--limit 2 --epochs 1 --group-size 2 --max-tokens 64` provides a short smoke run,
not a meaningful reproduction.

Each group is sampled before any update. Old log probabilities are detached;
per-completion backward passes accumulate the group mean before one optimizer
step, reducing activation memory. `--updates-per-group` optionally reuses the same
group, with its original likelihoods fixed. Existing nonempty output directories
are rejected rather than overwritten.

### Progress and saved model

Progress is printed to **standard output**, flushed immediately: dataset/model
loading, each sampled completion, each optimizer update, and final save paths.
Training logs include epoch/task/sample counters, token count, reward, execution
outcome, loss, gradient norm before clipping, mean group reward, and elapsed
seconds. For example (illustrative):

```text
[train] epoch=1/5 task=1/374 sampling 8 completions
[train] epoch=1/5 task=1/374 sample=1/8 tokens=120 reward=1.0000 outcome=passed
[train] epoch=1/5 task=1/374 step=1/1870 update=1/1 loss=0.000000 grad_norm=0.1200 mean_reward=0.2500 elapsed=12.3s
```

The directory passed to `--output` contains both the trained model adaptation and
the results:

```text
papers/slm-grpo-2026/runs/qwen-grpo-combined/
  config.json                # Resolved command-line options and hyperparameters
  train.jsonl                # Completions, rewards and verification diagnostics
  adapter/
    adapter_config.json      # LoRA configuration and base-model reference
    adapter_model.safetensors
    ...                      # Saved tokenizer files
```

The base model is frozen, so only its **LoRA adapter**, not a second full copy of
the base weights, is saved. Reload it using `--model` and `--adapter` together,
as in the evaluation command below.

## Evaluation

```bash
# Untuned baseline: omit --adapter.
python papers/slm-grpo-2026/run.py evaluate \
  --model Qwen/Qwen3-0.6B \
  --device cuda \
  --output papers/slm-grpo-2026/runs/qwen-baseline \
  --allow-code-execution

# Trained policy: use the same base checkpoint as training.
python papers/slm-grpo-2026/run.py evaluate \
  --model Qwen/Qwen3-0.6B \
  --device cuda \
  --adapter papers/slm-grpo-2026/runs/qwen-grpo-combined/adapter \
  --output papers/slm-grpo-2026/runs/qwen-grpo-combined-test \
  --allow-code-execution
```

Training defaults to MBPP `train`; evaluation defaults to `test`. The full MBPP
configuration is used, not the sanitized subset. `--split` can override the split.
For repeated evaluation, use separate output directories and `--seed` values.
Compare baseline and adapters with identical decoding settings and task subsets.
`metrics.json` contains pass@1 and diagnostics; `evaluate.jsonl` retains every
completion and its verification results. Evaluation always computes both unit-test
and Ruff diagnostics, regardless of the training reward. Both files and the
evaluation `config.json` are stored in its `--output` directory. Evaluation also
prints per-task progress and the running pass@1 to standard output. Use
`--device cpu` here too if CUDA is unavailable.

For offline/custom experiments, `--dataset /path/to/tasks.json` accepts a JSON list
of MBPP-shaped records: `text`, `test_list`, optional `task_id`,
`challenge_test_list`, and `test_setup_code`. In this mode the file itself defines
the split: use distinct training and held-out files. `--model` also accepts a local
Transformers checkpoint. `--help` lists all options.

## Tests

From the repository root, with pytest and Ruff available:

```bash
python -m pytest -q unittests/papers/test_slm_grpo.py
```

The tests cover real subprocess outcomes and timeout handling, setup/challenge
tests, prompt leakage, Ruff isolation, reward formulas, clipping and masking,
gradient behavior, and adapter-only updates. With the optional training dependencies,
they also train and reload both algorithms using a tiny, randomly initialized
Qwen3 checkpoint and a tokenizer created locally: **no model or dataset downloads**.
These integration tests run on CPU and also on CUDA when available, and check
the progress messages written to standard output.
PyTorch/PEFT-dependent tests skip when those optional dependencies are absent.
These checks do not establish the paper's reported benchmark gains; reproducing
those requires full GPU training and held-out evaluation.
