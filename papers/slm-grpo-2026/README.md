# RLVR for small code language models (2026)

An independent implementation of **Improving Small Language Models for Code
Generation with Reinforcement Learning from Verification Feedback**, E. Kotelnikov.

Paper: [arXiv:2605.30478](https://arxiv.org/abs/2605.30478)
([PDF](https://arxiv.org/pdf/2605.30478)).

## Paper summary

The paper trains small Python code-generation models on MBPP using unit-test
results and Ruff diagnostics as rewards. It compares GRPO and GSPO with LoRA
adapters on Qwen3-0.6B and Llama-3.2-1B-Instruct. The authors report up to
**13 percentage points** improvement in pass@1, but Ruff-only rewards can favor
short, incorrect programs. Neither algorithm consistently wins.

This standalone PyTorch implementation leaves the ONNX inference API unchanged.

**Safety:** generated Python is arbitrary code; the verification subprocess is
**not a security sandbox**. Run it in a disposable, unprivileged container/VM
without secrets, sensitive mounts, or network access during verification.
Download models/datasets beforehand. Execution requires POSIX and explicit
`--allow-code-execution`; Ruff-only training does not execute generated code.

## Installation

From the repository root, install the optional dependencies in a separate environment:

```bash
python -m venv papers/slm-grpo-2026/.venv
source papers/slm-grpo-2026/.venv/bin/activate
python -m pip install -r papers/slm-grpo-2026/requirements.txt
```

No project installation is needed. CUDA requires matching PyTorch/CUDA/cuDNN
libraries. For `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`, check that
`LD_LIBRARY_PATH` points to the active environment's `nvidia/cudnn/lib`.

## Training

From the repository root, inside the isolated environment described above, run:

```bash
# Train Qwen3 with GRPO and the combined reward.
python papers/slm-grpo-2026/run.py train \
  --model Qwen/Qwen3-0.6B \
  --device cuda \
  --algorithm grpo --reward combined \
  --output papers/slm-grpo-2026/runs/qwen-grpo-combined \
  --allow-code-execution
```

For CPU, replace **`--device cuda` with `--device cpu`** (much slower).
The defaults follow the paper: 8 completions per prompt, 5 epochs, learning rate
`1e-5`, temperature `0.9`, 768 new tokens, seed `42`, LoRA rank `32` and alpha `64`.
There is one prompt per update and no KL regularization.

| Option | Default | Other choices |
|---|---|---|
| `--model` | `Qwen/Qwen3-0.6B` | `meta-llama/Llama-3.2-1B-Instruct` (gated), or a compatible local Transformers checkpoint |
| `--device` | CUDA if available, otherwise CPU | `cuda`, `cuda:N`, `cpu` |
| `--algorithm` | `grpo` | `gspo` (paper-specific variant described above) |
| `--reward` | `combined` | `unit`, `ruff`, `ruff-normalized` |
| `--output` | Required; no implicit directory | Any empty or absent run directory |

For a smoke run, add `--limit 2 --epochs 1 --group-size 2 --max-tokens 64`.
Existing nonempty output directories are never overwritten.

### Progress and saved model

Progress appears immediately on **stdout**: loading, epoch/task/sample counters,
rewards, loss, gradient norm, elapsed time, and save paths. `--output` contains:

```text
papers/slm-grpo-2026/runs/qwen-grpo-combined/
  config.json                # Resolved command-line options and hyperparameters
  train.jsonl                # Completions, rewards and verification diagnostics
  adapter/
    adapter_config.json      # LoRA configuration and base-model reference
    adapter_model.safetensors
    ...                      # Saved tokenizer files
```

Only the **LoRA adapter** is saved; the frozen base model is loaded separately.

## Evaluation

```bash
# Evaluate the original model, without training.
python papers/slm-grpo-2026/run.py evaluate \
  --model Qwen/Qwen3-0.6B \
  --device cuda \
  --output papers/slm-grpo-2026/runs/qwen-baseline \
  --allow-code-execution
```

```bash
# Evaluate the trained adapter with its original base model.
python papers/slm-grpo-2026/run.py evaluate \
  --model Qwen/Qwen3-0.6B \
  --device cuda \
  --adapter papers/slm-grpo-2026/runs/qwen-grpo-combined/adapter \
  --output papers/slm-grpo-2026/runs/qwen-grpo-combined-test \
  --allow-code-execution
```

Evaluation uses MBPP `test`, temperature `0.6`, and one completion per task.
Each output directory receives `config.json`, `evaluate.jsonl` (completions and
diagnostics), and `metrics.json` (pass@1, error types, lengths, Ruff severities).
Per-task progress and running pass@1 appear on stdout. Use `--device cpu` if needed.

## Learning algorithm

For each MBPP task $x$, the prompt exposes only the first test, never the reference
solution. Sample $G=8$ programs $y_i$ with the current policy, then score them using
all standard/challenge tests and setup code.

### Reward

The unit-test reward is $R_{\mathrm{unit}}=1$ for success, $-0.3$ for an assertion
failure or timeout, $-0.6$ for a runtime error, and $-1$ for interpreter/other errors.
If $N_s(y)$ counts Ruff findings of severity $s$ and $L(y)$ counts lines:

$$
R_{\mathrm{raw}}(y)=\sum_s w_s N_s(y),
\qquad (w_{\mathrm{error}},w_{\mathrm{high}},w_{\mathrm{normal}},w_{\mathrm{low}})
=(-10,-5,-1,-0.2).
$$

$$
R_{\mathrm{ruff}}(y)=\operatorname{clip}
\left(\frac{R_{\mathrm{raw}}(y)}{\max(1,L(y))},-1,0\right),
\qquad
R_{\mathrm{short}}(y)=-0.5\,\mathbf{1}
\left[\operatorname{chars}(y)<30\ \text{or}\ L(y)<3\right].
$$

The default combined reward prioritizes correctness:

$$
R_i=R_{\mathrm{unit}}(y_i)+\lambda_i R_{\mathrm{ruff}}(y_i)
+R_{\mathrm{short}}(y_i),
\qquad
\lambda_i=
\begin{cases}
0.3 & \text{if all tests pass},\\
0.1 & \text{otherwise}.
\end{cases}
$$

### Policy update

Compute each program's advantage relative to its group:

$$
\bar R=\frac{1}{G}\sum_{j=1}^{G}R_j,\qquad A_i=R_i-\bar R.
$$

`--normalize-advantages` optionally divides $A_i$ by the group's population
standard deviation plus $10^{-4}$. Equal rewards give zero learning signal.
Let $T_i$ be the number of completion tokens and $\pi_{\mathrm{old}}$ the frozen
sampling policy. With clipping $\epsilon=0.2$, define:

$$
r_{i,t}=\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}
{\pi_{\mathrm{old}}(y_{i,t}\mid x,y_{i,<t})},
\qquad
f(r,A)=\min\left(rA,\operatorname{clip}(r,1-\epsilon,1+\epsilon)A\right).
$$

**GRPO** averages the clipped objective over tokens, then programs:

$$
J_{\mathrm{GRPO}}(\theta)=\frac{1}{G}\sum_{i=1}^{G}
\frac{1}{T_i}\sum_{t=1}^{T_i}f(r_{i,t},A_i).
$$

**GSPO (this paper's variant)** uses one importance ratio per program:

$$
r_i=\exp\left(\sum_{t=1}^{T_i}\log r_{i,t}\right),
\qquad
J_{\mathrm{GSPO}}(\theta)=\frac{1}{G}\sum_{i=1}^{G}f(r_i,A_i).
$$

The implementation minimizes $-J$ using AdamW on **LoRA parameters only**.
The base weights stay frozen; $\beta=0$ means no reference-model KL term.
Each group is sampled before updating; its old log probabilities remain fixed.
By default there is one optimizer step per group; `--updates-per-group` changes it.

## Implementation notes

- GSPO follows the paper's **summed** log-ratio equation, not canonical GSPO's
  length-normalized ratio.
- The exact Ruff severity mapping is unpublished; `rewards.py` documents the
  approximation. Ruff uses isolated rules `E4,E7,E9,F`; a failed invocation or
  invalid JSON is logged and receives raw penalty `-0.5` before normalization.
- Sampling/log probabilities use the same temperature, without top-k/top-p
  filtering. Dropout and Qwen thinking are disabled. AdamW uses zero weight decay,
  gradient clipping at norm 1, and a 3-second execution timeout.
- `--dataset tasks.json` accepts local MBPP-shaped records (`text`, `test_list`,
  optional `task_id`, `challenge_test_list`, `test_setup_code`). Use separate
  training/test files; `--split` applies to the hosted dataset. See `--help`.
- Full GPU benchmark gains, multi-seed statistics, Unsloth, and EvalPlus/vLLM
  experiments have **not** been reproduced.

## Tests

From the repository root, with pytest and Ruff available:

```bash
python -m pytest -q unittests/papers/test_slm_grpo.py
```

Tests cover rewards, verification, objectives, stdout progress, and adapter-only
updates. Tiny local Qwen3 models exercise both algorithms on CPU/CUDA without
downloads. Tests requiring optional dependencies or unavailable CUDA skip.
