# Reinforcement Learning for Code Optimization (2026)

A standalone research implementation of
[Reinforcement Learning for Code Optimization, arXiv:2607.25970v2](https://arxiv.org/abs/2607.25970v2)
([HTML](https://arxiv.org/html/2607.25970v2),
[PDF](https://arxiv.org/pdf/2607.25970v2)).

It trains a model to generate **correct and fast stdin/stdout Python programs**.
It follows the paper's environment/reward interfaces and stabilized GRPO
equations, with a synchronous, local LoRA implementation. It is independent
of locodellm's ONNX inference API.
Licensing follows the project; no separate study license, packaging manifest
or requirements files are added.

## Paper summary and implementation scope

Correctness rewards alone do not reliably produce fast code. The paper
separates three issues: larger tests that distinguish algorithms, calibrated
execution timing, and a learning rule that handles sparse/noisy speed rewards.
It constructs DMC-Optim from Code Contests, screens environments by replaying
human solutions, then trains with a stabilized GRPO recipe.

| Published observation | Paper result, not a local reproduction |
|---|---|
| Qwen 2.5 7B, strict top-50% pass@1 | 18.0% to 31.3% |
| CWM 32B, strict top-50% pass@1 | 30.7% to 50.4% |
| Timing data | 2,723 cleaned problems; 1,302 duration-filterable problems |
| Learning | Token-weighted prompt baseline, no standard-deviation normalization, fixed token horizon |

| Component | Implemented here | Difference from the paper |
|---|---|---|
| Test execution | Real stdin/stdout runs, private correctness and optimization tests, repeated median timings | Local Linux subprocesses, not the paper's remote CES service |
| Calibration | Remeasures verified reference programs on each worker's assigned physical CPU core | No affine CES service-state correction; local timing remains contention-sensitive |
| Environments | Absolute/relative/length pre-filters; absolute/relative/ranked-worst deadlines; QAR/QP ranking | User-provided verified tests/references, not automatic DMC-Optim construction |
| Rewards | Correctness, optimization, collapsed, two-gate, blend and multitask; binary/graded quality maps | Same appendix D.4 composition interface |
| Learning | Real LoRA updates with token-weighted baseline and fixed-horizon clipped objective | Synchronous DDP instead of asynchronous worker/trainer nodes; no token-budget packing |
| Offline screening | Quality-band replay and appendix E.2 curve diagnostics | Screens measured correct references, not a prediction of model learning |
| Evaluation | 20 samples by default; strict p100/p50/p30/p10 pass@1/pass@10; reusable traces | No LCB comparison campaign or claimed benchmark gains |

The original curated corpus, the authors' custom Qwen SFT checkpoint and CES
are not bundled or silently substituted. The default available model,
`Qwen/Qwen2.5-7B-Instruct`, differs from the paper's reasoning-SFT starting
checkpoint. LoRA rank 32/alpha 64 is a local memory-saving adaptation, not
the authors' full training configuration.

**Safety:** generated code is arbitrary Python. Verification has time, memory,
file-size and process-group cleanup limits, but **is not a security sandbox**.
Use a disposable, unprivileged Linux container/VM with no network, secrets or
sensitive mounts. Download model/data artifacts beforehand. Training,
evaluation and calibration require explicit `--allow-code-execution`.
Simulation and replay do not execute candidate code.

## Installation

Installs the optional study dependencies without installing the project:

```bash
python -m venv papers/code-optimization-2026/.venv
source papers/code-optimization-2026/.venv/bin/activate
python -m pip install "torch>=2.6" "transformers>=4.57,<6" \
  "peft>=0.18,<1" "scipy>=1.15,<2" "tqdm>=4.66,<5"
```

Install the PyTorch build matching the machine's CUDA libraries. The Linux
launcher reuses the cuDNN loader-path setup from the earlier SLM study.
All commands below are run from the repository root.

## Dataset: 100 training and 100 test problems

The ready-to-use dataset in `data/thealgorithms-python/` derives **200 problems**
from [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python), pinned to
commit `6e03a48579ca212a08b36be00378f887dc41a003`:

| File, relative to this study | Problems | Usage |
|---|---:|---|
| `data/thealgorithms-python/optimization-train.json` | 100 | Training |
| `data/thealgorithms-python/optimization-test.json` | 100 | Held-out evaluation |

Each problem comes from a different source file and has at least four distinct
upstream doctest calls with literal, known expected results. Its stdin is a JSON
batch of calls, each with `args` and `kwargs`; stdout is the ordered JSON result
array. Two correctness batches and two larger batches of **64 and 256 calls**
are included. Both reference programs passed **every** included test.

The references use the same upstream algorithm: one evaluates every call, the
other memoizes repeated arguments within the batch. Entire top-level source
categories are assigned to just one split, so source files, function names and
categories do not cross the train/test boundary. This is a category-held-out
split, not a stratified random split. `manifest.json` records provenance, split
counts, exclusions and dataset SHA-256 hashes. MIT notices are preserved in
`NOTICE.txt` and in each reference.

**Bootstrap dataset, not DMC-Optim:** batching increases workload, not individual
algorithm input sizes. The two wrappers are not independent correctness oracles;
both rely on upstream doctest answers. Timing separation, broader input coverage,
semantic deduplication and pretraining contamination have **not** been qualified.
Do not interpret results as reproducing the paper's optimization benchmark.

### Regeneration

The committed JSON files need no download or generation before training.
To rebuild from source into a new directory (Python 3.12, standard library only):

```bash
git clone https://github.com/TheAlgorithms/Python.git /tmp/thealgorithms-python
git -C /tmp/thealgorithms-python checkout 6e03a48579ca212a08b36be00378f887dc41a003
python -m papers.code-optimization-2026.optim_rl.algorithm_data \
  --source /tmp/thealgorithms-python \
  --output /tmp/thealgorithms-optimization-200 \
  --allow-code-execution
```

The builder checks the revision and rejects modified tracked sources or existing
output directories. It never imports entire upstream modules, rejects unsupported
dependencies, and validates references using the same resource-limited subprocess
backend as training. Execution still requires a disposable environment, not a
trusted host. Selection is fixed before execution: a reference failure or timeout
aborts publication rather than silently replacing a problem. The verification cap
is 10 seconds per reference/test, matching calibration. No GPU is needed.

### Task file format and private verification inputs

`--dataset` accepts a nonempty local JSON list:

```json
[
  {
    "task_id": "sum-positive",
    "problem": "Read n and print the sum of integers from 1 through n.",
    "correctness_tests": [{"input": "3\n", "output": "6\n"}],
    "optimization_tests": [{"input": "100000\n", "output": "5000050000\n"}],
    "references": [
      "n = int(input())\nprint(n * (n + 1) // 2)",
      "n = int(input())\nprint(sum(range(n + 1)))"
    ]
  }
]
```

Provide at least one test of each kind and two known-correct reference programs.
Each reference must pass every correctness and optimization test within the
10-second calibration cap; otherwise the run fails explicitly. Output comparison
is whitespace-token exact, not floating-point tolerant. Only `problem` goes
into the model prompt: reference code, expected outputs and hidden tests do not.
Separate training and evaluation datasets yourself, including related-problem
decontamination. The input file and per-task content receive SHA-256 identities;
offline commands reject calibration for changed task content.

`examples/sum.json` is a runnable, tiny integration example, **not DMC-Optim or a
valid performance benchmark**. Prepare real large-input optimization workloads
and verified controls before interpreting any learned speed gain.

## GPU selection: all eight when idle

Both training and evaluation use **every idle GPU visible to PyTorch** when
`--device cuda` is selected, including all eight GPUs on an idle eight-GPU
machine. There is no 50-problem boundary or fixed GPU limit.

`nvidia-smi` identifies GPUs with other compute processes; those GPUs are
excluded even if they have free memory. `CUDA_VISIBLE_DEVICES` is respected.
The selected devices and distinct physical verification CPU cores are printed
and saved in `config.json`. If no GPU is idle, the command fails instead of
silently sharing or falling back to CPU. `--device cpu` uses one CPU worker.
`--device cuda:N` explicitly permits sharing a chosen visible GPU.

Training launches one synchronized DDP replica per selected GPU; gradients are
scaled to compensate for DDP averaging and retain the paper's global
fixed-horizon **sum**. All workers must finish with identical adapter hashes.
Evaluation distributes task/sample pairs, so a small dataset can still use
all GPUs when enough samples are requested. Seeds depend on task/sample
identity, not completion order.

Availability is only a startup snapshot, not an exclusive reservation.
Verification cores are distinct across this run but are not reserved from
external workloads. More GPUs increase the global number of prompt groups per
update: record GPU count when comparing runs.

## Training

Trains a top-30% **per-test-percentile (QAR), binary collapsed-reward** policy
on the 100 included training problems, using all idle GPUs. Checkpoints are saved
**every 20 minutes**, after the next completed distributed update:

```bash
python papers/code-optimization-2026/run.py train \
  --dataset papers/code-optimization-2026/data/thealgorithms-python/optimization-train.json \
  --model Qwen/Qwen2.5-7B-Instruct --device cuda \
  --max-tokens 16384 \
  --output papers/code-optimization-2026/runs/qwen-qar-p30 \
  --allow-code-execution
```

For a tiny functional run, use `--dataset papers/code-optimization-2026/examples/sum.json`
and `--steps 1 --group-size 2 --max-tokens 32 --repeats 1`.
Defaults follow the paper where applicable: 16 rollouts, temperature 1, learning
rate `1e-7`, fixed horizon 32,768 and 10,000 training iterations. The default
generation cap is 32,768; lower it for the model's total context budget.
`--prompts-per-step` controls groups **per GPU**, not the global batch.
`--repeats` defaults to 3 independent runs per test.

All generated trajectories in a group use the same policy before any update.
The prompt-level baseline and advantage (appendix F, equation 50) are:

$$
\mu_x=\frac{\sum_i L_i R_i}{\sum_i L_i},\qquad A_i=R_i-\mu_x.
$$

There is **no division by reward standard deviation**. The loss is the negative
global sum of clipped token importance-ratio surrogates divided by the fixed
`--horizon`, not each trajectory's length (equation 51). Old probabilities are
detached, sampling/log probabilities share temperature, and dropout is disabled.
There is no KL penalty. Globally constant-reward batches skip the optimizer
step; a rank with no local signal still participates in DDP synchronization.
Synchronous sampling means rollout age is zero: there is no stale queue or
reward replay buffer. The paper's asynchronous 30-step age filter is therefore
not needed in this adaptation.

### Rewards and ranking

`c` means all correctness tests pass. `C` is stricter: `c` holds and every
executed optimization test either passes or times out; a wrong output or
runtime error invalidates `C`. Optimization timeouts are recorded at their
deadline, not treated as successful executions.

For QAR, insert the candidate duration into each test's reference timings,
compute its normalized rank and average across tests (equations 28-29).
QP additionally ranks that mean against reference means **recomputed with the
candidate inserted** (equation 30). Faster is lower; exact ties use the
strict-less rank. Ranking never compares raw seconds from different problems.

| Environment option | Optimization constraint | Quality `q` (lower is better) |
|---|---|---|
| `qar` | Mean per-test percentile <= `--threshold` | QAR |
| `qp` | Leaderboard percentile <= `--threshold` | QP |
| `absolute-filter` | Keep tests with reference mean < `--cutoff` seconds | Timeout fraction |
| `relative-filter` | Drop the slowest `--threshold` fraction by reference mean (ties retained) | Timeout fraction |
| `length-filter` | Keep tests with input + output length < `--cutoff` characters | Timeout fraction |
| `timeout` | Use `--timeout` seconds per optimization test | Timeout fraction |
| `relative-timeout` | Per-test reference quantile at `--threshold`, minimum 1 ms | Timeout fraction |
| `ranked-timeout` | Worst duration among the best `--threshold` fraction of ranked references, minimum 1 ms | Timeout fraction |

The optimization gate `g` also requires timeout fraction <= `--tolerance`.
Post-ranking defaults use tolerance 1 and a 10-second systems cap; pre/intra
experiments should explicitly set their intended timeout/tolerance, e.g.
`--environment absolute-filter --cutoff 2 --timeout 0.5 --tolerance 0.1`.
Correctness tests always retain the separate 10-second cap.

| Quality map | Unsigned efficiency credit `m` |
|---|---|
| `binary` (default) | 1 if `g`, otherwise 0 |
| `linear` | `1 - q` |
| `bucket` | Equation 44: `q=0` gives 1; otherwise ten decile bins down to 0 |
| `range-third` | `1/3 + (2/3) * (1-q)` |
| `range-half` | `1/2 + (1/2) * (1-q)` |

| Reward option | Scalar reward | Can speed help incorrect code? |
|---|---|---|
| `correctness` | `2*c - 1` | No speed term |
| `optimization` | `2*m - 1` | Yes; ablation/control only |
| `collapsed` | `2*C*m - 1` | No |
| `two-gate` | `m` if `C`, otherwise -1 | No |
| `blend` | `blend*(2*c-1) + (1-blend)*(2*m-1)` | Yes |
| `multitask` | 50/50 correctness or optimization stream, chosen per prompt group | Yes in optimization-only groups |

For the default binary collapsed reward, a strictly correct top-30% candidate
gets **+1**, a correct but slower candidate **-1**, and an incorrect candidate
**-1** regardless of speed. For `--reward two-gate --quality-map linear`, a
strictly correct candidate at `q=0.2` gets **0.8**, while incorrect code gets **-1**.
The `--blend` weight defaults to 0.5.

### Progress, checkpoints and continuation

A stdout tqdm bar reports iteration, actual optimizer updates, reward, loss,
gradient norm and useful group count. Calibration/verification happen before
the next update, so slow tests can temporarily leave the bar unchanged.
At the on-policy starting ratio, the token-weighted baseline can make the
displayed loss nearly zero while its gradient is nonzero; the gradient norm
and optimizer update count distinguish this from a skipped batch.

| Output | Contents |
|---|---|
| `config.json` | Resolved CLI/environment, dataset and implementation digests, devices, CPU cores, platform |
| `model-rank<N>.json` | Resolved model revision when available, dtype, device and Torch version |
| `train-rank<N>.jsonl` | Generated programs, rewards, individual test outcomes and durations |
| `calibration.jsonl` and `calibration-rank<N>.jsonl` | CPU-bound reference timing matrices with task content digests |
| `checkpoint-<update>/` | Atomically published LoRA weights, tokenizer, update metadata |
| `adapter/` | Final LoRA weights and tokenizer; base weights stay frozen |
| `training-rank<N>.json`, `training-metrics.json` | Update counts and matching adapter hashes |

Outputs must be absent; existing directories are never overwritten.
Checkpoints are saved every 20 minutes by default, after a whole distributed
iteration; long generation can delay a save. `--checkpoint-steps N` optionally
adds a step-based trigger (disabled by default).

Continues from learned weights using `--adapter <run>/checkpoint-50`, with a
**new** output directory. This is a warm start: optimizer, RNG and data-order
state are not restored. Only weights/tokenizer and update metadata are saved.

## Baseline and trained evaluation

Evaluates the **base model** on the 100 held-out problems using all idle GPUs.
This command works **before training**; no adapter is required:

```bash
python papers/code-optimization-2026/run.py evaluate \
  --dataset papers/code-optimization-2026/data/thealgorithms-python/optimization-test.json \
  --model Qwen/Qwen2.5-7B-Instruct --device cuda \
  --samples 20 --pass-k 1 10 --max-tokens 16384 \
  --output papers/code-optimization-2026/runs/qwen-baseline \
  --allow-code-execution
```

**After training**, add
`--adapter papers/code-optimization-2026/runs/qwen-qar-p30/adapter`
and change `--output` to `papers/code-optimization-2026/runs/qwen-qar-p30-test`.
That adapter directory only exists after a completed training run; an interrupted
run may instead have a usable `checkpoint-<update>/`. Missing local adapters are
rejected before model loading, GPU selection or output creation: evaluation never
silently falls back to the base model.

Each `--output` must be a **new directory**. Failed run directories are retained
for diagnosis; choose a different output path when retrying.

Evaluation always runs **all** optimization tests under the 10-second cap and
computes QP, independent of the selected training environment. `evaluate.jsonl`
is ordered by task/sample and retains timings, verdicts and reference keys.
Only after all workers succeed is `metrics.json` published.

For each task with `n` samples and `m` qualifying successes, pass@k is
`1 - choose(n-m,k)/choose(n,k)` (equation 2). p100 checks strict correctness
only, allowing optimization timeouts but not failures. p50/p30/p10 also require
QP <= 0.5/0.3/0.1. The metric is a mean over problems, not a pooled sample rate.
Use validation data for checkpoint selection; do not select on the final test set.

## CPU calibration, offline screening and trace replay

Measures reference programs without loading a model:

```bash
python papers/code-optimization-2026/run.py calibrate \
  --dataset papers/code-optimization-2026/examples/sum.json \
  --repeats 3 --output papers/code-optimization-2026/runs/sum-calibration \
  --allow-code-execution
```

Screens a reward/environment using recorded reference timings, without code
execution or GPUs:

```bash
python papers/code-optimization-2026/run.py simulate \
  --dataset papers/code-optimization-2026/examples/sum.json \
  --calibration papers/code-optimization-2026/runs/sum-calibration/calibration.jsonl \
  --environment qar --threshold 0.3 \
  --output papers/code-optimization-2026/runs/sum-simulation
```

`simulation.json` contains the quality-band activation/reward curves, raw and
smoothed AUC, clipped-linear steepness, moving-average variance, diagonal
deviation and Spearman ordering diagnostics (appendix E.2). The sampler moves
from slower to faster verified references; it does not simulate gradients,
correctness learning or predict the paper's benchmark gains. Constant curves
report null, not a fabricated correlation.

Replays a full evaluation trace through another environment/reward without
rerunning programs:

```bash
python papers/code-optimization-2026/run.py replay \
  --dataset papers/code-optimization-2026/data/thealgorithms-python/optimization-test.json \
  --calibration papers/code-optimization-2026/runs/qwen-baseline/calibration.jsonl \
  --records papers/code-optimization-2026/runs/qwen-baseline/evaluate.jsonl \
  --environment qar --threshold 0.5 --reward two-gate --quality-map linear \
  --output papers/code-optimization-2026/runs/qwen-replay
```

Replay requires complete optimization traces and matching task identities.
It preserves observed hard failures, supports tighter deadlines, and rejects
deadlines above the original 10-second recording cap. It cannot infer what
happened after a censored timeout. `replay.jsonl` contains the rescored records.

## Validation and limitations

Runs numerical-contract checks, real subprocess verification, a tiny offline
LoRA update/checkpoint/reload, evaluation and trace replay:

```bash
python -m pip install "pytest>=7,<10"
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=. python -m pytest -q unittests/papers/test_code_optimization.py
```

Local GPU checks used up to **seven idle H200s** (`cuda:1` through `cuda:7`).
A seven-rank correctness-control update and a six-rank default QAR/collapsed
update completed with identical adapter hashes on every participating rank;
the changing device count reflects another workload occupying GPU 1 at startup.
The always-occupied GPU 0 was excluded. Adapter reload and 14 evaluation samples
were checked across seven idle GPUs. These tiny local Qwen3 checks validate
execution/distribution, **not a multi-GPU benchmark gain**.
Eight-idle-device selection is additionally covered by a deterministic unit test.

Not reproduced: DMC-Optim data generation/cleaning and duration-filterability
qualification, remote CES isolation/affine correction, the authors' SFT/full-weight
training, async packing/staleness scheduling, multi-seed statistics, LCB and
published speed/correctness gains. Local CPU contention, interpreter/import
costs within the timed program and machine drift can change rankings.
References are freshly calibrated per training prompt group and cached per
task/CPU during evaluation; these are not cross-run calibrated CES timings.
