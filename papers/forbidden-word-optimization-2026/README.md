# Function-speed GRPO on forbidden-word-avoidance

An independent application of the algorithm from
[PR #39](https://github.com/xadupre/locodellm/pull/39) to the local
`forbidden-word-avoidance` function-generation dataset. The engine is copied
from commit `d5e4d32` into this directory; it does not import the other study.
This PR can be merged independently of #39.

**The objective is speed subject to correctness, not forbidden-word avoidance.**
The original dataset's lexical cost is not used. Original prompts are preserved,
but correctness means passing the supplied return-value cases, not proving all
algorithmic, complexity or generated-test requirements stated in each prompt.

The method keeps the stabilized GRPO objective: a token-weighted group baseline,
no standard-deviation normalization, fixed-horizon clipped token loss, and
synchronous LoRA updates. It keeps reference-relative QAR/QP rewards, repeated
median timings, all-idle-GPU selection, DDP gradient scaling, atomic adapter
checkpoints, evaluation, offline replay and simulator diagnostics from #39.

## Installation and safety

Use the source checkout through `PYTHONPATH`, without installing this project:

```bash
python -m pip install "torch>=2.6" "transformers>=4.57,<6" \
  "peft>=0.18,<1" "scipy>=1.15,<2" "tqdm>=4.66,<5"
export PYTHONPATH=.
```

Install the PyTorch build appropriate for the machine. The launcher configures
the active environment's cuDNN loader path when available.

Generated reference and candidate programs are arbitrary Python. Workers have
time, memory, output-size and process-group limits, but **are not a security
sandbox**. Run preparation, calibration, training and evaluation in a disposable,
unprivileged Linux container/VM without network, credentials or sensitive mounts.
Download model files beforehand. Execution requires `--allow-code-execution`.
Replay and simulation do not execute generated code.

## Dataset and reference preparation

The external dataset is not copied, modified or downloaded. Despite their `.json`
extension, its `train.json` and `test.json` contain JSON Lines, with 100 tasks
each. Each task has:

```json
{
  "id": "increment",
  "function": "solve",
  "prompt": "Write solve(n), returning n + 1, and include example tests.",
  "cases": [
    {"args": [1], "expected": 2},
    {"args": [-1], "expected": 0}
  ]
}
```

The loader also accepts a JSON array, optional case `kwargs`, French prompt
files and prefixed variants. Prefix identities include `source_dataset`, `id`
and `prefix_index`; repeated variants are not silently discarded. Training
and evaluation inputs remain separate. Only the original `prompt` is passed to
the model; case metadata, expected values and references are not appended.

The source data has no reference solutions or large-input timing tests.
`prepare` explicitly constructs both:

1. Each supplied case becomes a correctness test.
2. Optimization tests cycle through the known cases, by default in batches of
   64 and 256 calls. Arguments are decoded independently for each call. Every
   return value is JSON-encoded immediately, preserving strings, value types
   and snapshots of reused mutable return objects.
3. The starting model samples reference functions. At least two distinct source
   strings must pass every correctness and optimization test. Duplicate samples,
   failures and measurements are logged. If the attempt budget is exhausted,
   the command fails instead of dropping the task or publishing `tasks.json`.

These are **model-generated verified controls, not human references or
independent correctness oracles**. They can implement the same algorithm.
Batched small cases test repeated-call speed and caching, not asymptotic
performance on larger inputs. Incomplete cases can admit incorrect solutions;
add independent cases before interpreting improvements as semantic correctness.

Prepare training and test artifacts separately, using the same fixed starting
model. Example commands run from the repository root:

```bash
DATA=/path/to/forbidden-word-avoidance
STUDY=papers/forbidden-word-optimization-2026

python "$STUDY/run.py" prepare \
  --dataset "$DATA/train.json" --output "$STUDY/runs/train-prepared" \
  --model Qwen/Qwen2.5-7B-Instruct --device cuda \
  --reference-count 2 --reference-attempts 20 \
  --batch-sizes 64 256 --repeats 3 --allow-code-execution

python "$STUDY/run.py" prepare \
  --dataset "$DATA/test.json" --output "$STUDY/runs/test-prepared" \
  --model Qwen/Qwen2.5-7B-Instruct --device cuda \
  --reference-count 2 --reference-attempts 20 \
  --batch-sizes 64 256 --repeats 3 --allow-code-execution
```

The default model matches #39. A source-record/file digest, original file
location, record index and batch sizes are saved in each prepared task.
`config.json`, `model-rank<N>.json` and `reference-attempts-rank<N>.jsonl`
record preparation settings, model revision when available and all candidates.
`tasks.json` contains the prepared tasks and verified source references;
`calibration.jsonl` contains the preparation timings. A new output directory
is mandatory for every command.

## Training, evaluation and replay

```bash
STUDY=papers/forbidden-word-optimization-2026

python "$STUDY/run.py" train \
  --dataset "$STUDY/runs/train-prepared/tasks.json" \
  --output "$STUDY/runs/train" --model Qwen/Qwen2.5-7B-Instruct \
  --device cuda --steps 10000 --group-size 5 --prompts-per-step 1 \
  --environment qar --threshold 0.3 --quality-map binary \
  --reward collapsed --learning-rate 1e-7 \
  --max-tokens 32768 --horizon 32768 --repeats 3 \
  --checkpoint-minutes 20 --allow-code-execution

python "$STUDY/run.py" evaluate \
  --dataset "$STUDY/runs/test-prepared/tasks.json" \
  --output "$STUDY/runs/evaluation" --model Qwen/Qwen2.5-7B-Instruct \
  --adapter "$STUDY/runs/train/adapter" --device cuda \
  --samples 20 --pass-k 1 10 --repeats 3 --allow-code-execution

python "$STUDY/run.py" replay \
  --dataset "$STUDY/runs/test-prepared/tasks.json" \
  --calibration "$STUDY/runs/evaluation/calibration.jsonl" \
  --records "$STUDY/runs/evaluation/evaluate.jsonl" \
  --environment qar --threshold 0.5 --reward two-gate --quality-map linear \
  --output "$STUDY/runs/replay"
```

Omit `--adapter` for baseline evaluation. `--device cuda` uses all idle visible
GPUs; `--device cuda:N` explicitly selects one GPU, and `--device cpu` permits
small tests. Each verification worker is assigned a distinct physical CPU core.
The same reference sources are remeasured on that core during training and
evaluation, rather than importing another machine's timing matrix.

Each worker imports a candidate once into a registered `__solution__` module;
`if __name__ == "__main__"` blocks do not run. Generated `test_*` functions are
not automatically invoked. Private cases call the named function directly.
Timings include the batch's function calls and per-return JSON encoding;
interpreter startup, source compilation, module setup, input parsing, output
file writes and worker teardown are excluded. Module setup has a separate
10-second wall cap and shares the worker's CPU/memory limits. Non-JSON or
nonfinite returns, exceptions and mismatches are explicit failures.

Optimization timings remain sensitive to scheduling, CPU contention and very
short batches. Faster generation or shorter source does not itself earn speed
credit. With two references, evaluation QP is only 0, 0.5 or 1, so p30 and p10
select the same candidates; neither means "30% faster".

Training saves per-response rewards and timing traces, periodic
`checkpoint-<updates>/` adapters, and a final `adapter/`. All-equal rewards skip
the optimizer update. Evaluation produces strict correctness and speed-gated
pass@k plus uncensored batch-time summaries. Trace replay requires both the
task digest and exact calibration digest recorded in each response. Changed
cases, references or batch sizes require new artifacts.

`calibrate` and `simulate` are also available. The simulator samples progressively
faster verified references, including two-reference pools; it is not a forecast
of model learning. Do not select checkpoints on the held-out test split.

## Offline validation

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=. python -m pytest -q unittests/papers/test_forbidden_word_optimization.py
```

Tests execute real function workers and a tiny locally generated Qwen3 model:
reference sampling/rejection, source preparation, a real LoRA optimizer update,
checkpoint save/reload, evaluation and trace replay. Model-dependent tests skip
when optional Torch/PEFT dependencies are missing. No model is downloaded and
no external dataset path is needed in CI. No long training run or benchmark
gain is claimed by this implementation.
