# While without `while`: stabilized GRPO with the existing evaluator

This independent study applies the stabilized GRPO algorithm used in PR #39 to
the original **forbidden-word-avoidance cost**, not the function-speed objective
in PR #40. It does not import either study. Its objective, rollout/update loop,
LoRA settings, checkpoint publication and GPU selection were copied from
`papers/code-optimization-2026` at commit `06edeb2` and adapted here.

The evaluator already exists. This study **does not implement another cost**:

```python
reward = -float(response_cost(raw_completion, *reference_from_question(question)))
```

The original standalone evaluator is included in **`while_grpo/cost.py`** and
used automatically. Its `response_cost` and `reference_from_question` behavior
is preserved. **No other repository or evaluator path is required.** Only the
dataset remains an explicit local input.

Each run saves the evaluator as `evaluator.py` and records its SHA-256, local
source path, dataset hash and study implementation hash. An optional
`--evaluator` override can select a different trusted local cost module, but is
not needed for this experiment.

## What is graded

The existing evaluator receives the **raw decoded completion**. There is no
code-block extraction, reasoning removal, AST-based replacement of the lexical
rule, additional occurrence penalty or new correctness comparator.

For the bundled original evaluator:

- The case-sensitive substring `"while"` anywhere in the response adds one:
  comments, strings and `meanwhile` count; multiple occurrences still add one.
- Syntax errors, lack of actual work, missing top-level `test_*` functions and
  failing executable self-tests retain their original penalties.
- The entire response executes as `__main__`. Supplied `function` and `cases`
  are passed through to the original target-function checks.
- Return values retain the evaluator's JSON round-trip and Python equality
  semantics, including its treatment of booleans versus integers.

A cost of zero therefore means all of **that evaluator's** checks passed. It is
not proof of correctness beyond the supplied cases, or proof that self-tests are
meaningful. If a different evaluator version is selected, its own behavior wins;
the wrapper does not attempt to reproduce or amend these rules.

The local files `train.json` and `test.json` contain **JSON Lines**, despite
their extension. Each record must have `id`, `prompt`, `function` and nonempty
`cases`. Prefixed variants use `(source_dataset, id, prefix_index)` identities.
The loader checks the complete file even with `--limit`, rejects duplicate
identities and preserves all original prompt and grading fields. Only `prompt`
is rendered into the model's user message; no private cases are appended.
Thinking is disabled through the tokenizer's existing chat template.

## Algorithm

The cost reward replaces speed calibration; no generated reference programs,
timing matrices or optimization workloads are needed.

For completion rewards `r_i` and generated lengths `L_i`, the group baseline is
`sum(r_i * L_i) / sum(L_i)`. Advantages are centered without standard-deviation
normalization. The clipped token surrogates are summed and divided by a **fixed
horizon**, rather than a realized completion length or the number of groups.
Old probabilities and advantages are detached. DDP loss is multiplied by world
size to recover the global gradient sum after DDP averaging. All-equal rewards
skip the optimizer update.

Rollouts use evaluation mode with no gradients; updates use training mode with
non-reentrant gradient checkpointing and disabled dropout. Sampling uses the
same temperature as recomputed token probabilities, with neutral top-k, top-p
and repetition penalty. LoRA uses rank 32, alpha 64 and zero adapter dropout.
Gradient norm is clipped to one; AdamW has zero weight decay. Checkpoint numbers
count optimizer updates, not sampled iterations.

The default model is the original task's `Qwen/Qwen3.5-4B`; its text-only
conditional-generation forward is supported, as are causal language models such
as Qwen2.5/Qwen3. Set `--model` to an existing local checkpoint for offline work.
Model loading may download weights unless an offline local model is selected.

## Running

Requires Linux/POSIX and Python 3.12+, with PyTorch 2.6+, Transformers
`>=5.2,<6`, PEFT `>=0.18,<1` and tqdm. Use the existing Python environment;
the scripts do not require installing this repository.

**Generated code is not safe to run on a shared host.** Use a disposable,
offline VM/container with no credentials, sensitive mounts or host access.
`--allow-code-execution` acknowledges this requirement; it is not a sandbox.
The evaluator itself is trusted code and is imported even during a dry run.

From the repository root:

```bash
DATA=/path/to/forbidden-word-avoidance
STUDY=papers/while-without-while-2026

# Validates dataset/evaluator/provenance, without loading a model or creating output.
python "$STUDY/run.py" train \
  --dataset "$DATA/train.json" \
  --output "$STUDY/runs/input-check" --dry-run

# After entering the disposable execution environment:
python "$STUDY/run.py" train \
  --dataset "$DATA/train.json" \
  --output "$STUDY/runs/train" --device cuda \
  --steps 100 --group-size 5 --max-tokens 1024 --horizon 1024 \
  --allow-code-execution

python "$STUDY/run.py" evaluate \
  --dataset "$DATA/test.json" \
  --output "$STUDY/runs/evaluate" --device cuda \
  --adapter "$STUDY/runs/train/adapter" \
  --samples 20 --pass-k 1 10 --max-tokens 1024 --horizon 1024 \
  --allow-code-execution
```

Omit `--adapter` for base-model evaluation. Use separate output directories and
the same test data, evaluator, sampling parameters and seeds for comparisons.
`--device cuda` selects every idle visible GPU; `cuda:N` explicitly selects one,
and `cpu` runs small local models. GPU discovery failures are errors rather than
permission to use a busy GPU.

Each grading call starts a fresh process, with a five-second wall-clock deadline
by default (`--timeout`), a 2 GiB virtual-address limit and file/descriptor limits.
The process group is killed after grading, including on timeout. These limits
bound ordinary execution but **do not contain malicious Python** or prevent
processes from deliberately escaping a process group. Restrictive limits may
also make otherwise valid code fail.

Grading timeout, crash, missing output and invalid grader results are
**infrastructure failures**, not numeric costs. A failed verdict is flushed to
the rank trace, then the run aborts without publishing completed metrics. Such
failures are never replaced with a made-up negative reward.

Training saves raw completions/costs in `train-rank*.jsonl`, complete adapters in
`adapter/` and optional `checkpoint-N/`, and completed update counts in
`training-metrics.json`. These are adapter checkpoints, not full optimizer/RNG
resume states; `--adapter` starts a new optimizer. Evaluation produces
`evaluate.jsonl` and `metrics.json`, with mean cost/reward and unbiased
**zero-cost pass@k**. Existing output directories are never overwritten.

## Validation and limitations

```bash
# Checks the bundled original cost, subprocesses and real tiny models, without downloads.
PYTHONPATH=. CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 \
  python -m pytest -q unittests/papers/test_while_without_while.py

# Also validates the supplied local data.
FORBIDDEN_WORD_DATASET="$DATA" \
  PYTHONPATH=. CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 \
  python -m pytest -q unittests/papers/test_while_without_while.py
```

CI directly checks the bundled original cost: raw fences, substring/case
sensitivity, self-tests, expected outputs, JSON equality, pytest failures and
timeouts. The offline training/checkpoint/base-and-adapter evaluation cycle also
uses the bundled scorer with no evaluator argument. An explicit small known-cost
fixture separately checks the caller contract and worker failures. Only the
optional external-dataset validation skips when no dataset path is supplied.
Tiny Qwen3 and attention-only Qwen3.5 checkpoints exercise policy loading,
sampling, probability recomputation and backward passes. The latter
does not validate hybrid linear-attention kernels; CUDA-only optional kernel
packages can prevent running an actual hybrid Qwen3.5 model on CPU.

No full-size Qwen3.5 training, multi-GPU run or benchmark result is claimed.
The original grading semantics and supplied data remain unchanged.
