<p align="center">
  <img src="docs/_static/logo.svg" alt="locodellm logo" width="128" height="128">
</p>

# locodellm

[![CI](https://github.com/xadupre/locodellm/actions/workflows/ci.yml/badge.svg)](https://github.com/xadupre/locodellm/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/xadupre/locodellm/branch/main/graph/badge.svg)](https://codecov.io/gh/xadupre/locodellm)

**locodellm** is a small toolkit for experimenting with local large
language models. It wraps
[onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai) behind a
simple session API to load an ONNX model directory and generate text,
supports multi-turn conversations, and ships helpers to build tiny dummy
models for testing. A minimal command line interface is also provided.

## Install

```bash
pip install -e .
```

## Usage

### Command line

```bash
python -m locodellm version
```

Run a prompt with up to 100 Python or Bash script executions and export conversation
and execution events:

```bash
python -m locodellm generate path/to/model \
  "Write 42 to value.txt, read it back, and report its contents." \
  --chat-template chatml --harness python --max-tool-calls 100 \
  --output events.json
```

`generate --temperature 0.7` enables sampling with either backend, including
all harness turns. The default, `--temperature 0`, uses greedy decoding.
Temperatures must be finite and nonnegative. The effective sampling options
are recorded in the exported conversation events.

With `--harness`, each problem gets six minutes to explore, followed by a
request to write the final answer without further tool calls. At eight
minutes, generation stops for that problem, the command exports its partial
trace and repository diff, and the batch continues with the next problem.
The result has `timed_out: true` and an explicit `time_limit` event with the
elapsed time and limit.
Malformed or unsupported tool-call formats receive up to two requests to
reissue the call in the allowed format. No commands from a rejected response
are executed, and each rejection is recorded as `tool_call_rejected`.
If correction fails, the result has `protocol_error: true` and the batch
continues. These retries do not reset the time budget. An empty final answer,
including reasoning without a final response, is recorded with `empty_answer: true`
and an explicit error; its trace and diff are saved before the next problem runs.
Other model failures, such as requesting tools after finalization or exceeding
the tool budget, are also recorded without stopping the batch. Loading,
repository and infrastructure exceptions still propagate.
The budget starts after model
loading and repository cleanup, and resets for each problem. Generation
stops at token boundaries; script timeouts are capped by the remaining budget.

For changes in an existing local repository, create `tasks.json`:

```json
{
  "repo": "../my-project",
  "prompts": [
    "Update the README to explain the installation steps.",
    "Add a test for the behavior described in the README."
  ]
}
```

```bash
python -m locodellm generate path/to/model --input tasks.json \
  --chat-template chatml --max-tool-calls 100 --max-length 1024 \
  --output events.json
```

The JSON input enables the Python/Bash harness. Prompts run in order, each in an
independent conversation with its own script budget, against the same directory.
Relative `repo` paths are resolved from the JSON file's directory and must point
to the root of an existing Git working tree with a commit. Keep `tasks.json` and
`events.json` outside that working tree.

**Destructive cleanup:** before every prompt, the harness resets tracked files
to the run's starting revision and removes nonignored untracked files and
directories. This also discards any initial local modifications. Ignored files
are preserved. Changes persist between Python calls within a prompt, but are
cleared before the next prompt.

Output contains the resolved `repo`, `base_revision`, and a `results` list with
each prompt's events, final answer, and `diff`, saved before the next cleanup.
The patch includes modified, deleted, staged, and new nonignored files, including
binary changes. Execution stops at the first harness error after exporting its
events and diff. The last prompt's changes remain on disk. JSON batches use
the existing repository without cloning; no commit is performed.
Scripts run with your permissions, without a sandbox.

For native Transformers weights instead of ONNX, add ``--backend pytorch``
and choose a device:

```bash
python -m locodellm generate Qwen/Qwen3.5-35B-A3B-FP8 \
  --backend pytorch --device auto --input tasks.json \
  --max-tool-calls 100 --max-length 1024 --output events.json --verbose 1
```

This requires the optional `pytorch` dependencies and compatible FP8 hardware.
Use a Hugging Face model id (missing files will be downloaded), or a complete
local checkpoint directory, not the `models--OWNER--NAME` cache root.

With `--verbose 1`, progress is flushed immediately to stderr: model loading
stages, prompt and model-turn numbers, Python script execution, diff capture,
and saving. PyTorch generation also reports first-token waiting, periodic token
counts, and completion, rather than remaining silent until the answer is ready.

The model can request shell commands with `<bash>...</bash>` as well as Python
with `<python>...</python>`, outside Markdown fences. For example:

```xml
<bash>git status --short; find . -name '*.py' -print</bash>
```

Both tools share the script budget and working directory. Bash runs with
`pipefail` enabled; outputs and exit status are sent back to the model and saved
with `tool: "bash"` in the JSON trace. Bash must be available on PATH.

To expose only the Bash tool, use `--harness bash`, including with JSON input:

```bash
python -m locodellm generate Qwen/Qwen3.5-0.8B \
  --backend pytorch --device cuda --input tasks.json --harness bash \
  --max-tool-calls 100 --max-length 4096 --output events.json --verbose 1
```

The initial instructions advertise only Bash. Python tool requests are rejected;
this is not isolation, since Bash commands can still launch Python.

With PyTorch and a native Qwen3-Coder-compatible chat template (including
Nemotron), the harness automatically declares the allowed functions and uses
native `<tool_call>` requests instead of advertising custom script tags:

```xml
<tool_call>
<function=bash>
<parameter=command>git status --short</parameter>
</function>
</tool_call>
```

Bash takes the required string parameter `command`; Python takes `code`.
Execution results are returned as native tool messages and recorded in the trace.
Unknown functions, extra or duplicate parameters, incomplete calls, and empty
scripts are rejected without executing any command from that turn. Existing
`<bash>`/`<python>` requests remain supported. Explicit `--chat-template chatml`
keeps the existing tag-based prompting instead of native tool declarations.
For Nemotron, omit that option and use `--temperature 1` to follow NVIDIA's
recommended sampling; the checkpoint supplies `top_p=0.95`.

The parser accepts multiple protocols in the same response, independently of
`--chat-template`: script tags, native XML, wrapped `<tool_call><bash>...`
requests, and Qwen JSON:

```xml
<tool_call>{"name":"bash","arguments":{"command":"git status --short"}}</tool_call>
```

OpenAI-style `function` objects inside `<tool_call>` are also supported, including
JSON-encoded `arguments`. A bounded compatibility parser handles the legacy Bash
envelopes observed with Qwen, such as
`<tool_call><function=bash><parameter=pwd</parameter></function></tool_call>`.
It extracts the enclosed command verbatim, without reconstructing missing commands
or accepting unknown parameters. Unsupported envelopes and ambiguous requests are
rejected atomically, including duplicate JSON keys. Markdown examples and reasoning
are never executed. All protocols share the existing tool budget, time limits and
Bash-only restrictions; `tool_protocols` in model-response events records the
recognized formats.

Add `--live-log` to create a retained temporary transcript, or choose its path:

```bash
python -m locodellm generate Qwen/Qwen3.5-0.8B \
  --backend pytorch --device cuda --input tasks.json --harness bash \
  --max-tool-calls 100 --max-length 4096 --output events.json \
  --live-log live.log --verbose 1
```

In another terminal, `tail -f live.log` shows model text as it is generated and
tool stdout/stderr while scripts run, followed by their exit status. Logging
flushes immediately and also works without verbose. Keep this file outside the
target repository. Temporary logs are retained after the run for inspection.

### ATen failure-discovery datasets

`datasets/yobx/aten_failures_set_1.jsonl` and `datasets/yobx/aten_failures_set_2.jsonl`
contain 8 converter-discovery tasks each, covering 16 ATen
functions in `xadupre/yet-another-onnx-builder`. Related dispatchers and helpers
stay together; the sets share no functions, source issues, or reference examples.

Each JSONL record contains `function`, an English discovery `prompt`, and an
`expected_answer` with executable module-export examples, observed failures,
passing controls, and issue links. Every example converts a real
`torch.nn.Module` using `to_onnx(module, inputs)`; none calls a converter or
constructs a graph builder directly. Run its `setup_code` followed by one
example's `code` at a time. Export modes and actual converter routing are
recorded: some failures require `ExportOptions(tracing=True)` and pass with
default export. Invalid `tensor_split(sections=0)` is excluded because both
PyTorch and public export correctly reject it.

Verification records the source revision, dependency versions, and relevant
pre-existing source edits; the recorded embedding-bag value mismatches require
that edit, while unmodified sources reject the same inputs during conversion.

Both datasets record the current `xadupre/yet-another-onnx-builder` commit,
`2156c18e935b6ded3f73ba6e59c64f7f0f1119d9`, in every record's
`verification.revision`. To replay the problems at that revision without
changing an existing working tree, use a fresh clone:

```bash
git clone https://github.com/xadupre/yet-another-onnx-builder.git yobx-replay
git -C yobx-replay checkout --detach 2156c18e935b6ded3f73ba6e59c64f7f0f1119d9
```

Use the dependency versions in `verification.environment` and account for
`verification.relevant_preexisting_source_changes` before running the examples;
the commit alone does not reproduce the embedding-bag value mismatches.

Only send `prompt` to the model; keep `expected_answer` and issue links with the
evaluator. `generate` accepts these datasets directly and clones their
`repository` at `verification.revision` before loading the model:

```bash
python -m locodellm generate Qwen/Qwen3.8-27B \
  --input datasets/yobx/aten_failures_set_2.jsonl --clone-dir ./clones \
  --backend pytorch --device cuda --chat-template chatml --harness bash \
  --max-tool-calls 100 --max-length 16000 --temperature 0 \
  --output events.json --live-log live.log --verbose 1
```

`--clone-dir` defaults to `./clones`. Each run creates a fresh `run-*`
subdirectory, retained afterwards; it never reuses or cleans an existing
working tree. The resolved clone path and starting commit are recorded in the
output JSON. Records must share the same repository and revision. When no
revision is recorded, the clone uses the repository's default branch.

Only each record's `prompt` is sent to the model; reference answers and other
dataset metadata are not included. Blank lines are ignored. To use an existing
repository instead, pass `--repo PATH` instead of `--clone-dir`. Relative paths
are resolved from the current directory. JSON input keeps its existing
`{"repo": "...", "prompts": [...]}` format.
The selected repository is reset and nonignored untracked files are removed
before every prompt. Keep the dataset, output, and live log outside that
repository. No dependencies are installed in the clone.

### Python API

#### `create_session` and `session.generate`

```python
from locodellm.session import create_session
```

Creates a session from an ONNX model directory (compatible with
[onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai)) and
generates text.

```python
session = create_session(
    "path/to/model",                # model directory or og.Model
    providers=None,                  # e.g. ["CUDAExecutionProvider"]
)
session.generate(
    "Once upon a time",
    max_length=200,                  # max tokens (prompt + generated)
    temperature=0.7,                 # extra search options
)
print(session.text)
```

**`create_session` parameters:**

| Name | Type | Description |
|------|------|-------------|
| `model` | `str \| og.Model` | A path to the model directory or an already-loaded `onnxruntime_genai.Model`. |
| `providers` | `list[str] \| None` | Ordered list of execution providers. Ignored when `model` is not a path. |
| `verbose` | `int` | Verbosity level (0 = silent). |
| `chat_template` | `str \| None` | Chat template to wrap prompts (e.g. `"chatml"`). |

**`session.generate` parameters:**

| Name | Type | Description |
|------|------|-------------|
| `prompt` | `str` | The text prompt to send to the model. |
| `max_length` | `int` | Maximum number of tokens (including all tokens across turns). |
| `**search_options` | | Extra options forwarded to `GeneratorParams.set_search_options` (e.g. `temperature`, `top_k`, `top_p`). |

**Returns:** The `SessionState` instance (for chaining).

#### Multi-turn conversations

Call `session.generate` multiple times to continue the conversation with
full context:

```python
session = create_session("path/to/model")
session.generate("Hello, who are you?", max_length=200)
session.generate("Tell me more about that.", max_length=500)
print(session.text)  # only the latest turn's output
```

#### `SessionState`

```python
from locodellm.session import SessionState
```

Holds the state of a generation session.

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | `og.Model` | The loaded onnxruntime-genai model. |
| `tokenizer` | `og.Tokenizer` | The tokenizer bound to the model. |
| `tokens` | `np.ndarray` | All token ids accumulated so far (prompt + generated). |
| `text` | `str` | The text generated during the last `generate` call. |

#### `create_tiny_model`

```python
from locodellm.test_models import create_tiny_model
```

Creates a tiny dummy LLM model directory for testing. The model has
zero-initialized weights and produces meaningless output, but is valid
enough for onnxruntime-genai to load and run.

```python
model_path = create_tiny_model("/tmp/my-tiny-llm")
session = create_session(model_path)
session.generate("<s>", max_length=10)
```

## Benchmarks

### Multi-turn SLM-GRPO training on ATen discovery tasks

The `train` command applies the SLM-GRPO paper's LoRA/GRPO protocol to the
ATen JSONL datasets. Install the optional dependencies with
`python -m pip install ".[train]"` and prepare the target repository's dependencies
using `verification.environment` in the dataset. Models may be local checkpoints.

**Safety:** training executes arbitrary model-generated Python/Bash, including
independent verification. The subprocesses and fresh Git clones are **not security
sandboxes**. Run only inside a disposable, unprivileged container/VM without
secrets or sensitive mounts; prepare dependencies and model weights beforehand.
`--allow-code-execution` is mandatory. The initial automatic repository clone
requires network access; use `--repo` with a clean local checkout at the recorded
revision for offline training.

```bash
python -m locodellm train Qwen/Qwen3-0.6B \
  --protocol slm-grpo \
  --input datasets/yobx/aten_failures_set_1.jsonl \
  --output runs/aten-slm-grpo \
  --device cuda --epochs 2 --group-size 4 \
  --harness bash --max-tool-calls 10 --max-new-tokens 2048 \
  --allow-code-execution
```

Each group contains independent **multi-turn trajectories for the same task**.
Every trajectory starts a fresh conversation and repository clone. Tool results
become subsequent prompts; one final `export_failure_reward` is shared by all
assistant actions in that trajectory. The loss is averaged over all its generated
tokens, then over trajectories, **not over turns**. Prompt, tool-result and template
tokens are context only. Exact sampled token ids (including EOS) and frozen
old-policy probabilities are retained per turn; subsequent updates never
reconstruct actions by re-tokenizing text.

Defaults follow SLM-GRPO: group size 8, 5 epochs, learning rate `1e-5`,
temperature `0.9`, LoRA rank 32/alpha 64, one update per group, no KL term,
and centered but unnormalized advantages. `--normalize-advantages` enables
standard-deviation scaling. `--updates-per-group` reuses the same frozen old
policy probabilities for clipped GRPO updates.

The final answer must include a complete runnable script in Python Markdown
fences. The verifier instruments real `to_onnx(module, inputs, ...)` calls,
executes the reference module, validates `target_opset=18, optimize=False`,
and checks ONNX metadata and CPU ONNX Runtime outputs (`rtol=atol=2e-3`).
Use explicit `input_names` and a tuple of tensor inputs for runtime comparison;
nested or ambiguous input mappings are marked unverified, not failures.
Converter exceptions count only when the traceback identifies an actual ATen
converter. Invalid reference inputs and unrelated script exceptions do not count
as export failures. Failure/reproduction credit also requires reaching the task's
converter or a same-family helper route listed in its reference metadata.
Unrelated converters and unrecognized tracing bypasses cannot earn this credit.
Conversion routes and tracing mode are recorded.

Two fresh-process replays in separate clones of the recorded revision supply
the four factual criteria; the best verified example receives the additive
reward described below. Extraction alone gets the existing 0.05 floor.
Source modifications during exploration or an incomplete/erroring tool conversation
receive zero. Source modifications during replay invalidate all verified criteria.
This is **not** semantic reference matching: neither `expected_answer` nor issue
links are sent to the solving policy. Pre-existing patches recorded in reference
metadata are not applied; only failures reproduced on the pinned clean sources
can earn reproduction credit. Generated code is not an adversarially secure
verification boundary; isolation remains the caller's responsibility.

`--output` must be empty or absent. It receives `config.json`, `train.jsonl`
(all prompts, actions, tool responses, probabilities and verification diagnostics),
retained `repositories/run-*` clones, and the final `adapter/`. Checkpoints are
published every 10 minutes at an update boundary; `--checkpoint-minutes` and
`--checkpoint-steps` adjust this. `--adapter PATH` warm-starts saved LoRA weights,
not optimizer/RNG state. Use set 2 separately for held-out evaluation.

### Export-failure reward

`export_failure_reward` scores one clearly extracted example using four
independently verified boolean criteria. Each satisfied criterion adds `0.25`;
the score is maximized and ranges from `0` to `1`, without penalties or gating.
Setting `answer_extractable=True` grants a floor of `0.05` for a final answer
containing a clearly extractable code example, even if unverified. It is not an
extra bonus: `reward = max(0.05, 0.25 * satisfied_criteria)` in that case.
Rejected tool requests without such a final answer receive no extraction credit.

```python
from locodellm.bench import export_failure_reward

reward = export_failure_reward(
    reference_valid=True,       # PyTorch runs on the concrete inputs
    export_call_valid=True,     # to_onnx follows the required protocol
    failure_valid=True,         # an admissible failure is actually observed
    reproduction_valid=False,  # independent replay, route and stage not yet verified
    answer_extractable=True,   # final answer contains a clearly extractable code example
)
assert reward == 0.75
```

The function only computes the reward. Extracting examples and verifying these
criteria must happen separately; model assertions or a successful harness exit
are not verification. Non-boolean criteria are rejected.

### Semantic answer judging

The deterministic factual reward remains the default. Semantic judging is an
opt-in diagnostic and is not included in that reward.

`judge_export_answer` compares a generated report with the `expected_answer`
object of a failure-discovery dataset record. It returns the best semantic match
to one reference example, independently of the factual reward:

```python
from locodellm.bench import export_failure_reward, judge_export_answer

factual = export_failure_reward(
    reference_valid=True,
    export_call_valid=True,
    failure_valid=False,
    reproduction_valid=False,
)
semantic = judge_export_answer(
    session, generated_answer, record["expected_answer"], task=record["prompt"]
)
print(factual, semantic.similarity, semantic.closest_reference, semantic.reason)
```

The judge uses a fresh PyTorch conversation sharing the loaded weights, without
the solving history or tools. Native thinking is disabled and decoding is greedy.
It compares causes, triggering conditions, observations, conversion routes and
failure stages, not prose similarity. References are evaluator-only data.
One format-correction turn is allowed within the same time budget by default
(`max_format_retries=0` disables it). Judgments that remain malformed, out-of-range
or refer to unknown examples raise `JudgeProtocolError` with the raw completion
attached; they never become a default numeric score.
Schema-valid self-judgments can still contradict the candidate or references.
They require calibration before use as training rewards and do not replace
independent execution checks.

### Built-in benchmark

Run the built-in `basic` benchmark against a local ONNX model:

```bash
python -m locodellm bench path/to/model basic --chat-template chatml
```

Use `python -m locodellm benchmarks` to list the available built-in and LM-Eval
benchmarks.

### LM Evaluation Harness

Install the optional dependency and run a generation-based
[LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
task against an ONNX Runtime GenAI model:

```bash
pip install ".[eval]"
python -m locodellm bench path/to/model gsm8k --limit 10
```

The adapter supports LM-Eval tasks that use `generate_until`. Likelihood and
perplexity tasks are not supported by the ONNX Runtime GenAI generation API.

## Development

```bash
pip install -e ".[dev]"
pytest unittests
black . && ruff check .
```
