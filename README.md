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

### ATen failure-discovery datasets

`datasets/aten_failures_set_1.jsonl` and `datasets/aten_failures_set_2.jsonl`
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

Only send `prompt` to the model; keep `expected_answer` and issue links with the
evaluator. These JSONL files are evaluation data, not executable task files.
Load one JSON object per line and extract only its `prompt` when preparing
model inputs.

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
