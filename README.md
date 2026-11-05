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

Run a prompt with up to 100 Python script executions and export conversation
and execution events:

```bash
python -m locodellm generate path/to/model \
  "Write 42 to value.txt, read it back, and report its contents." \
  --chat-template chatml --harness python --max-tool-calls 100 \
  --output events.json
```

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

The JSON input enables the Python harness. Prompts run in order, each in an
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
events and diff. The last prompt's changes remain on disk. No clone or commit
is performed. Scripts run with your permissions, without a sandbox.

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
