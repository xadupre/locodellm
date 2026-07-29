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

## Development

```bash
pip install -e ".[dev]"
pytest unittests
black . && ruff check .
```

