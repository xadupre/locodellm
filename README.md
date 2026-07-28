<p align="center">
  <img src="docs/_static/logo.svg" alt="locodellm logo" width="128" height="128">
</p>

# locodellm

Experimentation around local LLM.

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

#### `generate`

```python
from locodellm.generate import generate
```

Calls a local LLM loaded from an ONNX model directory (compatible with
[onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai)) and
returns a `SessionState`.

```python
session = generate(
    prompt="Once upon a time",
    model="path/to/model",          # model directory, og.Model, or SessionState
    providers=None,                  # e.g. ["CUDAExecutionProvider"]
    max_length=200,                  # max tokens (prompt + generated)
    temperature=0.7,                 # extra search options
)
print(session.text)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `prompt` | `str` | The text prompt to send to the model. |
| `model` | `str \| SessionState \| og.Model` | A path to the model directory, an already-loaded `onnxruntime_genai.Model`, or a `SessionState` from a previous call to continue the conversation. |
| `providers` | `list[str] \| None` | Ordered list of execution providers. Ignored when `model` is not a path. |
| `max_length` | `int` | Maximum number of tokens to generate (including all tokens across turns). |
| `**search_options` | | Extra options forwarded to `GeneratorParams.set_search_options` (e.g. `temperature`, `top_k`, `top_p`). |

**Returns:** A `SessionState` instance.

#### Multi-turn conversations

Pass the returned `SessionState` back as the `model` argument to continue
the conversation with full context:

```python
session = generate("Hello, who are you?", "path/to/model")
session = generate("Tell me more about that.", session, max_length=500)
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
session = generate("<s>", model_path, max_length=10)
```

## Development

```bash
pip install -e ".[dev]"
pytest unittests
black . && ruff check .
```

