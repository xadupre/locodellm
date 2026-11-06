generate
========

Generates text from a prompt using a local LLM. The model can be a mock
model id, a local path to an ONNX model directory, or a HuggingFace
repository id (which is automatically downloaded and converted).
With ``--backend pytorch``, Transformers weights are loaded directly without
ONNX conversion.

Usage
-----

.. code-block:: bash

    python -m locodellm generate MODEL PROMPT [OPTIONS]
    python -m locodellm generate MODEL --input tasks.json [OPTIONS]

Options
-------

``MODEL``
    Model id or path. Use ``mock/generate`` for the mock model, a local
    directory path, or a HuggingFace id like
    ``Qwen/Qwen2.5-Coder-0.5B-Instruct``.

``PROMPT``
    The prompt text to send to the model. Mutually exclusive with ``--input``.

``--input``
    Reads a JSON object with exactly two fields: ``prompts``, a nonempty list
    of nonempty strings, and ``repo``, the root of an existing local Git
    working tree with a commit.
    Relative repository paths are resolved from the input file's directory.
    Enables the Python harness automatically. Prompts execute sequentially
    in independent conversations against the same directory, with a separate
    script budget per prompt. Before every prompt, tracked files are reset to
    the run's starting revision and nonignored untracked files and directories
    are removed, including modifications present before the run.
    Ignored files are preserved. Changes persist between Python calls within
    a prompt, but not between prompts. The last prompt's changes remain on disk.
    No repository is cloned or committed. Input and output JSON files must be
    outside the target working tree so cleanup cannot delete them.

``--precision``
    Precision qualifier for ONNX conversion (e.g. ``fp32``, ``fp16``, ``int4``).
    With PyTorch, loading dtype: ``auto``, ``fp32``, ``fp16``, or ``bf16``.
    Pre-quantized checkpoints retain their own quantization configuration.

``--backend``
    Generation backend: ``onnx`` (default) or ``pytorch``. The PyTorch backend
    requires the optional ``pytorch`` dependencies.
    Generation stops at either the model's configured EOS tokens or the
    tokenizer's EOS token, including chat turn endings when they differ.

``--device``
    Device for the PyTorch backend: ``cpu`` (default), ``cuda``, ``cuda:0``,
    or ``auto`` for automatic placement. Requires ``--backend pytorch``.

``--provider``
    Execution provider (e.g. ``CUDAExecutionProvider``).

``--max-length``
    Maximum token length for generation (default: 200). With ``--harness``,
    maximum new tokens per model turn (default: 1024).

``--harness python``
    Executes model-requested Python and Bash scripts and returns their output to the model.
    The historical option name ``python`` enables both tools.
    Requests use ``<python>...</python>`` or ``<bash>...</bash>`` outside thinking
    and Markdown fences. Mixed requests execute sequentially in message order,
    share a working directory and budget, and return their results together.
    Bash runs with ``pipefail`` enabled, without profile or rc files, and requires
    ``bash`` on PATH. Shell variables do not persist between executions.
    Scripts can read and write files. Each script starts a fresh process
    in a temporary directory shared across this prompt's calls and deleted at
    the end, unless ``--input`` selects a persistent repository directory.
    Internal harness scripts and counters always remain in temporary storage.
    Execution uses the current user's permissions: this is not a sandbox.

``--harness bash``
    Enables only Bash tool requests. The initial prompt advertises only
    ``<bash>...</bash>`` calls; Python tool requests are rejected before execution,
    including entire mixed batches. This explicit choice is preserved with
    ``--input``. It selects a tool protocol, not a sandbox: Bash can still
    launch installed programs, including Python.

``--max-tool-calls``
    Maximum combined number of Python and Bash scripts (default: 10).
    Each script in a grouped request consumes one execution.
    A request must include its closing tool tag before it can execute.
    An incomplete request, including one preceded by explanatory prose, is
    reported as a harness error rather than accepted as a final answer.

``--output, -o``
    Writes a JSON document containing ``initial_prompt``, ordered ``events``,
    ``initial_generation_options``, the final ``text``, ``token_count``,
    ``tool_call_count``, ``tool_calls``, and ``error``. Events include model
    responses, subsequent prompts, executed commands, stdout, stderr, and exit
    status. The file is written when the conversation ends, including on harness
    failure; a harness failure then raises an error instead of reporting success.
    With ``--input``, the document contains ``repo``, ``base_revision``, and a
    ``results`` list of per-prompt records, including the original ``prompt``
    and a ``diff`` patch against the starting revision. The patch captures
    tracked modifications and deletions, staged changes, and new nonignored
    files, including binary patches, without altering the real Git index.
    Records are saved after every prompt, before any subsequent cleanup.
    Execution stops at the first harness error, retaining the completed
    records, the failing record, and its diff; subsequent prompts are not executed.
    Each executed tool record and command event includes ``tool`` (``python``
    or ``bash``). Bash scripts contribute zero to Python function-call counts.

``--chat-template``
    Chat template to use (e.g. ``chatml``).

``--live-log [PATH]``
    Appends a human-readable live transcript to PATH. Without a path, creates a
    temporary ``locodellm-live-*.log`` file and prints its full path on stderr.
    The file is retained after completion or failure and can be followed with
    ``tail -f PATH``. Prompts, streamed model text, executed scripts, tool stdout,
    stderr, exit status, and harness errors are flushed as they become available,
    independently of ``--verbose``. Tool output is polled every 50 ms; a tool must
    flush its own output to make it visible immediately.
    Transformers text is emitted in decoded fragments, not necessarily every token.
    The path is included as ``live_log`` in saved results. Keep it outside the
    target repository and separate from the input/output JSON files.

``--verbose, -v``
    Verbosity level (default: 0). Level 1 immediately flushes progress messages
    to stderr: startup, input validation, loading, prompt number, repository
    reset, model turns, script execution and exit status, diff capture, and saving.
    The PyTorch backend reports configuration, weight and tokenizer loading,
    then first-token waiting, periodic generated-token counts, and completion.
    Each periodic token-count line includes ``text=...`` with the latest
    30 decoded characters, so the displayed text advances during generation.
    During PyTorch decoding, a response-start preview is flushed as soon as
    30 characters are available, without waiting for generation to finish.
    After each model response, the first 30 characters are also printed as a
    completed-response preview. Line breaks are escaped to keep previews on one line.
    Token counts appear during decoding; they do not indicate progress inside
    model loading or before the first token. The final answer remains on stdout.

Examples
--------

Using the mock model:

.. code-block:: bash

    python -m locodellm generate mock/generate \
        'write a python function which returns "hello"' --chat-template chatml

Output:

.. code-block:: text

    Here's a simple Python function that returns "hello":

    ```python
    def hello():
        return "hello"
    ```

    You can call this function and it will return the string "hello".

Using a HuggingFace model (automatically downloaded and converted):

.. code-block:: bash

    python -m locodellm generate Qwen/Qwen2.5-Coder-0.5B-Instruct \
        'write a python function which returns "hello"' \
        --chat-template chatml --precision fp32 --verbose 1

Running a custom prompt with up to 100 Python scripts and collecting events:

.. code-block:: bash

    python -m locodellm generate ./model \
        'Write 42 to value.txt, read it back, and report its contents.' \
        --chat-template chatml --harness python --max-tool-calls 100 \
        --max-length 1024 --output evenements.json

The final answer is printed to stdout; it need not be a Python function.
To keep generated files after the run, the prompt must specify an absolute
output path outside the temporary directory, or use JSON input:

.. code-block:: json

    {
      "repo": "../my-project",
      "prompts": [
        "Update README.md to explain the installation steps.",
        "Add a test for the documented behavior."
      ]
    }

Save this as ``tasks.json`` and run:

.. code-block:: bash

    python -m locodellm generate ./model --input tasks.json \
        --chat-template chatml --max-tool-calls 100 --max-length 1024 \
        --output evenements.json

For native Transformers weights, use a Hugging Face model id or a complete
local checkpoint directory containing its configuration, tokenizer, and weights,
not the Hugging Face cache root (``models--OWNER--NAME``):

.. code-block:: bash

    python -m locodellm generate Qwen/Qwen3.5-35B-A3B-FP8 \
        --backend pytorch --device auto --input tasks.json \
        --max-tool-calls 100 --max-length 1024 --output evenements.json

Missing checkpoint files are downloaded when using a model id. FP8 checkpoints
require hardware and quantization support compatible with Transformers.
