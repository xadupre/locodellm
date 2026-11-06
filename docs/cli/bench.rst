bench
=====

Runs either a built-in benchmark or one or more generation-based `LM
Evaluation Harness <https://github.com/EleutherAI/lm-evaluation-harness>`_
tasks against a model. For built-in benchmarks, generated code is compiled,
executed with test inputs, and compared to expected results.

Results are displayed as a markdown table on standard output, followed by
per-case statistics and an aggregated summary. File output includes detailed
JSON and an Excel workbook with ``aggregated`` and ``raw_data`` sheets, plus
CSV when requested.

When ``--verbose 1`` is used, a progress bar is shown on stderr during
generation.

Usage
-----

.. code-block:: bash

    python -m locodellm bench MODEL BENCHMARK [BENCHMARK ...] [OPTIONS]

Options
-------

``MODEL``
    Model id or path. Use ``mock/generate`` for the mock model, a local
    directory path, or a HuggingFace id like
    ``Qwen/Qwen2.5-Coder-0.5B-Instruct``.

``BENCHMARK``
    One built-in benchmark name or one or more LM-Eval task names. Use
    ``python -m locodellm benchmarks`` to list available benchmarks.

``--precision``
    Precision qualifier for conversion (e.g. ``fp32``, ``fp16``, ``int4``).
    With PyTorch, selects the loading dtype: ``auto`` (default), ``fp32``,
    ``fp16``, or ``bf16``. It does not quantize the model.

``--backend``
    Generation backend: ``onnx`` (default) or ``pytorch``. The PyTorch backend
    loads Hugging Face Transformers checkpoints directly, without ONNX
    conversion. It supports both built-in benchmarks (including the Python
    tool harness) and generation-based LM-Eval tasks.

``--device``
    PyTorch device, e.g. ``cpu`` (default), ``cuda``, or ``cuda:0``.
    ``auto`` lets Accelerate distribute model weights across available
    devices. Requires ``--backend pytorch``.

``--provider``
    ONNX execution provider (e.g. ``CUDAExecutionProvider``). This option,
    ``--provider-option``, and ``--session-option`` cannot be used with PyTorch.

``--provider-option NAME=VALUE``
    ONNX Runtime option for the selected provider. May be repeated.

``--session-option NAME=JSON_VALUE``
    ONNX Runtime session option. May be repeated.

``--max-length``
    Maximum token length for generation (default: 200 for built-in
    benchmarks and 2048 for LM-Eval). With ``--harness python``, limits
    **new tokens per model turn**, instead of total conversation length
    (default: 1024).

``--harness python``
    Enables model-requested Python script execution for a built-in benchmark.
    Each problem starts a fresh conversation with a budget of **10 executions**.
    The model decides whether to use the tool, receives its output, and can
    revise its solution before returning a final function. The default,
    without this option, remains a single generation with no tool.

``--chat-template``
    Chat template to use (e.g. ``chatml``).
    With PyTorch, the tokenizer's native chat template is used by default
    when available; otherwise prompts are sent as plain text. An explicit
    ``chatml`` overrides the native template.

``--output, -o``
    Built-in benchmark output path. Every output writes a detailed ``.json``
    file with interaction traces, generated code, and per-input results plus
    an ``.xlsx`` workbook with ``aggregated`` and ``raw_data`` sheets. JSON files are written
    incrementally during the run. A requested ``.csv`` file is written in
    addition to the JSON and Excel files. LM-Eval currently ignores this
    option.

``--num-fewshot``
    Number of few-shot examples for LM-Eval.

``--limit``
    Number or fraction of examples to evaluate with LM-Eval.

``--verbose, -v``
    Verbosity level (default: 0). At level 1, a progress bar is shown and
    model loading is silent. At level 2+, model loading details are also
    printed.

Output columns
--------------

**Results table** — one row per input set:

- ``prompt``: the prompt text
- ``duration``: generation time in seconds
- ``token_count``: number of generated tokens
- ``tokens_per_second``: generation speed
- ``tool_call_count``: Python tool executions for this problem, repeated on
  each input row (not a per-input counter)
- ``harness_error``: malformed tool request, exhausted budget, or empty answer,
  if the harness could not obtain a final answer
- ``compiled``: whether the generated code compiled
- ``ran``: whether the code ran without error
- ``input_index``: index of the input set within the prompt
- ``passed``: whether the output matched the expected value

**Statistics table** — one row per prompt:

- Same timing and compilation columns, plus ``inputs``, ``passed``,
  ``failed``, and ``score`` (fraction of inputs that passed).

**Summary table** — aggregated metrics:

- ``total_cases``, ``total_inputs``, ``total_passed``, ``total_failed``,
  ``cases_compiled``, ``cases_ran``, ``avg_duration``,
  ``avg_tokens_per_second``, ``avg_score``, ``total_tool_calls``.

The per-prompt statistics include ``tool_call_count`` once per problem.
``total_tool_calls`` sums the executions across problems, without counting
the repeated input rows. Failed scripts and timeouts count as tool uses;
final benchmark scoring does not. Without the harness these counts are zero.

Python tool harness
-------------------

Runs the same built-in problems with Python execution and feedback:

.. code-block:: bash

    python -m locodellm bench Qwen/Qwen2.5-Coder-0.5B-Instruct basic \
        --harness python --chat-template chatml --max-length 1024 \
        --output results.xlsx --verbose 1

The execution budget defaults to ten scripts per problem and can be increased,
for example with ``--max-tool-calls 100``. Instruction text, remaining-budget
feedback, and budget errors reflect the configured limit.

The model can request one or more executions per turn using
``<python>...script...</python>`` blocks. The application executes the scripts and
returns stdout, stderr, exit code, timeout status, and the remaining budget.
The same harness also accepts ``<bash>...commands...</bash>`` requests, sharing
the Python execution budget and working directory. Mixed scripts run in order.
Bash requires ``bash`` on PATH and runs with ``pipefail`` enabled, without
profile or rc files. Its commands, output, status, and ``tool: bash`` are exported;
Bash scripts do not contribute to Python function-call counts.
Use ``--harness bash`` to advertise and accept only Bash tool requests;
Python requests, including mixed batches, are rejected before execution.
The Nemotron form
``<tool_call><function=stateful_python_code_exec><parameter=code>...code...</parameter></function></tool_call>``
is also accepted, with ``stateful_python_code_exec`` or ``python`` as the tool
name, as an explicit Python execution request outside Markdown fences and
thinking. Despite the native tool name, it uses the same fresh-process
semantics as ``<python>``; Python variables do not persist between scripts.
The abbreviated Nemotron response starting with ``<parameter=code>`` is also
recognized when it occupies the entire answer after thinking and ends with
``</parameter></function></tool_call>``. Missing closing tags and unknown
explicit tool names remain errors.
Grouped calls execute sequentially in message order, not concurrently, because
scripts can depend on files in their shared working directory. All results are
returned together before the next model response, even if a script fails.
Each block consumes one execution; a batch exceeding the remaining budget is
rejected in full without executing any of its scripts.
After the last allowed execution the model can still provide its final function,
but any further request is refused and the problem is marked failed.
Hidden test inputs and expected results are not supplied to the tool.

Final-answer extraction prefers Markdown blocks labelled ``python`` or ``py``
and ignores accompanying unlabelled output or other-language blocks. When no
Python-labelled block is present, unlabelled blocks remain accepted as code.
Assertions included in the published Python code still execute and can fail;
the parser does not remove incorrect checks to improve a score.

Every ``<python>...</python>`` block outside Markdown fences requests execution,
even if it only defines functions or accompanies a proposed final answer.
The model always receives stdout, stderr, and execution status before another
generation. Defining a function alone produces no output; the model must include
its own calls, ``print``, or ``assert`` statements to check its behavior.
Final functions use Markdown fences without tool tags. Tool use is optional:
a final answer is accepted without requiring any script or function execution.
Only the explicit message format distinguishes tool requests from published code.

The summary reports ``success_rate`` over benchmark test inputs,
``avg_generated_tokens`` and ``avg_prompts`` per problem, and
``total_function_calls``. Token totals include thinking and all model responses;
prompt counts include the initial prompt and every subsequent model turn.
``function_call_count`` counts actual invocations of Python functions defined in
tool scripts, including recursive and helper calls, not definitions or hidden-test
evaluation. ``tool_call_count`` separately counts executed scripts.
``multi_tool_turn_count`` counts model responses requesting at least two scripts,
and ``max_tool_calls_per_turn`` records the largest requested batch, including a
batch rejected for exceeding the budget. These describe grouped requests, not
parallel execution.

Each script runs in a fresh Python process, with the current interpreter,
in a temporary working directory shared only across that problem's calls.
Executions time out after 10 seconds, and each stdout/stderr response is
limited to 16,000 bytes, with an explicit truncation marker. Detailed JSON
also records the scripts and their execution results in ``tool_calls``.
Generation token counts sum all model turns; duration includes the complete
conversation and tool execution, so tokens/s is an end-to-end rate in this mode.

Each JSON problem starts with ``initial_prompt``, the complete first message
sent to the model, including the harness instructions. It is followed
immediately by an ``events`` array in conversation order:

- ``model_response``: the complete model response, before code extraction
- ``command``: the actual process argument list (``argv``), working directory
  (``cwd``), Python script (``code``), and timeout
- ``stdout``: the tool's standard output, including an explicit empty string
  when the script prints nothing
- ``stderr``: the tool's standard error, as a separate event
- ``process_exit``: exit code and timeout status
- ``prompt``: the next message sent to the model
- Further responses, commands, and outputs as needed, then ``evaluation``

Stdout and stderr are collected after each command completes; these separate
events do not represent byte-level interleaving between the two streams.
They contain the same bounded outputs returned to the model, including any
truncation marker. The event array comes before the summary statistics and
other detailed result fields, so it can be read directly from start to finish.
Initial generation options are in ``initial_generation_options``; later
prompts include their own ``generation_options``.

Responses preserve any ``<think>`` sections or other reasoning text actually
emitted by the model. Internal reasoning that the model does not expose
cannot be recorded.
Tool-call detection and final code extraction ignore this thinking text,
including when the opening ``<think>`` was already supplied by the tokenizer's
chat template. The complete original responses remain unchanged in the JSON
events, and token counts still include reasoning tokens.

The final evaluation includes compilation/execution errors and the per-input
results, solely for inspection in the JSON file. It is **never sent to the
model** and does not trigger a retry or tool call. Neither the benchmark
problems nor the model's prompts or tool-selection behavior are changed by
tracing. The same trace format is written without a harness, with no tool
events. JSON is saved after each completed problem.

**This is not a sandbox:** generated scripts run with your user's permissions,
can access files and the network, and can start other processes. Run only in
an appropriately isolated environment. The timeout and output capture do not
provide process-tree, disk, or memory isolation. Final code evaluation retains
the existing in-process execution behavior.

Examples
--------

Run directly with PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~

Install the optional backend dependencies:

.. code-block:: bash

    pip install "locodellm[pytorch]"

Load a native Qwen3.5 checkpoint without converting it to ONNX:

.. code-block:: bash

    python -m locodellm bench Qwen/Qwen3.5-4B basic \
        --backend pytorch --device cuda --precision bf16 \
        --max-length 1024 --output results.xlsx

For a pre-quantized Transformers checkpoint, omit ``--precision int4``:
the model's own quantization configuration is used. For example:

.. code-block:: bash

    python -m locodellm bench QuantTrio/Qwen3.5-4B-AWQ basic \
        --backend pytorch --device cuda --max-length 1024 \
        --output results.xlsx

AWQ, GPTQ, and FP8 checkpoints require their respective Transformers
quantization dependencies, supported kernels, and compatible hardware.
The PyTorch extra alone does not guarantee support for every quantization.
GGUF files and ``mock/`` ONNX models are not PyTorch checkpoints.
Qwen3.5 vision-language checkpoints are loaded with
``AutoModelForImageTextToText`` for text-only benchmark prompts.

Use the Python tool harness independently of the inference backend:

.. code-block:: bash

    python -m locodellm bench Qwen/Qwen3.5-4B basic \
        --backend pytorch --device cuda --harness python --max-length 1024

CSV exports and the Excel ``raw_data`` sheet include ``backend`` and
``device`` columns so PyTorch runs can be distinguished from ONNX runs.
ONNX remains the default.

Run with ONNX Runtime GenAI
~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the optional dependency before running LM-Eval benchmarks:

.. code-block:: bash

    pip install ".[eval]"

Run ten samples from the LM-Eval ``gsm8k`` task:

.. code-block:: bash

    python -m locodellm bench path/to/model gsm8k --limit 10

Only LM-Eval tasks using ``generate_until`` are supported by the session
adapter, with either backend. Likelihood and perplexity tasks are not
implemented by this adapter.

Run with the mock model:

.. code-block:: bash

    python -m locodellm bench mock/generate basic --chat-template chatml

Run with a HuggingFace model and export to Excel:

.. code-block:: bash

    python -m locodellm bench Qwen/Qwen2.5-Coder-0.5B-Instruct basic \
        --chat-template chatml --output results.xlsx --verbose 1

Example output (selected columns, ``Qwen/Qwen2.5-Coder-0.5B-Instruct``,
``basic`` benchmark):

.. code-block:: text

    [██████████████████████████████] 10/10

Statistics:

.. code-block:: text

    | prompt                                   | duration | token_count | tokens_per_second | compiled | ran   | inputs | passed | failed | score |
    |:-----------------------------------------|---------:|------------:|------------------:|:---------|:------|-------:|-------:|-------:|------:|
    | ... hello ...                            |     3.37 |          66 |             19.56 | True     | True  |      2 |      2 |      0 |   1.0 |
    | ... add ...                              |     5.27 |         103 |             19.53 | True     | True  |      3 |      3 |      0 |   1.0 |
    | ... reverse_string ...                   |     6.50 |         127 |             19.54 | True     | True  |      3 |      3 |      0 |   1.0 |
    | ... find_max ...                         |     9.05 |         174 |             19.23 | True     | True  |      3 |      3 |      0 |   1.0 |
    | ... is_prime ...                         |     8.65 |         171 |             19.78 | True     | False |      4 |      4 |      0 |   1.0 |
    | ... factorial ...                        |     9.00 |         175 |             19.44 | True     | False |      4 |      4 |      0 |   1.0 |
    | ... char_count ...                       |     8.70 |         171 |             19.66 | True     | True  |      3 |      3 |      0 |   1.0 |
    | ... is_palindrome ...                    |     8.29 |         165 |             19.89 | True     | True  |      4 |      4 |      0 |   1.0 |
    | ... fibonacci ...                        |     8.26 |         161 |             19.48 | True     | False |      4 |      4 |      0 |   1.0 |
    | ... edit_distance ...                    |     7.96 |         153 |             19.22 | False    | False |      4 |      0 |      4 |   0.0 |

Summary:

.. code-block:: text

    | metric                |    value |
    |:----------------------|---------:|
    | total_cases           | 10       |
    | total_inputs          | 34       |
    | total_passed          | 30       |
    | total_failed          |  4       |
    | cases_compiled        |  9       |
    | cases_ran             |  6       |
    | avg_duration          |  7.51    |
    | avg_tokens_per_second | 19.53    |
    | avg_score             |  0.9     |
    | total_tool_calls      |  0       |

Export to detailed JSON and aggregated Excel (the JSON is written
incrementally during the run):

.. code-block:: bash

    python -m locodellm bench mock/generate basic \
        --chat-template chatml -o results.json

Export to CSV, detailed JSON, and aggregated Excel:

.. code-block:: bash

    python -m locodellm bench mock/generate basic --chat-template chatml -o results.csv
