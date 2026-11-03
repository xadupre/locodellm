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

``--provider``
    Execution provider (e.g. ``CUDAExecutionProvider``).

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

Runs the same built-in problems with optional tool use:

.. code-block:: bash

    python -m locodellm bench Qwen/Qwen2.5-Coder-0.5B-Instruct basic \
        --harness python --chat-template chatml --max-length 1024 \
        --output results.xlsx --verbose 1

The model is instructed to request one execution per turn using
``<python>...script...</python>``. The application executes the script and
returns stdout, stderr, exit code, timeout status, and the remaining budget.
After the tenth execution the model can still provide its final function,
but an eleventh request is refused and the problem is marked failed.
Hidden test inputs and expected results are not supplied to the tool.

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

Install the optional dependency before running LM-Eval benchmarks:

.. code-block:: bash

    pip install ".[eval]"

Run ten samples from the LM-Eval ``gsm8k`` task:

.. code-block:: bash

    python -m locodellm bench path/to/model gsm8k --limit 10

Only LM-Eval tasks using ``generate_until`` are supported; likelihood and
perplexity tasks require model logits, which ONNX Runtime GenAI does not
expose.

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
