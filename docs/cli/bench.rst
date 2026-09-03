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
    benchmarks and 2048 for LM-Eval).

``--chat-template``
    Chat template to use (e.g. ``chatml``).

``--output, -o``
    Built-in benchmark output path. Every output writes a detailed ``.json``
    file with generated code and per-input results plus an ``.xlsx`` workbook
    with ``aggregated`` and ``raw_data`` sheets. JSON files are written
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
  ``avg_tokens_per_second``, ``avg_score``.

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

Example output (``Qwen/Qwen2.5-Coder-0.5B-Instruct``, ``basic`` benchmark):

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

Export to detailed JSON and aggregated Excel (the JSON is written
incrementally during the run):

.. code-block:: bash

    python -m locodellm bench mock/generate basic \
        --chat-template chatml -o results.json

Export to CSV, detailed JSON, and aggregated Excel:

.. code-block:: bash

    python -m locodellm bench mock/generate basic --chat-template chatml -o results.csv
