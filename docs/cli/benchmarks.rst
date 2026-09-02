benchmarks
==========

Lists the available built-in benchmarks and an LM-Eval example. When LM-Eval
is installed, it also lists the benchmarks provided by LM Evaluation Harness
and links to the full benchmark list.

Usage
-----

.. code-block:: bash

    python -m locodellm benchmarks

Example output
--------------

.. code-block:: text

    basic  10 Python function prompts with growing difficulty, from returning a constant string to computing an edit distance.
    gsm8k  LM Evaluation Harness benchmark.

    Full LM-Eval benchmark list: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/
