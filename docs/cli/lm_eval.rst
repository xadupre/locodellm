lm-eval
*******

Runs generation-based `LM Evaluation Harness
<https://github.com/EleutherAI/lm-evaluation-harness>`_ tasks against an ONNX
Runtime GenAI model.

Install the optional dependency before using the command:

.. code-block:: bash

    pip install ".[eval]"

Usage
-----

.. code-block:: bash

    python -m locodellm lm-eval MODEL TASK [TASK ...] [OPTIONS]

For example, this evaluates ten samples from ``gsm8k``:

.. code-block:: bash

    python -m locodellm lm-eval path/to/model gsm8k --limit 10

The command accepts the same model options as ``generate``, as well as
``--num-fewshot`` and ``--limit``. Only tasks using LM-Eval's
``generate_until`` request type are supported; likelihood and perplexity tasks
require model logits, which the ONNX Runtime GenAI generation API does not
expose.
