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
repeatable ``--provider-option NAME=VALUE`` options, ``--num-fewshot``, and
``--limit``. For example, this selects CUDA device 1:

.. code-block:: bash

    python -m locodellm lm-eval path/to/model gsm8k \
        --provider CUDAExecutionProvider --provider-option device_id=1

Provider options are passed to ``onnxruntime_genai.Config.set_provider_option``.
Only tasks using LM-Eval's
``generate_until`` request type are supported; likelihood and perplexity tasks
require model logits, which the ONNX Runtime GenAI generation API does not
expose.
