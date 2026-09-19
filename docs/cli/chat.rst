chat
====

Loads a model with ONNX Runtime GenAI and streams answers as tokens are
generated. Each prompt builds on all preceding prompts and answers. The
same generator and its in-memory KV cache are reused between turns, rather
than replaying the conversation.

Usage
-----

.. code-block:: bash

    python -m locodellm chat MODEL [OPTIONS]
    python -m locodellm chat mock/generate --chat-template chatml
    python -m locodellm chat ./Qwen2.5-Coder-0.5B-onnx --chat-template chatml

``MODEL`` accepts the same local directories, mock model ids, and HuggingFace
ids as :doc:`generate`. HuggingFace models are downloaded and converted
automatically. ``--precision``, ``--provider``, and ``--verbose`` also work
as for ``generate``.

``--chat-template chatml`` formats turns for ChatML-based instruct models,
such as Qwen. Without this option, prompts are sent as-is.

``--max-length`` sets the maximum **total conversation length**, including
all prompts and answers (default: 2048 tokens). Choose a value supported by
the model. When the limit is reached, use ``/clear`` to start again.

Commands
--------

Enter one prompt per line. Empty lines are ignored.

* ``/clear`` clears the conversation and KV cache without reloading the model.
* ``/quit`` exits. End-of-file also exits (Ctrl-D on Unix).

For example, enter ``write a python function which returns "hello"``, then
``change hello into bonjour``. The second answer uses the preceding turn.
