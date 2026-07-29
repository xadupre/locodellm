generate
========

Generates text from a prompt using a local LLM. The model can be a mock
model id, a local path to an ONNX model directory, or a HuggingFace
repository id (which is automatically downloaded and converted).

Usage
-----

.. code-block:: bash

    python -m locodellm generate MODEL PROMPT [OPTIONS]

Options
-------

``MODEL``
    Model id or path. Use ``mock/generate`` for the mock model, a local
    directory path, or a HuggingFace id like
    ``Qwen/Qwen2.5-Coder-0.5B-Instruct``.

``PROMPT``
    The prompt text to send to the model.

``--precision``
    Precision qualifier for conversion (e.g. ``fp32``, ``fp16``, ``int4``).

``--provider``
    Execution provider (e.g. ``CUDAExecutionProvider``).

``--max-length``
    Maximum token length for generation (default: 200).

``--chat-template``
    Chat template to use (e.g. ``chatml``).

``--verbose, -v``
    Verbosity level (default: 0).

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
