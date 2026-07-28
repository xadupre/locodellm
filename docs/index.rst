locodellm
=========

Experimentation around local LLM using
`onnxruntime-genai <https://github.com/microsoft/onnxruntime-genai>`_.

Install
-------

.. code-block:: bash

    pip install -e .

Quick start
-----------

.. code-block:: python

    from locodellm.generate import generate

    session = generate("Once upon a time", "path/to/model")
    print(session.text)

.. toctree::
    :maxdepth: 2
    :caption: Contents

    api/index
