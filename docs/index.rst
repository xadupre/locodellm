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

    from locodellm.session import create_session

    session = create_session("path/to/model")
    session.generate("Once upon a time")
    print(session.text)

.. toctree::
    :maxdepth: 2
    :caption: Contents

    auto_examples/index
    api/index
