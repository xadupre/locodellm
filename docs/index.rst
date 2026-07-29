locodellm
=========

.. image:: https://github.com/xadupre/locodellm/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/xadupre/locodellm/actions/workflows/ci.yml
    :alt: CI

.. image:: https://codecov.io/gh/xadupre/locodellm/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/xadupre/locodellm
    :alt: Coverage

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
