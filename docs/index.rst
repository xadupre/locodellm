locodellm
=========

.. image:: https://github.com/xadupre/locodellm/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/xadupre/locodellm/actions/workflows/ci.yml
    :alt: CI

.. image:: https://codecov.io/gh/xadupre/locodellm/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/xadupre/locodellm
    :alt: Coverage

**locodellm** is a small toolkit for experimenting with local large
language models. It wraps
`onnxruntime-genai <https://github.com/microsoft/onnxruntime-genai>`_ behind a
simple session API to load an ONNX model directory and generate text,
supports multi-turn conversations, and ships helpers to build tiny dummy
models for testing. A minimal command line interface is also provided.

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
    cli/index
    api/index
