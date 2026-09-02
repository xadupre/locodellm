Command Line Interface
======================

The ``locodellm`` package provides several subcommands accessible via
``python -m locodellm``.

.. list-table::
    :header-rows: 1
    :widths: 20 80

    * - Command
      - Description
    * - :doc:`version`
      - Prints the installed package version.
    * - :doc:`benchmarks`
      - Lists the available built-in and LM-Eval benchmarks.
    * - :doc:`models`
      - Lists the available mock ONNX test models.
    * - :doc:`generate`
      - Generates text from a prompt using a local LLM.
    * - :doc:`bench`
      - Runs a built-in or LM-Eval benchmark against a model.

.. toctree::
    :maxdepth: 1
    :hidden:

    version
    benchmarks
    models
    generate
    bench
