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
      - Lists the available built-in benchmarks.
    * - :doc:`models`
      - Lists the available mock ONNX test models.
    * - :doc:`generate`
      - Generates text from a prompt using a local LLM.
    * - :doc:`bench`
      - Runs a benchmark against a model, evaluating generated code.
    * - :doc:`lm_eval`
      - Runs LM Evaluation Harness generation tasks against an ONNX model.

.. toctree::
    :maxdepth: 1
    :hidden:

    version
    benchmarks
    models
    generate
    bench
    lm_eval
