Available Benchmarks
=====================

Use :func:`~locodellm.bench.get_available_benchmarks` to list the built-in
benchmarks and :func:`~locodellm.bench.load_benchmark` to load one by name.

.. code-block:: python

    from locodellm.bench import get_available_benchmarks, load_benchmark

    # List available benchmarks
    for name, description in get_available_benchmarks().items():
        print(f"{name}: {description}")

    # Load a benchmark
    bench = load_benchmark("basic")

Built-in Benchmarks
--------------------

basic
^^^^^

.. runpython::
    :showcode:
    :rst:

    from locodellm.bench import load_benchmark

    bench = load_benchmark("basic")
    print(bench.description)
    print()
    print(f"**{len(bench.tests)} prompt tests:**")
    print()
    print(".. list-table::")
    print("   :header-rows: 1")
    print("   :widths: 5 60 10")
    print()
    print("   * - #")
    print("     - Task")
    print("     - Expected results")
    for i, test in enumerate(bench.tests, 1):
        prompt = test.prompt
        if len(prompt) > 80:
            prompt = prompt[:77] + "..."
        print(f"   * - {i}")
        print(f"     - {prompt}")
        print(f"     - {len(test.expected)}")
