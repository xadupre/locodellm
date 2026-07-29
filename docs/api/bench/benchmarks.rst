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
    tests = load_benchmark("basic")

Built-in Benchmarks
--------------------

basic
^^^^^

10 Python function prompts with growing difficulty.  Each prompt asks the
model to generate a single Python function.  The difficulty ranges from
returning a constant string to computing an edit distance (Levenshtein
distance).

.. list-table::
   :header-rows: 1
   :widths: 5 55 10

   * - #
     - Task
     - Expected results
   * - 1
     - Return the string ``"hello"``
     - 2
   * - 2
     - Add two numbers
     - 3
   * - 3
     - Reverse a string
     - 3
   * - 4
     - Find the maximum in a list
     - 3
   * - 5
     - Check if a number is prime
     - 4
   * - 6
     - Compute factorial
     - 4
   * - 7
     - Count character frequencies
     - 3
   * - 8
     - Palindrome check (ignoring case and spaces)
     - 4
   * - 9
     - Compute nth Fibonacci number
     - 4
   * - 10
     - Compute edit distance between two strings
     - 4
