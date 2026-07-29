"""A basic benchmark of 10 Python function prompts with growing difficulty."""

from locodellm.bench import ExpectedResult, PromptTest

BASIC_BENCHMARK: list[PromptTest] = [
    # 1. Trivial: return a constant string
    PromptTest(
        prompt='write a python function called hello that returns the string "hello"',
        expected=[
            ExpectedResult(args=(), expected="hello"),
            ExpectedResult(args=(), expected="hello"),
        ],
    ),
    # 2. Simple arithmetic: add two numbers
    PromptTest(
        prompt="write a python function called add that takes two numbers and returns their sum",
        expected=[
            ExpectedResult(args=(1, 2), expected=3),
            ExpectedResult(args=(-5, 5), expected=0),
            ExpectedResult(args=(0, 0), expected=0),
        ],
    ),
    # 3. String manipulation: reverse a string
    PromptTest(
        prompt=(
            "write a python function called reverse_string that takes a string"
            " and returns it reversed"
        ),
        expected=[
            ExpectedResult(args=("hello",), expected="olleh"),
            ExpectedResult(args=("",), expected=""),
            ExpectedResult(args=("a",), expected="a"),
        ],
    ),
    # 4. List operation: find the maximum value
    PromptTest(
        prompt=(
            "write a python function called find_max that takes a list of numbers"
            " and returns the maximum value"
        ),
        expected=[
            ExpectedResult(args=([3, 1, 4, 1, 5],), expected=5),
            ExpectedResult(args=([-10, -20, -3],), expected=-3),
            ExpectedResult(args=([42],), expected=42),
        ],
    ),
    # 5. Conditional logic: check if a number is prime
    PromptTest(
        prompt=(
            "write a python function called is_prime that takes an integer"
            " and returns True if it is prime, False otherwise"
        ),
        expected=[
            ExpectedResult(args=(2,), expected=True),
            ExpectedResult(args=(7,), expected=True),
            ExpectedResult(args=(1,), expected=False),
            ExpectedResult(args=(9,), expected=False),
        ],
    ),
    # 6. Recursion / iteration: compute factorial
    PromptTest(
        prompt=(
            "write a python function called factorial that takes a non-negative"
            " integer n and returns n!"
        ),
        expected=[
            ExpectedResult(args=(0,), expected=1),
            ExpectedResult(args=(1,), expected=1),
            ExpectedResult(args=(5,), expected=120),
            ExpectedResult(args=(10,), expected=3628800),
        ],
    ),
    # 7. Data structures: count character frequencies
    PromptTest(
        prompt=(
            "write a python function called char_count that takes a string"
            " and returns a dictionary mapping each character to its count"
        ),
        expected=[
            ExpectedResult(args=("aab",), expected={"a": 2, "b": 1}),
            ExpectedResult(args=("",), expected={}),
            ExpectedResult(args=("xyz",), expected={"x": 1, "y": 1, "z": 1}),
        ],
    ),
    # 8. Two-pointer / sorting: check if a string is a palindrome ignoring case and spaces
    PromptTest(
        prompt=(
            "write a python function called is_palindrome that takes a string and returns True "
            "if it is a palindrome ignoring case and spaces, False otherwise"
        ),
        expected=[
            ExpectedResult(args=("racecar",), expected=True),
            ExpectedResult(args=("Race Car",), expected=True),
            ExpectedResult(args=("hello",), expected=False),
            ExpectedResult(args=("",), expected=True),
        ],
    ),
    # 9. Dynamic programming: compute the nth Fibonacci number
    PromptTest(
        prompt=(
            "write a python function called fibonacci that takes a non-negative integer n "
            "and returns the nth Fibonacci number where fibonacci(0)=0 and fibonacci(1)=1"
        ),
        expected=[
            ExpectedResult(args=(0,), expected=0),
            ExpectedResult(args=(1,), expected=1),
            ExpectedResult(args=(10,), expected=55),
            ExpectedResult(args=(20,), expected=6765),
        ],
    ),
    # 10. Advanced: compute the edit distance (Levenshtein distance) between two strings
    PromptTest(
        prompt=(
            "write a python function called edit_distance that takes two strings a and b "
            "and returns the minimum number of single-character edits (insertions, deletions, "
            "or substitutions) needed to transform a into b"
        ),
        expected=[
            ExpectedResult(args=("kitten", "sitting"), expected=3),
            ExpectedResult(args=("", "abc"), expected=3),
            ExpectedResult(args=("abc", "abc"), expected=0),
            ExpectedResult(args=("saturday", "sunday"), expected=3),
        ],
    ),
]
