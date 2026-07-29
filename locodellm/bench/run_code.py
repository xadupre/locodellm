"""Compile and run generated Python functions with undefined arguments."""

from __future__ import annotations

import inspect
import types
from dataclasses import dataclass, field
from typing import Any


class _Undefined:
    """Sentinel value passed as argument when no real value is available.

    Behaves as a neutral element for common operations so that simple
    functions can execute without raising immediately.
    """

    def __repr__(self) -> str:
        return "<undefined>"

    def __str__(self) -> str:
        return "<undefined>"

    def __bool__(self) -> bool:
        return False

    def __add__(self, other: Any) -> _Undefined:
        return self

    def __radd__(self, other: Any) -> _Undefined:
        return self

    def __mul__(self, other: Any) -> _Undefined:
        return self

    def __rmul__(self, other: Any) -> _Undefined:
        return self

    def __sub__(self, other: Any) -> _Undefined:
        return self

    def __rsub__(self, other: Any) -> _Undefined:
        return self

    def __iter__(self):
        return iter([])

    def __len__(self) -> int:
        return 0

    def __getattr__(self, name: str) -> _Undefined:
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> _Undefined:
        return self

    def __getitem__(self, key: Any) -> _Undefined:
        return self


UNDEFINED = _Undefined()


@dataclass
class RunStatus:
    """Result of compiling and running a Python function string.

    Attributes:
        compiled: Whether the code compiled successfully.
        compile_error: The exception raised during compilation, if any.
        ran: Whether the function executed without error.
        run_error: The exception raised during execution, if any.
        result: The return value of the function, or *None* if it did
            not run or raised.
    """

    compiled: bool = False
    compile_error: BaseException | None = None
    ran: bool = False
    run_error: BaseException | None = None
    result: Any = None
    _function: Any = field(default=None, repr=False)

    @property
    def success(self) -> bool:
        """Returns *True* if the code both compiled and ran without error."""
        return self.compiled and self.ran


def run_function(source: str) -> RunStatus:
    """Compiles and executes a Python function defined in *source*.

    The function is called with :data:`UNDEFINED` for each of its
    positional parameters and as the default for keyword arguments.

    Args:
        source: Python source code defining exactly one function.

    Returns:
        A :class:`RunStatus` describing the outcome.
    """
    status = RunStatus()

    # Compile
    try:
        code = compile(source, "<generated>", "exec")
    except SyntaxError as exc:
        status.compile_error = exc
        return status

    status.compiled = True

    # Execute the module-level code to define the function
    namespace: dict[str, Any] = {}
    try:
        exec(code, namespace)  # noqa: S102
    except Exception as exc:
        status.ran = False
        status.run_error = exc
        return status

    # Find the function object
    functions = [v for v in namespace.values() if isinstance(v, types.FunctionType)]
    if not functions:
        status.run_error = RuntimeError("No function found in the provided source code.")
        return status

    func = functions[0]
    status._function = func

    # Build arguments from the signature
    sig = inspect.signature(func)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            if param.default is inspect.Parameter.empty:
                args.append(UNDEFINED)
            else:
                args.append(param.default)
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            if param.default is inspect.Parameter.empty:
                kwargs[name] = UNDEFINED
            else:
                kwargs[name] = param.default
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            pass  # *args — leave empty
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            pass  # **kwargs — leave empty

    # Call the function
    try:
        status.result = func(*args, **kwargs)
    except Exception as exc:
        status.run_error = exc
        return status

    status.ran = True
    return status
