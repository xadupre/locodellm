"""High-level generation from a model id."""

from __future__ import annotations

import os
import threading
from typing import Any

from locodellm.session import SessionState, create_session

# Thread-safe cache for loaded sessions keyed by (model_id, precision, provider).
_session_cache: dict[tuple[str, str | None, str | None], SessionState] = {}
_cache_lock = threading.Lock()

# Cache for converted model paths keyed by (model_id, precision).
_model_path_cache: dict[tuple[str, str | None], str] = {}
_path_lock = threading.Lock()

# Default directory for storing converted/generated models.
_DEFAULT_CACHE_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "locodellm", "models"
)


def _get_model_path(
    model_id: str, precision: str | None = None, cache_dir: str | None = None, verbose: int = 0
) -> str:
    """Returns the on-disk path for a model, creating it if needed.

    For mock models (``mock/...``), the model directory is generated once
    and cached. For other model ids the path is assumed to already exist.
    """
    key = (model_id, precision)
    with _path_lock:
        if key in _model_path_cache:
            return _model_path_cache[key]

    base_dir = cache_dir or _DEFAULT_CACHE_DIR
    os.makedirs(base_dir, exist_ok=True)

    if model_id.startswith("mock/"):
        from locodellm.test_models import _MODELS

        if model_id not in _MODELS:
            available = list(_MODELS.keys())
            raise KeyError(f"Unknown model id {model_id!r}. Available: {available}")

        safe_name = model_id.replace("/", "_")
        if precision:
            safe_name = f"{safe_name}_{precision}"
        model_dir = os.path.join(base_dir, safe_name)

        # Skip conversion if already done.
        marker = os.path.join(model_dir, "model.onnx")
        if not os.path.exists(marker):
            if verbose:
                print(f"[generate_from_model] creating model {model_id!r} at {model_dir}")
            if model_id == "mock/generate":
                from locodellm.test_models import create_mock_generate_model

                create_mock_generate_model(model_dir)
            elif model_id == "mock/tiny":
                from locodellm.test_models import create_tiny_model

                create_tiny_model(model_dir)
        elif verbose:
            print(f"[generate_from_model] reusing cached model at {model_dir}")

        path = os.path.abspath(model_dir)
    else:
        # Assume model_id is a path or HuggingFace id handled externally.
        path = model_id

    with _path_lock:
        _model_path_cache[key] = path
    return path


def _get_session(
    model_id: str,
    precision: str | None = None,
    provider: str | None = None,
    cache_dir: str | None = None,
    chat_template: str | None = None,
    verbose: int = 0,
) -> SessionState:
    """Returns a cached session, creating it on first access."""
    key = (model_id, precision, provider)
    with _cache_lock:
        if key in _session_cache:
            return _session_cache[key]

    model_path = _get_model_path(
        model_id, precision=precision, cache_dir=cache_dir, verbose=verbose
    )

    providers = [provider] if provider else None
    session = create_session(
        model_path, providers=providers, verbose=verbose, chat_template=chat_template
    )

    with _cache_lock:
        _session_cache[key] = session
    return session


def generate_from_model(
    model_id: str,
    prompt: str,
    precision: str | None = None,
    provider: str | None = None,
    max_length: int = 200,
    cache_dir: str | None = None,
    chat_template: str | None = None,
    verbose: int = 0,
    **search_options: Any,
) -> SessionState:
    """Generates text from a prompt using the specified model.

    Models are loaded once and cached for subsequent calls.  For mock
    models (ids starting with ``mock/``), the ONNX conversion is
    performed only on the first call and reused afterwards.

    Args:
        model_id: The model identifier.  Use ``mock/generate`` or
            ``mock/tiny`` for test models, a HuggingFace id like
            ``Qwen/Qwen2.5-Coder-0.5B-Instruct``, or a filesystem
            path for pre-converted models.
        prompt: The text prompt to send to the model.
        precision: Optional precision qualifier (e.g. ``"fp16"``,
            ``"int4"``).  Currently used only for cache keying.
        provider: Execution provider name (e.g.
            ``"CUDAExecutionProvider"``).  When *None*, the default
            provider is used.
        max_length: Maximum number of tokens to generate.
        cache_dir: Directory for storing converted models.  Defaults to
            ``~/.cache/locodellm/models``.
        chat_template: Chat template to wrap prompts with (e.g.
            ``"chatml"``).
        verbose: Verbosity level (0 = silent, 1+ = print progress).
        **search_options: Extra search options forwarded to generation
            (e.g. ``temperature``, ``top_k``, ``top_p``).

    Returns:
        A :class:`~locodellm.session.SessionState` with the generation
        results.  Access ``.text`` for the generated text.
    """
    session = _get_session(
        model_id,
        precision=precision,
        provider=provider,
        cache_dir=cache_dir,
        chat_template=chat_template,
        verbose=verbose,
    )

    current = session.new_session()
    current.generate(prompt, max_length=max_length, **search_options)
    return current
