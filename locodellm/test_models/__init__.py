"""Test model utilities for locodellm."""

from locodellm.test_models.mock_generate_model import create_mock_generate_model
from locodellm.test_models.tiny_model import create_tiny_model

_MODELS: dict[str, tuple[str, str]] = {
    "mock/generate": (
        "locodellm.test_models.mock_generate_model",
        "Mock LLM with hardcoded outputs for two Qwen2.5-Coder prompts.",
    ),
    "mock/tiny": (
        "locodellm.test_models.tiny_model",
        "Tiny model with zero weights for fast loading and smoke tests.",
    ),
}


def get_available_models() -> dict[str, str]:
    """Returns a dictionary of available mock model ids and descriptions.

    Returns:
        A mapping from model id to a short description.
    """
    return {name: desc for name, (_, desc) in _MODELS.items()}
