"""
Generate text with a tiny LLM
==============================

This example downloads a pretrained model from the HuggingFace Hub,
converts it to ONNX using `mbext <https://github.com/xadupre/mbext>`_,
and runs two consecutive prompts with
:meth:`locodellm.session.SessionState.generate`.

The model, precision, and execution provider can be changed from the
command line::

    python docs/examples/plot_generate.py \\
        --model Qwen/Qwen2.5-Coder-0.5B-Instruct --precision fp32 --provider cpu
"""

# %%
# Configuration
# -------------
#
# Default values can be overridden with ``--model``, ``--precision``, and
# ``--provider`` when running the script directly.
#
# When the environment variable ``UNITTEST_GOING`` is set to ``"1"`` (as done
# while running the unit tests or building the documentation), the example
# skips the download and conversion of the real model and uses a small mock
# ONNX model instead, so the example stays cheap and offline-friendly.

import argparse
import os
import sys

UNITTEST_GOING = os.environ.get("UNITTEST_GOING") == "1"

# Ensure the project root is on sys.path when running the script directly.
if "__file__" in dir():
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

_defaults = dict(
    model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    precision="fp32",
    provider="cpu",
    verbose=1,
    chat_template="chatml",
)

# sphinx-gallery passes no CLI args; parse only when run as a script.
if "__file__" in dir():
    _parser = argparse.ArgumentParser(description="Generate text with a tiny LLM.")
    _parser.add_argument("--model", default=_defaults["model"], help="HuggingFace model id.")
    _parser.add_argument(
        "--precision",
        default=_defaults["precision"],
        help="Conversion precision (fp32, fp16, int4).",
    )
    _parser.add_argument(
        "--provider", default=_defaults["provider"], help="Execution provider (cpu, cuda)."
    )
    _parser.add_argument(
        "--verbose", type=int, default=_defaults["verbose"], help="Verbosity level (0=silent)."
    )
    _parser.add_argument(
        "--chat-template",
        default=_defaults["chat_template"],
        help="Chat template (chatml, or empty for none).",
    )
    _args = _parser.parse_args()
    MODEL_ID = _args.model
    PRECISION = _args.precision
    PROVIDER = _args.provider
    VERBOSE = _args.verbose
    CHAT_TEMPLATE = _args.chat_template or None
else:
    MODEL_ID = _defaults["model"]
    PRECISION = _defaults["precision"]
    PROVIDER = _defaults["provider"]
    VERBOSE = _defaults["verbose"]
    CHAT_TEMPLATE = _defaults["chat_template"]

print(
    f"MODEL_ID={MODEL_ID}, PRECISION={PRECISION}, PROVIDER={PROVIDER}, "
    f"VERBOSE={VERBOSE}, CHAT_TEMPLATE={CHAT_TEMPLATE}"
)

# %%
# Download and convert the model
# -------------------------------
#
# We download the model from the HuggingFace Hub and convert it with
# :func:`modelbuilder.builder.create_model`.  All artefacts are written
# under a local subfolder so the weights survive the build and can be
# reused.
#
# The equivalent command line is:
#
# .. code-block:: bash
#
#     python -m modelbuilder.builder \
#         -m Qwen/Qwen2.5-Coder-0.5B-Instruct \
#         -o docs/examples/Qwen_Qwen2.5-Coder-0.5B-Instruct/onnx_model \
#         -p fp32 \
#         -e cpu \
#         -c docs/examples/Qwen_Qwen2.5-Coder-0.5B-Instruct/cache
#
# Under ``UNITTEST_GOING=1`` this step is replaced by
# :func:`locodellm.test_models.create_mock_generate_model`, which builds a
# tiny lookup-table ONNX model reproducing the exact outputs of this example.

from locodellm import extract_code  # noqa: E402
from locodellm.session import create_session  # noqa: E402

here = os.path.abspath(os.path.dirname(__file__)) if "__file__" in dir() else os.getcwd()
folder_name = MODEL_ID.replace("/", "_")
root_dir = os.path.join(here, folder_name)
output_dir = os.path.join(root_dir, "onnx_model")
cache_dir = os.path.join(root_dir, "cache")

if os.path.exists(os.path.join(output_dir, "model.onnx")):
    print("ONNX model already exists, skipping conversion.")
elif UNITTEST_GOING:
    from locodellm.test_models import create_mock_generate_model

    create_mock_generate_model(output_dir)
else:
    from modelbuilder.builder import create_model

    create_model(
        model_name=MODEL_ID,
        input_path="",
        output_dir=output_dir,
        precision=PRECISION,
        execution_provider=PROVIDER,
        cache_dir=cache_dir,
    )

print(
    "Converted model files:", sorted(f for f in os.listdir(output_dir) if not f.startswith("."))
)

# %%
# First prompt
# ------------
#
# Ask the model to write a Python function that returns ``"hello"``.


session = create_session(output_dir, verbose=VERBOSE, chat_template=CHAT_TEMPLATE)
session.generate('write a python function which returns "hello"', max_length=200)

print("First turn output:")
print(extract_code(session.text))

# %%
# Second prompt (continuation)
# ----------------------------
#
# We call :meth:`~locodellm.session.SessionState.generate` again on the
# same session. The full token history is replayed so the model sees the
# complete context.

session.generate("change hello into bonjour", max_length=500)

print("Second turn output:")
print(extract_code(session.text))
