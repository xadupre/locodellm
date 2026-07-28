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
        --model arnir0/Tiny-LLM --precision fp32 --provider cpu
"""

# %%
# Configuration
# -------------
#
# Default values can be overridden with ``--model``, ``--precision``, and
# ``--provider`` when running the script directly.

import argparse
import sys

_defaults = dict(model="arnir0/Tiny-LLM", precision="fp32", provider="cpu", verbose=1)

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
    _args = _parser.parse_args()
    MODEL_ID = _args.model
    PRECISION = _args.precision
    PROVIDER = _args.provider
    VERBOSE = _args.verbose
else:
    MODEL_ID = _defaults["model"]
    PRECISION = _defaults["precision"]
    PROVIDER = _defaults["provider"]
    VERBOSE = _defaults["verbose"]

print(f"MODEL_ID={MODEL_ID}, PRECISION={PRECISION}, PROVIDER={PROVIDER}, VERBOSE={VERBOSE}")

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
#         -m arnir0/Tiny-LLM \
#         -o docs/examples/tiny-llm/onnx_model \
#         -p fp32 \
#         -e cpu \
#         -c docs/examples/tiny-llm/cache

import os

from modelbuilder.builder import create_model

here = os.path.abspath(os.path.dirname(__file__)) if "__file__" in dir() else os.getcwd()
folder_name = MODEL_ID.replace("/", "_")
root_dir = os.path.join(here, folder_name)
output_dir = os.path.join(root_dir, "onnx_model")
cache_dir = os.path.join(root_dir, "cache")

if not os.path.exists(os.path.join(output_dir, "model.onnx")):
    create_model(
        model_name=MODEL_ID,
        input_path="",
        output_dir=output_dir,
        precision=PRECISION,
        execution_provider=PROVIDER,
        cache_dir=cache_dir,
    )
else:
    print("ONNX model already exists, skipping conversion.")

print(
    "Converted model files:", sorted(f for f in os.listdir(output_dir) if not f.startswith("."))
)

# %%
# First prompt
# ------------
#
# Ask the tiny LLM to write a Python function.
# ``verbose=True`` prints the main steps.

from locodellm.session import create_session

session = create_session(output_dir, verbose=VERBOSE)
session.generate('write a python function which returns "hello"', max_length=50)

print("First turn output:")
print(session.text)

# %%
# Second prompt (continuation)
# ----------------------------
#
# We call :meth:`~locodellm.session.SessionState.generate` again on the
# same session. The full token history is replayed so the model sees the
# complete context.

session.generate("change hello into bonjour", max_length=100)

print("Second turn output:")
print(session.text)
