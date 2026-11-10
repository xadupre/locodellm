"""Instruments public module exports in an explicitly isolated verification process."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
import random
import re
import runpy
import sys
import traceback


def main():
    """Records actual reference, export, runtime and comparison observations."""
    import numpy
    import onnx
    import onnxruntime
    import torch
    import yobx.torch.interpreter

    source, destination = map(Path, sys.argv[1:])
    torch.set_num_threads(1)
    torch.manual_seed(0)
    numpy.random.seed(0)
    random.seed(0)
    original = yobx.torch.interpreter.to_onnx
    signature = inspect.signature(original)
    observations = []
    export_errors = (
        AssertionError,
        RuntimeError,
        ValueError,
        TypeError,
        IndexError,
        KeyError,
        NotImplementedError,
    )
    ort_errors = (
        onnxruntime.capi.onnxruntime_pybind11_state.Fail,
        onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument,
        onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph,
        onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException,
        onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented,
    )

    def fingerprint(values):
        digest = hashlib.sha256()
        leaves, structure = torch.utils._pytree.tree_flatten(values)
        digest.update(str(structure).encode())
        for value in leaves:
            if isinstance(value, torch.Tensor):
                digest.update(str((value.shape, value.dtype)).encode())
                digest.update(
                    value.detach()
                    .cpu()
                    .contiguous()
                    .reshape(-1)
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                )
            else:
                digest.update(repr(value).encode())
        return digest.hexdigest()

    def instrumented(module, inputs, *args, **kwargs):
        record = {
            "reference_valid": False,
            "export_call_valid": False,
            "failure_valid": False,
            "stage": "pytorch",
            "converters": [],
            "error_type": None,
            "error": None,
            "tracing": bool(getattr(kwargs.get("export_options"), "tracing", False)),
        }
        observations.append(record)
        if not isinstance(module, torch.nn.Module) or not isinstance(inputs, tuple):
            raise TypeError("Verification requires to_onnx(torch.nn.Module, tuple_inputs).")
        with torch.no_grad():
            expected = copy.deepcopy(module)(*copy.deepcopy(inputs))
        expected, _ = torch.utils._pytree.tree_flatten(expected)
        if not expected or any(not isinstance(value, torch.Tensor) for value in expected):
            raise TypeError("Verification requires tensor outputs.")
        record["reference_valid"] = True
        record["inputs"] = fingerprint(inputs)
        record["reference_outputs"] = fingerprint(expected)
        record["stage"] = "export_call"
        signature.bind(module, inputs, *args, **kwargs)
        if args or kwargs.get("target_opset") != 18 or kwargs.get("optimize", True):
            raise ValueError("Verification requires target_opset=18, optimize=False.")
        for name, count in (("input_names", len(inputs)), ("output_names", len(expected))):
            names = kwargs.get(name)
            if names is not None and (
                not isinstance(names, (list, tuple))
                or len(names) != count
                or any(not isinstance(value, str) or not value for value in names)
                or len(set(names)) != count
            ):
                raise ValueError(f"{name} must uniquely name each tensor.")
        record["export_call_valid"] = True
        record["stage"] = "export"
        converters = set()

        def profile(frame, event, argument):
            if (
                event == "call"
                and "yobx" in frame.f_code.co_filename
                and frame.f_code.co_name.startswith("aten_")
            ):
                converters.add(frame.f_code.co_name)

        previous_profile = sys.getprofile()
        try:
            sys.setprofile(profile)
            model = original(module, inputs, **kwargs)
        except export_errors as error:
            record.update(
                failure_valid=any(
                    "yobx" in frame.filename and frame.name.startswith("aten_")
                    for frame in traceback.extract_tb(error.__traceback__)
                ),
                error_type=type(error).__name__,
                error=re.sub(r"0x[0-9a-fA-F]+", "0xADDRESS", str(error))[:2000],
            )
            raise
        finally:
            sys.setprofile(previous_profile)
            record["converters"] = sorted(converters)
        try:
            serialized = model.SerializeToString()
            proto = onnx.load_model_from_string(serialized)
            record["stage"] = "metadata"
            assert len(proto.graph.output) == len(expected), "Incorrect output count."
            for output, reference in zip(proto.graph.output, expected):
                tensor = output.type.tensor_type
                assert tensor.elem_type == onnx.helper.np_dtype_to_tensor_dtype(
                    reference.detach().cpu().numpy().dtype
                ), "Incorrect output metadata dtype."
                if tensor.HasField("shape"):
                    assert len(tensor.shape.dim) == reference.ndim, "Incorrect output rank."
                    for dimension, size in zip(tensor.shape.dim, reference.shape):
                        if dimension.HasField("dim_value"):
                            assert dimension.dim_value == size, "Incorrect static dimension."
            record["stage"] = "ort_load"
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            session = onnxruntime.InferenceSession(
                serialized, options, providers=["CPUExecutionProvider"]
            )
            record["stage"] = "ort_run"
            names = kwargs.get("input_names")
            if names is None:
                # Arbitrary nested/pruned input mapping cannot be inferred safely.
                record["stage"] = "unverified_input_mapping"
                return model
            if len(names) != len(inputs) or any(
                not isinstance(value, torch.Tensor) for value in inputs
            ):
                record["stage"] = "unverified_input_mapping"
                return model
            arrays = {name: value.detach().cpu().numpy() for name, value in zip(names, inputs)}
            if any(value.name not in arrays for value in session.get_inputs()):
                record["stage"] = "unverified_input_mapping"
                return model
            actual = session.run(
                None, {value.name: arrays[value.name] for value in session.get_inputs()}
            )
            record["stage"] = "comparison"
            assert len(actual) == len(expected), "Incorrect runtime output count."
            for got, reference in zip(actual, expected):
                want = reference.detach().cpu().numpy()
                assert got.shape == want.shape, "Incorrect runtime shape."
                assert got.dtype == want.dtype, "Incorrect runtime dtype."
                numpy.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-3)
        except (AssertionError, *ort_errors) as error:
            record.update(
                failure_valid=True,
                error_type=type(error).__name__,
                error=re.sub(r"0x[0-9a-fA-F]+", "0xADDRESS", str(error))[:2000],
            )
            raise
        return model

    yobx.torch.interpreter.to_onnx = instrumented
    try:
        runpy.run_path(str(source), run_name="__main__")
    finally:
        destination.write_text(json.dumps(observations), encoding="utf-8")


if __name__ == "__main__":
    main()
