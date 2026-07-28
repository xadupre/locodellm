"""Creates a tiny dummy LLM model directory for testing.

The model has uninitialized (zero) weights and produces meaningless output,
but the ONNX graph and configuration files are valid enough for
onnxruntime-genai to load and run a generation loop.
"""

from __future__ import annotations

import json
import os


# Architecture constants kept deliberately small so that the model
# loads and runs in a fraction of a second with negligible memory.
VOCAB_SIZE = 32
HIDDEN_SIZE = 64
NUM_ATTENTION_HEADS = 4
NUM_KEY_VALUE_HEADS = 4
HEAD_SIZE = HIDDEN_SIZE // NUM_ATTENTION_HEADS  # 16
NUM_HIDDEN_LAYERS = 1
CONTEXT_LENGTH = 128


def make_decoder_model():
    """Builds a minimal ONNX decoder graph.

    Inputs:
        input_ids            [batch, seq_len]            INT64
        attention_mask       [batch, total_seq_len]      INT64
        position_ids         [batch, seq_len]            INT64
        past_key_values.L.key   [batch, H, past_seq, D] FLOAT
        past_key_values.L.value [batch, H, past_seq, D] FLOAT

    Outputs:
        logits               [batch, seq_len, vocab]     FLOAT
        present.L.key        [batch, H, total_seq, D]    FLOAT
        present.L.value      [batch, H, total_seq, D]    FLOAT

    The logits are computed by a zero-weight embedding lookup followed by
    a zero-weight projection.  The KV cache is grown by padding one zero
    slice on the sequence axis.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    # --- graph inputs ---
    input_ids = helper.make_tensor_value_info(
        "input_ids", TensorProto.INT64, ["batch", "seq_len"]
    )
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, ["batch", "total_seq_len"]
    )
    position_ids = helper.make_tensor_value_info(
        "position_ids", TensorProto.INT64, ["batch", "seq_len"]
    )

    kv_inputs = []
    for i in range(NUM_HIDDEN_LAYERS):
        for kind in ("key", "value"):
            kv_inputs.append(
                helper.make_tensor_value_info(
                    f"past_key_values.{i}.{kind}",
                    TensorProto.FLOAT,
                    ["batch", NUM_KEY_VALUE_HEADS, "past_seq", HEAD_SIZE],
                )
            )

    # --- initializers (all zeros / constants) ---
    embed_weight = numpy_helper.from_array(
        np.zeros((VOCAB_SIZE, HIDDEN_SIZE), dtype=np.float32), name="embed_weight"
    )
    proj_weight = numpy_helper.from_array(
        np.zeros((HIDDEN_SIZE, VOCAB_SIZE), dtype=np.float32), name="proj_weight"
    )

    initializers = [embed_weight, proj_weight]

    # --- nodes ---
    nodes = [
        # Embedding lookup: input_ids → [batch, seq, hidden]
        helper.make_node("Gather", ["embed_weight", "input_ids"], ["embedded"], axis=0),
        # Project to vocab: [batch, seq, hidden] @ [hidden, vocab] → logits
        helper.make_node("MatMul", ["embedded", "proj_weight"], ["logits"]),
    ]

    # Build a zero KV-slice whose sequence dimension equals the input
    # seq_len so that Concat(past, new_kv) produces the correct total_seq
    # shape during both prefill (past is empty) and decode (past has
    # the accumulated sequence).
    #
    # new_kv_shape = [batch, NUM_KEY_VALUE_HEADS, seq_len, HEAD_SIZE]
    nodes.extend([
        helper.make_node("Shape", ["input_ids"], ["ids_shape"]),
        helper.make_node(
            "Constant", [], ["idx_0"],
            value=helper.make_tensor("idx_0", TensorProto.INT64, [], [0]),
        ),
        helper.make_node(
            "Constant", [], ["idx_1"],
            value=helper.make_tensor("idx_1", TensorProto.INT64, [], [1]),
        ),
        helper.make_node("Gather", ["ids_shape", "idx_0"], ["batch_dim"], axis=0),
        helper.make_node("Gather", ["ids_shape", "idx_1"], ["seq_dim"], axis=0),
        helper.make_node(
            "Constant", [], ["one_shape"],
            value=helper.make_tensor("one_shape", TensorProto.INT64, [1], [1]),
        ),
        helper.make_node("Reshape", ["batch_dim", "one_shape"], ["batch_1d"]),
        helper.make_node("Reshape", ["seq_dim", "one_shape"], ["seq_1d"]),
        helper.make_node(
            "Constant", [], ["kv_heads_dim"],
            value=helper.make_tensor(
                "kv_heads_dim", TensorProto.INT64, [1], [NUM_KEY_VALUE_HEADS]
            ),
        ),
        helper.make_node(
            "Constant", [], ["head_dim"],
            value=helper.make_tensor(
                "head_dim", TensorProto.INT64, [1], [HEAD_SIZE]
            ),
        ),
        # [batch, NUM_KEY_VALUE_HEADS, seq_len, HEAD_SIZE]
        helper.make_node(
            "Concat", ["batch_1d", "kv_heads_dim", "seq_1d", "head_dim"],
            ["new_kv_shape"], axis=0,
        ),
        helper.make_node(
            "ConstantOfShape", ["new_kv_shape"], ["new_kv_zeros"],
            value=helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0]),
        ),
    ])

    # Concat past KV with the new zero slice along the sequence axis (axis=2)
    for i in range(NUM_HIDDEN_LAYERS):
        for kind in ("key", "value"):
            nodes.append(
                helper.make_node(
                    "Concat",
                    [f"past_key_values.{i}.{kind}", "new_kv_zeros"],
                    [f"present.{i}.{kind}"],
                    axis=2,
                )
            )

    # --- graph outputs ---
    logits_out = helper.make_tensor_value_info(
        "logits", TensorProto.FLOAT, ["batch", "seq_len", VOCAB_SIZE]
    )
    kv_outputs = []
    for i in range(NUM_HIDDEN_LAYERS):
        for kind in ("key", "value"):
            kv_outputs.append(
                helper.make_tensor_value_info(
                    f"present.{i}.{kind}",
                    TensorProto.FLOAT,
                    ["batch", NUM_KEY_VALUE_HEADS, "total_seq", HEAD_SIZE],
                )
            )

    graph = helper.make_graph(
        nodes,
        "decoder",
        [input_ids, attention_mask, position_ids, *kv_inputs],
        [logits_out, *kv_outputs],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def make_genai_config() -> dict:
    """Returns a minimal ``genai_config.json`` dict."""
    return {
        "model": {
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "vocab_size": VOCAB_SIZE,
            "context_length": CONTEXT_LENGTH,
            "type": "llama",
            "decoder": {
                "filename": "model.onnx",
                "hidden_size": HIDDEN_SIZE,
                "head_size": HEAD_SIZE,
                "num_attention_heads": NUM_ATTENTION_HEADS,
                "num_key_value_heads": NUM_KEY_VALUE_HEADS,
                "num_hidden_layers": NUM_HIDDEN_LAYERS,
                "inputs": {
                    "input_ids": "input_ids",
                    "attention_mask": "attention_mask",
                    "position_ids": "position_ids",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                },
                "session_options": {
                    "provider_options": [],
                },
            },
        },
        "search": {
            "do_sample": False,
            "max_length": 32,
            "num_beams": 1,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
            "past_present_share_buffer": False,
        },
    }


def make_tokenizer_json() -> dict:
    """Returns a minimal HuggingFace ``tokenizer.json`` with a tiny BPE vocabulary."""
    vocab = {str(i): i for i in range(3, VOCAB_SIZE)}
    vocab["<unk>"] = 0
    vocab["<s>"] = 1
    vocab["</s>"] = 2
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 0,
                "content": "<unk>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 1,
                "content": "<s>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 2,
                "content": "</s>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": [],
        },
    }


def make_tokenizer_config() -> dict:
    """Returns a minimal ``tokenizer_config.json``."""
    return {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "model_max_length": CONTEXT_LENGTH,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
    }


def create_tiny_model(output_dir: str) -> str:
    """Creates a tiny dummy LLM model directory at *output_dir*.

    The directory will contain:

    * ``model.onnx`` – a valid but zero-weight ONNX decoder graph.
    * ``genai_config.json`` – onnxruntime-genai configuration.
    * ``tokenizer.json`` – minimal BPE tokenizer.
    * ``tokenizer_config.json`` – tokenizer metadata.

    Args:
        output_dir: Path where the model directory is created.  The directory
            and any missing parents are created automatically.

    Returns:
        The absolute path to *output_dir*.
    """
    import onnx

    os.makedirs(output_dir, exist_ok=True)

    onnx.save(make_decoder_model(), os.path.join(output_dir, "model.onnx"))

    for filename, data in [
        ("genai_config.json", make_genai_config()),
        ("tokenizer.json", make_tokenizer_json()),
        ("tokenizer_config.json", make_tokenizer_config()),
    ]:
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump(data, f, indent=4)

    return os.path.abspath(output_dir)
