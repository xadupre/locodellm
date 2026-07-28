"""Creates a mock LLM model directory for plot_generate.py.

The model produces the exact same outputs as the
Qwen/Qwen2.5-Coder-0.5B-Instruct model for the two prompts
used in ``docs/examples/plot_generate.py``, using a hardcoded
token lookup table instead of real transformer weights.

The two prompts (with ``do_sample=False, top_k=1``):

1. ``'write a python function which returns "hello"'``
2. ``"change hello into bonjour"``
"""

from __future__ import annotations

import json
import os

# Architecture constants – minimal, just enough for onnxruntime-genai.
VOCAB_SIZE = 151936
NUM_KEY_VALUE_HEADS = 2
HEAD_SIZE = 64
NUM_HIDDEN_LAYERS = 1
CONTEXT_LENGTH = 512

# EOS / special token ids matching the Qwen tokenizer.
EOS_TOKEN_ID = 151645
ENDOFTEXT_TOKEN_ID = 151643

# ──────────────────────────────────────────────────────────────────────
# Hardcoded token sequences captured from the real model with
# do_sample=False, top_k=1, top_p=1.0, temperature=1.0.
# ──────────────────────────────────────────────────────────────────────

# Turn 1 prompt (ChatML-wrapped):
#   <|im_start|>user\nwrite a python function which returns "hello"
#   <|im_end|>\n<|im_start|>assistant\n
# 17 prompt tokens + 39 generated tokens = 56 total
TURN1_TOKENS = [
    151644,
    872,
    198,
    4934,
    264,
    10135,
    729,
    892,
    4675,
    330,
    14990,
    1,
    151645,
    198,
    151644,
    77091,
    198,
    # --- generated tokens start here (index 17) ---
    8420,
    594,
    264,
    4285,
    13027,
    729,
    429,
    4675,
    330,
    14990,
    51418,
    73594,
    12669,
    198,
    750,
    23811,
    3932,
    262,
    470,
    330,
    14990,
    698,
    13874,
    19324,
    2610,
    646,
    1618,
    419,
    729,
    323,
    432,
    686,
    470,
    279,
    914,
    330,
    14990,
    3263,
    151645,
]

TURN1_PROMPT_LEN = 17

# Turn 2: context = turn1_tokens (56) + prompt2 tokens (15) = 71
# prompt2 (ChatML-wrapped):
#   <|im_end|>\n<|im_start|>user\nchange hello into bonjour
#   <|im_end|>\n<|im_start|>assistant\n
# 71 context tokens + 43 generated tokens = 114 total
TURN2_TOKENS = [
    # --- turn 1 tokens (0-55) ---
    151644,
    872,
    198,
    4934,
    264,
    10135,
    729,
    892,
    4675,
    330,
    14990,
    1,
    151645,
    198,
    151644,
    77091,
    198,
    8420,
    594,
    264,
    4285,
    13027,
    729,
    429,
    4675,
    330,
    14990,
    51418,
    73594,
    12669,
    198,
    750,
    23811,
    3932,
    262,
    470,
    330,
    14990,
    698,
    13874,
    19324,
    2610,
    646,
    1618,
    419,
    729,
    323,
    432,
    686,
    470,
    279,
    914,
    330,
    14990,
    3263,
    151645,
    # --- prompt 2 tokens (56-70) ---
    151645,
    198,
    151644,
    872,
    198,
    3373,
    23811,
    1119,
    7814,
    29262,
    151645,
    198,
    151644,
    77091,
    198,
    # --- generated tokens start here (index 71) ---
    8420,
    594,
    264,
    4285,
    13027,
    729,
    429,
    4675,
    330,
    5970,
    29262,
    51418,
    73594,
    12669,
    198,
    750,
    7814,
    29262,
    3932,
    262,
    470,
    330,
    5970,
    29262,
    698,
    13874,
    19324,
    2610,
    646,
    1618,
    419,
    729,
    323,
    432,
    686,
    470,
    279,
    914,
    330,
    5970,
    29262,
    3263,
    151645,
]

TURN2_CONTEXT_LEN = 71


def _build_lookup_table():
    """Builds a 1-D lookup table mapping *total_seq* → *next_token_id*.

    At generation step *s* the model has seen *total_seq* tokens in total
    (prompt + previously generated).  ``lookup[total_seq]`` gives the token
    that the model must predict next.

    Positions outside the two known generation ranges default to
    :data:`EOS_TOKEN_ID` so that the generator stops immediately.
    """
    import numpy as np

    max_pos = len(TURN2_TOKENS)  # 114
    lookup = np.full(max_pos + 1, EOS_TOKEN_ID, dtype=np.int64)

    for pos in range(TURN1_PROMPT_LEN, len(TURN1_TOKENS)):
        lookup[pos] = TURN1_TOKENS[pos]

    for pos in range(TURN2_CONTEXT_LEN, len(TURN2_TOKENS)):
        lookup[pos] = TURN2_TOKENS[pos]

    return lookup


def make_decoder_model():
    """Builds a minimal ONNX decoder graph with hardcoded token predictions.

    Inputs:
        input_ids                 [batch, seq_len]            INT64
        attention_mask            [batch, total_seq_len]      INT64
        past_key_values.0.key     [batch, H, past_seq, D]    FLOAT
        past_key_values.0.value   [batch, H, past_seq, D]    FLOAT

    Outputs:
        logits                    [batch, seq_len, vocab]     FLOAT
        present.0.key             [batch, H, total_seq, D]   FLOAT
        present.0.value           [batch, H, total_seq, D]   FLOAT

    The logits are computed by looking up the expected next token from a
    hardcoded table indexed by ``past_seq + seq_len`` (the total number
    of tokens seen so far).  A one-hot vector scaled to 100.0 is returned
    so that greedy / near-greedy decoding always picks the right token.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    lookup = _build_lookup_table()

    # ── graph inputs ──────────────────────────────────────────────────
    input_ids = helper.make_tensor_value_info(
        "input_ids", TensorProto.INT64, ["batch", "seq_len"]
    )
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, ["batch", "total_seq_len"]
    )
    kv_inputs = []
    for kind in ("key", "value"):
        kv_inputs.append(
            helper.make_tensor_value_info(
                f"past_key_values.0.{kind}",
                TensorProto.FLOAT,
                ["batch", NUM_KEY_VALUE_HEADS, "past_seq", HEAD_SIZE],
            )
        )

    # ── initializers / constants ─────────────────────────────────────
    lookup_init = numpy_helper.from_array(lookup, name="lookup_table")
    on_off_init = numpy_helper.from_array(
        np.array([0.0, 100.0], dtype=np.float32), name="on_off_values"
    )

    initializers = [lookup_init, on_off_init]

    # ── nodes ────────────────────────────────────────────────────────
    nodes = []

    # --- compute total_seq = past_seq + seq_len ----------------------
    nodes.append(helper.make_node("Shape", ["past_key_values.0.key"], ["kv_shape"]))
    nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["axis0"],
            value=helper.make_tensor("axis0", TensorProto.INT64, [], [0]),
        )
    )
    nodes.append(
        helper.make_node(
            "Constant", [], ["idx2"], value=helper.make_tensor("idx2", TensorProto.INT64, [], [2])
        )
    )
    nodes.append(
        helper.make_node(
            "Constant", [], ["idx1"], value=helper.make_tensor("idx1", TensorProto.INT64, [], [1])
        )
    )
    nodes.append(helper.make_node("Gather", ["kv_shape", "idx2"], ["past_seq"], axis=0))

    nodes.append(helper.make_node("Shape", ["input_ids"], ["ids_shape"]))
    nodes.append(helper.make_node("Gather", ["ids_shape", "idx1"], ["seq_len"], axis=0))

    nodes.append(helper.make_node("Add", ["past_seq", "seq_len"], ["total_seq"]))

    # --- clamp total_seq to lookup table bounds ----------------------
    max_idx = len(lookup) - 1
    nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["max_idx"],
            value=helper.make_tensor("max_idx", TensorProto.INT64, [], [max_idx]),
        )
    )
    nodes.append(helper.make_node("Min", ["total_seq", "max_idx"], ["clamped_pos"]))

    # --- lookup next token id ----------------------------------------
    nodes.append(
        helper.make_node("Gather", ["lookup_table", "clamped_pos"], ["next_token_id"], axis=0)
    )

    # --- build one-hot logits [1, vocab_size] ------------------------
    nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["one_shape"],
            value=helper.make_tensor("one_shape", TensorProto.INT64, [1], [1]),
        )
    )
    nodes.append(helper.make_node("Reshape", ["next_token_id", "one_shape"], ["next_token_1d"]))
    nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["vocab_depth"],
            value=helper.make_tensor("vocab_depth", TensorProto.INT64, [], [VOCAB_SIZE]),
        )
    )
    nodes.append(
        helper.make_node(
            "OneHot", ["next_token_1d", "vocab_depth", "on_off_values"], ["onehot"], axis=1
        )
    )
    # onehot shape: [1, VOCAB_SIZE]

    # --- expand to [batch, seq_len, vocab_size] ----------------------
    nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["unsqueeze_axes"],
            value=helper.make_tensor("unsqueeze_axes", TensorProto.INT64, [1], [0]),
        )
    )
    nodes.append(helper.make_node("Unsqueeze", ["onehot", "unsqueeze_axes"], ["logits_1x1xV"]))
    # logits_1x1xV shape: [1, 1, VOCAB_SIZE]

    nodes.append(helper.make_node("Gather", ["ids_shape", "axis0"], ["batch_dim"], axis=0))
    nodes.append(helper.make_node("Reshape", ["batch_dim", "one_shape"], ["batch_1d"]))
    nodes.append(helper.make_node("Reshape", ["seq_len", "one_shape"], ["seq_1d"]))
    nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["vocab_1d"],
            value=helper.make_tensor("vocab_1d", TensorProto.INT64, [1], [VOCAB_SIZE]),
        )
    )
    nodes.append(
        helper.make_node("Concat", ["batch_1d", "seq_1d", "vocab_1d"], ["logits_shape"], axis=0)
    )
    nodes.append(helper.make_node("Expand", ["logits_1x1xV", "logits_shape"], ["logits"]))

    # --- KV cache: concat zeros along sequence axis ------------------
    nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["kv_heads_1d"],
            value=helper.make_tensor(
                "kv_heads_1d", TensorProto.INT64, [1], [NUM_KEY_VALUE_HEADS]
            ),
        )
    )
    nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["head_1d"],
            value=helper.make_tensor("head_1d", TensorProto.INT64, [1], [HEAD_SIZE]),
        )
    )
    nodes.append(
        helper.make_node(
            "Concat", ["batch_1d", "kv_heads_1d", "seq_1d", "head_1d"], ["new_kv_shape"], axis=0
        )
    )
    nodes.append(
        helper.make_node(
            "ConstantOfShape",
            ["new_kv_shape"],
            ["new_kv_zeros"],
            value=helper.make_tensor("zero_f", TensorProto.FLOAT, [1], [0.0]),
        )
    )
    for kind in ("key", "value"):
        nodes.append(
            helper.make_node(
                "Concat",
                [f"past_key_values.0.{kind}", "new_kv_zeros"],
                [f"present.0.{kind}"],
                axis=2,
            )
        )

    # ── graph outputs ────────────────────────────────────────────────
    logits_out = helper.make_tensor_value_info(
        "logits", TensorProto.FLOAT, ["batch", "seq_len", VOCAB_SIZE]
    )
    kv_outputs = []
    for kind in ("key", "value"):
        kv_outputs.append(
            helper.make_tensor_value_info(
                f"present.0.{kind}",
                TensorProto.FLOAT,
                ["batch", NUM_KEY_VALUE_HEADS, "total_seq", HEAD_SIZE],
            )
        )

    graph = helper.make_graph(
        nodes,
        "mock_decoder",
        [input_ids, attention_mask, *kv_inputs],
        [logits_out, *kv_outputs],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def make_genai_config() -> dict:
    """Returns a ``genai_config.json`` dict compatible with the Qwen tokenizer."""
    return {
        "model": {
            "bos_token_id": ENDOFTEXT_TOKEN_ID,
            "eos_token_id": [EOS_TOKEN_ID, ENDOFTEXT_TOKEN_ID],
            "pad_token_id": ENDOFTEXT_TOKEN_ID,
            "vocab_size": VOCAB_SIZE,
            "context_length": CONTEXT_LENGTH,
            "type": "qwen2",
            "decoder": {
                "filename": "model.onnx",
                "hidden_size": NUM_KEY_VALUE_HEADS * HEAD_SIZE,
                "head_size": HEAD_SIZE,
                "num_attention_heads": NUM_KEY_VALUE_HEADS,
                "num_key_value_heads": NUM_KEY_VALUE_HEADS,
                "num_hidden_layers": NUM_HIDDEN_LAYERS,
                "inputs": {
                    "input_ids": "input_ids",
                    "attention_mask": "attention_mask",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                },
                "session_options": {"provider_options": []},
            },
        },
        "search": {
            "do_sample": False,
            "max_length": CONTEXT_LENGTH,
            "num_beams": 1,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
            "past_present_share_buffer": False,
        },
    }


def _save_tokenizer(output_dir: str) -> None:
    """Downloads the Qwen tokenizer and saves it to *output_dir*."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    tokenizer.save_pretrained(output_dir)


def create_mock_generate_model(output_dir: str) -> str:
    """Creates a mock model directory that reproduces plot_generate.py outputs.

    The directory will contain:

    * ``model.onnx`` – a lookup-table-based ONNX decoder graph.
    * ``genai_config.json`` – onnxruntime-genai configuration.
    * ``tokenizer.json`` – the Qwen tokenizer (downloaded from HuggingFace).
    * ``tokenizer_config.json`` – tokenizer metadata.

    Args:
        output_dir: Path where the model directory is created.

    Returns:
        The absolute path to *output_dir*.
    """
    import onnx

    os.makedirs(output_dir, exist_ok=True)

    onnx.save(make_decoder_model(), os.path.join(output_dir, "model.onnx"))

    with open(os.path.join(output_dir, "genai_config.json"), "w") as f:
        json.dump(make_genai_config(), f, indent=4)

    _save_tokenizer(output_dir)

    return os.path.abspath(output_dir)
