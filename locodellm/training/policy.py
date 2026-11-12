"""Loads and saves the SLM-GRPO paper's LoRA policies."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]


def load_policy(model_name, adapter, *, train, device, lora_layers="all"):
    """Loads a frozen base with trainable or saved LoRA adapters."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model

    if lora_layers not in ("all", "last-half"):
        raise ValueError(f"Unknown LoRA layer selection: {lora_layers!r}")
    if adapter and lora_layers != "all":
        raise ValueError("--lora-layers cannot override a saved adapter's layer selection.")
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device).index or 0)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = (
        torch.bfloat16
        if device.startswith("cuda") and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    if adapter:
        model = PeftModel.from_pretrained(
            model, adapter, is_trainable=train, autocast_adapter_dtype=False
        )
    elif train:
        layers_to_transform = None
        if lora_layers == "last-half":
            config = model.config.get_text_config()
            count = config.num_hidden_layers
            layers_to_transform = list(range(count // 2, count))
        model = get_peft_model(
            model,
            LoraConfig(
                r=32,
                lora_alpha=64,
                lora_dropout=0.0,
                target_modules=TARGET_MODULES,
                task_type="CAUSAL_LM",
                layers_to_transform=layers_to_transform,
                layers_pattern="layers" if layers_to_transform is not None else None,
            ),
            autocast_adapter_dtype=False,
        )
    model.to(dtype=dtype)
    if train:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # Non-reentrant checkpoints track LoRA gradients without unfreezing the prefix.
        if hasattr(model, "_require_grads_hook") or getattr(model, "_require_grads_hooks", None):
            model.disable_input_require_grads()
    model.eval()
    return model, tokenizer


def save_adapter(model, tokenizer, path: Path, *, checkpoint_state=None):
    """Publishes an adapter snapshot only after all of its files are saved."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing adapter: {path}")
    with tempfile.TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as directory:
        model.save_pretrained(directory)
        tokenizer.save_pretrained(directory)
        if checkpoint_state is not None:
            Path(directory, "checkpoint.json").write_text(
                json.dumps(checkpoint_state, indent=2) + "\n"
            )
        Path(directory).rename(path)
