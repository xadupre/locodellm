"""Loads trainable policies and samples one-shot stdin/stdout solutions."""

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_policy(model_name, adapter, device, *, train):
    """Loads a frozen base plus LoRA, leaving model downloads explicit to the caller."""
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = (
        torch.bfloat16
        if (device.startswith("cuda") and torch.cuda.is_bf16_supported())
        else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=train)
    elif train:
        model = get_peft_model(
            model,
            LoraConfig(
                r=32,
                lora_alpha=64,
                lora_dropout=0.0,
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "up_proj",
                    "gate_proj",
                    "down_proj",
                ],
            ),
        )
    if train:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    model.eval()
    return model, tokenizer


def prompt_tokens(tokenizer, task, device):
    """Exposes only the problem statement, never private tests or references."""
    prompt = (
        "Write an efficient, correct Python program for this problem. "
        "Read from standard input and write to standard output. "
        "Return the program in a Python code block.\n\n" + task["problem"]
    )
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    if encoded["input_ids"].numel() == 0:
        raise ValueError(
            "The tokenizer produced an empty prompt; check model/tokenizer compatibility."
        )
    return encoded


def sample(model, tokenizer, prompt, args):
    """Samples without top-k/top-p truncation, using the same temperature as GRPO."""
    with torch.no_grad():
        tokens = model.generate(
            **prompt,
            do_sample=True,
            temperature=args.temperature,
            top_k=0,
            top_p=1.0,
            max_new_tokens=args.max_tokens,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    length = prompt["input_ids"].shape[1]
    return tokens, tokenizer.decode(tokens[0, length:], skip_special_tokens=True)
