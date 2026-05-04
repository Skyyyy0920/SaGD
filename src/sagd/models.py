"""Model loading utilities for teacher and student LLMs."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer


def _ensure_pad_token(tokenizer: PreTrainedTokenizer) -> None:
    """Set pad_token = eos_token if not already set."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


def load_teacher(
    model_name: str,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float16,
    load_in_8bit: bool = False,
) -> tuple[nn.Module, PreTrainedTokenizer]:
    """Load teacher model in eval mode.

    Args:
        model_name: HuggingFace model name.
        device: Target device. Ignored when ``load_in_8bit=True``
            (handled by accelerate's device_map).
        dtype: Model dtype (default float16). Ignored when ``load_in_8bit=True``.
        load_in_8bit: If True, load via bitsandbytes int8 quantization to fit
            on smaller GPUs (e.g. 24GB 3090). Linear weights stored as int8;
            forward dequantizes per-matmul to fp16. Adds bitsandbytes dep.

    Returns:
        (model, tokenizer) with model in eval mode, all params frozen.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _ensure_pad_token(tokenizer)

    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": device},
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, tokenizer


def load_student(
    model_name: str,
    device: str = "cuda:0",
    gradient_checkpointing: bool = False,
) -> tuple[nn.Module, PreTrainedTokenizer]:
    """Load student model in train mode (float32).

    Args:
        model_name: HuggingFace model name.
        device: Target device.
        gradient_checkpointing: If True, enable activation recomputation to
            reduce activation memory (useful on 24GB GPUs). Trades ~30% extra
            compute for ~50% activation memory savings.

    Returns:
        (model, tokenizer) with model in train mode.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _ensure_pad_token(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    model.train()
    return model, tokenizer
