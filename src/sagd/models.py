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
) -> tuple[nn.Module, PreTrainedTokenizer]:
    """Load teacher model in eval mode.

    Args:
        model_name: HuggingFace model name.
        device: Target device.
        dtype: Model dtype (default float16 for memory efficiency).

    Returns:
        (model, tokenizer) with model in eval mode, all params frozen.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _ensure_pad_token(tokenizer)

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
) -> tuple[nn.Module, PreTrainedTokenizer]:
    """Load student model in train mode (float32).

    Args:
        model_name: HuggingFace model name.
        device: Target device.

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

    model.train()
    return model, tokenizer
