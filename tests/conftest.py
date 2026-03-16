"""Shared test fixtures using a tiny GPT-2 model."""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="session")
def tiny_model() -> nn.Module:
    """Load a tiny GPT-2 model for testing."""
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    model.eval()
    return model


@pytest.fixture(scope="session")
def tiny_tokenizer():
    """Load tokenizer for tiny GPT-2."""
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


@pytest.fixture
def sample_batch() -> dict[str, torch.Tensor]:
    """Create a simple batch for testing (B=2, L=10)."""
    B, L = 2, 10
    return {
        "input_ids": torch.randint(0, 100, (B, L)),       # (B, L)
        "attention_mask": torch.ones(B, L, dtype=torch.long),  # (B, L)
        "labels_mask": torch.tensor([
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # 5 prompt, 5 response
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],  # 3 prompt, 7 response
        ], dtype=torch.long),                               # (B, L)
        "index": torch.tensor([0, 1], dtype=torch.long),   # (B,)
    }


@pytest.fixture
def sample_batch_with_padding() -> dict[str, torch.Tensor]:
    """Batch with padding (B=2, L=10, second sample padded)."""
    B, L = 2, 10
    return {
        "input_ids": torch.randint(0, 100, (B, L)),
        "attention_mask": torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # 3 padding positions
        ], dtype=torch.long),
        "labels_mask": torch.tensor([
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],  # padding positions
        ], dtype=torch.long),
        "index": torch.tensor([0, 1], dtype=torch.long),
    }
