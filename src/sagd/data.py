"""Instruction dataset for sequence-level KD on Dolly-15K.

Loads data from HuggingFace, tokenizes into [prompt | response] sequences,
provides masks distinguishing prompt (labels_mask=0) from response (labels_mask=1).
"""

from __future__ import annotations

from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def _format_prompt(instruction: str, context: str) -> str:
    """Format Dolly sample into prompt string (without response)."""
    parts = ["Below is an instruction that describes a task.\n"]
    parts.append(f"### Instruction:\n{instruction}\n")
    if context and context.strip():
        parts.append(f"### Input:\n{context}\n")
    parts.append("### Response:\n")
    return "\n".join(parts)


class InstructionDataset(Dataset):
    """Dolly-15K instruction dataset for knowledge distillation.

    Args:
        tokenizer: HuggingFace tokenizer.
        dataset_name: HF dataset name.
        max_seq_len: Maximum sequence length.
        max_samples: Limit number of samples (None = all).
        split: HuggingFace dataset split (default "train").
        seed: Random seed for shuffling.
        subset: Which subset after shuffled split: "train", "val", or "test".
            train = first N-1000, val = next 500, test = last 500.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_name: str = "databricks/databricks-dolly-15k",
        max_seq_len: int = 512,
        max_samples: int | None = None,
        split: str = "train",
        seed: int = 42,
        subset: str = "train",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        raw = load_dataset(dataset_name, split=split)
        raw = raw.shuffle(seed=seed)

        # Split into train / val / test
        n_total = len(raw)
        n_test = 500
        n_val = 500
        n_train = n_total - n_test - n_val

        if subset == "train":
            raw = raw.select(range(n_train))
        elif subset == "val":
            raw = raw.select(range(n_train, n_train + n_val))
        elif subset == "test":
            raw = raw.select(range(n_train + n_val, n_total))
        else:
            raise ValueError(f"Unknown subset: {subset}. Must be train/val/test")

        if max_samples is not None:
            raw = raw.select(range(min(max_samples, len(raw))))

        self.samples: list[dict[str, Any]] = []
        for i, row in enumerate(raw):
            prompt_str = _format_prompt(row["instruction"], row.get("context", ""))
            full_str = prompt_str + row["response"]

            prompt_enc = tokenizer(
                prompt_str, add_special_tokens=True, truncation=True,
                max_length=max_seq_len,
            )
            full_enc = tokenizer(
                full_str, add_special_tokens=True, truncation=True,
                max_length=max_seq_len, padding=False,
            )

            input_ids = full_enc["input_ids"]
            attention_mask = full_enc["attention_mask"]
            prompt_len = len(prompt_enc["input_ids"])
            seq_len = len(input_ids)  # (L,)

            # labels_mask: 0 for prompt, 1 for response
            labels_mask = [0] * min(prompt_len, seq_len) + [1] * max(0, seq_len - prompt_len)

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_mask": labels_mask,
                "index": i,
                "category": row.get("category", "unknown"),
                "instruction": row["instruction"],
                "response": row["response"],
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "input_ids": torch.tensor(s["input_ids"], dtype=torch.long),       # (L,)
            "attention_mask": torch.tensor(s["attention_mask"], dtype=torch.long),  # (L,)
            "labels_mask": torch.tensor(s["labels_mask"], dtype=torch.long),    # (L,)
            "index": torch.tensor(s["index"], dtype=torch.long),               # scalar
        }

    def get_metadata(self, idx: int) -> dict[str, str]:
        """Get non-tensor metadata for a sample."""
        s = self.samples[idx]
        return {
            "category": s["category"],
            "instruction": s["instruction"],
            "response": s["response"],
        }


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad to longest in batch, stack index."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    pad_id = 0  # padding value

    input_ids = []
    attention_mask = []
    labels_mask = []
    indices = []

    for b in batch:
        seq_len = b["input_ids"].size(0)  # (L_i,)
        pad_len = max_len - seq_len

        input_ids.append(torch.cat([b["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        attention_mask.append(torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        labels_mask.append(torch.cat([b["labels_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        indices.append(b["index"])

    return {
        "input_ids": torch.stack(input_ids),          # (B, L)
        "attention_mask": torch.stack(attention_mask),  # (B, L)
        "labels_mask": torch.stack(labels_mask),        # (B, L)
        "index": torch.stack(indices),                  # (B,)
    }
