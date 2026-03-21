"""Datasets for sequence-level KD: Dolly-15K and SQuAD 2.0.

Loads data from HuggingFace, tokenizes into [prompt | response] sequences,
provides masks distinguishing prompt (labels_mask=0) from response (labels_mask=1).

SQuAD 2.0 additionally tracks answer span token positions for evidence
concentration evaluation.
"""

from __future__ import annotations

import re
import string
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


def _format_squad_prompt(context: str, question: str) -> str:
    """Format SQuAD sample into prompt string (without answer)."""
    return (
        "Answer the question based on the context below.\n\n"
        f"### Context:\n{context}\n\n"
        f"### Question:\n{question}\n\n"
        "### Answer:\n"
    )


def normalize_answer(s: str) -> str:
    """Normalize answer string for EM/F1 evaluation (SQuAD standard)."""
    s = s.lower()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    s = " ".join(s.split())
    return s


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


class SquadDataset(Dataset):
    """SQuAD 2.0 extractive QA dataset for knowledge distillation.

    Each sample formats context + question as prompt, answer as response.
    Tracks answer span token positions for evidence concentration evaluation.

    Unanswerable questions (SQuAD 2.0) are filtered out.

    Args:
        tokenizer: HuggingFace tokenizer (must be a fast tokenizer for offset mapping).
        dataset_name: HF dataset name (default ``"rajpurkar/squad_v2"``).
        max_seq_len: Maximum sequence length.
        max_samples: Limit number of samples (None = all).
        seed: Random seed for shuffling.
        subset: ``"train"`` uses HF train split; ``"val"``/``"test"`` split
            HF validation set in half (first half = val, second half = test).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_name: str = "rajpurkar/squad_v2",
        max_seq_len: int = 512,
        max_samples: int | None = None,
        seed: int = 42,
        subset: str = "train",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Load and filter unanswerable questions
        if subset == "train":
            raw = load_dataset(dataset_name, split="train")
        elif subset in ("val", "test"):
            raw = load_dataset(dataset_name, split="validation")
        else:
            raise ValueError(f"Unknown subset: {subset}. Must be train/val/test")

        # Filter unanswerable (empty answers)
        raw = raw.filter(lambda x: len(x["answers"]["text"]) > 0)
        raw = raw.shuffle(seed=seed)

        # Split validation into val / test halves
        if subset in ("val", "test"):
            n_half = len(raw) // 2
            if subset == "val":
                raw = raw.select(range(n_half))
            else:
                raw = raw.select(range(n_half, len(raw)))

        if max_samples is not None:
            raw = raw.select(range(min(max_samples, len(raw))))

        self.samples: list[dict[str, Any]] = []
        n_span_mapped = 0

        for i, row in enumerate(raw):
            context = row["context"]
            question = row["question"]
            answer_text = row["answers"]["text"][0]
            answer_start_char = row["answers"]["answer_start"][0]  # char offset in context

            prompt_str = _format_squad_prompt(context, question)
            full_str = prompt_str + answer_text

            # Tokenize prompt separately to get prompt_len
            prompt_enc = tokenizer(
                prompt_str, add_special_tokens=True, truncation=True,
                max_length=max_seq_len,
            )
            full_enc = tokenizer(
                full_str, add_special_tokens=True, truncation=True,
                max_length=max_seq_len, padding=False,
                return_offsets_mapping=True,
            )

            input_ids = full_enc["input_ids"]
            attention_mask = full_enc["attention_mask"]
            offset_mapping = full_enc.get("offset_mapping")
            prompt_len = len(prompt_enc["input_ids"])
            seq_len = len(input_ids)

            # labels_mask: 0 for prompt, 1 for response
            labels_mask = [0] * min(prompt_len, seq_len) + [1] * max(0, seq_len - prompt_len)

            # Map answer span character offsets to token positions
            # answer_start_char is relative to context; find context start in prompt_str
            answer_token_start = -1
            answer_token_end = -1

            if offset_mapping is not None:
                context_marker = "### Context:\n"
                context_start_in_prompt = prompt_str.find(context_marker)
                if context_start_in_prompt >= 0:
                    context_start_in_prompt += len(context_marker)
                    abs_answer_start = context_start_in_prompt + answer_start_char
                    abs_answer_end = abs_answer_start + len(answer_text)

                    # Find token indices that overlap with [abs_answer_start, abs_answer_end)
                    for tok_idx, (cs, ce) in enumerate(offset_mapping):
                        if cs == 0 and ce == 0:
                            continue  # special token
                        if ce > abs_answer_start and cs < abs_answer_end:
                            if answer_token_start == -1:
                                answer_token_start = tok_idx
                            answer_token_end = tok_idx

            if answer_token_start >= 0:
                n_span_mapped += 1

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_mask": labels_mask,
                "index": i,
                "context": context,
                "question": question,
                "answer_text": answer_text,
                "answer_token_start": answer_token_start,
                "answer_token_end": answer_token_end,
            })

        self._n_span_mapped = n_span_mapped

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "input_ids": torch.tensor(s["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(s["attention_mask"], dtype=torch.long),
            "labels_mask": torch.tensor(s["labels_mask"], dtype=torch.long),
            "index": torch.tensor(s["index"], dtype=torch.long),
            "answer_token_start": torch.tensor(s["answer_token_start"], dtype=torch.long),
            "answer_token_end": torch.tensor(s["answer_token_end"], dtype=torch.long),
        }

    def get_metadata(self, idx: int) -> dict[str, str]:
        """Get non-tensor metadata for a sample."""
        s = self.samples[idx]
        return {
            "instruction": s["question"],
            "response": s["answer_text"],
            "context": s["context"],
            "category": "extractive_qa",
        }

    @property
    def span_mapping_rate(self) -> float:
        """Fraction of samples with successfully mapped answer spans."""
        return self._n_span_mapped / max(len(self.samples), 1)


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad to longest in batch, stack index and optional answer span fields."""
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

    result = {
        "input_ids": torch.stack(input_ids),          # (B, L)
        "attention_mask": torch.stack(attention_mask),  # (B, L)
        "labels_mask": torch.stack(labels_mask),        # (B, L)
        "index": torch.stack(indices),                  # (B,)
    }

    # Optional answer span fields (present in SquadDataset, absent in InstructionDataset)
    if "answer_token_start" in batch[0]:
        result["answer_token_start"] = torch.stack(
            [b["answer_token_start"] for b in batch],
        )  # (B,)
        result["answer_token_end"] = torch.stack(
            [b["answer_token_end"] for b in batch],
        )  # (B,)

    return result
