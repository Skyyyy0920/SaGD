#!/usr/bin/env python3
"""Precompute teacher saliency and cache to disk.

Run once before SaGD training. Output format:
{
    "saliency": List[Tensor],  # each (L_i,) per sample
    "metadata": {"model": str, "data": str, "n_samples": int, "max_seq_len": int}
}
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sagd.data import InstructionDataset, collate_fn
from sagd.models import load_teacher
from sagd.saliency import SaliencyComputer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute teacher saliency")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--tokenizer_name", type=str, default=None,
                    help="Tokenizer to use. Defaults to --model_name. "
                         "For cross-arch experiments, set to the STUDENT model name.")
    p.add_argument("--data_source", type=str, default="databricks/databricks-dolly-15k")
    p.add_argument("--output_path", type=str, default="data/teacher_saliency.pt")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"Loading teacher: {args.model_name}")
    tokenizer_name = args.tokenizer_name or args.model_name
    teacher, tokenizer = load_teacher(args.model_name, args.device)

    # If a different tokenizer is specified (cross-arch), use it for data
    if args.tokenizer_name and args.tokenizer_name != args.model_name:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = InstructionDataset(
        tokenizer=tokenizer,
        dataset_name=args.data_source,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
        seed=args.seed,
        subset="train",
    )
    print(f"Dataset size: {len(dataset)}")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
    )

    computer = SaliencyComputer()
    all_saliency: list[torch.Tensor] = [None] * len(dataset)

    for batch in tqdm(dataloader, desc="Computing teacher saliency"):
        input_ids = batch["input_ids"].to(args.device)          # (B, L)
        attention_mask = batch["attention_mask"].to(args.device)  # (B, L)
        labels_mask = batch["labels_mask"].to(args.device)       # (B, L)
        indices = batch["index"]                                 # (B,)

        with torch.amp.autocast("cuda", enabled=False):
            sal = computer.compute(teacher, input_ids, attention_mask, labels_mask)  # (B, L)

        # Store per-sample saliency (variable length, trimmed of trailing padding)
        for i, idx in enumerate(indices.tolist()):
            sample_mask = attention_mask[i].cpu()  # (L,)
            real_len = sample_mask.sum().item()
            all_saliency[idx] = sal[i, :real_len].cpu()  # (L_real,)

    # Verify all samples computed
    assert all(s is not None for s in all_saliency), "Some samples missing saliency"

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save({
        "saliency": all_saliency,
        "metadata": {
            "model": args.model_name,
            "data": args.data_source,
            "n_samples": len(dataset),
            "max_seq_len": args.max_seq_len,
        },
    }, args.output_path)

    print(f"Saved {len(all_saliency)} saliency vectors to {args.output_path}")


if __name__ == "__main__":
    main()
