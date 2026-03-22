#!/usr/bin/env python3
"""Pre-generate student responses and save to JSONL.

This decouples generation from metric computation, allowing:
  1. Run generation once (expensive), then compute multiple metrics cheaply.
  2. Run GPT-as-Judge comparisons without needing GPU access.
  3. Inspect generated responses manually.

Output format (one JSON object per line):
    {"index": 0, "instruction": "...", "reference": "...", "generated": "...", "category": "..."}
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sagd.data import InstructionDataset, SquadDataset
from sagd.evaluation import generate_responses, save_responses
from sagd.models import load_student


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate student responses to JSONL")
    p.add_argument("--student_model", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--student_ckpt", type=str, required=True,
                    help="Path to student checkpoint (.pt)")
    p.add_argument("--dataset", type=str, default="dolly", choices=["dolly", "squad"],
                    help="Dataset: 'dolly' (Dolly-15K) or 'squad' (SQuAD 2.0)")
    p.add_argument("--data_source", type=str, default=None,
                    help="HF dataset name. Auto-set from --dataset if not provided.")
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=500)
    p.add_argument("--max_new_tokens", type=int, default=None,
                    help="Max tokens to generate. Default: 32 for squad, 256 for dolly.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--subset", type=str, default="test",
                    choices=["train", "val", "test"])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_path", type=str, required=True,
                    help="Output JSONL file path")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.data_source is None:
        args.data_source = {
            "dolly": "databricks/databricks-dolly-15k",
            "squad": "rajpurkar/squad_v2",
        }[args.dataset]

    if args.max_new_tokens is None:
        args.max_new_tokens = 32 if args.dataset == "squad" else 256

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load model
    student, tokenizer = load_student(args.student_model, args.device)
    state_dict = torch.load(args.student_ckpt, map_location=args.device, weights_only=True)
    student.load_state_dict(state_dict)
    student.eval()

    # Dataset
    if args.dataset == "squad":
        dataset = SquadDataset(
            tokenizer=tokenizer,
            dataset_name=args.data_source,
            max_seq_len=args.max_seq_len,
            max_samples=args.max_samples,
            seed=args.seed,
            subset=args.subset,
        )
    else:
        dataset = InstructionDataset(
            tokenizer=tokenizer,
            dataset_name=args.data_source,
            max_seq_len=args.max_seq_len,
            max_samples=args.max_samples,
            seed=args.seed,
            subset=args.subset,
        )
    print(f"Dataset: {len(dataset)} samples (subset={args.subset})")

    # Generate
    responses = generate_responses(
        student, tokenizer, dataset,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Save
    save_responses(responses, args.output_path)
    print(f"Saved {len(responses)} responses to {args.output_path}")


if __name__ == "__main__":
    main()
