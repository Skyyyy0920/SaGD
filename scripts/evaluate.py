#!/usr/bin/env python3
"""Standalone evaluation script for trained student models."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sagd.data import InstructionDataset
from sagd.evaluation import evaluate_rouge
from sagd.models import load_student


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate student model")
    p.add_argument("--student_model", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--student_ckpt", type=str, required=True)
    p.add_argument("--data_source", type=str, default="databricks/databricks-dolly-15k")
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=500)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_path", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    student, tokenizer = load_student(args.student_model, args.device)
    state_dict = torch.load(args.student_ckpt, map_location=args.device, weights_only=True)
    student.load_state_dict(state_dict)
    student.eval()

    dataset = InstructionDataset(
        tokenizer=tokenizer,
        dataset_name=args.data_source,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    metrics = evaluate_rouge(
        student, tokenizer, dataset,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=args.device,
    )

    print(f"ROUGE-L F1:  {metrics['rouge_l_f']:.4f}")
    print(f"ROUGE-L P:   {metrics['rouge_l_p']:.4f}")
    print(f"ROUGE-L R:   {metrics['rouge_l_r']:.4f}")

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
