#!/usr/bin/env python3
"""Evaluate a trained student on all DA-KD-style instruction-following benchmarks.

Evaluates on: DollyEval, SelfInst, Super-Natural, Unnatural, VicunaEval.
Reports ROUGE-L for each benchmark and the average.

Usage:
    python scripts/evaluate_benchmarks.py \
        --student_ckpt outputs/standard_kd/seed_42/student_final.pt \
        --output_path outputs/standard_kd/seed_42/benchmark_rouge.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sagd.data import EvalInstructionDataset
from sagd.evaluation import compute_rouge, generate_responses
from sagd.models import load_student

# VicunaEval removed: no reliable HF source; it requires GPT-4 judging, not ROUGE-L.
BENCHMARKS = ["dolly_eval", "self_inst", "super_natural", "unnatural"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-benchmark evaluation")
    p.add_argument("--student_model", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--student_ckpt", type=str, required=True)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=500,
                    help="Max samples per benchmark (default 500)")
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

    all_metrics = {}
    rouge_scores = []

    for bench in BENCHMARKS:
        print(f"\n=== Evaluating on {bench} ===")
        try:
            dataset = EvalInstructionDataset(
                tokenizer=tokenizer,
                eval_name=bench,
                max_seq_len=args.max_seq_len,
                max_samples=args.max_samples,
                seed=args.seed,
            )
            if len(dataset) == 0:
                print(f"  WARNING: {bench} has 0 samples, skipping.")
                continue

            print(f"  Samples: {len(dataset)}")
            responses = generate_responses(
                student, tokenizer, dataset,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                device=args.device,
            )
            metrics = compute_rouge(responses)
            all_metrics[bench] = metrics
            rouge_scores.append(metrics["rouge_l_f"])
            print(f"  ROUGE-L F1: {metrics['rouge_l_f']:.4f}")

        except Exception as e:
            print(f"  ERROR on {bench}: {e}")

    # Average
    if rouge_scores:
        avg = sum(rouge_scores) / len(rouge_scores)
        all_metrics["average_rouge_l"] = avg
        print(f"\n=== Average ROUGE-L F1: {avg:.4f} ===")

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
