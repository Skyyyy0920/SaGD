#!/usr/bin/env python3
"""Standalone evaluation script for trained student models.

Computes ROUGE-L, BERTScore (optional), Perplexity, and for SQuAD: EM/F1.
Optionally saves pre-generated responses to JSONL for later GPT-as-Judge use.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sagd.data import InstructionDataset, SquadDataset
from sagd.evaluation import evaluate_all, generate_responses, save_responses
from sagd.models import load_student


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate student model")
    p.add_argument("--student_model", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--student_ckpt", type=str, required=True)
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
    p.add_argument("--output_path", type=str, default=None)
    p.add_argument("--save_responses", type=str, default=None,
                    help="Path to save generated responses as JSONL (for GPT-as-Judge)")
    p.add_argument("--skip_bertscore", action="store_true",
                    help="Skip BERTScore computation (faster)")
    p.add_argument("--bertscore_model", type=str, default="microsoft/deberta-xlarge-mnli",
                    help="BERTScore encoder model")
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

    student, tokenizer = load_student(args.student_model, args.device)
    state_dict = torch.load(args.student_ckpt, map_location=args.device, weights_only=True)
    student.load_state_dict(state_dict)
    student.eval()

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

    # Run all metrics
    metrics = evaluate_all(
        student, tokenizer, dataset,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=args.device,
        skip_bertscore=args.skip_bertscore,
        bertscore_model=args.bertscore_model,
        dataset_type=args.dataset,
    )

    # Print results
    if "exact_match" in metrics:
        print(f"Exact Match:  {metrics['exact_match']:.4f}")
        print(f"Token F1:     {metrics['token_f1']:.4f}")
    print(f"ROUGE-L F1:   {metrics['rouge_l_f']:.4f}")
    print(f"ROUGE-L P:    {metrics['rouge_l_p']:.4f}")
    print(f"ROUGE-L R:    {metrics['rouge_l_r']:.4f}")
    if "bertscore_f" in metrics:
        print(f"BERTScore F1: {metrics['bertscore_f']:.4f}")
        print(f"BERTScore P:  {metrics['bertscore_p']:.4f}")
        print(f"BERTScore R:  {metrics['bertscore_r']:.4f}")
    print(f"Perplexity:   {metrics['perplexity']:.2f}")
    print(f"Avg NLL:      {metrics['avg_loss']:.4f}")

    # Save metrics
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {args.output_path}")

    # Optionally save responses for GPT-as-Judge
    if args.save_responses:
        responses = generate_responses(
            student, tokenizer, dataset,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            device=args.device,
        )
        save_responses(responses, args.save_responses)
        print(f"Saved responses to {args.save_responses}")


if __name__ == "__main__":
    main()
