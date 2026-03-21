#!/usr/bin/env python3
"""Main training entry point for SaGD knowledge distillation."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sagd.data import InstructionDataset, SquadDataset
from sagd.evaluation import evaluate_all
from sagd.models import load_student, load_teacher
from sagd.trainer import METHODS, Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SaGD Knowledge Distillation Training")

    # Method
    p.add_argument("--method", type=str, default="standard_kd", choices=sorted(METHODS))

    # Models
    p.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--student_model", type=str, default="Qwen/Qwen3-0.6B")

    # Data
    p.add_argument("--dataset", type=str, default="dolly", choices=["dolly", "squad"],
                    help="Dataset: 'dolly' (Dolly-15K) or 'squad' (SQuAD 2.0)")
    p.add_argument("--data_source", type=str, default=None,
                    help="HF dataset name. Auto-set from --dataset if not provided.")
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)

    # Training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no_fp16", action="store_true")

    # SaGD-specific
    p.add_argument("--teacher_saliency_path", type=str, default=None)
    p.add_argument("--lambda_sal", type=float, default=0.5)
    p.add_argument("--sagd_every_n_steps", type=int, default=5)
    p.add_argument("--sagd_tau_w", type=float, default=1.0)
    p.add_argument("--saliency_temperature", type=float, default=2.0)

    # Output
    p.add_argument("--output_dir", type=str, default="outputs/")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--skip_bertscore", action="store_true",
                    help="Skip BERTScore in post-training eval")
    p.add_argument("--log_every", type=int, default=50)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_fp16:
        args.fp16 = False

    # Auto-set data_source from --dataset if not explicitly provided
    if args.data_source is None:
        args.data_source = {
            "dolly": "databricks/databricks-dolly-15k",
            "squad": "rajpurkar/squad_v2",
        }[args.dataset]

    # Reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Save dir
    save_dir = os.path.join(args.output_dir, args.method, f"seed_{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Method: {args.method}")
    print(f"Dataset: {args.dataset} ({args.data_source})")
    print(f"Teacher: {args.teacher_model}")
    print(f"Student: {args.student_model}")
    print(f"Save dir: {save_dir}")

    # Load models
    teacher, t_tokenizer = load_teacher(args.teacher_model, args.device)
    student, s_tokenizer = load_student(args.student_model, args.device)

    # Use student tokenizer for data (student is the one being trained)
    if args.dataset == "squad":
        dataset = SquadDataset(
            tokenizer=s_tokenizer,
            dataset_name=args.data_source,
            max_seq_len=args.max_seq_len,
            max_samples=args.max_train_samples,
            seed=args.seed,
            subset="train",
        )
        print(f"Answer span mapping rate: {dataset.span_mapping_rate:.1%}")
    else:
        dataset = InstructionDataset(
            tokenizer=s_tokenizer,
            dataset_name=args.data_source,
            max_seq_len=args.max_seq_len,
            max_samples=args.max_train_samples,
            seed=args.seed,
            subset="train",
        )
    print(f"Dataset size: {len(dataset)}")

    # Config dict
    config = {
        "method": args.method,
        "dataset": args.dataset,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "temperature": args.temperature,
        "fp16": args.fp16,
        "log_every": args.log_every,
        "teacher_saliency_path": args.teacher_saliency_path,
        "lambda_sal": args.lambda_sal,
        "sagd_every_n_steps": args.sagd_every_n_steps,
        "sagd_tau_w": args.sagd_tau_w,
        "saliency_temperature": args.saliency_temperature,
    }

    # Save config
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Train
    trainer = Trainer(teacher, student, s_tokenizer, dataset, config)
    history = trainer.train(save_dir)

    # Evaluate
    if not args.skip_eval:
        print("Evaluating...")
        if args.dataset == "squad":
            eval_dataset = SquadDataset(
                tokenizer=s_tokenizer,
                dataset_name=args.data_source,
                max_seq_len=args.max_seq_len,
                max_samples=500,
                seed=args.seed,
                subset="test",
            )
        else:
            eval_dataset = InstructionDataset(
                tokenizer=s_tokenizer,
                dataset_name=args.data_source,
                max_seq_len=args.max_seq_len,
                max_samples=500,
                seed=args.seed,
                subset="test",
            )
        metrics = evaluate_all(
            student, s_tokenizer, eval_dataset,
            device=args.device,
            skip_bertscore=args.skip_bertscore,
            dataset_type=args.dataset,
        )
        print(f"ROUGE-L F1:  {metrics['rouge_l_f']:.4f}")
        if "exact_match" in metrics:
            print(f"Exact Match: {metrics['exact_match']:.4f}")
            print(f"Token F1:    {metrics['token_f1']:.4f}")
        if "bertscore_f" in metrics:
            print(f"BERTScore F1: {metrics['bertscore_f']:.4f}")
        print(f"Perplexity:  {metrics['perplexity']:.2f}")
        with open(os.path.join(save_dir, "eval_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
