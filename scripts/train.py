#!/usr/bin/env python3
"""Main training entry point for SaGD knowledge distillation.

Supports all baseline methods from DA-KD (ICML 2025) comparison:
  SFT, KD-KL (standard_kd), KD-RKL (reverse_kl), SeqKD, GKD, DistiLLM, DA-KD, SaGD.

Supports datasets: dolly, squad, samsum, gsm8k.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sagd.data import GSM8KDataset, InstructionDataset, SAMSumDataset, SquadDataset
from sagd.evaluation import evaluate_all
from sagd.models import load_student, load_teacher
from sagd.trainer import METHODS, Trainer


DATASET_CHOICES = ["dolly", "squad", "samsum", "gsm8k"]

DATASET_HF_NAMES = {
    "dolly": "databricks/databricks-dolly-15k",
    "squad": "rajpurkar/squad_v2",
    "samsum": "knkarthick/samsum",
    "gsm8k": "openai/gsm8k",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SaGD Knowledge Distillation Training")

    # Method
    p.add_argument("--method", type=str, default="standard_kd", choices=sorted(METHODS))

    # Models
    p.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--student_model", type=str, default="Qwen/Qwen3-0.6B")

    # Data
    p.add_argument("--dataset", type=str, default="dolly", choices=DATASET_CHOICES,
                    help="Dataset for training")
    p.add_argument("--data_source", type=str, default=None,
                    help="HF dataset name. Auto-set from --dataset if not provided.")
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)

    # Training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
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
    p.add_argument("--lambda_noise", type=float, default=0.5,
                    help="Weight for noise KL loss (implicit Jacobian matching)")
    p.add_argument("--noise_sigma", type=float, default=0.01,
                    help="Std of Gaussian noise added to embeddings")
    p.add_argument("--sagd_every_n_steps", type=int, default=5)
    p.add_argument("--sagd_tau_w", type=float, default=1.0)
    p.add_argument("--saliency_temperature", type=float, default=2.0)

    # GKD-specific
    p.add_argument("--gkd_beta", type=float, default=0.5,
                    help="JSD mixing coefficient for GKD")
    p.add_argument("--gkd_on_policy_prob", type=float, default=0.0,
                    help="Probability of using on-policy (student-generated) sequences for GKD")

    # DistiLLM-specific
    p.add_argument("--distillm_alpha", type=float, default=0.5,
                    help="Skew coefficient for DistiLLM")

    # DA-KD-specific
    p.add_argument("--bdl_lambda", type=float, default=0.9,
                    help="BDL mixing coefficient for DA-KD")

    # Curriculum
    p.add_argument("--curriculum_path", type=str, default=None,
                    help="Path to curriculum order file (.pt) from compute_curriculum.py. "
                         "Trains all samples every epoch, ordered by structural score.")

    # Output
    p.add_argument("--output_dir", type=str, default="outputs/")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--load_8bit_teacher", action="store_true",
                    help="Load teacher with bitsandbytes int8 quantization (fits 24GB GPUs).")
    p.add_argument("--gradient_checkpointing", action="store_true",
                    help="Enable gradient checkpointing on student to reduce activation memory.")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--skip_bertscore", action="store_true",
                    help="Skip BERTScore in post-training eval")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every_n_epochs", type=int, default=0,
                    help="Save intermediate checkpoint every N epochs. "
                         "0 (default) = only save student_final.pt.")

    return p.parse_args()


def create_dataset(args, tokenizer, subset="train"):
    """Create the appropriate dataset based on --dataset flag."""
    if args.dataset == "squad":
        return SquadDataset(
            tokenizer=tokenizer,
            dataset_name=args.data_source,
            max_seq_len=args.max_seq_len,
            max_samples=args.max_train_samples if subset == "train" else 500,
            seed=args.seed,
            subset=subset,
        )
    elif args.dataset == "samsum":
        return SAMSumDataset(
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            max_samples=args.max_train_samples if subset == "train" else None,
            seed=args.seed,
            subset=subset,
        )
    elif args.dataset == "gsm8k":
        return GSM8KDataset(
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            max_samples=args.max_train_samples if subset == "train" else None,
            seed=args.seed,
            subset=subset,
        )
    else:  # dolly
        return InstructionDataset(
            tokenizer=tokenizer,
            dataset_name=args.data_source,
            max_seq_len=args.max_seq_len,
            max_samples=args.max_train_samples if subset == "train" else 500,
            seed=args.seed,
            subset=subset,
        )


def main() -> None:
    args = parse_args()

    if args.no_fp16:
        args.fp16 = False

    # Auto-set data_source from --dataset if not explicitly provided
    if args.data_source is None:
        args.data_source = DATASET_HF_NAMES[args.dataset]

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
    # SFT is the only method that doesn't need a teacher
    if args.method == "sft":
        teacher, t_tokenizer = None, None
    else:
        teacher, t_tokenizer = load_teacher(
            args.teacher_model, args.device,
            load_in_8bit=args.load_8bit_teacher,
        )

    student, s_tokenizer = load_student(
        args.student_model, args.device,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Use student tokenizer for data
    dataset = create_dataset(args, s_tokenizer, subset="train")
    if hasattr(dataset, 'span_mapping_rate'):
        print(f"Answer span mapping rate: {dataset.span_mapping_rate:.1%}")
    print(f"Dataset size: {len(dataset)}")

    # Load curriculum if provided
    curriculum_stages = None
    if args.curriculum_path:
        print(f"Loading curriculum: {args.curriculum_path}")
        curriculum_data = torch.load(args.curriculum_path, map_location="cpu",
                                     weights_only=False)
        sorted_indices = curriculum_data["sorted_indices"].tolist()
        # Pass the full sorted index list — every epoch trains ALL samples,
        # but in this fixed order (high structural score first).
        curriculum_stages = sorted_indices
        print(f"  Curriculum: {len(sorted_indices)} samples, ordered by PC score")

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
        "save_every_n_epochs": args.save_every_n_epochs,
        "curriculum_path": args.curriculum_path,
        "teacher_saliency_path": args.teacher_saliency_path,
        "lambda_noise": args.lambda_noise,
        "noise_sigma": args.noise_sigma,
        "sagd_every_n_steps": args.sagd_every_n_steps,
        "sagd_tau_w": args.sagd_tau_w,
        "saliency_temperature": args.saliency_temperature,
        "gkd_beta": args.gkd_beta,
        "gkd_on_policy_prob": args.gkd_on_policy_prob,
        "distillm_alpha": args.distillm_alpha,
        "bdl_lambda": args.bdl_lambda,
    }

    # Save config
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Train
    trainer = Trainer(teacher, student, s_tokenizer, dataset, config)
    history = trainer.train(save_dir, curriculum_stages=curriculum_stages)

    # Evaluate
    if not args.skip_eval:
        print("Evaluating...")
        eval_dataset = create_dataset(args, s_tokenizer, subset="test")
        max_new = 32 if args.dataset == "squad" else 256
        metrics = evaluate_all(
            student, s_tokenizer, eval_dataset,
            max_new_tokens=max_new,
            device=args.device,
            skip_bertscore=args.skip_bertscore,
            dataset_type=args.dataset,
        )
        print(f"ROUGE-L F1:  {metrics['rouge_l_f']:.4f}")
        if "exact_match" in metrics:
            print(f"Exact Match: {metrics['exact_match']:.4f}")
            print(f"Token F1:    {metrics['token_f1']:.4f}")
        if "gsm8k_accuracy" in metrics:
            print(f"GSM8K Acc:   {metrics['gsm8k_accuracy']:.4f}")
        if "bertscore_f" in metrics:
            print(f"BERTScore F1: {metrics['bertscore_f']:.4f}")
        print(f"Perplexity:  {metrics['perplexity']:.2f}")
        with open(os.path.join(save_dir, "eval_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
