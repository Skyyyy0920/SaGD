#!/usr/bin/env python3
"""Saliency divergence diagnosis for a trained student checkpoint.

For SQuAD datasets, also computes evidence concentration (fraction of
saliency mass on the answer span) for both teacher and student.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sagd.data import InstructionDataset, SquadDataset, collate_fn
from sagd.evaluation import compute_evidence_concentration
from sagd.models import load_student
from sagd.saliency import SaliencyComputer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Saliency divergence diagnosis")
    p.add_argument("--student_model", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--student_ckpt", type=str, required=True)
    p.add_argument("--teacher_saliency_path", type=str, required=True)
    p.add_argument("--dataset", type=str, default="dolly", choices=["dolly", "squad"],
                    help="Dataset: 'dolly' (Dolly-15K) or 'squad' (SQuAD 2.0)")
    p.add_argument("--data_source", type=str, default=None,
                    help="HF dataset name. Auto-set from --dataset if not provided.")
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--max_samples", type=int, default=500)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--subset", type=str, default="val",
                    choices=["train", "val", "test"])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.data_source is None:
        args.data_source = {
            "dolly": "databricks/databricks-dolly-15k",
            "squad": "rajpurkar/squad_v2",
        }[args.dataset]

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load student
    student, tokenizer = load_student(args.student_model, args.device)
    state_dict = torch.load(args.student_ckpt, map_location=args.device, weights_only=True)
    student.load_state_dict(state_dict)
    student.eval()

    # Load teacher saliency cache
    cache = torch.load(args.teacher_saliency_path, map_location="cpu", weights_only=False)
    teacher_saliency_cache = cache["saliency"]

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

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
    )

    computer = SaliencyComputer()
    all_jsd: list[tuple[int, float, str, str]] = []  # (index, jsd, category, instruction)

    # Accumulators for evidence concentration (SQuAD only)
    all_teacher_ec: list[float] = []
    all_student_ec: list[float] = []

    for batch in tqdm(dataloader, desc="Computing student saliency"):
        input_ids = batch["input_ids"].to(args.device)          # (B, L)
        attention_mask = batch["attention_mask"].to(args.device)  # (B, L)
        labels_mask = batch["labels_mask"].to(args.device)       # (B, L)
        indices = batch["index"]                                 # (B,)

        seq_len = input_ids.size(1)

        # Student saliency
        with torch.amp.autocast("cuda", enabled=False):
            student_sal = computer.compute(
                student, input_ids, attention_mask, labels_mask,
            )  # (B, L)

        # Teacher saliency from cache
        teacher_sal_list = []
        for idx in indices.tolist():
            sal = teacher_saliency_cache[idx]  # (L_i,)
            if sal.size(0) >= seq_len:
                sal = sal[:seq_len]
            else:
                sal = torch.cat([sal, torch.zeros(seq_len - sal.size(0))])
            teacher_sal_list.append(sal)
        teacher_sal = torch.stack(teacher_sal_list).to(args.device)  # (B, L)

        # JSD
        jsd = computer.divergence(
            teacher_sal, student_sal, labels_mask, attention_mask,
        )  # (B,)

        for i, idx in enumerate(indices.tolist()):
            meta = dataset.get_metadata(idx)
            all_jsd.append((idx, jsd[i].item(), meta["category"], meta["instruction"]))

        # Evidence concentration (SQuAD only)
        if args.dataset == "squad" and "answer_token_start" in batch:
            ans_start = batch["answer_token_start"].to(args.device)
            ans_end = batch["answer_token_end"].to(args.device)

            t_ec = compute_evidence_concentration(
                teacher_sal, ans_start, ans_end, attention_mask,
            )
            s_ec = compute_evidence_concentration(
                student_sal, ans_start, ans_end, attention_mask,
            )

            # Collect per-sample evidence concentrations for valid samples
            B = input_ids.size(0)
            for i in range(B):
                if ans_start[i].item() >= 0 and ans_end[i].item() >= 0:
                    t_total = teacher_sal[i].sum().item()
                    s_total = student_sal[i].sum().item()
                    if t_total > 1e-10:
                        start_i = max(0, min(ans_start[i].item(), seq_len - 1))
                        end_i = max(0, min(ans_end[i].item(), seq_len - 1))
                        all_teacher_ec.append(
                            teacher_sal[i, start_i:end_i + 1].sum().item() / t_total,
                        )
                    if s_total > 1e-10:
                        start_i = max(0, min(ans_start[i].item(), seq_len - 1))
                        end_i = max(0, min(ans_end[i].item(), seq_len - 1))
                        all_student_ec.append(
                            student_sal[i, start_i:end_i + 1].sum().item() / s_total,
                        )

    # Aggregate JSD
    jsd_values = [x[1] for x in all_jsd]
    jsd_tensor = torch.tensor(jsd_values)

    # Per-category
    cat_jsds = defaultdict(list)
    for _, jsd_val, cat, _ in all_jsd:
        cat_jsds[cat].append(jsd_val)
    per_category = {cat: sum(v) / len(v) for cat, v in cat_jsds.items()}

    # Top-20
    sorted_jsd = sorted(all_jsd, key=lambda x: x[1], reverse=True)
    top20 = [
        {"index": x[0], "jsd": x[1], "instruction_preview": x[3][:100]}
        for x in sorted_jsd[:20]
    ]

    result = {
        "mean_jsd": jsd_tensor.mean().item(),
        "std_jsd": jsd_tensor.std().item(),
        "median_jsd": jsd_tensor.median().item(),
        "top20_samples": top20,
        "per_category_jsd": per_category,
    }

    # Add evidence concentration for SQuAD
    if all_teacher_ec:
        result["teacher_evidence_concentration"] = sum(all_teacher_ec) / len(all_teacher_ec)
        result["student_evidence_concentration"] = (
            sum(all_student_ec) / len(all_student_ec) if all_student_ec else 0.0
        )
        result["n_ec_samples"] = len(all_teacher_ec)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Mean JSD: {result['mean_jsd']:.4f} ± {result['std_jsd']:.4f}")
    if "teacher_evidence_concentration" in result:
        print(f"Teacher Evidence Concentration: {result['teacher_evidence_concentration']:.4f}")
        print(f"Student Evidence Concentration: {result['student_evidence_concentration']:.4f}")
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
