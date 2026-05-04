#!/usr/bin/env python3
"""Summarize Phase 2 (task-specific) evaluation results.

Reads eval_metrics.json from outputs_task/ and prints DA-KD Table 2-style
summary with 2 decimal places.

Usage:
    python scripts/summarize_phase2.py
"""

from __future__ import annotations

import json
import os

import numpy as np

METHODS = ["sft", "standard_kd", "reverse_kl", "seqkd", "gkd", "distillm", "dakd", "sagd"]
METHOD_DISPLAY = {
    "sft": "SFT",
    "standard_kd": "KD-KL",
    "reverse_kl": "KD-RKL",
    "seqkd": "SeqKD",
    "gkd": "GKD",
    "distillm": "DistiLLM",
    "dakd": "DA-KD",
    "sagd": "SaGD (ours)",
}
SEEDS = [42, 123, 456]
STUDENTS = [
    ("qwen3_1.7B", "Qwen3-1.7B"),
    ("qwen3_0.6B", "Qwen3-0.6B"),
]
# (dataset, metric_key, display_name, higher_is_better)
DATASET_METRICS = [
    ("samsum", "rouge_l_f", "SAMSum RL", True),
    ("gsm8k", "gsm8k_accuracy", "GSM8K Acc", True),
    ("squad", "exact_match", "SQuAD EM", True),
    ("squad", "token_f1", "SQuAD F1", True),
    ("squad", "perplexity", "SQuAD PPL", False),
]


def fmt(values: list[float], scale: float = 100.0) -> str:
    if not values:
        return "     —     "
    m = np.mean(values) * scale
    s = np.std(values) * scale
    return f"{m:5.2f}±{s:4.2f}"


def fmt_ppl(values: list[float]) -> str:
    if not values:
        return "     —     "
    m = np.mean(values)
    s = np.std(values)
    return f"{m:5.2f}±{s:4.2f}"


def main():
    base = "outputs_task"

    for tag, student_name in STUDENTS:
        print(f"\n{'='*90}")
        print(f"  {student_name} (Teacher: Qwen3-8B)")
        print(f"{'='*90}")

        # Header
        metric_names = [dm[2] for dm in DATASET_METRICS]
        header = f"{'Method':<15}" + "".join(f" | {mn:>11}" for mn in metric_names)
        print(header)
        print("-" * len(header))

        for method in METHODS:
            display = METHOD_DISPLAY.get(method, method)
            row = f"{display:<15}"

            for ds, key, _, higher in DATASET_METRICS:
                values = []
                for seed in SEEDS:
                    path = os.path.join(base, tag, ds, method, f"seed_{seed}", "eval_metrics.json")
                    if os.path.exists(path):
                        with open(path) as f:
                            data = json.load(f)
                        if key in data:
                            values.append(data[key])

                if key == "perplexity":
                    row += f" | {fmt_ppl(values):>11}"
                else:
                    row += f" | {fmt(values):>11}"

            print(row)

        # Completeness
        print()
        for ds in ["samsum", "gsm8k", "squad"]:
            missing = []
            for method in METHODS:
                n = sum(
                    1 for seed in SEEDS
                    if os.path.exists(os.path.join(base, tag, ds, method, f"seed_{seed}", "eval_metrics.json"))
                )
                if n < len(SEEDS):
                    missing.append(f"{method}({n}/{len(SEEDS)})")
            if missing:
                print(f"  {ds} incomplete: {', '.join(missing)}")
            else:
                print(f"  {ds}: all {len(METHODS)}×{len(SEEDS)} complete.")

    # LaTeX
    print(f"\n{'='*90}")
    print("  LaTeX (copy-paste ready)")
    print(f"{'='*90}")
    metric_names = [dm[2] for dm in DATASET_METRICS]
    print(r"\begin{tabular}{l l " + " ".join(["c"] * len(DATASET_METRICS)) + "}")
    print(r"\toprule")
    print(r"Model & Method & " + " & ".join(metric_names) + r" \\")
    print(r"\midrule")

    for tag, student_name in STUDENTS:
        first = True
        for method in METHODS:
            display = METHOD_DISPLAY.get(method, method)
            if method == "sagd":
                display = r"\textbf{" + display + "}"
            parts = []
            for ds, key, _, _ in DATASET_METRICS:
                values = []
                for seed in SEEDS:
                    path = os.path.join(base, tag, ds, method, f"seed_{seed}", "eval_metrics.json")
                    if os.path.exists(path):
                        with open(path) as f:
                            data = json.load(f)
                        if key in data:
                            values.append(data[key])
                if values:
                    if key == "perplexity":
                        parts.append(f"{np.mean(values):.2f}")
                    else:
                        parts.append(f"{np.mean(values)*100:.2f}")
                else:
                    parts.append("—")

            model_col = student_name if first else ""
            first = False
            print(f"{model_col} & {display} & " + " & ".join(parts) + r" \\")
        print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
