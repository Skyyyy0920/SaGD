#!/usr/bin/env python3
"""Summarize all Phase 1 evaluation results into a formatted table.

Reads benchmark_rouge.json from all method/seed directories and prints
a DA-KD Table 1-style summary with 2 decimal places.

Usage:
    python scripts/summarize_results.py
    python scripts/summarize_results.py --output_dir outputs_dolly
"""

from __future__ import annotations

import argparse
import json
import os
import sys

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
BENCHMARKS = ["dolly_eval", "super_natural", "unnatural"]
BENCH_DISPLAY = {
    "dolly_eval": "DollyEval",
    "super_natural": "S-NatInst",
    "unnatural": "Unnatural",
}
SEEDS = [42, 123, 456, 789, 2024]
STUDENTS = [
    ("qwen3_1.7B", "Qwen3-1.7B"),
    ("qwen3_0.6B", "Qwen3-0.6B"),
]


def load_results(output_dir: str) -> dict:
    """Load all benchmark_rouge.json files into a nested dict."""
    results = {}
    for tag, name in STUDENTS:
        results[tag] = {}
        for method in METHODS:
            results[tag][method] = {}
            for seed in SEEDS:
                path = os.path.join(
                    output_dir, tag, method, f"seed_{seed}", "benchmark_rouge.json"
                )
                if os.path.exists(path):
                    with open(path) as f:
                        results[tag][method][seed] = json.load(f)
    return results


def fmt(values: list[float], bold: bool = False) -> str:
    """Format mean±std with 2 decimal places."""
    if not values:
        return "   —   "
    m = np.mean(values)
    s = np.std(values)
    text = f"{m:5.2f}±{s:4.2f}"
    if bold:
        text = f"**{text}**"
    return text


def print_table(results: dict, output_dir: str) -> None:
    """Print a DA-KD Table 1-style summary."""
    bench_names = [BENCH_DISPLAY.get(b, b) for b in BENCHMARKS]

    for tag, student_name in STUDENTS:
        print(f"\n{'='*80}")
        print(f"  {student_name} (Teacher: Qwen3-8B)")
        print(f"{'='*80}")

        # Header
        header = f"{'Method':<15}"
        for bn in bench_names:
            header += f" | {bn:>12}"
        header += f" | {'Avg.':>10}"
        print(header)
        print("-" * len(header))

        # Find best avg for bolding
        method_avgs = {}
        for method in METHODS:
            all_avgs = []
            for bench in BENCHMARKS:
                scores = []
                for seed in SEEDS:
                    data = results[tag].get(method, {}).get(seed)
                    if data and bench in data:
                        scores.append(data[bench]["rouge_l_f"] * 100)
                if scores:
                    all_avgs.append(np.mean(scores))
            if all_avgs:
                method_avgs[method] = np.mean(all_avgs)

        best_method = max(method_avgs, key=method_avgs.get) if method_avgs else None

        # Rows
        for method in METHODS:
            display = METHOD_DISPLAY.get(method, method)
            is_best = (method == best_method)
            row = f"{display:<15}"

            bench_means = []
            for bench in BENCHMARKS:
                scores = []
                for seed in SEEDS:
                    data = results[tag].get(method, {}).get(seed)
                    if data and bench in data:
                        scores.append(data[bench]["rouge_l_f"] * 100)
                row += f" | {fmt(scores):>12}"
                if scores:
                    bench_means.append(np.mean(scores))

            # Avg column
            if bench_means:
                avg = np.mean(bench_means)
                avg_str = f"{avg:5.2f}"
                if is_best:
                    avg_str = f"*{avg_str}*"
                row += f" | {avg_str:>10}"
            else:
                row += f" | {'—':>10}"

            print(row)

        # Print count of completed seeds
        print()
        completed = {}
        for method in METHODS:
            n = len(results[tag].get(method, {}))
            completed[method] = n
        missing = [m for m, n in completed.items() if n < len(SEEDS)]
        if missing:
            print(f"  Incomplete: {', '.join(f'{m}({completed[m]}/{len(SEEDS)})' for m in missing)}")
        else:
            print(f"  All {len(METHODS)} methods × {len(SEEDS)} seeds complete.")

    # LaTeX table
    print(f"\n{'='*80}")
    print("  LaTeX (copy-paste ready)")
    print(f"{'='*80}")
    print(r"\begin{tabular}{ll" + "c" * len(BENCHMARKS) + "c}")
    print(r"\toprule")
    print(
        r"Model & Method & "
        + " & ".join(BENCH_DISPLAY.get(b, b) for b in BENCHMARKS)
        + r" & Avg. \\"
    )
    print(r"\midrule")

    for tag, student_name in STUDENTS:
        first = True
        for method in METHODS:
            display = METHOD_DISPLAY.get(method, method)
            if method == "sagd":
                display = r"\textbf{" + display + "}"

            row_parts = []
            bench_means = []
            for bench in BENCHMARKS:
                scores = []
                for seed in SEEDS:
                    data = results[tag].get(method, {}).get(seed)
                    if data and bench in data:
                        scores.append(data[bench]["rouge_l_f"] * 100)
                if scores:
                    m, s = np.mean(scores), np.std(scores)
                    row_parts.append(f"{m:.2f}")
                    bench_means.append(m)
                else:
                    row_parts.append("—")

            if bench_means:
                avg = np.mean(bench_means)
                row_parts.append(f"{avg:.2f}")
            else:
                row_parts.append("—")

            model_col = student_name if first else ""
            first = False
            print(f"{model_col} & {display} & " + " & ".join(row_parts) + r" \\")

        print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")


def main():
    parser = argparse.ArgumentParser(description="Summarize Phase 1 results")
    parser.add_argument("--output_dir", default="outputs_dolly",
                        help="Base output directory (default: outputs_dolly)")
    args = parser.parse_args()

    results = load_results(args.output_dir)
    print_table(results, args.output_dir)


if __name__ == "__main__":
    main()
