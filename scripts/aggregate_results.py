#!/usr/bin/env python3
"""Aggregate all benchmark_rouge.json under one or more root directories.

Auto-discovers any path matching `**/seed_<int>/benchmark_rouge.json`. The
"label" used in the output table is the relative path from --root to the
parent of `seed_*`. This handles both flat layouts:

    outputs_dolly/llama_1B/standard_kd/seed_42/benchmark_rouge.json
    label = "standard_kd"

and nested curriculum layouts:

    outputs_ours/llama_1B/dolly/sagd_grad_curriculum/sagd/seed_42/...
    label = "sagd_grad_curriculum/sagd"

Usage:
    # Single root
    python scripts/aggregate_results.py --root outputs_dolly/llama_1B

    # Multiple roots (e.g. baselines + curriculum)
    python scripts/aggregate_results.py \\
        --root outputs_dolly/llama_1B \\
        --root outputs_ours/llama_1B/dolly \\
        --tag llama

    # CSV / Markdown / LaTeX outputs
    python scripts/aggregate_results.py --root outputs_dolly/qwen3_0.6B \\
        --csv qwen.csv --markdown qwen.md --latex qwen.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


SEED_RE = re.compile(r"^seed_(\d+)$")


def discover(root: Path) -> dict[str, dict[int, dict]]:
    """Walk root, group by label → seed → metrics_dict.

    label = relpath from root to parent of seed_* directory.
    """
    grouped: dict[str, dict[int, dict]] = defaultdict(dict)
    for json_path in root.rglob("benchmark_rouge.json"):
        seed_dir = json_path.parent
        m = SEED_RE.match(seed_dir.name)
        if not m:
            continue
        seed = int(m.group(1))
        label_path = seed_dir.parent.relative_to(root)
        label = str(label_path).replace("\\", "/")
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [WARN] failed to read {json_path}: {e}")
            continue
        grouped[label][seed] = data
    return grouped


def collect_benchmarks(grouped: dict[str, dict[int, dict]]) -> list[str]:
    """Find benchmark keys present in any json (excluding 'average_rouge_l')."""
    keys: set[str] = set()
    for seeds in grouped.values():
        for data in seeds.values():
            for k, v in data.items():
                if isinstance(v, dict) and "rouge_l_f" in v:
                    keys.add(k)
    preferred = ["dolly_eval", "self_inst", "super_natural", "unnatural", "vicuna_eval"]
    ordered = [k for k in preferred if k in keys]
    ordered += sorted(k for k in keys if k not in preferred)
    return ordered


def extract(data: dict, bench: str) -> float | None:
    if bench not in data:
        return None
    v = data[bench]
    if isinstance(v, dict) and "rouge_l_f" in v:
        return float(v["rouge_l_f"]) * 100.0
    return None


def compute_rows(
    grouped: dict[str, dict[int, dict]], benchmarks: list[str]
) -> list[dict]:
    rows = []
    for label, seeds in sorted(grouped.items()):
        seed_ids = sorted(seeds.keys())
        per_bench: dict[str, list[float]] = {b: [] for b in benchmarks}
        for s in seed_ids:
            for b in benchmarks:
                v = extract(seeds[s], b)
                if v is not None:
                    per_bench[b].append(v)
        bench_means = []
        bench_stats = {}
        for b in benchmarks:
            xs = per_bench[b]
            if xs:
                m, sd = float(np.mean(xs)), float(np.std(xs))
                bench_stats[b] = (m, sd, len(xs))
                bench_means.append(m)
            else:
                bench_stats[b] = None
        avg = float(np.mean(bench_means)) if bench_means else None
        rows.append(
            {
                "label": label,
                "seeds": seed_ids,
                "stats": bench_stats,
                "avg": avg,
            }
        )
    return rows


def print_text_table(rows: list[dict], benchmarks: list[str], title: str) -> None:
    if not rows:
        print(f"\n[{title}] no results found.")
        return

    label_w = max(len(r["label"]) for r in rows)
    label_w = max(label_w, len("Method"))
    col_w = 13

    header = f"{'Method':<{label_w}}"
    for b in benchmarks:
        header += f" | {b:>{col_w}}"
    header += f" | {'Avg':>{col_w}} | seeds"
    print(f"\n{'=' * len(header)}")
    print(f"  {title}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    best_avg = max((r["avg"] for r in rows if r["avg"] is not None), default=None)

    for r in rows:
        row = f"{r['label']:<{label_w}}"
        for b in benchmarks:
            stat = r["stats"].get(b)
            cell = "—" if stat is None else f"{stat[0]:5.2f}±{stat[1]:4.2f}"
            row += f" | {cell:>{col_w}}"
        if r["avg"] is None:
            avg_cell = "—"
        else:
            mark = "*" if best_avg is not None and abs(r["avg"] - best_avg) < 1e-9 else " "
            avg_cell = f"{mark}{r['avg']:5.2f}{mark}"
        row += f" | {avg_cell:>{col_w}} | {len(r['seeds'])} ({','.join(map(str, r['seeds']))})"
        print(row)


def write_csv(rows: list[dict], benchmarks: list[str], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["method"]
        for b in benchmarks:
            header += [f"{b}_mean", f"{b}_std", f"{b}_n"]
        header += ["avg", "seeds"]
        w.writerow(header)
        for r in rows:
            line = [r["label"]]
            for b in benchmarks:
                stat = r["stats"].get(b)
                if stat is None:
                    line += ["", "", 0]
                else:
                    line += [f"{stat[0]:.4f}", f"{stat[1]:.4f}", stat[2]]
            line += [
                f"{r['avg']:.4f}" if r["avg"] is not None else "",
                ";".join(map(str, r["seeds"])),
            ]
            w.writerow(line)
    print(f"  CSV → {path}")


def write_markdown(
    rows: list[dict], benchmarks: list[str], title: str, path: Path
) -> None:
    lines = [f"## {title}", ""]
    head = ["Method"] + benchmarks + ["Avg", "n"]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join(["---"] * len(head)) + "|")
    best_avg = max((r["avg"] for r in rows if r["avg"] is not None), default=None)
    for r in rows:
        cells = [r["label"]]
        for b in benchmarks:
            stat = r["stats"].get(b)
            cells.append("—" if stat is None else f"{stat[0]:.2f}±{stat[1]:.2f}")
        if r["avg"] is None:
            cells.append("—")
        else:
            cell = f"{r['avg']:.2f}"
            if best_avg is not None and abs(r["avg"] - best_avg) < 1e-9:
                cell = f"**{cell}**"
            cells.append(cell)
        cells.append(str(len(r["seeds"])))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Markdown → {path}")


def write_latex(
    rows: list[dict], benchmarks: list[str], title: str, path: Path
) -> None:
    cols = "l" + "c" * (len(benchmarks) + 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + title + r"}",
        r"\begin{tabular}{" + cols + "}",
        r"\toprule",
        "Method & " + " & ".join(benchmarks) + r" & Avg \\",
        r"\midrule",
    ]
    best_avg = max((r["avg"] for r in rows if r["avg"] is not None), default=None)
    for r in rows:
        cells = [r["label"].replace("_", r"\_")]
        for b in benchmarks:
            stat = r["stats"].get(b)
            cells.append("—" if stat is None else f"{stat[0]:.2f}\\tiny$\\pm${stat[1]:.2f}")
        if r["avg"] is None:
            cells.append("—")
        else:
            cell = f"{r['avg']:.2f}"
            if best_avg is not None and abs(r["avg"] - best_avg) < 1e-9:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  LaTeX → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate benchmark_rouge.json results.")
    parser.add_argument(
        "--root", action="append", required=True,
        help="Root directory to scan (repeat for multiple).",
    )
    parser.add_argument("--tag", default=None, help="Title for table (default: root path).")
    parser.add_argument("--csv", default=None, help="Optional CSV output path.")
    parser.add_argument("--markdown", default=None, help="Optional Markdown output path.")
    parser.add_argument("--latex", default=None, help="Optional LaTeX output path.")
    parser.add_argument(
        "--method_filter", default=None,
        help="Substring filter on label (e.g. 'sagd' to keep only sagd-related rows).",
    )
    args = parser.parse_args()

    grouped: dict[str, dict[int, dict]] = defaultdict(dict)
    for root_str in args.root:
        root = Path(root_str)
        if not root.exists():
            print(f"[WARN] root not found: {root}")
            continue
        sub = discover(root)
        if not sub:
            print(f"[WARN] no benchmark_rouge.json under {root}")
            continue
        prefix = root.name if len(args.root) > 1 else ""
        for label, seeds in sub.items():
            key = f"{prefix}/{label}" if prefix else label
            grouped[key].update(seeds)

    if not grouped:
        print("No results found.")
        return

    if args.method_filter:
        grouped = {k: v for k, v in grouped.items() if args.method_filter in k}
        if not grouped:
            print(f"No results match filter '{args.method_filter}'.")
            return

    benchmarks = collect_benchmarks(grouped)
    rows = compute_rows(grouped, benchmarks)

    title = args.tag or " | ".join(args.root)
    print_text_table(rows, benchmarks, title)

    if args.csv:
        write_csv(rows, benchmarks, Path(args.csv))
    if args.markdown:
        write_markdown(rows, benchmarks, title, Path(args.markdown))
    if args.latex:
        write_latex(rows, benchmarks, title, Path(args.latex))

    n_runs = sum(len(r["seeds"]) for r in rows)
    print(f"\n  Total: {len(rows)} configs × {n_runs} runs total.")


if __name__ == "__main__":
    main()
