#!/usr/bin/env python3
"""CLI for GPT-as-Judge pairwise evaluation.

Reads two JSONL files (pre-generated responses from generate_responses.py)
and asks GPT-4o-mini to pick the better response for each instruction.
Uses position debiasing (judges each pair twice, swapping order).

Example:
    python scripts/gpt_judge.py \
        --responses_a outputs/standard_kd/seed_42/responses.jsonl \
        --responses_b outputs/sagd/seed_42/responses.jsonl \
        --label_a "Standard KD" \
        --label_b "SaGD" \
        --output_path outputs/gpt_judge_std_vs_sagd.json

Environment:
    Set OPENAI_API_KEY or pass --api_key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sagd.evaluation import load_responses
from sagd.gpt_judge import GPTJudge, save_judge_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-as-Judge pairwise evaluation")
    p.add_argument("--responses_a", type=str, required=True,
                    help="JSONL file for model A responses")
    p.add_argument("--responses_b", type=str, required=True,
                    help="JSONL file for model B responses")
    p.add_argument("--label_a", type=str, default="Model_A",
                    help="Display label for model A")
    p.add_argument("--label_b", type=str, default="Model_B",
                    help="Display label for model B")
    p.add_argument("--output_path", type=str, required=True,
                    help="Output JSON file for results")
    p.add_argument("--max_samples", type=int, default=None,
                    help="Limit number of samples to judge (default: all)")
    p.add_argument("--model", type=str, default="gpt-4o-mini",
                    help="OpenAI model for judging")
    p.add_argument("--api_key", type=str, default=None,
                    help="OpenAI API key (or set OPENAI_API_KEY)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load responses
    responses_a = load_responses(args.responses_a)
    responses_b = load_responses(args.responses_b)
    print(f"Loaded {len(responses_a)} responses from {args.responses_a}")
    print(f"Loaded {len(responses_b)} responses from {args.responses_b}")

    if len(responses_a) != len(responses_b):
        print(f"WARNING: Different lengths ({len(responses_a)} vs {len(responses_b)}), "
              f"truncating to shorter.")
        n = min(len(responses_a), len(responses_b))
        responses_a = responses_a[:n]
        responses_b = responses_b[:n]

    if args.max_samples is not None:
        responses_a = responses_a[:args.max_samples]
        responses_b = responses_b[:args.max_samples]
        print(f"Limited to {args.max_samples} samples")

    # Judge
    judge = GPTJudge(api_key=args.api_key, model=args.model)
    results = judge.judge_pairwise(
        responses_a, responses_b,
        label_a=args.label_a,
        label_b=args.label_b,
    )

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"GPT-as-Judge Results: {args.label_a} vs {args.label_b}")
    print(f"{'=' * 50}")
    print(f"  {args.label_a} wins: {results['wins_a']} ({results['win_rate_a']:.1%})")
    print(f"  {args.label_b} wins: {results['wins_b']} ({results['win_rate_b']:.1%})")
    print(f"  Ties:         {results['ties']} ({results['tie_rate']:.1%})")
    print(f"  Total:        {results['n_samples']}")

    # Save
    save_judge_results(results, args.output_path)
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
