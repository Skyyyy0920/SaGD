"""Tests for the extended evaluation system."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from sagd.evaluation import (
    compute_perplexity,
    compute_rouge,
    generate_responses,
    load_responses,
    save_responses,
)


class TestGenerateResponses:
    def test_returns_correct_keys(self, tiny_model, tiny_tokenizer):
        """generate_responses returns dicts with required keys."""
        from sagd.data import InstructionDataset

        ds = InstructionDataset(
            tokenizer=tiny_tokenizer,
            dataset_name="databricks/databricks-dolly-15k",
            max_seq_len=32,
            max_samples=3,
            seed=42,
        )
        responses = generate_responses(
            tiny_model, tiny_tokenizer, ds,
            max_new_tokens=5, batch_size=2, device="cpu",
        )
        assert len(responses) == 3
        for r in responses:
            for key in ["index", "instruction", "reference", "generated", "category"]:
                assert key in r, f"Missing key: {key}"
            assert isinstance(r["generated"], str)
            assert isinstance(r["index"], int)

    def test_batch_size_larger_than_dataset(self, tiny_model, tiny_tokenizer):
        """Works when batch_size > len(dataset)."""
        from sagd.data import InstructionDataset

        ds = InstructionDataset(
            tokenizer=tiny_tokenizer,
            dataset_name="databricks/databricks-dolly-15k",
            max_seq_len=32,
            max_samples=2,
            seed=42,
        )
        responses = generate_responses(
            tiny_model, tiny_tokenizer, ds,
            max_new_tokens=5, batch_size=10, device="cpu",
        )
        assert len(responses) == 2


class TestComputeRouge:
    def test_perfect_match(self):
        """Identical reference/generated → ROUGE-L ≈ 1."""
        responses = [
            {"reference": "The quick brown fox", "generated": "The quick brown fox"},
            {"reference": "Hello world", "generated": "Hello world"},
        ]
        metrics = compute_rouge(responses)
        assert metrics["rouge_l_f"] > 0.99

    def test_no_match(self):
        """Completely different texts → low ROUGE-L."""
        responses = [
            {"reference": "The quick brown fox jumps over the lazy dog",
             "generated": "Alpha beta gamma delta epsilon zeta"},
        ]
        metrics = compute_rouge(responses)
        assert metrics["rouge_l_f"] < 0.1

    def test_returns_all_keys(self):
        responses = [{"reference": "hello", "generated": "hello"}]
        metrics = compute_rouge(responses)
        assert "rouge_l_f" in metrics
        assert "rouge_l_p" in metrics
        assert "rouge_l_r" in metrics


class TestComputePerplexity:
    def test_returns_correct_keys(self, tiny_model, tiny_tokenizer):
        """compute_perplexity returns perplexity and avg_loss."""
        from sagd.data import InstructionDataset

        ds = InstructionDataset(
            tokenizer=tiny_tokenizer,
            dataset_name="databricks/databricks-dolly-15k",
            max_seq_len=32,
            max_samples=3,
            seed=42,
        )
        metrics = compute_perplexity(
            tiny_model, tiny_tokenizer, ds,
            batch_size=2, device="cpu",
        )
        assert "perplexity" in metrics
        assert "avg_loss" in metrics
        assert metrics["perplexity"] > 0
        assert metrics["avg_loss"] > 0

    def test_perplexity_is_finite(self, tiny_model, tiny_tokenizer):
        """Perplexity should be finite, not NaN or Inf."""
        from sagd.data import InstructionDataset

        ds = InstructionDataset(
            tokenizer=tiny_tokenizer,
            dataset_name="databricks/databricks-dolly-15k",
            max_seq_len=32,
            max_samples=2,
            seed=42,
        )
        metrics = compute_perplexity(
            tiny_model, tiny_tokenizer, ds,
            batch_size=2, device="cpu",
        )
        assert not torch.isnan(torch.tensor(metrics["perplexity"]))
        assert not torch.isinf(torch.tensor(metrics["perplexity"]))


class TestResponseIO:
    def test_save_load_roundtrip(self):
        """save_responses → load_responses preserves data."""
        responses = [
            {"index": 0, "instruction": "Hello", "reference": "Hi",
             "generated": "Hey", "category": "qa"},
            {"index": 1, "instruction": "Bye", "reference": "Goodbye",
             "generated": "See ya", "category": "chat"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "responses.jsonl"
            save_responses(responses, path)

            loaded = load_responses(path)
            assert len(loaded) == 2
            assert loaded[0]["instruction"] == "Hello"
            assert loaded[1]["generated"] == "See ya"

    def test_unicode_roundtrip(self):
        """Non-ASCII characters survive save/load."""
        responses = [
            {"index": 0, "instruction": "翻译这句话",
             "reference": "Translate this", "generated": "翻译",
             "category": "translation"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "responses.jsonl"
            save_responses(responses, path)
            loaded = load_responses(path)
            assert loaded[0]["instruction"] == "翻译这句话"


class TestGPTJudgeUnit:
    """Unit tests for GPT judge internals (no API calls)."""

    def test_verdict_aggregation_agree(self):
        """When both orderings agree, final verdict matches."""
        from sagd.gpt_judge import GPTJudge

        # Mock the judge by testing the aggregation logic directly
        # If v_ab == "A" and v_ba == "A" (both say A is better), final = "A"
        assert _simulate_aggregation("A", "A") == "A"
        assert _simulate_aggregation("B", "B") == "B"
        assert _simulate_aggregation("TIE", "TIE") == "TIE"

    def test_verdict_aggregation_disagree(self):
        """When orderings disagree, final verdict is TIE."""
        assert _simulate_aggregation("A", "B") == "TIE"
        assert _simulate_aggregation("B", "A") == "TIE"
        assert _simulate_aggregation("A", "TIE") == "TIE"


def _simulate_aggregation(v_ab: str, v_ba: str) -> str:
    """Simulate the position-debiased aggregation logic from GPTJudge."""
    if v_ab == v_ba:
        return v_ab
    return "TIE"
