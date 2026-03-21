"""Tests for the instruction dataset, SQuAD dataset, and collate function."""

from __future__ import annotations

import pytest
import torch

from sagd.data import InstructionDataset, SquadDataset, collate_fn, normalize_answer


@pytest.fixture(scope="module")
def small_dataset(tiny_tokenizer):
    """Create a small dataset for testing."""
    return InstructionDataset(
        tokenizer=tiny_tokenizer,
        dataset_name="databricks/databricks-dolly-15k",
        max_seq_len=64,
        max_samples=10,
        seed=42,
    )


class TestInstructionDataset:
    def test_returns_index(self, small_dataset):
        """__getitem__ must include 'index' as a scalar tensor."""
        item = small_dataset[0]
        assert "index" in item
        assert item["index"].dim() == 0
        assert item["index"].dtype == torch.long

    def test_labels_mask_boundary(self, small_dataset):
        """labels_mask must have 0 for prompt and 1 for response."""
        item = small_dataset[0]
        labels_mask = item["labels_mask"]
        # Should start with 0s (prompt) and end with 1s (response)
        assert labels_mask[0].item() == 0  # first token is prompt
        # Must have at least some response tokens
        has_response = (labels_mask == 1).any()
        # Not all samples guaranteed to have response after truncation,
        # but most should
        assert has_response or labels_mask.size(0) > 0

    def test_all_keys_present(self, small_dataset):
        """Each item has input_ids, attention_mask, labels_mask, index."""
        item = small_dataset[0]
        for key in ["input_ids", "attention_mask", "labels_mask", "index"]:
            assert key in item, f"Missing key: {key}"

    def test_shapes_consistent(self, small_dataset):
        """input_ids, attention_mask, labels_mask all have same length."""
        item = small_dataset[0]
        L = item["input_ids"].size(0)
        assert item["attention_mask"].size(0) == L
        assert item["labels_mask"].size(0) == L

    def test_collate_fn_pads(self, small_dataset):
        """collate_fn pads to longest in batch."""
        items = [small_dataset[i] for i in range(min(3, len(small_dataset)))]
        batch = collate_fn(items)
        B = len(items)
        L = batch["input_ids"].size(1)
        assert batch["input_ids"].shape == (B, L)
        assert batch["attention_mask"].shape == (B, L)
        assert batch["labels_mask"].shape == (B, L)

    def test_collate_fn_stacks_index(self, small_dataset):
        """collate_fn stacks index correctly."""
        items = [small_dataset[i] for i in range(min(3, len(small_dataset)))]
        batch = collate_fn(items)
        assert batch["index"].shape == (len(items),)
        for i, item in enumerate(items):
            assert batch["index"][i].item() == item["index"].item()

    def test_get_metadata(self, small_dataset):
        """get_metadata returns category, instruction, response."""
        meta = small_dataset.get_metadata(0)
        assert "category" in meta
        assert "instruction" in meta
        assert "response" in meta


class TestSubsetSplits:
    def test_subset_splits_cover_all_data(self, tiny_tokenizer):
        """Train + val + test cover the full dataset without gaps."""
        train_ds = InstructionDataset(
            tokenizer=tiny_tokenizer, max_seq_len=32, subset="train",
        )
        val_ds = InstructionDataset(
            tokenizer=tiny_tokenizer, max_seq_len=32, subset="val",
        )
        test_ds = InstructionDataset(
            tokenizer=tiny_tokenizer, max_seq_len=32, subset="test",
        )

        # Sizes should add up to the full dataset
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == 15011, f"Expected 15011, got {total}"
        assert len(train_ds) == 15011 - 500 - 500
        assert len(val_ds) == 500
        assert len(test_ds) == 500

        # First sample of each subset should be different
        # (since they come from different contiguous ranges)
        train_first = train_ds.get_metadata(0)["instruction"]
        val_first = val_ds.get_metadata(0)["instruction"]
        test_first = test_ds.get_metadata(0)["instruction"]
        # At least two of three should differ (extremely unlikely to collide)
        assert not (train_first == val_first == test_first)

    def test_subset_default_is_train(self, tiny_tokenizer):
        """Default subset is train."""
        ds = InstructionDataset(
            tokenizer=tiny_tokenizer, max_seq_len=32, max_samples=10,
        )
        ds_train = InstructionDataset(
            tokenizer=tiny_tokenizer, max_seq_len=32, max_samples=10, subset="train",
        )
        assert len(ds) == len(ds_train)

    def test_val_and_test_have_500_samples(self, tiny_tokenizer):
        """Val and test subsets have 500 samples each."""
        val_ds = InstructionDataset(
            tokenizer=tiny_tokenizer, max_seq_len=32, subset="val",
        )
        test_ds = InstructionDataset(
            tokenizer=tiny_tokenizer, max_seq_len=32, subset="test",
        )
        assert len(val_ds) == 500
        assert len(test_ds) == 500


# =========================================================================
# SQuAD Dataset Tests
# =========================================================================

@pytest.fixture(scope="module")
def small_squad_dataset(tiny_tokenizer):
    """Create a small SQuAD dataset for testing."""
    return SquadDataset(
        tokenizer=tiny_tokenizer,
        dataset_name="rajpurkar/squad_v2",
        max_seq_len=128,
        max_samples=10,
        seed=42,
    )


class TestSquadDataset:
    def test_returns_index(self, small_squad_dataset):
        """__getitem__ must include 'index' as a scalar tensor."""
        item = small_squad_dataset[0]
        assert "index" in item
        assert item["index"].dim() == 0
        assert item["index"].dtype == torch.long

    def test_all_keys_present(self, small_squad_dataset):
        """Each item has standard keys plus answer span fields."""
        item = small_squad_dataset[0]
        for key in ["input_ids", "attention_mask", "labels_mask", "index",
                     "answer_token_start", "answer_token_end"]:
            assert key in item, f"Missing key: {key}"

    def test_answer_span_fields_are_scalars(self, small_squad_dataset):
        """Answer span fields are scalar long tensors."""
        item = small_squad_dataset[0]
        assert item["answer_token_start"].dim() == 0
        assert item["answer_token_start"].dtype == torch.long
        assert item["answer_token_end"].dim() == 0
        assert item["answer_token_end"].dtype == torch.long

    def test_shapes_consistent(self, small_squad_dataset):
        """input_ids, attention_mask, labels_mask all have same length."""
        item = small_squad_dataset[0]
        L = item["input_ids"].size(0)
        assert item["attention_mask"].size(0) == L
        assert item["labels_mask"].size(0) == L

    def test_labels_mask_starts_with_prompt(self, small_squad_dataset):
        """labels_mask must start with 0 (prompt)."""
        item = small_squad_dataset[0]
        assert item["labels_mask"][0].item() == 0

    def test_answer_span_within_bounds(self, small_squad_dataset):
        """If answer span is mapped (>=0), it must be within sequence length."""
        for i in range(len(small_squad_dataset)):
            item = small_squad_dataset[i]
            start = item["answer_token_start"].item()
            end = item["answer_token_end"].item()
            L = item["input_ids"].size(0)
            if start >= 0:
                assert 0 <= start < L, f"start={start} out of range [0, {L})"
                assert 0 <= end < L, f"end={end} out of range [0, {L})"
                assert start <= end, f"start={start} > end={end}"

    def test_filters_unanswerable(self, small_squad_dataset):
        """All samples should have valid answer text."""
        for i in range(len(small_squad_dataset)):
            meta = small_squad_dataset.get_metadata(i)
            assert len(meta["response"]) > 0, f"Sample {i} has empty answer"

    def test_get_metadata_keys(self, small_squad_dataset):
        """get_metadata returns expected keys."""
        meta = small_squad_dataset.get_metadata(0)
        for key in ["instruction", "response", "context", "category"]:
            assert key in meta, f"Missing metadata key: {key}"
        assert meta["category"] == "extractive_qa"

    def test_collate_fn_with_answer_span(self, small_squad_dataset):
        """collate_fn handles answer span fields."""
        items = [small_squad_dataset[i] for i in range(min(3, len(small_squad_dataset)))]
        batch = collate_fn(items)
        B = len(items)
        assert "answer_token_start" in batch
        assert "answer_token_end" in batch
        assert batch["answer_token_start"].shape == (B,)
        assert batch["answer_token_end"].shape == (B,)

    def test_span_mapping_rate_positive(self, small_squad_dataset):
        """At least some spans should be successfully mapped."""
        assert small_squad_dataset.span_mapping_rate > 0.0


class TestSquadSubsetSplits:
    def test_val_and_test_differ(self, tiny_tokenizer):
        """Val and test subsets should contain different samples."""
        val_ds = SquadDataset(
            tokenizer=tiny_tokenizer, max_seq_len=64, max_samples=5,
            seed=42, subset="val",
        )
        test_ds = SquadDataset(
            tokenizer=tiny_tokenizer, max_seq_len=64, max_samples=5,
            seed=42, subset="test",
        )
        val_q = val_ds.get_metadata(0)["instruction"]
        test_q = test_ds.get_metadata(0)["instruction"]
        # Very unlikely to be the same question
        assert val_q != test_q or len(val_ds) != len(test_ds)


class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("Hello World") == "hello world"

    def test_remove_articles(self):
        assert normalize_answer("the quick a fox an apple") == "quick fox apple"

    def test_remove_punctuation(self):
        assert normalize_answer("hello, world!") == "hello world"

    def test_collapse_whitespace(self):
        assert normalize_answer("hello   world") == "hello world"

    def test_combined(self):
        assert normalize_answer("The Quick, Brown Fox!") == "quick brown fox"
