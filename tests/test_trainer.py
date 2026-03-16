"""Integration tests for the Trainer using a tiny model."""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch

from sagd.data import InstructionDataset
from sagd.saliency import SaliencyComputer
from sagd.trainer import Trainer


@pytest.fixture(scope="module")
def tiny_dataset(tiny_tokenizer):
    """Small dataset for integration tests."""
    return InstructionDataset(
        tokenizer=tiny_tokenizer,
        dataset_name="databricks/databricks-dolly-15k",
        max_seq_len=32,
        max_samples=16,
        seed=42,
    )


@pytest.fixture
def teacher_model(tiny_model):
    """Teacher (frozen, eval)."""
    model = tiny_model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@pytest.fixture
def student_model(tiny_tokenizer):
    """Separate student model instance."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    model.train()
    return model


def _make_fake_saliency_cache(
    dataset: InstructionDataset,
    model,
    cache_path: str,
) -> None:
    """Create a fake teacher saliency cache for testing."""
    from sagd.data import collate_fn
    computer = SaliencyComputer()
    all_sal = [None] * len(dataset)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, collate_fn=collate_fn,
    )
    for batch in loader:
        sal = computer.compute(
            model, batch["input_ids"], batch["attention_mask"], batch["labels_mask"],
        )
        for i, idx in enumerate(batch["index"].tolist()):
            real_len = batch["attention_mask"][i].sum().item()
            all_sal[idx] = sal[i, :real_len].cpu()

    torch.save({
        "saliency": all_sal,
        "metadata": {"model": "tiny", "data": "dolly", "n_samples": len(dataset), "max_seq_len": 32},
    }, cache_path)


class TestTrainer:
    def test_standard_kd_trains(
        self, teacher_model, student_model, tiny_tokenizer, tiny_dataset,
    ):
        """Standard KD runs without error."""
        config = {
            "method": "standard_kd",
            "device": "cpu",
            "epochs": 1,
            "batch_size": 4,
            "gradient_accumulation": 1,
            "lr": 1e-4,
            "weight_decay": 0.01,
            "warmup_ratio": 0.0,
            "max_grad_norm": 1.0,
            "temperature": 2.0,
            "fp16": False,
            "log_every": 1,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                teacher_model, student_model, tiny_tokenizer, tiny_dataset, config,
            )
            history = trainer.train(tmpdir)
            assert len(history["loss"]) > 0
            assert os.path.exists(os.path.join(tmpdir, "student_final.pt"))

    def test_sagd_trains(
        self, teacher_model, student_model, tiny_tokenizer, tiny_dataset,
    ):
        """SaGD runs without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "teacher_sal.pt")
            _make_fake_saliency_cache(tiny_dataset, teacher_model, cache_path)

            config = {
                "method": "sagd",
                "device": "cpu",
                "epochs": 1,
                "batch_size": 4,
                "gradient_accumulation": 1,
                "lr": 1e-4,
                "weight_decay": 0.01,
                "warmup_ratio": 0.0,
                "max_grad_norm": 1.0,
                "temperature": 2.0,
                "fp16": False,
                "log_every": 1,
                "teacher_saliency_path": cache_path,
                "lambda_sal": 0.5,
                "sagd_every_n_steps": 2,
                "sagd_tau_w": 1.0,
                "saliency_temperature": 2.0,
            }
            trainer = Trainer(
                teacher_model, student_model, tiny_tokenizer, tiny_dataset, config,
            )
            history = trainer.train(tmpdir)
            assert len(history["loss"]) > 0

    def test_sagd_logs_metrics(
        self, teacher_model, student_model, tiny_tokenizer, tiny_dataset,
    ):
        """SaGD logs sagd/ fields in training_stats.jsonl."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "teacher_sal.pt")
            _make_fake_saliency_cache(tiny_dataset, teacher_model, cache_path)

            config = {
                "method": "sagd",
                "device": "cpu",
                "epochs": 1,
                "batch_size": 4,
                "gradient_accumulation": 1,
                "lr": 1e-4,
                "weight_decay": 0.01,
                "warmup_ratio": 0.0,
                "max_grad_norm": 1.0,
                "temperature": 2.0,
                "fp16": False,
                "log_every": 1,
                "teacher_saliency_path": cache_path,
                "lambda_sal": 0.5,
                "sagd_every_n_steps": 1,  # Every step for testing
                "sagd_tau_w": 1.0,
                "saliency_temperature": 2.0,
            }
            trainer = Trainer(
                teacher_model, student_model, tiny_tokenizer, tiny_dataset, config,
            )
            trainer.train(tmpdir)

            stats_path = os.path.join(tmpdir, "training_stats.jsonl")
            assert os.path.exists(stats_path)

            with open(stats_path) as f:
                lines = f.readlines()
            assert len(lines) > 0

            # At least one logged line should have sagd/ fields
            sagd_logged = False
            for line in lines:
                entry = json.loads(line)
                if "sagd/sal_loss" in entry:
                    sagd_logged = True
                    assert "sagd/mean_jsd" in entry
                    assert "sagd/max_weight" in entry
                    assert "sagd/min_weight" in entry
                    break
            assert sagd_logged, "No sagd/ metrics found in training_stats.jsonl"

    def test_baselines_unaffected(
        self, teacher_model, student_model, tiny_tokenizer, tiny_dataset,
    ):
        """Standard KD does not initialize SaGD components."""
        config = {
            "method": "standard_kd",
            "device": "cpu",
            "epochs": 1,
            "batch_size": 4,
            "gradient_accumulation": 1,
            "lr": 1e-4,
            "weight_decay": 0.01,
            "warmup_ratio": 0.0,
            "max_grad_norm": 1.0,
            "temperature": 2.0,
            "fp16": False,
            "log_every": 1,
        }
        trainer = Trainer(
            teacher_model, student_model, tiny_tokenizer, tiny_dataset, config,
        )
        assert trainer.saliency_computer is None
        assert trainer.sal_align_loss is None
        assert trainer.teacher_saliency_cache is None
