"""Tests for saliency computation and alignment."""

from __future__ import annotations

import copy

import pytest
import torch

from sagd.saliency import SaliencyAlignmentLoss, SaliencyComputer


class TestSaliencyComputer:
    def test_output_shape(self, tiny_model, sample_batch):
        """Saliency output has shape (B, L)."""
        computer = SaliencyComputer()
        sal = computer.compute(
            tiny_model,
            sample_batch["input_ids"],
            sample_batch["attention_mask"],
            sample_batch["labels_mask"],
        )
        B, L = sample_batch["input_ids"].shape
        assert sal.shape == (B, L)

    def test_response_positions_zeroed(self, tiny_model, sample_batch):
        """Saliency at response positions (labels_mask=1) must be zero."""
        computer = SaliencyComputer()
        sal = computer.compute(
            tiny_model,
            sample_batch["input_ids"],
            sample_batch["attention_mask"],
            sample_batch["labels_mask"],
        )
        response_mask = sample_batch["labels_mask"].bool()
        assert (sal[response_mask] == 0).all()

    def test_padding_positions_zeroed(self, tiny_model, sample_batch_with_padding):
        """Saliency at padding positions (attention_mask=0) must be zero."""
        computer = SaliencyComputer()
        sal = computer.compute(
            tiny_model,
            sample_batch_with_padding["input_ids"],
            sample_batch_with_padding["attention_mask"],
            sample_batch_with_padding["labels_mask"],
        )
        pad_mask = (sample_batch_with_padding["attention_mask"] == 0)
        assert (sal[pad_mask] == 0).all()

    def test_no_model_params_updated(self, tiny_model, sample_batch):
        """Model parameters must not change during saliency computation."""
        params_before = {n: p.clone() for n, p in tiny_model.named_parameters()}
        computer = SaliencyComputer()
        computer.compute(
            tiny_model,
            sample_batch["input_ids"],
            sample_batch["attention_mask"],
            sample_batch["labels_mask"],
        )
        for n, p in tiny_model.named_parameters():
            assert torch.equal(p, params_before[n]), f"Parameter {n} changed"

    def test_no_param_grad_pollution(self, tiny_model, sample_batch):
        """Model parameters must not accumulate gradients."""
        # Clear any existing grads
        for p in tiny_model.parameters():
            p.grad = None
        computer = SaliencyComputer()
        computer.compute(
            tiny_model,
            sample_batch["input_ids"],
            sample_batch["attention_mask"],
            sample_batch["labels_mask"],
        )
        for n, p in tiny_model.named_parameters():
            assert p.grad is None, f"Parameter {n} has gradient"


class TestSaliencyDistribution:
    def test_sums_to_one_at_prompt(self, tiny_model, sample_batch):
        """Saliency distribution sums to ~1 at prompt positions."""
        computer = SaliencyComputer()
        sal = computer.compute(
            tiny_model,
            sample_batch["input_ids"],
            sample_batch["attention_mask"],
            sample_batch["labels_mask"],
        )
        dist = computer.to_distribution(
            sal, sample_batch["labels_mask"], sample_batch["attention_mask"],
        )
        # Sum per sample should be ~1
        sums = dist.sum(dim=-1)  # (B,)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_handles_all_response(self, tiny_model):
        """All-response sample (no prompt) does not crash."""
        computer = SaliencyComputer()
        B, L = 1, 5
        sal = torch.rand(B, L)
        labels_mask = torch.ones(B, L, dtype=torch.long)  # all response
        attention_mask = torch.ones(B, L, dtype=torch.long)
        dist = computer.to_distribution(sal, labels_mask, attention_mask)
        assert dist.shape == (B, L)
        assert not torch.isnan(dist).any()


class TestSaliencyDivergence:
    def test_identical_is_zero(self):
        """Identical saliency → JSD = 0."""
        computer = SaliencyComputer()
        B, L = 2, 8
        sal = torch.rand(B, L)
        labels_mask = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]] * B, dtype=torch.long)
        attention_mask = torch.ones(B, L, dtype=torch.long)
        # Zero out response positions to match pre-masking
        sal = sal * (1 - labels_mask).float()
        jsd = computer.divergence(sal, sal.clone(), labels_mask, attention_mask)
        assert torch.allclose(jsd, torch.zeros(B), atol=1e-6)

    def test_different_is_positive(self):
        """Different saliency → JSD > 0."""
        computer = SaliencyComputer()
        B, L = 2, 8
        labels_mask = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]] * B, dtype=torch.long)
        attention_mask = torch.ones(B, L, dtype=torch.long)
        prompt_mask = (1 - labels_mask).float()
        sal_T = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0]] * B) * prompt_mask
        sal_S = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0]] * B) * prompt_mask
        jsd = computer.divergence(sal_T, sal_S, labels_mask, attention_mask)
        assert (jsd > 0).all()


class TestSaliencyAlignmentLoss:
    def test_identical_near_zero(self):
        """Identical saliency → loss ≈ 0."""
        loss_fn = SaliencyAlignmentLoss()
        B, L = 2, 8
        sal = torch.rand(B, L)
        labels_mask = torch.zeros(B, L, dtype=torch.long)
        attention_mask = torch.ones(B, L, dtype=torch.long)
        loss, stats = loss_fn(sal, sal.clone(), labels_mask, attention_mask)
        assert loss.item() < 1e-6
        assert abs(stats["mean_cos_sim"] - 1.0) < 1e-6


class TestSaliencyComputeDifferentiable:
    def test_output_shape(self, tiny_model, sample_batch):
        """Differentiable saliency has shape (B, L)."""
        model = copy.deepcopy(tiny_model)
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
        computer = SaliencyComputer()
        sal = computer.compute_differentiable(
            model, sample_batch["input_ids"],
            sample_batch["attention_mask"], sample_batch["labels_mask"],
        )
        B, L = sample_batch["input_ids"].shape
        assert sal.shape == (B, L)

    def test_requires_grad(self, tiny_model, sample_batch):
        """Differentiable saliency has requires_grad=True."""
        model = copy.deepcopy(tiny_model)
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
        computer = SaliencyComputer()
        sal = computer.compute_differentiable(
            model, sample_batch["input_ids"],
            sample_batch["attention_mask"], sample_batch["labels_mask"],
        )
        assert sal.requires_grad, "Differentiable saliency must have requires_grad=True"

    def test_gradient_reaches_model_params(self, tiny_model, sample_batch):
        """sal_loss.backward() produces non-zero gradients on model parameters."""
        model = copy.deepcopy(tiny_model)
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
            p.grad = None

        computer = SaliencyComputer()
        loss_fn = SaliencyAlignmentLoss()

        student_sal = computer.compute_differentiable(
            model, sample_batch["input_ids"],
            sample_batch["attention_mask"], sample_batch["labels_mask"],
        )
        # Fake teacher saliency (fixed, no grad)
        teacher_sal = torch.rand_like(student_sal).detach()
        prompt_mask = (1 - sample_batch["labels_mask"]).float() * sample_batch["attention_mask"].float()
        teacher_sal = teacher_sal * prompt_mask

        sal_loss, _ = loss_fn(teacher_sal, student_sal,
                              sample_batch["labels_mask"], sample_batch["attention_mask"])
        sal_loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "sal_loss.backward() produced no gradients on model parameters"

    def test_non_differentiable_has_no_grad(self, tiny_model, sample_batch):
        """Contrast: compute() returns tensor without grad."""
        computer = SaliencyComputer()
        sal = computer.compute(
            tiny_model, sample_batch["input_ids"],
            sample_batch["attention_mask"], sample_batch["labels_mask"],
        )
        assert not sal.requires_grad, "Non-differentiable compute() should return detached tensor"
