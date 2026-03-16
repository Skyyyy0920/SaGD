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

    def test_gradient_flows_to_student(self, tiny_model):
        """Saliency alignment loss produces gradients for student parameters."""
        model = copy.deepcopy(tiny_model)
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)

        computer = SaliencyComputer()
        loss_fn = SaliencyAlignmentLoss()

        B, L = 1, 6
        input_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)
        labels_mask = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long)

        # Teacher saliency (fixed)
        teacher_sal = torch.rand(B, L) * (1 - labels_mask).float()

        # Student saliency (needs grad flow)
        embed_layer = model.get_input_embeddings()
        embed = embed_layer(input_ids)  # NOT detached — grad flows through params
        outputs = model(inputs_embeds=embed, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        mask = labels_mask[:, 1:].float()
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        response_ll = (token_lp * mask).sum()
        # Compute saliency-like gradient on embed
        grad = torch.autograd.grad(response_ll, embed, create_graph=True)[0]
        student_sal = grad.norm(dim=-1) * (1 - labels_mask).float()

        loss, _ = loss_fn(teacher_sal, student_sal, labels_mask, attention_mask)
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        assert has_grad, "No gradients flowed to model parameters"
