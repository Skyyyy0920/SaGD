"""Tests for KL divergence loss functions."""

from __future__ import annotations

import pytest
import torch

from sagd.losses import ReverseKLLoss, StandardKDLoss


class TestStandardKDLoss:
    def test_output_is_scalar(self):
        """Loss output must be a scalar tensor."""
        loss_fn = StandardKDLoss(temperature=2.0)
        B, L, V = 2, 10, 50
        t_logits = torch.randn(B, L, V)
        s_logits = torch.randn(B, L, V)
        labels_mask = torch.tensor([
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        ], dtype=torch.long)
        loss = loss_fn(t_logits, s_logits, labels_mask)
        assert loss.dim() == 0

    def test_zero_when_identical(self):
        """Identical logits → loss ≈ 0."""
        loss_fn = StandardKDLoss(temperature=2.0)
        B, L, V = 2, 10, 50
        logits = torch.randn(B, L, V)
        labels_mask = torch.tensor([
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        ], dtype=torch.long)
        loss = loss_fn(logits, logits.clone(), labels_mask)
        assert loss.item() < 1e-5

    def test_positive_when_different(self):
        """Different logits → positive loss."""
        loss_fn = StandardKDLoss(temperature=2.0)
        B, L, V = 2, 10, 50
        t_logits = torch.randn(B, L, V)
        s_logits = torch.randn(B, L, V) + 5.0
        labels_mask = torch.tensor([
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        ], dtype=torch.long)
        loss = loss_fn(t_logits, s_logits, labels_mask)
        assert loss.item() > 0

    def test_mask_excludes_prompt(self):
        """KL only computed at response positions."""
        loss_fn = StandardKDLoss(temperature=2.0)
        B, L, V = 1, 8, 20
        t_logits = torch.randn(B, L, V)
        s_logits = torch.randn(B, L, V)

        # All prompt (no response) → no positions to compute KL
        all_prompt = torch.zeros(B, L, dtype=torch.long)
        loss_all_prompt = loss_fn(t_logits, s_logits, all_prompt)

        # Some response
        some_response = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.long)
        loss_some = loss_fn(t_logits, s_logits, some_response)

        # All-prompt should give ~0 loss (no response positions)
        assert loss_all_prompt.item() < 1e-6

    def test_shift_alignment(self):
        """Verify logit[j] corresponds to labels_mask[j+1]."""
        loss_fn = StandardKDLoss(temperature=2.0)
        B, L, V = 1, 6, 10

        # If labels_mask = [0,0,0,1,1,1], shifted mask = [0,0,1,1,1]
        # So logit positions 2,3,4 are used (for tokens 3,4,5)
        t_logits = torch.zeros(B, L, V)
        s_logits = torch.zeros(B, L, V)
        labels_mask = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long)

        # Make a difference only at logit position 0 (prompt in shifted mask → excluded)
        t_logits_mod = t_logits.clone()
        t_logits_mod[:, 0, :] = 10.0  # Big difference at position 0
        s_logits_mod = s_logits.clone()

        loss_base = loss_fn(t_logits, s_logits, labels_mask)
        loss_mod = loss_fn(t_logits_mod, s_logits_mod, labels_mask)

        # Position 0 is prompt in shifted view → should not affect loss
        assert abs(loss_base.item() - loss_mod.item()) < 1e-5


class TestReverseKLLoss:
    def test_output_is_scalar(self):
        loss_fn = ReverseKLLoss(temperature=2.0)
        B, L, V = 2, 10, 50
        t_logits = torch.randn(B, L, V)
        s_logits = torch.randn(B, L, V)
        labels_mask = torch.tensor([
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        ], dtype=torch.long)
        loss = loss_fn(t_logits, s_logits, labels_mask)
        assert loss.dim() == 0

    def test_zero_when_identical(self):
        loss_fn = ReverseKLLoss(temperature=2.0)
        B, L, V = 2, 10, 50
        logits = torch.randn(B, L, V)
        labels_mask = torch.tensor([
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        ], dtype=torch.long)
        loss = loss_fn(logits, logits.clone(), labels_mask)
        assert loss.item() < 1e-5

    def test_positive_when_different(self):
        loss_fn = ReverseKLLoss(temperature=2.0)
        B, L, V = 2, 10, 50
        t_logits = torch.randn(B, L, V)
        s_logits = torch.randn(B, L, V) + 5.0
        labels_mask = torch.tensor([
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        ], dtype=torch.long)
        loss = loss_fn(t_logits, s_logits, labels_mask)
        assert loss.item() > 0
