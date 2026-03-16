"""KL divergence losses for knowledge distillation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardKDLoss(nn.Module):
    """Forward KL: KL(P_T || P_S), sequence-level with response mask.

    Args:
        temperature: Softmax temperature for logit scaling.
        eps: Numerical stability constant.
    """

    def __init__(self, temperature: float = 2.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        teacher_logits: torch.Tensor,  # (B, L, V)
        student_logits: torch.Tensor,  # (B, L, V)
        labels_mask: torch.Tensor,     # (B, L)
    ) -> torch.Tensor:
        """Compute forward KL with shift alignment.

        logit[j] predicts token[j+1], so use logits[:, :-1] with labels_mask[:, 1:].
        """
        # Shift alignment: logit[j] predicts token[j+1]
        t_logits = teacher_logits[:, :-1, :]  # (B, L-1, V)
        s_logits = student_logits[:, :-1, :]  # (B, L-1, V)
        mask = labels_mask[:, 1:].float()     # (B, L-1)

        # Softmax with temperature
        t_probs = F.softmax(t_logits / self.temperature, dim=-1)  # (B, L-1, V)
        s_log_probs = F.log_softmax(s_logits / self.temperature, dim=-1)  # (B, L-1, V)
        t_log_probs = torch.log(t_probs.clamp(min=self.eps))  # (B, L-1, V)

        # Per-position KL: sum over vocab
        per_pos_kl = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)  # (B, L-1)

        # Masked mean over response positions
        per_pos_kl = per_pos_kl * mask  # (B, L-1)
        mask_count = mask.sum(dim=-1).clamp(min=1)  # (B,)
        per_sample_kl = per_pos_kl.sum(dim=-1) / mask_count  # (B,)

        # Scale by T² and return batch mean
        loss = (per_sample_kl * self.temperature ** 2).mean()  # scalar
        return loss


class ReverseKLLoss(nn.Module):
    """Reverse KL: KL(P_S || P_T), same interface as StandardKDLoss.

    Args:
        temperature: Softmax temperature for logit scaling.
        eps: Numerical stability constant.
    """

    def __init__(self, temperature: float = 2.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        teacher_logits: torch.Tensor,  # (B, L, V)
        student_logits: torch.Tensor,  # (B, L, V)
        labels_mask: torch.Tensor,     # (B, L)
    ) -> torch.Tensor:
        """Compute reverse KL with shift alignment."""
        # Shift alignment
        t_logits = teacher_logits[:, :-1, :]  # (B, L-1, V)
        s_logits = student_logits[:, :-1, :]  # (B, L-1, V)
        mask = labels_mask[:, 1:].float()     # (B, L-1)

        # Softmax with temperature
        s_probs = F.softmax(s_logits / self.temperature, dim=-1)  # (B, L-1, V)
        s_log_probs = torch.log(s_probs.clamp(min=self.eps))  # (B, L-1, V)
        t_log_probs = F.log_softmax(t_logits / self.temperature, dim=-1)  # (B, L-1, V)

        # Per-position reverse KL: KL(S || T)
        per_pos_kl = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)  # (B, L-1)

        # Masked mean over response positions
        per_pos_kl = per_pos_kl * mask  # (B, L-1)
        mask_count = mask.sum(dim=-1).clamp(min=1)  # (B,)
        per_sample_kl = per_pos_kl.sum(dim=-1) / mask_count  # (B,)

        loss = (per_sample_kl * self.temperature ** 2).mean()  # scalar
        return loss
