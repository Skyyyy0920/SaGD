"""Loss functions for knowledge distillation.

Implements all baseline methods from DA-KD (ICML 2025) comparison:
  - StandardKDLoss: Forward KL (Hinton, 2015)
  - ReverseKLLoss: Reverse KL / MiniLLM (Gu et al., 2024)
  - SFTLoss: Supervised fine-tuning (cross-entropy on ground truth)
    Also used by SeqKD (Kim & Rush, 2016) with teacher argmax as targets.
  - JSDLoss: Generalized JSD for GKD (Agarwal et al., 2023)
  - SkewKLLoss: Skew KL for DistiLLM (Ko et al., 2024)
  - BDLLoss: Bidirectional Discrepancy Loss for DA-KD (He et al., 2025)
"""

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


class SFTLoss(nn.Module):
    """Supervised fine-tuning loss (cross-entropy on ground truth labels).

    No teacher involved — baseline for comparison.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        student_logits: torch.Tensor,  # (B, L, V)
        input_ids: torch.Tensor,       # (B, L)
        labels_mask: torch.Tensor,     # (B, L)
    ) -> torch.Tensor:
        """Compute cross-entropy on response tokens with shift alignment."""
        # Shift alignment
        shift_logits = student_logits[:, :-1, :].contiguous()  # (B, L-1, V)
        shift_labels = input_ids[:, 1:].contiguous()            # (B, L-1)
        mask = labels_mask[:, 1:].float()                       # (B, L-1)

        # Per-token cross-entropy
        log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, L-1, V)
        token_nll = -log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # (B, L-1)

        # Masked mean
        masked_nll = token_nll * mask  # (B, L-1)
        mask_count = mask.sum(dim=-1).clamp(min=1)  # (B,)
        per_sample = masked_nll.sum(dim=-1) / mask_count  # (B,)

        return per_sample.mean()


class JSDLoss(nn.Module):
    """Generalized JSD loss for GKD (Agarwal et al., 2023).

    D_JSD(P_T, P_S) = β·KL(P_T||M) + (1-β)·KL(P_S||M)
    where M = β·P_T + (1-β)·P_S.

    Args:
        temperature: Softmax temperature.
        beta: Mixing coefficient (default 0.5 = symmetric JSD).
        eps: Numerical stability constant.
    """

    def __init__(
        self, temperature: float = 2.0, beta: float = 0.5, eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.beta = beta
        self.eps = eps

    def forward(
        self,
        teacher_logits: torch.Tensor,  # (B, L, V)
        student_logits: torch.Tensor,  # (B, L, V)
        labels_mask: torch.Tensor,     # (B, L)
    ) -> torch.Tensor:
        """Compute generalized JSD with shift alignment."""
        t_logits = teacher_logits[:, :-1, :]  # (B, L-1, V)
        s_logits = student_logits[:, :-1, :]  # (B, L-1, V)
        mask = labels_mask[:, 1:].float()     # (B, L-1)

        t_probs = F.softmax(t_logits / self.temperature, dim=-1)  # (B, L-1, V)
        s_probs = F.softmax(s_logits / self.temperature, dim=-1)  # (B, L-1, V)

        # Mixture distribution
        m_probs = self.beta * t_probs + (1 - self.beta) * s_probs  # (B, L-1, V)
        m_log = torch.log(m_probs.clamp(min=self.eps))

        # KL(P_T || M)
        t_log = torch.log(t_probs.clamp(min=self.eps))
        kl_t_m = (t_probs * (t_log - m_log)).sum(dim=-1)  # (B, L-1)

        # KL(P_S || M)
        s_log = torch.log(s_probs.clamp(min=self.eps))
        kl_s_m = (s_probs * (s_log - m_log)).sum(dim=-1)  # (B, L-1)

        per_pos = self.beta * kl_t_m + (1 - self.beta) * kl_s_m  # (B, L-1)

        # Masked mean
        per_pos = per_pos * mask
        mask_count = mask.sum(dim=-1).clamp(min=1)
        per_sample = per_pos.sum(dim=-1) / mask_count

        loss = (per_sample * self.temperature ** 2).mean()
        return loss


class SkewKLLoss(nn.Module):
    """Skew KL divergence for DistiLLM (Ko et al., 2024).

    SKL: KL(α·P_T + (1-α)·P_S || P_S)
    Smooths teacher distribution with student to stabilize training.

    Args:
        temperature: Softmax temperature.
        alpha: Skew coefficient (default 0.5).
        eps: Numerical stability constant.
    """

    def __init__(
        self, temperature: float = 2.0, alpha: float = 0.5, eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.eps = eps

    def forward(
        self,
        teacher_logits: torch.Tensor,  # (B, L, V)
        student_logits: torch.Tensor,  # (B, L, V)
        labels_mask: torch.Tensor,     # (B, L)
    ) -> torch.Tensor:
        """Compute skew KL with shift alignment."""
        t_logits = teacher_logits[:, :-1, :]  # (B, L-1, V)
        s_logits = student_logits[:, :-1, :]  # (B, L-1, V)
        mask = labels_mask[:, 1:].float()     # (B, L-1)

        t_probs = F.softmax(t_logits / self.temperature, dim=-1)
        s_probs = F.softmax(s_logits / self.temperature, dim=-1)
        s_log_probs = F.log_softmax(s_logits / self.temperature, dim=-1)

        # Skewed distribution: α·P_T + (1-α)·P_S
        skew_probs = self.alpha * t_probs + (1 - self.alpha) * s_probs
        skew_log = torch.log(skew_probs.clamp(min=self.eps))

        # KL(skew || P_S)
        per_pos = (skew_probs * (skew_log - s_log_probs)).sum(dim=-1)  # (B, L-1)

        per_pos = per_pos * mask
        mask_count = mask.sum(dim=-1).clamp(min=1)
        per_sample = per_pos.sum(dim=-1) / mask_count

        loss = (per_sample * self.temperature ** 2).mean()
        return loss


class BDLLoss(nn.Module):
    """Bidirectional Discrepancy Loss for DA-KD (He et al., ICML 2025).

    D_BDL(p, q) = KL((1-λ)p + λq || λp + (1-λ)q)

    Stabilizes gradients on difficult samples by bounding the gradient
    coefficient C(x) within a finite range determined by λ.

    Args:
        temperature: Softmax temperature.
        bdl_lambda: Mixing coefficient λ (default 0.9, from DA-KD paper).
        eps: Numerical stability constant.
    """

    def __init__(
        self, temperature: float = 2.0, bdl_lambda: float = 0.9, eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.bdl_lambda = bdl_lambda
        self.eps = eps

    def forward(
        self,
        teacher_logits: torch.Tensor,  # (B, L, V)
        student_logits: torch.Tensor,  # (B, L, V)
        labels_mask: torch.Tensor,     # (B, L)
    ) -> torch.Tensor:
        """Compute BDL with shift alignment."""
        t_logits = teacher_logits[:, :-1, :]  # (B, L-1, V)
        s_logits = student_logits[:, :-1, :]  # (B, L-1, V)
        mask = labels_mask[:, 1:].float()     # (B, L-1)

        lam = self.bdl_lambda
        t_probs = F.softmax(t_logits / self.temperature, dim=-1)
        s_probs = F.softmax(s_logits / self.temperature, dim=-1)

        # P_m = (1-λ)p + λq,  Q_m = λp + (1-λ)q
        p_m = (1 - lam) * t_probs + lam * s_probs
        q_m = lam * t_probs + (1 - lam) * s_probs

        # KL(P_m || Q_m)
        p_m_log = torch.log(p_m.clamp(min=self.eps))
        q_m_log = torch.log(q_m.clamp(min=self.eps))
        per_pos = (p_m * (p_m_log - q_m_log)).sum(dim=-1)  # (B, L-1)

        per_pos = per_pos * mask
        mask_count = mask.sum(dim=-1).clamp(min=1)
        per_sample = per_pos.sum(dim=-1) / mask_count

        loss = (per_sample * self.temperature ** 2).mean()
        return loss
