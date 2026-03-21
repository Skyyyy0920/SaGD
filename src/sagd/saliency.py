"""Input saliency computation and alignment for SaGD.

Saliency measures how much each input token affects response generation:
  s_i = ||∂ log P(response) / ∂ embed_i||

See CLAUDE.md §2.3 for the canonical implementation with all critical details.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SaliencyComputer:
    """Compute input saliency via embedding gradients.

    Args:
        temperature: Softmax temperature for saliency → distribution conversion.
        eps: Numerical stability constant.
    """

    def __init__(self, temperature: float = 2.0, eps: float = 1e-8) -> None:
        self.temperature = temperature
        self.eps = eps

    @torch.enable_grad()
    def compute(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,      # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
        labels_mask: torch.Tensor,     # (B, L)
    ) -> torch.Tensor:
        """Non-differentiable saliency. Used for:
        - Teacher saliency precomputation
        - Saliency diagnosis
        - Reweighting signal (JSD) computation in training

        CRITICAL: temporarily disables model param grads (§5.1).
        CRITICAL: masks with attention_mask, not just labels_mask (§5.2).

        Returns:
            saliency: (B, L) — raw saliency, pre-masked, detached.
        """
        # Save and disable all param grads to prevent pollution
        param_states = {n: p.requires_grad for n, p in model.named_parameters()}
        for p in model.parameters():
            p.requires_grad_(False)

        try:
            # 1. Embed: create leaf tensor disconnected from model parameters
            embed_layer = model.get_input_embeddings()
            embed = embed_layer(input_ids).detach().requires_grad_(True)  # (B, L, d)

            # 2. Forward through the full model
            outputs = model(inputs_embeds=embed, attention_mask=attention_mask)
            logits = outputs.logits  # (B, L, V)

            # 3. Response log-prob with shift alignment
            # logit[j] predicts token[j+1]
            shifted_logits = logits[:, :-1, :]       # (B, L-1, V)
            shifted_targets = input_ids[:, 1:]        # (B, L-1)
            shifted_mask = labels_mask[:, 1:].float()  # (B, L-1)

            log_probs = F.log_softmax(shifted_logits, dim=-1)  # (B, L-1, V)
            token_log_probs = log_probs.gather(
                dim=-1, index=shifted_targets.unsqueeze(-1)
            ).squeeze(-1)  # (B, L-1)

            # Sum log-probs at response positions only
            response_ll = (token_log_probs * shifted_mask).sum()  # scalar

            # 4. Backward: gradients flow to embed only
            response_ll.backward()

            # 5. Saliency: L2 norm of embedding gradient per position
            saliency = embed.grad.norm(dim=-1)  # (B, L)

            # Mask: keep only prompt positions (exclude response AND padding)
            prompt_mask = (1 - labels_mask).float() * attention_mask.float()  # (B, L)
            saliency = saliency * prompt_mask  # (B, L)

        finally:
            # Restore param grad states
            for n, p in model.named_parameters():
                p.requires_grad_(param_states[n])

        return saliency.detach()  # (B, L)

    @torch.enable_grad()
    def compute_differentiable(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,      # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
        labels_mask: torch.Tensor,     # (B, L)
    ) -> torch.Tensor:
        """Differentiable saliency for saliency alignment loss.

        Uses torch.autograd.grad with create_graph=True so that
        sal_loss.backward() propagates second-order gradients to model parameters.

        ONLY used for student saliency in the saliency alignment loss path.
        NOT used for reweighting (reweighting uses non-differentiable compute()).

        Returns:
            saliency: (B, L) — differentiable, gradients flow to model parameters.
        """
        # 1. Embed: DO NOT detach — must stay in the computation graph
        embed = model.get_input_embeddings()(input_ids)  # (B, L, d)
        embed.retain_grad()  # need grad on this intermediate tensor

        # 2. Forward through the full model
        # CRITICAL: Disable efficient/flash SDPA backends — they don't support
        # second-order gradients (create_graph=True). Only the math backend does.
        # Use global flags (more reliable than context managers across PyTorch versions).
        _flash_prev = torch.backends.cuda.flash_sdp_enabled()
        _mem_prev = torch.backends.cuda.mem_efficient_sdp_enabled()
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        try:
            outputs = model(inputs_embeds=embed, attention_mask=attention_mask)
        finally:
            torch.backends.cuda.enable_flash_sdp(_flash_prev)
            torch.backends.cuda.enable_mem_efficient_sdp(_mem_prev)
        logits = outputs.logits  # (B, L, V)

        # 3. Response log-prob with shift alignment
        shifted_logits = logits[:, :-1, :]        # (B, L-1, V)
        shifted_targets = input_ids[:, 1:]         # (B, L-1)
        shifted_mask = labels_mask[:, 1:].float()  # (B, L-1)

        log_probs = F.log_softmax(shifted_logits, dim=-1)  # (B, L-1, V)
        token_log_probs = log_probs.gather(
            dim=-1, index=shifted_targets.unsqueeze(-1)
        ).squeeze(-1)  # (B, L-1)

        response_ll = (token_log_probs * shifted_mask).sum()  # scalar

        # 4. Grad with create_graph=True — this is the key difference
        if response_ll.abs().item() < 1e-9:
            return torch.zeros(
                input_ids.shape[0], input_ids.shape[1],
                device=input_ids.device, requires_grad=True,
            )

        grad = torch.autograd.grad(
            response_ll, embed, create_graph=True, retain_graph=True,
        )[0]  # (B, L, d) — differentiable w.r.t. model parameters

        # 5. Saliency: norm of gradient per position
        saliency = grad.norm(dim=-1)  # (B, L) — differentiable!

        # 6. Mask prompt positions (same logic as compute())
        prompt_mask = (1 - labels_mask).float() * attention_mask.float()  # (B, L)
        saliency = saliency * prompt_mask  # (B, L) — still differentiable

        return saliency  # (B, L) — NOT detached, gradients flow to model params

    def to_distribution(
        self,
        saliency: torch.Tensor,       # (B, L)
        labels_mask: torch.Tensor,     # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:
        """Convert raw saliency to probability distribution over prompt positions.

        Applies softmax(saliency / temperature) over prompt positions only.
        Response and padding positions are set to 0.

        Returns:
            dist: (B, L) — sums to ~1 at prompt positions per sample.
        """
        prompt_mask = (1 - labels_mask).float() * attention_mask.float()  # (B, L)

        # Replace non-prompt positions with -inf for softmax
        scaled = saliency / self.temperature  # (B, L)
        scaled = scaled.masked_fill(prompt_mask == 0, float("-inf"))  # (B, L)

        dist = F.softmax(scaled, dim=-1)  # (B, L)
        # Zero out non-prompt (softmax may give NaN for all-inf rows)
        dist = dist * prompt_mask  # (B, L)
        # Handle all-response samples (no prompt positions)
        dist = torch.nan_to_num(dist, nan=0.0)  # (B, L)
        return dist

    def divergence(
        self,
        saliency_T: torch.Tensor,     # (B, L)
        saliency_S: torch.Tensor,     # (B, L)
        labels_mask: torch.Tensor,    # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:
        """Per-sample JSD between teacher/student saliency distributions.

        Returns:
            jsd: (B,) — Jensen-Shannon divergence per sample.
        """
        p = self.to_distribution(saliency_T, labels_mask, attention_mask)  # (B, L)
        q = self.to_distribution(saliency_S, labels_mask, attention_mask)  # (B, L)
        m = 0.5 * (p + q)  # (B, L)

        # KL(P||M) and KL(Q||M), with log clamping for stability
        def _kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # Only compute where a > 0
            log_ratio = torch.log(a.clamp(min=self.eps)) - torch.log(b.clamp(min=self.eps))
            return (a * log_ratio).sum(dim=-1)  # (B,)

        jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)  # (B,)
        return jsd.clamp(min=0.0)  # (B,)


class SaliencyAlignmentLoss(nn.Module):
    """Cosine distance on raw saliency vectors at prompt positions.

    Input saliency vectors are pre-masked (response/padding = 0).
    This function does NOT apply additional masking.
    """

    def forward(
        self,
        saliency_T: torch.Tensor,     # (B, L)
        saliency_S: torch.Tensor,     # (B, L)
        labels_mask: torch.Tensor,    # (B, L)  — unused, kept for interface consistency
        attention_mask: torch.Tensor,  # (B, L)  — unused, kept for interface consistency
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute cosine alignment loss.

        Returns:
            loss: scalar — mean (1 - cos_sim) over batch.
            stats: dict with "mean_cos_sim".
        """
        # Cosine similarity per sample
        cos_sim = F.cosine_similarity(saliency_T, saliency_S, dim=-1, eps=1e-8)  # (B,)

        loss = (1.0 - cos_sim).mean()  # scalar
        stats = {"mean_cos_sim": cos_sim.mean().item()}
        return loss, stats
