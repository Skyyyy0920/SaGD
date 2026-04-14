"""Unified trainer for SaGD and all baselines.

Supports 8 methods: sft, standard_kd, reverse_kl, seqkd, gkd, distillm, dakd, sagd.
See CLAUDE.md §2.5 for the SaGD training flow pseudocode.

SaGD uses noise KL for implicit Jacobian matching (Srinivas & Fleuret 2018):
  E[KL(f_T(x+ξ) || f_S(x+ξ))] = KL(f_T(x) || f_S(x)) + σ² ||J_T - J_S||²_F + O(σ⁴)
Combined with saliency-guided sample reweighting (DRO).
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from sagd.data import InstructionDataset, collate_fn
from sagd.losses import (
    BDLLoss,
    JSDLoss,
    ReverseKLLoss,
    SFTLoss,
    SkewKLLoss,
    StandardKDLoss,
)
from sagd.saliency import SaliencyComputer

METHODS = {
    "standard_kd",    # Forward KL (Hinton, 2015)
    "reverse_kl",     # Reverse KL / MiniLLM (Gu et al., 2024)
    "sft",            # Supervised fine-tuning (no teacher)
    "seqkd",          # Sequence-level KD (Kim & Rush, 2016) — SFT on teacher outputs
    "gkd",            # Generalized KD with JSD (Agarwal et al., 2023)
    "distillm",       # DistiLLM with Skew KL (Ko et al., 2024)
    "dakd",           # DA-KD with BDL (He et al., 2025)
    "sagd",           # Our method
}


class Trainer:
    """Knowledge distillation trainer.

    Args:
        teacher: Teacher model (frozen, eval mode).
        student: Student model (train mode).
        tokenizer: HuggingFace tokenizer.
        dataset: Training dataset.
        config: Dict with all hyperparameters.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        tokenizer: Any,
        dataset: InstructionDataset,
        config: dict[str, Any],
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.config = config

        method = config.get("method", "standard_kd")
        assert method in METHODS, f"Unknown method: {method}. Must be one of {METHODS}"
        self.method = method

        self.device = config.get("device", "cuda:0")
        self.epochs = config.get("epochs", 3)
        self.batch_size = config.get("batch_size", 4)
        self.grad_accum = config.get("gradient_accumulation", 4)

        # SaGD does extra forward passes (noise KL on both teacher + student),
        # so peak memory ~2× standard KD. Auto-halve batch_size and double
        # grad_accum to keep effective batch unchanged.
        if method == "sagd" and self.batch_size > 1:
            new_bs = max(1, self.batch_size // 2)
            new_ga = self.grad_accum * (self.batch_size // new_bs)
            print(
                f"[SaGD] auto-reducing batch_size {self.batch_size} -> {new_bs} "
                f"and increasing grad_accum {self.grad_accum} -> {new_ga} "
                f"(effective batch unchanged)"
            )
            self.batch_size = new_bs
            self.grad_accum = new_ga
        self.lr = config.get("lr", 2e-5)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.warmup_ratio = config.get("warmup_ratio", 0.03)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.temperature = config.get("temperature", 2.0)
        self.fp16 = config.get("fp16", True)
        self.log_every = config.get("log_every", 50)
        # save_every_n_epochs <= 0 means: only save the final checkpoint
        # (avoids filling disk with 10x intermediate checkpoints across 80 runs).
        self.save_every_n_epochs = config.get("save_every_n_epochs", 0)

        # Loss functions
        if method == "reverse_kl":
            self.kl_loss_fn = ReverseKLLoss(temperature=self.temperature)
        elif method == "sft" or method == "seqkd":
            self.sft_loss_fn = SFTLoss()
            self.kl_loss_fn = None  # not used
        elif method == "gkd":
            self.kl_loss_fn = JSDLoss(
                temperature=self.temperature,
                beta=config.get("gkd_beta", 0.5),
            )
            # On-policy probability: fraction of steps using student-generated outputs
            self.gkd_on_policy_prob = config.get("gkd_on_policy_prob", 0.0)
        elif method == "distillm":
            self.kl_loss_fn = SkewKLLoss(
                temperature=self.temperature,
                alpha=config.get("distillm_alpha", 0.5),
            )
        elif method == "dakd":
            self.kl_loss_fn = BDLLoss(
                temperature=self.temperature,
                bdl_lambda=config.get("bdl_lambda", 0.9),
            )
        else:
            self.kl_loss_fn = StandardKDLoss(temperature=self.temperature)

        # SaGD components — only initialized when method == "sagd"
        self.saliency_computer: SaliencyComputer | None = None
        self.teacher_saliency_cache: list[torch.Tensor] | None = None

        # DA-KD components — DiffUp strategy
        if method == "dakd":
            self.dakd_tau = config.get("dakd_tau", 0.1)  # stratified mixing

        if method == "sagd":
            sal_temp = config.get("saliency_temperature", 2.0)
            self.saliency_computer = SaliencyComputer(temperature=sal_temp)
            self.lambda_noise = config.get("lambda_noise", 0.5)
            self.noise_sigma = config.get("noise_sigma", 0.01)
            self.sagd_every_n = config.get("sagd_every_n_steps", 5)
            self.sagd_tau_w = config.get("sagd_tau_w", 1.0)

            # Load teacher saliency cache
            cache_path = config.get("teacher_saliency_path")
            assert cache_path is not None, "sagd requires teacher_saliency_path"
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.teacher_saliency_cache = cache["saliency"]

    def _compute_dakd_subset(
        self,
        epoch: int,
    ) -> Subset | None:
        """DA-KD DiffUp: compute DDS scores and select a subset for this epoch.

        DDS(x) = L_student(x) / L_teacher(x), where L is cross-entropy loss.
        Selection ratio decays with cosine schedule: r = 0.5*(cos(πe/E) + 1).

        Returns:
            Subset of self.dataset, or None if epoch==0 (use full dataset).
        """
        if epoch == 0:
            return None  # first epoch uses full dataset

        E = self.epochs
        r = 0.5 * (math.cos(math.pi * epoch / E) + 1)  # cosine decay
        r = max(r, 0.1)  # minimum 10% of data

        # Use a non-shuffled dataloader for deterministic index tracking
        score_loader = DataLoader(
            self.dataset, batch_size=self.batch_size,
            shuffle=False, collate_fn=collate_fn, drop_last=False,
        )

        # Compute per-sample cross-entropy for teacher and student
        self.student.eval()
        all_dds: list[tuple[int, float]] = []  # (dataset_index, dds_score)

        with torch.no_grad():
            for batch in tqdm(score_loader, desc=f"DDS scoring (epoch {epoch+1})", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels_mask = batch["labels_mask"].to(self.device)
                indices = batch["index"]  # (B,) — dataset indices

                t_out = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
                s_out = self.student(input_ids=input_ids, attention_mask=attention_mask)

                # Per-sample cross-entropy on response tokens
                for logits, tag in [(t_out.logits.float(), "t"),
                                    (s_out.logits.float(), "s")]:
                    shift_logits = logits[:, :-1, :]
                    shift_labels = input_ids[:, 1:]
                    mask = labels_mask[:, 1:].float()

                    log_probs = F.log_softmax(shift_logits, dim=-1)
                    token_nll = -log_probs.gather(
                        dim=-1, index=shift_labels.unsqueeze(-1)
                    ).squeeze(-1)  # (B, L-1)

                    per_sample = (token_nll * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
                    if tag == "t":
                        t_losses = per_sample.cpu()
                    else:
                        s_losses = per_sample.cpu()

                # DDS = student_loss / teacher_loss
                dds = s_losses / t_losses.clamp(min=1e-6)
                for idx_val, dds_val in zip(indices.tolist(), dds.tolist()):
                    all_dds.append((idx_val, dds_val))

        self.student.train()

        # Sort by DDS descending
        all_dds.sort(key=lambda x: x[1], reverse=True)
        n_total = len(all_dds)
        n_select = max(int(n_total * r), 1)

        # Stratified sampling: split into high-DDS and low-DDS partitions
        high_items = all_dds[:n_select]
        low_items = all_dds[n_select:]

        tau = self.dakd_tau
        n_from_high = max(int((1 - tau) * len(high_items)), 1)
        n_from_low = max(int(tau * len(high_items)), 0)

        rng = random.Random(epoch)  # deterministic per epoch
        selected_high = rng.sample(high_items, min(n_from_high, len(high_items)))
        selected_low = rng.sample(low_items, min(n_from_low, len(low_items))) if low_items else []

        # Extract dataset indices
        selected_indices = [item[0] for item in selected_high + selected_low]
        return Subset(self.dataset, selected_indices)

    def _compute_per_sample_kl(
        self,
        t_logits: torch.Tensor,   # (B, L, V)
        s_logits: torch.Tensor,   # (B, L, V)
        labels_mask: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:
        """Per-sample KL divergence. See CLAUDE.md §2.4.

        Returns:
            per_sample_kl: (B,) — scaled by T².
        """
        # Shift alignment
        t_shifted = t_logits[:, :-1, :]  # (B, L-1, V)
        s_shifted = s_logits[:, :-1, :]  # (B, L-1, V)
        mask = labels_mask[:, 1:].float()  # (B, L-1)

        t_probs = F.softmax(t_shifted / self.temperature, dim=-1)  # (B, L-1, V)
        t_log = torch.log(t_probs.clamp(min=1e-8))  # (B, L-1, V)
        s_log = F.log_softmax(s_shifted / self.temperature, dim=-1)  # (B, L-1, V)

        per_pos = (t_probs * (t_log - s_log)).sum(dim=-1)  # (B, L-1)
        per_pos = per_pos * mask  # (B, L-1)

        mask_count = mask.sum(dim=-1).clamp(min=1)  # (B,)
        per_sample = per_pos.sum(dim=-1) / mask_count  # (B,)
        return per_sample * self.temperature ** 2  # (B,)

    def _get_cached_teacher_saliency(
        self,
        indices: torch.Tensor,  # (B,)
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Retrieve teacher saliency from cache, pad/trim to seq_len.

        Returns:
            saliency: (B, seq_len)
        """
        batch_sal = []
        for idx in indices.tolist():
            sal = self.teacher_saliency_cache[idx]  # (L_i,)
            if sal.size(0) >= seq_len:
                sal = sal[:seq_len]
            else:
                pad = torch.zeros(seq_len - sal.size(0))
                sal = torch.cat([sal, pad])
            batch_sal.append(sal)
        return torch.stack(batch_sal).to(device)  # (B, seq_len)

    def _compute_adaptive_sigma(
        self,
        teacher_sal: torch.Tensor,     # (B, L)
        student_sal: torch.Tensor,     # (B, L)
        embed_norm: float,             # mean embedding norm for this batch
    ) -> torch.Tensor:
        """Compute per-position noise scale from saliency difference.

        noise_sigma is interpreted as a FRACTION of embedding norm (not absolute).
        σ_j = σ_base * embed_norm * clamp(sal_diff_j / mean(sal_diff), max=5)

        Returns:
            sigma_per_pos: (B, L)
        """
        sal_diff = (teacher_sal - student_sal).abs()  # (B, L)
        sal_diff = sal_diff.clamp(min=1e-6)
        sal_diff_mean = sal_diff.mean(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, 1)
        ratio = (sal_diff / sal_diff_mean).clamp(max=5.0)  # cap at 5× mean
        # noise_sigma is relative to embed_norm (e.g., 0.01 = 1% of embedding)
        return self.noise_sigma * embed_norm * ratio  # (B, L)

    def _compute_noisy_kl(
        self,
        input_ids: torch.Tensor,      # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
        labels_mask: torch.Tensor,     # (B, L)
        teacher_sal: torch.Tensor,     # (B, L) for adaptive noise
        student_sal: torch.Tensor,     # (B, L) for adaptive noise
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute per-sample KL on noise-perturbed embeddings.

        Both models see the "same" perturbation: shared per-position noise
        scale σ_j (from saliency difference), applied to each model's own
        embeddings. For cross-architecture (different d_t, d_s), the noise
        is generated independently per dimension but with the same σ_j scale,
        preserving the Jacobian matching interpretation.

        Returns:
            per_sample_kl_noisy: (B,) — per-sample KL on noisy input.
            stats: dict with noise diagnostics.
        """
        # Get embeddings first to compute norm for relative noise scaling
        with torch.no_grad():
            t_embed = self.teacher.get_input_embeddings()(input_ids)  # (B, L, d_t)
            s_embed = self.student.get_input_embeddings()(input_ids)  # (B, L, d_s)
            s_embed_norm = s_embed.norm(dim=-1).mean().item()

        # Shared per-position noise scale (relative to embedding norm)
        sigma_per_pos = self._compute_adaptive_sigma(
            teacher_sal, student_sal, s_embed_norm,
        )  # (B, L)

        with torch.no_grad():
            t_noisy_embed = t_embed + torch.randn_like(t_embed) * sigma_per_pos.unsqueeze(-1)

        s_noisy_embed = s_embed + torch.randn_like(s_embed) * sigma_per_pos.unsqueeze(-1)

        # Teacher forward on noisy input (frozen, no_grad)
        with torch.no_grad():
            t_out_noisy = self.teacher(
                inputs_embeds=t_noisy_embed, attention_mask=attention_mask,
            )
            t_logits_noisy = t_out_noisy.logits.float()  # (B, L, V)

        # Student forward on noisy input (differentiable through model layers)
        s_out_noisy = self.student(
            inputs_embeds=s_noisy_embed, attention_mask=attention_mask,
        )
        s_logits_noisy = s_out_noisy.logits.float()  # (B, L, V)

        # Per-sample KL on noisy logits
        per_sample_kl_noisy = self._compute_per_sample_kl(
            t_logits_noisy, s_logits_noisy, labels_mask,
        )  # (B,)

        # NaN/Inf guard: if ANY sample is NaN/Inf, detach the entire noisy KL
        # to prevent corrupted gradients from flowing into model parameters.
        # masked_fill alone is NOT safe: 0 * NaN_gradient = NaN in autograd.
        if torch.isnan(per_sample_kl_noisy).any() or torch.isinf(per_sample_kl_noisy).any():
            per_sample_kl_noisy = torch.zeros_like(per_sample_kl_noisy)  # fully detached, no graph

        stats = {
            "embed_norm": s_embed_norm,
            "noise_ratio": self.noise_sigma,  # relative to embed norm
        }

        return per_sample_kl_noisy, stats

    def train(self, save_dir: str) -> dict[str, list[float]]:
        """Run training loop.

        Args:
            save_dir: Directory to save checkpoints and logs.

        Returns:
            history: Dict of metric lists.
        """
        os.makedirs(save_dir, exist_ok=True)

        dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size,
            shuffle=True, collate_fn=collate_fn, drop_last=True,
        )

        optimizer = torch.optim.AdamW(
            self.student.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )

        total_steps = len(dataloader) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8 / self.lr if self.lr > 0 else 1.0,
            end_factor=1.0, total_iters=max(warmup_steps, 1),
        )

        scaler = torch.amp.GradScaler("cuda", enabled=self.fp16)
        stats_path = Path(save_dir) / "training_stats.jsonl"
        history: dict[str, list[float]] = {"loss": []}
        global_step = 0

        for epoch in range(self.epochs):
            # DA-KD DiffUp: select data subset for this epoch
            if self.method == "dakd" and epoch > 0:
                subset = self._compute_dakd_subset(epoch)
                if subset is not None:
                    epoch_dataloader = DataLoader(
                        subset, batch_size=self.batch_size,
                        shuffle=True, collate_fn=collate_fn, drop_last=True,
                    )
                else:
                    epoch_dataloader = dataloader
            else:
                epoch_dataloader = dataloader

            self.student.train()
            epoch_loss = 0.0
            pbar = tqdm(epoch_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")

            for step, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(self.device)          # (B, L)
                attention_mask = batch["attention_mask"].to(self.device)  # (B, L)
                labels_mask = batch["labels_mask"].to(self.device)       # (B, L)
                indices = batch["index"]                                 # (B,)

                # SFT doesn't need teacher forward
                if self.method == "sft":
                    t_logits = None
                else:
                    # Teacher forward (frozen, no_grad)
                    with torch.no_grad():
                        t_out = self.teacher(
                            input_ids=input_ids, attention_mask=attention_mask,
                        )
                        t_logits = t_out.logits.float()  # (B, L, V)

                # Student forward
                with torch.amp.autocast("cuda", enabled=self.fp16):
                    s_out = self.student(
                        input_ids=input_ids, attention_mask=attention_mask,
                    )
                    s_logits = s_out.logits.float()  # (B, L, V)

                    step_stats: dict[str, Any] = {"step": global_step, "epoch": epoch}

                    if self.method == "sft":
                        # SFT: cross-entropy on ground truth labels
                        loss = self.sft_loss_fn(s_logits, input_ids, labels_mask)

                    elif self.method == "seqkd":
                        # SeqKD (Kim & Rush, 2016): SFT on teacher's argmax tokens.
                        # Replace target tokens with teacher's greedy predictions
                        # at response positions (prompt tokens stay as ground truth).
                        teacher_tokens = t_logits.argmax(dim=-1)  # (B, L)
                        seqkd_targets = torch.where(
                            labels_mask.bool(), teacher_tokens, input_ids,
                        )  # (B, L)
                        loss = self.sft_loss_fn(s_logits, seqkd_targets, labels_mask)

                    elif self.method == "sagd" and global_step % self.sagd_every_n == 0:
                        # === SaGD step ===

                        # 1. Per-sample KL on clean input
                        per_sample_kl = self._compute_per_sample_kl(
                            t_logits, s_logits, labels_mask,
                        )  # (B,)

                        # 2. Saliency (needed for both adaptive noise and reweighting)
                        with torch.no_grad():
                            student_sal = self.saliency_computer.compute(
                                self.student, input_ids, attention_mask, labels_mask,
                            )  # (B, L), detached

                        teacher_sal = self._get_cached_teacher_saliency(
                            indices, input_ids.size(1), input_ids.device,
                        )  # (B, L)

                        # 3. Noise KL with position-adaptive noise
                        per_sample_kl_noisy, noise_stats = self._compute_noisy_kl(
                            input_ids, attention_mask, labels_mask,
                            teacher_sal, student_sal,
                        )  # (B,)

                        # 4. Saliency-guided reweighting (DRO)
                        jsd = self.saliency_computer.divergence(
                            teacher_sal, student_sal, labels_mask, attention_mask,
                        )  # (B,)
                        weights = F.softmax(jsd / self.sagd_tau_w, dim=0) * jsd.size(0)  # (B,)

                        # 5. Combined loss: weighted (clean KL + λ * noisy KL)
                        loss = (weights.detach() * (
                            per_sample_kl + self.lambda_noise * per_sample_kl_noisy
                        )).mean()

                        step_stats.update({
                            "sagd/kl_noisy": per_sample_kl_noisy.mean().item(),
                            "sagd/kl_clean": per_sample_kl.mean().item(),
                            "sagd/mean_jsd": jsd.mean().item(),
                            "sagd/max_weight": weights.max().item(),
                            "sagd/min_weight": weights.min().item(),
                            "sagd/embed_norm": noise_stats["embed_norm"],
                            "sagd/noise_ratio": noise_stats["noise_ratio"],
                        })
                    elif self.method == "gkd" and self.gkd_on_policy_prob > 0 and torch.rand(1).item() < self.gkd_on_policy_prob:
                        # GKD on-policy: generate student outputs, then compute JSD
                        # on student-generated sequences (both teacher and student
                        # do a forward pass on the student's generated tokens).
                        with torch.no_grad():
                            prompt_lens = (labels_mask == 0).sum(dim=-1)  # (B,)
                            B_size = input_ids.size(0)
                            L_orig = input_ids.size(1)
                            pad_id = self.tokenizer.pad_token_id or 0
                            max_new = L_orig - prompt_lens.min().item()

                            # Left-pad prompts for batch generation (each sample
                            # has different prompt length)
                            max_pl = prompt_lens.max().item()
                            gen_input = torch.full(
                                (B_size, max_pl), pad_id,
                                dtype=torch.long, device=self.device,
                            )
                            gen_attn = torch.zeros(
                                (B_size, max_pl),
                                dtype=torch.long, device=self.device,
                            )
                            for bi in range(B_size):
                                pl = prompt_lens[bi].item()
                                gen_input[bi, max_pl - pl:] = input_ids[bi, :pl]
                                gen_attn[bi, max_pl - pl:] = 1

                            # Suppress <think> token for Qwen3 (prevents
                            # thinking-mode traces in on-policy generation)
                            _think_id = self.tokenizer.convert_tokens_to_ids("<think>")
                            _suppress = [_think_id] if _think_id is not None and _think_id != self.tokenizer.unk_token_id else None

                            gen_kwargs = dict(
                                input_ids=gen_input,
                                attention_mask=gen_attn,
                                max_new_tokens=max(max_new, 1),
                                do_sample=True,
                                temperature=1.0,
                                pad_token_id=pad_id,
                            )
                            if _suppress is not None:
                                gen_kwargs["suppress_tokens"] = _suppress

                            gen_out = self.student.generate(**gen_kwargs)  # (B, max_pl + generated)

                            # Trim and build attention mask from non-pad tokens
                            gen_out = gen_out[:, :max_pl + max(max_new, 1)]
                            gen_mask = (gen_out != pad_id).long()

                            # Labels mask: tokens after each sample's prompt are response
                            gen_labels = torch.zeros_like(gen_out)
                            for bi in range(B_size):
                                # prompt starts at max_pl - prompt_lens[bi]
                                gen_labels[bi, max_pl:] = 1

                            # Teacher forward on student-generated tokens
                            t_out_gen = self.teacher(input_ids=gen_out, attention_mask=gen_mask)
                            t_logits_gen = t_out_gen.logits.float()

                        # Student forward on its own generated tokens (differentiable)
                        s_out_gen = self.student(input_ids=gen_out, attention_mask=gen_mask)
                        s_logits_gen = s_out_gen.logits.float()
                        loss = self.kl_loss_fn(t_logits_gen, s_logits_gen, gen_labels)

                    else:
                        # Standard KD, Reverse KL, GKD (off-policy), DistiLLM, DA-KD, or non-SaGD-step
                        loss = self.kl_loss_fn(t_logits, s_logits, labels_mask)

                # Gradient accumulation
                loss_scaled = loss / self.grad_accum
                scaler.scale(loss_scaled).backward()

                if (step + 1) % self.grad_accum == 0 or (step + 1) == len(epoch_dataloader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(), self.max_grad_norm,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                step_stats["loss"] = loss_val
                history["loss"].append(loss_val)

                # Logging
                if global_step % self.log_every == 0:
                    pbar.set_postfix(loss=f"{loss_val:.4f}")
                    with open(stats_path, "a") as f:
                        f.write(json.dumps(step_stats) + "\n")

                global_step += 1

            # End of epoch
            avg_loss = epoch_loss / max(len(epoch_dataloader), 1)
            print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

            # Only save intermediate checkpoints if explicitly requested
            # (save_every_n_epochs > 0). Default 0 → only the final
            # checkpoint is written at end of training.
            if self.save_every_n_epochs > 0 and (epoch + 1) % self.save_every_n_epochs == 0:
                ckpt_path = Path(save_dir) / f"student_epoch{epoch+1}.pt"
                torch.save(self.student.state_dict(), ckpt_path)

        # Save final checkpoint
        final_path = Path(save_dir) / "student_final.pt"
        torch.save(self.student.state_dict(), final_path)
        return history

    def evaluate(self) -> dict[str, float]:
        """Evaluate student model with ROUGE-L."""
        from sagd.evaluation import evaluate_rouge

        return evaluate_rouge(
            self.student, self.tokenizer, self.dataset,
            device=self.device,
        )
