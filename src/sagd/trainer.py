"""Unified trainer for SaGD and baselines.

Supports three methods: standard_kd, reverse_kl, sagd.
See CLAUDE.md §2.5 for the complete training flow pseudocode.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sagd.data import InstructionDataset, collate_fn
from sagd.losses import ReverseKLLoss, StandardKDLoss
from sagd.saliency import SaliencyAlignmentLoss, SaliencyComputer

METHODS = {"standard_kd", "reverse_kl", "sagd"}


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
        self.batch_size = config.get("batch_size", 8)
        self.grad_accum = config.get("gradient_accumulation", 4)
        self.lr = config.get("lr", 2e-5)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.warmup_ratio = config.get("warmup_ratio", 0.03)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.temperature = config.get("temperature", 2.0)
        self.fp16 = config.get("fp16", True)
        self.log_every = config.get("log_every", 50)
        self.save_every_n_epochs = config.get("save_every_n_epochs", 1)

        # Loss functions
        if method == "reverse_kl":
            self.kl_loss_fn = ReverseKLLoss(temperature=self.temperature)
        else:
            self.kl_loss_fn = StandardKDLoss(temperature=self.temperature)

        # SaGD components — only initialized when method == "sagd"
        self.saliency_computer: SaliencyComputer | None = None
        self.sal_align_loss: SaliencyAlignmentLoss | None = None
        self.teacher_saliency_cache: list[torch.Tensor] | None = None

        if method == "sagd":
            sal_temp = config.get("saliency_temperature", 2.0)
            self.saliency_computer = SaliencyComputer(temperature=sal_temp)
            self.sal_align_loss = SaliencyAlignmentLoss()
            self.lambda_sal = config.get("lambda_sal", 0.5)
            self.sagd_every_n = config.get("sagd_every_n_steps", 5)
            self.sagd_tau_w = config.get("sagd_tau_w", 1.0)

            # Load teacher saliency cache
            cache_path = config.get("teacher_saliency_path")
            assert cache_path is not None, "sagd requires teacher_saliency_path"
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.teacher_saliency_cache = cache["saliency"]

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
            self.student.train()
            epoch_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")

            for step, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(self.device)          # (B, L)
                attention_mask = batch["attention_mask"].to(self.device)  # (B, L)
                labels_mask = batch["labels_mask"].to(self.device)       # (B, L)
                indices = batch["index"]                                 # (B,)

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

                    if self.method == "sagd" and global_step % self.sagd_every_n == 0:
                        # SaGD step
                        per_sample_kl = self._compute_per_sample_kl(
                            t_logits, s_logits, labels_mask,
                        )  # (B,)

                        # NON-differentiable student saliency — for reweighting only
                        with torch.no_grad():
                            student_sal_for_reweight = self.saliency_computer.compute(
                                self.student, input_ids, attention_mask, labels_mask,
                            )  # (B, L), detached

                        teacher_sal = self._get_cached_teacher_saliency(
                            indices, input_ids.size(1), input_ids.device,
                        )  # (B, L)

                        # Reweighting: uses NON-differentiable saliency
                        jsd = self.saliency_computer.divergence(
                            teacher_sal, student_sal_for_reweight, labels_mask, attention_mask,
                        )  # (B,)
                        weights = F.softmax(jsd / self.sagd_tau_w, dim=0) * jsd.size(0)  # (B,)

                        # DIFFERENTIABLE student saliency — for alignment loss
                        with torch.amp.autocast("cuda", enabled=False):
                            student_sal_for_loss = self.saliency_computer.compute_differentiable(
                                self.student, input_ids, attention_mask, labels_mask,
                            )  # (B, L), differentiable!

                        sal_loss, sal_stats = self.sal_align_loss(
                            teacher_sal, student_sal_for_loss, labels_mask, attention_mask,
                        )

                        loss = (weights.detach() * per_sample_kl).mean() + \
                               self.lambda_sal * sal_loss

                        step_stats.update({
                            "sagd/sal_loss": sal_loss.item(),
                            "sagd/mean_jsd": jsd.mean().item(),
                            "sagd/max_weight": weights.max().item(),
                            "sagd/min_weight": weights.min().item(),
                            "sagd/mean_cos_sim": sal_stats["mean_cos_sim"],
                        })
                    else:
                        # Standard or non-SaGD-step
                        loss = self.kl_loss_fn(t_logits, s_logits, labels_mask)

                # Gradient accumulation
                loss_scaled = loss / self.grad_accum
                scaler.scale(loss_scaled).backward()

                if (step + 1) % self.grad_accum == 0 or (step + 1) == len(dataloader):
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
            avg_loss = epoch_loss / max(len(dataloader), 1)
            print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

            if (epoch + 1) % self.save_every_n_epochs == 0:
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
