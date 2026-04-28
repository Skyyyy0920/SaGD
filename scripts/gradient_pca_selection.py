"""
Gradient-PCA Data Selection for Knowledge Distillation.

Profiles the per-sample loss gradient structure via random projection + SVD,
then selects a subset that spans the principal gradient subspace.

Theory:
  - Fine-tuning gradients are low-rank (LoRA insight)
  - JL projection preserves inner products in gradient space
  - D-optimal / per-PC-quota selection covers the gradient subspace
  - Coreset guarantee: subset gradient ≈ full-data gradient

Usage:
    # Step 1: Profile gradient structure (using teacher's NLL gradient)
    python scripts/gradient_pca_selection.py profile \
        --teacher_model Qwen/Qwen3-8B \
        --dataset dolly \
        --output_dir outputs/gradient_pca \
        --device cuda:0

    # Step 2: Select subset
    python scripts/gradient_pca_selection.py select \
        --profile_dir outputs/gradient_pca \
        --select_ratio 0.5 \
        --output_path outputs/gradient_pca/selected_indices.pt
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sagd.data import InstructionDataset, SquadDataset, collate_fn
from sagd.models import load_teacher, load_student


# =============================================================================
#  Count Sketch Projection
# =============================================================================

class CountSketchProjector:
    """Project high-dimensional gradient vectors to d dimensions via Count Sketch.

    For each model parameter index j, a hash function maps it to:
      - a projection dimension h(j) in {0, ..., d-1}
      - a sign s(j) in {-1, +1}

    The projection is: proj[k] = sum_{j: h(j)=k} s(j) * grad[j]

    This is O(|theta|) per sample and O(d) memory for the result.
    JL-like guarantee: preserves norms with d = O(1/eps^2).

    Note: hash tables are pre-generated and stored on the specified device.
    For a 600M-param model with proj_dim=1024, this uses ~7GB device memory
    for the hash tables (int64 indices + float32 signs).
    """

    def __init__(self, param_shapes: dict, proj_dim: int = 1024, seed: int = 42,
                 device: str = "cpu"):
        """
        Args:
            param_shapes: {param_name: numel} for all parameters
            proj_dim: projection dimensionality
            seed: random seed for hash functions
            device: device to store hash tables and compute projection
        """
        self.proj_dim = proj_dim
        self.device = device
        self.hash_dims = {}   # param_name → LongTensor of hash bucket indices
        self.hash_signs = {}  # param_name → FloatTensor of signs {-1, +1}

        rng = torch.Generator().manual_seed(seed)
        for name, numel in param_shapes.items():
            self.hash_dims[name] = torch.randint(
                0, proj_dim, (numel,), generator=rng
            ).to(device)
            self.hash_signs[name] = (
                torch.randint(0, 2, (numel,), generator=rng).float() * 2 - 1
            ).to(device)

    def project(self, model: torch.nn.Module) -> torch.Tensor:
        """Project current .grad of all parameters to proj_dim dimensions.

        Must be called AFTER loss.backward().
        Runs on self.device (GPU if available) for speed.
        Returns (proj_dim,) tensor on CPU.
        """
        projected = torch.zeros(self.proj_dim, device=self.device)
        for name, p in model.named_parameters():
            if p.grad is None or name not in self.hash_dims:
                continue
            grad_flat = p.grad.detach().view(-1).float()
            signed_grad = grad_flat * self.hash_signs[name]
            projected.scatter_add_(0, self.hash_dims[name], signed_grad)
        return projected.cpu()


# =============================================================================
#  Per-Sample Gradient Profiling
# =============================================================================

def compute_per_sample_kl(t_logits, s_logits, labels_mask, temperature=2.0):
    """Compute per-sample KL divergence (scalar per sample, NOT batch-averaged).

    Follows the same shift-alignment as trainer._compute_per_sample_kl().
    Returns (B,) tensor.
    """
    t_shifted = t_logits[:, :-1, :]
    s_shifted = s_logits[:, :-1, :]
    mask = labels_mask[:, 1:].float()

    t_probs = F.softmax(t_shifted / temperature, dim=-1)
    t_log = torch.log(t_probs.clamp(min=1e-8))
    s_log = F.log_softmax(s_shifted / temperature, dim=-1)

    per_pos = (t_probs * (t_log - s_log)).sum(dim=-1)  # (B, L-1)
    per_pos = per_pos * mask

    mask_count = mask.sum(dim=-1).clamp(min=1)
    per_sample = per_pos.sum(dim=-1) / mask_count
    return per_sample * temperature ** 2  # (B,)


def compute_teacher_nll(logits, input_ids, labels_mask):
    """Compute teacher's NLL (next-token prediction loss) on response tokens.

    Returns scalar loss for a single sample.
    """
    shifted_logits = logits[:, :-1, :]  # (1, L-1, V)
    shifted_targets = input_ids[:, 1:]   # (1, L-1)
    mask = labels_mask[:, 1:].float()    # (1, L-1)

    log_probs = F.log_softmax(shifted_logits, dim=-1)  # (1, L-1, V)
    token_log_probs = log_probs.gather(
        dim=-1, index=shifted_targets.unsqueeze(-1)
    ).squeeze(-1)  # (1, L-1)

    nll = -(token_log_probs * mask).sum() / mask.sum().clamp(min=1)
    return nll  # scalar


def profile_gradients(teacher, student, dataloader, projector, device,
                      temperature=2.0):
    """Compute projected per-sample TEACHER gradient for all training samples.

    Uses the teacher's NLL gradient to determine data structure —
    the teacher (as the expert) identifies which samples activate
    the most important learning directions.

    Uses batch_size=1 internally for per-sample gradient isolation.

    Returns:
        projected_grads: np.ndarray (n_samples, proj_dim)
        losses: np.ndarray (n_samples,) — per-sample teacher NLL
        indices: np.ndarray (n_samples,) — dataset indices
    """
    teacher.eval()

    all_proj = []
    all_loss = []
    all_idx = []

    for batch in tqdm(dataloader, desc="Gradient profiling (teacher)"):
        input_ids = batch["input_ids"].to(device)       # (1, L)
        attention_mask = batch["attention_mask"].to(device)
        labels_mask = batch["labels_mask"].to(device)
        idx = batch["index"]

        # Teacher forward WITH grad (for gradient profiling)
        teacher.zero_grad()
        t_out = teacher(input_ids=input_ids, attention_mask=attention_mask)
        t_logits = t_out.logits.float()  # (1, L, V)

        # Teacher NLL loss on response tokens
        loss = compute_teacher_nll(t_logits, input_ids, labels_mask)

        # Backward through teacher
        loss.backward()

        # Project teacher gradient via Count Sketch
        proj = projector.project(teacher)  # (proj_dim,)

        all_proj.append(proj.numpy())
        all_loss.append(loss.item())
        all_idx.append(idx.item())

    return (
        np.stack(all_proj, axis=0),    # (n, proj_dim)
        np.array(all_loss),            # (n,)
        np.array(all_idx),             # (n,)
    )


# =============================================================================
#  PCA Analysis and Selection
# =============================================================================

def analyze_and_select(projected_grads, losses, indices, select_ratio=0.5,
                       null_permutations=3):
    """SVD analysis on projected gradients + subset selection.

    Returns:
        analysis: dict with eigenvalue spectrum, effective rank, etc.
        selected_indices: np.ndarray of selected dataset indices
    """
    n, d = projected_grads.shape
    K = int(n * select_ratio)

    # --- Check for degenerate values ---
    nan_count = np.isnan(projected_grads).any(axis=1).sum()
    inf_count = np.isinf(projected_grads).any(axis=1).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"WARNING: {nan_count} NaN rows, {inf_count} Inf rows. Replacing with zeros.")
        bad = np.isnan(projected_grads).any(axis=1) | np.isinf(projected_grads).any(axis=1)
        projected_grads[bad] = 0.0

    # --- Center and SVD ---
    G = projected_grads - projected_grads.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    n_components = len(S)

    eigenvalues = S ** 2
    total_var = eigenvalues.sum()
    if total_var < 1e-12:
        print("ERROR: total gradient variance is near-zero.")
        return {}, indices[:K]

    cumvar = np.cumsum(eigenvalues) / total_var

    print("\n=== Gradient PCA Analysis ===")
    print(f"Matrix shape: ({n}, {d})")
    for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
        r = int(np.searchsorted(cumvar, threshold)) + 1
        print(f"  Effective rank (>{threshold*100:.0f}% var): {r} / {n_components}")

    print(f"\nTop-10 eigenvalues:")
    for k in range(min(10, n_components)):
        print(f"  PC{k+1}: {eigenvalues[k]/total_var*100:.2f}%  "
              f"(cum: {cumvar[k]*100:.2f}%)")

    # --- Null model: random gradients with same norms ---
    print(f"\n=== Null model ({null_permutations} permutations) ===")
    grad_norms = np.linalg.norm(projected_grads, axis=1, keepdims=True)
    null_spectra = []
    for _ in range(null_permutations):
        random_dirs = np.random.randn(n, d)
        random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
        G_null = random_dirs * grad_norms
        G_null -= G_null.mean(axis=0, keepdims=True)
        _, S_null, _ = np.linalg.svd(G_null, full_matrices=False)
        null_spectra.append(S_null ** 2)
    null_eigenvalues = np.mean(null_spectra, axis=0)
    null_total = null_eigenvalues.sum()
    null_cumvar = np.cumsum(null_eigenvalues) / null_total

    for threshold in [0.5, 0.8, 0.9]:
        r_real = int(np.searchsorted(cumvar, threshold)) + 1
        r_null = int(np.searchsorted(null_cumvar, threshold)) + 1
        ratio = r_real / max(r_null, 1)
        flag = " ← LOWER RANK ✓" if ratio < 0.8 else ""
        print(f"  >{threshold*100:.0f}%: real={r_real}, null={r_null}, "
              f"ratio={ratio:.2f}{flag}")

    top10_real = cumvar[min(9, n_components - 1)]
    top10_null = null_cumvar[min(9, len(null_eigenvalues) - 1)]
    print(f"\n  Top-10 PCs: real={top10_real*100:.1f}%, null={top10_null*100:.1f}%")

    # --- Per-PC quota selection ---
    # Determine effective rank (90% variance)
    r_star = int(np.searchsorted(cumvar, 0.9)) + 1
    r_star = min(r_star, n_components, 100)  # cap at 100 PCs
    print(f"\n=== Selection: r*={r_star}, K={K} ({select_ratio*100:.0f}% of {n}) ===")

    # Quota per PC, proportional to eigenvalue
    eig_r = eigenvalues[:r_star]
    quotas = np.maximum(np.round(K * eig_r / eig_r.sum()).astype(int), 1)
    # Adjust total to match K
    while quotas.sum() > K:
        quotas[np.argmax(quotas)] -= 1
    while quotas.sum() < K:
        quotas[np.argmin(quotas)] += 1

    # Sample projections on each PC
    projections = U[:, :r_star] * S[:r_star]  # (n, r_star)

    selected_set = set()
    for k in range(r_star):
        proj_k = np.abs(projections[:, k])
        # Select top samples by projection magnitude, excluding already selected
        ranking = np.argsort(proj_k)[::-1]
        count = 0
        for idx in ranking:
            if idx not in selected_set:
                selected_set.add(idx)
                count += 1
                if count >= quotas[k]:
                    break

    # If we haven't reached K (due to overlap), fill with highest-loss unselected
    if len(selected_set) < K:
        remaining = [i for i in range(n) if i not in selected_set]
        remaining_sorted = sorted(remaining, key=lambda i: losses[i], reverse=True)
        for idx in remaining_sorted:
            selected_set.add(idx)
            if len(selected_set) >= K:
                break

    selected_positions = sorted(selected_set)
    selected_dataset_indices = indices[selected_positions]

    print(f"  Selected {len(selected_dataset_indices)} samples")
    print(f"  Selected avg loss: {losses[selected_positions].mean():.4f} "
          f"(full: {losses.mean():.4f})")

    # --- Compile analysis results ---
    analysis = {
        "n_samples": int(n),
        "proj_dim": int(d),
        "select_ratio": float(select_ratio),
        "K": int(K),
        "r_star": int(r_star),
        "eigenvalues": eigenvalues[:min(100, n_components)].tolist(),
        "cumulative_variance": cumvar[:min(100, n_components)].tolist(),
        "null_eigenvalues": null_eigenvalues[:min(100, len(null_eigenvalues))].tolist(),
        "null_cumulative_variance": null_cumvar[:min(100, len(null_cumvar))].tolist(),
        "effective_rank": {
            str(t): int(np.searchsorted(cumvar, t) + 1)
            for t in [0.5, 0.8, 0.9, 0.95, 0.99]
        },
        "top10_variance_real": float(top10_real),
        "top10_variance_null": float(top10_null),
        "loss_stats": {
            "mean": float(losses.mean()),
            "std": float(losses.std()),
            "selected_mean": float(losses[selected_positions].mean()),
        },
    }

    return analysis, selected_dataset_indices


# =============================================================================
#  CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    # --- profile ---
    p = sub.add_parser("profile", help="Compute projected gradients and run PCA")
    p.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--student_model", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--student_ckpt", type=str, default=None)
    p.add_argument("--dataset", type=str, default="dolly", choices=["dolly", "squad"])
    p.add_argument("--output_dir", type=str, default="outputs/gradient_pca")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--proj_dim", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--select_ratio", type=float, default=0.5,
                   help="Fraction of data to select")

    # --- select (from saved profile) ---
    s = sub.add_parser("select", help="Select subset from saved profile")
    s.add_argument("--profile_dir", type=str, required=True)
    s.add_argument("--select_ratio", type=float, default=0.5)
    s.add_argument("--output_path", type=str, default=None)

    return parser.parse_args()


def cmd_profile(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load models ---
    print(f"Loading teacher: {args.teacher_model}")
    teacher, tokenizer = load_teacher(args.teacher_model, args.device)

    # Enable gradients on teacher attention layers for gradient profiling.
    # We only profile a subset of teacher params to keep memory feasible:
    # attention Q/K/V/O projections capture the main processing patterns.
    profiled_params = {}
    for name, p in teacher.named_parameters():
        # Match attention projection layers (q_proj, k_proj, v_proj, o_proj)
        if any(k in name for k in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            p.requires_grad_(True)
            profiled_params[name] = p.numel()
        # Keep other params frozen (no grad)

    total_profiled = sum(profiled_params.values())
    print(f"Teacher profiled parameters (attention only): {total_profiled:,} "
          f"/ {sum(p.numel() for p in teacher.parameters()):,}")

    # --- Load data ---
    print(f"Loading dataset: {args.dataset}")
    if args.dataset == "dolly":
        dataset = InstructionDataset(
            tokenizer, max_seq_len=args.max_seq_len,
            max_samples=args.max_samples, seed=args.seed, subset="train"
        )
    elif args.dataset == "squad":
        dataset = SquadDataset(
            tokenizer, max_seq_len=args.max_seq_len,
            max_samples=args.max_samples, seed=args.seed, subset="train"
        )
    print(f"Dataset size: {len(dataset)}")

    # batch_size=1 for per-sample gradient isolation
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    # --- Setup projector on teacher's profiled params ---
    print(f"Projection dimension: {args.proj_dim}")

    projector = CountSketchProjector(
        profiled_params, args.proj_dim, seed=args.seed, device=args.device
    )

    # --- Profile using teacher gradients ---
    t0 = time.time()
    projected_grads, losses, indices = profile_gradients(
        teacher, None, dataloader, projector, args.device, args.temperature
    )
    elapsed = time.time() - t0
    print(f"\nProfiling done in {elapsed:.1f}s ({elapsed/len(dataset):.3f}s/sample)")

    # Save raw profile
    np.savez(
        os.path.join(args.output_dir, "gradient_profile.npz"),
        projected_grads=projected_grads,
        losses=losses,
        indices=indices,
    )

    # --- Analyze and select ---
    analysis, selected_indices = analyze_and_select(
        projected_grads, losses, indices,
        select_ratio=args.select_ratio
    )
    analysis["profiling_time_seconds"] = float(elapsed)
    analysis["args"] = vars(args)

    # Save analysis
    with open(os.path.join(args.output_dir, "gradient_pca_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    # Save selected indices
    torch.save(
        torch.tensor(selected_indices, dtype=torch.long),
        os.path.join(args.output_dir, "selected_indices.pt")
    )
    print(f"\nSelected {len(selected_indices)} indices saved to "
          f"{os.path.join(args.output_dir, 'selected_indices.pt')}")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        eigenvalues = np.array(analysis["eigenvalues"])
        cumvar = np.array(analysis["cumulative_variance"])
        null_eigenvalues = np.array(analysis["null_eigenvalues"])
        null_cumvar = np.array(analysis["null_cumulative_variance"])
        total_var = eigenvalues.sum()
        null_total = null_eigenvalues.sum()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        n_plot = min(50, len(eigenvalues))
        k_range = np.arange(1, n_plot + 1)

        ax = axes[0]
        ax.semilogy(k_range, eigenvalues[:n_plot] / total_var, "b-o",
                     markersize=3, label="Real")
        ax.semilogy(k_range, null_eigenvalues[:n_plot] / null_total, "r--x",
                     markersize=3, label="Null (random dir.)")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Fraction of Variance (log)")
        ax.set_title("(a) Gradient Eigenvalue Spectrum")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(k_range, cumvar[:n_plot] * 100, "b-o", markersize=3, label="Real")
        ax.plot(k_range, null_cumvar[:n_plot] * 100, "r--x", markersize=3,
                label="Null")
        ax.axhline(y=90, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(y=80, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Variance (%)")
        ax.set_title("(b) Cumulative Explained Variance")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # PC1 vs PC2, colored by loss
        G = projected_grads - projected_grads.mean(axis=0, keepdims=True)
        U_plot, S_plot, _ = np.linalg.svd(G, full_matrices=False)
        proj_plot = U_plot[:, :2] * S_plot[:2]

        ax = axes[2]
        sc = ax.scatter(proj_plot[:, 0], proj_plot[:, 1],
                        c=losses, cmap="viridis", s=5, alpha=0.6)
        plt.colorbar(sc, ax=ax, label="KL Loss")
        ax.set_xlabel(f"PC1 ({eigenvalues[0]/total_var*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({eigenvalues[1]/total_var*100:.1f}%)")
        ax.set_title("(c) Samples in Gradient PC Space")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(args.output_dir, "gradient_pca_analysis.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {fig_path}")
        plt.close()
    except ImportError:
        print("matplotlib not available, skipping plot")


def cmd_select(args):
    """Re-run selection with a different ratio from saved profile."""
    data = np.load(os.path.join(args.profile_dir, "gradient_profile.npz"))
    projected_grads = data["projected_grads"]
    losses = data["losses"]
    indices = data["indices"]

    analysis, selected_indices = analyze_and_select(
        projected_grads, losses, indices,
        select_ratio=args.select_ratio
    )

    output_path = args.output_path or os.path.join(
        args.profile_dir, f"selected_indices_{int(args.select_ratio*100)}pct.pt"
    )
    torch.save(torch.tensor(selected_indices, dtype=torch.long), output_path)
    print(f"Selected {len(selected_indices)} indices saved to {output_path}")

    with open(output_path.replace(".pt", "_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)


def main():
    args = parse_args()
    if args.command == "profile":
        cmd_profile(args)
    elif args.command == "select":
        cmd_select(args)
    else:
        print("Usage: python gradient_pca_selection.py {profile|select} ...")


if __name__ == "__main__":
    main()
