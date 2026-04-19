"""
Verification experiment: PCA on saliency-weighted embedding differences.

Tests whether teacher-student saliency misalignment has low-rank structure.

Usage:
    python scripts/verify_saliency_pca.py \
        --student_model Qwen/Qwen3-0.6B \
        --student_ckpt outputs_dolly/qwen3_0.6B/sagd/seed_42/student_final.pt \
        --teacher_saliency_path data/teacher_saliency_dolly.pt \
        --dataset dolly \
        --output_dir outputs/saliency_pca_analysis \
        --device cuda:0

If no --student_ckpt is provided, uses the pretrained student (before distillation)
to analyze the INITIAL saliency error structure.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sagd.data import InstructionDataset, SquadDataset, collate_fn
from sagd.saliency import SaliencyComputer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--student_ckpt", type=str, default=None,
                        help="Path to student checkpoint. If None, uses pretrained.")
    parser.add_argument("--teacher_saliency_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="dolly", choices=["dolly", "squad"])
    parser.add_argument("--output_dir", type=str, default="outputs/saliency_pca_analysis")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for quick testing)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--saliency_temperature", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true",
                        help="Load model in float16 to save GPU memory")
    parser.add_argument("--null_permutations", type=int, default=3,
                        help="Number of random permutations for null model")
    return parser.parse_args()


def load_teacher_saliency_cache(path):
    """Load precomputed teacher saliency cache."""
    cache = torch.load(path, map_location="cpu", weights_only=False)
    print(f"Loaded teacher saliency: {cache['metadata']}")
    return cache["saliency"], cache["metadata"]


def get_cached_teacher_saliency(cache, indices, seq_len):
    """Retrieve and pad/trim teacher saliency for given indices."""
    batch = []
    for idx in indices:
        idx_val = idx.item() if isinstance(idx, torch.Tensor) else int(idx)
        sal = cache[idx_val]
        if sal.size(0) >= seq_len:
            batch.append(sal[:seq_len])
        else:
            batch.append(F.pad(sal, (0, seq_len - sal.size(0))))
    return torch.stack(batch)


def saliency_to_distribution(saliency, labels_mask, attention_mask, temperature=2.0):
    """Convert raw saliency to normalized distribution over prompt positions."""
    prompt_mask = (1 - labels_mask).float() * attention_mask.float()
    masked = saliency / temperature
    masked = masked.masked_fill(prompt_mask == 0, float("-inf"))
    dist = F.softmax(masked, dim=-1)
    dist = dist * prompt_mask
    # Handle all-response samples (no prompt positions → NaN from softmax)
    dist = torch.nan_to_num(dist, nan=0.0)
    return dist


def compute_delta_z(student_model, saliency_computer, teacher_cache,
                    dataloader, saliency_temperature, device):
    """
    Compute saliency-weighted embedding difference for all samples.

    For each sample i:
        delta_z_i = sum_j (s_T_hat_j - s_S_hat_j) * e_j
    where s_T_hat, s_S_hat are normalized saliency distributions and e_j
    are student embeddings.

    Returns:
        delta_z: np.ndarray (n_samples, d) — saliency error vectors
        jsd_values: np.ndarray (n_samples,) — per-sample JSD
        norms: np.ndarray (n_samples,) — ||delta_z|| per sample
        indices: np.ndarray (n_samples,) — dataset indices
    """
    student_model.eval()
    embed_layer = student_model.get_input_embeddings()

    all_delta_z = []
    all_jsd = []
    all_norms = []
    all_indices = []

    for batch in tqdm(dataloader, desc="Computing delta_z"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_mask = batch["labels_mask"].to(device)
        batch_indices = batch["index"]
        B, L = input_ids.shape

        # 1. Student saliency (non-differentiable)
        # Note: compute() uses @torch.enable_grad() internally,
        # so it works correctly even though model params are frozen.
        student_sal = saliency_computer.compute(
            student_model, input_ids, attention_mask, labels_mask
        ).float()  # (B, L), detached, ensure float32 even if model is fp16

        # 2. Teacher saliency (from cache, already float32)
        teacher_sal = get_cached_teacher_saliency(
            teacher_cache, batch_indices, L
        ).to(device)  # (B, L), float32

        # 3. Normalized saliency distributions
        s_T_dist = saliency_to_distribution(
            teacher_sal, labels_mask, attention_mask, saliency_temperature
        )  # (B, L)
        s_S_dist = saliency_to_distribution(
            student_sal, labels_mask, attention_mask, saliency_temperature
        )  # (B, L)

        # 4. Saliency difference weights (sum to ~0 per sample since both are distributions)
        delta_s = s_T_dist - s_S_dist  # (B, L)

        # 5. Student embeddings (detached, cast to float32 for precision)
        with torch.no_grad():
            embeds = embed_layer(input_ids).float()  # (B, L, d)

        # 6. Saliency-weighted embedding difference: delta_z_i = sum_j delta_s_j * e_j
        delta_z = torch.bmm(
            delta_s.unsqueeze(1),  # (B, 1, L)
            embeds                  # (B, L, d)
        ).squeeze(1)  # (B, d)

        # 7. Per-sample JSD (reuses SaliencyComputer.divergence for consistency)
        jsd = saliency_computer.divergence(
            teacher_sal, student_sal, labels_mask, attention_mask
        )  # (B,)

        all_delta_z.append(delta_z.cpu().numpy())
        all_jsd.append(jsd.cpu().numpy())
        all_norms.append(delta_z.norm(dim=-1).cpu().numpy())
        all_indices.append(batch_indices.numpy())

    return (
        np.concatenate(all_delta_z, axis=0),
        np.concatenate(all_jsd, axis=0),
        np.concatenate(all_norms, axis=0),
        np.concatenate(all_indices, axis=0),
    )


def compute_null_model(student_model, teacher_cache, dataloader,
                       saliency_temperature, device, n_permutations=3):
    """
    Null model: replace student saliency with random values, keeping the same
    teacher saliency and embeddings.

    This tests whether the low-rank structure in delta_z comes from meaningful
    teacher-student saliency interaction or merely from embedding geometry.

    If real eigenvalues decay faster than null → saliency structure is genuine.
    If similar → low rank is an embedding artifact.
    """
    embed_layer = student_model.get_input_embeddings()
    null_spectra = []

    for perm_idx in range(n_permutations):
        all_delta_z_null = []

        for batch in tqdm(dataloader, desc=f"Null model [{perm_idx+1}/{n_permutations}]",
                          leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_mask = batch["labels_mask"].to(device)
            batch_indices = batch["index"]
            B, L = input_ids.shape

            prompt_mask = (1 - labels_mask).float() * attention_mask.float()

            # Teacher saliency (real)
            teacher_sal = get_cached_teacher_saliency(
                teacher_cache, batch_indices, L
            ).to(device)
            s_T_dist = saliency_to_distribution(
                teacher_sal, labels_mask, attention_mask, saliency_temperature
            )

            # Null student saliency: random values over prompt positions
            rand_sal = torch.rand(B, L, device=device) * prompt_mask
            s_null_dist = saliency_to_distribution(
                rand_sal, labels_mask, attention_mask, saliency_temperature
            )

            delta_s = s_T_dist - s_null_dist

            with torch.no_grad():
                embeds = embed_layer(input_ids).float()

            delta_z = torch.bmm(delta_s.unsqueeze(1), embeds).squeeze(1)
            all_delta_z_null.append(delta_z.cpu().numpy())

        Z_null = np.concatenate(all_delta_z_null, axis=0)
        Z_null = Z_null - Z_null.mean(axis=0, keepdims=True)
        _, S_null, _ = np.linalg.svd(Z_null, full_matrices=False)
        null_spectra.append(S_null ** 2)

    return np.mean(null_spectra, axis=0)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load model ---
    print(f"Loading student model: {args.student_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.float16 if args.fp16 else torch.float32
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model, torch_dtype=dtype, trust_remote_code=True
    )

    if args.student_ckpt:
        print(f"Loading checkpoint: {args.student_ckpt}")
        state_dict = torch.load(args.student_ckpt, map_location="cpu", weights_only=True)
        # Checkpoints are saved in float32 (training dtype). If loading into fp16
        # model, explicitly cast to avoid dtype mismatch errors.
        if args.fp16:
            state_dict = {k: v.half() for k, v in state_dict.items()}
        student.load_state_dict(state_dict)

    student = student.to(args.device).eval()
    # Freeze all params — we never need param gradients in this script.
    # SaliencyComputer.compute() will save/restore these states internally,
    # but since they're already False, it's effectively a no-op.
    for p in student.parameters():
        p.requires_grad_(False)

    d_model = student.config.hidden_size
    print(f"Student hidden size: {d_model}, dtype: {dtype}")

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

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    # --- Load teacher saliency cache ---
    teacher_cache, cache_meta = load_teacher_saliency_cache(args.teacher_saliency_path)

    # Sanity check: cache and dataset size should match
    if len(teacher_cache) != len(dataset):
        print(f"WARNING: teacher cache has {len(teacher_cache)} entries "
              f"but dataset has {len(dataset)} samples. "
              f"Index alignment may be wrong!")
        print(f"  Cache metadata: {cache_meta}")
        if len(teacher_cache) < len(dataset):
            print(f"  Truncating dataset to cache size.")
            dataset = torch.utils.data.Subset(dataset, range(len(teacher_cache)))
            dataloader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                collate_fn=collate_fn, num_workers=0
            )

    # --- Compute delta_z for all samples ---
    saliency_computer = SaliencyComputer(temperature=args.saliency_temperature)

    print("\n=== Computing saliency-weighted embedding differences ===")
    delta_z, jsd_values, norms, indices = compute_delta_z(
        student, saliency_computer, teacher_cache,
        dataloader, args.saliency_temperature, args.device
    )
    n, d = delta_z.shape
    print(f"Delta Z matrix shape: ({n}, {d})")
    print(f"JSD  — mean: {jsd_values.mean():.6f}, std: {jsd_values.std():.6f}, "
          f"max: {jsd_values.max():.6f}, min: {jsd_values.min():.6f}")
    print(f"||Δz|| — mean: {norms.mean():.4f}, std: {norms.std():.4f}, "
          f"max: {norms.max():.4f}, min: {norms.min():.4f}")

    # Check for degenerate cases
    zero_norm_count = (norms < 1e-8).sum()
    if zero_norm_count > 0:
        print(f"WARNING: {zero_norm_count} samples have near-zero ||Δz|| "
              f"(saliency distributions already aligned)")

    nan_count = np.isnan(delta_z).any(axis=1).sum()
    inf_count = np.isinf(delta_z).any(axis=1).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"WARNING: {nan_count} NaN rows, {inf_count} Inf rows in delta_z. "
              f"Replacing with zeros.")
        bad_mask = np.isnan(delta_z).any(axis=1) | np.isinf(delta_z).any(axis=1)
        delta_z[bad_mask] = 0.0

    # --- SVD on centered delta_z ---
    print("\n=== SVD on saliency error matrix ===")
    Z = delta_z - delta_z.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    n_components = len(S)

    eigenvalues = S ** 2
    total_var = eigenvalues.sum()
    if total_var < 1e-12:
        print("ERROR: total variance is near-zero. All saliency patterns are identical.")
        return

    cumvar = np.cumsum(eigenvalues) / total_var

    for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
        r = int(np.searchsorted(cumvar, threshold)) + 1
        print(f"  Effective rank (>{threshold*100:.0f}% var): "
              f"{r} / {n_components}")

    print(f"\n  Top-10 eigenvalues (% of total):")
    for k in range(min(10, n_components)):
        print(f"    PC{k+1}: {eigenvalues[k]/total_var*100:.2f}%  "
              f"(cumulative: {cumvar[k]*100:.2f}%)")

    # --- Null model comparison ---
    print(f"\n=== Null model ({args.null_permutations} random permutations) ===")
    null_eigenvalues = compute_null_model(
        student, teacher_cache, dataloader,
        args.saliency_temperature, args.device,
        n_permutations=args.null_permutations
    )

    null_total = null_eigenvalues.sum()
    if null_total < 1e-12:
        print("ERROR: null model total variance is near-zero.")
        return

    null_cumvar = np.cumsum(null_eigenvalues) / null_total

    print("  Effective rank comparison (real vs null):")
    for threshold in [0.5, 0.8, 0.9, 0.95]:
        r_real = int(np.searchsorted(cumvar, threshold)) + 1
        r_null = int(np.searchsorted(null_cumvar, threshold)) + 1
        ratio = r_real / max(r_null, 1)
        print(f"    >{threshold*100:.0f}%: real={r_real}, null={r_null}, "
              f"ratio={ratio:.2f} {'← LOWER RANK ✓' if ratio < 0.8 else ''}")

    # Concentration ratio: what fraction of variance is in top-10 PCs?
    top10_real = cumvar[min(9, n_components - 1)]
    top10_null = null_cumvar[min(9, len(null_eigenvalues) - 1)]
    print(f"\n  Top-10 PCs explain: real={top10_real*100:.1f}%, null={top10_null*100:.1f}%")
    if top10_real > top10_null + 0.05:
        print("  → Saliency errors are MORE concentrated than null ✓")
    else:
        print("  → Saliency errors are NOT more concentrated than null ✗")

    # --- Per-PC sample analysis ---
    print("\n=== Per-PC sample analysis ===")
    n_pcs = min(20, n_components)
    projections = U[:, :n_pcs] * S[:n_pcs]  # (n, n_pcs)
    for k in range(min(5, n_pcs)):
        proj_k = np.abs(projections[:, k])
        top_idx = np.argsort(proj_k)[-5:][::-1]
        jsd_str = ", ".join(f"{jsd_values[i]:.4f}" for i in top_idx)
        norm_str = ", ".join(f"{norms[i]:.3f}" for i in top_idx)
        print(f"  PC{k+1}: top-5 JSD = [{jsd_str}]")
        print(f"         top-5 ||Δz|| = [{norm_str}]")

    # --- Save results ---
    results = {
        "n_samples": int(n),
        "d_model": int(d),
        "n_components": int(n_components),
        "eigenvalues": eigenvalues[:min(100, n_components)].tolist(),
        "cumulative_variance": cumvar[:min(100, n_components)].tolist(),
        "null_eigenvalues": null_eigenvalues[:min(100, len(null_eigenvalues))].tolist(),
        "null_cumulative_variance": null_cumvar[:min(100, len(null_cumvar))].tolist(),
        "effective_rank": {
            str(t): int(np.searchsorted(cumvar, t) + 1)
            for t in [0.5, 0.8, 0.9, 0.95, 0.99]
        },
        "null_effective_rank": {
            str(t): int(np.searchsorted(null_cumvar, t) + 1)
            for t in [0.5, 0.8, 0.9, 0.95, 0.99]
        },
        "top10_variance_real": float(top10_real),
        "top10_variance_null": float(top10_null),
        "jsd_stats": {
            "mean": float(jsd_values.mean()),
            "std": float(jsd_values.std()),
            "max": float(jsd_values.max()),
            "min": float(jsd_values.min()),
        },
        "norm_stats": {
            "mean": float(norms.mean()),
            "std": float(norms.std()),
            "max": float(norms.max()),
            "min": float(norms.min()),
        },
        "args": vars(args),
    }

    results_path = os.path.join(args.output_dir, "pca_analysis.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    np.savez(
        os.path.join(args.output_dir, "pca_data.npz"),
        eigenvalues=eigenvalues,
        null_eigenvalues=null_eigenvalues,
        cumvar=cumvar,
        null_cumvar=null_cumvar,
        projections=projections,
        jsd_values=jsd_values,
        norms=norms,
        indices=indices,
        Vt=Vt[:n_pcs],  # top PC directions for later visualization
    )
    print(f"Raw data saved to {os.path.join(args.output_dir, 'pca_data.npz')}")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # (a) Eigenvalue spectrum (log scale)
        ax = axes[0]
        n_plot = min(50, n_components)
        k_range = np.arange(1, n_plot + 1)
        ax.semilogy(k_range, eigenvalues[:n_plot] / total_var,
                     "b-o", markersize=3, label="Real")
        ax.semilogy(k_range, null_eigenvalues[:n_plot] / null_total,
                     "r--x", markersize=3, label="Null (random sal.)")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Fraction of Variance (log)")
        ax.set_title("(a) Eigenvalue Spectrum")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (b) Cumulative variance
        ax = axes[1]
        ax.plot(k_range, cumvar[:n_plot] * 100, "b-o", markersize=3, label="Real")
        ax.plot(k_range, null_cumvar[:n_plot] * 100, "r--x", markersize=3, label="Null")
        ax.axhline(y=90, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(y=80, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Variance (%)")
        ax.set_title("(b) Cumulative Explained Variance")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (c) PC1 vs PC2 scatter, colored by JSD
        ax = axes[2]
        sc = ax.scatter(
            projections[:, 0], projections[:, 1],
            c=jsd_values, cmap="viridis", s=5, alpha=0.6
        )
        plt.colorbar(sc, ax=ax, label="JSD")
        ax.set_xlabel(f"PC1 ({eigenvalues[0]/total_var*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({eigenvalues[1]/total_var*100:.1f}%)")
        ax.set_title("(c) Samples in PC1-PC2 Space")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(args.output_dir, "pca_analysis.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {fig_path}")
        plt.close()

    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
