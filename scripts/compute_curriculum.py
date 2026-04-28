"""
Compute training curriculum order from SVD results.

Takes PCA data (from verify_saliency_pca.py or gradient_pca_selection.py)
and produces a sorted index file for curriculum training.

Scoring: score(i) = sum_k |u_ik| * sigma_k  for top-r PCs
  High score = sample aligns with dominant principal directions = train first

Usage:
    # From saliency PCA results
    python scripts/compute_curriculum.py \
        --pca_path outputs/saliency_pca_analysis/kd_0.6B/pca_data.npz \
        --output_path outputs/curriculum/saliency_order.pt \
        --top_r 50

    # From gradient PCA results
    python scripts/compute_curriculum.py \
        --pca_path outputs/gradient_pca/kd_0.6B/gradient_profile.npz \
        --output_path outputs/curriculum/gradient_order.pt \
        --top_r 50 --is_gradient
"""

import argparse
import json
import os

import numpy as np
import torch


def compute_scores_from_svd(U, S, top_r):
    """Compute structural alignment score per sample.

    score(i) = sum_{k=1}^{top_r} |U[i,k]| * S[k]

    Samples with high score contribute most to the top principal directions.
    """
    r = min(top_r, len(S), U.shape[1])
    scores = np.abs(U[:, :r]) @ S[:r]  # (n,)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca_path", type=str, required=True,
                        help="Path to pca_data.npz or gradient_profile.npz")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--top_r", type=int, default=50,
                        help="Number of top PCs to use for scoring")
    parser.add_argument("--is_gradient", action="store_true",
                        help="If set, input is gradient_profile.npz (needs SVD)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    data = np.load(args.pca_path)

    if args.is_gradient:
        # gradient_profile.npz has raw projected_grads, need to do SVD
        G = data["projected_grads"]
        indices = data["indices"]
        G_centered = G - G.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(G_centered, full_matrices=False)
    else:
        # pca_data.npz from verify_saliency_pca.py already has eigenvalues
        # but we need U and S. Recompute from projections if available,
        # or load eigenvalues and compute scores differently.
        if "projections" in data:
            # projections = U[:, :k] * S[:k], shape (n, k)
            projections = data["projections"]
            eigenvalues = data["eigenvalues"]
            S = np.sqrt(np.maximum(eigenvalues[:projections.shape[1]], 1e-12))
            # Recover U: projections[:, k] = U[:, k] * S[k]
            U = projections / S[np.newaxis, :]  # safe: S clamped above
            indices = data["indices"]
        else:
            raise ValueError("pca_data.npz must contain 'projections' and 'eigenvalues'")

    # Compute scores
    scores = compute_scores_from_svd(U, S, args.top_r)
    n = len(scores)

    # Sort: highest score first (most structurally aligned → train first)
    sorted_positions = np.argsort(scores)[::-1]
    sorted_dataset_indices = indices[sorted_positions]
    sorted_scores = scores[sorted_positions]

    # Print summary
    print(f"=== Curriculum Order ===")
    print(f"Total samples: {n}")
    print(f"Top-r for scoring: {min(args.top_r, len(S))}")
    print(f"Score range: [{sorted_scores[-1]:.4f}, {sorted_scores[0]:.4f}]")
    print(f"Score mean: {scores.mean():.4f}, std: {scores.std():.4f}")

    # Show phases
    for pct in [0.3, 0.5, 0.7, 1.0]:
        k = int(n * pct)
        print(f"  Top {pct*100:.0f}% ({k} samples): "
              f"score range [{sorted_scores[k-1]:.4f}, {sorted_scores[0]:.4f}]")

    # Save
    result = {
        "sorted_indices": torch.tensor(sorted_dataset_indices, dtype=torch.long),
        "scores": torch.tensor(sorted_scores, dtype=torch.float32),
        "metadata": {
            "source": args.pca_path,
            "top_r": args.top_r,
            "n_samples": int(n),
        }
    }
    torch.save(result, args.output_path)
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
