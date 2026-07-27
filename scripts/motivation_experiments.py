#!/usr/bin/env python3
"""SaKD motivation experiments M1 and M2.

M1: Failure-mode scatter
    For each sample x_i, compute
        X = D_KL(f_T(x_i) || f_S(x_i))                       (clean teacher-student KL)
        Y = E_xi[D_KL(f_T(x_i+xi) || f_S(x_i+xi))] - X       (neighborhood degradation)
    Plot (X, Y); highlight "failure mode" cluster where X is small but Y is large.

M2: Neighborhood-gap distribution
    Overlay distributions of E_xi[D_KL(f_T(x_i) || f_M(x_i+xi))] for M in
        {teacher (self-consistency control), Standard-KD student, optional SaKD student}.

Both subcommands use isotropic Gaussian noise on input embeddings with sigma = relative
fraction of mean embedding norm (default 0.01), averaged over `--noise_repeats` draws,
forward-KL on response positions only.

Mirrors the embedding-perturbation conventions in src/sagd/trainer.py (`_compute_noisy_kl`)
but with isotropic noise instead of saliency-adaptive noise (motivation only).

Outputs:
  - PDF figure (NeurIPS single-column 0.7 linewidth, vector graphics).
  - JSON sidecar with raw per-sample numbers for re-plotting / supplementary tables.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sagd.data import InstructionDataset, collate_fn
from sagd.models import load_student, load_teacher


# ---------------------------------------------------------------------------
# Core KL primitives
# ---------------------------------------------------------------------------


def per_sample_forward_kl(
    t_logits: torch.Tensor,    # (B, L, V)
    s_logits: torch.Tensor,    # (B, L, V)
    labels_mask: torch.Tensor,  # (B, L)
    temperature: float = 1.0,
) -> torch.Tensor:
    """Per-sample forward KL D_KL(p_T || p_S) over response positions.

    Shift alignment matches the trainer: logit[j] predicts token[j+1] so the
    response mask at position j+1 selects which positions count.
    """
    t_shifted = t_logits[:, :-1, :]
    s_shifted = s_logits[:, :-1, :]
    mask = labels_mask[:, 1:].float()  # (B, L-1)

    t_log = F.log_softmax(t_shifted / temperature, dim=-1)
    s_log = F.log_softmax(s_shifted / temperature, dim=-1)
    t_probs = t_log.exp()

    per_pos = (t_probs * (t_log - s_log)).sum(dim=-1)  # (B, L-1)
    per_pos = per_pos * mask  # zero out non-response positions
    mask_count = mask.sum(dim=-1).clamp(min=1)  # (B,)
    per_sample = per_pos.sum(dim=-1) / mask_count  # (B,)
    return per_sample * (temperature ** 2)


def isotropic_noisy_embeds(embed: torch.Tensor, sigma_rel: float) -> torch.Tensor:
    """Isotropic Gaussian noise scaled to a fraction of mean embedding norm.

    Returns embed + xi where xi ~ N(0, sigma^2 I), sigma = sigma_rel * <||embed||>.
    """
    embed_norm = embed.norm(dim=-1).mean().item()
    sigma = sigma_rel * embed_norm
    return embed + torch.randn_like(embed) * sigma


# ---------------------------------------------------------------------------
# Per-batch routine: clean and noisy KL between two models
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_clean_and_noisy_kl(
    model_p: torch.nn.Module,    # reference distribution (always teacher in our usage)
    model_q: torch.nn.Module,    # comparison distribution (teacher OR student)
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels_mask: torch.Tensor,
    sigma_rel: float,
    noise_repeats: int,
) -> dict[str, torch.Tensor]:
    """Compute per-sample KL(p || q) at clean input and averaged over noise draws.

    For M1, model_p = teacher, model_q = student.
    For M2, the same call is used with model_q in {teacher, kd_student, sagd_student}
    against the same teacher reference.

    Both models are run on independently-sampled noise of the same isotropic scale,
    matching the trainer convention (different embedding dims => independent z, same sigma).
    """
    e_p = model_p.get_input_embeddings()(input_ids)  # (B, L, d_p)
    e_q = model_q.get_input_embeddings()(input_ids)  # (B, L, d_q)

    p_logits_clean = model_p(inputs_embeds=e_p, attention_mask=attention_mask).logits.float()
    q_logits_clean = model_q(inputs_embeds=e_q, attention_mask=attention_mask).logits.float()

    kl_clean = per_sample_forward_kl(p_logits_clean, q_logits_clean, labels_mask)

    kl_noisy_accum = torch.zeros_like(kl_clean)
    for _ in range(noise_repeats):
        e_p_noisy = isotropic_noisy_embeds(e_p, sigma_rel)
        e_q_noisy = isotropic_noisy_embeds(e_q, sigma_rel)

        p_logits_noisy = model_p(inputs_embeds=e_p_noisy, attention_mask=attention_mask).logits.float()
        q_logits_noisy = model_q(inputs_embeds=e_q_noisy, attention_mask=attention_mask).logits.float()

        kl_noisy_accum = kl_noisy_accum + per_sample_forward_kl(
            p_logits_noisy, q_logits_noisy, labels_mask,
        )

    kl_noisy_mean = kl_noisy_accum / max(noise_repeats, 1)

    return {
        "kl_clean": kl_clean.detach().cpu(),
        "kl_noisy": kl_noisy_mean.detach().cpu(),
    }


# ---------------------------------------------------------------------------
# Plotting helpers (NeurIPS single-column style)
# ---------------------------------------------------------------------------


def _setup_neurips_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
    })


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.5)


# ---------------------------------------------------------------------------
# Dataset / student loaders
# ---------------------------------------------------------------------------


def build_dataset(tokenizer, n_samples: int, max_seq_len: int, seed: int) -> InstructionDataset:
    """Dolly-15K val (MiniLLM/dolly valid.jsonl, 500 samples) — first n_samples."""
    return InstructionDataset(
        tokenizer=tokenizer,
        dataset_name="MiniLLM/dolly",
        max_seq_len=max_seq_len,
        max_samples=n_samples,
        seed=seed,
        subset="val",
    )


def make_dataloader(dataset, batch_size: int) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
    )


def load_student_with_ckpt(
    student_model_name: str, ckpt_path: str, device: str,
) -> torch.nn.Module:
    """Load a student architecture and overlay a checkpoint state_dict.

    Returns the model in eval() with grads disabled. We never train here.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Student checkpoint not found: {ckpt_path}\n"
            f"Cannot run motivation experiment without it. "
            f"Train a {student_model_name} student first via "
            f"`python scripts/train.py --method standard_kd ...`."
        )
    student, _ = load_student(student_model_name, device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    student.load_state_dict(state_dict)
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)
    return student


# ---------------------------------------------------------------------------
# M1: Failure-mode scatter
# ---------------------------------------------------------------------------


def run_m1(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading teacher: {args.teacher}")
    teacher, _ = load_teacher(args.teacher, args.device)

    print(f"Loading student arch: {args.student_model_name}  ckpt: {args.student}")
    student = load_student_with_ckpt(args.student_model_name, args.student, args.device)

    _, student_tok = load_student(args.student_model_name, args.device)
    dataset = build_dataset(student_tok, args.n_samples, args.max_seq_len, args.seed)
    print(f"Dolly-15K val: {len(dataset)} samples")

    loader = make_dataloader(dataset, args.batch_size)

    indices: list[int] = []
    kl_clean_all: list[float] = []
    kl_noisy_all: list[float] = []

    for batch in tqdm(loader, desc="M1 scoring"):
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels_mask = batch["labels_mask"].to(args.device)
        idx_batch = batch["index"].tolist()

        n_resp = labels_mask[:, 1:].sum(dim=-1)
        valid = (n_resp > 0).cpu().tolist()

        out = compute_clean_and_noisy_kl(
            teacher, student,
            input_ids, attention_mask, labels_mask,
            sigma_rel=args.noise_sigma, noise_repeats=args.noise_repeats,
        )

        for i, ok in enumerate(valid):
            if not ok:
                continue
            indices.append(idx_batch[i])
            kl_clean_all.append(float(out["kl_clean"][i].item()))
            kl_noisy_all.append(float(out["kl_noisy"][i].item()))

    kl_clean = np.asarray(kl_clean_all)
    kl_noisy = np.asarray(kl_noisy_all)
    delta = kl_noisy - kl_clean

    # Failure-mode threshold: bottom-quartile X, top-quartile Y.
    x_low = float(np.quantile(kl_clean, 0.25))
    y_high = float(np.quantile(delta, 0.75))
    failure_mask = (kl_clean <= x_low) & (delta >= y_high)
    failure_frac = float(failure_mask.mean())

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({
            "experiment": "M1",
            "teacher": args.teacher,
            "student_arch": args.student_model_name,
            "student_ckpt": args.student,
            "dataset": "MiniLLM/dolly val (first {} samples)".format(args.n_samples),
            "noise_sigma_rel": args.noise_sigma,
            "noise_repeats": args.noise_repeats,
            "n_samples": int(len(kl_clean)),
            "failure_mode": {
                "x_low_threshold_q25": x_low,
                "y_high_threshold_q75": y_high,
                "fraction": failure_frac,
                "count": int(failure_mask.sum()),
                "indices": [int(indices[i]) for i in np.where(failure_mask)[0]],
            },
            "samples": [
                {"index": int(indices[i]),
                 "kl_clean": float(kl_clean[i]),
                 "kl_noisy": float(kl_noisy[i]),
                 "delta": float(delta[i])}
                for i in range(len(indices))
            ],
        }, f, indent=2)
    print(f"Wrote {args.output_json}")

    _setup_neurips_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.85, 2.6))
    _despine(ax)

    other_mask = ~failure_mask
    ax.scatter(kl_clean[other_mask], delta[other_mask],
               s=10, alpha=0.35, c="#3274A1", edgecolors="none",
               label="all samples")
    ax.scatter(kl_clean[failure_mask], delta[failure_mask],
               s=14, alpha=0.85, c="#C03A2B", edgecolors="none",
               label=f"failure mode ({failure_frac*100:.1f}%)")

    ax.set_xlabel(r"$D_{\mathrm{KL}}(f_T(x)\,\|\,f_S(x))$ at training point")
    ax.set_ylabel(r"$\mathbb{E}_\xi[D(x{+}\xi)] - D(x)$")
    ax.legend(frameon=False, loc="upper right")

    ax.axvline(x_low, color="#888888", linestyle=":", linewidth=0.6)
    ax.axhline(y_high, color="#888888", linestyle=":", linewidth=0.6)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output_pdf) or ".", exist_ok=True)
    fig.savefig(args.output_pdf, format="pdf")
    plt.close(fig)
    print(f"Wrote {args.output_pdf}")

    print()
    print(f"[M1] processed {len(kl_clean)} samples")
    print(f"[M1] failure-mode fraction: {failure_frac*100:.2f}%  "
          f"({'PASS' if failure_frac >= 0.05 else 'FAIL'} >=5%)")


# ---------------------------------------------------------------------------
# M2: Neighborhood-gap distribution
# ---------------------------------------------------------------------------


def run_m2(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if len(args.students) != len(args.student_labels):
        raise ValueError(
            f"--students ({len(args.students)}) and --student_labels "
            f"({len(args.student_labels)}) must have the same length."
        )

    # Filter to checkpoints that actually exist (graceful degradation per
    # acceptance criterion 6: at minimum Standard-KD must be present).
    present_students: list[tuple[str, str]] = []
    for path, label in zip(args.students, args.student_labels):
        if os.path.isfile(path):
            present_students.append((path, label))
        else:
            print(f"[M2] checkpoint missing, skipping: {label} -> {path}")
    if not present_students:
        raise FileNotFoundError(
            "No student checkpoints available for M2. At minimum the "
            "Standard-KD checkpoint is required.",
        )

    print(f"Loading teacher: {args.teacher}")
    teacher, _ = load_teacher(args.teacher, args.device)

    _, student_tok = load_student(args.student_model_name, args.device)
    dataset = build_dataset(student_tok, args.n_samples, args.max_seq_len, args.seed)
    print(f"Dolly-15K val: {len(dataset)} samples")

    loader = make_dataloader(dataset, args.batch_size)

    # Distribution 0: teacher self-consistency = D_KL(f_T(x) || f_T(x+xi)).
    # We reuse compute_clean_and_noisy_kl with model_p = model_q = teacher;
    # the meaningful quantity is `kl_noisy` (kl_clean ≈ 0 by construction).
    print("[M2] computing teacher self-consistency control...")
    teacher_self_kl: list[float] = []
    for batch in tqdm(loader, desc="teacher self-consistency"):
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels_mask = batch["labels_mask"].to(args.device)
        n_resp = labels_mask[:, 1:].sum(dim=-1).cpu()

        out = compute_clean_and_noisy_kl(
            teacher, teacher,
            input_ids, attention_mask, labels_mask,
            sigma_rel=args.noise_sigma, noise_repeats=args.noise_repeats,
        )
        for i in range(out["kl_noisy"].size(0)):
            if n_resp[i].item() > 0:
                teacher_self_kl.append(float(out["kl_noisy"][i].item()))

    # Distributions 1+: teacher vs each student under noise.
    student_distributions: dict[str, list[float]] = {}
    for ckpt_path, label in present_students:
        print(f"[M2] computing teacher-vs-{label} ({ckpt_path})...")
        student = load_student_with_ckpt(args.student_model_name, ckpt_path, args.device)

        per_sample_kl: list[float] = []
        for batch in tqdm(loader, desc=label):
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels_mask = batch["labels_mask"].to(args.device)
            n_resp = labels_mask[:, 1:].sum(dim=-1).cpu()

            out = compute_clean_and_noisy_kl(
                teacher, student,
                input_ids, attention_mask, labels_mask,
                sigma_rel=args.noise_sigma, noise_repeats=args.noise_repeats,
            )
            for i in range(out["kl_noisy"].size(0)):
                if n_resp[i].item() > 0:
                    per_sample_kl.append(float(out["kl_noisy"][i].item()))
        student_distributions[label] = per_sample_kl

        del student
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Acceptance: Standard-KD 95th percentile >= 1.5x teacher's.
    teacher_p95 = float(np.percentile(np.asarray(teacher_self_kl), 95))
    standard_kd_label = next(
        (lab for _, lab in present_students if "Standard" in lab),
        present_students[0][1],
    )
    kd_p95 = float(np.percentile(np.asarray(student_distributions[standard_kd_label]), 95))
    ratio = kd_p95 / teacher_p95 if teacher_p95 > 0 else float("inf")

    summary = {
        "teacher_self": {
            "mean": float(np.mean(teacher_self_kl)),
            "median": float(np.median(teacher_self_kl)),
            "p95": teacher_p95,
            "values": teacher_self_kl,
        },
    }
    for label, vals in student_distributions.items():
        arr = np.asarray(vals)
        summary[label] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "values": vals,
        }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({
            "experiment": "M2",
            "teacher": args.teacher,
            "student_arch": args.student_model_name,
            "students": [{"label": lab, "ckpt": p} for p, lab in present_students],
            "dataset": f"MiniLLM/dolly val (first {args.n_samples} samples)",
            "noise_sigma_rel": args.noise_sigma,
            "noise_repeats": args.noise_repeats,
            "n_samples": len(teacher_self_kl),
            "acceptance": {
                "teacher_p95": teacher_p95,
                "standard_kd_label_used": standard_kd_label,
                "standard_kd_p95": kd_p95,
                "ratio": ratio,
                "passes_1.5x": bool(ratio >= 1.5),
            },
            "distributions": summary,
        }, f, indent=2)
    print(f"Wrote {args.output_json}")

    # Plot — log-x KDE on the gap distribution. Forward-KL has a long right tail;
    # log-x makes the heavier-tail comparison visible at single-column size.
    _setup_neurips_style()
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(3.85, 2.6))
    _despine(ax)

    palette = {
        "teacher": "#666666",
        "Standard KD": "#3274A1",
        "SaKD": "#C03A2B",
    }

    def _kde_plot(values: np.ndarray, label: str, color: str) -> None:
        v = np.clip(np.asarray(values, dtype=np.float64), 1e-8, None)
        v = np.log10(v)
        kde = gaussian_kde(v, bw_method=0.35)
        grid = np.linspace(v.min() - 0.3, v.max() + 0.3, 400)
        dens = kde(grid)
        ax.plot(grid, dens, color=color, linewidth=1.3, label=label)
        ax.fill_between(grid, dens, color=color, alpha=0.18, linewidth=0)

    _kde_plot(np.asarray(teacher_self_kl), "Teacher self-consistency",
              palette["teacher"])
    for label, vals in student_distributions.items():
        color = palette.get(label, None)
        if color is None:
            color = "#3274A1" if "Standard" in label else "#C03A2B"
        _kde_plot(np.asarray(vals), label, color)

    ax.set_xlabel(r"$\log_{10}\,\mathbb{E}_\xi[D_{\mathrm{KL}}(f_T(x)\,\|\,f_M(x{+}\xi))]$")
    ax.set_ylabel("density")
    ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output_pdf) or ".", exist_ok=True)
    fig.savefig(args.output_pdf, format="pdf")
    plt.close(fig)
    print(f"Wrote {args.output_pdf}")

    print()
    print(f"[M2] processed {len(teacher_self_kl)} samples")
    print(f"[M2] teacher self-consistency p95: {teacher_p95:.6f}")
    print(f"[M2] {standard_kd_label} p95: {kd_p95:.6f}")
    print(f"[M2] ratio: {ratio:.2f}x  "
          f"({'PASS' if ratio >= 1.5 else 'FAIL'} >=1.5x)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--teacher", default="Qwen/Qwen3-8B")
    p.add_argument("--student_model_name", default="Qwen/Qwen3-0.6B",
                   help="Student architecture used to load the checkpoint(s).")
    p.add_argument("--n_samples", type=int, default=500,
                   help="Number of Dolly val samples to process (max 500).")
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--noise_sigma", type=float, default=0.01,
                   help="Noise std as fraction of mean embedding norm.")
    p.add_argument("--noise_repeats", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SaKD motivation experiments")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("m1", help="Failure-mode scatter")
    _add_common_args(p1)
    p1.add_argument("--student", required=True,
                    help="Path to Standard-KD student checkpoint (.pt).")
    p1.add_argument("--output_pdf", required=True)
    p1.add_argument("--output_json", required=True)

    p2 = sub.add_parser("m2", help="Neighborhood-gap distribution")
    _add_common_args(p2)
    p2.add_argument("--students", nargs="+", required=True,
                    help="One or more student checkpoint paths.")
    p2.add_argument("--student_labels", nargs="+", required=True,
                    help="Display labels matching --students (e.g. 'Standard KD' 'SaKD').")
    p2.add_argument("--output_pdf", required=True)
    p2.add_argument("--output_json", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "m1":
        run_m1(args)
    elif args.cmd == "m2":
        run_m2(args)
    else:  # pragma: no cover
        raise SystemExit(f"Unknown subcommand: {args.cmd}")


if __name__ == "__main__":
    main()
