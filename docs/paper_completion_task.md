# Task: Bring SaKD NeurIPS 2026 Paper to Camera-Ready Quality

**Status:** Submitted version is in `writing/NeurIPS26-SaGD/`, last Overleaf commit is the
project's current HEAD. The paper compiles and tells a coherent story, but several
experiments are placeholder, several figures still have known issues, and the manuscript
needs a final polish + page-budget audit. This document is the **single source of truth**
for the work needed to take the paper to a top-tier finished state.

**Owner:** Autonomous server-side Claude Code agent. Execute via `/goal` against this
file. Do not return control to the human until the acceptance criteria of PHASE 0–10 are
all met (or you have written an irrecoverable BLOCKER to `tmp/PAPER_COMPLETION_DONE.md`).

**Authorization:** end-to-end on everything under the repo root. Specifically:
- **Writable**: `tmp/`, `writing/NeurIPS26-SaGD/`, `scripts/`, new `outputs_dolly/qwen_*/`
  directories, this `docs/` directory.
- **Append-only / do not delete**: existing `outputs_dolly/llama_1B/` checkpoints, any
  existing `outputs_dolly/qwen_*/` checkpoints.
- **Read-only**: `data/teacher_saliency_*.pt` (regenerate to a new file if you must;
  never overwrite). Existing precomputed dataset caches in `data/`.

**Operating constraints**
- Run with `--dangerously-skip-permissions`. Do NOT prompt the human for confirmation.
- Run inside a tmux session named `sakd_finish` so the human can `tmux attach` to inspect.
- Append a one-line status to `tmp/exp_progress.log` whenever a phase starts/ends or a
  sub-job finishes. Format: `HH:MM:SS PHASE-N <event>`.
- All deliverable artefacts go under `tmp/` with stable filenames listed in each phase.
- LaTeX-side edits go under `writing/NeurIPS26-SaGD/` and are pushed to Overleaf
  (`origin master`) after every commit. Always `git stash && git pull --rebase &&
  git stash pop` before staging to handle concurrent Overleaf-web edits.
- 4× A100 80GB available. Reuse the parallel lockfile pattern in
  `scripts/a100_parallel_eval.sh` to avoid GPU contention.
- Hard time budget for the full run: ≤ 36 hours wall-clock. If a phase exceeds its
  individual budget, mark PARTIAL in `EXPERIMENTS_DONE.md` and continue.

---

## 0. Background: current paper state (read before doing anything)

```
writing/NeurIPS26-SaGD/
├── neurips_2026.tex         # root; abstract is inline; \input{sections/...}
├── sections/
│   ├── introduction.tex     # has Fig 1 (M1+M2 motivation, LLaMA)
│   ├── preliminary.tex
│   ├── method.tex           # 3 theorems + Algorithm 1 (currently in §3.4 of main text)
│   ├── background.tex       # 2 short subsections (KD for LLMs, gradient attribution)
│   ├── experiments.tex      # §4.1 setup, §4.2 main result, §4.3 ablation, §4.4 EC
│   ├── conclusion.tex       # §5 Conclusion + Outlook + §6 Limitations
│   └── appendix.tex         # proofs, noise-coupling remark, implementation, compute cost
├── tables/                  # dolly_main(.tex / _t.tex), ablation, ec, compute_cost
├── figures/                 # framework, ablation (hidden bar chart)
├── algorithms/sakd.tex      # Algorithm 1, included in §3.4
├── sources/                 # motivation_M{1,2}.pdf, framework.pdf
├── checklist.tex            # NeurIPS reproducibility checklist (all 16 items filled)
└── references.bib
```

The **submitted version** is the last commit on Overleaf `master`. Everything below
treats the submitted version as the baseline and adds completeness / polish on top.

### What is known to be missing or weak (the gap this doc closes)

1. **EC table only has Qwen3-0.6B and LLaMA-1B single-seed; no 1.7B and no std.**
2. **No hyperparameter-sensitivity sweep** for λ, σ, τ_w, N.
3. **No training-dynamics curves** (loss / saliency-divergence over training).
4. **No benchmark defense** (MMLU / ARC-C / TruthfulQA via lm-eval-harness).
5. **Saliency heatmap** in `tmp/saliency_heatmap.pdf` is LLaMA-only; the paper expects
   Qwen3-0.6B for consistency with §4.4.
6. **Figure 1 left panel (M1)** does not actually have a shaded region, but the caption
   claims one — caption + figure mismatch.
7. **Page budget** is over 9 (NeurIPS limit). Either trim or accept overflow per
   PHASE 9's tactic ladder.
8. **References.bib** has ~6–14 orphan entries left from earlier scope changes.
9. **Qwen3 checkpoints may not exist on the GPU server** (per previous PHASE 0 inventory).
   PHASE 1 handles this BLOCKER first.

---

## 1. Acceptance summary (the bar for "done")

The work is **complete** when **all** of the following are true:

- A2.0: Every phase below has a `tmp/EXPERIMENTS_DONE.md` row with status DONE or
  PARTIAL (and PARTIAL is justified with a 1-paragraph note).
- A2.1: `writing/NeurIPS26-SaGD/main.pdf` compiles cleanly (zero `??` references, zero
  undefined citations, zero LaTeX errors) and is committed to Overleaf `master`.
- A2.2: Main text is ≤ 9 pages excluding references / appendix / checklist; if you
  cannot get to 9 pages with PHASE 9 tactics, document the overflow and which content
  was prioritised in `tmp/PAGE_BUDGET.md`.
- A2.3: Tables `tab:dolly`, `tab:ablation`, `tab:ec`, `tab:compute-cost` all reflect
  numbers reproducible from `outputs_dolly/` + the eval scripts in `scripts/`.
- A2.4: New table `tab:hp_sensitivity` and new figures `fig:training_dynamics`,
  `fig:saliency_heatmap_qwen` exist and are referenced from the paper.
- A2.5: All `\GH{...}` / `\CC{...}` reviewer notes and all `% TODO` markers in
  rendered files are removed (commented-out legacy block in method.tex may stay).
- A2.6: Final summary `tmp/PAPER_COMPLETION_DONE.md` lists every artefact produced and
  every paper section / table / figure touched, with a 1-line diff summary.

---

## 2. Repository assumptions (verify in PHASE 0)

| Asset | Expected location | Required by |
|---|---|---|
| Qwen3-8B teacher saliency cache | `data/teacher_saliency_dolly.pt`, `data/teacher_saliency_squad.pt` | PHASE 2, 3, 6 |
| LLaMA-3.1-8B teacher saliency cache (Dolly) | `data/teacher_saliency_dolly_llama.pt` | already used; not modified |
| Trained student checkpoints | `outputs_dolly/{qwen_0.6B,qwen_1.7B,llama_1B}/<method>/seed_<S>/student_final.pt` | every eval phase |
| Training scripts | `scripts/a100_qwen_train.sh` (create if missing — model on `a100_llama_train.sh`), `scripts/a100_llama_train.sh`, `scripts/a100_parallel_eval.sh` | PHASE 1, 3 |
| Saliency / diagnosis | `scripts/diagnose_saliency.py`, `scripts/precompute_teacher_saliency.py` | PHASE 2, 6 |
| lm-eval-harness | installed in env; cache at `~/.cache/lm-eval` | PHASE 5 |

---

## 3. Phases

Each phase has:
- **Goal** — what success looks like
- **Prereqs** — phases that must finish first
- **Steps** — concrete, ordered actions
- **Deliverables** — exact filenames the phase must produce
- **Acceptance** — checkable post-conditions
- **If blocked** — fallback / escalation
- **DO NOT** — explicit out-of-scope items

---

### PHASE 0 — Inventory & sanity (≤ 20 min)

**Goal:** Know exactly what checkpoints / caches exist before scheduling any training or
eval. Decide which downstream phases are blocked.

**Prereqs:** none.

**Steps:**
1. Walk `outputs_dolly/` and emit `tmp/phase0_inventory.json` listing every
   `(student, method, seed)` triple with: ckpt path, mtime, size, `nan_check` (load
   state_dict on CPU; `torch.isnan(v).any()` for every tensor).
2. Verify presence of both teacher saliency caches; emit `tmp/phase0_caches.json` with
   `n_samples`, `max_seq_len`, `model`, `dataset` for each.
3. Verify `lm-eval-harness` is installed (`pip show lm-eval`). Verify GPU availability
   with `nvidia-smi --query-gpu=index,memory.free --format=csv`.
4. List which cells of the EC matrix (PHASE 2) are missing.
5. Append to `tmp/EXPERIMENTS_DONE.md`: row for PHASE 0 with status DONE.

**Deliverables:**
- `tmp/phase0_inventory.json`
- `tmp/phase0_caches.json`
- `tmp/phase0_gpu.json`

**Acceptance:** all three files exist; `tmp/EXPERIMENTS_DONE.md` has a PHASE 0 row.

**If blocked:** N/A (this phase only reads).

**DO NOT:** modify any checkpoint or cache.

---

### PHASE 1 — Restore Qwen3 checkpoints if absent (≤ 22 GPU-hours, OPTIONAL)

**Goal:** Ensure both `outputs_dolly/qwen_0.6B/` and `outputs_dolly/qwen_1.7B/` contain
checkpoints for every method × seed cell needed by Table 1 of the paper.

**Prereqs:** PHASE 0.

**Required matrix (must exist at end of PHASE 1):**
- Students: `qwen_0.6B`, `qwen_1.7B`
- Methods: `sft, standard_kd, reverse_kl, seqkd, gkd, distillm, dakd, sagd`
- Seeds: `{42, 123, 456, 789, 2024}` (5 seeds, matching Table 1 caption)
- Total cells: 2 × 8 × 5 = **80** checkpoints

**Steps:**
1. From PHASE 0 inventory, list missing cells. If 0 missing, skip to acceptance.
2. If `data/teacher_saliency_dolly.pt` (Qwen3-8B) is missing, precompute first:
   ```
   python scripts/precompute_teacher_saliency.py \
       --model_name Qwen/Qwen3-8B --dataset dolly \
       --output_path data/teacher_saliency_dolly.pt \
       --batch_size 4 --max_seq_len 512 --device cuda:0
   ```
   ~2 hours on a free A100. Same for SQuAD cache if PHASE 6 will run on Qwen.
3. Create `scripts/a100_qwen_train.sh` modelled on `a100_llama_train.sh` if it does
   not exist; parameterise via env vars `QWEN_STUDENT`, `QWEN_OUT_DIR`, `METHOD`,
   `SEED`.
4. Schedule training across 4 A100s using the parallel-eval lockfile pattern.
   Approximate budget: 30 min / cell × 80 cells / 4 GPUs ≈ 10 hours.
5. After each cell finishes, run NaN check + eval ROUGE-L on Dolly test split. Persist
   per-cell metrics to `outputs_dolly/qwen_*/...seed_S/eval.json`.

**Deliverables:**
- All 80 checkpoints under `outputs_dolly/qwen_*/`
- `tmp/phase1_training_log.json` (per-cell wall time, final loss, ROUGE-L)

**Acceptance:**
- Every required `(student, method, seed)` checkpoint exists with `nan_check == "ok"`.
- `tmp/phase1_training_log.json` lists all 80 cells.

**If blocked:**
- GPU contention preventing parallel: serialise to 1 GPU. Document the slowdown and
  reduce seed count to {42, 123, 456} (3 seeds, 48 cells) with a note in
  `EXPERIMENTS_DONE.md`. Update Table 1 caption accordingly later in PHASE 9.
- Out-of-memory at batch 8: drop to batch 4 with grad_accum 8 (already supported).
- > 36 hour budget breach: stop after 24 hours; save whatever finished; mark PARTIAL.

**DO NOT:**
- Delete or overwrite existing checkpoints.
- Touch `outputs_dolly/llama_1B/`.
- Train with non-canonical hyperparameters (see §4 of `CLAUDE.md` for the canon: 10
  epochs, lr 1e-5, batch 8 × grad-accum 4, cosine 3% warmup, weight decay 0.01,
  max_seq_len 512, T=2, fp16, λ=0.5, σ=0.005, τ_w=1.0, N=5 for SaKD).

---

### PHASE 2 — Full Evidence-Concentration matrix (≤ 8 GPU-hours)

**Goal:** Replace the current 2-pair EC table (Qwen3-0.6B + LLaMA-1B) with the full
3-pair × 7-method × 3-seed matrix, with mean ± std for Qwen and single-seed for LLaMA.

**Prereqs:** PHASE 0 (caches); PHASE 1 (Qwen checkpoints, if any were missing).

**Required matrix:**
- Students: `qwen_0.6B`, `qwen_1.7B`, `llama_1B`
- Methods: KD-KL, KD-RKL, GKD, DistiLLM, DA-KD, SaKD (+ Teacher row)
- Seeds: Qwen {42, 123, 456}; LLaMA {42} only
- Dataset: SQuAD 2.0 validation (answerable subset, 500 samples)

**Steps:**
1. Use `scripts/diagnose_saliency.py`; extend to take `--methods`, `--seeds`,
   `--students` lists, and emit per-sample EC arrays + aggregated mean/std.
2. For each cell, compute mean EC; for cells with std (Qwen), aggregate across seeds.
3. Build `tmp/ec_per_sample_v2.json` keyed by
   `f"{student}|{method}|{seed}|{teacher_or_student}"` → `list[float]`.
4. Emit `tmp/ec_table_v2.tex` as a booktabs table with the 3-pair × 7-row layout.
   Bold the row closest to teacher per column.
5. Emit `tmp/ec_distribution_v2.pdf` — overlay teacher / KD-KL / SaKD EC distributions
   for the Qwen3-0.6B column (most informative single architecture).
6. Replace `writing/NeurIPS26-SaGD/tables/ec.tex` with the new layout (preserve label
   `tab:ec`). Update the §4.4 prose to reflect 3 architectures (current prose covers 2).

**Deliverables:**
- `tmp/ec_per_sample_v2.json`
- `tmp/ec_table_v2.tex`
- `tmp/ec_distribution_v2.pdf`
- Replaced `writing/NeurIPS26-SaGD/tables/ec.tex`
- Updated §4.4 "Results" paragraph in `writing/NeurIPS26-SaGD/sections/experiments.tex`

**Acceptance:**
- `tab:ec` has 3 columns (Qwen3-0.6B / Qwen3-1.7B / LLaMA-3.2-1B) × 7 rows
  (Teacher + 6 methods).
- Mean ± std present for Qwen rows; single value for LLaMA rows.
- SaKD row is the row closest to teacher in every column (sanity check); if not, do not
  silently rewrite — file an issue note in `tmp/PHASE2_NOTES.md`.
- Paper compiles with new `tab:ec`; no broken `\ref`.

**If blocked:**
- A method checkpoint produces NaN saliency: re-train with seed+1000, document.
- A method checkpoint missing for a specific seed: drop to single-seed for that row;
  add asterisk in caption.

**DO NOT:**
- Touch the LLaMA-1B EC values from the prior run unless re-computing produces a
  different result; if it does, prefer the fresh value and note the delta.
- Change SQuAD 2.0 evaluation parameters (still answerable subset, 500 val samples,
  fast tokenizer with `return_offsets_mapping=True`).

---

### PHASE 3 — Hyperparameter sensitivity (≤ 12 GPU-hours)

**Goal:** Produce one new table and one new figure showing SaKD's robustness to its
four hyperparameters (λ, σ, τ_w, N) on Qwen3-0.6B.

**Prereqs:** PHASE 1 (Qwen3-0.6B SaKD checkpoint at default settings).

**Sweep (4 axes, 3 values each; one axis varies, others fixed at default):**
- λ ∈ {0.1, 0.5, 2.0} (default 0.5)
- σ ∈ {0.001, 0.005, 0.02} (default 0.005)
- τ_w ∈ {0.5, 1.0, 5.0} (default 1.0)
- N ∈ {1, 5, 20} (default 5)

Total: 4 × 3 = 12 SaKD training runs, but the default cell (λ=0.5, σ=0.005, τ_w=1.0,
N=5) is shared across all four sweeps — reuse the existing checkpoint, so 4 × 2 = 8
new training runs.

**Steps:**
1. For each non-default cell, train SaKD with the canonical recipe (PHASE 1 settings),
   single seed (42). Persist `outputs_dolly/qwen_0.6B/sagd_sweep/<axis>_<value>/seed_42/`.
2. Eval each on Dolly test split: DollyEval, S-NatInst, Unnatural, Avg ROUGE-L.
3. Persist per-cell metrics to `tmp/hp_sensitivity.json`.
4. Emit `tmp/hp_sensitivity.tex` — a 4-subtable booktabs layout (one sub-table per
   axis), columns = DollyEval / S-NatInst / Unnatural / Avg, rows = the three values.
   Mark the default cell.
5. Emit `tmp/hp_sensitivity.pdf` — 1×4 panel line plot, x = parameter value, y = Avg
   ROUGE-L, default marked with a star, baseline KD-KL Avg as a horizontal dashed
   reference line.
6. Add a new appendix subsection `\subsection{Hyperparameter Sensitivity}` with label
   `app:hp-sensitivity` that `\input`s `tmp/hp_sensitivity.tex` (move to
   `writing/NeurIPS26-SaGD/tables/hp_sensitivity.tex`) and a one-paragraph discussion:
   SaKD's Avg ROUGE-L stays within ±0.3 of the default across all four sweeps; the
   most sensitive axis is X, the least is Y.
7. Add `\ref{app:hp-sensitivity}` from §4.1 or end of §4.3.

**Deliverables:**
- `outputs_dolly/qwen_0.6B/sagd_sweep/<axis>_<value>/seed_42/student_final.pt` (8 new)
- `tmp/hp_sensitivity.json`
- `writing/NeurIPS26-SaGD/tables/hp_sensitivity.tex`
- `writing/NeurIPS26-SaGD/figures/hp_sensitivity.pdf` (copy from tmp)
- New appendix subsection in `writing/NeurIPS26-SaGD/sections/appendix.tex`

**Acceptance:**
- 8 new SaKD checkpoints exist with NaN-clean weights.
- Appendix renders the new table and figure; both labels resolve.
- Paragraph identifies the most and least sensitive axis with concrete numbers.

**If blocked:**
- > 12 GPU-hour budget: sweep λ and σ only (most theoretically interesting), defer τ_w
  and N. Document in `tmp/PHASE3_NOTES.md`.

**DO NOT:**
- Sweep on Qwen3-1.7B or LLaMA (out of compute scope).
- Sweep multiple axes simultaneously (one-at-a-time only).

---

### PHASE 4 — Training dynamics (≤ 3 GPU-hours)

**Goal:** Produce one figure showing convergence behaviour of KD-KL vs SaKD over
training, on Qwen3-0.6B seed 42.

**Prereqs:** PHASE 1 (Qwen3-0.6B Qwen pipeline working).

**Steps:**
1. Re-train KD-KL and SaKD on Qwen3-0.6B seed 42 with per-step logging every 50
   optimiser steps. Log:
   - train loss
   - val loss on Dolly val (200-sample fixed subset for speed)
   - mean saliency divergence on the same fixed val subset (only for SaKD; for KD-KL,
     compute the same metric to show it does *not* shrink)
2. Persist to `tmp/training_dynamics.json` with keys
   `[{method, step, train_loss, val_loss, sal_div}]`.
3. Emit `tmp/training_dynamics.pdf` — two-panel figure:
   - (a) train + val loss curves, KD-KL vs SaKD
   - (b) saliency-divergence over training, KD-KL vs SaKD (SaKD should drop faster)
4. Add a one-paragraph subsection `\subsection{Training Dynamics}` to the appendix
   (label `app:training-dynamics`); `\input` the figure (move PDF to
   `writing/NeurIPS26-SaGD/sources/training_dynamics.pdf`).
5. Reference from §4.3 ablation discussion: "see Appendix~\ref{app:training-dynamics}
   for convergence curves."

**Deliverables:**
- `tmp/training_dynamics.json`
- `writing/NeurIPS26-SaGD/sources/training_dynamics.pdf`
- New appendix subsection.

**Acceptance:**
- The two-panel PDF compiles into the paper.
- The val-loss curves are monotonic-downward (sanity: no training divergence).
- Saliency-divergence curve for SaKD ends below KD-KL's; if not, file note in
  `tmp/PHASE4_NOTES.md` (do not invent results).

**If blocked:**
- Re-training takes too long: use the existing PHASE 1 checkpoints' periodic
  intermediate saves if they exist; otherwise drop val-loss panel and ship only the
  saliency-divergence panel.

**DO NOT:**
- Hand-fit a curve. Plot the raw measurements only.

---

### PHASE 5 — Benchmark defense (≤ 6 GPU-hours)

**Goal:** Show that SaKD does not regress on general capability benchmarks vs the
KD-KL baseline. One new appendix table.

**Prereqs:** PHASE 1.

**Eval matrix:**
- Models: Qwen3-0.6B {KD-KL, SaKD}, Qwen3-1.7B {KD-KL, SaKD} (4 ckpts, seed 42)
- Benchmarks: MMLU (5-shot), ARC-Challenge (25-shot), TruthfulQA (0-shot, mc2)
- Tool: `lm-eval-harness`

**Steps:**
1. Run `lm-eval` for each (model, benchmark) cell. Persist raw outputs to
   `tmp/benchmark_defense_raw/<model>_<benchmark>.json`.
2. Aggregate accuracy / mc2 into `tmp/benchmark_defense.json`.
3. Emit `tmp/benchmark_defense.tex` — booktabs table, columns MMLU / ARC-C /
   TruthfulQA / Avg, rows {KD-KL, SaKD} × {0.6B, 1.7B} (4 rows).
4. Add appendix subsection `\subsection{General-Capability Defence}` (label
   `app:benchmark-defense`); `\input` the table.
5. Reference from §5 Limitations: "SaKD does not regress on general capability
   benchmarks (Appendix~\ref{app:benchmark-defense})."

**Deliverables:**
- `tmp/benchmark_defense_raw/` (4 × 3 = 12 JSON files)
- `tmp/benchmark_defense.json`
- `writing/NeurIPS26-SaGD/tables/benchmark_defense.tex`
- New appendix subsection.

**Acceptance:**
- SaKD's Avg ≥ KD-KL's Avg − 0.5 absolute on at least one student. If SaKD strictly
  regresses everywhere, write a candid note in `tmp/PHASE5_NOTES.md` and adjust the
  limitations section to acknowledge this — do not hide.

**If blocked:**
- TruthfulQA dataset download fails: skip TruthfulQA only, ship MMLU + ARC-C; document.

**DO NOT:**
- Cherry-pick benchmarks. Report every cell that runs.

---

### PHASE 6 — Qualitative saliency heatmap (Qwen) (≤ 1 GPU-hour)

**Goal:** Replace `tmp/saliency_heatmap.pdf` (LLaMA) with a Qwen3-0.6B version that
matches §4.4's primary architecture.

**Prereqs:** PHASE 1; SQuAD teacher saliency cache for Qwen3-8B.

**Steps:**
1. Pick 2 SQuAD val samples: one short context (≤ 200 tokens), one long context
   (≥ 400 tokens). Seed = 42.
2. For each, compute saliency arrays for Teacher (Qwen3-8B), KD-KL student
   (Qwen3-0.6B), SaKD student (Qwen3-0.6B).
3. Plot a 2-row × 3-col heatmap. Red bar = answer span. Use a perceptually uniform
   colormap (viridis) and consistent scale across panels.
4. Save `tmp/saliency_heatmap_qwen.pdf` + `tmp/saliency_heatmap_qwen.json` (raw arrays).
5. Move `tmp/saliency_heatmap_qwen.pdf` → `writing/NeurIPS26-SaGD/sources/`.
6. Add a `\begin{figure}` block in §4.4 or in the appendix (whichever fits the page
   budget) with label `fig:saliency_heatmap`. Caption: which 2 samples were chosen,
   teacher / KD-KL / SaKD comparison, what to look for (KD-KL concentrates on red bar,
   SaKD spreads).

**Deliverables:**
- `writing/NeurIPS26-SaGD/sources/saliency_heatmap_qwen.pdf`
- `tmp/saliency_heatmap_qwen.json`
- New figure block in `experiments.tex` or `appendix.tex`.

**Acceptance:**
- Figure renders; `fig:saliency_heatmap` resolves.
- KD-KL panel visibly more concentrated on the answer span than SaKD panel.

**If blocked:**
- Teacher saliency cache for SQuAD missing: precompute it as PHASE 1 step 2.

**DO NOT:**
- Pick samples post-hoc to make SaKD look good. Use sample indices 0 (short) and 69
  (long) as the previous LLaMA heatmap did.

---

### PHASE 7 — Fix Figure 1 left panel "shaded region" mismatch (≤ 30 min)

**Goal:** Either (a) redraw `sources/motivation_M1.pdf` with an actual shaded
rectangle around the failure cluster, or (b) reword the caption to match what's
visible.

**Prereqs:** none (this is a writing/figure fix, not an experiment).

**Decision rule:** Open `writing/NeurIPS26-SaGD/sources/motivation_M1.pdf`. If there
is any visible shaded / boxed region, choose (b) and edit the caption to match. If
there is no shaded region, choose (a) and redraw the figure.

**For (a) redraw:**
1. Use the existing scatter data in `tmp/motivation/m1_data.json` if present (from
   the original M1 generation run); otherwise re-run the M1 generation script.
2. Overlay a translucent shaded rectangle covering `clean_KL ≤ 25th-percentile AND
   perturbed_KL ≥ 75th-percentile` (the failure cluster).
3. Save to `tmp/motivation_M1_v2.pdf`, then move to
   `writing/NeurIPS26-SaGD/sources/motivation_M1.pdf` (overwrite).

**For (b) reword:** Open `writing/NeurIPS26-SaGD/sections/introduction.tex`, locate
the Figure 1 caption, replace "the shaded region" with a phrase matching what is
visible (e.g. "the upper-left cluster" or "points above the dashed line" — match the
actual panel).

**Deliverables:**
- Either updated `sources/motivation_M1.pdf` or updated caption in `introduction.tex`.

**Acceptance:**
- The Figure 1 left panel's visible markup matches the caption text exactly.

**DO NOT:**
- Alter the actual data points. Only the markup / annotation.

---

### PHASE 8 — Final manuscript polish (≤ 1 hour)

**Goal:** Final cleanup of writing-side hygiene.

**Prereqs:** PHASES 2, 3, 5, 6, 7 done so labels are stable.

**Steps:**
1. **Bib orphan scrub:** for each entry in `references.bib`, grep for `\cite[a-z]*{<key>`
   across `sections/`, `tables/`, `figures/`, `algorithms/`, `checklist.tex`. Delete
   entries with zero hits. Persist deleted keys to `tmp/PHASE8_bib_orphans.md`.
2. **Reviewer-note scrub:** `grep -rE '\\(GH|CC|cc)\{|% TODO' writing/NeurIPS26-SaGD/`.
   For each hit in a *rendered* line, remove the annotation. Comment-only `% \GH{...}`
   inside dead blocks of `method.tex` may stay (they don't render); flag them in
   `tmp/PHASE8_notes.md`.
3. **Em-dash audit (`---` in rendered prose):** grep `'---'` in `sections/`. For each
   hit, rewrite to use commas, semicolons, or parentheses (memory rule). Skip
   `---` inside `\verb`, `\caption{}` of a table row, or already-correct math context.
4. **Italics / bold scrub:** `grep -rE '\\(emph|textit)\{' writing/NeurIPS26-SaGD/`.
   Remove or convert per the user's preference (no random italics; `\textbf{}` only
   for short paragraph-leading labels like `\textbf{(i)}`).
5. **Widow re-check:** for each paragraph in `sections/`, if the last sentence ends
   with ≤ 2 short words, extend with 3–5 words while preserving meaning. (PHASE 8's
   acceptance criterion does not require this — only flag in `tmp/PHASE8_widow.md` if
   more than 5 paragraphs are at risk.)

**Deliverables:**
- `tmp/PHASE8_bib_orphans.md` (list of deleted keys)
- `tmp/PHASE8_notes.md`
- Updated `references.bib`, `sections/*.tex`.

**Acceptance:**
- `grep -rE '\\(GH|CC|cc)\{' writing/NeurIPS26-SaGD/sections/` returns 0 hits in
  rendered text.
- `grep -rE '\\(emph|textit)\{' writing/NeurIPS26-SaGD/sections/` returns 0 hits.
- `references.bib` orphan count = 0 (verify with the same grep loop).
- Paper still compiles after all cleanups.

**DO NOT:**
- Rewrite full paragraphs. Only the surgical cleanup actions listed.

---

### PHASE 9 — Page budget audit + recovery (≤ 1 hour, mostly LaTeX)

**Goal:** Get the main text to ≤ 9 pages without losing substantive content. NeurIPS
hard limit; appendix / references / checklist do not count.

**Prereqs:** PHASES 2, 3, 5, 6 done (final content settled).

**Tactic ladder (apply in order, stop when ≤ 9 pages):**
1. **Move PHASE 3 sensitivity discussion to appendix only** (do not duplicate the
   one-paragraph summary in main text).
2. **Hide `tables/dolly_main_t.tex`** (the transposed view; comment out the
   `\input{tables/dolly_main_t}` line). Keep `tables/dolly_main.tex`.
3. **Move Algorithm 1 from `method.tex` §3.4 back to `appendix.tex`**. Replace the
   in-text algorithm with a one-line "see Algorithm~\ref{alg:sakd} in
   Appendix~\ref{app:algorithms}" pointer. (This was previously done; reverse the
   recent in-text placement if needed.)
4. **Drop the `\paragraph{Datasets.}` block** in §4.1 if step 3 wasn't enough. The
   essential content lives in Table 1 caption + §4.4.
5. **Tighten Figure 1 caption** to 3 lines; tighten Figure 2 caption to 3 lines.
6. **Remove the abstract's "Concretely, SaKD combines..." sentence** if step 5 wasn't
   enough.

Document which tactics were applied to get under 9 pages in `tmp/PAGE_BUDGET.md`.

**Steps:**
1. Compile the paper: `cd writing/NeurIPS26-SaGD && pdflatex -interaction=nonstopmode
   neurips_2026.tex; bibtex neurips_2026; pdflatex neurips_2026; pdflatex neurips_2026`.
2. Count main-text pages via `pdftk neurips_2026.pdf dump_data | grep NumberOfPages`
   minus appendix / references / checklist pages.
3. If > 9 pages, apply tactics 1–6 in order until ≤ 9.
4. After every tactic application, re-compile and recount.

**Deliverables:**
- `tmp/PAGE_BUDGET.md` (which tactics applied; final page count)
- Updated paper source.

**Acceptance:**
- Main text page count ≤ 9 (or, if all 6 tactics applied and still > 9, a candid
  `tmp/PAGE_BUDGET.md` documenting what was prioritised and what was sacrificed).

**DO NOT:**
- Reduce font size, line spacing, or NeurIPS-prescribed margins.
- Drop a real result (Table 1, Table 3, ablation, EC, training dynamics, benchmark
  defense). These are non-negotiable.

---

### PHASE 10 — Final compile, verification, and bundle (≤ 30 min)

**Goal:** Produce a clean PDF + a written summary the human can pull and ship.

**Prereqs:** all previous phases.

**Steps:**
1. Final compile (4 passes): pdflatex → bibtex → pdflatex → pdflatex.
2. Verify zero `Undefined reference`, zero `Citation undefined`, zero `Overfull`
   warnings of severity ≥ 10pt. Persist build log to `tmp/PHASE10_build.log`.
3. Generate `tmp/PAPER_COMPLETION_DONE.md` with:
   - For each PHASE: status (DONE / PARTIAL / FAILED), wall time, deliverables,
     headline finding.
   - A "What changed in the paper" cheat-sheet: list every section / table / figure
     touched, with one line per change.
   - A "next steps for human" subsection: anything that's still PARTIAL.
4. `git add . && git commit -m "Paper completion run: PHASES 0-10" && git push` for
   both the main repo and `writing/NeurIPS26-SaGD/` submodule (rebase first).

**Deliverables:**
- `writing/NeurIPS26-SaGD/neurips_2026.pdf` (final)
- `tmp/PHASE10_build.log`
- `tmp/PAPER_COMPLETION_DONE.md`

**Acceptance:**
- A2.0 – A2.6 (top of this document) all check.

**DO NOT:**
- Push to a tag, branch, or remote other than `origin master` of the two repos.
- Open a PR — the user will review the pushed master directly.

---

## 4. Global failure-handling rules

- Any unrecoverable failure (e.g. dataset corruption, repo permission lost,
  hardware unavailable) → write a 1-page SOS file `tmp/PAPER_COMPLETION_SOS.md`
  with traceback + which phase + what was attempted + recommended human action.
  Then stop.
- Any minor / recoverable failure → write a `tmp/PHASE{N}_NOTES.md` row and continue.
- A run that hits the 36-hour total budget → finalise PHASE 10 with whatever state
  exists; mark un-run phases as TODO in `PAPER_COMPLETION_DONE.md`.

---

## 5. What is explicitly out of scope

- Adding new methods or new theorems.
- Re-running on datasets other than Dolly-15K (training) / SQuAD 2.0 (EC) / the three
  Dolly-derived held-out benchmarks / the three general-capability benchmarks.
- Adding a new related-work entry (other than to fix a missing citation discovered in
  PHASE 8).
- Changing the paper title, author block, abstract claims, or method name (SaKD).
- Re-rendering `sources/framework.pdf` (user already finalised this).

---

## 6. How the human will consume your output

The human will, after notification:

1. Read `tmp/PAPER_COMPLETION_DONE.md` first.
2. Spot-check `writing/NeurIPS26-SaGD/neurips_2026.pdf`.
3. For each PARTIAL phase, decide whether to accept or open a follow-up task.
4. Pull from Overleaf and re-publish.

Your job ends when the human can do these four steps without needing to ask you any
clarifying question.
