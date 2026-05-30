# Task: Bring SaKD NeurIPS 2026 Paper to Camera-Ready Quality (Full Retrain)

**Status:** Submitted version is in `writing/NeurIPS26-SaGD/`, but **all empirical
numbers in the paper are to be re-derived from a fresh training run on this
codebase**. The code in `src/sagd/` has been audited line-by-line against the paper
(see "Code consistency" section below) and is confirmed faithful; therefore the
right action is to retrain everything from scratch, then re-do every downstream
table / figure / EC / sweep / dynamics / defense.

**Owner:** Autonomous server-side Claude Code agent (the "worker"). Execute via
`/goal docs/paper_completion_task.md`. A separate monitor agent
(`docs/paper_completion_monitor.md`) watches this worker; you can ignore the
monitor — it never modifies your work.

**Hard time budget:** ≤ 96 hours wall-clock (~4 days). This is a long-running job by
design; the monitor agent will detect stalls and restart you if needed.

**Authorization:**
- **Writable**: `tmp/`, `writing/NeurIPS26-SaGD/`, `scripts/`, `outputs/`,
  `outputs_dolly/`, `outputs_squad/`, `data/teacher_saliency_*.pt`, this `docs/`
  directory.
- **Read-only**: `src/sagd/` (the audited code — do not modify; if a bug is
  discovered, halt and write `tmp/PAPER_COMPLETION_SOS.md` instead of patching).
- **Off-limits**: `.git/`, `.agents/`, `CLAUDE.md`, `tests/` (do not extend tests),
  `README.md` (the README is human-curated).

**Operating constraints**
- Run with `--dangerously-skip-permissions`. Do NOT prompt the human for
  confirmation.
- Run inside a tmux session named `sakd_finish` so the monitor and human can
  `tmux attach` to inspect.
- Append a one-line status to `tmp/exp_progress.log` whenever a phase
  starts/ends, a sub-job finishes, or every 15 minutes during long jobs (so
  monitor can detect stalls). Format: `HH:MM:SS PHASE-N <event>`. The monitor
  considers > 30 minutes of silence a stall.
- All deliverable artefacts go under `tmp/` with stable filenames listed per
  phase.
- LaTeX-side edits go under `writing/NeurIPS26-SaGD/` and are pushed to Overleaf
  (`origin master`) after every commit. Always `git stash && git pull --rebase &&
  git stash pop` before staging to handle concurrent Overleaf-web edits.
- 4× A100 80GB available. Reuse `scripts/a100_parallel_eval.sh`'s lockfile
  pattern to avoid GPU contention. Persist GPU-to-cell assignment to
  `tmp/gpu_assignment.json` so the monitor can correlate `nvidia-smi` output
  with which cell each GPU is running.

---

## 0. Code consistency status (already verified — do not re-audit)

A line-by-line audit on **2026-05-30** confirmed all 23 method-side claims from
the paper (CLAUDE.md §2) are faithfully implemented:

- Loss formula `(1/B) Σ w_i (KL_clean_i + λ KL_noise_i)` — `trainer.py:538-540` ✓
- Non-SaKD steps use only clean KL — `trainer.py:507,616-617` ✓
- Weights detached before multiplication — `trainer.py:538` ✓
- Weights = `softmax(JSD/τ_w) * B`, mean-normalised — `trainer.py:535` ✓
- Shift-aligned per-sample KL — `trainer.py:256-270` ✓
- Both teacher & student forward on noised embeddings — `trainer.py:347-362` ✓
- Embedding lookup detached before noise add — `trainer.py:337-338,349` ✓
- Adaptive σ_j per Eq. with [δ, 5×] clamp — `trainer.py:308-313` ✓
- Independent ξ_T, ξ_S at shared σ_j — `trainer.py:347,349` ✓
- Saliency: param.requires_grad save/restore — `saliency.py:48-50` ✓
- Saliency mask uses `(1 - labels_mask) * attention_mask` — `saliency.py:82` ✓
- Saliency shift convention — `saliency.py:63-65` ✓
- Saliency output pre-masked — `saliency.py:83,90` ✓
- Teacher saliency cache format & metadata — `precompute_teacher_saliency.py:139-148` ✓
- Cache by dataset index — `trainer.py:272-292` ✓
- METHODS dict {sft, standard_kd, reverse_kl, seqkd, gkd, distillm, dakd, sagd} —
  `trainer.py:38-47` ✓
- Non-sagd methods don't initialise SaKD components — `trainer.py:146-152` ✓
- Dolly splits: shuffle(seed=42), train/val/test — `data.py:128,163-175` ✓
- SquadDataset: answerable, returns answer span — `data.py:279,354-360` ✓
- collate_fn handles index + answer span — `data.py:1028,1032-1038` ✓
- Teacher frozen in eval() + no_grad — `models.py:72-74`, `trainer.py:478-482` ✓

**Implication:** training results produced now are method-faithful. You do not
need to re-validate the code; assume PASS and proceed.

If during a training run you observe behaviour that contradicts the audit (e.g. a
loss explosion that the code shouldn't permit, or a saliency NaN under conditions
the audit said were safe), STOP, write `tmp/PAPER_COMPLETION_SOS.md` with the
divergence, and wait for human intervention. Do NOT silently patch `src/sagd/`.

---

## 1. Paper structure (read once for label-ref hygiene)

```
writing/NeurIPS26-SaGD/
├── neurips_2026.tex         # root; abstract is inline; \input{sections/...}
├── sections/
│   ├── introduction.tex     # Fig 1 motivation
│   ├── preliminary.tex
│   ├── method.tex           # 3 theorems + Algorithm 1
│   ├── background.tex
│   ├── experiments.tex      # §4.1 setup, §4.2 main, §4.3 ablation, §4.4 EC
│   ├── conclusion.tex       # §5 Conclusion + Outlook + §6 Limitations
│   └── appendix.tex         # proofs, noise-coupling, implementation, compute cost
├── tables/                  # dolly_main(.tex / _t.tex), ablation, ec, compute_cost
├── figures/                 # framework, ablation (hidden bar chart)
├── algorithms/sakd.tex      # Algorithm 1, included in §3.4
├── sources/                 # motivation_M{1,2}.pdf, framework.pdf
├── checklist.tex
└── references.bib
```

The submitted version's NUMBERS will be REPLACED by this run's fresh results.
The submitted version's STRUCTURE (sections, tables, figures, labels) is the
target — preserve all `\label{}`s.

---

## 2. Acceptance summary (the bar for "done")

The work is **complete** when **all** of the following are true:

- A2.0: Every phase has a `tmp/EXPERIMENTS_DONE.md` row with status DONE or
  PARTIAL (PARTIAL requires a 1-paragraph justification).
- A2.1: `writing/NeurIPS26-SaGD/main.pdf` compiles cleanly (zero `??` refs, zero
  undefined citations, zero LaTeX errors) and is pushed to Overleaf `master`.
- A2.2: Main text ≤ 9 pages (excluding refs / appendix / checklist); PHASE 9
  tactic ladder documents which tactics applied.
- A2.3: Tables `tab:dolly`, `tab:ablation`, `tab:ec`, `tab:compute-cost` all
  reflect numbers reproducible from `outputs_dolly/` + `outputs_squad/` + the
  eval scripts in `scripts/`. **No legacy numbers from the submitted version
  remain in any table.**
- A2.4: New `tab:hp_sensitivity`, `fig:training_dynamics`,
  `fig:saliency_heatmap_qwen`, `tab:benchmark_defense` exist and are referenced.
- A2.5: All `\GH{...}` / `\CC{...}` reviewer notes and rendered `% TODO` removed.
- A2.6: `tmp/PAPER_COMPLETION_DONE.md` itemises every artefact and every paper
  section/table/figure touched, with a 1-line diff per item, plus a "checkpoint
  manifest" listing every trained ckpt with `(student, method, seed, ckpt path,
  ROUGE-L Avg, ec, train-wall-time)`.

---

## 3. Resource matrix (verify in PHASE 0)

| Asset | Expected location | Must exist before |
|---|---|---|
| Teacher saliency caches | `data/teacher_saliency_{dolly_qwen,dolly_llama,squad_qwen,squad_llama}.pt` | PHASE 2 (Qwen Dolly cache), PHASE 3 (EC needs SQuAD caches), PHASE 6 (sweep), PHASE 7 (dynamics), PHASE 8 (heatmap). PHASE 1 creates them all. |
| Trained student checkpoints | `outputs_dolly/{qwen_0.6B,qwen_1.7B,llama_1B}/<method>/seed_<S>/student_final.pt` | every eval phase. PHASE 2 creates them all. |
| Training scripts | `scripts/a100_qwen_train.sh`, `scripts/a100_llama_train.sh`, `scripts/a100_parallel_eval.sh`, `scripts/train.py` | PHASE 2, PHASE 6, PHASE 7. Create `a100_qwen_train.sh` if missing (model on `a100_llama_train.sh`). |
| Diagnosis script | `scripts/diagnose_saliency.py` | PHASE 3, PHASE 8 |
| lm-eval-harness | installed; cache at `~/.cache/lm-eval` | PHASE 5 |

---

## 4. Phases

Each phase: **Goal / Prereqs / Steps / Deliverables / Acceptance / If blocked / DO NOT**.

---

### PHASE 0 — Inventory & sanity (≤ 20 min)

**Goal:** Snapshot current disk state so PHASE 1/2 know what's missing.

**Prereqs:** none.

**Steps:**
1. `tmp/phase0_inventory.json` — every `(student, method, seed)` under
   `outputs_dolly/` with `(path, mtime, size, nan_check)`.
2. `tmp/phase0_caches.json` — for each teacher saliency cache in `data/`,
   record `n_samples, max_seq_len, model, dataset`.
3. `tmp/phase0_disk.json` — `df -h .` parsed; assert ≥ 800 GB free (88 ckpts
   × ~5 GB + caches + logs ≈ 500 GB; double for safety).
4. `tmp/phase0_gpu.json` — `nvidia-smi --query-gpu=index,memory.free,name
   --format=csv,noheader`; assert 4 A100 80GB.
5. `pip show lm-eval` to confirm install; write to `tmp/phase0_envcheck.json`.
6. Append PHASE 0 row to `tmp/EXPERIMENTS_DONE.md`.

**Deliverables:** the 5 json files above + `EXPERIMENTS_DONE.md` row.

**Acceptance:** all 5 json files exist; `EXPERIMENTS_DONE.md` PHASE 0 row.

**If blocked:** if `nvidia-smi` shows < 4 GPUs free, wait 10 min and re-try
once. If still < 4, document in NOTES and proceed with what's available — the
training loop already serialises gracefully.

**DO NOT:** modify any existing checkpoint or cache.

---

### PHASE 1 — Precompute teacher saliency caches (≤ 8 GPU-hours)

**Goal:** Have all 4 teacher saliency caches on disk before any student training
(every cell of SaKD training needs the cache).

**Prereqs:** PHASE 0.

**Steps (run 4 caches in parallel across 4 GPUs):**
1. Qwen3-8B on Dolly-15K → `data/teacher_saliency_dolly_qwen.pt`
2. Qwen3-8B on SQuAD 2.0 → `data/teacher_saliency_squad_qwen.pt`
3. LLaMA-3.1-8B on Dolly-15K → `data/teacher_saliency_dolly_llama.pt` (may exist
   from prior run — verify and reuse)
4. LLaMA-3.1-8B on SQuAD 2.0 → `data/teacher_saliency_squad_llama.pt`

Use `scripts/precompute_teacher_saliency.py` with `--batch_size 4
--max_seq_len 512`. For cross-arch caches, pass `--tokenizer_name` pointing to
the STUDENT model (Qwen3-0.6B for the Qwen caches, LLaMA-3.2-1B for the LLaMA
caches) per CLAUDE.md §5.7.

**Deliverables:**
- 4 cache files in `data/`
- `tmp/phase1_caches.json` (per-cache metadata: tokeniser, n_samples,
  max_seq_len, wall time, model, dataset)

**Acceptance:**
- Each cache loads successfully on CPU and has `metadata.n_samples >= 13K` for
  Dolly, `>= 85K` for SQuAD.
- Tokeniser identity in metadata matches the intended STUDENT model for that
  pair.

**If blocked:** OOM at batch 4 → drop to batch 2 for that cache only and
document the slowdown.

**DO NOT:** reuse a cache whose tokeniser does not match its target student
(silent corruption of `index → saliency` mapping per CLAUDE.md §5.7).

---

### PHASE 2 — Train ALL student checkpoints from scratch (≤ 36 GPU-hours)

**Goal:** Fresh checkpoints for every cell of Table 1 (`tab:dolly`). The
submitted version's numbers are to be replaced entirely.

**Prereqs:** PHASE 1.

**Required matrix:**
- Students: `qwen_0.6B`, `qwen_1.7B`, `llama_1B`
- Methods: `sft, standard_kd, reverse_kl, seqkd, gkd, distillm, dakd, sagd` (8)
- Seeds: Qwen × {42, 123, 456, 789, 2024}; LLaMA × {42} only (single seed per
  the submitted Table 1 caption)
- Total cells: (2 × 8 × 5) + (1 × 8 × 1) = **88 checkpoints**

**Canonical hyperparameters (do not deviate):** 10 epochs, lr 1e-5 with cosine
3% warmup, batch 8 × grad-accum 4, weight decay 0.01, max_seq_len 512, KL
temperature T=2, fp16. For SaKD: λ=0.5, σ=0.005 (relative to embedding norm),
τ_w=1.0, N=5. See CLAUDE.md §4.

**Steps:**
1. Create `scripts/a100_qwen_train.sh` modelled on `a100_llama_train.sh` if it
   does not exist. Parameterise via env vars `STUDENT, METHOD, SEED, OUT_DIR`.
2. Build the queue of 88 cells. Persist queue to `tmp/phase2_queue.json` with
   columns `(student, method, seed, status, gpu, started_at, finished_at,
   wall_time_s, eval_avg_rougeL, nan_check)`. Status ∈
   {PENDING, RUNNING, DONE, FAILED, RETRIED}.
3. Run queue across 4 GPUs using `scripts/a100_parallel_eval.sh`'s lockfile
   pattern. Each finished cell:
   - Runs NaN check on `student_final.pt`
   - Runs ROUGE-L eval on the three Dolly held-out benchmarks (DollyEval,
     S-NatInst, Unnatural)
   - Persists per-cell metrics to
     `outputs_dolly/<student>/<method>/seed_<S>/eval.json`
   - Updates `tmp/phase2_queue.json` row
4. Approximate budget: 30 min/cell × 88 cells / 4 GPUs ≈ 11 hours. Plus eval
   ~5 min/cell × 88 / 4 ≈ 1.5 hours. Total ~13 GPU-hours wall-clock.
5. After all DONE, aggregate per-cell `eval.json` into the four metrics tables:
   - `tmp/phase2_table_qwen17b.json` (8 methods × 5 seeds, mean ± std)
   - `tmp/phase2_table_qwen06b.json`
   - `tmp/phase2_table_llama1b.json` (8 methods × 1 seed)
   - `tmp/phase2_table_dolly_combined.tex` — the booktabs version of `tab:dolly`
   - Move into `writing/NeurIPS26-SaGD/tables/dolly_main.tex`, preserving
     `\label{tab:dolly}`.
6. Re-write the §4.2 analysis prose to reflect the actual numbers. The
   submitted version's narrative claims (SaKD wins everywhere, S-NatInst is
   biggest gain, variance lower than baselines, cross-arch margin narrower)
   may NOT all hold on the fresh run. **Report what the numbers say, do not
   adapt the numbers to the narrative.**

**Deliverables:**
- 88 fresh checkpoints in `outputs_dolly/`
- 88 `eval.json` files
- `tmp/phase2_queue.json` (final state, all DONE/FAILED)
- `writing/NeurIPS26-SaGD/tables/dolly_main.tex` (replaced)
- Updated `writing/NeurIPS26-SaGD/sections/experiments.tex` §4.2 prose
- `tmp/phase2_narrative_diff.md` — bullet list of which narrative claims still
  hold vs. which had to change

**Acceptance:**
- 88/88 cells in DONE state (or DONE + a documented set of FAILED with
  retry-and-failed reasoning).
- New `tab:dolly` compiles; SaKD row present; all `\ref{tab:dolly}` callers
  still resolve.
- §4.2 prose's headline claims match the new numbers (no contradictions).

**If blocked:**
- OOM at batch 8 → drop to batch 4 with grad_accum 8 for that cell.
- A cell produces NaN at the end of training → retry with seed+1000, document
  the swap in `tmp/phase2_narrative_diff.md`.
- Budget overrun (> 36 GPU-hours) → finish whatever cells are RUNNING; mark
  remaining PENDING cells as PARTIAL in `EXPERIMENTS_DONE.md`; downgrade Table
  1 caption to "3 seeds" if only 3 of the 5 Qwen seeds completed.

**DO NOT:**
- Touch `src/sagd/` code.
- Use different hyperparameters per cell (the only knob that varies is `seed`).
- Skip the NaN check.
- Adapt training numbers to fit narrative.

---

### PHASE 3 — Evidence-Concentration matrix (≤ 8 GPU-hours)

**Goal:** Full 3-pair × 7-method × 3-seed EC table to replace
`tables/ec.tex`.

**Prereqs:** PHASE 1 (SQuAD caches for both architectures), PHASE 2 (all 88
checkpoints).

**Required matrix:**
- Pairs: Qwen3-0.6B, Qwen3-1.7B, LLaMA-3.2-1B
- Methods: KD-KL, KD-RKL, GKD, DistiLLM, DA-KD, SaKD (+ Teacher row)
- Seeds: Qwen {42, 123, 456}; LLaMA {42}
- Dataset: SQuAD 2.0 val, answerable subset, 500 samples

**Steps:**
1. Extend `scripts/diagnose_saliency.py` (if not already) to take `--methods`,
   `--seeds`, `--students` lists and emit per-sample EC arrays.
2. For each cell, compute mean EC; aggregate seeds for Qwen rows (mean ± std).
3. `tmp/ec_per_sample_v2.json` keyed
   `"<student>|<method>|<seed>|<teacher_or_student>"` → list[float].
4. `tmp/ec_table_v2.tex` booktabs layout; bold row closest to teacher per
   column.
5. `tmp/ec_distribution_v2.pdf` — teacher/KD-KL/SaKD EC distribution overlay
   for Qwen3-0.6B.
6. Replace `writing/NeurIPS26-SaGD/tables/ec.tex` (preserve label `tab:ec`).
7. Update §4.4 "Results" paragraph to reflect 3 architectures.

**Deliverables:**
- `tmp/ec_per_sample_v2.json`
- `writing/NeurIPS26-SaGD/tables/ec.tex` (replaced)
- `writing/NeurIPS26-SaGD/sources/ec_distribution.pdf` (new)
- Updated §4.4 prose in `experiments.tex`

**Acceptance:**
- `tab:ec` is 3 cols × 7 rows; std present for Qwen.
- SaKD is the row closest to teacher in **at least 2 of 3** columns. If SaKD is
  not closest in any column, do not silently rewrite the narrative — file
  `tmp/PHASE3_NOTES.md` flagging the regression.
- Paper compiles.

**DO NOT:** change SQuAD eval parameters (still answerable subset, 500 val
samples, fast tokeniser with `return_offsets_mapping=True`).

---

### PHASE 4 — Ablation Table (≤ 12 GPU-hours)

**Goal:** Reproduce the four-row ablation in `tab:ablation` from fresh
checkpoints (currently 3 seeds for non-baseline, 5 for baseline).

**Prereqs:** PHASE 1 (cache), PHASE 2 (baseline KD-KL and full SaKD ckpts).

**Required matrix (Qwen3-0.6B only):**
- KD-KL baseline — reuse 5-seed KD-KL ckpts from PHASE 2
- +Noise KL only (λ=0.5, τ_w=∞ → uniform weights) — 3 new ckpts (seeds 42, 123, 456)
- +Reweight only (λ=0.0, τ_w=1.0) — 3 new ckpts
- Full SaKD (λ=0.5, τ_w=1.0) — reuse PHASE 2 ckpts for 3 of 5 seeds

**Steps:**
1. Add a "config" knob to the SaKD trainer for `noise_only` (τ_w=∞) and
   `reweight_only` (λ=0). Verify via dry-run on 10 samples that the two
   degenerate to the expected losses.
2. Train 6 new ckpts: 2 configs × 3 seeds.
3. Eval on the 3 held-out Dolly benchmarks.
4. Build `tmp/ablation.json` with mean ± std per (config, benchmark).
5. Replace `writing/NeurIPS26-SaGD/tables/ablation.tex` (preserve label
   `tab:ablation`).
6. Update §4.3 prose to reflect the new numbers — preserve the
   "constructively combine, sub-additive" framing only if the numbers still
   support it.

**Deliverables:** 6 new ckpts under `outputs_dolly/qwen_0.6B/sagd_ablation/`,
new `ablation.tex`, updated §4.3 prose.

**Acceptance:** ablation table compiles; SaKD full ≥ both single-component
configs on Avg ROUGE-L (sanity).

---

### PHASE 5 — Hyperparameter sensitivity (≤ 12 GPU-hours)

**Goal:** New table + figure showing SaKD robustness to λ, σ, τ_w, N.

**Prereqs:** PHASE 2 (Qwen3-0.6B default-config SaKD ckpt).

**Sweep:** λ ∈ {0.1, 0.5, 2.0}; σ ∈ {0.001, 0.005, 0.02}; τ_w ∈ {0.5, 1.0,
5.0}; N ∈ {1, 5, 20}. One-axis-at-a-time, default cell shared = 8 new ckpts
total. Single seed (42).

**Steps:**
1. Train 8 new SaKD ckpts. Persist to
   `outputs_dolly/qwen_0.6B/sagd_sweep/<axis>_<value>/seed_42/`.
2. Eval on the 3 held-out Dolly benchmarks.
3. `tmp/hp_sensitivity.json` per-cell metrics.
4. `writing/NeurIPS26-SaGD/tables/hp_sensitivity.tex` — 4 sub-tables, mark
   default cell.
5. `writing/NeurIPS26-SaGD/sources/hp_sensitivity.pdf` — 1×4 panel line plot, y
   = Avg ROUGE-L, KD-KL baseline as horizontal dashed line.
6. Add `\subsection{Hyperparameter Sensitivity}` to appendix (label
   `app:hp-sensitivity`); reference from end of §4.3.

**Deliverables:** 8 ckpts, table, figure, appendix subsection.

**Acceptance:** all 4 sweeps render; appendix paragraph identifies most/least
sensitive axis with concrete numbers.

**If blocked:** > 12 GPU-hr → sweep λ and σ only (8 → 4 new ckpts); document.

**DO NOT:** sweep multiple axes simultaneously.

---

### PHASE 6 — Training dynamics (≤ 4 GPU-hours)

**Goal:** Two-panel figure of train+val loss + saliency-divergence over
training, KD-KL vs SaKD.

**Prereqs:** PHASE 1.

**Steps:**
1. Re-train KD-KL and SaKD on Qwen3-0.6B seed 42 with per-50-step logging:
   train loss, val loss (Dolly val, fixed 200-sample subset), mean saliency
   divergence on the same subset.
2. `tmp/training_dynamics.json` `[{method, step, train_loss, val_loss,
   sal_div}]`.
3. `writing/NeurIPS26-SaGD/sources/training_dynamics.pdf` — 2 panels: (a)
   train+val loss curves, KD-KL vs SaKD; (b) saliency-divergence over training.
4. Add `\subsection{Training Dynamics}` to appendix (label
   `app:training-dynamics`); reference from §4.3.

**Deliverables:** json, PDF, appendix subsection.

**Acceptance:** PDF compiles; val-loss monotonic; SaKD's saliency divergence
ends below KD-KL's (else flag in NOTES, do not invent).

---

### PHASE 7 — Benchmark defense (≤ 6 GPU-hours)

**Goal:** Appendix table showing SaKD does not regress on general capability
benchmarks vs KD-KL.

**Prereqs:** PHASE 2.

**Matrix:** Qwen3-{0.6B, 1.7B} × {KD-KL, SaKD} (seed 42) × {MMLU 5-shot, ARC-C
25-shot, TruthfulQA mc2}. 12 cells total via lm-eval-harness.

**Steps:**
1. Run `lm-eval` for each cell. Raw outputs to `tmp/benchmark_defense_raw/`.
2. Aggregate to `tmp/benchmark_defense.json`.
3. `writing/NeurIPS26-SaGD/tables/benchmark_defense.tex` — 4 rows × 4 cols
   (MMLU/ARC-C/TruthfulQA/Avg).
4. Add `\subsection{General-Capability Defence}` to appendix (label
   `app:benchmark-defense`); reference from §Limitations.

**Acceptance:** SaKD Avg ≥ KD-KL Avg − 0.5 on at least one student. If SaKD
strictly regresses, candidly note it in `tmp/PHASE7_NOTES.md` and update
§Limitations.

**If blocked:** TruthfulQA download fails → ship MMLU + ARC-C only; document.

**DO NOT:** cherry-pick benchmarks.

---

### PHASE 8 — Qualitative saliency heatmap (Qwen) (≤ 1 GPU-hour)

**Goal:** Qwen3-0.6B version of the saliency heatmap (the existing one is
LLaMA, which doesn't match §4.4's primary architecture).

**Prereqs:** PHASE 1 (SQuAD Qwen cache), PHASE 2 (Qwen3-0.6B KD-KL and SaKD
ckpts, seed 42).

**Steps:** sample indices 0 (short context) and 69 (long context) per the
prior LLaMA heatmap; compute saliency arrays for Teacher + KD-KL student +
SaKD student; plot 2 × 3 heatmap with answer span as red bar; consistent
colour scale across panels.

**Deliverables:**
- `writing/NeurIPS26-SaGD/sources/saliency_heatmap_qwen.pdf`
- `tmp/saliency_heatmap_qwen.json` (raw arrays)
- New `\begin{figure}` in §4.4 or appendix (whichever page budget permits)
  with label `fig:saliency_heatmap`.

**Acceptance:** figure renders; KD-KL panel visibly more concentrated on the
answer span than the SaKD panel.

**DO NOT:** swap samples post-hoc to make SaKD look better. Indices 0 and 69
are fixed.

---

### PHASE 9 — Fig 1 shaded-region fix + manuscript polish (≤ 1.5 hours)

**Goal:** Final cleanup pass.

**Prereqs:** PHASES 3, 4, 5, 6, 7, 8 done so all labels are final.

**Steps:**
1. **Figure 1 shaded region:** open `sources/motivation_M1.pdf`. If there's no
   visible shaded region but the caption claims one, redraw the figure using
   the stored M1 data + a translucent rectangle on the failure cluster
   (`clean_KL ≤ 25%-ile AND perturbed_KL ≥ 75%-ile`). Save back to
   `sources/motivation_M1.pdf`.
2. **Bib orphan scrub:** for each entry in `references.bib`, grep for
   `\cite[a-z]*{<key>` across `sections/`, `tables/`, `figures/`,
   `algorithms/`, `checklist.tex`. Delete entries with 0 hits; record to
   `tmp/PHASE9_bib_orphans.md`.
3. **Reviewer-note scrub:** `grep -rE '\\(GH|CC|cc)\{|% TODO'
   writing/NeurIPS26-SaGD/`. Remove all rendered ones.
4. **Em-dash audit:** rewrite `---` in rendered prose to comma/semicolon/parens
   per memory rule.
5. **Italics/bold scrub:** `grep -rE '\\(emph|textit)\{'` — remove or convert
   per the user's preference.
6. **Widow re-check:** flag paragraphs ending in ≤ 2 short words to
   `tmp/PHASE9_widow.md`; extend the worst 5–10.

**Acceptance:**
- `grep` for `\\GH`, `\\CC`, `\\cc`, `\\emph`, `\\textit`, `% TODO` in
  rendered sections returns 0.
- Fig 1 caption matches figure markup.
- `references.bib` orphan count = 0.
- Paper still compiles.

---

### PHASE 10 — Page budget audit (≤ 1 hour)

**Goal:** Main text ≤ 9 pages without dropping substantive results.

**Prereqs:** PHASES 3–9 done.

**Tactic ladder (apply in order, stop when ≤ 9):**
1. Move PHASE 5 sensitivity discussion to appendix only.
2. Hide `tables/dolly_main_t.tex` (`% \input{tables/dolly_main_t}`).
3. Move Algorithm 1 from `method.tex` §3.4 back to `appendix.tex` with a
   one-line pointer in §3.4.
4. Drop the `\paragraph{Datasets.}` block in §4.1.
5. Tighten Fig 1 and Fig 2 captions to ≤ 3 lines each.
6. Remove the abstract's "Concretely, SaKD combines..." sentence.

**Steps:**
1. Compile 4-pass; count main-text pages via `pdftk` − appendix/refs/checklist.
2. If > 9, apply tactics 1–6 in order. After each, recompile and recount.
3. Document applied tactics in `tmp/PAGE_BUDGET.md`.

**Acceptance:** main text ≤ 9 (or candid documentation of overflow with what
was prioritised).

**DO NOT:** change font size, line spacing, margins, or drop a real result
table.

---

### PHASE 11 — Final compile, verification, and bundle (≤ 30 min)

**Goal:** Clean PDF + written summary.

**Prereqs:** all previous phases.

**Steps:**
1. 4-pass compile: pdflatex → bibtex → pdflatex → pdflatex.
2. Verify zero `Undefined reference`, zero `Citation undefined`, zero
   `Overfull > 10pt`. Persist build log to `tmp/PHASE11_build.log`.
3. Generate `tmp/PAPER_COMPLETION_DONE.md` with:
   - Per-phase status (DONE / PARTIAL / FAILED), wall time, deliverables,
     headline finding.
   - **Checkpoint manifest** — `tmp/CKPT_MANIFEST.csv` listing every trained
     ckpt: `(student, method, seed, config, ckpt_path, eval_avg_rougeL, ec,
     wall_time_s, nan_check)`.
   - "What changed in the paper" cheat-sheet.
   - "Next steps for human" subsection for anything PARTIAL.
4. `git add . && git commit -m "Full retrain + paper completion: PHASES 0-11"
   && git push` for both the main repo and `writing/NeurIPS26-SaGD/`.

**Deliverables:**
- `writing/NeurIPS26-SaGD/neurips_2026.pdf` (final)
- `tmp/PHASE11_build.log`
- `tmp/PAPER_COMPLETION_DONE.md`
- `tmp/CKPT_MANIFEST.csv`

**Acceptance:** A2.0 – A2.6 all check.

---

## 5. Global failure-handling rules

- **Unrecoverable failure** (dataset corrupt, repo permission lost, hardware
  unavailable, **code bug discovered**) → write `tmp/PAPER_COMPLETION_SOS.md`
  with traceback + phase + recommended human action. Stop.
- **Recoverable failure** → `tmp/PHASE{N}_NOTES.md` row + continue.
- **96-hour budget breach** → finalise PHASE 11 with whatever state exists;
  mark un-run phases TODO in `PAPER_COMPLETION_DONE.md`.
- **Monitor agent restart**: if the monitor detects a stall (no
  `exp_progress.log` update in > 30 min) it will `tmux send-keys` a wake signal
  to your tmux session. On wake, re-read this doc and resume from the last
  COMPLETED phase (i.e. find the latest DONE row in
  `tmp/EXPERIMENTS_DONE.md` and start from PHASE n+1).

---

## 6. Out of scope (do NOT do)

- Adding new methods, new theorems, new datasets.
- Changing paper title, author block, abstract claims, or method name (SaKD).
- Re-rendering `sources/framework.pdf`.
- Patching `src/sagd/` (write SOS instead).
- Modifying `tests/`, `README.md`, `CLAUDE.md`.
- Opening a PR (push directly to `origin master`).
- Using non-canonical hyperparameters per cell.

---

## 7. How the human will consume your output

After notification, the human will:
1. Read `tmp/PAPER_COMPLETION_DONE.md`.
2. Read `tmp/CKPT_MANIFEST.csv` to verify the training matrix is complete.
3. Spot-check `writing/NeurIPS26-SaGD/neurips_2026.pdf`.
4. For each PARTIAL phase, decide accept-or-followup.
5. Pull from Overleaf and re-publish.

Your job ends when the human can do these 5 steps without needing to ask any
clarifying question.

---

## 8. Companion: monitor agent

A separate agent reads `docs/paper_completion_monitor.md`. Its job is to
babysit you (detect stalls, restart, summarise progress to the human). You
never read or write the monitor's files; the monitor never modifies your
deliverables. If you see `tmp/MONITOR_*.md` files appear, ignore them.
