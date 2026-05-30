# Task: Bring SaKD NeurIPS 2026 Paper to Camera-Ready Quality (Full Retrain)

**Status:** Submitted version is in `writing/NeurIPS26-SaGD/`, but **all empirical
numbers in the paper are to be re-derived from a fresh training run on this
codebase**. The code in `src/sagd/` has been audited line-by-line against the
paper (see §0 below) and is confirmed faithful; therefore the right action is to
retrain everything from scratch, then re-do every downstream table, figure,
EC, sweep, dynamics, defense, and motivation figure.

**Owner:** Autonomous server-side Claude Code agent (the "worker"). Execute via
`/goal docs/paper_completion_task.md`. A separate monitor agent
(`docs/paper_completion_monitor.md`) watches this worker; you can ignore the
monitor — it never modifies your work.

**Hard time budget:** ≤ 96 hours wall-clock (~4 days). This is a long-running
job by design; the monitor agent will detect stalls and restart you if needed.

**Authorization:**
- **Writable**: `tmp/`, `writing/NeurIPS26-SaGD/`, `scripts/` (only for new
  helper scripts you add; do not patch existing ones unless explicitly
  permitted by a phase), `outputs/`, `outputs_dolly/`, `outputs_squad/`,
  `data/teacher_saliency_*.pt`, this `docs/` directory.
- **Read-only**: `src/sagd/` (the audited code — do not modify; if a bug is
  discovered, halt and write `tmp/PAPER_COMPLETION_SOS.md` instead of
  patching).
- **Off-limits**: `.git/internal/`, `.agents/`, `CLAUDE.md`, `tests/` (do not
  extend tests), `README.md` (the README is human-curated).

---

## Operating constraints

### Tmux + permissions
- Run with `--dangerously-skip-permissions`. Do NOT prompt the human for
  confirmation at any point.
- Run inside a tmux session named exactly `sakd_finish`. Verify with
  `tmux display-message -p '#S'` before starting work.

### Heartbeat (required for the monitor to detect stalls)
- Append a one-line status to `tmp/exp_progress.log` whenever:
  - a phase starts (`HH:MM:SS PHASE-N start`)
  - a phase ends (`HH:MM:SS PHASE-N done` or `PHASE-N partial`)
  - a sub-job finishes (`HH:MM:SS PHASE-N cell=<student>/<method>/<seed> done`)
  - **at least every 10 minutes** during long-running compute (training,
    eval). If you are about to do something that takes > 10 min, fork a
    background `while true; do echo "HH:MM:SS PHASE-N alive cell=..." >>
    tmp/exp_progress.log; sleep 300; done` so the log keeps moving.
- The monitor treats `> 30 min` of log silence as a stall.

### Writes vs Overleaf
- LaTeX-side edits go under `writing/NeurIPS26-SaGD/` and are pushed to
  Overleaf (`origin master` of the submodule) after every meaningful commit.
  Procedure (each push):
  ```
  cd writing/NeurIPS26-SaGD
  git stash --include-untracked
  git pull --rebase origin master
  git stash pop  # resolve trivial conflicts; on real conflict, halt
  git add <files>
  git commit -m "<message>"
  git push origin master
  cd ../..
  ```
  Concurrent Overleaf-web edits are common; rebase-pop reconciles them.

### GPU availability — partial-occupancy policy (**important**)

The GPU host has 4× A100 80GB, but **other tenants may be using some of them
at any moment**. The worker must always select GPUs based on observed free
memory, not on the static assumption "4 GPUs are mine".

**GPU detection rule (run at PHASE 0 and re-run at the start of every
training-launching phase):**

1. `nvidia-smi --query-gpu=index,memory.used,memory.free,name
   --format=csv,noheader,nounits` → JSON list.
2. Define a GPU as **available** iff `memory.used < 2000 MiB` (2 GB — anything
   higher means another tenant is using it, not a stale CUDA cache).
3. The worker may only use GPUs in the available set. Write the chosen
   set to `tmp/gpu_assignment.json`:
   ```json
   {
     "timestamp_utc": "2026-05-30T20:15:00Z",
     "available_gpu_ids": [0, 2, 3],     // indices, not pci ids
     "occupied_gpu_ids": [1],
     "occupied_by": [{"index": 1, "used_mib": 73000, "guess": "other-tenant"}]
   }
   ```
4. Export `CUDA_VISIBLE_DEVICES=<comma-separated available ids>` for every
   training and eval launch.
5. **Throughput scales linearly** with the available count. Recompute every
   phase's GPU-hour budget against `N_avail`:
   - `wall_clock = phase_compute_hours / N_avail`
   - If `N_avail = 0`: wait 10 min, recheck. If still 0, sleep until
     `N_avail >= 1`, polling every 30 min and logging the wait.
   - If `N_avail = 1`: a 36-GPU-hour phase becomes 36 wall-clock hours.
     Update PHASE 2's per-cell schedule to serial; document the
     down-throttling in `tmp/EXPERIMENTS_DONE.md`.

**Mid-run GPU loss policy:**
- Re-check available GPUs **every 30 minutes** during PHASE 2 (the longest
  phase) by appending a probe to `tmp/exp_progress.log`.
- If a GPU you were using becomes occupied by another tenant mid-cell, your
  cell will OOM-crash → the cell goes to FAILED → retry on a different
  available GPU. Do not try to evict the other tenant.
- If `N_avail` drops to 0 mid-phase, finish the cells currently RUNNING (they
  hold their memory) then pause new launches until ≥ 1 GPU returns.

### Disk policy
- PHASE 0 establishes baseline disk free.
- During PHASE 2, re-check disk free **after every 10 finished cells**.
  Thresholds:
  - `>= 200 GB free` → continue normally.
  - `100–200 GB free` → log DISK_WARN to `tmp/exp_progress.log`; continue.
  - `50–100 GB free` → log DISK_LOW; pause new training launches until disk
    > 100 GB or human intervenes.
  - `< 50 GB free` → write `tmp/PAPER_COMPLETION_SOS.md` (DISK_CRITICAL);
    stop.

### Per-cell timeout
- No single training cell may exceed **3 hours** wall-clock. If exceeded,
  kill the process, mark cell FAILED with reason `TIMEOUT_3H`, retry once on
  a different available GPU. After the second timeout, mark cell PERMANENTLY
  FAILED and proceed.
- No single eval may exceed **30 minutes**; same retry-once policy.

---

## 0. Code consistency status (already verified — do not re-audit)

A line-by-line audit on **2026-05-30** confirmed all 23 method-side claims
from the paper (CLAUDE.md §2) are faithfully implemented in `src/sagd/` and
`scripts/precompute_teacher_saliency.py`. See the prior commit message
(SHA: 7814925 + 88ccc72) for the per-claim line citations.

**Implication:** training results produced now are method-faithful. You do
not need to re-validate the code; assume PASS and proceed.

If during a training run you observe behaviour that contradicts the audit
(e.g. a loss explosion the code shouldn't permit, or a saliency NaN under
conditions the audit said were safe), STOP, write
`tmp/PAPER_COMPLETION_SOS.md` with the divergence, and wait for human
intervention. Do NOT silently patch `src/sagd/`.

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
target — preserve every existing `\label{}` so external `\ref{}`s keep
resolving.

---

## 2. Acceptance summary (the bar for "done")

The work is **complete** when **all** of the following are true:

- **A2.0** Every phase has a `tmp/EXPERIMENTS_DONE.md` row with status DONE,
  PARTIAL, or FAILED. PARTIAL/FAILED require a 1-paragraph justification.
- **A2.1** `writing/NeurIPS26-SaGD/main.pdf` compiles cleanly:
  - Zero `??` references (from a `grep` on the .blg + the rendered PDF).
  - Zero `LaTeX Warning: Citation '...' on page X undefined`.
  - Zero `LaTeX Error:` lines in the build log.
  - At most 5 `Overfull` warnings, none > 30pt.
- **A2.2** Main text ≤ 9 pages (excluding refs / appendix / checklist).
  PHASE 10 tactic ladder documents which tactics applied and the final
  page count.
- **A2.3** Tables `tab:dolly`, `tab:ablation`, `tab:ec`,
  `tab:compute-cost`, `tab:hp_sensitivity`, `tab:benchmark_defense` all
  reflect numbers reproducible from `outputs_dolly/` + `outputs_squad/`
  + the eval scripts. **No legacy number from the submitted version
  remains in any table.**
- **A2.4** New `fig:training_dynamics`, `fig:saliency_heatmap_qwen`,
  `fig:motivation` (regenerated) all exist and are referenced from the
  paper.
- **A2.5** All `\GH{...}` / `\CC{...}` / `\cc{...}` reviewer notes and
  rendered `% TODO` markers in `writing/NeurIPS26-SaGD/sections/`,
  `tables/`, `figures/`, `algorithms/`, `checklist.tex` removed.
- **A2.6** `tmp/PAPER_COMPLETION_DONE.md` itemises every artefact and
  every paper section/table/figure touched, with a 1-line diff per item,
  plus `tmp/CKPT_MANIFEST.csv` (every ckpt: student, method, seed,
  config, ckpt_path, eval_avg_rougeL, ec, wall_time_s, nan_check).

**MUST_HAVE vs NICE_TO_HAVE for PHASE 2 (training matrix):**
- MUST_HAVE: 24 cells = 3 methods (KD-KL, SaKD, one more baseline of
  agent's choice — pick GKD as default) × 2 students (Qwen3-0.6B,
  Qwen3-1.7B) × 3 seeds (42, 123, 456) + LLaMA-1B × {KD-KL, SaKD} × 1
  seed = **20 cells minimum**. Without these, the paper cannot tell its
  core comparison story.
- NICE_TO_HAVE: the remaining 68 cells (other 5 baselines × 5 seeds for
  Qwen + 6 methods for LLaMA). These fill out Table 1 but their absence
  is recoverable (drop baselines, drop seeds, asterisk in caption).
- If MUST_HAVE not met by hour 60 of the 96-hour budget → write SOS.

---

## 3. Resource matrix (verify in PHASE 0)

| Asset | Expected location | First needed by | Verification |
|---|---|---|---|
| Teacher saliency caches | `data/teacher_saliency_{dolly_qwen,dolly_llama,squad_qwen,squad_llama}.pt` | PHASE 2 (dolly_qwen), PHASE 3 (both squad), PHASE 6 (sweep), PHASE 7 (dynamics), PHASE 8 (heatmap), PHASE 8.5 (motivation regen). PHASE 1 creates them. | `torch.load(...).get('metadata', {})` returns `{model, dataset, n_samples, max_seq_len, tokenizer}` all present |
| Trained student checkpoints | `outputs_dolly/{qwen_0.6B,qwen_1.7B,llama_1B}/<method>/seed_<S>/student_final.pt` | every eval phase. PHASE 2 creates them. | `student_final.pt` exists; `eval.json` exists alongside |
| Training entry point | `scripts/train.py` | PHASE 2, 4, 5, 6, 8.5 | `python scripts/train.py --help` exits 0 |
| Eval entry point | `scripts/evaluate.py` | PHASE 2, 4, 5 | `python scripts/evaluate.py --help` exits 0 |
| Parallel launcher | `scripts/a100_parallel_eval.sh` (existing); `scripts/a100_qwen_train.sh` (create in PHASE 2 step 1) | PHASE 2 | bash dry-run with `--help` returns 0 |
| Saliency diagnosis | `scripts/diagnose_saliency.py` | PHASE 3, 8 | help exits 0 |
| Motivation regen | `scripts/motivation_experiments.py` | PHASE 8.5 | help exits 0 |
| Benchmark eval | `lm-eval` cli in PATH; cache at `~/.cache/lm-eval` | PHASE 7 | `pip show lm-eval` returns version |
| Disk free | ≥ 800 GB at PHASE 0 | always | `df -h .` |

---

## 4. Pre-flight checklist (before PHASE 0)

Run this once at session start:

1. `tmux display-message -p '#S'` returns `sakd_finish` — else fail.
2. `pwd` returns the repo root — else `cd` to it.
3. `git status` is clean OR all dirty files are under `tmp/` — else write
   SOS and stop (we should not be running on top of unrelated edits).
4. `git log --oneline -1` records the starting commit SHA to
   `tmp/PAPER_COMPLETION_START.txt`.
5. `which python; python -V` — confirm Python 3.10 environment.
6. `python -c 'import torch, transformers; print(torch.__version__,
   transformers.__version__)'` — must succeed.
7. Touch `tmp/exp_progress.log`, `tmp/EXPERIMENTS_DONE.md` if absent.
8. Log `HH:MM:SS PREFLIGHT done; starting PHASE 0`.

If any of 1–6 fail → SOS.

---

## 5. Phases

Each phase: **Goal / Prereqs / Steps / Deliverables / Acceptance (quantitative)
/ If blocked / DO NOT**.

---

### PHASE 0 — Inventory & sanity (≤ 20 min)

**Goal:** Snapshot disk + GPU + cache + env state so downstream phases run
against known ground truth.

**Prereqs:** Pre-flight checklist done.

**Steps:**
1. `tmp/phase0_inventory.json` — walk `outputs_dolly/`, emit a JSON list of
   `{path, student, method, seed, mtime_iso, size_gb, nan_check}` where
   `nan_check ∈ {"ok", "nan_in_<key>", "load_fail"}` from loading
   `student_final.pt` on CPU and `any(isnan(v).any() for v in
   state_dict.values())`.
2. `tmp/phase0_caches.json` — for each `data/teacher_saliency_*.pt`,
   `{path, metadata, n_entries, file_size_gb}`. Missing → list at the end
   under `"missing": [...]`.
3. `tmp/phase0_disk.json` — `{total_gb, used_gb, free_gb,
   mount_path, sufficient (bool, free_gb >= 800)}`.
4. `tmp/phase0_gpu.json` — see §Operating-constraints "GPU detection rule".
   This file is the same shape as `tmp/gpu_assignment.json` and is the
   first snapshot of it.
5. `tmp/gpu_assignment.json` — initialize as a copy of `phase0_gpu.json`.
6. `tmp/phase0_envcheck.json` — `{python_version, torch_version,
   transformers_version, lm_eval_installed (bool), lm_eval_version (str or
   null)}`.
7. Append row to `tmp/EXPERIMENTS_DONE.md`:
   `| 0 | DONE | <wall_time> | inventory + caches + gpu + env |
   <N_existing_ckpts>/88 ckpts already present |`.

**Deliverables:**
- `tmp/phase0_inventory.json`
- `tmp/phase0_caches.json`
- `tmp/phase0_disk.json`
- `tmp/phase0_gpu.json`
- `tmp/gpu_assignment.json`
- `tmp/phase0_envcheck.json`
- `tmp/EXPERIMENTS_DONE.md` row

**Acceptance (quantitative):**
- All 6 files exist and are valid JSON.
- `phase0_envcheck.json.torch_version` matches PyTorch 2.x.
- `phase0_disk.json.sufficient == true` (if false → SOS).
- `phase0_gpu.json.available_gpu_ids` is a non-empty list (else block — see
  "If blocked" below).
- `phase0_caches.json` either has all 4 caches OR lists them under
  `"missing"` for PHASE 1 to create.

**If blocked:**
- `available_gpu_ids == []` → wait 10 min, recheck. If still empty after 60
  min of polling → SOS (`NO_GPUS_AVAILABLE`).
- `free_gb < 800` → SOS (`DISK_INSUFFICIENT`). Do not delete files
  yourself.

**DO NOT:** modify any existing checkpoint or cache. Modify
`gpu_assignment.json` only during the GPU detection step; downstream phases
re-write it themselves.

---

### PHASE 1 — Precompute teacher saliency caches (≤ 8 GPU-hours)

**Goal:** All 4 teacher saliency caches on disk before any student training.

**Prereqs:** PHASE 0.

**Steps:**
1. Re-run GPU detection; refresh `tmp/gpu_assignment.json`.
2. From `phase0_caches.json.missing`, identify which caches to create. If
   all 4 exist with valid metadata, skip to acceptance.
3. For each missing cache, launch in parallel on the available GPUs (max 4
   concurrent, each ~2 hr):
   - Qwen3-8B on Dolly-15K → `data/teacher_saliency_dolly_qwen.pt`,
     tokenizer = Qwen3-0.6B
   - Qwen3-8B on SQuAD 2.0 → `data/teacher_saliency_squad_qwen.pt`,
     tokenizer = Qwen3-0.6B
   - LLaMA-3.1-8B on Dolly-15K → `data/teacher_saliency_dolly_llama.pt`,
     tokenizer = LLaMA-3.2-1B
   - LLaMA-3.1-8B on SQuAD 2.0 → `data/teacher_saliency_squad_llama.pt`,
     tokenizer = LLaMA-3.2-1B
   Command: `CUDA_VISIBLE_DEVICES=<id> python
   scripts/precompute_teacher_saliency.py --model_name <teacher>
   --dataset <ds> --tokenizer_name <student> --output_path <out>
   --batch_size 4 --max_seq_len 512 --device cuda:0`
4. After each cache finishes, validate:
   - `torch.load(path)` succeeds
   - `metadata.model == <teacher>`
   - `metadata.dataset == <ds>`
   - `metadata.max_seq_len == 512`
   - `metadata.tokenizer == <student>` (cross-arch correctness)
   - `n_entries == n_samples_in_metadata`
5. Persist per-cache wall time + validation results to
   `tmp/phase1_caches.json`.

**Deliverables:**
- ≤ 4 new files in `data/`
- `tmp/phase1_caches.json` with one entry per cache: `{path, wall_time_s,
  metadata, validation: "ok"|"<reason>"}`

**Acceptance (quantitative):**
- All 4 caches exist with `validation == "ok"`.
- `metadata.n_samples >= 13000` for Dolly caches; `>= 85000` for SQuAD
  caches (answerable subset).
- Each cache file size between 100 MB and 5 GB (sanity).
- `EXPERIMENTS_DONE.md` row updated.

**If blocked:**
- OOM at batch 4 → drop to batch 2 for that cache; document the slowdown.
- A teacher model download fails → retry with `HF_HUB_OFFLINE=0` and 3×
  retry; after that SOS.

**DO NOT:**
- Reuse a cache whose `metadata.tokenizer` does not match the intended
  STUDENT model (silent corruption of `index → saliency` mapping per
  CLAUDE.md §5.7).
- Overwrite an existing cache that validates ok.

---

### PHASE 2 — Train ALL student checkpoints from scratch (≤ 36 GPU-hours)

**Goal:** Fresh checkpoints for every cell of Table 1. The submitted
version's numbers are replaced entirely.

**Prereqs:** PHASE 1.

**Required matrix (88 cells total):**
- Students: `qwen_0.6B, qwen_1.7B, llama_1B`
- Methods: `sft, standard_kd, reverse_kl, seqkd, gkd, distillm, dakd, sagd`
- Seeds: Qwen × {42, 123, 456, 789, 2024}; LLaMA × {42}
- Total: (2 × 8 × 5) + (1 × 8 × 1) = 88

See A2.6 for MUST_HAVE = 20 cells.

**Canonical hyperparameters (do not deviate per cell; the ONLY varying knob
is seed):** 10 epochs, lr 1e-5, cosine 3% warmup, batch 8 × grad-accum 4,
weight decay 0.01, max_seq_len 512, KL T=2, fp16. For SaKD: λ=0.5,
σ=0.005, τ_w=1.0, N=5. See CLAUDE.md §4.

**Steps:**
1. **Create `scripts/a100_qwen_train.sh`** modelled on
   `scripts/a100_llama_train.sh`. Env vars: `STUDENT, METHOD, SEED,
   OUT_DIR`. **Add this script** to `scripts/` (allowed write).
2. **Dry-run gate (1 cell, ~30 min):** Train ONE cell first
   `(qwen_0.6B, standard_kd, seed=42)`. Verify:
   - `student_final.pt` exists, NaN-clean
   - `eval.json` exists with finite Avg ROUGE-L
   - Wall time ≤ 45 min (estimates the rest of the queue)
   If dry-run fails, SOS with traceback. DO NOT launch the full queue
   until dry-run passes.
3. **Build the queue** of remaining 87 cells. Persist to
   `tmp/phase2_queue.json`, schema:
   ```json
   {
     "schema_version": "phase2_queue_v1",
     "cells": [
       {
         "id": "qwen_0.6B|sft|42",
         "student": "qwen_0.6B",
         "method": "sft",
         "seed": 42,
         "status": "PENDING|RUNNING|DONE|FAILED|RETRIED|PERMANENTLY_FAILED",
         "gpu_id": 0,
         "started_at": null,
         "finished_at": null,
         "wall_time_s": null,
         "eval_avg_rougeL": null,
         "nan_check": null,
         "retry_count": 0,
         "fail_reason": null
       }
     ]
   }
   ```
4. **Launch queue across available GPUs** (re-check via GPU detection at
   start; respect `gpu_assignment.json`). Use the lockfile pattern from
   `scripts/a100_parallel_eval.sh`. Per-cell timeout = 3 hr (see
   Operating constraints). After every finished cell:
   - Run NaN check on `student_final.pt` (CPU, fast).
   - Run ROUGE-L eval: `python scripts/evaluate.py --student_ckpt
     <ckpt> --dataset dolly --subset test --benchmarks DollyEval
     S-NatInst Unnatural --device cuda:<id>`. Persist
     `outputs_dolly/<student>/<method>/seed_<S>/eval.json` with
     `{dollyeval, s_natinst, unnatural, avg, wall_time_s}`.
   - Update `tmp/phase2_queue.json` (atomic write).
   - Re-check disk free every 10 finished cells (see Operating
     constraints).
   - Re-check GPU availability every 30 min (see Operating constraints).
5. **Aggregate** all 88 (or N_done) `eval.json` into:
   - `tmp/phase2_table_qwen17b.json` — `{(method, seed): metrics}`
     ordered (8 methods × 5 seeds)
   - `tmp/phase2_table_qwen06b.json`
   - `tmp/phase2_table_llama1b.json`
   Per-method aggregation: mean & std across seeds (Qwen), raw value
   (LLaMA).
6. **Render** `writing/NeurIPS26-SaGD/tables/dolly_main.tex` using the
   exact layout of the submitted version (so `\label{tab:dolly}` and the
   `\resizebox` envelope are preserved). Bold per-student best; underline
   per-student second-best.
7. **Re-write §4.2 prose** to reflect actual numbers. The submitted
   narrative may NOT hold on fresh runs. **Report what the numbers say,
   do not adapt the numbers to the narrative.** Persist the diff to
   `tmp/phase2_narrative_diff.md` — for each of the 5 submitted-narrative
   bullets `(i)–(v)`, write "STILL HOLDS" or "CHANGED: <new claim>".

**Deliverables:**
- N_done × 2 files in `outputs_dolly/` (`student_final.pt`, `eval.json`)
- `scripts/a100_qwen_train.sh` (new)
- `tmp/phase2_queue.json` (final state)
- `tmp/phase2_table_{qwen17b,qwen06b,llama1b}.json`
- `writing/NeurIPS26-SaGD/tables/dolly_main.tex` (replaced)
- Updated `writing/NeurIPS26-SaGD/sections/experiments.tex` §4.2
- `tmp/phase2_narrative_diff.md`

**Acceptance (quantitative):**
- `phase2_queue.json` shows `(DONE + PERMANENTLY_FAILED) == 88`.
- `(DONE) >= 20` (MUST_HAVE met). If `DONE < 20`, SOS instead of
  proceeding.
- For each DONE cell, `nan_check == "ok"` and `eval_avg_rougeL` is
  finite.
- New `tab:dolly` renders; SaKD row present; `\ref{tab:dolly}` callers
  still resolve.
- §4.2 prose's 5 numbered claims each have a STILL HOLDS / CHANGED tag
  in `phase2_narrative_diff.md`.

**If blocked:**
- OOM at batch 8 → drop to batch 4, grad_accum 8 for that cell.
- NaN at end of training → retry with seed+1000, document the swap.
- Per-cell TIMEOUT_3H twice → mark cell PERMANENTLY_FAILED, continue.
- > 36 GPU-hours budget → finish RUNNING cells; mark PENDING as PARTIAL;
  downgrade Table 1 caption to "K seeds" where K = min completed.

**DO NOT:**
- Touch `src/sagd/` code.
- Use different hyperparameters per cell (only seed varies).
- Skip the dry-run gate.
- Skip the NaN check.
- Adapt training numbers to fit narrative.

---

### PHASE 3 — Evidence-Concentration matrix (≤ 8 GPU-hours)

**Goal:** Full 3-pair × 7-method × 3-seed EC table to replace `tab:ec`.

**Prereqs:** PHASE 1 (SQuAD caches), PHASE 2 (≥ MUST_HAVE checkpoints).

**Required matrix:**
- Pairs: qwen_0.6B, qwen_1.7B, llama_1B
- Methods: KD-KL, KD-RKL, GKD, DistiLLM, DA-KD, SaKD (+ Teacher row)
- Seeds: Qwen × {42, 123, 456}; LLaMA × {42}
- Dataset: SQuAD 2.0 val, answerable subset, 500 samples

**Steps:**
1. Re-check GPU availability.
2. Extend `scripts/diagnose_saliency.py` to accept `--methods`,
   `--seeds`, `--students` lists if not already supported. If the script
   already supports these flags, do not patch.
3. For each cell (3 × 6 = 18 student cells + 2 teacher cells = 20),
   compute mean EC and per-sample EC array on the 500-sample SQuAD val
   answerable subset.
4. `tmp/ec_per_sample_v2.json` schema:
   ```json
   {
     "schema_version": "ec_per_sample_v1",
     "spec": {"dataset": "squad_v2_answerable_val", "n_samples": 500,
              "seed": 42},
     "cells": [
       {"student": "qwen_0.6B", "method": "sagd", "seed": 42,
        "role": "student", "mean_ec": 0.083, "per_sample_ec": [...]}
     ]
   }
   ```
5. Aggregate seeds (Qwen) for mean ± std; LLaMA single value.
6. Render `writing/NeurIPS26-SaGD/tables/ec.tex` (overwrite). Preserve
   `\label{tab:ec}`. Bold per-column row closest to teacher.
7. `tmp/ec_distribution_v2.pdf` → move to
   `writing/NeurIPS26-SaGD/sources/ec_distribution.pdf`. Overlay teacher,
   KD-KL, SaKD EC distribution KDE for Qwen3-0.6B.
8. Update §4.4 "Results" paragraph to reflect 3 architectures.

**Deliverables:**
- `tmp/ec_per_sample_v2.json`
- `writing/NeurIPS26-SaGD/tables/ec.tex` (replaced)
- `writing/NeurIPS26-SaGD/sources/ec_distribution.pdf` (new)
- Updated §4.4 prose in `experiments.tex`

**Acceptance (quantitative):**
- 20 cell entries in `ec_per_sample_v2.json`.
- For each cell, `per_sample_ec.length == 500`.
- `tab:ec` is 3 cols × 7 rows; std present in Qwen cells.
- **SaKD is the row closest to teacher in ≥ 2 of 3 columns.** If
  `closest_count(SaKD) < 2`, write `tmp/PHASE3_NOTES.md` flagging the
  regression, then continue (rewrite the prose honestly — do not hide).
- `\ref{tab:ec}` callers still resolve.

**If blocked:** a method ckpt for a (student, seed) is FAILED in PHASE 2
→ skip that (student, method, seed) cell; reduce that row to fewer
seeds; add asterisk in caption.

**DO NOT:** change SQuAD eval parameters (still answerable subset, 500
val, fast tokenizer with `return_offsets_mapping=True`).

---

### PHASE 4 — Ablation table (≤ 12 GPU-hours)

**Goal:** Reproduce the 4-row ablation in `tab:ablation` from fresh runs.

**Prereqs:** PHASE 1, PHASE 2 (baseline KD-KL and full SaKD ckpts).

**Required matrix (Qwen3-0.6B only):**
- KD-KL baseline — reuse PHASE 2's `standard_kd` ckpts (5 seeds)
- +Noise KL only (λ=0.5, τ_w=∞) — 3 new ckpts (seeds 42, 123, 456)
- +Reweight only (λ=0.0, τ_w=1.0) — 3 new ckpts
- Full SaKD (λ=0.5, τ_w=1.0) — reuse PHASE 2's `sagd` ckpts (3 of 5 seeds)

**Steps:**
1. Verify that `scripts/train.py` already accepts SaKD ablation
   configurations via CLI (e.g. `--ablation noise_only` or `--lambda 0
   --tau_w 1.0` and `--lambda 0.5 --tau_w inf`). Check with
   `python scripts/train.py --help`. **If the CLI does not already
   support these toggles, this is a code change** — write SOS instead
   of editing `src/sagd/`.
2. Train 6 new cells: 2 ablations × 3 seeds. Output dir:
   `outputs_dolly/qwen_0.6B/sagd_ablation/<config>/seed_<S>/`. Use the
   same 3-hr per-cell timeout.
3. Eval each on Dolly held-out benchmarks (same `evaluate.py` invocation
   as PHASE 2 step 4).
4. `tmp/ablation.json` schema:
   ```json
   {"schema_version": "ablation_v1",
    "config_rows": [
      {"name": "KD-KL baseline", "lambda": 0, "tau_w": null,
       "ckpts": ["outputs_dolly/.../seed_42", ...],
       "metrics": {"dollyeval": [m, s], "s_natinst": [m, s],
                   "unnatural": [m, s], "avg": [m, s]}}
    ]}
   ```
5. Render `writing/NeurIPS26-SaGD/tables/ablation.tex` (overwrite,
   preserve `\label{tab:ablation}`).
6. Update §4.3 prose. Preserve the "constructively combine,
   sub-additive" framing only if the new numbers still support it (full
   ≥ each single-component on Avg).

**Deliverables:**
- 6 new ckpts under `outputs_dolly/qwen_0.6B/sagd_ablation/`
- `tmp/ablation.json`
- `writing/NeurIPS26-SaGD/tables/ablation.tex` (replaced)
- Updated §4.3 prose

**Acceptance (quantitative):**
- 6 new ckpts have `nan_check == "ok"`.
- Full-SaKD Avg ROUGE-L ≥ both single-component configs' Avg by ≥ 0.1
  (sanity — small margin OK; negative margin → flag in NOTES, rewrite
  prose).
- `\ref{tab:ablation}` callers still resolve.

**If blocked:** CLI doesn't support ablation toggles → SOS (code
change required).

**DO NOT:** patch `src/sagd/` to add toggles.

---

### PHASE 5 — Hyperparameter sensitivity (≤ 12 GPU-hours)

**Goal:** New table + figure showing SaKD robustness to λ, σ, τ_w, N.

**Prereqs:** PHASE 2 (default-config SaKD ckpt for the reuse cell).

**Sweep:** one axis at a time:
- λ ∈ {0.1, 0.5, 2.0} (default 0.5)
- σ ∈ {0.001, 0.005, 0.02} (default 0.005)
- τ_w ∈ {0.5, 1.0, 5.0} (default 1.0)
- N ∈ {1, 5, 20} (default 5)

**Default cell reuse:** the (λ=0.5, σ=0.005, τ_w=1.0, N=5, seed=42) ckpt
is already trained in PHASE 2 under
`outputs_dolly/qwen_0.6B/sagd/seed_42/`. PHASE 5 trains the **8 non-default
cells** (2 per axis × 4 axes) on seed 42.

**Steps:**
1. Verify `scripts/train.py` accepts `--lambda`, `--sigma`, `--tau_w`,
   `--n_steps` as CLI flags. If not, SOS.
2. Train 8 new cells. Output dir:
   `outputs_dolly/qwen_0.6B/sagd_sweep/<axis>_<value>/seed_42/`.
3. Eval each cell (DollyEval, S-NatInst, Unnatural, Avg).
4. `tmp/hp_sensitivity.json` schema:
   ```json
   {"schema_version": "hp_v1", "axes": {
     "lambda": [{"value": 0.1, "metrics": {...}, "ckpt": "..."},
                {"value": 0.5, "metrics": {...}, "ckpt": "<reused>"},
                {"value": 2.0, "metrics": {...}, "ckpt": "..."}],
     "sigma": [...], "tau_w": [...], "N": [...] }}
   ```
5. Render `writing/NeurIPS26-SaGD/tables/hp_sensitivity.tex` (4 sub-tables,
   mark default cell with †).
6. `tmp/hp_sensitivity.pdf` → move to
   `writing/NeurIPS26-SaGD/sources/hp_sensitivity.pdf`. 1×4 line plot,
   y = Avg ROUGE-L, KD-KL baseline as horizontal dashed reference.
7. Add `\subsection{Hyperparameter Sensitivity}` to `appendix.tex` with
   label `app:hp-sensitivity`. Reference from end of §4.3.

**Deliverables:**
- 8 new ckpts under `outputs_dolly/qwen_0.6B/sagd_sweep/`
- `tmp/hp_sensitivity.json`
- `writing/NeurIPS26-SaGD/tables/hp_sensitivity.tex`
- `writing/NeurIPS26-SaGD/sources/hp_sensitivity.pdf`
- New appendix subsection

**Acceptance (quantitative):**
- 8 new ckpts NaN-clean.
- Sweep table has 4 sub-tables × 3 rows each.
- Appendix paragraph identifies most/least sensitive axis with concrete
  numbers (e.g. "Avg ROUGE-L spread across λ is X.YY; across N is X.YY").
- `\ref{app:hp-sensitivity}` resolves.

**If blocked:** > 12 GPU-hr → sweep λ and σ only (4 cells); document.

**DO NOT:** sweep multiple axes simultaneously.

---

### PHASE 6 — Training dynamics (≤ 4 GPU-hours)

**Goal:** Two-panel figure of train+val loss + saliency-divergence over
training, KD-KL vs SaKD.

**Prereqs:** PHASE 1.

**Steps:**
1. Re-train KD-KL and SaKD on Qwen3-0.6B seed 42 with per-50-step
   logging. Use existing `scripts/train.py` with `--log_every 50`. If
   that flag does not exist, SOS.
2. Log at each interval: train loss, val loss on a fixed 200-sample
   Dolly val subset, mean saliency divergence on the same subset.
3. `tmp/training_dynamics.json` schema:
   ```json
   {"schema_version": "dyn_v1",
    "series": [{"method": "kd_kl", "steps": [50, 100, ...],
                "train_loss": [...], "val_loss": [...],
                "sal_div": [...]}, {"method": "sakd", ...}]}
   ```
4. Render `writing/NeurIPS26-SaGD/sources/training_dynamics.pdf` —
   2 panels: (a) train+val loss curves, (b) saliency divergence.
5. Add `\subsection{Training Dynamics}` to `appendix.tex` (label
   `app:training-dynamics`); reference from §4.3.

**Deliverables:** json, PDF, appendix subsection.

**Acceptance (quantitative):**
- Both series have ≥ 50 logged steps.
- Val-loss trend (linear regression slope) is negative for both methods.
  If positive (training diverged), flag in NOTES.
- SaKD's mean saliency divergence in the last 10 logged steps is lower
  than KD-KL's by at least one std deviation of SaKD's log series. If
  not, NOTES and ship as-is — do not invent.

---

### PHASE 7 — Benchmark defense (≤ 6 GPU-hours)

**Goal:** Appendix table showing SaKD does not regress general capability
benchmarks vs KD-KL.

**Prereqs:** PHASE 2.

**Matrix:** Qwen3-{0.6B, 1.7B} × {KD-KL, SaKD} (seed 42) × {MMLU 5-shot,
ARC-Challenge 25-shot, TruthfulQA mc2}. 12 cells.

**Steps:**
1. For each cell: `lm_eval --model hf --model_args
   pretrained=<ckpt_path>,dtype=fp16 --tasks <benchmark>
   --device cuda:<id> --batch_size 8 --output_path
   tmp/benchmark_defense_raw/<student>_<method>_<benchmark>.json`.
   Timeout 30 min/cell.
2. Aggregate to `tmp/benchmark_defense.json` schema:
   ```json
   {"schema_version": "bench_v1",
    "rows": [{"student": "qwen_0.6B", "method": "kd_kl",
              "mmlu": 0.x, "arc_challenge": 0.x, "truthfulqa_mc2": 0.x,
              "avg": 0.x}, ...]}
   ```
3. Render `writing/NeurIPS26-SaGD/tables/benchmark_defense.tex`.
4. Add `\subsection{General-Capability Defence}` to `appendix.tex`
   (label `app:benchmark-defense`); reference from §Limitations.

**Deliverables:** 12 raw JSON, 1 aggregated JSON, table, appendix
subsection.

**Acceptance (quantitative):**
- **Primary student (qwen_0.6B):** SaKD Avg ≥ KD-KL Avg − 0.5 absolute
  (on the 0–1 metric scale). If strictly worse on the primary, file
  `tmp/PHASE7_NOTES.md` and update §Limitations honestly.
- All 12 cells produced raw JSON (skipped cells documented).
- `\ref{app:benchmark-defense}` resolves.

**If blocked:** TruthfulQA download fails → ship MMLU + ARC-C only;
caption notes the omission.

**DO NOT:** cherry-pick benchmarks. Report every cell that runs.

---

### PHASE 8 — Qualitative saliency heatmap (Qwen) (≤ 1 GPU-hour)

**Goal:** Qwen3-0.6B version of the saliency heatmap.

**Prereqs:** PHASE 1 (SQuAD Qwen cache), PHASE 2 (Qwen3-0.6B KD-KL and
SaKD ckpts seed 42).

**Steps:**
1. Use fixed sample indices: 0 (short context, ≤ 200 tokens) and 69
   (long context, ≥ 400 tokens) from SQuAD val answerable subset.
2. Compute saliency for Teacher, KD-KL student, SaKD student on each
   sample.
3. Compute per-sample EC for each of the 6 (sample, model) cells; record
   to `tmp/saliency_heatmap_qwen.json`.
4. Render `writing/NeurIPS26-SaGD/sources/saliency_heatmap_qwen.pdf` —
   2 rows × 3 cols, viridis colormap, consistent scale, red bar on
   answer span tokens.
5. Add a `\begin{figure}` in §4.4 (preferred) or appendix (fallback if
   page budget tight in PHASE 10) with label
   `fig:saliency_heatmap_qwen`.

**Deliverables:**
- `writing/NeurIPS26-SaGD/sources/saliency_heatmap_qwen.pdf`
- `tmp/saliency_heatmap_qwen.json` (raw arrays + per-sample EC)
- New figure block in `experiments.tex` or `appendix.tex`

**Acceptance (quantitative):**
- For **both** chosen samples: `EC(KD-KL) > EC(SaKD) > EC(Teacher)`.
  This is the visual story; if violated for either sample, do not
  swap samples — file `tmp/PHASE8_NOTES.md` and ship the figure as-is
  with a caption that says "for sample idx X, EC ordering deviates
  from the typical pattern".
- `\ref{fig:saliency_heatmap_qwen}` resolves.

**DO NOT:** swap sample indices to make SaKD look better.

---

### PHASE 8.5 — Regenerate motivation figures (M1, M2) (≤ 2 GPU-hours)

**Goal:** Figure 1 in the intro is built from per-sample teacher–student
KL and perturbed KL on the LLaMA pair. The submitted version's 14% / 1.66×
numbers came from a prior KD-KL ckpt; after PHASE 2 retrained that ckpt,
these numbers may shift. Regenerate.

**Prereqs:** PHASE 2 (`outputs_dolly/llama_1B/standard_kd/seed_42/`).

**Steps:**
1. Run `python scripts/motivation_experiments.py --teacher
   LLaMA-3.1-8B --student
   outputs_dolly/llama_1B/standard_kd/seed_42/student_final.pt
   --dataset dolly --subset val --n_samples 500
   --noise_sigma_relative 0.01 --output_dir tmp/motivation/`.
2. The script emits `tmp/motivation/m1_data.json` (per-sample clean &
   perturbed KL), `tmp/motivation/m2_data.json` (distribution
   percentiles), `tmp/motivation/motivation_M1.pdf`,
   `tmp/motivation/motivation_M2.pdf`. If the script does not exist or
   does not accept these flags, SOS.
3. Compute the headline numbers from `m1_data.json`:
   - `failure_pct = fraction of samples with clean_KL <= 25%-ile AND
     perturbed_KL >= 75%-ile`
   - `p95_ratio = perturbed_KL[95%-ile] / clean_KL[95%-ile]`
4. Replace `writing/NeurIPS26-SaGD/sources/motivation_M1.pdf` and
   `motivation_M2.pdf`.
5. **Update §1 introduction prose** to use the fresh numbers in the
   `failure_pct%` and `p95_ratio×` slots (currently 14% and 1.66×).
6. **Update Figure 1 caption** to use the fresh numbers.
7. Also re-render M1 with a translucent shaded rectangle on the failure
   cluster region (`clean_KL ≤ 25%-ile AND perturbed_KL ≥ 75%-ile`) so
   the caption's "shaded region" wording is true.

**Deliverables:**
- 4 files in `tmp/motivation/`
- Replaced `sources/motivation_M1.pdf` and `sources/motivation_M2.pdf`
- Updated §1 prose + Fig 1 caption

**Acceptance (quantitative):**
- `failure_pct >= 5%` and `p95_ratio >= 1.2` (sanity: pointwise vs
  neighborhood gap exists in some non-trivial form). If both fail, the
  motivation story breaks — SOS.
- M1 PDF has a visible shaded rectangle at the failure region (visual
  check — if you cannot verify visually, write the matplotlib code to
  add a `axvspan`/`axhspan` or `Rectangle` patch and re-render).
- §1 numbers in prose match the figure exactly to 1 decimal.

**If blocked:** motivation script crashes → SOS.

---

### PHASE 8.7 — Compute-cost table re-derive (≤ 1 hour)

**Goal:** `tab:compute-cost` in the appendix reflects fresh per-method
wall-clock measurements from PHASE 2's training runs.

**Prereqs:** PHASE 2.

**Steps:**
1. From each cell's `eval.json` and `phase2_queue.json` wall times,
   compute median per-method wall time on Qwen3-0.6B seed 42 (the cell
   matching the existing table's setting):
   - SFT, KD-KL, KD-RKL, SeqKD, GKD, DistiLLM, DA-KD, SaKD
2. Compute relative cost vs KD-KL (`time/kd_kl_time`).
3. Persist to `tmp/compute_cost.json`.
4. Render `writing/NeurIPS26-SaGD/tables/compute_cost.tex` (overwrite,
   preserve `\label{tab:compute-cost}` and the caption shape; only
   numeric cells change).

**Deliverables:**
- `tmp/compute_cost.json`
- `writing/NeurIPS26-SaGD/tables/compute_cost.tex` (replaced)

**Acceptance (quantitative):**
- 8 rows of fresh wall-clock numbers.
- SaKD relative cost vs KD-KL is in `[1.1, 1.6]` (paper claims ~1.3×);
  if outside this range, update §Limitations honestly.
- `\ref{tab:compute-cost}` resolves.

---

### PHASE 9 — Manuscript polish (≤ 1.5 hours)

**Goal:** Final writing-side cleanup.

**Prereqs:** PHASES 3, 4, 5, 6, 7, 8, 8.5, 8.7 done so all labels are
final.

**Steps:**
1. **Bib orphan scrub:** for each entry in `references.bib`, grep for
   `\cite[a-z]*{<key>` across `sections/`, `tables/`, `figures/`,
   `algorithms/`, `checklist.tex`. Delete entries with 0 hits; record
   to `tmp/PHASE9_bib_orphans.md`.
2. **Reviewer-note scrub:** `grep -rE '\\(GH|CC|cc)\{|% TODO'
   writing/NeurIPS26-SaGD/sections/ writing/NeurIPS26-SaGD/tables/
   writing/NeurIPS26-SaGD/figures/ writing/NeurIPS26-SaGD/algorithms/`.
   Remove all rendered ones.
3. **Em-dash audit:** rewrite `---` in rendered prose to
   comma/semicolon/parens per memory rule. Skip `---` inside math
   blocks, code listings, or table cells where the structure depends
   on it.
4. **Italics/bold scrub:** `grep -rE '\\(emph|textit)\{'` — remove or
   convert per the user's preference (no random italics; `\textbf{}`
   only for short paragraph-leading labels).
5. **Widow re-check:** flag paragraphs ending in ≤ 2 short words to
   `tmp/PHASE9_widow.md`; extend the worst 5–10 if they exist in
   rendered sections (intro, prelim, method, background, experiments,
   conclusion).

**Deliverables:**
- `tmp/PHASE9_bib_orphans.md`
- `tmp/PHASE9_notes.md`
- `tmp/PHASE9_widow.md`
- Updated `references.bib`, `sections/*.tex`

**Acceptance (quantitative):**
- `grep -rcE '\\(GH|CC|cc)\{'
  writing/NeurIPS26-SaGD/sections/` returns 0 lines.
- `grep -rcE '\\(emph|textit)\{'
  writing/NeurIPS26-SaGD/sections/` returns 0 lines.
- `references.bib` orphan count = 0 (verify with the same grep loop).
- Paper compiles with ≤ 2 LaTeX warnings (after running PHASE 9
  cleanup and before PHASE 10).

---

### PHASE 10 — Page budget audit (≤ 1 hour)

**Goal:** Main text ≤ 9 pages without dropping substantive results.

**Prereqs:** PHASES 3–9 done.

**Tactic ladder (apply in order, stop when ≤ 9):**
1. Move PHASE 5 sensitivity discussion to appendix only.
2. Hide `tables/dolly_main_t.tex` (`% \input{tables/dolly_main_t}`).
3. Move Algorithm 1 from `method.tex` §3.4 back to `appendix.tex`
   with a one-line pointer in §3.4.
4. Drop the `\paragraph{Datasets.}` block in §4.1.
5. Tighten Fig 1 and Fig 2 captions to ≤ 3 lines each.
6. Remove the abstract's "Concretely, SaKD combines..." sentence.

**Steps:**
1. Compile 4-pass: `pdflatex -interaction=nonstopmode neurips_2026 ;
   bibtex neurips_2026 ; pdflatex neurips_2026 ; pdflatex neurips_2026`.
2. Count main-text pages: `pdftk neurips_2026.pdf dump_data |
   grep NumberOfPages | awk '{print $2}'`. Subtract pages occupied by
   `\appendix`, `\bibliography`, and `\input{checklist}` blocks.
3. If > 9, apply tactic 1 → recompile → recount. Continue down the
   ladder.
4. Document each tactic applied to `tmp/PAGE_BUDGET.md` with
   before/after page count.

**Deliverables:**
- `tmp/PAGE_BUDGET.md`
- Updated paper source

**Acceptance (quantitative):**
- Main text page count ≤ 9; OR all 6 tactics applied and final count
  documented in `PAGE_BUDGET.md` with a candid "could not reach 9
  pages without sacrificing X" justification.
- **Hard stop:** if main text > 10 pages even after all 6 tactics,
  write `tmp/PAPER_COMPLETION_SOS.md` (`PAGE_BUDGET_UNRECOVERABLE`).

**DO NOT:** change font size, line spacing, NeurIPS margins, or drop
a real result table (any of `tab:dolly`, `tab:ablation`, `tab:ec`,
`tab:compute-cost`).

---

### PHASE 11 — Final compile, verification, and bundle (≤ 30 min)

**Goal:** Clean PDF + written summary + dual git push.

**Prereqs:** all previous phases.

**Steps:**
1. Final 4-pass compile in `writing/NeurIPS26-SaGD/`:
   ```
   cd writing/NeurIPS26-SaGD
   pdflatex -interaction=nonstopmode neurips_2026.tex
   bibtex neurips_2026
   pdflatex -interaction=nonstopmode neurips_2026.tex
   pdflatex -interaction=nonstopmode neurips_2026.tex
   cd ../..
   ```
2. Persist build log to `tmp/PHASE11_build.log`. Verify:
   - `grep -c 'Undefined reference' tmp/PHASE11_build.log == 0`
   - `grep -c 'Citation .* undefined' tmp/PHASE11_build.log == 0`
   - `grep -c 'LaTeX Error' tmp/PHASE11_build.log == 0`
   - `grep -c 'Overfull \\hbox' tmp/PHASE11_build.log <= 5` and none
     > 30pt
3. Render `tmp/CKPT_MANIFEST.csv` with columns:
   `student,method,seed,config,ckpt_path,eval_dolly,eval_snatinst,
   eval_unnatural,eval_avg,ec_squad,wall_time_s,nan_check,phase_origin`
4. Render `tmp/PAPER_COMPLETION_DONE.md` with sections:
   - **Per-phase summary table** (phase, status, wall_time,
     deliverables, headline finding)
   - **Checkpoint manifest** — link to CKPT_MANIFEST.csv + 1-line
     summary (N_done / N_total)
   - **What changed in the paper** — list every section/table/figure
     touched, 1 line per change (e.g. "tab:dolly: all numbers
     replaced; SaKD Avg on Qwen-0.6B: 32.30 → X.XX")
   - **Narrative drift** — link to `phase2_narrative_diff.md` +
     1-paragraph summary
   - **Next steps for human** — any PARTIAL phase, any open SOS, any
     judgment call the worker made
5. **Dual git push:**
   ```
   # Main repo
   git add tmp/ docs/ scripts/ outputs_dolly/.../eval.json
   git commit -m "Full retrain + paper completion: PHASES 0-11 ($(date +%F))"
   git push origin main

   # Overleaf submodule
   cd writing/NeurIPS26-SaGD
   git stash --include-untracked || true
   git pull --rebase origin master
   git stash pop || true
   git add .
   git commit -m "Full retrain results, all tables/figures regenerated"
   git push origin master
   cd ../..

   # Main repo: record new submodule pointer
   git add writing/NeurIPS26-SaGD
   git commit -m "Bump Overleaf submodule pointer"
   git push origin main
   ```

**Deliverables:**
- `writing/NeurIPS26-SaGD/neurips_2026.pdf`
- `tmp/PHASE11_build.log`
- `tmp/PAPER_COMPLETION_DONE.md`
- `tmp/CKPT_MANIFEST.csv`

**Acceptance:** A2.0 – A2.6 all check; both git pushes succeed.

**DO NOT:**
- Open a PR. Push directly to `origin master`.
- Push to a tag or non-master branch.

---

## 6. Global failure-handling rules

- **Unrecoverable failure** (dataset corrupt, repo permission lost, hardware
  permanently lost, **code bug discovered**) → write
  `tmp/PAPER_COMPLETION_SOS.md` with traceback + phase + recommended human
  action. Stop. The monitor will detect the SOS and finalize a report.
- **Recoverable failure** → `tmp/PHASE{N}_NOTES.md` row + continue.
- **96-hour budget breach** → finalise PHASE 11 with whatever state exists;
  mark un-run phases TODO in `PAPER_COMPLETION_DONE.md`.
- **MUST_HAVE breach** (PHASE 2 DONE count < 20 at hour 60) → SOS.
- **Monitor agent restart**: if the monitor detects a stall and you wake
  back up via `tmux send-keys`, re-read this doc and resume from the last
  COMPLETED phase (find the latest DONE row in
  `tmp/EXPERIMENTS_DONE.md` and start from PHASE n+1). For PHASE 2
  resumption, consult `tmp/phase2_queue.json` and start from the first
  non-DONE cell.

---

## 7. Out of scope (do NOT do)

- Add new methods, new theorems, new datasets.
- Change paper title, author block, abstract claims, or method name (SaKD).
- Re-render `sources/framework.pdf` (user-curated).
- Patch `src/sagd/` (write SOS instead).
- Modify `tests/`, `README.md`, `CLAUDE.md`.
- Open a PR (push directly to `origin master`).
- Use non-canonical hyperparameters per cell in PHASE 2 (only `seed`
  varies).
- Evict another tenant from a GPU.

---

## 8. How the human will consume your output

After notification, the human will:
1. Read `tmp/PAPER_COMPLETION_DONE.md`.
2. Read `tmp/CKPT_MANIFEST.csv` to verify the training matrix.
3. Spot-check `writing/NeurIPS26-SaGD/neurips_2026.pdf`.
4. For each PARTIAL phase, decide accept-or-followup.
5. Pull from Overleaf and re-publish.

Your job ends when the human can do these 5 steps without needing to ask
any clarifying question.

---

## 9. Companion: monitor agent

A separate agent reads `docs/paper_completion_monitor.md`. Its job is to
babysit you (detect stalls, restart, summarise progress to the human). You
never read or write the monitor's files; the monitor never modifies your
deliverables. If you see `tmp/MONITOR_*.md` files appear, ignore them.

---

## 10. JSON schemas index (for cross-phase consistency)

All `tmp/phase*.json` and `tmp/*.json` deliverables follow the
schemas embedded in their producing phases (see PHASE 2, 3, 4, 5, 6, 7).
Every JSON file MUST include a top-level `"schema_version": "<name>_v<n>"`
key so the monitor and any post-hoc analysis can detect format drift.

Phase → schema version table:
- PHASE 0: no schema (just JSON dumps of nvidia-smi / df output)
- PHASE 1: `phase1_caches` (ad-hoc)
- PHASE 2: `phase2_queue_v1`
- PHASE 3: `ec_per_sample_v1`
- PHASE 4: `ablation_v1`
- PHASE 5: `hp_v1`
- PHASE 6: `dyn_v1`
- PHASE 7: `bench_v1`
- PHASE 8: ad-hoc (raw saliency arrays)
- PHASE 8.5: ad-hoc (motivation pre-aggregated)
- PHASE 8.7: ad-hoc (compute cost per method)

If a downstream phase reads a JSON whose `schema_version` does not match
expectation, halt and SOS.
