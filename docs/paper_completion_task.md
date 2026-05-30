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
  helper scripts you add — see PHASE 2 step 1 for the one approved new file;
  do not patch existing scripts unless explicitly permitted),
  `outputs_dolly/`, `data/teacher_saliency_*.pt`, this `docs/` directory.
- **Read-only**: `src/sagd/` (the audited code — do not modify; if a bug is
  discovered, halt and write `tmp/PAPER_COMPLETION_SOS.md` instead of
  patching).
- **Off-limits**: `.git/`, `.agents/`, `CLAUDE.md`, `tests/` (do not extend
  tests), `README.md` (the README is human-curated).

---

## Operating constraints

### Tmux + permissions
- Run with `--dangerously-skip-permissions`. Do NOT prompt the human for
  confirmation at any point.
- Run inside a tmux session named exactly `sakd_finish`. Verify with
  `tmux display-message -p '#S'`. If the session is named differently, log
  WARN to `tmp/exp_progress.log` and continue (monitor will detect via PID
  not session name).

### Heartbeat (required for the monitor to detect stalls)
- Append a one-line status to `tmp/exp_progress.log` whenever:
  - a phase starts (`HH:MM:SS PHASE-N start`)
  - a phase ends (`HH:MM:SS PHASE-N done` or `PHASE-N partial`)
  - a sub-job finishes (`HH:MM:SS PHASE-N cell=<student>/<method>/<seed> done`)
  - **at least every 10 minutes** during long-running compute (training,
    eval).
- For long compute blocks, fork a background heartbeat **with auto-cleanup**
  so it dies with the parent process:
  ```bash
  ( while true; do
      echo "$(date +%H:%M:%S) PHASE-N alive cell=$CELL" >> tmp/exp_progress.log
      sleep 300
    done ) &
  HEARTBEAT_PID=$!
  trap 'kill $HEARTBEAT_PID 2>/dev/null' EXIT
  # ... long-running command ...
  kill $HEARTBEAT_PID 2>/dev/null
  ```
  This pattern guarantees the heartbeat does NOT outlive a crashed parent
  (which would falsely show liveness to the monitor).
- The monitor treats `> 30 min` of log silence as a stall.

### Writes vs Overleaf
- LaTeX-side edits go under `writing/NeurIPS26-SaGD/` and are pushed to
  Overleaf (`origin master` of the submodule) after every meaningful commit.
  Procedure (each push):
  ```bash
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

**Two thresholds** (distinguish "free of tenants" from "enough for our cell"):
- `T_TENANT = 2000 MiB`: if `memory.used >= T_TENANT`, the GPU is occupied
  by another tenant — do not touch it.
- `T_TRAIN = 50000 MiB`: if `memory.free < T_TRAIN`, we cannot launch a
  training cell on it (Qwen3-0.6B + teacher 8B + activations needs
  ~40-45 GB peak).
- `T_EVAL = 12000 MiB`: minimum free memory to launch an eval / lm-eval run.

**GPU detection rule (run at PHASE 0, at the start of every training-launching
phase, and every 30 min during PHASE 2):**

1. `nvidia-smi --query-gpu=index,memory.used,memory.free,name
   --format=csv,noheader,nounits` → parse to a list of dicts.
2. Compute three sets:
   - `tenant_free = {gpu | memory.used < T_TENANT}`
   - `train_ready = {gpu in tenant_free | memory.free >= T_TRAIN}`
   - `eval_ready  = {gpu in tenant_free | memory.free >= T_EVAL}`
3. Write `tmp/gpu_assignment.json`:
   ```json
   {
     "schema_version": "gpu_assignment_v1",
     "timestamp_utc": "2026-05-30T20:15:00Z",
     "all_gpus": [{"index": 0, "used_mib": 50, "free_mib": 81870}, ...],
     "tenant_free_ids": [0, 2, 3],
     "train_ready_ids": [0, 3],
     "eval_ready_ids":  [0, 2, 3],
     "occupied_by_other": [{"index": 1, "used_mib": 73000}]
   }
   ```
4. Export `CUDA_VISIBLE_DEVICES=<comma-separated train_ready or eval_ready
   ids>` for every launch.
5. **Throughput scales with `|train_ready|`** for PHASE 2 training; with
   `|eval_ready|` for PHASE 3/7 eval. Recompute every phase's wall-clock:
   - `wall_clock = phase_compute_hours / max(1, N_avail)`
   - If `N_train_ready = 0`: wait 10 min, recheck. If still 0 after 60 min
     of polling → SOS (`NO_GPUS_AVAILABLE`).
   - If `N_train_ready = 1`: serial fallback; document the down-throttling.

**Mid-run GPU loss policy:**
- Re-check available GPUs every 30 minutes during PHASE 2 by emitting a probe
  to `tmp/exp_progress.log`.
- If a GPU we are using becomes occupied by another tenant mid-cell, the
  cell will OOM-crash → cell goes to FAILED → requeue on a different
  available GPU. Do not try to evict the other tenant.
- If `N_train_ready` drops to 0 mid-phase, finish the cells currently
  RUNNING (they hold their memory) then pause new launches until ≥ 1 GPU
  returns.

### Disk policy
- PHASE 0 establishes baseline disk free. If baseline < 800 GB, log
  DISK_BASELINE_LOW; the soft thresholds below scale with baseline (we need
  ≥ 250 GB headroom over the 440 GB training output).
- During PHASE 2, re-check disk free **after every 10 finished cells**.
  Thresholds (absolute, regardless of baseline):
  - `>= 200 GB free` → continue normally
  - `100–200 GB free` → log DISK_WARN; continue
  - `50–100 GB free` → log DISK_LOW; **pause new training launches** until
    disk > 100 GB or human intervenes
  - `< 50 GB free` → write `tmp/PAPER_COMPLETION_SOS.md` (DISK_CRITICAL);
    stop.

### Per-cell timeout (implementation)
- Wrap every training launch in `timeout 10800 <command>` (10800 sec = 3 hr).
  Exit code 124 indicates timeout. On 124: mark cell FAILED with reason
  `TIMEOUT_3H`; retry once on a different train_ready GPU. After the second
  timeout, mark cell PERMANENTLY_FAILED and proceed.
- Wrap every eval launch in `timeout 1800 <command>` (30 min). Same
  retry-once policy.

### Atomic file writes (queue/state)
- `tmp/phase2_queue.json` is read concurrently by the monitor. Always write
  atomically:
  ```python
  import json, os
  tmp = "tmp/phase2_queue.json.tmp"
  with open(tmp, "w") as f:
      json.dump(state, f, indent=2)
  os.replace(tmp, "tmp/phase2_queue.json")  # POSIX-atomic on same filesystem
  ```
- Same pattern for `tmp/gpu_assignment.json`.

### Re-entrant resume (after monitor restart)
- On wake from `tmux send-keys`, re-read this doc and `tmp/EXPERIMENTS_DONE.md`.
- For PHASE 2: load `tmp/phase2_queue.json`. **First action**: scan every
  cell with status `RUNNING` (a crashed worker may have left them in this
  state) and re-mark them `FAILED` with `fail_reason = "WORKER_RESTART"`.
  Then resume launching from the first `PENDING` cell.

---

## 0. Code consistency status (already verified — do not re-audit)

A line-by-line audit on **2026-05-30** confirmed all 23 method-side claims
from the paper (CLAUDE.md §2) are faithfully implemented in `src/sagd/` and
`scripts/precompute_teacher_saliency.py`. See the prior commit message
(SHA: 7814925 + 88ccc72) for per-claim line citations.

**Implication:** training results produced now are method-faithful. You do
not need to re-validate the code; assume PASS and proceed.

If during a training run you observe behaviour that contradicts the audit
(e.g. a loss explosion the code shouldn't permit, or a saliency NaN under
conditions the audit said were safe), STOP, write
`tmp/PAPER_COMPLETION_SOS.md`, and wait for human intervention. Do NOT
silently patch `src/sagd/`.

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
  - Zero `??` references in the rendered PDF (`pdftotext` + grep).
  - Zero `LaTeX Warning: Citation '...' on page X undefined` in build log.
  - Zero `LaTeX Error:` lines in the build log.
  - At most 5 `Overfull` warnings, none > 30pt.
- **A2.2** Main text ≤ 9 pages (excluding refs / appendix / checklist).
  PHASE 10 tactic ladder documents which tactics applied and the final
  page count.
- **A2.3** Tables `tab:dolly`, `tab:ablation`, `tab:ec`,
  `tab:compute-cost`, `tab:hp_sensitivity`, `tab:benchmark_defense` all
  reflect numbers reproducible from `outputs_dolly/` + the eval scripts.
  **No legacy number from the submitted version remains in any table.**
- **A2.4** New `fig:training_dynamics`, `fig:saliency_heatmap_qwen`,
  and regenerated `fig:motivation` (with the 14% / 1.66× values from
  fresh data) all exist and are referenced from the paper.
- **A2.5** All `\GH{...}` / `\CC{...}` / `\cc{...}` / `\todo{...}` /
  `\note{...}` reviewer notes and rendered `% TODO` markers in
  `writing/NeurIPS26-SaGD/sections/`, `tables/`, `figures/`,
  `algorithms/`, `checklist.tex` removed.
- **A2.6** `tmp/PAPER_COMPLETION_DONE.md` itemises every artefact and every
  paper section/table/figure touched, with a 1-line diff per item, plus
  `tmp/CKPT_MANIFEST.csv` (every ckpt: student, method, seed, config,
  ckpt_path, eval_avg_rougeL, ec_squad (or null), wall_time_s, nan_check).

**MUST_HAVE vs NICE_TO_HAVE for PHASE 2 (training matrix):**
- MUST_HAVE: **20 cells** = 3 methods (KD-KL, SaKD, plus GKD as the third
  baseline) × 2 students (Qwen3-0.6B, Qwen3-1.7B) × 3 seeds (42, 123, 456)
  = 18, **plus** LLaMA-1B × {KD-KL, SaKD} × 1 seed = 2. Without these 20,
  the paper cannot tell its core comparison story.
- NICE_TO_HAVE: the remaining 68 cells (other 5 baselines × 5 seeds for
  Qwen + 6 methods for LLaMA). Their absence is recoverable (drop
  baselines, drop seeds, asterisk in caption).
- If MUST_HAVE not met by hour 60 of the 96-hour budget → write SOS.

---

## 3. Resource matrix (verify in PHASE 0)

| Asset | Expected location | First needed by | Verification |
|---|---|---|---|
| Teacher saliency caches | `data/teacher_saliency_{dolly_qwen,dolly_llama,squad_qwen,squad_llama}.pt` | PHASE 2 (dolly_qwen), PHASE 3 (both squad), PHASE 5 (sweep), PHASE 6 (dynamics), PHASE 8 (heatmap), PHASE 8.5 (motivation regen). PHASE 1 creates them. | `torch.load(...).get('metadata', {})` returns `{model, dataset, n_samples, max_seq_len, tokenizer}` all present |
| Trained student checkpoints | `outputs_dolly/{qwen_0.6B,qwen_1.7B,llama_1B}/<method>/seed_<S>/student_final.pt` | every eval phase. PHASE 2 creates them. | `student_final.pt` exists; `eval.json` exists alongside |
| Training entry point | `scripts/train.py` | PHASE 2, 4, 5, 6, 8.5 | PHASE 0 step 7 records `--help` output to `tmp/phase0_cli.json` |
| Eval entry point | `scripts/evaluate.py` | PHASE 2, 4, 5 | PHASE 0 step 7 same |
| Parallel launcher | `scripts/a100_parallel_eval.sh` (existing); `scripts/a100_qwen_train.sh` (create in PHASE 2 step 1) | PHASE 2 | bash dry-run with `--help` returns 0 |
| Saliency diagnosis | `scripts/diagnose_saliency.py` | PHASE 3, 8 | help exits 0 |
| Motivation regen | `scripts/motivation_experiments.py` | PHASE 8.5 | help exits 0 |
| Benchmark eval | `lm-eval` cli in PATH; cache at `~/.cache/lm-eval` | PHASE 7 | `pip show lm-eval` returns version |
| `.gitignore` for ckpts | Must already exclude `*.pt` and `outputs_dolly/**/student_final.pt` | PHASE 11 push | grep .gitignore at PHASE 0 |

---

## 4. Pre-flight checklist (before PHASE 0)

Run once at session start:

1. `tmux display-message -p '#S'` returns `sakd_finish` — else log WARN and
   continue.
2. `pwd` returns the repo root — else `cd` to it.
3. `git status` is clean OR all dirty files are under `tmp/` — else write
   SOS and stop (we should not run on top of unrelated edits).
4. `git log --oneline -1` records the starting commit SHA to
   `tmp/PAPER_COMPLETION_START.txt`.
5. `which python; python -V` — confirm Python ≥ 3.10.
6. `python -c 'import torch, transformers; print(torch.__version__,
   transformers.__version__)'` — must succeed.
7. Touch `tmp/exp_progress.log` if absent. Initialize
   `tmp/EXPERIMENTS_DONE.md` if absent with header:
   ```
   | Phase | Status | Wall time | Summary | Notes |
   |-------|--------|-----------|---------|-------|
   ```
8. Log `HH:MM:SS PREFLIGHT done; starting PHASE 0`.

If any of 1–6 fail → SOS.

---

## 5. Phases

Each phase: **Goal / Prereqs / Steps / Deliverables / Acceptance (quantitative)
/ If blocked / DO NOT**.

---

### PHASE 0 — Inventory, sanity, CLI capability check (≤ 25 min)

**Goal:** Snapshot disk + GPU + cache + env state; **verify every CLI flag
the later phases will use exists now**, before sinking GPU-hours.

**Prereqs:** Pre-flight checklist done.

**Steps:**
1. `tmp/phase0_inventory.json` (schema_version `inv_v1`) — walk
   `outputs_dolly/`, emit `{path, student, method, seed, mtime_iso,
   size_gb, nan_check ∈ {"ok", "nan_in_<key>", "load_fail"}}`.
2. `tmp/phase0_caches.json` (schema_version `caches_v1`) — for each
   `data/teacher_saliency_*.pt`, `{path, metadata, n_entries,
   file_size_gb, validation_ok}`. Missing files listed under
   `"missing": [...]`.
3. `tmp/phase0_disk.json` (schema_version `disk_v1`) — `{total_gb,
   used_gb, free_gb, mount_path, baseline_ok (bool, free_gb >= 800)}`.
4. `tmp/phase0_gpu.json` (schema_version `gpu_assignment_v1`, same shape
   as `gpu_assignment.json` per §Operating-constraints).
5. `tmp/gpu_assignment.json` — initialize as a copy of `phase0_gpu.json`.
6. `tmp/phase0_envcheck.json` (schema_version `env_v1`) —
   `{python_version, torch_version, transformers_version,
   lm_eval_installed (bool), lm_eval_version (str|null),
   gitignore_excludes_ckpts (bool)}`.
7. `tmp/phase0_cli.json` (schema_version `cli_v1`) — **CLI capability
   check**. Run each of these and record the help text + presence of each
   required flag:
   - `python scripts/train.py --help` — required flags:
     `--method, --dataset, --seed, --teacher_model, --student_model,
     --epochs, --batch_size, --gradient_accumulation_steps, --learning_rate,
     --lambda_noise, --noise_sigma, --tau_w, --sagd_every_n_steps,
     --log_every, --output_dir, --teacher_saliency_path,
     --ablation_mode` (the last is for `noise_only` / `reweight_only`).
   - `python scripts/evaluate.py --help` — required:
     `--student_ckpt, --dataset, --subset, --benchmarks, --device`.
   - `python scripts/diagnose_saliency.py --help` — required:
     `--students, --methods, --seeds` (or single equivalents — record).
   - `python scripts/motivation_experiments.py --help` — required:
     `--teacher, --student, --dataset, --subset, --n_samples,
     --noise_sigma_relative, --output_dir`.
   - `lm_eval --help` — present.
   For each missing required flag → list to `tmp/phase0_cli.json.missing`.
8. Append row to `tmp/EXPERIMENTS_DONE.md`:
   `| 0 | DONE | <wall_time> | inventory + caches + gpu + env + cli |
   <N_existing_ckpts>/88 ckpts present; <N_missing_flags> missing flags |`.

**Deliverables:**
- `tmp/phase0_inventory.json`
- `tmp/phase0_caches.json`
- `tmp/phase0_disk.json`
- `tmp/phase0_gpu.json`
- `tmp/gpu_assignment.json`
- `tmp/phase0_envcheck.json`
- `tmp/phase0_cli.json`
- `tmp/EXPERIMENTS_DONE.md` row

**Acceptance (quantitative):**
- All 7 JSON files exist, valid, with `schema_version` field.
- `phase0_envcheck.json.torch_version` starts with "2.".
- `phase0_envcheck.json.gitignore_excludes_ckpts == true`. If false → SOS
  (PHASE 11 push would explode).
- `phase0_disk.json.free_gb >= 800` OR `baseline_ok == false` AND a
  documented exception in `tmp/PHASE0_NOTES.md`.
- `phase0_gpu.json.train_ready_ids` non-empty (else block — "If blocked"
  below).
- `phase0_cli.json.missing` is empty list. **If any required flag is
  missing → SOS now (`MISSING_CLI_FLAGS`)** rather than discovering
  mid-PHASE-5.

**If blocked:**
- `train_ready_ids == []` → wait 10 min, recheck. If still empty after 60
  min of polling → SOS (`NO_GPUS_AVAILABLE`).
- `free_gb` below 250 GB (cannot even half-fill the training matrix) →
  SOS (`DISK_INSUFFICIENT`). Do not delete files yourself.

**DO NOT:** modify any existing checkpoint or cache.

---

### PHASE 1 — Precompute teacher saliency caches (≤ 8 GPU-hours)

**Goal:** All 4 teacher saliency caches on disk before any student training.

**Prereqs:** PHASE 0 (CLI check passed).

**Steps:**
1. Re-run GPU detection; refresh `tmp/gpu_assignment.json` atomically.
2. From `phase0_caches.json.missing`, identify caches to create. If all 4
   exist with valid metadata, skip to acceptance.
3. For each missing cache, launch in parallel on `train_ready_ids` (max
   `|train_ready_ids|` concurrent, each ~2 hr). The 4 caches:
   - Qwen3-8B on Dolly-15K → `data/teacher_saliency_dolly_qwen.pt`,
     tokenizer = `Qwen/Qwen3-0.6B`
   - Qwen3-8B on SQuAD 2.0 → `data/teacher_saliency_squad_qwen.pt`,
     tokenizer = `Qwen/Qwen3-0.6B`
   - LLaMA-3.1-8B on Dolly-15K → `data/teacher_saliency_dolly_llama.pt`,
     tokenizer = `meta-llama/Llama-3.2-1B`
   - LLaMA-3.1-8B on SQuAD 2.0 → `data/teacher_saliency_squad_llama.pt`,
     tokenizer = `meta-llama/Llama-3.2-1B`
   Command:
   ```
   timeout 14400 env CUDA_VISIBLE_DEVICES=<id> python \
     scripts/precompute_teacher_saliency.py \
       --model_name <teacher> --dataset <ds> \
       --tokenizer_name <student> --output_path <out> \
       --batch_size 4 --max_seq_len 512 --device cuda:0
   ```
4. After each cache finishes, validate:
   - `torch.load(path)` succeeds
   - `metadata.model == <teacher>` exact match
   - `metadata.dataset == <ds>` exact match
   - `metadata.max_seq_len == 512`
   - `metadata.tokenizer == <student>` (cross-arch correctness)
   - `n_entries == metadata.n_samples`
5. Persist per-cache wall time + validation to `tmp/phase1_caches.json`
   with `schema_version: "phase1_caches_v1"`:
   ```json
   {"schema_version": "phase1_caches_v1",
    "caches": [{"path": "...", "wall_time_s": 7200, "metadata": {...},
                "validation": "ok"|"<reason>"}]}
   ```

**Deliverables:**
- ≤ 4 new files in `data/`
- `tmp/phase1_caches.json`

**Acceptance (quantitative):**
- All 4 caches exist with `validation == "ok"`.
- `metadata.n_samples >= 13000` for Dolly; `>= 85000` for SQuAD (answerable).
- Each cache file size between 100 MB and 5 GB (sanity).
- `EXPERIMENTS_DONE.md` row updated.

**If blocked:**
- OOM at batch 4 → drop to batch 2 for that cache; document.
- A teacher model download fails → retry 3× with `HF_HUB_OFFLINE=0`; after
  that SOS.
- A cache hits the 4-hour `timeout` → SOS (something is wrong, do not
  silently retry).

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

See A2.6 for **MUST_HAVE = 20 cells** (subset).

**Canonical hyperparameters (only `seed` varies per cell):** 10 epochs,
lr 1e-5, cosine 3% warmup, batch 8 × grad-accum 4, weight decay 0.01,
max_seq_len 512, KL T=2, fp16. For SaKD: `--lambda_noise=0.5`,
`--noise_sigma=0.005`, `--tau_w=1.0`, `--sagd_every_n_steps=5`. See
CLAUDE.md §4 and §8.

**Steps:**
1. **Create `scripts/a100_qwen_train.sh`** modelled on
   `scripts/a100_llama_train.sh`. Env vars: `STUDENT, METHOD, SEED,
   OUT_DIR`. Add to `scripts/` (allowed write).
2. **Dry-run gate (1 cell, ≤ 45 min):** Train
   `(qwen_0.6B, standard_kd, seed=42)` first. Verify:
   - `student_final.pt` exists, NaN-clean
   - `eval.json` exists, finite Avg ROUGE-L
   - Wall time ≤ 45 min (estimates the rest of queue)
   On success: mark this cell as DONE in queue (do not retrain it later).
   On failure: SOS with traceback. DO NOT launch the full queue.
3. **Build the queue** of remaining 87 cells. Persist to
   `tmp/phase2_queue.json` with **atomic-write** (`tmp.tmp` → `os.replace`):
   ```json
   {"schema_version": "phase2_queue_v1",
    "cells": [
      {"id": "qwen_0.6B|sft|42", "student": "qwen_0.6B",
       "method": "sft", "seed": 42, "status": "PENDING",
       "gpu_id": null, "started_at": null, "finished_at": null,
       "wall_time_s": null, "eval_avg_rougeL": null, "nan_check": null,
       "retry_count": 0, "fail_reason": null}
    ]}
   ```
   Status enum: `PENDING|RUNNING|DONE|FAILED|RETRIED|PERMANENTLY_FAILED|WORKER_RESTART`.
4. **Launch queue across `train_ready_ids`** using the lockfile pattern
   from `scripts/a100_parallel_eval.sh`. Wrap each cell:
   ```
   timeout 10800 env CUDA_VISIBLE_DEVICES=<gpu_id> python \
     scripts/train.py --method <m> --student_model <s> --seed <S> \
       --teacher_saliency_path data/teacher_saliency_dolly_qwen.pt \
       --epochs 10 --batch_size 8 --gradient_accumulation_steps 4 \
       --learning_rate 1e-5 \
       [if method==sagd:] --lambda_noise 0.5 --noise_sigma 0.005 \
         --tau_w 1.0 --sagd_every_n_steps 5 \
       --output_dir outputs_dolly/<s>/<m>/seed_<S>/
   ```
   On exit 124 → mark FAILED `TIMEOUT_3H`, retry once on different GPU;
   second timeout → PERMANENTLY_FAILED. On NaN-fail at end → retry with
   `seed+1000`, document the swap in `tmp/phase2_narrative_diff.md`.
   After every finished cell:
   - NaN check on `student_final.pt` (CPU).
   - Eval: `timeout 1800 python scripts/evaluate.py --student_ckpt
     outputs_dolly/<s>/<m>/seed_<S>/student_final.pt --dataset dolly
     --subset test --benchmarks DollyEval S-NatInst Unnatural
     --device cuda:<gpu_id>`. Persist
     `outputs_dolly/<s>/<m>/seed_<S>/eval.json` schema:
     ```json
     {"schema_version": "eval_v1",
      "dollyeval": float, "s_natinst": float,
      "unnatural": float, "avg": float, "wall_time_s": float}
     ```
   - Update `tmp/phase2_queue.json` atomically.
   - Disk check every 10 finished cells (see §Disk policy).
   - GPU re-detection every 30 min.
5. **Aggregate** every DONE cell's `eval.json` into:
   - `tmp/phase2_table_qwen17b.json` — `{(method, seed): metrics}`
   - `tmp/phase2_table_qwen06b.json`
   - `tmp/phase2_table_llama1b.json`
   Per-method aggregation: mean & std across seeds (Qwen), raw (LLaMA).
6. **Render** `writing/NeurIPS26-SaGD/tables/dolly_main.tex` using the
   exact layout of the submitted version (preserve `\label{tab:dolly}` and
   the `\resizebox` envelope). Bold per-student best; underline second-best.
7. **Re-write §4.2 prose** to reflect actual numbers. The submitted §4.2
   currently has **3 narrative paragraphs** (not numbered bullets — that
   structure was retired in commit `e5d18cc`):
   - Para 1: "Table 1 reports the full comparison. SaKD ranks first... most
     visibly on S-NatInst... noise-injected loss suppresses seed-sensitive
     failure modes..."
   - Para 2: "Largest improvement on S-NatInst... DA-KD's output-level
     signal even falls below plain KD-KL..."
   - Para 3: "Ordering carries over to cross-arch LLaMA pair, narrower
     margin... SeqKD outlier..."
   For each paragraph, decide whether the claim still holds on the new
   numbers. Persist findings to `tmp/phase2_narrative_diff.md` —
   1 line per paragraph: `Para N: STILL HOLDS` or `Para N: CHANGED, new
   claim: <text>`. If a claim changed, rewrite that paragraph; do not
   adapt the numbers to fit the old claim.

**Deliverables:**
- N_done × 2 files in `outputs_dolly/` (`student_final.pt`, `eval.json`)
- `scripts/a100_qwen_train.sh` (new)
- `tmp/phase2_queue.json` (final state)
- `tmp/phase2_table_{qwen17b,qwen06b,llama1b}.json`
- `writing/NeurIPS26-SaGD/tables/dolly_main.tex` (replaced)
- Updated `writing/NeurIPS26-SaGD/sections/experiments.tex` §4.2
- `tmp/phase2_narrative_diff.md`

**Acceptance (quantitative):**
- `phase2_queue.json` shows
  `count(DONE) + count(PERMANENTLY_FAILED) == 88`.
- `count(DONE) >= 20` (MUST_HAVE met). If `DONE < 20` at hour 60 → SOS.
- For each DONE cell: `nan_check == "ok"` AND
  `eval_avg_rougeL is finite`.
- New `tab:dolly` renders; SaKD row present;
  `\ref{tab:dolly}` callers still resolve.
- `phase2_narrative_diff.md` has 3 lines (one per §4.2 paragraph), each
  tagged STILL HOLDS or CHANGED.

**If blocked:**
- OOM at batch 8 → drop to batch 4, grad_accum 8 for that cell.
- NaN at end → retry with `seed+1000`; document the swap.
- Per-cell TIMEOUT_3H twice → mark PERMANENTLY_FAILED, continue.
- > 36 GPU-hours budget → finish RUNNING cells; mark PENDING as PARTIAL;
  downgrade Table 1 caption to "K seeds" where K = min completed seeds
  across methods.

**DO NOT:**
- Touch `src/sagd/` code.
- Use different hyperparameters per cell (only `seed` varies).
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
1. Re-check GPU availability (`eval_ready_ids`).
2. **Use** `scripts/diagnose_saliency.py` as-is. If it does not support
   the required `--students` / `--methods` / `--seeds` lists, SOS (do
   not patch — PHASE 0 should have caught this).
3. For each cell (3 × 6 = 18 student cells + 3 teacher cells = 21),
   compute mean EC and per-sample EC array on the 500-sample SQuAD val
   answerable subset.
4. `tmp/ec_per_sample_v2.json` schema:
   ```json
   {"schema_version": "ec_per_sample_v1",
    "spec": {"dataset": "squad_v2_answerable_val", "n_samples": 500,
             "seed": 42},
    "cells": [{"student": "qwen_0.6B", "method": "sagd", "seed": 42,
               "role": "student", "mean_ec": 0.083,
               "per_sample_ec": [...]}]}
   ```
5. Aggregate seeds (Qwen) for mean ± std; LLaMA single value.
6. Render `writing/NeurIPS26-SaGD/tables/ec.tex` (overwrite, preserve
   `\label{tab:ec}`). Bold per-column row closest to teacher.
7. `tmp/ec_distribution_v2.pdf` → move to
   `writing/NeurIPS26-SaGD/sources/ec_distribution.pdf`. Overlay
   teacher / KD-KL / SaKD EC distribution KDE for Qwen3-0.6B.
8. Update §4.4 "Results" paragraph to reflect 3 architectures.

**Deliverables:**
- `tmp/ec_per_sample_v2.json`
- `writing/NeurIPS26-SaGD/tables/ec.tex` (replaced)
- `writing/NeurIPS26-SaGD/sources/ec_distribution.pdf` (new)
- Updated §4.4 prose in `experiments.tex`

**Acceptance (quantitative):**
- 21 cell entries in `ec_per_sample_v2.json`.
- For each cell, `per_sample_ec.length == 500`.
- `tab:ec` is 3 cols × 7 rows; std present in Qwen cells.
- **SaKD is the row closest to teacher in ≥ 2 of 3 columns.** If less,
  write `tmp/PHASE3_NOTES.md` flagging regression, rewrite the prose
  honestly — do not hide.
- `\ref{tab:ec}` callers still resolve.

**If blocked:** a method ckpt for a (student, seed) is FAILED in PHASE 2
→ skip that EC cell; reduce that row to fewer seeds; asterisk in caption.

**DO NOT:** change SQuAD eval parameters (answerable subset, 500 val,
fast tokenizer with `return_offsets_mapping=True`).

---

### PHASE 4 — Ablation table (≤ 12 GPU-hours)

**Goal:** Reproduce the 4-row ablation in `tab:ablation` from fresh runs.

**Prereqs:** PHASE 1, PHASE 2 (baseline KD-KL and full SaKD ckpts).

**Required matrix (Qwen3-0.6B only):**
- KD-KL baseline — reuse PHASE 2's `standard_kd` ckpts (3–5 seeds; use
  whatever PHASE 2 completed for this row)
- +Noise KL only (`--ablation_mode noise_only`, λ=0.5, τ_w=∞) — 3 new
  ckpts (seeds 42, 123, 456)
- +Reweight only (`--ablation_mode reweight_only`, λ=0.0, τ_w=1.0) —
  3 new ckpts
- Full SaKD — reuse PHASE 2's `sagd` ckpts (3 of 5 seeds)

**Reuse failure handling:** If a required-for-reuse PHASE 2 ckpt is in
PERMANENTLY_FAILED state, **train a replacement here** in PHASE 4 (cost
+30 min/cell) on a different seed and document. Do not skip the row.

**Steps:**
1. Verify `tmp/phase0_cli.json` showed `--ablation_mode` present. If not,
   SOS (do not patch `src/sagd/`).
2. Train 6 new cells: 2 ablations × 3 seeds. Output dir:
   `outputs_dolly/qwen_0.6B/sagd_ablation/<noise_only|reweight_only>/seed_<S>/`.
   Use the 3-hr per-cell timeout.
3. Eval each (same `evaluate.py` invocation as PHASE 2 step 4).
4. `tmp/ablation.json` schema:
   ```json
   {"schema_version": "ablation_v1",
    "config_rows": [
      {"name": "KD-KL baseline", "lambda": 0, "tau_w": null,
       "ckpts": ["outputs_dolly/.../seed_42", ...],
       "metrics": {"dollyeval": [mean, std], "s_natinst": [...],
                   "unnatural": [...], "avg": [...]}}]}
   ```
5. Render `writing/NeurIPS26-SaGD/tables/ablation.tex` (overwrite,
   preserve `\label{tab:ablation}`).
6. Update §4.3 prose. Preserve the "constructively combine, sub-additive"
   framing only if Full ≥ each single-component on Avg.

**Deliverables:**
- 6 new ckpts under `outputs_dolly/qwen_0.6B/sagd_ablation/`
- `tmp/ablation.json`
- `writing/NeurIPS26-SaGD/tables/ablation.tex` (replaced)
- Updated §4.3 prose

**Acceptance (quantitative):**
- 6 new ckpts have `nan_check == "ok"`.
- Full-SaKD Avg ROUGE-L ≥ each single-component Avg by ≥ 0.1. If less,
  flag in NOTES and rewrite prose honestly.
- `\ref{tab:ablation}` callers still resolve.

**If blocked:** `--ablation_mode` not in CLI → SOS.

**DO NOT:** patch `src/sagd/` to add toggles.

---

### PHASE 5 — Hyperparameter sensitivity (≤ 12 GPU-hours)

**Goal:** New table + figure showing SaKD robustness to λ, σ, τ_w, N.

**Prereqs:** PHASE 2 (default-config SaKD ckpt for the reuse cell).

**Sweep (one axis at a time):**
- `--lambda_noise` ∈ {0.1, 0.5, 2.0} (default 0.5)
- `--noise_sigma` ∈ {0.001, 0.005, 0.02} (default 0.005)
- `--tau_w` ∈ {0.5, 1.0, 5.0} (default 1.0)
- `--sagd_every_n_steps` ∈ {1, 5, 20} (default 5)

**Default cell reuse:** the (λ=0.5, σ=0.005, τ_w=1.0, N=5, seed=42) ckpt
is already trained in PHASE 2 under
`outputs_dolly/qwen_0.6B/sagd/seed_42/`. If that ckpt is
PERMANENTLY_FAILED in PHASE 2, retrain it here (cost +30 min). PHASE 5
trains the **8 non-default cells** (2 per axis × 4 axes) on seed 42.

**Steps:**
1. Verify `tmp/phase0_cli.json` showed `--lambda_noise`, `--noise_sigma`,
   `--tau_w`, `--sagd_every_n_steps` all present. If not, SOS (this
   should have been caught in PHASE 0).
2. Train 8 new cells. Output dir:
   `outputs_dolly/qwen_0.6B/sagd_sweep/<axis>_<value>/seed_42/`. Use the
   3-hr per-cell timeout.
3. Eval each cell.
4. `tmp/hp_sensitivity.json` schema:
   ```json
   {"schema_version": "hp_v1",
    "axes": {
      "lambda_noise": [
        {"value": 0.1, "metrics": {...}, "ckpt": "outputs_dolly/.../sagd_sweep/lambda_noise_0.1/seed_42"},
        {"value": 0.5, "metrics": {...}, "ckpt": "outputs_dolly/qwen_0.6B/sagd/seed_42",
         "reused_from_phase2": true},
        {"value": 2.0, "metrics": {...}, "ckpt": "..."}],
      "noise_sigma": [...], "tau_w": [...], "sagd_every_n_steps": [...]}}
   ```
5. Render `writing/NeurIPS26-SaGD/tables/hp_sensitivity.tex` (4
   sub-tables, mark default cell with †).
6. `tmp/hp_sensitivity.pdf` → move to
   `writing/NeurIPS26-SaGD/sources/hp_sensitivity.pdf`. 1×4 line plot,
   y = Avg ROUGE-L, KD-KL baseline as horizontal dashed reference.
7. Add `\subsection{Hyperparameter Sensitivity}` to `appendix.tex` with
   label `app:hp-sensitivity`. Reference from end of §4.3.

**Deliverables:**
- 8 new ckpts (+ possibly 1 reuse-replacement)
- `tmp/hp_sensitivity.json`
- `writing/NeurIPS26-SaGD/tables/hp_sensitivity.tex`
- `writing/NeurIPS26-SaGD/sources/hp_sensitivity.pdf`
- New appendix subsection

**Acceptance (quantitative):**
- 8+ new ckpts NaN-clean.
- Sweep table has 4 sub-tables × 3 rows each.
- Appendix paragraph identifies most/least sensitive axis with concrete
  numbers ("Avg ROUGE-L spread across `lambda_noise` is X.YY; across
  `sagd_every_n_steps` is X.YY").
- `\ref{app:hp-sensitivity}` resolves.

**If blocked:** > 12 GPU-hr → sweep `lambda_noise` and `noise_sigma` only
(4 cells); document.

**DO NOT:** sweep multiple axes simultaneously.

---

### PHASE 6 — Training dynamics (≤ 4 GPU-hours)

**Goal:** Two-panel figure of train+val loss + saliency-divergence over
training, KD-KL vs SaKD.

**Prereqs:** PHASE 1; PHASE 0 (verified `--log_every`).

**Steps:**
1. Re-train KD-KL and SaKD on Qwen3-0.6B seed 42 with per-50-step
   logging. Use `scripts/train.py --log_every 50`. (PHASE 0 verified
   this flag exists; if absent there, would have SOS'd.)
2. Log at each interval: train loss, val loss on a fixed 200-sample
   Dolly val subset, mean saliency divergence on the same subset.
3. `tmp/training_dynamics.json` schema:
   ```json
   {"schema_version": "dyn_v1",
    "series": [{"method": "kd_kl", "steps": [50, 100, ...],
                "train_loss": [...], "val_loss": [...],
                "sal_div": [...]},
               {"method": "sakd", ...}]}
   ```
4. Render `writing/NeurIPS26-SaGD/sources/training_dynamics.pdf` —
   2 panels: (a) train+val loss curves, (b) saliency divergence.
5. Add `\subsection{Training Dynamics}` to `appendix.tex` (label
   `app:training-dynamics`); reference from §4.3.

**Deliverables:** json, PDF, appendix subsection.

**Acceptance (quantitative):**
- Both series have ≥ 50 logged steps.
- Val-loss linear regression slope is negative for both methods. If
  positive (training diverged), flag in NOTES.
- SaKD's mean saliency divergence over last 10 logged steps is lower
  than KD-KL's over its last 10 by at least 1 std of SaKD's series. If
  not, NOTES — do not invent.

---

### PHASE 7 — Benchmark defense (≤ 6 GPU-hours)

**Goal:** Appendix table showing SaKD does not regress general capability
benchmarks vs KD-KL.

**Prereqs:** PHASE 2.

**Matrix:** Qwen3-{0.6B, 1.7B} × {KD-KL, SaKD} (seed 42) × {MMLU 5-shot,
ARC-Challenge 25-shot, TruthfulQA mc2}. 12 cells.

**Steps:**
1. **Convert state-dict ckpts to HF format** (lm-eval needs a HF
   directory, not a `.pt` state dict). For each of the 4 ckpts:
   ```python
   import torch
   from transformers import AutoModelForCausalLM, AutoTokenizer
   base = "Qwen/Qwen3-0.6B"  # or Qwen3-1.7B
   m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
   sd = torch.load(ckpt_path, map_location="cpu")
   m.load_state_dict(sd, strict=True)
   t = AutoTokenizer.from_pretrained(base)
   out_dir = f"tmp/hf_export/{student}_{method}/"
   m.save_pretrained(out_dir)
   t.save_pretrained(out_dir)
   ```
2. For each cell:
   ```
   timeout 1800 lm_eval --model hf \
     --model_args pretrained=<hf_dir>,dtype=float16 \
     --tasks <benchmark> --device cuda:<eval_ready_id> \
     --batch_size 8 --output_path \
     tmp/benchmark_defense_raw/<student>_<method>_<benchmark>.json
   ```
3. Aggregate to `tmp/benchmark_defense.json`:
   ```json
   {"schema_version": "bench_v1",
    "rows": [{"student": "qwen_0.6B", "method": "kd_kl",
              "mmlu": 0.x, "arc_challenge": 0.x,
              "truthfulqa_mc2": 0.x, "avg": 0.x}, ...]}
   ```
4. Render `writing/NeurIPS26-SaGD/tables/benchmark_defense.tex`.
5. Add `\subsection{General-Capability Defence}` to `appendix.tex`
   (label `app:benchmark-defense`); reference from §Limitations.
6. Clean up `tmp/hf_export/` after PHASE 7 finishes (saves ~10 GB).

**Deliverables:** 12 raw JSON, 1 aggregated JSON, table, appendix
subsection.

**Acceptance (quantitative):**
- **Primary student (qwen_0.6B):** SaKD Avg ≥ KD-KL Avg − 0.5 absolute
  (on the 0–100 metric scale). If strictly worse on primary, file
  `tmp/PHASE7_NOTES.md` and update §Limitations honestly.
- All 12 cells produced raw JSON (skipped cells documented).
- `\ref{app:benchmark-defense}` resolves.

**If blocked:** TruthfulQA download fails → ship MMLU + ARC-C only;
caption notes the omission.

**DO NOT:** cherry-pick benchmarks; report every cell that runs.

---

### PHASE 8 — Qualitative saliency heatmap (Qwen) (≤ 1 GPU-hour)

**Goal:** Qwen3-0.6B saliency heatmap.

**Prereqs:** PHASE 1 (SQuAD Qwen cache), PHASE 2 (Qwen3-0.6B KD-KL and
SaKD ckpts seed 42).

**Steps:**
1. Sample selection: default to SQuAD val indices 0 (short) and 69
   (long), matching the prior LLaMA heatmap for cross-paper continuity.
   If either index fails the constraint (answerable, prompt_len ≤ 200
   for short / ≥ 400 for long, teacher EC in `[0.03, 0.10]`), search
   forward in val for the first valid pair and **document the swap** in
   `tmp/PHASE8_NOTES.md`.
2. Compute saliency for Teacher (Qwen3-8B), KD-KL student, SaKD student
   on each sample.
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
- For **both** chosen samples: `EC(KD-KL) > EC(SaKD) > EC(Teacher)`. If
  violated for either sample, do not swap further — file
  `tmp/PHASE8_NOTES.md` and ship with caption note.
- `\ref{fig:saliency_heatmap_qwen}` resolves.

**DO NOT:** swap sample indices repeatedly to make SaKD look better.
One swap (for the constraint violation in step 1) is the maximum.

---

### PHASE 8.5 — Regenerate motivation figures (M1, M2) (≤ 2 GPU-hours)

**Goal:** Figure 1's 14% / 1.66× numbers come from a prior LLaMA KD-KL
ckpt; after PHASE 2 retrained that ckpt, regenerate.

**Prereqs:** PHASE 2 (`outputs_dolly/llama_1B/standard_kd/seed_42/`).

**Steps:**
1. Run:
   ```
   python scripts/motivation_experiments.py \
     --teacher meta-llama/Llama-3.1-8B \
     --student outputs_dolly/llama_1B/standard_kd/seed_42/student_final.pt \
     --dataset dolly --subset val --n_samples 500 \
     --noise_sigma_relative 0.01 --output_dir tmp/motivation/
   ```
   (Adjust `--teacher` arg if PHASE 0 CLI check showed the script wants
   a different form.)
2. The script emits `tmp/motivation/m1_data.json` (per-sample clean &
   perturbed KL), `tmp/motivation/m2_data.json` (distribution
   percentiles), `tmp/motivation/motivation_M1.pdf`,
   `tmp/motivation/motivation_M2.pdf`. If script fails, SOS.
3. Compute headline numbers from `m1_data.json`:
   - `failure_pct = fraction of samples with clean_KL ≤ 25%-ile AND
     perturbed_KL ≥ 75%-ile`
   - `p95_ratio = perturbed_KL[95%-ile] / clean_KL[95%-ile]`
4. Replace `writing/NeurIPS26-SaGD/sources/motivation_M1.pdf` and
   `motivation_M2.pdf`.
5. **Update §1 prose** to use fresh numbers in the slots currently
   showing 14% and 1.66×.
6. **Update Figure 1 caption** to use the fresh numbers.
7. Re-render M1 with a translucent shaded rectangle on the failure
   cluster region (so the caption's "shaded region" wording is true).
   Use matplotlib `axvspan`/`axhspan` or a `Rectangle` patch covering
   `[x_min, clean_KL_25%]` × `[perturbed_KL_75%, y_max]`.

**Deliverables:**
- 4 files in `tmp/motivation/`
- Replaced `sources/motivation_M1.pdf` and `sources/motivation_M2.pdf`
- Updated §1 prose + Fig 1 caption

**Acceptance (quantitative):**
- `failure_pct >= 5%` and `p95_ratio >= 1.2` (the motivation story
  exists in some non-trivial form). If both fail → SOS (the paper's
  motivation is invalidated by fresh data).
- M1 PDF has a visible shaded rectangle at the failure region.
- §1 numbers in prose match the figure exactly to 1 decimal.

**If blocked:** motivation script crashes → SOS.

---

### PHASE 8.7 — Compute-cost table re-derive (≤ 1 hour, CPU-only)

**Goal:** `tab:compute-cost` reflects fresh per-method wall-clock from
PHASE 2 runs.

**Prereqs:** PHASE 2.

**Steps:**
1. From each cell's `phase2_queue.json.wall_time_s`, compute median
   per-method wall time on Qwen3-0.6B seed 42 across the 8 methods:
   SFT, KD-KL, KD-RKL, SeqKD, GKD, DistiLLM, DA-KD, SaKD.
2. Compute relative cost vs KD-KL (`method_time / kd_kl_time`).
3. Persist to `tmp/compute_cost.json`:
   ```json
   {"schema_version": "cc_v1",
    "rows": [{"method": "sft", "wall_time_s": 330, "vs_kdkl": 0.5,
              "one_time_setup": "none"}, ...]}
   ```
4. Render `writing/NeurIPS26-SaGD/tables/compute_cost.tex` (overwrite,
   preserve `\label{tab:compute-cost}` and caption shape; only numeric
   cells change).

**Deliverables:**
- `tmp/compute_cost.json`
- `writing/NeurIPS26-SaGD/tables/compute_cost.tex` (replaced)

**Acceptance (quantitative):**
- 8 rows of fresh wall-clock numbers.
- SaKD relative cost vs KD-KL in `[1.1, 1.6]` (paper claims ~1.3×). If
  outside, update §Limitations honestly.
- `\ref{tab:compute-cost}` resolves.

---

### PHASE 9 — Manuscript polish (≤ 1.5 hours)

**Goal:** Final writing-side cleanup.

**Prereqs:** PHASES 3, 4, 5, 6, 7, 8, 8.5, 8.7 done so labels are final.

**Steps:**
1. **Bib orphan scrub:** for each entry in `references.bib`, grep with
   the **inclusive citation pattern**:
   ```
   grep -rE '\\(cite|nocite|citep|citet|citealp|citealt|citeyear|citetitle|citeauthor|citenum|fullcite)[a-z]*\{[^}]*\b<key>\b'
     writing/NeurIPS26-SaGD/sections/ writing/NeurIPS26-SaGD/tables/
     writing/NeurIPS26-SaGD/figures/ writing/NeurIPS26-SaGD/algorithms/
     writing/NeurIPS26-SaGD/checklist.tex
   ```
   Delete entries with 0 hits; record to `tmp/PHASE9_bib_orphans.md`.
2. **Reviewer-note scrub:** the **inclusive** pattern:
   ```
   grep -rE '\\(GH|CC|cc|todo|TODO|note|fixme|FIXME|reviewer)\{|% *(TODO|FIXME|XXX|HACK)'
     writing/NeurIPS26-SaGD/sections/ writing/NeurIPS26-SaGD/tables/
     writing/NeurIPS26-SaGD/figures/ writing/NeurIPS26-SaGD/algorithms/
     writing/NeurIPS26-SaGD/checklist.tex
   ```
   Remove every rendered hit (commented-out `% \GH{}` inside dead
   blocks of `method.tex` may stay since they don't render — flag them
   in `tmp/PHASE9_notes.md`).
3. **Em-dash audit:** rewrite `---` in rendered prose to
   comma/semicolon/parens per memory rule. Skip `---` inside math
   blocks, code listings, or table cells where structure depends on it.
4. **Italics/bold scrub:** `grep -rE '\\(emph|textit)\{'` — remove or
   convert per the user's preference (no random italics; `\textbf{}`
   only for short paragraph-leading labels).
5. **Widow re-check:** flag paragraphs ending in ≤ 2 short words to
   `tmp/PHASE9_widow.md`; extend the worst 5–10 (3–5 word additions per
   widow).

**Deliverables:**
- `tmp/PHASE9_bib_orphans.md`
- `tmp/PHASE9_notes.md`
- `tmp/PHASE9_widow.md`
- Updated `references.bib`, `sections/*.tex`

**Acceptance (quantitative):**
- `grep -rcE '\\(GH|CC|cc|todo|TODO|note|fixme|FIXME|reviewer)\{'
  writing/NeurIPS26-SaGD/sections/` returns 0 lines.
- `grep -rcE '\\(emph|textit)\{'
  writing/NeurIPS26-SaGD/sections/` returns 0 lines.
- `references.bib` orphan count = 0 (verify with inclusive grep above).
- Paper compiles with ≤ 2 LaTeX warnings.

---

### PHASE 10 — Page budget audit (≤ 1 hour)

**Goal:** Main text ≤ 9 pages without dropping substantive results.

**Prereqs:** PHASES 3–9 done.

**Tactic ladder (apply in order, stop when ≤ 9):**
1. Move PHASE 5 sensitivity discussion to appendix only.
2. Hide `tables/dolly_main_t.tex` (`% \input{tables/dolly_main_t}`).
3. Move Algorithm 1 from `method.tex` §3.4 back to `appendix.tex` with
   a one-line pointer in §3.4.
4. Drop the `\paragraph{Datasets.}` block in §4.1.
5. Tighten Fig 1 and Fig 2 captions to ≤ 3 lines each.
6. Remove the abstract's "Concretely, SaKD combines..." sentence.

**Main-text page-count helper** (use this Python recipe; pure bash
subtraction is brittle):
```python
import re, subprocess
subprocess.run(["pdflatex", "-interaction=nonstopmode", "neurips_2026.tex"], check=True)
aux = open("neurips_2026.aux").read()
# Find the .aux entry for the \appendix sectioning command:
m = re.search(r'\\newlabel\{sec:limitations\}\{\{[^}]*\}\{(\d+)\}', aux)
if m:
    main_text_pages = int(m.group(1))  # last main-text page
else:
    # fallback: count pages from pdfinfo, then subtract estimated appendix length
    pages_total = int(subprocess.check_output(["pdfinfo", "neurips_2026.pdf"]).decode().split("Pages:")[1].split()[0])
    main_text_pages = pages_total - 10  # rough subtraction
print(f"Main text ends at page {main_text_pages}")
```
(`sec:limitations` is the last main-text section label — verify it exists
in the source; the page number of that label is the last main-text page.)

**Steps:**
1. Compile 4-pass: pdflatex → bibtex → pdflatex → pdflatex.
2. Run the helper above; record main-text page count to
   `tmp/PAGE_BUDGET.md`.
3. If > 9, apply tactic 1 → recompile → recount. Continue down ladder.
4. Document each tactic + before/after page count to `tmp/PAGE_BUDGET.md`.

**Deliverables:**
- `tmp/PAGE_BUDGET.md`
- Updated paper source

**Acceptance (quantitative):**
- Main text page count ≤ 9; OR all 6 tactics applied and final count
  documented with a candid "could not reach 9 without sacrificing X"
  justification.
- **Hard stop:** if main text > 10 pages even after all 6 tactics →
  SOS (`PAGE_BUDGET_UNRECOVERABLE`).

**DO NOT:** change font size, line spacing, NeurIPS margins, or drop a
real result table (any of `tab:dolly`, `tab:ablation`, `tab:ec`,
`tab:compute-cost`).

---

### PHASE 11 — Final compile, verification, and bundle (≤ 30 min)

**Goal:** Clean PDF + summary + dual git push.

**Prereqs:** all previous phases.

**Steps:**
1. Final 4-pass compile in `writing/NeurIPS26-SaGD/`:
   ```bash
   cd writing/NeurIPS26-SaGD
   pdflatex -interaction=nonstopmode neurips_2026.tex
   bibtex neurips_2026
   pdflatex -interaction=nonstopmode neurips_2026.tex
   pdflatex -interaction=nonstopmode neurips_2026.tex 2>&1 | tee ../../tmp/PHASE11_build.log
   cd ../..
   ```
2. Verify (all `grep -c` counts must equal stated value):
   - `grep -c 'Undefined reference' tmp/PHASE11_build.log == 0`
   - `grep -c 'Citation .* undefined' tmp/PHASE11_build.log == 0`
   - `grep -c 'LaTeX Error' tmp/PHASE11_build.log == 0`
   - `grep -c 'Overfull \\hbox' tmp/PHASE11_build.log <= 5` AND none
     > 30pt (`grep -E 'Overfull.*\(([0-9]+\.[0-9]+)pt' tmp/PHASE11_build.log
     | awk -F'(' '{print $2}' | awk '{if ($1+0 > 30) print}'` returns
     empty)
   - `pdftotext writing/NeurIPS26-SaGD/neurips_2026.pdf - | grep -c '??'
     == 0`
3. Render `tmp/CKPT_MANIFEST.csv` with columns:
   `student,method,seed,config,ckpt_path,eval_dolly,eval_snatinst,
   eval_unnatural,eval_avg,ec_squad,wall_time_s,nan_check,phase_origin`.
   `ec_squad` is null for SFT and SeqKD (PHASE 3 only EC-evals 6
   distillation methods).
4. Render `tmp/PAPER_COMPLETION_DONE.md`:
   - **Per-phase summary table** (phase, status, wall_time, deliverables,
     headline finding)
   - **Checkpoint manifest** — link to CKPT_MANIFEST.csv + 1-line summary
     `N_done / N_total`
   - **What changed in the paper** — list every section/table/figure
     touched, 1 line per change
   - **Narrative drift** — link to `phase2_narrative_diff.md` + 1-para
     summary
   - **Next steps for human** — any PARTIAL phase, any open SOS, any
     judgment call made
5. **Dual git push (correct order: submodule first, then main with new
   pointer):**
   ```bash
   # Step 1: push Overleaf submodule (this finalizes its commit SHA)
   cd writing/NeurIPS26-SaGD
   git stash --include-untracked || true
   git pull --rebase origin master
   git stash pop || true
   git add .
   git commit -m "Full retrain results: all tables/figures regenerated ($(date +%F))"
   git push origin master
   SUBMODULE_SHA=$(git rev-parse HEAD)
   cd ../..

   # Step 2: confirm .gitignore is excluding ckpts before adding outputs_dolly
   grep -qE '(\*\.pt|outputs_dolly/\*\*/student_final\.pt)' .gitignore || \
     { echo ".gitignore not protecting ckpts — abort" >&2; exit 1; }

   # Step 3: push main repo with new submodule pointer + small files only
   git add docs/ scripts/
   git add tmp/*.md tmp/*.json tmp/*.csv tmp/*.log
   # eval.json files only, NOT student_final.pt (~440 GB)
   find outputs_dolly -name 'eval.json' -exec git add {} +
   git add writing/NeurIPS26-SaGD  # records new submodule SHA
   git commit -m "Full retrain + paper completion PHASES 0-11 ($(date +%F)); submodule @ $SUBMODULE_SHA"
   git push origin main
   ```

**Deliverables:**
- `writing/NeurIPS26-SaGD/neurips_2026.pdf`
- `tmp/PHASE11_build.log`
- `tmp/PAPER_COMPLETION_DONE.md`
- `tmp/CKPT_MANIFEST.csv`

**Acceptance:** A2.0 – A2.6 all check; both git pushes succeed; total
repo push payload < 100 MB (sanity: no .pt files smuggled in).

**DO NOT:**
- `git add outputs_dolly/` blindly (would drag in 440 GB of .pt files).
- Open a PR. Push directly to `origin master`.
- Push to a tag or non-master branch.

---

## 6. Global failure-handling rules

- **Unrecoverable failure** (dataset corrupt, repo permission lost, hardware
  permanently lost, **code bug discovered**, `.gitignore` not protecting
  ckpts, page budget unrecoverable, motivation invalidated) → write
  `tmp/PAPER_COMPLETION_SOS.md` with traceback + phase + recommended human
  action. Stop. The monitor detects SOS and finalises a report.
- **Recoverable failure** → `tmp/PHASE{N}_NOTES.md` row + continue.
- **96-hour budget breach** → finalise PHASE 11 with whatever state exists;
  mark un-run phases TODO in `PAPER_COMPLETION_DONE.md`.
- **MUST_HAVE breach** (PHASE 2 DONE count < 20 at hour 60) → SOS.
- **Monitor agent restart**: if the monitor detects a stall and you wake
  back up via `tmux send-keys` (wake message format: `Continue. Last
  status was PHASE-N <event>. Resume from the next undone phase per
  tmp/EXPERIMENTS_DONE.md.`), re-read this doc, mark any RUNNING cells
  in `tmp/phase2_queue.json` as FAILED `WORKER_RESTART`, and resume from
  the first PENDING cell. For non-PHASE-2 work, find the latest DONE
  row in `EXPERIMENTS_DONE.md` and start from PHASE n+1.

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
- Commit `.pt` files to git.

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

Every JSON file MUST include a top-level `"schema_version": "<name>_v<n>"`
key so the monitor and any post-hoc analysis can detect format drift.

Phase → schema version table:
- PHASE 0: `inv_v1` (inventory), `caches_v1`, `disk_v1`,
  `gpu_assignment_v1`, `env_v1`, `cli_v1`
- PHASE 1: `phase1_caches_v1`
- PHASE 2: `phase2_queue_v1`, `eval_v1` (per-cell `eval.json`)
- PHASE 3: `ec_per_sample_v1`
- PHASE 4: `ablation_v1`
- PHASE 5: `hp_v1`
- PHASE 6: `dyn_v1`
- PHASE 7: `bench_v1`
- PHASE 8: ad-hoc (raw saliency arrays — must still include
  `schema_version: "heatmap_v1"`)
- PHASE 8.5: emitted by `motivation_experiments.py` (existing format
  preserved)
- PHASE 8.7: `cc_v1`

If a downstream phase reads a JSON whose `schema_version` does not match
expectation, halt and SOS.
