# Supplementary Experiments E1–E4 — autonomous server task doc

**Purpose:** Four discussion-period supplementary experiments on the
LLaMA-3.1-8B → LLaMA-3.2-1B pair. Results feed time-critical discussion
replies — treat as deadline-critical. Numbers will be quoted verbatim in
public responses, so the honesty rule below is absolute.

**HONESTY RULE:** Report every number exactly as measured. A negative or
inconclusive result is a valid deliverable — write it down as-is. Never
tune, rerun-until-favorable, or omit a completed measurement.

**Priority order (execute in this order):** E2 → E4 → E1 → E3.
E2/E4 are eval-only on existing checkpoints (~2–3 GPU-h total) and are the
highest-value results. E1 is one training run. E3 is six training runs and
may be truncated if time runs out (see §6).

Total budget: ≈ 7–9 GPU-hours on one A100.

---

## 0. Environment & GPU policy

- Repo: `/data/tianhao/SaGD`, conda env `py310_torch24`.
- `export PYTHONPATH="$(pwd)/src:$PYTHONPATH"` before any python invocation.
- Shared GPUs: pick a GPU with ≥ 70 GB free via `nvidia-smi`; single-GPU
  only; NEVER kill or contend with other tenants' processes. If no GPU is
  free, poll every 15 min and log the wait.
- All outputs go under `tmp/rebuttal_experiments/`. Create it first.
- Append one line to `tmp/rebuttal_experiments/PROGRESS.log` at every
  start/finish/failure: `HH:MM:SS <exp> <event>`.
- On any single-run failure: retry once; on second failure mark the
  experiment `BLOCKED` in RESULTS.md with the full error and move to the
  next priority item. Do not debug indefinitely.

## 1. Preflight inventory (15 min, no GPU)

Verify and record in `tmp/rebuttal_experiments/PREFLIGHT.md`:

1. Checkpoints (per `tmp/EXPERIMENTS_DONE.md` these exist):
   - `outputs_dolly/llama_1B/standard_kd/seed_*/student_final.pt`
   - `outputs_dolly/llama_1B/sagd/seed_*/student_final.pt`
   List every `<method>/<seed>` actually present; E2 uses all of
   {standard_kd, sagd} seeds found.
2. Teacher saliency cache: `data/teacher_saliency_dolly_llama.pt`
   (metadata must show LLaMA tokenizer, dolly, max_seq_len=512).
3. Scripts: `scripts/motivation_experiments.py` (pushed together with this
   doc — `git pull` first), `scripts/diagnose_saliency.py`,
   `scripts/train.py`, the LLaMA train wrapper `scripts/a100_llama_train.sh`,
   and whatever eval path produced each training cell's `eval.json`
   (reuse it verbatim for E1/E3 ROUGE-L).
4. Per-sample EC file: `tmp/ec_per_sample.json` (produced by the earlier EC
   run; contains `per_sample_teacher_ec` / `per_sample_student_ec` arrays
   for SQuAD val). Needed by E4b.
5. NaN-check is NOT required (already passed per EXPERIMENTS_DONE.md).

If any item is missing, log it in PREFLIGHT.md, mark dependent experiments
BLOCKED, and continue with the rest.

---

## 2. E2 — Local-robustness evaluation (eval-only, ~1.5–2 GPU-h)

**Question:** Does SaKD actually reduce the teacher–student gap in input
neighborhoods, relative to standard KD?

**Protocol** (same convention as the paper's Figure 1 / motivation run):
for each student checkpoint M ∈ {standard_kd, sagd} × available seeds, on
the SAME 500 Dolly validation samples (fixed split, seed 42), isotropic
Gaussian embedding noise at relative scale σ ∈ {0.005, 0.01, 0.02}·‖e‖,
8 noise draws per sample:

- per-sample clean KL `X_i = KL(f_T(x_i) ‖ f_S(x_i))`
- per-sample degradation `Y_i = E_ξ[KL(f_T(x_i+ξ) ‖ f_S(x_i+ξ))] − X_i`

Use `scripts/motivation_experiments.py` (subcommand `m1`) once per
(checkpoint, σ); it already computes X/Y and writes a JSON sidecar with
per-sample numbers. Pass the student checkpoint path; teacher is
LLaMA-3.1-8B. Keep batch small (2) to fit alongside tenants.

**Report per (method, seed, σ)** in `tmp/rebuttal_experiments/E2_table.md`
(+ raw `E2_robustness.json`):
1. 95th percentile of perturbed KL `X_i + Y_i`
2. mean perturbed/clean ratio
3. failure-cluster fraction — use the SAME definition as the original
   Figure-1 JSON/plot code (read it from the script; do not invent a new
   threshold). If the original definition is parameterized, record the
   parameters used.
Also report the across-seed mean for each method at each σ.

**Success criterion:** SaKD < KD-KL on (1) and (3) at every σ. If any cell
goes the other way, report it unchanged.

## 3. E4 — Saliency divergence ↔ vulnerability correlation (eval-only, ~1 GPU-h)

**E4a (Dolly, uses E2 outputs).** For the standard_kd seed-42 student on
the same 500 Dolly val samples:
- per-sample saliency JSD_i between teacher and student: teacher saliency
  from `data/teacher_saliency_dolly_llama.pt`; student saliency computed
  with the repo's `SaliencyComputer.compute()` (same masking rules as
  training: prompt positions only, attention-masked); normalize both with
  the training convention (softmax over prompt positions, τ_s = 2.0);
  JSD between the two distributions.
- vulnerability_i = Y_i from E2 at σ = 0.01 (primary; also report 0.005
  and 0.02 as robustness of the conclusion).
- Report Spearman ρ and p-value, n = 500. Scatter-plot PDF optional.

**E4b (SQuAD, EC deviation).** For the standard_kd seed-42 student:
- |EC_i − EC_teacher,i| per sample from `tmp/ec_per_sample.json`.
- vulnerability on the SAME SQuAD val samples: run the E2 protocol once on
  SQuAD val (500 samples, σ = 0.01, 8 draws) for this checkpoint. If
  `motivation_experiments.py` only supports the instruction dataset, adapt
  it minimally to load `SquadDataset` (repo `src/sagd/data.py`) — keep the
  KL computation identical; ~30 min extra.
- Report Spearman ρ and p-value.

**Output:** `tmp/rebuttal_experiments/E4_correlation.json` + a 5-line
summary in RESULTS.md. A weak or negative ρ is reported as-is (it changes
paper §3.3 framing, not this task).

## 4. E1 — Uniform vs saliency-adaptive noise, equal budget (1 training run, ~1 GPU-h)

**Code change (small, guarded):** add a `--uniform_noise` flag to
`scripts/train.py`, passed through to the trainer. In the SaKD noisy-KL
branch of `src/sagd/trainer.py`, when the flag is set, SKIP the
saliency-adaptive per-position allocation (Eq. 11 path) and use the
uniform scale `σ_j = σ0 · ‖e‖` at every position — i.e., force the
adaptive ratio to 1 everywhere. Everything else (λ, σ0, τ_w, N,
reweighting, seeds) unchanged, so the total noise budget matches the
adaptive run by construction of Eq. 11's mean normalization. Commit the
change with a clear message.

**Run:** LLaMA pair, Dolly, seed 42, identical hyperparameters to the main
sagd cell (mirror the sagd invocation in `scripts/a100_llama_train.sh`,
add `--uniform_noise`), output to
`outputs_dolly/llama_1B/sagd_uniform/seed_42/`.

**Eval:** (a) ROUGE-L on DollyEval / S-NatInst / Unnatural via the same
eval path as the main table; (b) EC on SQuAD val via
`scripts/diagnose_saliency.py` (teacher computed on the fly, same as the
earlier EC run).

**Output:** `tmp/rebuttal_experiments/E1_table.md` — three rows at seed 42:
KD-KL / SaKD-uniform / SaKD-adaptive; columns: 3 benchmarks + avg + EC.
Whatever the ordering comes out to be, report it.

## 5. E3 — Hyperparameter sensitivity mini-sweep (6 training runs, ~3.5 GPU-h)

One-at-a-time around defaults (λ=0.5, σ0=0.005, τ_w=1.0, N=5), LLaMA pair,
Dolly, seed 42, method sagd:

| Run | λ | σ0 | τ_w |
|----|-----|-------|-----|
| 1 | 0.1 | 0.005 | 1.0 |
| 2 | 2.0 | 0.005 | 1.0 |
| 3 | 0.5 | 0.001 | 1.0 |
| 4 | 0.5 | 0.02  | 1.0 |
| 5 | 0.5 | 0.005 | 0.1 |
| 6 | 0.5 | 0.005 | 10  |

The default cell (0.5, 0.005, 1.0) is the EXISTING sagd seed-42 run — do
not retrain it. Output dirs: `outputs_dolly/llama_1B/sagd_hp_<name>/seed_42/`.
Eval: ROUGE-L on the 3 benchmarks, same eval path as the main table.

**Output:** `tmp/rebuttal_experiments/E3_sensitivity.md` — per-axis table
(value → 3 benchmarks + avg) and ONE line per axis stating: flat or
peaked, and the observed safe range.

**Truncation rule:** if wall-clock runs out, complete axes in the order
λ → σ0 → τ_w and mark the rest PARTIAL.

---

## 6. Results contract

Aggregate everything into `tmp/rebuttal_experiments/RESULTS.md`:

1. One section per experiment: status (DONE / PARTIAL / BLOCKED), wall
   time, the tables above, and a 2–3 sentence plain-language takeaway
   each. Takeaways must be supported by the numbers in the same file.
2. Append one row per experiment to `tmp/EXPERIMENTS_DONE.md`
   (`| E1 | DONE | 55 min | uniform vs adaptive | ... |` style).
3. Copy the raw JSONs alongside (E2_robustness.json, E4_correlation.json).
4. Everything under `tmp/` — do NOT commit results; the local author pulls
   them manually.

**Acceptance bar:** E2 and E4 DONE (both are eval-only — there is no
GPU-budget excuse); E1 DONE; E3 at least the λ axis DONE.
