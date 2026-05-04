# scripts/

Scripts are organized by **GPU/host target** and **Python entry point vs orchestration shell**.

## Quick Map: which script for which job

| I want to ... | Run |
|---|---|
| Train a single config (any method, any dataset) | `python scripts/train.py --method ... --dataset ...` |
| Re-eval all LLaMA ckpts with 5-bench (A100) | `bash scripts/a100_llama_eval.sh [all\|seeds_42_123\|seed_456]` |
| Re-eval all Qwen ckpts with 5-bench (A100) | `bash scripts/a100_qwen_eval.sh [all\|baselines_only\|curriculum_only\|extension_only]` |
| Run Qwen loss ablation + λ sweep + GPDS-random (A100) | `bash scripts/a100_qwen_extension.sh` |
| Run any LLaMA experiment on 3090 | `CUDA_VISIBLE_DEVICES=N bash scripts/3090/run.sh <task>` (see below) |
| Verify 3090 setup works | `bash scripts/3090/smoketest.sh` then `bash scripts/3090/verify_8bit.sh` |
| Aggregate results into a paper table | `python scripts/aggregate_results.py --root <dir>` |
| Compute teacher saliency cache | `python scripts/precompute_teacher_saliency.py --model_name ... --dataset ... --output_path ...` |
| Compute gradient PCA + curriculum order | `python scripts/gradient_pca_selection.py profile ...` then `python scripts/compute_curriculum.py ...` |
| Diagnose evidence concentration on a ckpt | `python scripts/diagnose_saliency.py --student_ckpt ...` |

## 3090 dispatcher tasks

```
CUDA_VISIBLE_DEVICES=0 bash scripts/3090/run.sh curriculum_sagd_grad
CUDA_VISIBLE_DEVICES=1 bash scripts/3090/run.sh curriculum_kd_grad
CUDA_VISIBLE_DEVICES=2 bash scripts/3090/run.sh curriculum_kd_pocl
CUDA_VISIBLE_DEVICES=3 bash scripts/3090/run.sh curriculum_sagd_pocl
CUDA_VISIBLE_DEVICES=4 bash scripts/3090/run.sh samsum
CUDA_VISIBLE_DEVICES=5 bash scripts/3090/run.sh gsm8k
CUDA_VISIBLE_DEVICES=6 bash scripts/3090/run.sh loss_ablation
CUDA_VISIBLE_DEVICES=7 bash scripts/3090/run.sh lambda_sweep
```

All tasks are skip-if-exists idempotent. `bash scripts/3090/run.sh help` shows full task list.

## Layout

```
scripts/
├── README.md                       this file
│
├── # Python entry points
├── train.py                        ★ training entry (all methods, all datasets)
├── evaluate_benchmarks.py          ★ 5-benchmark eval (DA-KD-style)
├── evaluate.py                     legacy single-benchmark eval
├── generate_responses.py           dump generated responses to JSONL (for GPT judge etc.)
├── aggregate_results.py            ★ aggregate benchmark_rouge.json → table
├── precompute_teacher_saliency.py  build teacher saliency cache (.pt)
├── gradient_pca_selection.py       Count Sketch teacher gradient profiling + per-PC select
├── compute_curriculum.py           PC-score-based curriculum order
├── diagnose_saliency.py            evidence concentration + per-method saliency stats
├── verify_saliency_pca.py          sanity-check PCA decomposition vs null
├── gpt_judge.py                    GPT-as-judge win-rate over response JSONL
│
├── # A100 (uvavast) shell entry points
├── a100_llama_eval.sh              5-bench re-eval for all LLaMA ckpts
├── a100_qwen_eval.sh               5-bench re-eval for all Qwen ckpts
├── a100_qwen_extension.sh          loss ablation + λ sweep + GPDS-random training
│
├── # A100 historical training scripts (already ran; kept for reproducibility)
├── llama_prereq.sh                 LLaMA teacher saliency + gradient profiling
├── llama_baselines_gpu1.sh         LLaMA baselines part 1 (sft/kd/rkl/seqkd)
├── llama_baselines_gpu2.sh         LLaMA baselines part 2 (gkd/distillm/dakd) + sagd
├── llama_curriculum_gpu0.sh        LLaMA curriculum 4 configs × 3 seeds (slow path)
├── gpu1_curriculum.sh              Qwen curriculum × seed 42
├── gpu2_curriculum.sh              Qwen curriculum × seed 123
├── gpu3_curriculum.sh              Qwen curriculum × seed 456
├── gpu1_profiling.sh               Qwen gradient profiling
│
├── 3090/                           # 3090 (lyg1086) — bitsandbytes 8-bit teacher
│   ├── _common.sh                  shared config (paths, flags, eval helper)
│   ├── run.sh                      8-task dispatcher (see "3090 dispatcher tasks" above)
│   ├── smoketest.sh                10-min sanity check
│   └── verify_8bit.sh              fp16 vs 8-bit fidelity verification
│
└── _archive/                       # stale scripts from earlier phases (do not use)
    ├── run_phase1_*.sh, run_phase2_*.sh    Phase 1/2 launchers (now via per-task scripts)
    ├── eval_phase1_*.sh, eval_phase2_*.sh  Phase 1/2 evaluators (now a100_*_eval.sh)
    ├── fix_phase2.sh                       one-off Phase 2 hotfix
    ├── run_baselines.sh, run_ours.sh       old monolithic launchers
    ├── run_all_gpus.sh                     old 3-GPU parallel launcher
    ├── gpu1_ours.sh, gpu2_squad.sh,        old GPU-pinned launchers
    │   gpu3_samsum_gsm8k.sh
    ├── sweep.sh                            old hyperparameter sweep
    └── summarize_results.py,               replaced by aggregate_results.py
        summarize_phase2.py
```

## Conventions

- **Skip-if-exists**: every shell script checks for the output file (ckpt or eval JSON) and skips if present. Re-running picks up where left off.
- **CUDA_VISIBLE_DEVICES** picks the GPU; scripts internally use `--device cuda:0`.
- **Paths**: scripts call `cd "$(dirname "$0")/.."` first, so they are path-agnostic.
- **Outputs**: training writes to `outputs_dolly/<student>/<method>/seed_<N>/`, `outputs_task/<student>/<dataset>/<method>/seed_<N>/`, or `outputs_ours/<student>/<dataset>/<config>/<method>/seed_<N>/`. Eval JSON co-locates as `benchmark_rouge.json`.

## When to use each shell entry point

| Scenario | Script |
|---|---|
| Fresh install on a 3090 box, want to verify it works | `3090/smoketest.sh` |
| Decide whether 8-bit teacher is faithful enough | `3090/verify_8bit.sh` (PASS / MARGINAL / FAIL verdict) |
| Train one of 8 LLaMA experiment families on 3090 | `3090/run.sh <task>` |
| Re-evaluate all existing A100 ckpts with 5-bench eval | `a100_llama_eval.sh` + `a100_qwen_eval.sh` |
| Add Qwen ablation/sweep data for paper Tables 5-6 / Figure 4 | `a100_qwen_extension.sh` |
