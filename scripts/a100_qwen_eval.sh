#!/bin/bash
# A100 eval-only re-run of all existing Qwen ckpts with the full 5-benchmark
# eval suite. Use after BENCHMARKS was extended to 5 (self_inst + vicuna_eval)
# AND the rep_penalty fix in evaluation.py.
#
# Run BEFORE this script:
#     find outputs_dolly/qwen3_0.6B outputs_dolly/qwen3_1.7B outputs_ours/qwen3_0.6B \
#         -name "benchmark_rouge.json" -delete
#
# Usage:
#     CUDA_VISIBLE_DEVICES=3 bash scripts/a100_qwen_eval.sh                # all
#     CUDA_VISIBLE_DEVICES=3 bash scripts/a100_qwen_eval.sh baselines_only
#     CUDA_VISIBLE_DEVICES=3 bash scripts/a100_qwen_eval.sh curriculum_only
#     CUDA_VISIBLE_DEVICES=3 bash scripts/a100_qwen_eval.sh extension_only
#
# ~10 min per ckpt × 5 benchmarks → ~5h for full Qwen suite (~60 ckpts).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

GROUP="${1:-all}"
SEEDS="42 123 456 789 2024"
SEEDS_OURS="42 123 456"

eval_path () {
    local _student="$1" _ckpt="$2" _eval="$3"
    [ ! -f "$_ckpt" ] && return 0
    [ -f "$_eval" ] && { echo "SKIP $_eval"; return 0; }
    echo ">>> EVAL $_eval"
    python scripts/evaluate_benchmarks.py \
        --student_model "$_student" \
        --student_ckpt "$_ckpt" \
        --output_path "$_eval" \
        --device cuda:0
}

# ---------- Baselines: 8 methods × 5 seeds × 2 students = 80 ckpts ----------
do_baselines () {
    echo "===== Section: Qwen baselines ====="
    for STUDENT_TAG in "qwen3_0.6B:Qwen/Qwen3-0.6B" "qwen3_1.7B:Qwen/Qwen3-1.7B"; do
        TAG="${STUDENT_TAG%%:*}"
        MODEL="${STUDENT_TAG##*:}"
        BASE="outputs_dolly/${TAG}"
        for METHOD in sft standard_kd reverse_kl seqkd gkd distillm dakd sagd; do
            for SEED in $SEEDS; do
                eval_path "$MODEL" \
                    "${BASE}/${METHOD}/seed_${SEED}/student_final.pt" \
                    "${BASE}/${METHOD}/seed_${SEED}/benchmark_rouge.json"
            done
        done
    done
}

# ---------- Curriculum: 4 configs (+ sagd_random) × 3 seeds ----------
do_curriculum () {
    echo "===== Section: Qwen curriculum ====="
    BASE="outputs_ours/qwen3_0.6B/dolly"
    for CFG in sagd_grad_curriculum kd_grad_curriculum kd_pocl sagd_pocl sagd_random; do
        case "$CFG" in
            sagd_*) METHOD="sagd" ;;
            kd_*)   METHOD="standard_kd" ;;
        esac
        for SEED in $SEEDS_OURS; do
            eval_path "Qwen/Qwen3-0.6B" \
                "${BASE}/${CFG}/${METHOD}/seed_${SEED}/student_final.pt" \
                "${BASE}/${CFG}/${METHOD}/seed_${SEED}/benchmark_rouge.json"
        done
    done
}

# ---------- Extension: loss ablation + lambda sweep ----------
do_extension () {
    echo "===== Section: Qwen extension (loss ablation, lambda sweep) ====="
    BASE="outputs_ours/qwen3_0.6B/dolly"

    for METHOD in reverse_kl gkd distillm dakd; do
        for SEED in $SEEDS_OURS; do
            eval_path "Qwen/Qwen3-0.6B" \
                "${BASE}/loss_ablation/${METHOD}/seed_${SEED}/student_final.pt" \
                "${BASE}/loss_ablation/${METHOD}/seed_${SEED}/benchmark_rouge.json"
        done
    done

    for LAMBDA in 0.1 1.0 2.0 5.0; do
        for SEED in $SEEDS_OURS; do
            eval_path "Qwen/Qwen3-0.6B" \
                "${BASE}/lambda_${LAMBDA}/sagd/seed_${SEED}/student_final.pt" \
                "${BASE}/lambda_${LAMBDA}/sagd/seed_${SEED}/benchmark_rouge.json"
        done
    done
}

case "$GROUP" in
all)              do_baselines; do_curriculum; do_extension ;;
baselines_only)   do_baselines ;;
curriculum_only)  do_curriculum ;;
extension_only)   do_extension ;;
*)                echo "Unknown group: $GROUP"; exit 1 ;;
esac

echo "===== a100_qwen_eval done ====="
