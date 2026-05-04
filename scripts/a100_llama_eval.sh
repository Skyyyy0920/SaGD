#!/bin/bash
# A100 eval-only re-run of all existing LLaMA ckpts with the full 5-benchmark
# eval suite. Use this after BENCHMARKS was extended to 5 benchmarks AND
# the rep_penalty fix in evaluation.py.
#
# Run BEFORE this script:
#     # Delete old 3-bench JSONs to force re-eval
#     find outputs_dolly/llama_1B outputs_ours/llama_1B \
#         -name "benchmark_rouge.json" -delete
#
# Usage (split by seed across the 2 free A100 GPUs):
#     CUDA_VISIBLE_DEVICES=2 bash scripts/a100_llama_eval.sh seeds_42_123
#     CUDA_VISIBLE_DEVICES=3 bash scripts/a100_llama_eval.sh seed_456
#
# ~10 min per ckpt × 5 benchmarks → ~2h per GPU.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

STUDENT="meta-llama/Llama-3.2-1B-Instruct"
GROUP="${1:-all}"

case "$GROUP" in
seeds_42_123) SEEDS="42 123" ;;
seed_456)     SEEDS="456" ;;
all)          SEEDS="42 123 456" ;;
*)            echo "Unknown group: $GROUP"; exit 1 ;;
esac

eval_path () {
    local _ckpt="$1" _eval="$2"
    [ ! -f "$_ckpt" ] && return 0
    [ -f "$_eval" ] && { echo "SKIP $_eval"; return 0; }
    echo ">>> EVAL $_eval"
    python scripts/evaluate_benchmarks.py \
        --student_model "$STUDENT" \
        --student_ckpt "$_ckpt" \
        --output_path "$_eval" \
        --device cuda:0
}

echo "===== a100_llama_eval: seeds = $SEEDS ====="

# ---------- Baselines (8 methods) ----------
BASE="outputs_dolly/llama_1B"
for METHOD in sft standard_kd reverse_kl seqkd gkd distillm dakd sagd; do
    for SEED in $SEEDS; do
        eval_path "${BASE}/${METHOD}/seed_${SEED}/student_final.pt" \
                  "${BASE}/${METHOD}/seed_${SEED}/benchmark_rouge.json"
    done
done

# ---------- Curriculum (4 configs) ----------
BASE="outputs_ours/llama_1B/dolly"
for CFG in sagd_grad_curriculum kd_grad_curriculum kd_pocl sagd_pocl; do
    case "$CFG" in
        sagd_*) METHOD="sagd" ;;
        kd_*)   METHOD="standard_kd" ;;
    esac
    for SEED in $SEEDS; do
        eval_path "${BASE}/${CFG}/${METHOD}/seed_${SEED}/student_final.pt" \
                  "${BASE}/${CFG}/${METHOD}/seed_${SEED}/benchmark_rouge.json"
    done
done

# ---------- Loss ablation / lambda sweep (if 3090 produced any) ----------
for ROOT in "${BASE}/loss_ablation" "${BASE}"/lambda_*; do
    [ ! -d "$ROOT" ] && continue
    for METHOD_DIR in "$ROOT"/*/; do
        METHOD=$(basename "$METHOD_DIR")
        for SEED in $SEEDS; do
            eval_path "${METHOD_DIR}seed_${SEED}/student_final.pt" \
                      "${METHOD_DIR}seed_${SEED}/benchmark_rouge.json"
        done
    done
done

echo "===== a100_llama_eval done ====="
