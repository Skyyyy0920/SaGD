#!/bin/bash
# =============================================================================
# BASELINES: 7 KD methods on Dolly-15K
#
# Methods: SFT, standard_kd, reverse_kl, seqkd, gkd, distillm, dakd
# Student: Qwen3-0.6B (default) or Qwen3-1.7B
# Seeds: 42, 123, 456
#
# Usage:
#   bash scripts/run_baselines.sh train           # training only
#   bash scripts/run_baselines.sh eval            # evaluation only
#   bash scripts/run_baselines.sh all             # both
#   STUDENT=Qwen/Qwen3-1.7B bash scripts/run_baselines.sh all  # 1.7B student
#   CUDA_DEVICE=cuda:1 bash scripts/run_baselines.sh train      # specific GPU
# =============================================================================

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# ---- Configuration ----
DEVICE="${CUDA_DEVICE:-cuda:0}"
TEACHER="Qwen/Qwen3-8B"
STUDENT="${STUDENT:-Qwen/Qwen3-0.6B}"
STUDENT_TAG=$(echo "$STUDENT" | sed 's/Qwen\/Qwen3-/qwen3_/')
DATASET="dolly"

EPOCHS=10
LR=1e-5
BATCH_SIZE=4
MAX_SEQ_LEN=512
SEEDS=(42 123 456)

OUTPUT_BASE="outputs_dolly/${STUDENT_TAG}"
LOG_DIR="logs/baselines_${STUDENT_TAG}"

mkdir -p "$OUTPUT_BASE" "$LOG_DIR"

BASELINES=(sft standard_kd reverse_kl seqkd gkd distillm dakd)

STAGE="${1:-all}"

# =============================================================================
# Training
# =============================================================================
run_train() {
    echo ""
    echo "===== Baseline Training: ${STUDENT_TAG} on ${DATASET} ====="

    COMMON_ARGS="--dataset $DATASET --student_model $STUDENT \
        --epochs $EPOCHS --lr $LR --batch_size $BATCH_SIZE \
        --max_seq_len $MAX_SEQ_LEN --skip_eval --device $DEVICE"

    for SEED in "${SEEDS[@]}"; do
        for METHOD in "${BASELINES[@]}"; do
            CKPT="${OUTPUT_BASE}/${METHOD}/seed_${SEED}/student_final.pt"
            if [ -f "$CKPT" ]; then
                echo "[TRAIN] SKIP ${METHOD}/seed_${SEED} (exists)"
                continue
            fi

            # Method-specific args
            EXTRA=""
            case $METHOD in
                sft)      EXTRA="" ;;
                gkd)      EXTRA="--gkd_beta 0.5" ;;
                distillm) EXTRA="--distillm_alpha 0.5" ;;
                dakd)     EXTRA="--bdl_lambda 0.9" ;;
            esac

            echo "[TRAIN] >>> ${METHOD}/seed_${SEED}"
            python scripts/train.py \
                --method $METHOD $COMMON_ARGS $EXTRA \
                --seed $SEED \
                --output_dir "${OUTPUT_BASE}/" \
                2>&1 | tee "${LOG_DIR}/train_${METHOD}_s${SEED}.log"
        done
    done

    echo "[TRAIN] All baseline training done."
}

# =============================================================================
# Evaluation
# =============================================================================
run_eval() {
    echo ""
    echo "===== Baseline Evaluation: ${STUDENT_TAG} ====="

    for METHOD in "${BASELINES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            CKPT="${OUTPUT_BASE}/${METHOD}/seed_${SEED}/student_final.pt"
            EVAL_OUT="${OUTPUT_BASE}/${METHOD}/seed_${SEED}/benchmark_rouge.json"

            if [ ! -f "$CKPT" ]; then
                echo "[EVAL] SKIP ${METHOD}/seed_${SEED} (no checkpoint)"
                continue
            fi
            if [ -f "$EVAL_OUT" ]; then
                echo "[EVAL] SKIP ${METHOD}/seed_${SEED} (eval exists)"
                continue
            fi

            echo "[EVAL] >>> ${METHOD}/seed_${SEED}"
            python scripts/evaluate_benchmarks.py \
                --student_model "$STUDENT" \
                --student_ckpt "$CKPT" \
                --output_path "$EVAL_OUT" \
                --device "$DEVICE" \
                2>&1 | tee "${LOG_DIR}/eval_${METHOD}_s${SEED}.log"
        done
    done

    echo "[EVAL] All baseline evaluation done."
    echo ""

    # Summarize
    echo "===== Baseline Results: ${STUDENT_TAG} ====="
    python -c "
import json, numpy as np, os

methods = ['sft', 'standard_kd', 'reverse_kl', 'seqkd', 'gkd', 'distillm', 'dakd']
seeds = [42, 123, 456]
base = '${OUTPUT_BASE}'

header = f\"{'Method':<15} | {'DollyEval':>12} | {'S-NatInst':>12} | {'Unnatural':>12} | {'Avg':>8}\"
print(header)
print('-' * len(header))

for method in methods:
    benchmarks = {'dolly_eval': [], 'super_natural': [], 'unnatural': []}
    for seed in seeds:
        path = os.path.join(base, method, f'seed_{seed}', 'benchmark_rouge.json')
        try:
            with open(path) as f:
                data = json.load(f)
            for b in benchmarks:
                if b in data:
                    benchmarks[b].append(data[b].get('rouge_l_f', 0))
        except:
            pass
    def fmt(lst):
        if not lst: return '—'
        return f'{np.mean(lst):.2f}±{np.std(lst):.2f}'
    avgs = [np.mean(v) for v in benchmarks.values() if v]
    avg_str = f'{np.mean(avgs):.2f}' if avgs else '—'
    print(f\"{method:<15} | {fmt(benchmarks['dolly_eval']):>12} | {fmt(benchmarks['super_natural']):>12} | {fmt(benchmarks['unnatural']):>12} | {avg_str:>8}\")
"
}

# =============================================================================
# Main
# =============================================================================
case "$STAGE" in
    train) run_train ;;
    eval)  run_eval ;;
    all)   run_train && run_eval ;;
    *)     echo "Usage: $0 {all|train|eval}" ; exit 1 ;;
esac
