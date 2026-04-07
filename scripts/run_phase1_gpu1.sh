#!/bin/bash
# =============================================================================
# Phase 0 (half) + Phase 1 (half) on GPU 1
#   Phase 0:  precompute teacher saliency for dolly + squad
#   Phase 1:  Qwen3-8B → Qwen3-0.6B, 8 methods × 5 seeds = 40 runs (Dolly)
# =============================================================================

set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# CUDA_VISIBLE_DEVICES maps physical GPU 1 to logical cuda:0
DEVICE="cuda:0"
# Non-SaGD methods use batch_size=8; SaGD auto-halves to 4 inside trainer
# (effective batch = batch_size * grad_accum = 32 in both cases)
BATCH_SIZE=8
PRECOMPUTE_BATCH=4
MAX_SEQ_LEN=512

TEACHER="Qwen/Qwen3-8B"
STUDENT="Qwen/Qwen3-0.6B"
STUDENT_TAG="qwen3_0.6B"

DATA_DIR="data"
OUTPUT_DIR="outputs_dolly/${STUDENT_TAG}"
LOG_DIR="logs/gpu1_${STUDENT_TAG}"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$LOG_DIR"

# =============================================================================
# Phase 0 (half): teacher saliency for dolly + squad
# =============================================================================
echo "===== Phase 0 (GPU1): precompute teacher saliency ====="

if [ ! -f "${DATA_DIR}/teacher_saliency_dolly.pt" ]; then
    echo "[GPU1] precomputing dolly saliency..."
    python scripts/precompute_teacher_saliency.py \
        --model_name "$TEACHER" \
        --dataset dolly \
        --output_path "${DATA_DIR}/teacher_saliency_dolly.pt" \
        --batch_size "$PRECOMPUTE_BATCH" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --device "$DEVICE" \
        2>&1 | tee "${LOG_DIR}/precompute_dolly.log"
else
    echo "[GPU1] dolly saliency already exists, skipping."
fi

if [ ! -f "${DATA_DIR}/teacher_saliency_squad.pt" ]; then
    echo "[GPU1] precomputing squad saliency..."
    python scripts/precompute_teacher_saliency.py \
        --model_name "$TEACHER" \
        --dataset squad \
        --output_path "${DATA_DIR}/teacher_saliency_squad.pt" \
        --batch_size "$PRECOMPUTE_BATCH" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --device "$DEVICE" \
        2>&1 | tee "${LOG_DIR}/precompute_squad.log"
else
    echo "[GPU1] squad saliency already exists, skipping."
fi

# Touch a marker file so the other GPU can detect completion
touch "${DATA_DIR}/.dolly_saliency_ready"

# =============================================================================
# Phase 1 (half): 40 runs on Qwen3-0.6B, all 8 methods × 5 seeds
# =============================================================================
echo "===== Phase 1 (GPU1): training Qwen3-0.6B (40 runs) ====="

SEEDS=(42 123 456 789 2024)
METHODS=(sft standard_kd reverse_kl seqkd gkd distillm dakd sagd)

run_one() {
    local METHOD=$1
    local SEED=$2
    local LOG_FILE="${LOG_DIR}/${METHOD}_seed${SEED}.log"
    local CKPT="${OUTPUT_DIR}/${METHOD}/seed_${SEED}/student_final.pt"

    if [ -f "$CKPT" ]; then
        echo "[GPU1] SKIP ${METHOD}/seed_${SEED} (checkpoint exists)"
        return 0
    fi

    echo "[GPU1] >>> ${METHOD}/seed_${SEED}  (log: ${LOG_FILE})"

    local EXTRA_ARGS=""
    case $METHOD in
        gkd)      EXTRA_ARGS="--gkd_beta 0.5" ;;
        distillm) EXTRA_ARGS="--distillm_alpha 0.5" ;;
        dakd)     EXTRA_ARGS="--bdl_lambda 0.9" ;;
        sagd)     EXTRA_ARGS="--teacher_saliency_path ${DATA_DIR}/teacher_saliency_dolly.pt --lambda_noise 0.5 --noise_sigma 0.005 --sagd_every_n_steps 5 --sagd_tau_w 1.0" ;;
    esac

    python scripts/train.py \
        --method "$METHOD" \
        --dataset dolly \
        --teacher_model "$TEACHER" \
        --student_model "$STUDENT" \
        --batch_size "$BATCH_SIZE" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --epochs 10 \
        --lr 1e-5 \
        --seed "$SEED" \
        --output_dir "$OUTPUT_DIR" \
        --skip_eval \
        --device "$DEVICE" \
        $EXTRA_ARGS \
        2>&1 | tee "$LOG_FILE"
}

for SEED in "${SEEDS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        run_one "$METHOD" "$SEED"
    done
done

echo "===== GPU1 done: Phase 0 (dolly+squad) + Phase 1 (Qwen3-0.6B × 40 runs) ====="
