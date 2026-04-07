#!/bin/bash
# =============================================================================
# Phase 0 (half) + Phase 1 (half) on GPU 3
#   Phase 0:  precompute teacher saliency for samsum + gsm8k
#   Phase 1:  Qwen3-8B → Qwen3-1.7B, 8 methods × 5 seeds = 40 runs (Dolly)
#
# Note: Phase 1 SaGD runs need data/teacher_saliency_dolly.pt which is
#       computed by run_phase1_gpu1.sh. We wait for it before starting SaGD.
# =============================================================================

set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# CUDA_VISIBLE_DEVICES maps physical GPU 3 to logical cuda:0
DEVICE="cuda:0"
# Non-SaGD methods use batch_size=4; SaGD auto-halves to 2 inside trainer
# (effective batch = batch_size * grad_accum = 16 in both cases)
BATCH_SIZE=4
PRECOMPUTE_BATCH=4
MAX_SEQ_LEN=512
EPOCHS=5

TEACHER="Qwen/Qwen3-8B"
STUDENT="Qwen/Qwen3-1.7B"
STUDENT_TAG="qwen3_1.7B"

DATA_DIR="data"
OUTPUT_DIR="outputs_dolly/${STUDENT_TAG}"
LOG_DIR="logs/gpu3_${STUDENT_TAG}"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$LOG_DIR"

# =============================================================================
# Phase 0 (half): teacher saliency for samsum + gsm8k
# =============================================================================
echo "===== Phase 0 (GPU3): precompute teacher saliency ====="

if [ ! -f "${DATA_DIR}/teacher_saliency_samsum.pt" ]; then
    echo "[GPU3] precomputing samsum saliency..."
    python scripts/precompute_teacher_saliency.py \
        --model_name "$TEACHER" \
        --dataset samsum \
        --output_path "${DATA_DIR}/teacher_saliency_samsum.pt" \
        --batch_size "$PRECOMPUTE_BATCH" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --device "$DEVICE" \
        2>&1 | tee "${LOG_DIR}/precompute_samsum.log"
else
    echo "[GPU3] samsum saliency already exists, skipping."
fi

if [ ! -f "${DATA_DIR}/teacher_saliency_gsm8k.pt" ]; then
    echo "[GPU3] precomputing gsm8k saliency..."
    python scripts/precompute_teacher_saliency.py \
        --model_name "$TEACHER" \
        --dataset gsm8k \
        --output_path "${DATA_DIR}/teacher_saliency_gsm8k.pt" \
        --batch_size "$PRECOMPUTE_BATCH" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --device "$DEVICE" \
        2>&1 | tee "${LOG_DIR}/precompute_gsm8k.log"
else
    echo "[GPU3] gsm8k saliency already exists, skipping."
fi

# =============================================================================
# Phase 1 (half): 40 runs on Qwen3-1.7B, all 8 methods × 5 seeds
# =============================================================================
echo "===== Phase 1 (GPU3): training Qwen3-1.7B (40 runs) ====="

SEEDS=(42 123 456 789 2024)
# Non-SaGD methods can run immediately; SaGD waits for the dolly cache.
NON_SAGD_METHODS=(sft standard_kd reverse_kl seqkd gkd distillm dakd)

run_one() {
    local METHOD=$1
    local SEED=$2
    local LOG_FILE="${LOG_DIR}/${METHOD}_seed${SEED}.log"
    local CKPT="${OUTPUT_DIR}/${METHOD}/seed_${SEED}/student_final.pt"

    if [ -f "$CKPT" ]; then
        echo "[GPU3] SKIP ${METHOD}/seed_${SEED} (checkpoint exists)"
        return 0
    fi

    echo "[GPU3] >>> ${METHOD}/seed_${SEED}  (log: ${LOG_FILE})"

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
        --epochs "$EPOCHS" \
        --lr 1e-5 \
        --seed "$SEED" \
        --output_dir "$OUTPUT_DIR" \
        --skip_eval \
        --device "$DEVICE" \
        $EXTRA_ARGS \
        2>&1 | tee "$LOG_FILE"
}

# 1) Run all non-SaGD methods first (no dependency on dolly cache)
for SEED in "${SEEDS[@]}"; do
    for METHOD in "${NON_SAGD_METHODS[@]}"; do
        run_one "$METHOD" "$SEED"
    done
done

# 2) Wait for the dolly saliency cache produced by GPU1.
#    Poll the marker file (created by GPU1 *after* the cache write completes),
#    not the cache file itself, to avoid reading a half-written file.
echo "[GPU3] Waiting for ${DATA_DIR}/.dolly_saliency_ready ..."
while [ ! -f "${DATA_DIR}/.dolly_saliency_ready" ]; do
    sleep 30
done
echo "[GPU3] dolly saliency ready, starting SaGD runs."

# 3) Run SaGD across all seeds
for SEED in "${SEEDS[@]}"; do
    run_one "sagd" "$SEED"
done

echo "===== GPU3 done: Phase 0 (samsum+gsm8k) + Phase 1 (Qwen3-1.7B × 40 runs) ====="
