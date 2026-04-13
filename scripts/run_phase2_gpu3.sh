#!/bin/bash
# =============================================================================
# Phase 2 (half): Task-specific training on GPU 3
#   Student: Qwen3-1.7B
#   Datasets: SQuAD, SAMSum, GSM8K
#   Methods: sft, standard_kd, reverse_kl, seqkd, gkd, distillm, dakd, sagd
#   Seeds: 42, 123, 456
# =============================================================================

set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

DEVICE="cuda:0"
BATCH_SIZE=4
MAX_SEQ_LEN=512
EPOCHS=3

TEACHER="Qwen/Qwen3-8B"
STUDENT="Qwen/Qwen3-1.7B"
STUDENT_TAG="qwen3_1.7B"

DATA_DIR="data"
LOG_DIR="logs/gpu3_phase2_${STUDENT_TAG}"

mkdir -p "$LOG_DIR"

SEEDS=(42 123 456)
METHODS=(sft standard_kd reverse_kl seqkd gkd distillm dakd sagd)
DATASETS=(squad samsum gsm8k)

run_one() {
    local DATASET=$1
    local METHOD=$2
    local SEED=$3
    local OUTPUT_DIR="outputs_task/${STUDENT_TAG}/${DATASET}"
    local CKPT="${OUTPUT_DIR}/${METHOD}/seed_${SEED}/student_final.pt"
    local LOG_FILE="${LOG_DIR}/${DATASET}_${METHOD}_seed${SEED}.log"

    mkdir -p "$OUTPUT_DIR"

    if [ -f "$CKPT" ]; then
        echo "[GPU3] SKIP ${DATASET}/${METHOD}/seed_${SEED} (checkpoint exists)"
        return 0
    fi

    echo "[GPU3] >>> ${DATASET}/${METHOD}/seed_${SEED}"

    local EXTRA_ARGS=""
    case $METHOD in
        gkd)      EXTRA_ARGS="--gkd_beta 0.5" ;;
        distillm) EXTRA_ARGS="--distillm_alpha 0.5" ;;
        dakd)     EXTRA_ARGS="--bdl_lambda 0.9" ;;
        sagd)     EXTRA_ARGS="--teacher_saliency_path ${DATA_DIR}/teacher_saliency_${DATASET}.pt --lambda_noise 0.5 --noise_sigma 0.005 --sagd_every_n_steps 5 --sagd_tau_w 1.0" ;;
    esac

    python scripts/train.py \
        --method "$METHOD" \
        --dataset "$DATASET" \
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

echo "===== Phase 2 (GPU3): ${STUDENT_TAG} × 3 datasets × 8 methods × 3 seeds ====="

for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            run_one "$DATASET" "$METHOD" "$SEED"
        done
    done
done

echo "===== GPU3 Phase 2 training done ====="
