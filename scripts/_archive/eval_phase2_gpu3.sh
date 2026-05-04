#!/bin/bash
# =============================================================================
# Phase 2 evaluation on GPU 3 — Qwen3-1.7B
# Evaluates on SQuAD (EM/F1/ROUGE-L/PPL), SAMSum (ROUGE-L/PPL), GSM8K (Acc/PPL)
# Runs 2 eval processes in parallel for 2× throughput.
# =============================================================================

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

DEVICE="cuda:0"
STUDENT="Qwen/Qwen3-1.7B"
STUDENT_TAG="qwen3_1.7B"
LOG_DIR="logs/gpu3_phase2_eval_${STUDENT_TAG}"

mkdir -p "$LOG_DIR"

SEEDS=(42 123 456)
METHODS=(sft standard_kd reverse_kl seqkd gkd distillm dakd sagd)
DATASETS=(squad samsum gsm8k)

echo "===== Phase 2 Eval (GPU3): ${STUDENT_TAG} × 3 datasets (2 parallel) ====="

# Build job list
JOBS=()
for DATASET in "${DATASETS[@]}"; do
    MAX_NEW=256
    [ "$DATASET" = "squad" ] && MAX_NEW=32

    for SEED in "${SEEDS[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            OUTPUT_DIR="outputs_task/${STUDENT_TAG}/${DATASET}"
            CKPT="${OUTPUT_DIR}/${METHOD}/seed_${SEED}/student_final.pt"
            OUT_FILE="${OUTPUT_DIR}/${METHOD}/seed_${SEED}/eval_metrics.json"

            if [ -f "$OUT_FILE" ]; then
                echo "[GPU3] SKIP ${DATASET}/${METHOD}/seed_${SEED} (eval exists)"
                continue
            fi
            if [ ! -f "$CKPT" ]; then
                echo "[GPU3] SKIP ${DATASET}/${METHOD}/seed_${SEED} (no ckpt)"
                continue
            fi
            JOBS+=("${DATASET}|${METHOD}|${SEED}|${MAX_NEW}")
        done
    done
done

N_JOBS=${#JOBS[@]}
echo "[GPU3] ${N_JOBS} eval jobs (2 at a time)"

i=0
while [ $i -lt $N_JOBS ]; do
    # Job A
    IFS='|' read -r DS_A M_A S_A MN_A <<< "${JOBS[$i]}"
    CKPT_A="outputs_task/${STUDENT_TAG}/${DS_A}/${M_A}/seed_${S_A}/student_final.pt"
    OUT_A="outputs_task/${STUDENT_TAG}/${DS_A}/${M_A}/seed_${S_A}/eval_metrics.json"
    LOG_A="${LOG_DIR}/eval_${DS_A}_${M_A}_seed${S_A}.log"

    echo "[GPU3] >>> [A] ${DS_A}/${M_A}/seed_${S_A}"
    python scripts/evaluate.py \
        --student_model "$STUDENT" \
        --student_ckpt "$CKPT_A" \
        --dataset "$DS_A" --subset test \
        --max_new_tokens "$MN_A" \
        --output_path "$OUT_A" \
        --skip_bertscore \
        --device "$DEVICE" \
        --seed "$S_A" \
        > "$LOG_A" 2>&1 &
    PID_A=$!

    # Job B
    j=$((i + 1))
    PID_B=""
    if [ $j -lt $N_JOBS ]; then
        IFS='|' read -r DS_B M_B S_B MN_B <<< "${JOBS[$j]}"
        CKPT_B="outputs_task/${STUDENT_TAG}/${DS_B}/${M_B}/seed_${S_B}/student_final.pt"
        OUT_B="outputs_task/${STUDENT_TAG}/${DS_B}/${M_B}/seed_${S_B}/eval_metrics.json"
        LOG_B="${LOG_DIR}/eval_${DS_B}_${M_B}_seed${S_B}.log"

        echo "[GPU3] >>> [B] ${DS_B}/${M_B}/seed_${S_B}"
        python scripts/evaluate.py \
            --student_model "$STUDENT" \
            --student_ckpt "$CKPT_B" \
            --dataset "$DS_B" --subset test \
            --max_new_tokens "$MN_B" \
            --output_path "$OUT_B" \
            --skip_bertscore \
            --device "$DEVICE" \
            --seed "$S_B" \
            > "$LOG_B" 2>&1 &
        PID_B=$!
    fi

    wait $PID_A
    echo "[GPU3]     done [A] ${DS_A}/${M_A}/seed_${S_A}"
    if [ -n "$PID_B" ]; then
        wait $PID_B
        echo "[GPU3]     done [B] ${DS_B}/${M_B}/seed_${S_B}"
    fi

    i=$((i + 2))
done

echo "===== GPU3 Phase 2 eval done ====="
