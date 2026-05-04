#!/bin/bash
# =============================================================================
# Phase 1 evaluation on GPU 3 — Qwen3-1.7B
# Evaluates all 40 trained models on 4 instruction-following benchmarks
# (DollyEval, SelfInst, Super-Natural, Unnatural)
#
# Runs 2 eval processes in parallel on the same GPU for 2× throughput.
# =============================================================================

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

DEVICE="cuda:0"
STUDENT="Qwen/Qwen3-1.7B"
STUDENT_TAG="qwen3_1.7B"
OUTPUT_DIR="outputs_dolly/${STUDENT_TAG}"
LOG_DIR="logs/gpu3_${STUDENT_TAG}"

mkdir -p "$LOG_DIR"

SEEDS=(42 123 456 789 2024)
METHODS=(sft standard_kd reverse_kl seqkd gkd distillm dakd sagd)

echo "===== Phase 1 Eval (GPU3): Qwen3-1.7B × 4 benchmarks (2 parallel) ====="

# Build a flat list of (METHOD, SEED) jobs
JOBS=()
for SEED in "${SEEDS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        CKPT="${OUTPUT_DIR}/${METHOD}/seed_${SEED}/student_final.pt"
        OUT_FILE="${OUTPUT_DIR}/${METHOD}/seed_${SEED}/benchmark_rouge.json"

        if [ -f "$OUT_FILE" ]; then
            echo "[GPU3] SKIP ${METHOD}/seed_${SEED} (already evaluated)"
            continue
        fi
        if [ ! -f "$CKPT" ]; then
            echo "[GPU3] SKIP ${METHOD}/seed_${SEED} (no checkpoint)"
            continue
        fi
        JOBS+=("${METHOD}|${SEED}")
    done
done

N_JOBS=${#JOBS[@]}
echo "[GPU3] ${N_JOBS} jobs to run (2 at a time)"

# Run jobs two at a time
i=0
while [ $i -lt $N_JOBS ]; do
    # Job A
    IFS='|' read -r METHOD_A SEED_A <<< "${JOBS[$i]}"
    CKPT_A="${OUTPUT_DIR}/${METHOD_A}/seed_${SEED_A}/student_final.pt"
    OUT_A="${OUTPUT_DIR}/${METHOD_A}/seed_${SEED_A}/benchmark_rouge.json"
    LOG_A="${LOG_DIR}/eval_${METHOD_A}_seed${SEED_A}.log"

    echo "[GPU3] >>> [A] ${METHOD_A}/seed_${SEED_A}"
    python scripts/evaluate_benchmarks.py \
        --student_model "$STUDENT" \
        --student_ckpt "$CKPT_A" \
        --output_path "$OUT_A" \
        --device "$DEVICE" \
        --seed "$SEED_A" \
        > "$LOG_A" 2>&1 &
    PID_A=$!

    # Job B (if exists)
    j=$((i + 1))
    PID_B=""
    if [ $j -lt $N_JOBS ]; then
        IFS='|' read -r METHOD_B SEED_B <<< "${JOBS[$j]}"
        CKPT_B="${OUTPUT_DIR}/${METHOD_B}/seed_${SEED_B}/student_final.pt"
        OUT_B="${OUTPUT_DIR}/${METHOD_B}/seed_${SEED_B}/benchmark_rouge.json"
        LOG_B="${LOG_DIR}/eval_${METHOD_B}_seed${SEED_B}.log"

        echo "[GPU3] >>> [B] ${METHOD_B}/seed_${SEED_B}"
        python scripts/evaluate_benchmarks.py \
            --student_model "$STUDENT" \
            --student_ckpt "$CKPT_B" \
            --output_path "$OUT_B" \
            --device "$DEVICE" \
            --seed "$SEED_B" \
            > "$LOG_B" 2>&1 &
        PID_B=$!
    fi

    # Wait for both to finish before launching next pair
    wait $PID_A
    echo "[GPU3]     done [A] ${METHOD_A}/seed_${SEED_A}"
    if [ -n "$PID_B" ]; then
        wait $PID_B
        echo "[GPU3]     done [B] ${METHOD_B}/seed_${SEED_B}"
    fi

    i=$((i + 2))
done

# =============================================================================
# Summary table
# =============================================================================
echo ""
echo "===== Results Summary: Qwen3-1.7B ====="
python -c "
import json, os, numpy as np

methods = ['sft', 'standard_kd', 'reverse_kl', 'seqkd', 'gkd', 'distillm', 'dakd', 'sagd']
benchmarks = ['dolly_eval', 'super_natural', 'unnatural']
seeds = [42, 123, 456, 789, 2024]
tag = '${STUDENT_TAG}'

header = f\"{'Method':<15}\" + ''.join(f' | {b:>14}' for b in benchmarks) + ' |    Avg.'
print(header)
print('-' * len(header))
for method in methods:
    bench_scores = {b: [] for b in benchmarks}
    for seed in seeds:
        path = f'outputs_dolly/{tag}/{method}/seed_{seed}/benchmark_rouge.json'
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            for b in benchmarks:
                if b in data:
                    bench_scores[b].append(data[b]['rouge_l_f'] * 100)
    row = f'{method:<15}'
    avgs = []
    for b in benchmarks:
        if bench_scores[b]:
            m = np.mean(bench_scores[b])
            s = np.std(bench_scores[b])
            row += f' | {m:>5.2f}±{s:>4.2f}'
            avgs.append(m)
        else:
            row += f' |       —     '
    if avgs:
        row += f' | {np.mean(avgs):>5.2f}'
    print(row)
"

echo "===== GPU3 eval done ====="
