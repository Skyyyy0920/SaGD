#!/bin/bash
# =============================================================================
# Phase 1 evaluation on GPU 1 — Qwen3-0.6B
# Evaluates all 40 trained models on 5 instruction-following benchmarks
# (DollyEval, SelfInst, Super-Natural, Unnatural, VicunaEval)
# =============================================================================

set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

DEVICE="cuda:0"
STUDENT="Qwen/Qwen3-0.6B"
STUDENT_TAG="qwen3_0.6B"
OUTPUT_DIR="outputs_dolly/${STUDENT_TAG}"
LOG_DIR="logs/gpu1_${STUDENT_TAG}"

mkdir -p "$LOG_DIR"

SEEDS=(42 123 456 789 2024)
METHODS=(sft standard_kd reverse_kl seqkd gkd distillm dakd sagd)

echo "===== Phase 1 Eval (GPU1): Qwen3-0.6B × 5 benchmarks ====="

for SEED in "${SEEDS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        CKPT="${OUTPUT_DIR}/${METHOD}/seed_${SEED}/student_final.pt"
        OUT_FILE="${OUTPUT_DIR}/${METHOD}/seed_${SEED}/benchmark_rouge.json"
        LOG_FILE="${LOG_DIR}/eval_${METHOD}_seed${SEED}.log"

        # Skip if already evaluated
        if [ -f "$OUT_FILE" ]; then
            echo "[GPU1] SKIP eval ${METHOD}/seed_${SEED} (benchmark_rouge.json exists)"
            continue
        fi

        # Skip if checkpoint missing
        if [ ! -f "$CKPT" ]; then
            echo "[GPU1] SKIP eval ${METHOD}/seed_${SEED} (no checkpoint)"
            continue
        fi

        echo "[GPU1] >>> eval ${METHOD}/seed_${SEED}"
        python scripts/evaluate_benchmarks.py \
            --student_model "$STUDENT" \
            --student_ckpt "$CKPT" \
            --output_path "$OUT_FILE" \
            --device "$DEVICE" \
            --seed "$SEED" \
            2>&1 | tee "$LOG_FILE"
    done
done

# =============================================================================
# Summary table
# =============================================================================
echo ""
echo "===== Results Summary: Qwen3-0.6B ====="
python -c "
import json, os, numpy as np

methods = ['sft', 'standard_kd', 'reverse_kl', 'seqkd', 'gkd', 'distillm', 'dakd', 'sagd']
benchmarks = ['dolly_eval', 'self_inst', 'super_natural', 'unnatural', 'vicuna_eval']
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
            row += f' | {m:>5.1f}±{s:>4.1f}'
            avgs.append(m)
        else:
            row += f' |       —     '
    if avgs:
        row += f' | {np.mean(avgs):>5.1f}'
    print(row)
"

echo "===== GPU1 eval done ====="
