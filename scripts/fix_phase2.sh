#!/bin/bash
# =============================================================================
# Fix Phase 2: retrain missing SQuAD seed_42 + eval all pending
#
# Split across TWO GPUs: each GPU handles both training AND eval.
#   GPU_A: retrain 3 missing methods + eval half of pending jobs
#   GPU_B: retrain 3 missing methods + eval other half
#
# Usage:
#   GPU_A=2 GPU_B=3 bash scripts/fix_phase2.sh
# =============================================================================

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"

TEACHER="Qwen/Qwen3-8B"
STUDENT="Qwen/Qwen3-0.6B"
STUDENT_TAG="qwen3_0.6B"
BASE="outputs_task/${STUDENT_TAG}"

EPOCHS=10
LR=1e-5
BATCH_SIZE=4
MAX_SEQ_LEN=512

mkdir -p logs

# =============================================================================
# Helper: train one method on squad seed_42
# =============================================================================
train_one() {
    local GPU=$1 METHOD=$2
    CKPT="${BASE}/squad/${METHOD}/seed_42/student_final.pt"
    if [ -f "$CKPT" ]; then
        echo "[GPU${GPU}] SKIP train squad/${METHOD}/seed_42 (exists)"
        return
    fi
    EXTRA=""
    case $METHOD in
        gkd)      EXTRA="--gkd_beta 0.5" ;;
        distillm) EXTRA="--distillm_alpha 0.5" ;;
        dakd)     EXTRA="--bdl_lambda 0.9" ;;
    esac
    echo "[GPU${GPU}] TRAIN >>> squad/${METHOD}/seed_42"
    CUDA_VISIBLE_DEVICES=$GPU python scripts/train.py \
        --method $METHOD --dataset squad --student_model $STUDENT \
        --epochs $EPOCHS --lr $LR --batch_size $BATCH_SIZE \
        --max_seq_len $MAX_SEQ_LEN --skip_eval --device cuda:0 \
        $EXTRA --seed 42 --output_dir "${BASE}/squad/" \
        > "logs/retrain_squad_${METHOD}_s42.log" 2>&1
    echo "[GPU${GPU}] DONE train squad/${METHOD}/seed_42"
}

# =============================================================================
# Helper: eval one job
# =============================================================================
eval_one() {
    local GPU=$1 DS=$2 METHOD=$3 SEED=$4
    CKPT="${BASE}/${DS}/${METHOD}/seed_${SEED}/student_final.pt"
    EVAL_OUT="${BASE}/${DS}/${METHOD}/seed_${SEED}/eval_metrics.json"
    [ ! -f "$CKPT" ] && return
    [ -f "$EVAL_OUT" ] && return

    MAX_NEW=256
    [ "$DS" = "squad" ] && MAX_NEW=32

    echo "[GPU${GPU}] EVAL >>> ${DS}/${METHOD}/seed_${SEED}"
    CUDA_VISIBLE_DEVICES=$GPU python scripts/evaluate.py \
        --student_model "$STUDENT" --student_ckpt "$CKPT" \
        --dataset "$DS" --subset test --max_new_tokens $MAX_NEW \
        --output_path "$EVAL_OUT" --skip_bertscore \
        --device cuda:0 --seed $SEED \
        > "logs/eval_${DS}_${METHOD}_s${SEED}.log" 2>&1
    echo "[GPU${GPU}] DONE eval ${DS}/${METHOD}/seed_${SEED}"
}

# =============================================================================
# GPU_A worker: train first 3 missing + eval odd-indexed jobs
# =============================================================================
worker_a() {
    echo "===== GPU_A ($GPU_A): train 3 + eval half ====="

    # Train: standard_kd, reverse_kl, seqkd
    train_one $GPU_A standard_kd
    train_one $GPU_A reverse_kl
    train_one $GPU_A seqkd

    # Eval: samsum (all) + gsm8k (sft, standard_kd, reverse_kl, seqkd)
    for METHOD in sft standard_kd reverse_kl seqkd gkd distillm dakd sagd; do
        for SEED in 42 123 456; do
            eval_one $GPU_A samsum $METHOD $SEED
        done
    done
    for METHOD in sft standard_kd reverse_kl seqkd; do
        for SEED in 42 123 456; do
            eval_one $GPU_A gsm8k $METHOD $SEED
        done
    done
    # Eval: squad — methods that should have checkpoints by now
    for METHOD in sft standard_kd reverse_kl seqkd sagd; do
        for SEED in 42 123 456; do
            eval_one $GPU_A squad $METHOD $SEED
        done
    done

    echo "===== GPU_A done ====="
}

# =============================================================================
# GPU_B worker: train last 3 missing + eval even-indexed jobs
# =============================================================================
worker_b() {
    echo "===== GPU_B ($GPU_B): train 3 + eval half ====="

    # Train: gkd, distillm, dakd
    train_one $GPU_B gkd
    train_one $GPU_B distillm
    train_one $GPU_B dakd

    # Eval: gsm8k (gkd, distillm, dakd, sagd) + squad (gkd, distillm, dakd)
    for METHOD in gkd distillm dakd sagd; do
        for SEED in 42 123 456; do
            eval_one $GPU_B gsm8k $METHOD $SEED
        done
    done
    for METHOD in gkd distillm dakd; do
        for SEED in 42 123 456; do
            eval_one $GPU_B squad $METHOD $SEED
        done
    done

    echo "===== GPU_B done ====="
}

# =============================================================================
# Summary
# =============================================================================
summarize() {
    echo ""
    echo "===== Phase 2 Results: ${STUDENT_TAG} ====="
    python -c "
import json, numpy as np, os

methods = ['sft', 'standard_kd', 'reverse_kl', 'seqkd', 'gkd', 'distillm', 'dakd', 'sagd']
seeds = [42, 123, 456]
base = '${BASE}'

header = f\"{'Method':<15} | {'SAMSum RL':>12} | {'GSM8K Acc':>12} | {'SQuAD EM':>12} | {'SQuAD F1':>12}\"
print(header)
print('-' * len(header))

for method in methods:
    samsum_rl, gsm_acc, squad_em, squad_f1 = [], [], [], []
    for seed in seeds:
        for ds, lst, key in [
            ('samsum', samsum_rl, 'rouge_l_f'),
            ('gsm8k', gsm_acc, 'gsm8k_accuracy'),
            ('squad', squad_em, 'exact_match'),
            ('squad', squad_f1, 'token_f1'),
        ]:
            try:
                path = f'{base}/{ds}/{method}/seed_{seed}/eval_metrics.json'
                with open(path) as f:
                    m = json.load(f)
                val = m[key]
                if key == 'rouge_l_f':
                    val *= 100
                elif key in ('exact_match', 'token_f1', 'gsm8k_accuracy'):
                    val *= 100
                lst.append(val)
            except: pass
    def fmt(lst):
        if not lst: return '—'
        return f'{np.mean(lst):.2f}+/-{np.std(lst):.2f}'
    print(f'{method:<15} | {fmt(samsum_rl):>12} | {fmt(gsm_acc):>12} | {fmt(squad_em):>12} | {fmt(squad_f1):>12}')
"
}

# =============================================================================
# Main: run both workers in parallel
# =============================================================================
worker_a &
PID_A=$!
worker_b &
PID_B=$!

wait $PID_A
wait $PID_B

summarize
