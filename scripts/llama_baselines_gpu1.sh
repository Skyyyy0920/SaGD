#!/bin/bash
# LLaMA baselines on GPU 1: sft, standard_kd, reverse_kl, seqkd × 3 seeds
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1

TEACHER="meta-llama/Llama-3.1-8B-Instruct"
STUDENT="meta-llama/Llama-3.2-1B-Instruct"
BASE="outputs_dolly/llama_1B"
COMMON="--dataset dolly --student_model $STUDENT --teacher_model $TEACHER \
    --epochs 10 --lr 1e-5 --batch_size 4 --max_seq_len 512 \
    --skip_eval --device cuda:0"

echo "===== GPU 1: LLaMA baselines (4 methods × 3 seeds) ====="

for METHOD in sft standard_kd reverse_kl seqkd; do
    for SEED in 42 123 456; do
        CKPT="${BASE}/${METHOD}/seed_${SEED}/student_final.pt"
        [ -f "$CKPT" ] && echo "SKIP ${METHOD}/seed_${SEED}" && continue
        echo ">>> ${METHOD}/seed_${SEED}"
        python scripts/train.py --method $METHOD $COMMON \
            --seed $SEED --output_dir "${BASE}/"
    done
done

echo "===== Eval ====="
for METHOD in sft standard_kd reverse_kl seqkd; do
    for SEED in 42 123 456; do
        CKPT="${BASE}/${METHOD}/seed_${SEED}/student_final.pt"
        EVAL="${BASE}/${METHOD}/seed_${SEED}/benchmark_rouge.json"
        [ ! -f "$CKPT" ] && continue
        [ -f "$EVAL" ] && continue
        echo "EVAL >>> ${METHOD}/seed_${SEED}"
        python scripts/evaluate_benchmarks.py --student_model "$STUDENT" \
            --student_ckpt "$CKPT" --output_path "$EVAL" --device cuda:0
    done
done

echo "===== GPU 1 done ====="
