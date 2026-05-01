#!/bin/bash
# LLaMA baselines on GPU 2: gkd, distillm, dakd × 3 seeds + sagd_random × 3 seeds
# (sagd needs teacher_saliency, so it must wait for prereq on GPU 0)
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=2

TEACHER="meta-llama/Llama-3.1-8B-Instruct"
STUDENT="meta-llama/Llama-3.2-1B-Instruct"
SAL_PATH="data/teacher_saliency_dolly_llama.pt"
BASE="outputs_dolly/llama_1B"
COMMON="--dataset dolly --student_model $STUDENT --teacher_model $TEACHER \
    --epochs 10 --lr 1e-5 --batch_size 4 --max_seq_len 512 \
    --skip_eval --device cuda:0"

echo "===== GPU 2: LLaMA baselines (gkd, distillm, dakd) ====="

for SEED in 42 123 456; do
    # GKD
    CKPT="${BASE}/gkd/seed_${SEED}/student_final.pt"
    [ ! -f "$CKPT" ] && echo ">>> gkd/seed_${SEED}" && \
    python scripts/train.py --method gkd --gkd_beta 0.5 $COMMON \
        --seed $SEED --output_dir "${BASE}/"

    # DistiLLM
    CKPT="${BASE}/distillm/seed_${SEED}/student_final.pt"
    [ ! -f "$CKPT" ] && echo ">>> distillm/seed_${SEED}" && \
    python scripts/train.py --method distillm --distillm_alpha 0.5 $COMMON \
        --seed $SEED --output_dir "${BASE}/"

    # DA-KD
    CKPT="${BASE}/dakd/seed_${SEED}/student_final.pt"
    [ ! -f "$CKPT" ] && echo ">>> dakd/seed_${SEED}" && \
    python scripts/train.py --method dakd --bdl_lambda 0.9 $COMMON \
        --seed $SEED --output_dir "${BASE}/"
done

# Wait for prereq if needed
echo "===== Waiting for teacher saliency (prereq on GPU 0) ====="
while [ ! -f "$SAL_PATH" ]; do
    echo "Waiting for $SAL_PATH..."
    sleep 60
done

echo "===== SaGD (no curriculum) ====="
for SEED in 42 123 456; do
    CKPT="${BASE}/sagd/seed_${SEED}/student_final.pt"
    [ -f "$CKPT" ] && echo "SKIP sagd/seed_${SEED}" && continue
    echo ">>> sagd/seed_${SEED}"
    python scripts/train.py --method sagd $COMMON \
        --teacher_saliency_path "$SAL_PATH" \
        --lambda_noise 0.5 --noise_sigma 0.005 \
        --sagd_every_n_steps 5 --sagd_tau_w 1.0 \
        --seed $SEED --output_dir "${BASE}/"
done

echo "===== Eval ====="
for METHOD in gkd distillm dakd sagd; do
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

echo "===== GPU 2 done ====="
