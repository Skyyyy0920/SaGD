#!/bin/bash
# LLaMA curriculum experiments on GPU 0 (run AFTER llama_prereq.sh).
# Trains 4 curriculum configs × 3 seeds (12 runs) sequentially, then evaluates.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

TEACHER="meta-llama/Llama-3.1-8B-Instruct"
STUDENT="meta-llama/Llama-3.2-1B-Instruct"
SAL="data/teacher_saliency_dolly_llama.pt"
GRAD_ORDER="outputs_ours/curriculum/llama_1B/dolly/gradient_order.pt"
POCL_ORDER="outputs_ours/curriculum/llama_1B/dolly/pocl_order.pt"
BASE="outputs_ours/llama_1B/dolly"
COMMON="--dataset dolly --student_model $STUDENT --teacher_model $TEACHER \
    --epochs 10 --lr 1e-5 --batch_size 4 --max_seq_len 512 \
    --skip_eval --device cuda:0"
SAGD_ARGS="--teacher_saliency_path $SAL --lambda_noise 0.5 --noise_sigma 0.005 \
    --sagd_every_n_steps 5 --sagd_tau_w 1.0"

# Verify prereq outputs exist
for F in "$SAL" "$GRAD_ORDER" "$POCL_ORDER"; do
    if [ ! -f "$F" ]; then
        echo "ERROR: missing $F. Run scripts/llama_prereq.sh first."
        exit 1
    fi
done

echo "===== GPU 0: LLaMA curriculum (4 configs × 3 seeds) ====="

for SEED in 42 123 456; do
    echo "----- seed $SEED -----"

    # sagd + gradient curriculum
    CKPT="${BASE}/sagd_grad_curriculum/sagd/seed_${SEED}/student_final.pt"
    [ ! -f "$CKPT" ] && echo ">>> sagd_grad_curriculum seed_${SEED}" && \
    python scripts/train.py --method sagd $COMMON $SAGD_ARGS \
        --curriculum_path $GRAD_ORDER --seed $SEED \
        --output_dir "${BASE}/sagd_grad_curriculum/"

    # kd + gradient curriculum
    CKPT="${BASE}/kd_grad_curriculum/standard_kd/seed_${SEED}/student_final.pt"
    [ ! -f "$CKPT" ] && echo ">>> kd_grad_curriculum seed_${SEED}" && \
    python scripts/train.py --method standard_kd $COMMON \
        --curriculum_path $GRAD_ORDER --seed $SEED \
        --output_dir "${BASE}/kd_grad_curriculum/"

    # kd + pocl
    CKPT="${BASE}/kd_pocl/standard_kd/seed_${SEED}/student_final.pt"
    [ ! -f "$CKPT" ] && echo ">>> kd_pocl seed_${SEED}" && \
    python scripts/train.py --method standard_kd $COMMON \
        --curriculum_path $POCL_ORDER --seed $SEED \
        --output_dir "${BASE}/kd_pocl/"

    # sagd + pocl
    CKPT="${BASE}/sagd_pocl/sagd/seed_${SEED}/student_final.pt"
    [ ! -f "$CKPT" ] && echo ">>> sagd_pocl seed_${SEED}" && \
    python scripts/train.py --method sagd $COMMON $SAGD_ARGS \
        --curriculum_path $POCL_ORDER --seed $SEED \
        --output_dir "${BASE}/sagd_pocl/"
done

echo "===== Eval ====="
for CFG in sagd_grad_curriculum kd_grad_curriculum kd_pocl sagd_pocl; do
    case "$CFG" in
        sagd_*) METHOD="sagd" ;;
        kd_*)   METHOD="standard_kd" ;;
    esac
    for SEED in 42 123 456; do
        CKPT="${BASE}/${CFG}/${METHOD}/seed_${SEED}/student_final.pt"
        EVAL="${BASE}/${CFG}/${METHOD}/seed_${SEED}/benchmark_rouge.json"
        [ ! -f "$CKPT" ] && continue
        [ -f "$EVAL" ] && continue
        echo "EVAL >>> ${CFG}/seed_${SEED}"
        python scripts/evaluate_benchmarks.py --student_model "$STUDENT" \
            --student_ckpt "$CKPT" --output_path "$EVAL" --device cuda:0
    done
done

echo "===== GPU 0 curriculum done ====="
