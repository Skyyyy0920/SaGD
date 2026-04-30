#!/bin/bash
# GPU 1: Train 4 curriculum configs × seed 42
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1

SEED=42
SAL="data/teacher_saliency_dolly.pt"
GRAD_ORDER="outputs_ours/curriculum/qwen3_0.6B/dolly/gradient_order.pt"
POCL_ORDER="outputs_ours/curriculum/qwen3_0.6B/dolly/pocl_order.pt"
BASE="outputs_ours/qwen3_0.6B/dolly"
COMMON="--dataset dolly --student_model Qwen/Qwen3-0.6B --epochs 10 --lr 1e-5 --skip_eval --device cuda:0"
SAGD_ARGS="--teacher_saliency_path $SAL --lambda_noise 0.5 --noise_sigma 0.005 --sagd_every_n_steps 5 --sagd_tau_w 1.0"

echo "===== GPU 1: seed $SEED ====="

# sagd + gradient curriculum
CKPT="${BASE}/sagd_grad_curriculum/sagd/seed_${SEED}/student_final.pt"
[ ! -f "$CKPT" ] && echo ">>> sagd_grad_curriculum seed_${SEED}" && \
python scripts/train.py --method sagd $COMMON $SAGD_ARGS --curriculum_path $GRAD_ORDER --seed $SEED --output_dir "${BASE}/sagd_grad_curriculum/"

# kd + gradient curriculum
CKPT="${BASE}/kd_grad_curriculum/standard_kd/seed_${SEED}/student_final.pt"
[ ! -f "$CKPT" ] && echo ">>> kd_grad_curriculum seed_${SEED}" && \
python scripts/train.py --method standard_kd $COMMON --curriculum_path $GRAD_ORDER --seed $SEED --output_dir "${BASE}/kd_grad_curriculum/"

# kd + pocl
CKPT="${BASE}/kd_pocl/standard_kd/seed_${SEED}/student_final.pt"
[ ! -f "$CKPT" ] && echo ">>> kd_pocl seed_${SEED}" && \
python scripts/train.py --method standard_kd $COMMON --curriculum_path $POCL_ORDER --seed $SEED --output_dir "${BASE}/kd_pocl/"

# sagd + pocl
CKPT="${BASE}/sagd_pocl/sagd/seed_${SEED}/student_final.pt"
[ ! -f "$CKPT" ] && echo ">>> sagd_pocl seed_${SEED}" && \
python scripts/train.py --method sagd $COMMON $SAGD_ARGS --curriculum_path $POCL_ORDER --seed $SEED --output_dir "${BASE}/sagd_pocl/"

echo "===== GPU 1 Done ====="
