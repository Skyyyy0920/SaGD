#!/bin/bash
# 3090 smoke test — confirms 8-bit teacher + LLaMA-1B student fits 24GB
# and trains for a few steps without crashing. Runs in ~5-10 minutes.
#
# Usage:
#     CUDA_VISIBLE_DEVICES=0 bash scripts/3090/smoketest.sh
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "===== 3090 smoke test (LLaMA 3.1-8B int8 + 3.2-1B fp32 student) ====="
nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv

python scripts/train.py \
    --method standard_kd --dataset dolly \
    --teacher_model meta-llama/Llama-3.1-8B-Instruct \
    --student_model meta-llama/Llama-3.2-1B-Instruct \
    --load_8bit_teacher --gradient_checkpointing --use_8bit_optimizer \
    --epochs 1 --batch_size 1 --gradient_accumulation 8 \
    --max_seq_len 256 --max_train_samples 50 --skip_eval \
    --output_dir /tmp/smoke_3090/ --device cuda:0

echo "===== Smoke test PASSED — 3090 setup OK ====="
