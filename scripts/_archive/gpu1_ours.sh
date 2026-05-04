#!/bin/bash
# GPU 1: Teacher gradient profiling + curriculum training + eval
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE=cuda:0

# Clean old student-gradient results
rm -rf outputs_ours/gradient_pca/
rm -f outputs_ours/curriculum/qwen3_0.6B/dolly/gradient_order.pt
rm -f outputs_ours/curriculum/qwen3_0.6B/dolly/pocl_order.pt
rm -rf outputs_ours/qwen3_0.6B/dolly/sagd_grad_curriculum
rm -rf outputs_ours/qwen3_0.6B/dolly/kd_grad_curriculum
rm -rf outputs_ours/qwen3_0.6B/dolly/kd_pocl
rm -rf outputs_ours/qwen3_0.6B/dolly/sagd_pocl

bash scripts/run_ours.sh all
