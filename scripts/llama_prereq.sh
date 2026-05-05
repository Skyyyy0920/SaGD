#!/bin/bash
# LLaMA: Teacher saliency + gradient profiling.
# Run this FIRST. Takes ~2h on a single A100.
# Pin GPU via CUDA_VISIBLE_DEVICES, e.g.:
#     CUDA_VISIBLE_DEVICES=3 bash scripts/llama_prereq.sh
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

TEACHER="meta-llama/Llama-3.1-8B-Instruct"
STUDENT="meta-llama/Llama-3.2-1B-Instruct"
SAL_PATH="data/teacher_saliency_dolly_llama.pt"
PCA_DIR="outputs_ours/gradient_pca/llama_1B/dolly"

# 1. Teacher saliency
if [ ! -f "$SAL_PATH" ]; then
    echo "===== Step 1: Teacher saliency (~30 min) ====="
    python scripts/precompute_teacher_saliency.py \
        --model_name "$TEACHER" --tokenizer_name "$STUDENT" \
        --dataset dolly --output_path "$SAL_PATH" \
        --batch_size 4 --max_seq_len 512 --device cuda:0
else
    echo "[SKIP] Teacher saliency exists"
fi

# 2. Gradient profiling
if [ ! -f "${PCA_DIR}/gradient_profile.npz" ]; then
    echo "===== Step 2: Gradient profiling (~1.5h) ====="
    mkdir -p "$PCA_DIR"
    python scripts/gradient_pca_selection.py profile \
        --teacher_model "$TEACHER" --dataset dolly \
        --output_dir "$PCA_DIR" --device cuda:0
else
    echo "[SKIP] Gradient profile exists"
fi

# 3. Generate curriculum orders
CUR_DIR="outputs_ours/curriculum/llama_1B/dolly"
mkdir -p "$CUR_DIR"

if [ ! -f "${CUR_DIR}/gradient_order.pt" ]; then
    echo "===== Step 3: Gradient curriculum order ====="
    python scripts/compute_curriculum.py \
        --pca_path "${PCA_DIR}/gradient_profile.npz" \
        --output_path "${CUR_DIR}/gradient_order.pt" \
        --top_r 50 --is_gradient
fi

if [ ! -f "${CUR_DIR}/pocl_order.pt" ]; then
    echo "===== Step 4: POCL order ====="
    python -c "
import torch, numpy as np
data = np.load('${PCA_DIR}/gradient_profile.npz')
losses, indices = data['losses'], data['indices']
order = indices[np.argsort(losses)]
torch.save({'sorted_indices': torch.tensor(order, dtype=torch.long),
            'scores': torch.tensor(np.sort(losses), dtype=torch.float32),
            'metadata': {'source': 'pocl_easy_first'}},
           '${CUR_DIR}/pocl_order.pt')
print(f'POCL order saved: {len(order)} samples')
"
fi

echo "===== Prereq done. Now run llama_curriculum.sh on GPU 0 ====="
