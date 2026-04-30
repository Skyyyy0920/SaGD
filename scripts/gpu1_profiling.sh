#!/bin/bash
# Step 1: Teacher gradient profiling (single GPU, ~1-2 hours)
# Run this FIRST, wait for it to finish, then run gpu1/2/3_curriculum.sh
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1

rm -rf outputs_ours/gradient_pca/
mkdir -p outputs_ours/gradient_pca/qwen3_0.6B/dolly

echo "===== Teacher Gradient Profiling (GPU 1) ====="
python scripts/gradient_pca_selection.py profile \
    --teacher_model Qwen/Qwen3-8B --dataset dolly \
    --output_dir outputs_ours/gradient_pca/qwen3_0.6B/dolly \
    --device cuda:0

echo "===== Generating curriculum orders ====="
rm -f outputs_ours/curriculum/qwen3_0.6B/dolly/gradient_order.pt
rm -f outputs_ours/curriculum/qwen3_0.6B/dolly/pocl_order.pt
mkdir -p outputs_ours/curriculum/qwen3_0.6B/dolly

# Gradient PC order
python scripts/compute_curriculum.py \
    --pca_path outputs_ours/gradient_pca/qwen3_0.6B/dolly/gradient_profile.npz \
    --output_path outputs_ours/curriculum/qwen3_0.6B/dolly/gradient_order.pt \
    --top_r 50 --is_gradient

# POCL order (easy-first by teacher NLL)
python -c "
import torch, numpy as np
data = np.load('outputs_ours/gradient_pca/qwen3_0.6B/dolly/gradient_profile.npz')
losses, indices = data['losses'], data['indices']
order = indices[np.argsort(losses)]
scores = np.sort(losses)
torch.save({'sorted_indices': torch.tensor(order, dtype=torch.long),
            'scores': torch.tensor(scores, dtype=torch.float32),
            'metadata': {'source': 'pocl_easy_first', 'n_samples': len(order)}},
           'outputs_ours/curriculum/qwen3_0.6B/dolly/pocl_order.pt')
print(f'POCL order saved: {len(order)} samples')
"

echo "===== Done! Now run gpu1/2/3_curriculum.sh ====="
