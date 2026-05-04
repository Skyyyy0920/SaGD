#!/bin/bash
# SaGD Hyperparameter Sweep
# 用法: bash scripts/sweep.sh <GPU_ID>
# 例如: bash scripts/sweep.sh 3

set -e
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

GPU=${1:-0}
SEED=42
BASE_DIR="outputs_sweep"

echo "=== SaGD Hyperparameter Sweep on GPU $GPU ==="

# ---------------------------------------------------------
# 1. σ sweep (most critical — controls Jacobian matching strength)
# ---------------------------------------------------------
echo ""
echo ">>> σ sweep (λ=0.5, τ_w=1.0, N=5 fixed) <<<"
for SIGMA in 0.001 0.005 0.01 0.02 0.05 0.1; do
    DIR="${BASE_DIR}/sigma_${SIGMA}/sagd/seed_${SEED}"
    if [ -f "$DIR/eval_metrics.json" ]; then
        echo "[SKIP] σ=$SIGMA already done"
        continue
    fi
    echo "[RUN] σ=$SIGMA"
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_noise 0.5 --noise_sigma $SIGMA --sagd_tau_w 1.0 --sagd_every_n_steps 5 \
        --seed $SEED --output_dir ${BASE_DIR}/sigma_${SIGMA}/ \
        --epochs 1 --device cuda:$GPU
done

# ---------------------------------------------------------
# 2. λ sweep (noise KL weight)
# ---------------------------------------------------------
echo ""
echo ">>> λ sweep (σ=0.01, τ_w=1.0, N=5 fixed) <<<"
for LAMBDA in 0.1 0.2 0.5 1.0 2.0 5.0; do
    DIR="${BASE_DIR}/lambda_${LAMBDA}/sagd/seed_${SEED}"
    if [ -f "$DIR/eval_metrics.json" ]; then
        echo "[SKIP] λ=$LAMBDA already done"
        continue
    fi
    echo "[RUN] λ=$LAMBDA"
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_noise $LAMBDA --noise_sigma 0.01 --sagd_tau_w 1.0 --sagd_every_n_steps 5 \
        --seed $SEED --output_dir ${BASE_DIR}/lambda_${LAMBDA}/ \
        --epochs 1 --device cuda:$GPU
done

# ---------------------------------------------------------
# 3. τ_w sweep (DRO strength)
# ---------------------------------------------------------
echo ""
echo ">>> τ_w sweep (λ=0.5, σ=0.01, N=5 fixed) <<<"
for TAU in 0.1 0.5 1.0 2.0 5.0 100.0; do
    DIR="${BASE_DIR}/tau_${TAU}/sagd/seed_${SEED}"
    if [ -f "$DIR/eval_metrics.json" ]; then
        echo "[SKIP] τ_w=$TAU already done"
        continue
    fi
    echo "[RUN] τ_w=$TAU"
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_noise 0.5 --noise_sigma 0.01 --sagd_tau_w $TAU --sagd_every_n_steps 5 \
        --seed $SEED --output_dir ${BASE_DIR}/tau_${TAU}/ \
        --epochs 1 --device cuda:$GPU
done

# ---------------------------------------------------------
# 4. N sweep (SaGD step frequency)
# ---------------------------------------------------------
echo ""
echo ">>> N sweep (λ=0.5, σ=0.01, τ_w=1.0 fixed) <<<"
for N in 1 3 5 10; do
    DIR="${BASE_DIR}/every_n_${N}/sagd/seed_${SEED}"
    if [ -f "$DIR/eval_metrics.json" ]; then
        echo "[SKIP] N=$N already done"
        continue
    fi
    echo "[RUN] N=$N"
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_noise 0.5 --noise_sigma 0.01 --sagd_tau_w 1.0 --sagd_every_n_steps $N \
        --seed $SEED --output_dir ${BASE_DIR}/every_n_${N}/ \
        --epochs 1 --device cuda:$GPU
done

# ---------------------------------------------------------
# 5. Collect results
# ---------------------------------------------------------
echo ""
echo "=== Sweep Results ==="
python -c "
import json, os, glob

print(f'{'Config':<35} {'EM':>6} {'F1':>6} {'ROUGE':>6} {'PPL':>6}')
print('-' * 65)

# Standard KD baseline
bl = 'outputs/standard_kd/seed_42/eval_metrics.json'
if os.path.exists(bl):
    m = json.load(open(bl))
    print(f'{'standard_kd (baseline)':<35} {m[\"exact_match\"]:6.3f} {m[\"token_f1\"]:6.3f} {m[\"rouge_l_f\"]:6.3f} {m[\"perplexity\"]:6.2f}')

# Sweep results
for pattern in ['sigma_*', 'lambda_*', 'tau_*', 'every_n_*']:
    dirs = sorted(glob.glob(f'${BASE_DIR}/{pattern}/sagd/seed_${SEED}/eval_metrics.json'))
    for path in dirs:
        parts = path.split('/')
        config = parts[-4]  # e.g. sigma_0.01
        try:
            m = json.load(open(path))
            print(f'{config:<35} {m[\"exact_match\"]:6.3f} {m[\"token_f1\"]:6.3f} {m[\"rouge_l_f\"]:6.3f} {m[\"perplexity\"]:6.2f}')
        except:
            print(f'{config:<35} ERROR')
"

echo ""
echo "=== Sweep Complete ==="
