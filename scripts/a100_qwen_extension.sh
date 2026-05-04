#!/bin/bash
# A100 Qwen extension — loss ablation + λ sweep + GPDS random control.
# Fills the missing tables/figures DA-KD has but we don't yet:
#   - Loss replacement ablation (Table 5 equivalent)
#   - λ hyperparameter sweep (Figure 4 equivalent)
#   - GPDS w/o gradient-PCA (random data order control row)
#
# Usage:
#     CUDA_VISIBLE_DEVICES=3 bash scripts/a100_qwen_extension.sh
#
# ~30h total on a single A100 (loss_abl 12h + λ sweep 15h + sagd_random 3h).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

TEACHER="Qwen/Qwen3-8B"
STUDENT="Qwen/Qwen3-0.6B"
SAL="data/teacher_saliency_dolly.pt"
BASE_OURS="outputs_ours/qwen3_0.6B/dolly"

COMMON="--dataset dolly --teacher_model $TEACHER --student_model $STUDENT \
    --epochs 10 --lr 1e-5 --batch_size 4 --gradient_accumulation 8 \
    --max_seq_len 512 --skip_eval --device cuda:0"

eval_ckpt () {
    local _ckpt="$1" _eval="$2"
    [ ! -f "$_ckpt" ] && return 0
    [ -f "$_eval" ] && { echo "SKIP $_eval"; return 0; }
    echo ">>> EVAL $_eval"
    python scripts/evaluate_benchmarks.py \
        --student_model "$STUDENT" --student_ckpt "$_ckpt" \
        --output_path "$_eval" --device cuda:0
}

SEED=42

# ===== Section 1: Loss ablation (Table 5) =====
OUT="${BASE_OURS}/loss_ablation/"
echo "===== Section 1: Loss ablation ====="
for METHOD in reverse_kl gkd distillm dakd; do
    CKPT="${OUT}${METHOD}/seed_${SEED}/student_final.pt"
    if [ ! -f "$CKPT" ]; then
        echo ">>> loss_ablation/${METHOD}/seed_${SEED}"
        case "$METHOD" in
            gkd)      EXTRA="--gkd_beta 0.5" ;;
            distillm) EXTRA="--distillm_alpha 0.5" ;;
            dakd)     EXTRA="--bdl_lambda 0.9" ;;
            *)        EXTRA="" ;;
        esac
        python scripts/train.py --method "$METHOD" $COMMON $EXTRA \
            --seed "$SEED" --output_dir "$OUT"
    fi
    eval_ckpt "${OUT}${METHOD}/seed_${SEED}/student_final.pt" \
              "${OUT}${METHOD}/seed_${SEED}/benchmark_rouge.json"
done

# ===== Section 2: λ hyperparameter sweep (Figure 4) =====
echo "===== Section 2: λ sweep ====="
for LAMBDA in 0.1 1.0 2.0 5.0; do
    OUT="${BASE_OURS}/lambda_${LAMBDA}/"
    CKPT="${OUT}sagd/seed_${SEED}/student_final.pt"
    if [ ! -f "$CKPT" ]; then
        echo ">>> lambda=${LAMBDA} seed_${SEED}"
        python scripts/train.py --method sagd $COMMON \
            --teacher_saliency_path "$SAL" \
            --lambda_noise "$LAMBDA" --noise_sigma 0.005 \
            --sagd_every_n_steps 5 --sagd_tau_w 1.0 \
            --seed "$SEED" --output_dir "$OUT"
    fi
    eval_ckpt "$CKPT" "${OUT}sagd/seed_${SEED}/benchmark_rouge.json"
done

# ===== Section 3: GPDS random data ordering control =====
echo "===== Section 3: GPDS random control ====="
RAND_ORDER="outputs_ours/curriculum/qwen3_0.6B/dolly/random_order.pt"
if [ ! -f "$RAND_ORDER" ]; then
    python -c "
import torch, random
random.seed(42)
indices = list(range(14011))
random.shuffle(indices)
torch.save({
    'sorted_indices': torch.tensor(indices, dtype=torch.long),
    'scores': torch.zeros(14011, dtype=torch.float32),
    'metadata': {'source': 'random_order_seed42'},
}, '$RAND_ORDER')
print(f'Random order saved to $RAND_ORDER ({len(indices)} samples)')
"
fi
OUT="${BASE_OURS}/sagd_random/"
CKPT="${OUT}sagd/seed_${SEED}/student_final.pt"
if [ ! -f "$CKPT" ]; then
    echo ">>> sagd_random seed_${SEED}"
    python scripts/train.py --method sagd $COMMON \
        --teacher_saliency_path "$SAL" \
        --lambda_noise 0.5 --noise_sigma 0.005 \
        --sagd_every_n_steps 5 --sagd_tau_w 1.0 \
        --curriculum_path "$RAND_ORDER" \
        --seed "$SEED" --output_dir "$OUT"
fi
eval_ckpt "$CKPT" "${OUT}sagd/seed_${SEED}/benchmark_rouge.json"

echo "===== a100_qwen_extension done ====="
