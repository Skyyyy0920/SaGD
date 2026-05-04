#!/bin/bash
# GPU 3: SAMSum + GSM8K training + eval (8 methods × 3 seeds × 2 datasets)
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=3

STUDENT="Qwen/Qwen3-0.6B"
DEVICE="cuda:0"

echo "===== GPU 3: SAMSum + GSM8K Training ====="
for DATASET in samsum gsm8k; do
  SALIENCY="data/teacher_saliency_${DATASET}.pt"
  for METHOD in sft standard_kd reverse_kl seqkd gkd distillm dakd sagd; do
    EXTRA=""
    case $METHOD in
      gkd) EXTRA="--gkd_beta 0.5" ;;
      distillm) EXTRA="--distillm_alpha 0.5" ;;
      dakd) EXTRA="--bdl_lambda 0.9" ;;
      sagd) EXTRA="--teacher_saliency_path $SALIENCY --lambda_noise 0.5 --noise_sigma 0.005 --sagd_every_n_steps 5 --sagd_tau_w 1.0" ;;
    esac
    for SEED in 42 123 456; do
      CKPT="outputs_task/qwen3_0.6B/${DATASET}/${METHOD}/seed_${SEED}/student_final.pt"
      [ -f "$CKPT" ] && echo "SKIP train ${DATASET}/${METHOD}/seed_${SEED}" && continue
      echo ">>> TRAIN ${DATASET}/${METHOD}/seed_${SEED}"
      python scripts/train.py --method $METHOD --dataset $DATASET \
        --student_model $STUDENT $EXTRA \
        --seed $SEED --output_dir "outputs_task/qwen3_0.6B/${DATASET}/" \
        --epochs 10 --lr 1e-5 --skip_eval --device $DEVICE
    done
  done
done

echo "===== GPU 3: SAMSum + GSM8K Evaluation ====="
for DATASET in samsum gsm8k; do
  MAX_NEW=256
  for METHOD in sft standard_kd reverse_kl seqkd gkd distillm dakd sagd; do
    for SEED in 42 123 456; do
      CKPT="outputs_task/qwen3_0.6B/${DATASET}/${METHOD}/seed_${SEED}/student_final.pt"
      EVAL="outputs_task/qwen3_0.6B/${DATASET}/${METHOD}/seed_${SEED}/eval_metrics.json"
      [ ! -f "$CKPT" ] && continue
      [ -f "$EVAL" ] && echo "SKIP eval ${DATASET}/${METHOD}/seed_${SEED}" && continue
      echo ">>> EVAL ${DATASET}/${METHOD}/seed_${SEED}"
      python scripts/evaluate.py --student_model $STUDENT \
        --student_ckpt "$CKPT" --dataset $DATASET --subset test \
        --max_new_tokens $MAX_NEW --output_path "$EVAL" \
        --skip_bertscore --device $DEVICE --seed $SEED
    done
  done
done

echo "===== GPU 3 Done ====="
