#!/bin/bash
# =============================================================================
# Run all remaining experiments on GPU 1/2/3 in parallel
#   GPU 1: Teacher gradient profiling + curriculum training + eval
#   GPU 2: SQuAD training (8 methods × 3 seeds)
#   GPU 3: SAMSum + GSM8K training (8 methods × 3 seeds each)
# =============================================================================

cd "$(dirname "$0")/.."
mkdir -p logs

echo "Starting 3 GPU jobs at $(date)"

# === GPU 1: Our method (teacher gradient profiling + curriculum) ===
CUDA_VISIBLE_DEVICES=1 bash -c '
cd /data/tianhao/SaGD
rm -rf outputs_ours/gradient_pca/
rm -f outputs_ours/curriculum/qwen3_0.6B/dolly/gradient_order.pt
rm -f outputs_ours/curriculum/qwen3_0.6B/dolly/pocl_order.pt
rm -rf outputs_ours/qwen3_0.6B/dolly/sagd_grad_curriculum
rm -rf outputs_ours/qwen3_0.6B/dolly/kd_grad_curriculum
rm -rf outputs_ours/qwen3_0.6B/dolly/kd_pocl
rm -rf outputs_ours/qwen3_0.6B/dolly/sagd_pocl
CUDA_DEVICE=cuda:0 bash scripts/run_ours.sh all
' > logs/gpu1_ours.log 2>&1 &
PID1=$!
echo "[GPU 1] PID=$PID1 — teacher gradient profiling + curriculum"

# === GPU 2: SQuAD training ===
CUDA_VISIBLE_DEVICES=2 bash -c '
cd /data/tianhao/SaGD
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
for METHOD in sft standard_kd reverse_kl seqkd gkd distillm dakd sagd; do
  EXTRA=""
  case $METHOD in
    gkd) EXTRA="--gkd_beta 0.5" ;;
    distillm) EXTRA="--distillm_alpha 0.5" ;;
    dakd) EXTRA="--bdl_lambda 0.9" ;;
    sagd) EXTRA="--teacher_saliency_path data/teacher_saliency_squad.pt --lambda_noise 0.5 --noise_sigma 0.005 --sagd_every_n_steps 5 --sagd_tau_w 1.0" ;;
  esac
  for SEED in 42 123 456; do
    CKPT="outputs_task/qwen3_0.6B/squad/${METHOD}/seed_${SEED}/student_final.pt"
    [ -f "$CKPT" ] && echo "SKIP squad/${METHOD}/seed_${SEED}" && continue
    echo ">>> squad/${METHOD}/seed_${SEED}"
    python scripts/train.py --method $METHOD --dataset squad \
      --student_model Qwen/Qwen3-0.6B $EXTRA \
      --seed $SEED --output_dir outputs_task/qwen3_0.6B/squad/ \
      --epochs 10 --lr 1e-5 --skip_eval --device cuda:0
  done
done
echo "GPU2 training done. Starting eval..."
for METHOD in sft standard_kd reverse_kl seqkd gkd distillm dakd sagd; do
  for SEED in 42 123 456; do
    CKPT="outputs_task/qwen3_0.6B/squad/${METHOD}/seed_${SEED}/student_final.pt"
    EVAL="outputs_task/qwen3_0.6B/squad/${METHOD}/seed_${SEED}/eval_metrics.json"
    [ ! -f "$CKPT" ] && continue
    [ -f "$EVAL" ] && continue
    echo "EVAL >>> squad/${METHOD}/seed_${SEED}"
    python scripts/evaluate.py --student_model Qwen/Qwen3-0.6B \
      --student_ckpt "$CKPT" --dataset squad --subset test \
      --max_new_tokens 32 --output_path "$EVAL" \
      --skip_bertscore --device cuda:0 --seed $SEED
  done
done
' > logs/gpu2_squad.log 2>&1 &
PID2=$!
echo "[GPU 2] PID=$PID2 — SQuAD train + eval"

# === GPU 3: SAMSum + GSM8K training ===
CUDA_VISIBLE_DEVICES=3 bash -c '
cd /data/tianhao/SaGD
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
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
      [ -f "$CKPT" ] && echo "SKIP ${DATASET}/${METHOD}/seed_${SEED}" && continue
      echo ">>> ${DATASET}/${METHOD}/seed_${SEED}"
      python scripts/train.py --method $METHOD --dataset $DATASET \
        --student_model Qwen/Qwen3-0.6B $EXTRA \
        --seed $SEED --output_dir "outputs_task/qwen3_0.6B/${DATASET}/" \
        --epochs 10 --lr 1e-5 --skip_eval --device cuda:0
    done
  done
done
echo "GPU3 training done. Starting eval..."
for DATASET in samsum gsm8k; do
  MAX_NEW=256
  for METHOD in sft standard_kd reverse_kl seqkd gkd distillm dakd sagd; do
    for SEED in 42 123 456; do
      CKPT="outputs_task/qwen3_0.6B/${DATASET}/${METHOD}/seed_${SEED}/student_final.pt"
      EVAL="outputs_task/qwen3_0.6B/${DATASET}/${METHOD}/seed_${SEED}/eval_metrics.json"
      [ ! -f "$CKPT" ] && continue
      [ -f "$EVAL" ] && continue
      echo "EVAL >>> ${DATASET}/${METHOD}/seed_${SEED}"
      python scripts/evaluate.py --student_model Qwen/Qwen3-0.6B \
        --student_ckpt "$CKPT" --dataset $DATASET --subset test \
        --max_new_tokens $MAX_NEW --output_path "$EVAL" \
        --skip_bertscore --device cuda:0 --seed $SEED
    done
  done
done
' > logs/gpu3_samsum_gsm8k.log 2>&1 &
PID3=$!
echo "[GPU 3] PID=$PID3 — SAMSum + GSM8K train + eval"

echo ""
echo "All 3 jobs launched. Monitor with:"
echo "  tail -f logs/gpu1_ours.log"
echo "  tail -f logs/gpu2_squad.log"
echo "  tail -f logs/gpu3_samsum_gsm8k.log"

wait $PID1 $PID2 $PID3
echo "All done at $(date)"
