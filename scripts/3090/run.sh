#!/bin/bash
# 3090 LLaMA experiment dispatcher.
# Usage (one terminal per GPU):
#     CUDA_VISIBLE_DEVICES=0 bash scripts/3090/run.sh curriculum_sagd_grad
#     CUDA_VISIBLE_DEVICES=1 bash scripts/3090/run.sh curriculum_kd_grad
#     CUDA_VISIBLE_DEVICES=2 bash scripts/3090/run.sh curriculum_kd_pocl
#     CUDA_VISIBLE_DEVICES=3 bash scripts/3090/run.sh curriculum_sagd_pocl
#     CUDA_VISIBLE_DEVICES=4 bash scripts/3090/run.sh samsum
#     CUDA_VISIBLE_DEVICES=5 bash scripts/3090/run.sh gsm8k
#     CUDA_VISIBLE_DEVICES=6 bash scripts/3090/run.sh loss_ablation
#     CUDA_VISIBLE_DEVICES=7 bash scripts/3090/run.sh lambda_sweep
# All tasks are skip-if-exists idempotent.
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
source "$(dirname "$0")/_common.sh"

TASK="${1:-help}"
SEEDS_DEFAULT="42 123 456"

usage () {
    cat <<EOF
Tasks:
  curriculum_sagd_grad   sagd_grad_curriculum × {42,123,456}      (~18h)
  curriculum_kd_grad     kd_grad_curriculum   × {42,123,456}      (~18h)
  curriculum_kd_pocl     kd_pocl              × {42,123,456}      (~18h)
  curriculum_sagd_pocl   sagd_pocl            × {42,123,456}      (~18h)
  samsum                 8 methods × seed 42 SAMSum               (~48h)
  gsm8k                  8 methods × seed 42 GSM8K                (~48h)
  loss_ablation          {RKL,GKD,DistiLLM,DA-KD} × seed 42 Dolly (~36h)
  lambda_sweep           lambda ∈ {0.1,1.0,2.0,5.0} × seed 42     (~24h)
Each task ends with eval (5 benchmarks) on each ckpt produced.
EOF
}

case "$TASK" in
help|--help|-h)
    usage; exit 0 ;;

curriculum_sagd_grad)
    for SEED in $SEEDS_DEFAULT; do
        OUT="${BASE_OURS}/sagd_grad_curriculum/"
        CKPT="${OUT}sagd/seed_${SEED}/student_final.pt"
        if [ ! -f "$CKPT" ]; then
            echo ">>> sagd_grad_curriculum seed_${SEED}"
            python scripts/train.py --method sagd $COMMON_DOLLY $SAGD_ARGS \
                --curriculum_path "$GRAD_ORDER" --seed "$SEED" --output_dir "$OUT"
        fi
        eval_ckpt "${BASE_OURS}/sagd_grad_curriculum" "sagd" "$SEED"
    done
    ;;

curriculum_kd_grad)
    for SEED in $SEEDS_DEFAULT; do
        OUT="${BASE_OURS}/kd_grad_curriculum/"
        CKPT="${OUT}standard_kd/seed_${SEED}/student_final.pt"
        if [ ! -f "$CKPT" ]; then
            echo ">>> kd_grad_curriculum seed_${SEED}"
            python scripts/train.py --method standard_kd $COMMON_DOLLY \
                --curriculum_path "$GRAD_ORDER" --seed "$SEED" --output_dir "$OUT"
        fi
        eval_ckpt "${BASE_OURS}/kd_grad_curriculum" "standard_kd" "$SEED"
    done
    ;;

curriculum_kd_pocl)
    for SEED in $SEEDS_DEFAULT; do
        OUT="${BASE_OURS}/kd_pocl/"
        CKPT="${OUT}standard_kd/seed_${SEED}/student_final.pt"
        if [ ! -f "$CKPT" ]; then
            echo ">>> kd_pocl seed_${SEED}"
            python scripts/train.py --method standard_kd $COMMON_DOLLY \
                --curriculum_path "$POCL_ORDER" --seed "$SEED" --output_dir "$OUT"
        fi
        eval_ckpt "${BASE_OURS}/kd_pocl" "standard_kd" "$SEED"
    done
    ;;

curriculum_sagd_pocl)
    for SEED in $SEEDS_DEFAULT; do
        OUT="${BASE_OURS}/sagd_pocl/"
        CKPT="${OUT}sagd/seed_${SEED}/student_final.pt"
        if [ ! -f "$CKPT" ]; then
            echo ">>> sagd_pocl seed_${SEED}"
            python scripts/train.py --method sagd $COMMON_DOLLY $SAGD_ARGS \
                --curriculum_path "$POCL_ORDER" --seed "$SEED" --output_dir "$OUT"
        fi
        eval_ckpt "${BASE_OURS}/sagd_pocl" "sagd" "$SEED"
    done
    ;;

samsum)
    SEED=42
    OUT="${BASE_TASKS}/samsum/"
    for METHOD in sft standard_kd reverse_kl seqkd gkd distillm dakd; do
        CKPT="${OUT}${METHOD}/seed_${SEED}/student_final.pt"
        [ -f "$CKPT" ] && continue
        echo ">>> samsum/${METHOD}/seed_${SEED}"
        case "$METHOD" in
            gkd)      EXTRA="--gkd_beta 0.5" ;;
            distillm) EXTRA="--distillm_alpha 0.5" ;;
            dakd)     EXTRA="--bdl_lambda 0.9" ;;
            *)        EXTRA="" ;;
        esac
        python scripts/train.py --method "$METHOD" $COMMON_SAMSUM $EXTRA \
            --seed "$SEED" --output_dir "$OUT"
    done
    SAGD_CKPT="${OUT}sagd/seed_${SEED}/student_final.pt"
    if [ ! -f "$SAGD_CKPT" ]; then
        SAL_SAMSUM="data/teacher_saliency_samsum_llama.pt"
        if [ ! -f "$SAL_SAMSUM" ]; then
            echo ">>> precompute saliency for samsum"
            python scripts/precompute_teacher_saliency.py \
                --model_name "$TEACHER" --tokenizer_name "$STUDENT" \
                --dataset samsum --output_path "$SAL_SAMSUM" \
                --batch_size 1 --max_seq_len 512 --device cuda:0
        fi
        echo ">>> samsum/sagd/seed_${SEED}"
        python scripts/train.py --method sagd $COMMON_SAMSUM \
            --teacher_saliency_path "$SAL_SAMSUM" \
            --lambda_noise 0.5 --noise_sigma 0.005 \
            --sagd_every_n_steps 5 --sagd_tau_w 1.0 \
            --seed "$SEED" --output_dir "$OUT"
    fi
    for METHOD in sft standard_kd reverse_kl seqkd gkd distillm dakd sagd; do
        eval_ckpt "$OUT" "$METHOD" "$SEED"
    done
    ;;

gsm8k)
    SEED=42
    OUT="${BASE_TASKS}/gsm8k/"
    for METHOD in sft standard_kd reverse_kl seqkd gkd distillm dakd; do
        CKPT="${OUT}${METHOD}/seed_${SEED}/student_final.pt"
        [ -f "$CKPT" ] && continue
        echo ">>> gsm8k/${METHOD}/seed_${SEED}"
        case "$METHOD" in
            gkd)      EXTRA="--gkd_beta 0.5" ;;
            distillm) EXTRA="--distillm_alpha 0.5" ;;
            dakd)     EXTRA="--bdl_lambda 0.9" ;;
            *)        EXTRA="" ;;
        esac
        python scripts/train.py --method "$METHOD" $COMMON_GSM8K $EXTRA \
            --seed "$SEED" --output_dir "$OUT"
    done
    SAGD_CKPT="${OUT}sagd/seed_${SEED}/student_final.pt"
    if [ ! -f "$SAGD_CKPT" ]; then
        SAL_GSM="data/teacher_saliency_gsm8k_llama.pt"
        if [ ! -f "$SAL_GSM" ]; then
            echo ">>> precompute saliency for gsm8k"
            python scripts/precompute_teacher_saliency.py \
                --model_name "$TEACHER" --tokenizer_name "$STUDENT" \
                --dataset gsm8k --output_path "$SAL_GSM" \
                --batch_size 1 --max_seq_len 512 --device cuda:0
        fi
        echo ">>> gsm8k/sagd/seed_${SEED}"
        python scripts/train.py --method sagd $COMMON_GSM8K \
            --teacher_saliency_path "$SAL_GSM" \
            --lambda_noise 0.5 --noise_sigma 0.005 \
            --sagd_every_n_steps 5 --sagd_tau_w 1.0 \
            --seed "$SEED" --output_dir "$OUT"
    fi
    for METHOD in sft standard_kd reverse_kl seqkd gkd distillm dakd sagd; do
        eval_ckpt "$OUT" "$METHOD" "$SEED"
    done
    ;;

loss_ablation)
    SEED=42
    OUT="${BASE_OURS}/loss_ablation/"
    for METHOD in reverse_kl gkd distillm dakd; do
        CKPT="${OUT}${METHOD}/seed_${SEED}/student_final.pt"
        [ -f "$CKPT" ] && continue
        echo ">>> loss_ablation/${METHOD}/seed_${SEED}"
        case "$METHOD" in
            gkd)      EXTRA="--gkd_beta 0.5" ;;
            distillm) EXTRA="--distillm_alpha 0.5" ;;
            dakd)     EXTRA="--bdl_lambda 0.9" ;;
            *)        EXTRA="" ;;
        esac
        python scripts/train.py --method "$METHOD" $COMMON_DOLLY $EXTRA \
            --seed "$SEED" --output_dir "$OUT"
    done
    for METHOD in reverse_kl gkd distillm dakd; do
        eval_ckpt "$OUT" "$METHOD" "$SEED"
    done
    ;;

lambda_sweep)
    SEED=42
    for LAMBDA in 0.1 1.0 2.0 5.0; do
        OUT="${BASE_OURS}/lambda_${LAMBDA}/"
        CKPT="${OUT}sagd/seed_${SEED}/student_final.pt"
        if [ ! -f "$CKPT" ]; then
            echo ">>> lambda_sweep λ=${LAMBDA} seed_${SEED}"
            python scripts/train.py --method sagd $COMMON_DOLLY \
                --teacher_saliency_path "$SAL" \
                --lambda_noise "$LAMBDA" --noise_sigma 0.005 \
                --sagd_every_n_steps 5 --sagd_tau_w 1.0 \
                --seed "$SEED" --output_dir "$OUT"
        fi
        eval_ckpt "$OUT" "sagd" "$SEED"
    done
    ;;

*)
    echo "Unknown task: $TASK"
    usage
    exit 1
    ;;
esac

echo "===== run.sh task=$TASK done ====="
