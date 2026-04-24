#!/bin/bash
# =============================================================================
# OUR METHOD: SaGD + PC-Guided Curriculum
#
# Three stages:
#   Stage 1: Prerequisites (saliency cache + PCA profiling + curriculum order)
#   Stage 2: Training (SaGD × curriculum strategies × seeds)
#   Stage 3: Evaluation
#
# Usage:
#   bash scripts/run_ours.sh           # run everything
#   bash scripts/run_ours.sh prereq    # only prerequisites
#   bash scripts/run_ours.sh train     # only training (prereqs must exist)
#   bash scripts/run_ours.sh eval      # only evaluation (checkpoints must exist)
# =============================================================================

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# ---- Configuration ----
DEVICE="${CUDA_DEVICE:-cuda:0}"          # override with CUDA_DEVICE=cuda:1 etc.
TEACHER="Qwen/Qwen3-8B"
STUDENT="Qwen/Qwen3-0.6B"
STUDENT_TAG="qwen3_0.6B"
DATASET="dolly"

EPOCHS=10
LR=1e-5
BATCH_SIZE=4                             # SaGD auto-halves internally
MAX_SEQ_LEN=512
SEEDS=(42 123 456)

# SaGD hyperparameters
LAMBDA_NOISE=0.5
NOISE_SIGMA=0.005
SAGD_TAU_W=1.0
SAGD_N=5

# Curriculum schedule: 30% for first third of epochs, 60% for second, 100% for rest
CURRICULUM_SCHEDULE="0.3,0.6,1.0"

# Directories
DATA_DIR="data"
OUTPUT_BASE="outputs_ours/${STUDENT_TAG}/${DATASET}"
CURRICULUM_DIR="outputs_ours/curriculum/${STUDENT_TAG}/${DATASET}"
LOG_DIR="logs/ours_${STUDENT_TAG}"

mkdir -p "$DATA_DIR" "$OUTPUT_BASE" "$CURRICULUM_DIR" "$LOG_DIR"

STAGE="${1:-all}"

# =============================================================================
# Stage 1: Prerequisites
# =============================================================================
run_prereq() {
    echo ""
    echo "===== Stage 1: Prerequisites ====="

    # 1a. Teacher saliency cache
    SALIENCY_PATH="${DATA_DIR}/teacher_saliency_${DATASET}.pt"
    if [ -f "$SALIENCY_PATH" ]; then
        echo "[PREREQ] Teacher saliency cache exists: $SALIENCY_PATH"
    else
        echo "[PREREQ] Computing teacher saliency..."
        python scripts/precompute_teacher_saliency.py \
            --model_name "$TEACHER" --dataset "$DATASET" \
            --output_path "$SALIENCY_PATH" \
            --batch_size 4 --max_seq_len $MAX_SEQ_LEN \
            --device "$DEVICE" \
            2>&1 | tee "${LOG_DIR}/precompute_saliency.log"
    fi

    # 1b. Saliency PCA (for saliency-based curriculum)
    SAL_PCA_DIR="outputs_ours/saliency_pca/${STUDENT_TAG}/${DATASET}"
    SAL_PCA_DATA="${SAL_PCA_DIR}/pca_data.npz"
    if [ -f "$SAL_PCA_DATA" ]; then
        echo "[PREREQ] Saliency PCA exists: $SAL_PCA_DATA"
    else
        echo "[PREREQ] Running saliency PCA..."
        mkdir -p "$SAL_PCA_DIR"
        python scripts/verify_saliency_pca.py \
            --student_model "$STUDENT" \
            --teacher_saliency_path "$SALIENCY_PATH" \
            --dataset "$DATASET" \
            --output_dir "$SAL_PCA_DIR" \
            --device "$DEVICE" --fp16 \
            2>&1 | tee "${LOG_DIR}/saliency_pca.log"
    fi

    # 1c. Gradient PCA (for gradient-based curriculum)
    GRAD_PCA_DIR="outputs_ours/gradient_pca/${STUDENT_TAG}/${DATASET}"
    GRAD_PROFILE="${GRAD_PCA_DIR}/gradient_profile.npz"
    if [ -f "$GRAD_PROFILE" ]; then
        echo "[PREREQ] Gradient PCA exists: $GRAD_PROFILE"
    else
        echo "[PREREQ] Running gradient PCA profiling (~40 min)..."
        mkdir -p "$GRAD_PCA_DIR"
        python scripts/gradient_pca_selection.py profile \
            --teacher_model "$TEACHER" \
            --student_model "$STUDENT" \
            --dataset "$DATASET" \
            --output_dir "$GRAD_PCA_DIR" \
            --device "$DEVICE" \
            2>&1 | tee "${LOG_DIR}/gradient_pca.log"
    fi

    # 1d. Generate curriculum orders
    SAL_ORDER="${CURRICULUM_DIR}/saliency_order.pt"
    if [ -f "$SAL_ORDER" ]; then
        echo "[PREREQ] Saliency curriculum order exists: $SAL_ORDER"
    else
        echo "[PREREQ] Computing saliency curriculum order..."
        python scripts/compute_curriculum.py \
            --pca_path "$SAL_PCA_DATA" \
            --output_path "$SAL_ORDER" --top_r 50
    fi

    GRAD_ORDER="${CURRICULUM_DIR}/gradient_order.pt"
    if [ -f "$GRAD_ORDER" ]; then
        echo "[PREREQ] Gradient curriculum order exists: $GRAD_ORDER"
    else
        echo "[PREREQ] Computing gradient curriculum order..."
        python scripts/compute_curriculum.py \
            --pca_path "$GRAD_PROFILE" \
            --output_path "$GRAD_ORDER" --top_r 50 --is_gradient
    fi

    echo "[PREREQ] All prerequisites ready."
}

# =============================================================================
# Stage 2: Training
# =============================================================================
run_train() {
    echo ""
    echo "===== Stage 2: Training ====="

    SALIENCY_PATH="${DATA_DIR}/teacher_saliency_${DATASET}.pt"
    SAL_ORDER="${CURRICULUM_DIR}/saliency_order.pt"
    GRAD_ORDER="${CURRICULUM_DIR}/gradient_order.pt"

    COMMON_ARGS="--dataset $DATASET --student_model $STUDENT \
        --epochs $EPOCHS --lr $LR --batch_size $BATCH_SIZE \
        --max_seq_len $MAX_SEQ_LEN --skip_eval --device $DEVICE"

    SAGD_ARGS="--teacher_saliency_path $SALIENCY_PATH \
        --lambda_noise $LAMBDA_NOISE --noise_sigma $NOISE_SIGMA \
        --sagd_every_n_steps $SAGD_N --sagd_tau_w $SAGD_TAU_W"

    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "--- Seed $SEED ---"

        # A. SaGD (no curriculum, baseline for our method)
        CKPT="${OUTPUT_BASE}/sagd_random/seed_${SEED}/student_final.pt"
        if [ -f "$CKPT" ]; then
            echo "[TRAIN] SKIP sagd_random/seed_${SEED} (exists)"
        else
            echo "[TRAIN] >>> sagd_random/seed_${SEED}"
            python scripts/train.py \
                --method sagd $COMMON_ARGS $SAGD_ARGS \
                --seed $SEED \
                --output_dir "${OUTPUT_BASE}/sagd_random/" \
                2>&1 | tee "${LOG_DIR}/train_sagd_random_s${SEED}.log"
        fi

        # B. SaGD + saliency PC curriculum
        CKPT="${OUTPUT_BASE}/sagd_sal_curriculum/seed_${SEED}/student_final.pt"
        if [ -f "$CKPT" ]; then
            echo "[TRAIN] SKIP sagd_sal_curriculum/seed_${SEED} (exists)"
        else
            echo "[TRAIN] >>> sagd_sal_curriculum/seed_${SEED}"
            python scripts/train.py \
                --method sagd $COMMON_ARGS $SAGD_ARGS \
                --curriculum_path "$SAL_ORDER" \
                --curriculum_schedule "$CURRICULUM_SCHEDULE" \
                --seed $SEED \
                --output_dir "${OUTPUT_BASE}/sagd_sal_curriculum/" \
                2>&1 | tee "${LOG_DIR}/train_sagd_sal_cur_s${SEED}.log"
        fi

        # C. SaGD + gradient PC curriculum
        CKPT="${OUTPUT_BASE}/sagd_grad_curriculum/seed_${SEED}/student_final.pt"
        if [ -f "$CKPT" ]; then
            echo "[TRAIN] SKIP sagd_grad_curriculum/seed_${SEED} (exists)"
        else
            echo "[TRAIN] >>> sagd_grad_curriculum/seed_${SEED}"
            python scripts/train.py \
                --method sagd $COMMON_ARGS $SAGD_ARGS \
                --curriculum_path "$GRAD_ORDER" \
                --curriculum_schedule "$CURRICULUM_SCHEDULE" \
                --seed $SEED \
                --output_dir "${OUTPUT_BASE}/sagd_grad_curriculum/" \
                2>&1 | tee "${LOG_DIR}/train_sagd_grad_cur_s${SEED}.log"
        fi

        # D. Standard KD + gradient PC curriculum (test curriculum alone)
        CKPT="${OUTPUT_BASE}/kd_grad_curriculum/seed_${SEED}/student_final.pt"
        if [ -f "$CKPT" ]; then
            echo "[TRAIN] SKIP kd_grad_curriculum/seed_${SEED} (exists)"
        else
            echo "[TRAIN] >>> kd_grad_curriculum/seed_${SEED}"
            python scripts/train.py \
                --method standard_kd $COMMON_ARGS \
                --curriculum_path "$GRAD_ORDER" \
                --curriculum_schedule "$CURRICULUM_SCHEDULE" \
                --seed $SEED \
                --output_dir "${OUTPUT_BASE}/kd_grad_curriculum/" \
                2>&1 | tee "${LOG_DIR}/train_kd_grad_cur_s${SEED}.log"
        fi

        # E. SaGD ablation: noise-only (τ_w=100 ≈ uniform)
        CKPT="${OUTPUT_BASE}/sagd_noise_only/seed_${SEED}/student_final.pt"
        if [ -f "$CKPT" ]; then
            echo "[TRAIN] SKIP sagd_noise_only/seed_${SEED} (exists)"
        else
            echo "[TRAIN] >>> sagd_noise_only/seed_${SEED}"
            python scripts/train.py \
                --method sagd $COMMON_ARGS \
                --teacher_saliency_path "$SALIENCY_PATH" \
                --lambda_noise $LAMBDA_NOISE --noise_sigma $NOISE_SIGMA \
                --sagd_every_n_steps $SAGD_N --sagd_tau_w 100.0 \
                --seed $SEED \
                --output_dir "${OUTPUT_BASE}/sagd_noise_only/" \
                2>&1 | tee "${LOG_DIR}/train_sagd_noise_only_s${SEED}.log"
        fi

        # F. SaGD ablation: reweight-only (λ=0)
        CKPT="${OUTPUT_BASE}/sagd_reweight_only/seed_${SEED}/student_final.pt"
        if [ -f "$CKPT" ]; then
            echo "[TRAIN] SKIP sagd_reweight_only/seed_${SEED} (exists)"
        else
            echo "[TRAIN] >>> sagd_reweight_only/seed_${SEED}"
            python scripts/train.py \
                --method sagd $COMMON_ARGS \
                --teacher_saliency_path "$SALIENCY_PATH" \
                --lambda_noise 0.0 --noise_sigma $NOISE_SIGMA \
                --sagd_every_n_steps $SAGD_N --sagd_tau_w $SAGD_TAU_W \
                --seed $SEED \
                --output_dir "${OUTPUT_BASE}/sagd_reweight_only/" \
                2>&1 | tee "${LOG_DIR}/train_sagd_reweight_only_s${SEED}.log"
        fi
    done

    echo "[TRAIN] All training done."
}

# =============================================================================
# Stage 3: Evaluation
# =============================================================================
run_eval() {
    echo ""
    echo "===== Stage 3: Evaluation ====="

    METHODS_TO_EVAL=(
        sagd_random
        sagd_sal_curriculum
        sagd_grad_curriculum
        kd_grad_curriculum
        sagd_noise_only
        sagd_reweight_only
    )

    for METHOD in "${METHODS_TO_EVAL[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            CKPT="${OUTPUT_BASE}/${METHOD}/seed_${SEED}/student_final.pt"
            EVAL_OUT="${OUTPUT_BASE}/${METHOD}/seed_${SEED}/benchmark_rouge.json"

            if [ ! -f "$CKPT" ]; then
                echo "[EVAL] SKIP ${METHOD}/seed_${SEED} (no checkpoint)"
                continue
            fi
            if [ -f "$EVAL_OUT" ]; then
                echo "[EVAL] SKIP ${METHOD}/seed_${SEED} (eval exists)"
                continue
            fi

            echo "[EVAL] >>> ${METHOD}/seed_${SEED}"
            python scripts/evaluate_benchmarks.py \
                --student_model "$STUDENT" \
                --student_ckpt "$CKPT" \
                --output_path "$EVAL_OUT" \
                --device "$DEVICE" \
                2>&1 | tee "${LOG_DIR}/eval_${METHOD}_s${SEED}.log"
        done
    done

    echo "[EVAL] All evaluation done."
    echo ""

    # Summarize results
    echo "===== Results Summary ====="
    python -c "
import json, numpy as np, os

methods = ['sagd_random', 'sagd_sal_curriculum', 'sagd_grad_curriculum',
           'kd_grad_curriculum', 'sagd_noise_only', 'sagd_reweight_only']
seeds = [42, 123, 456]
base = '${OUTPUT_BASE}'

header = f\"{'Method':<25} | {'DollyEval':>10} | {'S-NatInst':>10} | {'Unnatural':>10} | {'Avg':>10}\"
print(header)
print('-' * len(header))

for method in methods:
    benchmarks = {'dolly_eval': [], 'super_natural': [], 'unnatural': []}
    for seed in seeds:
        path = os.path.join(base, method, f'seed_{seed}', 'benchmark_rouge.json')
        try:
            with open(path) as f:
                data = json.load(f)
            for b in benchmarks:
                if b in data:
                    benchmarks[b].append(data[b].get('rouge_l_f', 0))
        except:
            pass
    def fmt(lst):
        if not lst: return '—'
        return f'{np.mean(lst):.2f}±{np.std(lst):.2f}'
    avgs = [np.mean(v) for v in benchmarks.values() if v]
    avg_str = f'{np.mean(avgs):.2f}' if avgs else '—'
    print(f\"{method:<25} | {fmt(benchmarks['dolly_eval']):>10} | {fmt(benchmarks['super_natural']):>10} | {fmt(benchmarks['unnatural']):>10} | {avg_str:>10}\")
"
}

# =============================================================================
# Main
# =============================================================================
case "$STAGE" in
    prereq) run_prereq ;;
    train)  run_train ;;
    eval)   run_eval ;;
    all)    run_prereq && run_train && run_eval ;;
    *)      echo "Usage: $0 {all|prereq|train|eval}" ; exit 1 ;;
esac
