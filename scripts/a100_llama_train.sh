#!/bin/bash
# A100 LLaMA full training — lock-based parallel worker.
#
# Trains the complete LLaMA-3.1-8B -> 3.2-1B experiment matrix on A100:
#   - 8 baselines (sft, standard_kd, reverse_kl, seqkd, gkd, distillm, dakd, sagd)
#     × 3 seeds = 24 runs
#   - 3 ablation configs (noise_only, reweight_only, no_grad_curr)
#     × 3 seeds = 9 runs
#   - 4 curriculum configs (sagd_grad, kd_grad, kd_pocl, sagd_pocl)
#     × 3 seeds = 12 runs
# Total: 45 runs. ~25 min/run on A100 -> ~6h with 3 parallel workers.
#
# Concurrency: each task acquires a directory-based lock (mkdir is atomic) so
# multiple concurrent invocations on different GPUs (or same GPU) won't collide.
# Skip-if-exists: tasks whose ckpt already exists are skipped.
# Skip-if-no-prereq: curriculum tasks skip silently when their order file
# isn't ready yet (so workers can launch before prereq finishes).
#
# Usage (concurrent on multiple GPUs/slots):
#     CUDA_VISIBLE_DEVICES=1 tmux new -d -s g1 \
#         'bash scripts/a100_llama_train.sh 2>&1 | tee logs/llama_train_g1.log'
#     CUDA_VISIBLE_DEVICES=3 tmux new -d -s g3a \
#         'bash scripts/a100_llama_train.sh 2>&1 | tee logs/llama_train_g3a.log'
#     CUDA_VISIBLE_DEVICES=3 tmux new -d -s g3b \
#         'bash scripts/a100_llama_train.sh 2>&1 | tee logs/llama_train_g3b.log'
#
# Prereq: scripts/llama_prereq.sh must produce
#     data/teacher_saliency_dolly_llama.pt
#     outputs_ours/curriculum/llama_1B/dolly/{gradient_order,pocl_order}.pt
# Workers will skip prereq-dependent tasks until those files appear, so the
# script is safe to launch in parallel with the prereq.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
mkdir -p logs

TEACHER="meta-llama/Llama-3.1-8B-Instruct"
STUDENT="meta-llama/Llama-3.2-1B-Instruct"
SAL="data/teacher_saliency_dolly_llama.pt"
GRAD_ORDER="outputs_ours/curriculum/llama_1B/dolly/gradient_order.pt"
POCL_ORDER="outputs_ours/curriculum/llama_1B/dolly/pocl_order.pt"

# A100 fp16 teacher fits easily; 8 bs * 4 grad-accum = 32 effective batch.
COMMON="--dataset dolly --student_model $STUDENT --teacher_model $TEACHER \
    --epochs 10 --batch_size 8 --gradient_accumulation 4 \
    --lr 1e-5 --max_seq_len 512 --skip_eval --device cuda:0"

SAGD_ARGS="--teacher_saliency_path $SAL \
    --lambda_noise 0.5 --noise_sigma 0.005 \
    --sagd_every_n_steps 5 --sagd_tau_w 1.0"

# ---------- Build task list ----------
declare -a TASKS

SEEDS_BL="42 123 456"
for SEED in $SEEDS_BL; do
    # Baselines (7 non-SaGD methods)
    for METHOD in sft standard_kd reverse_kl seqkd gkd distillm dakd; do
        TASKS+=("baseline|${METHOD}|${SEED}|outputs_dolly/llama_1B/")
    done
    # SaGD main entry (= full SaGD with grad curriculum)
    TASKS+=("sagd_full|sagd|${SEED}|outputs_dolly/llama_1B/")

    # Loss-level ablations (no curriculum)
    TASKS+=("noise_only|sagd|${SEED}|outputs_ours/llama_1B/dolly/sagd_noise_only/")
    TASKS+=("reweight_only|sagd|${SEED}|outputs_ours/llama_1B/dolly/sagd_reweight_only/")
    TASKS+=("no_grad_curr|sagd|${SEED}|outputs_ours/llama_1B/dolly/sagd_no_grad_curr/")

    # Curriculum
    TASKS+=("sagd_grad|sagd|${SEED}|outputs_ours/llama_1B/dolly/sagd_grad_curriculum/")
    TASKS+=("kd_grad|standard_kd|${SEED}|outputs_ours/llama_1B/dolly/kd_grad_curriculum/")
    TASKS+=("kd_pocl|standard_kd|${SEED}|outputs_ours/llama_1B/dolly/kd_pocl/")
    TASKS+=("sagd_pocl|sagd|${SEED}|outputs_ours/llama_1B/dolly/sagd_pocl/")
done

N=${#TASKS[@]}
echo "===== a100_llama_train ====="
echo "Worker PID: $$"
echo "GPU:        ${CUDA_VISIBLE_DEVICES:-(unset)}"
echo "Task slots: $N"
echo ""

# ---------- Process tasks ----------
DONE=0; SKIPPED=0; LOCKED=0
for ((I=0; I<N; I++)); do
    IFS='|' read -r KIND METHOD SEED OUT <<< "${TASKS[I]}"
    SAVE_DIR="${OUT}${METHOD}/seed_${SEED}"
    CKPT="${SAVE_DIR}/student_final.pt"
    LOCK="${SAVE_DIR}/.lock"

    if [ -f "$CKPT" ]; then
        DONE=$((DONE+1))
        continue
    fi

    mkdir -p "$SAVE_DIR"
    if ! mkdir "$LOCK" 2>/dev/null; then
        LOCKED=$((LOCKED+1))
        continue
    fi

    trap "rmdir '$LOCK' 2>/dev/null || true" EXIT INT TERM

    case "$KIND" in
        sagd_full|noise_only|reweight_only|no_grad_curr|sagd_grad|sagd_pocl)
            if [ ! -f "$SAL" ]; then
                echo "[$(date +%H:%M)] [skip prereq] $KIND $METHOD seed=$SEED  (saliency not ready)"
                rmdir "$LOCK" 2>/dev/null || true
                SKIPPED=$((SKIPPED+1))
                continue
            fi ;;
    esac
    case "$KIND" in
        sagd_full|sagd_grad|kd_grad)
            if [ ! -f "$GRAD_ORDER" ]; then
                echo "[$(date +%H:%M)] [skip prereq] $KIND $METHOD seed=$SEED  (gradient_order not ready)"
                rmdir "$LOCK" 2>/dev/null || true
                SKIPPED=$((SKIPPED+1))
                continue
            fi ;;
        kd_pocl|sagd_pocl)
            if [ ! -f "$POCL_ORDER" ]; then
                echo "[$(date +%H:%M)] [skip prereq] $KIND $METHOD seed=$SEED  (pocl_order not ready)"
                rmdir "$LOCK" 2>/dev/null || true
                SKIPPED=$((SKIPPED+1))
                continue
            fi ;;
    esac

    echo "[$(date +%H:%M)] [$((I+1))/$N] start: $KIND $METHOD seed=$SEED -> $OUT"
    START=$(date +%s)

    case "$KIND" in
        baseline)
            EXTRA=""
            [ "$METHOD" = "gkd" ]      && EXTRA="--gkd_beta 0.5"
            [ "$METHOD" = "distillm" ] && EXTRA="--distillm_alpha 0.5"
            [ "$METHOD" = "dakd" ]     && EXTRA="--bdl_lambda 0.9"
            python scripts/train.py --method "$METHOD" $COMMON $EXTRA \
                --seed "$SEED" --output_dir "$OUT" || true
            ;;
        sagd_full)
            python scripts/train.py --method sagd $COMMON $SAGD_ARGS \
                --curriculum_path "$GRAD_ORDER" \
                --seed "$SEED" --output_dir "$OUT" || true
            ;;
        noise_only)
            python scripts/train.py --method sagd $COMMON \
                --teacher_saliency_path "$SAL" \
                --lambda_noise 0.5 --noise_sigma 0.005 \
                --sagd_every_n_steps 5 --sagd_tau_w 100 \
                --seed "$SEED" --output_dir "$OUT" || true
            ;;
        reweight_only)
            python scripts/train.py --method sagd $COMMON \
                --teacher_saliency_path "$SAL" \
                --lambda_noise 0 --noise_sigma 0.005 \
                --sagd_every_n_steps 5 --sagd_tau_w 1.0 \
                --seed "$SEED" --output_dir "$OUT" || true
            ;;
        no_grad_curr)
            python scripts/train.py --method sagd $COMMON $SAGD_ARGS \
                --seed "$SEED" --output_dir "$OUT" || true
            ;;
        sagd_grad)
            python scripts/train.py --method sagd $COMMON $SAGD_ARGS \
                --curriculum_path "$GRAD_ORDER" \
                --seed "$SEED" --output_dir "$OUT" || true
            ;;
        kd_grad)
            python scripts/train.py --method standard_kd $COMMON \
                --curriculum_path "$GRAD_ORDER" \
                --seed "$SEED" --output_dir "$OUT" || true
            ;;
        kd_pocl)
            python scripts/train.py --method standard_kd $COMMON \
                --curriculum_path "$POCL_ORDER" \
                --seed "$SEED" --output_dir "$OUT" || true
            ;;
        sagd_pocl)
            python scripts/train.py --method sagd $COMMON $SAGD_ARGS \
                --curriculum_path "$POCL_ORDER" \
                --seed "$SEED" --output_dir "$OUT" || true
            ;;
    esac

    ELAPSED=$(( $(date +%s) - START ))
    if [ -f "$CKPT" ]; then
        echo "[$(date +%H:%M)] [$((I+1))/$N] done in $((ELAPSED/60))m: $KIND $METHOD seed=$SEED"
    else
        echo "[$(date +%H:%M)] [$((I+1))/$N] FAILED ($((ELAPSED/60))m): $KIND $METHOD seed=$SEED"
    fi

    rmdir "$LOCK" 2>/dev/null || true
done

echo ""
echo "===== Worker done ====="
echo "  Completed (already-existing or just-trained): $DONE"
echo "  Skipped (prereq missing):                     $SKIPPED"
echo "  Locked-out (other worker handling):           $LOCKED"
echo ""
echo "If 'Skipped' > 0, wait for prereq to finish and re-launch this script."
