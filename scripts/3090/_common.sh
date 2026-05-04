# shellcheck shell=bash
# Shared config for 3090 LLaMA training scripts.
# Source from each per-GPU script:
#     source "$(dirname "$0")/_common.sh"

# Defragment-friendly allocator — recovers 1-3 GB of "reserved but unallocated"
# memory that would otherwise cause OOM at backward time on tight 24GB GPUs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ----- Models -----
TEACHER="meta-llama/Llama-3.1-8B-Instruct"
STUDENT="meta-llama/Llama-3.2-1B-Instruct"

# ----- Pre-computed artifacts (rsync these from A100 before running) -----
SAL="data/teacher_saliency_dolly_llama.pt"
GRAD_ORDER="outputs_ours/curriculum/llama_1B/dolly/gradient_order.pt"
POCL_ORDER="outputs_ours/curriculum/llama_1B/dolly/pocl_order.pt"

# ----- Output bases -----
BASE_DOLLY="outputs_dolly/llama_1B"
BASE_OURS="outputs_ours/llama_1B/dolly"
BASE_TASKS="outputs_task/llama_1B"

# ----- 3090 24GB constraints -----
# load_8bit_teacher    : bitsandbytes int8 → 8B teacher fits
# gradient_checkpointing: ~50% activation memory savings on student
# batch=1, accum=32    : effective batch 32 (matches A100 baseline)
THREE090_FLAGS="--load_4bit_teacher --gradient_checkpointing --use_8bit_optimizer --bf16 \
    --batch_size 1 --gradient_accumulation 32"

COMMON_DOLLY="--dataset dolly --student_model $STUDENT --teacher_model $TEACHER \
    --epochs 10 --lr 1e-5 --max_seq_len 192 \
    --skip_eval --device cuda:0 $THREE090_FLAGS"

COMMON_SAMSUM="--dataset samsum --student_model $STUDENT --teacher_model $TEACHER \
    --epochs 10 --lr 1e-5 --max_seq_len 192 \
    --skip_eval --device cuda:0 $THREE090_FLAGS"

COMMON_GSM8K="--dataset gsm8k --student_model $STUDENT --teacher_model $TEACHER \
    --epochs 10 --lr 1e-5 --max_seq_len 192 \
    --skip_eval --device cuda:0 $THREE090_FLAGS"

SAGD_ARGS="--teacher_saliency_path $SAL \
    --lambda_noise 0.5 --noise_sigma 0.005 \
    --sagd_every_n_steps 5 --sagd_tau_w 1.0"

# Skip-if-exists eval helper.
# Usage: eval_ckpt <BASE> <method> <seed>
eval_ckpt () {
    local _base="$1" _method="$2" _seed="$3"
    local _ckpt="${_base}/${_method}/seed_${_seed}/student_final.pt"
    local _eval="${_base}/${_method}/seed_${_seed}/benchmark_rouge.json"
    [ ! -f "$_ckpt" ] && return 0
    [ -f "$_eval" ] && return 0
    echo "EVAL >>> ${_method}/seed_${_seed}"
    python scripts/evaluate_benchmarks.py \
        --student_model "$STUDENT" \
        --student_ckpt "$_ckpt" \
        --output_path "$_eval" \
        --device cuda:0
}
