#!/bin/bash
# Parallel eval — N workers on a single GPU, sharded round-robin.
# Designed for A100 80GB which can fit ~6 concurrent Qwen3-0.6B eval procs
# or ~3 concurrent Qwen3-1.7B eval procs.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/a100_parallel_eval.sh [N_WORKERS] [FILTER]
#
# FILTER values:
#   all              — all Qwen + LLaMA, baselines + curriculum + extension
#   qwen3_0.6B       — only 0.6B baselines
#   qwen3_1.7B       — only 1.7B baselines
#   qwen_curriculum  — only Qwen curriculum / loss_ablation / lambda_sweep
#   llama_1B         — LLaMA-3.2-1B baselines + curriculum + task-specific
#
# Examples:
#   # GPU 1, 6 workers, all Qwen 0.6B (best for one A100 80GB)
#   CUDA_VISIBLE_DEVICES=1 bash scripts/a100_parallel_eval.sh 6 qwen3_0.6B
#
#   # GPU 1, 3 workers, Qwen 1.7B (1.7B is ~3x larger so fewer workers)
#   CUDA_VISIBLE_DEVICES=1 bash scripts/a100_parallel_eval.sh 3 qwen3_1.7B
#
#   # GPU 1, 4 workers, everything
#   CUDA_VISIBLE_DEVICES=1 bash scripts/a100_parallel_eval.sh 4 all
#
# Memory budget per worker:
#   Qwen3-0.6B  ~4 GB   -> 6 workers fit on a 50GB-free A100
#   Qwen3-1.7B  ~10 GB  -> 3-4 workers
#   LLaMA-3.2-1B ~6 GB  -> 5-6 workers
# Each worker eval task = one ckpt × 5 benchmarks ≈ 5-15 min.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

N_WORKERS="${1:-4}"
FILTER="${2:-all}"
mkdir -p logs

# ---------- Task discovery ----------
declare -a TASKS

add_task () {
    local _model="$1" _ckpt="$2" _eval="$3"
    [ ! -f "$_ckpt" ] && return 0
    [ -f "$_eval" ] && return 0
    TASKS+=("${_model}|${_ckpt}|${_eval}")
}

scan_baselines () {
    # outputs_dolly/<student>/<method>/seed_X/student_final.pt
    local _base="$1" _model="$2"
    [ ! -d "$_base" ] && return 0
    for METHOD_DIR in "$_base"/*/; do
        for SEED_DIR in "$METHOD_DIR"seed_*/; do
            add_task "$_model" \
                "${SEED_DIR}student_final.pt" \
                "${SEED_DIR}benchmark_rouge.json"
        done
    done
}

scan_nested () {
    # outputs_ours/<student>/<dataset>/<config>/<method>/seed_X/student_final.pt
    # also handles outputs_task/<student>/<dataset>/<method>/seed_X/...
    local _base="$1" _model="$2"
    [ ! -d "$_base" ] && return 0
    for CFG_DIR in "$_base"/*/; do
        for METHOD_DIR in "$CFG_DIR"*/; do
            # If METHOD_DIR contains seed_X dirs, treat as 2-level (CFG/method/seed)
            for SEED_DIR in "$METHOD_DIR"seed_*/; do
                add_task "$_model" \
                    "${SEED_DIR}student_final.pt" \
                    "${SEED_DIR}benchmark_rouge.json"
            done
        done
    done
}

case "$FILTER" in
    all|qwen3_0.6B)
        scan_baselines "outputs_dolly/qwen3_0.6B" "Qwen/Qwen3-0.6B" ;;
esac
case "$FILTER" in
    all|qwen3_1.7B)
        scan_baselines "outputs_dolly/qwen3_1.7B" "Qwen/Qwen3-1.7B" ;;
esac
case "$FILTER" in
    all|qwen_curriculum)
        scan_nested "outputs_ours/qwen3_0.6B/dolly" "Qwen/Qwen3-0.6B"
        scan_nested "outputs_ours/qwen3_1.7B/dolly" "Qwen/Qwen3-1.7B" ;;
esac
case "$FILTER" in
    all|llama_1B)
        scan_baselines "outputs_dolly/llama_1B" "meta-llama/Llama-3.2-1B-Instruct"
        scan_nested "outputs_ours/llama_1B/dolly" "meta-llama/Llama-3.2-1B-Instruct"
        scan_nested "outputs_task/llama_1B" "meta-llama/Llama-3.2-1B-Instruct" ;;
esac

N_TASKS=${#TASKS[@]}
echo "===== a100_parallel_eval ====="
echo "Filter:  $FILTER"
echo "Workers: $N_WORKERS"
echo "Found $N_TASKS ckpts needing eval"
echo ""

if [ "$N_TASKS" -eq 0 ]; then
    echo "Nothing to do — all ckpts already have benchmark_rouge.json."
    exit 0
fi

echo "First 5 tasks:"
for ((i=0; i < 5 && i < N_TASKS; i++)); do
    EVAL_PATH="${TASKS[i]##*|}"
    echo "  $((i+1)). ${EVAL_PATH}"
done
[ "$N_TASKS" -gt 5 ] && echo "  ... ($((N_TASKS - 5)) more)"
echo ""

# ---------- Launch workers ----------
PIDS=()
for ((W=0; W < N_WORKERS; W++)); do
    (
        for ((I=W; I < N_TASKS; I+=N_WORKERS)); do
            IFS='|' read -r MODEL CKPT EVAL <<< "${TASKS[I]}"
            POS=$((I+1))
            echo "[W${W}] [${POS}/${N_TASKS}] ${EVAL}"
            python scripts/evaluate_benchmarks.py \
                --student_model "$MODEL" \
                --student_ckpt "$CKPT" \
                --output_path "$EVAL" \
                --device cuda:0 \
                || echo "[W${W}] FAILED ${EVAL}"
        done
        echo "[W${W}] done"
    ) > "logs/parallel_eval_W${W}.log" 2>&1 &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} workers (PIDs: ${PIDS[*]})"
echo ""
echo "Monitor commands:"
echo "  tail -f logs/parallel_eval_W*.log"
echo "  watch -n 30 'ls logs/parallel_eval_W*.log | xargs -I {} sh -c \"echo === {} ===; tail -2 {}\"'"
echo "  watch -n 30 'find outputs_dolly outputs_ours outputs_task -name benchmark_rouge.json | wc -l'"
echo ""

wait
echo ""
echo "===== All workers done ====="
DONE=$(find outputs_dolly outputs_ours outputs_task \
    -name "benchmark_rouge.json" 2>/dev/null | wc -l)
echo "Total benchmark_rouge.json files now: $DONE"
