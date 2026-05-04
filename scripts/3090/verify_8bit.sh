#!/bin/bash
# Verify 8-bit teacher fidelity vs the existing fp16 reference.
# Trains standard_kd on Dolly-15K with 8-bit teacher (3090 mode), then
# compares ROUGE-L to fp16 baseline at outputs_dolly/llama_1B/standard_kd/seed_42/
#
# Decision rule (judged on average ROUGE-L delta):
#   |Δ| < 0.5  → PASS: run all LLaMA on 3090
#   0.5 ≤ |Δ| < 1.0 → MARGINAL: 3090 for ablations only, A100 for main
#   |Δ| ≥ 1.0  → FAIL: train all LLaMA on A100 GPU 2/3
#
# Usage:
#     CUDA_VISIBLE_DEVICES=0 bash scripts/3090/verify_8bit.sh
#     FAST=1 CUDA_VISIBLE_DEVICES=0 bash scripts/3090/verify_8bit.sh   # 3 epochs
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
source "$(dirname "$0")/_common.sh"

EPOCHS=${EPOCHS:-10}
[ "${FAST:-0}" = "1" ] && EPOCHS=3

OUT="outputs_verify/llama_8bit/"
SEED=42
CKPT="${OUT}standard_kd/seed_${SEED}/student_final.pt"
EVAL="${OUT}standard_kd/seed_${SEED}/benchmark_rouge.json"

if [ ! -f "$CKPT" ]; then
    echo ">>> training standard_kd with 8-bit teacher (epochs=$EPOCHS)"
    python scripts/train.py \
        --method standard_kd --dataset dolly \
        --teacher_model "$TEACHER" --student_model "$STUDENT" \
        --load_8bit_teacher --gradient_checkpointing --use_8bit_optimizer \
        --epochs "$EPOCHS" --batch_size 1 --gradient_accumulation 32 \
        --lr 1e-5 --max_seq_len 384 --skip_eval \
        --seed "$SEED" --output_dir "$OUT" --device cuda:0
fi

if [ ! -f "$EVAL" ]; then
    echo ">>> eval"
    python scripts/evaluate_benchmarks.py \
        --student_model "$STUDENT" --student_ckpt "$CKPT" \
        --output_path "$EVAL" --device cuda:0
fi

echo ""
echo "===== Verification result ====="
python - <<PY
import json, glob
me = "$EVAL"
ref_path = "outputs_dolly/llama_1B/standard_kd/seed_42/benchmark_rouge.json"
import os
if not os.path.exists(ref_path):
    print(f"[WARN] no fp16 reference at {ref_path}")
    print("       rsync the A100 result over before judging fidelity")
    raise SystemExit(0)
me_d = json.load(open(me)); ref_d = json.load(open(ref_path))

print(f"  8-bit:  {me}")
print(f"  fp16:   {ref_path}\n")
print(f"{'benchmark':<16} {'8-bit':>8} {'fp16':>8} {'Delta':>8}")
print("-" * 44)
for k in sorted(set(me_d) | set(ref_d)):
    if k == "average_rouge_l": continue
    if not (isinstance(me_d.get(k), dict) and isinstance(ref_d.get(k), dict)): continue
    a = me_d[k]["rouge_l_f"] * 100
    b = ref_d[k]["rouge_l_f"] * 100
    print(f"{k:<16} {a:>8.2f} {b:>8.2f} {a-b:>+8.2f}")
a_avg = me_d.get("average_rouge_l", 0) * 100
b_avg = ref_d.get("average_rouge_l", 0) * 100
print("-" * 44)
print(f"{'AVERAGE':<16} {a_avg:>8.2f} {b_avg:>8.2f} {a_avg-b_avg:>+8.2f}")

dlt = abs(a_avg - b_avg)
if dlt < 0.5:
    verdict = "PASS  - run all LLaMA on 3090"
elif dlt < 1.0:
    verdict = "MARGINAL - 3090 only for ablations"
else:
    verdict = "FAIL  - fall back to A100 for LLaMA"
print(f"\nVerdict: |Delta|={dlt:.2f} ROUGE-L -> {verdict}")
PY
