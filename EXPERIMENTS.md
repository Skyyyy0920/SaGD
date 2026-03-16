# SaGD 实验指南

本文档对应论文 §4 的全部实验。按顺序执行，后续实验依赖前序实验的输出。

---

## 总览

```
Phase 0   预计算 teacher saliency               1 GPU    ~1h
Phase 1   Exp 1: Saliency 诊断（动机实验）       1 GPU    ~1h     → §4.2
Phase 2   Exp 2: 主实验表（3方法 × 3种子）        4 GPU    ~6h     → §4.3
Phase 3   Exp 3: 消融实验（~15 runs）             4 GPU    ~8h     → §4.4
Phase 4   Exp 4: 训练动态曲线                     1 GPU    ~2h     → §4.5
Phase 5   Exp 5: 错误样本分析                     1 GPU    ~1h     → §4.6
Phase 6   Exp 6: 跨架构泛化（LLaMA）             1 GPU    ~4h     → §4.7
Phase 7   Exp 7: Benchmark 防御（MMLU 等）        1 GPU    ~2h     → Appendix
```

**硬件**: 4× A100 80GB（Phase 2-3 可并行；其他 phase 单卡即可）
**固定超参**: epochs=3, batch_size=8, grad_accum=4, lr=2e-5, max_seq_len=512, T=2.0, fp16=true
**种子**: 42, 123, 456

---

## Phase 0: 预计算 Teacher Saliency（运行一次）

**目的**: Teacher 是冻结的，saliency 只需算一次，缓存到磁盘，后续所有实验复用。

**关键**: 必须与训练使用完全相同的 data_source、seed、max_seq_len、tokenizer、subset，否则 index→saliency 映射会静默错乱。预计算和训练都使用 subset="train"（默认值）。

```bash
python scripts/precompute_teacher_saliency.py \
    --model_name Qwen/Qwen3-8B \
    --data_source "databricks/databricks-dolly-15k" \
    --output_path data/teacher_saliency.pt \
    --batch_size 4 \
    --max_seq_len 512 \
    --device cuda:0
```

**输出**: `data/teacher_saliency.pt`（约 200MB，包含 15k 样本的变长 saliency 向量）
**耗时**: ~1 小时（单卡 A100）
**验证**: 确认文件大小合理，加载后检查 `len(cache["saliency"]) == 15011`

---

## Phase 1: Exp 1 — Saliency Divergence 诊断（§4.2）

**论文问题**: "Standard KD 是否保留了 teacher 的 saliency 模式？"
**预期结论**: 不保留。Standard KD 训出的 student 与 teacher 存在显著的 saliency 偏差，这为 SaGD 提供了动机。

### Step 1.1: 训练 Standard KD baseline

```bash
# 三个种子
for SEED in 42 123 456; do
    python scripts/train.py \
        --method standard_kd \
        --seed $SEED \
        --output_dir outputs/ \
        --device cuda:0
done
```

**输出**: `outputs/standard_kd/seed_{42,123,456}/student_final.pt`

### Step 1.2: 对每个 checkpoint 做 saliency 诊断

```bash
for SEED in 42 123 456; do
    python scripts/diagnose_saliency.py \
        --student_ckpt outputs/standard_kd/seed_${SEED}/student_final.pt \
        --teacher_saliency_path data/teacher_saliency.pt \
        --output_path outputs/standard_kd/seed_${SEED}/saliency_diagnosis.json \
        --subset val \
        --max_samples 500 \
        --device cuda:0
done
```

**输出**: 每个种子一个 JSON，包含:
```json
{
    "mean_jsd": 0.xxx,        // 越高说明 saliency 偏差越大
    "std_jsd": 0.xxx,
    "median_jsd": 0.xxx,
    "top20_samples": [...],   // JSD 最大的 20 个样本
    "per_category_jsd": {...} // 按 Dolly 类别分解
}
```

### Step 1.3（可选）: 对未训练的原始 student 也做诊断，作为 "before KD" 的 baseline

```bash
# 需要先保存原始 student 的权重
python -c "
from sagd.models import load_student
import torch
student, _ = load_student('Qwen/Qwen3-0.6B', 'cpu')
torch.save(student.state_dict(), 'outputs/pretrained_student.pt')
"

python scripts/diagnose_saliency.py \
    --student_ckpt outputs/pretrained_student.pt \
    --teacher_saliency_path data/teacher_saliency.pt \
    --output_path outputs/pretrained_saliency_diagnosis.json \
    --subset val \
    --max_samples 500 \
    --device cuda:0
```

### 要报告的数据

| Model | Mean JSD ↓ | 解读 |
|-------|-----------|------|
| Pretrained student (no KD) | 较高 | 起始状态 |
| Standard KD student | 仍然较高 | 说明 standard KD 未对齐 saliency |

---

## Phase 2: Exp 2 — 主实验表（§4.3）

**论文问题**: SaGD vs baselines 的 ROUGE-L 和 Saliency Loyalty 比较。
**这是论文最核心的表格。**

### 三个方法 × 三个种子 = 9 runs

```bash
# === Standard KD (已在 Phase 1 训完，只需补充评测) ===
for SEED in 42 123 456; do
    python scripts/evaluate.py \
        --student_ckpt outputs/standard_kd/seed_${SEED}/student_final.pt \
        --output_path outputs/standard_kd/seed_${SEED}/eval_metrics.json \
        --device cuda:0
done

# === Reverse KL ===
for SEED in 42 123 456; do
    python scripts/train.py \
        --method reverse_kl \
        --seed $SEED \
        --output_dir outputs/ \
        --device cuda:0  # 可分配不同 GPU: cuda:1
done

# === SaGD (our method) ===
for SEED in 42 123 456; do
    python scripts/train.py \
        --method sagd \
        --teacher_saliency_path data/teacher_saliency.pt \
        --lambda_sal 0.5 \
        --sagd_every_n_steps 5 \
        --sagd_tau_w 1.0 \
        --seed $SEED \
        --output_dir outputs/ \
        --device cuda:0  # 可分配不同 GPU: cuda:2
done
```

### 评测每个 checkpoint

```bash
for METHOD in standard_kd reverse_kl sagd; do
    for SEED in 42 123 456; do
        # ROUGE-L (on test subset)
        python scripts/evaluate.py \
            --student_ckpt outputs/${METHOD}/seed_${SEED}/student_final.pt \
            --output_path outputs/${METHOD}/seed_${SEED}/eval_metrics.json \
            --subset test \
            --device cuda:0

        # Saliency Loyalty (Mean JSD, on val subset)
        python scripts/diagnose_saliency.py \
            --student_ckpt outputs/${METHOD}/seed_${SEED}/student_final.pt \
            --teacher_saliency_path data/teacher_saliency.pt \
            --output_path outputs/${METHOD}/seed_${SEED}/saliency_diagnosis.json \
            --subset val \
            --device cuda:0
    done
done
```

### 要报告的表格

| Method | ROUGE-L F1 (mean±std) | Mean JSD ↓ (mean±std) |
|--------|----------------------|----------------------|
| Standard KD | x.xx ± x.xx | x.xx ± x.xx |
| Reverse KL | x.xx ± x.xx | x.xx ± x.xx |
| **SaGD (ours)** | **x.xx ± x.xx** | **x.xx ± x.xx** |

三个种子取 mean 和 std。

---

## Phase 3: Exp 3 — 消融实验（§4.4）

**论文问题**: Saliency loss 和 reweighting 各自贡献多少？

### 消融配置表

| 配置名 | λ | τ_w | 效果 |
|--------|---|-----|------|
| `sagd` (full) | 0.5 | 1.0 | 完整方法 |
| `sagd_loss_only` | 0.5 | 100.0 | τ_w≈∞ → 均匀权重 → 只有 saliency loss |
| `sagd_reweight_only` | 0.0 | 1.0 | 无 saliency loss → 只有 reweighting |
| `standard_kd` | — | — | baseline（已有） |

### 运行消融

```bash
# --- Ablation: saliency loss only (λ=0.5, τ_w=100 → uniform weights) ---
for SEED in 42 123 456; do
    python scripts/train.py \
        --method sagd \
        --teacher_saliency_path data/teacher_saliency.pt \
        --lambda_sal 0.5 \
        --sagd_tau_w 100.0 \
        --sagd_every_n_steps 5 \
        --seed $SEED \
        --output_dir outputs_ablation/sagd_loss_only/ \
        --device cuda:0
done

# --- Ablation: reweight only (λ=0, τ_w=1.0 → no saliency loss) ---
for SEED in 42 123 456; do
    python scripts/train.py \
        --method sagd \
        --teacher_saliency_path data/teacher_saliency.pt \
        --lambda_sal 0.0 \
        --sagd_tau_w 1.0 \
        --sagd_every_n_steps 5 \
        --seed $SEED \
        --output_dir outputs_ablation/sagd_reweight_only/ \
        --device cuda:0
done
```

### 超参敏感性（Appendix）

```bash
# --- λ sweep ---
for LAMBDA in 0.01 0.1 0.5 1.0 2.0; do
    python scripts/train.py \
        --method sagd \
        --teacher_saliency_path data/teacher_saliency.pt \
        --lambda_sal $LAMBDA \
        --sagd_tau_w 1.0 \
        --seed 42 \
        --output_dir outputs_sweep/lambda_${LAMBDA}/ \
        --device cuda:0
done

# --- τ_w sweep ---
for TAU in 0.1 0.5 1.0 2.0 5.0; do
    python scripts/train.py \
        --method sagd \
        --teacher_saliency_path data/teacher_saliency.pt \
        --lambda_sal 0.5 \
        --sagd_tau_w $TAU \
        --seed 42 \
        --output_dir outputs_sweep/tau_${TAU}/ \
        --device cuda:0
done

# --- N (saliency update frequency) sweep ---
for N in 1 3 5 10 20; do
    python scripts/train.py \
        --method sagd \
        --teacher_saliency_path data/teacher_saliency.pt \
        --lambda_sal 0.5 \
        --sagd_tau_w 1.0 \
        --sagd_every_n_steps $N \
        --seed 42 \
        --output_dir outputs_sweep/every_n_${N}/ \
        --device cuda:0
done
```

### 评测所有消融/sweep checkpoints

```bash
# 消融
for CONFIG in sagd_loss_only sagd_reweight_only; do
    for SEED in 42 123 456; do
        python scripts/evaluate.py \
            --student_ckpt outputs_ablation/${CONFIG}/sagd/seed_${SEED}/student_final.pt \
            --output_path outputs_ablation/${CONFIG}/sagd/seed_${SEED}/eval_metrics.json \
            --subset test \
            --device cuda:0
    done
done

# sweep（只跑 seed=42，只看趋势）
for LAMBDA in 0.01 0.1 0.5 1.0 2.0; do
    python scripts/evaluate.py \
        --student_ckpt outputs_sweep/lambda_${LAMBDA}/sagd/seed_42/student_final.pt \
        --output_path outputs_sweep/lambda_${LAMBDA}/sagd/seed_42/eval_metrics.json \
        --subset test \
        --device cuda:0
done

# τ_w 和 N 的 sweep 同理
```

### 要报告的表格

**消融表 (§4.4)**:

| Config | KL (zero-order) | Sal loss (first-order) | Reweight | ROUGE-L ↑ | Mean JSD ↓ |
|--------|:---:|:---:|:---:|:---:|:---:|
| Standard KD | uniform | — | — | x.xx | x.xx |
| + Sal loss only | uniform | ✓ | — | x.xx | x.xx |
| + Reweight only | weighted | — | ✓ | x.xx | x.xx |
| **SaGD (full)** | weighted | ✓ | ✓ | **x.xx** | **x.xx** |

**超参敏感性图 (Appendix)**: λ vs ROUGE-L 折线图，τ_w vs ROUGE-L 折线图

---

## Phase 4: Exp 4 — 训练动态（§4.5）

**论文问题**: 训练过程中 saliency alignment 和 JSD 如何变化？

**数据来源**: 直接从 SaGD 训练的 `training_stats.jsonl` 中提取，无需额外训练。

```bash
# 从 Phase 2 的 SaGD run 中提取
cat outputs/sagd/seed_42/training_stats.jsonl | python -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    if 'sagd/sal_loss' in d:
        print(f\"{d['step']}\t{d['loss']:.4f}\t{d['sagd/sal_loss']:.4f}\t{d['sagd/mean_jsd']:.4f}\t{d['sagd/max_weight']:.2f}\")
" > outputs/sagd/seed_42/dynamics.tsv
```

### 需要额外记录的训练动态（需在每个 epoch 末尾做诊断）

如果想画出 "JSD vs training step" 的曲线，需要在训练中间保存 checkpoint 并逐一诊断：

```bash
# SaGD 训练时每个 epoch 都会保存 student_epoch{1,2,3}.pt
for EPOCH in 1 2 3; do
    python scripts/diagnose_saliency.py \
        --student_ckpt outputs/sagd/seed_42/student_epoch${EPOCH}.pt \
        --teacher_saliency_path data/teacher_saliency.pt \
        --output_path outputs/sagd/seed_42/saliency_epoch${EPOCH}.json \
        --subset val \
        --device cuda:0
done

# 对比: standard_kd 的同样 epoch checkpoints
for EPOCH in 1 2 3; do
    python scripts/diagnose_saliency.py \
        --student_ckpt outputs/standard_kd/seed_42/student_epoch${EPOCH}.pt \
        --teacher_saliency_path data/teacher_saliency.pt \
        --output_path outputs/standard_kd/seed_42/saliency_epoch${EPOCH}.json \
        --subset val \
        --device cuda:0
done
```

### 要报告的图

1. **Saliency loss vs step**: 从 `training_stats.jsonl` 提取 `sagd/sal_loss`
2. **Mean JSD vs epoch**: 对比 SaGD 和 Standard KD 的 JSD 随 epoch 下降曲线
3. **Max weight vs step**: 展示 reweighting 的动态（从 `sagd/max_weight`）

---

## Phase 5: Exp 5 — 错误样本分析（§4.6）

**论文问题**: "Student 在哪些样本上 saliency 偏差最大？有什么模式？"

**数据来源**: Phase 2 的 `saliency_diagnosis.json` 中的 `top20_samples` 和 `per_category_jsd`。

```bash
# 提取 standard_kd 和 sagd 的 per-category JSD 对比
python -c "
import json

for method in ['standard_kd', 'sagd']:
    with open(f'outputs/{method}/seed_42/saliency_diagnosis.json') as f:
        d = json.load(f)
    print(f'\n=== {method} ===')
    print(f'Mean JSD: {d[\"mean_jsd\"]:.4f}')
    print('Per-category JSD:')
    for cat, jsd in sorted(d['per_category_jsd'].items(), key=lambda x: -x[1]):
        print(f'  {cat}: {jsd:.4f}')
    print('Top-5 worst samples:')
    for s in d['top20_samples'][:5]:
        print(f'  idx={s[\"index\"]}, JSD={s[\"jsd\"]:.4f}: {s[\"instruction_preview\"][:80]}...')
"
```

### 要报告的数据

1. **Per-category JSD 柱状图**: Standard KD vs SaGD，按 Dolly 类别分解
2. **Case study**: 挑 2-3 个 top JSD 样本，可视化 teacher vs student saliency heatmap
3. **分析**: 哪些类别（如 creative_writing vs QA）saliency 偏差最大，SaGD 是否修复

---

## Phase 6: Exp 6 — 跨架构泛化（§4.7，可选）

**论文问题**: SaGD 是否泛化到不同的模型架构？
**模型对**: LLaMA 3.1-8B (teacher) → LLaMA 3.1-1B (student)

### Step 6.1: 预计算 LLaMA teacher saliency

```bash
python scripts/precompute_teacher_saliency.py \
    --model_name meta-llama/Llama-3.1-8B \
    --tokenizer_name meta-llama/Llama-3.1-1B \
    --output_path data/teacher_saliency_llama.pt \
    --batch_size 4 \
    --max_seq_len 512 \
    --device cuda:0
```

### Step 6.2: 训练

```bash
for METHOD in standard_kd sagd; do
    for SEED in 42 123 456; do
        EXTRA_ARGS=""
        if [ "$METHOD" == "sagd" ]; then
            EXTRA_ARGS="--teacher_saliency_path data/teacher_saliency_llama.pt \
                        --lambda_sal 0.5 --sagd_every_n_steps 5 --sagd_tau_w 1.0"
        fi

        python scripts/train.py \
            --method $METHOD \
            --teacher_model meta-llama/Llama-3.1-8B \
            --student_model meta-llama/Llama-3.1-1B \
            --seed $SEED \
            --output_dir outputs_llama/ \
            $EXTRA_ARGS \
            --device cuda:0
    done
done
```

### Step 6.3: 评测

```bash
for METHOD in standard_kd sagd; do
    for SEED in 42 123 456; do
        python scripts/evaluate.py \
            --student_model meta-llama/Llama-3.1-1B \
            --student_ckpt outputs_llama/${METHOD}/seed_${SEED}/student_final.pt \
            --output_path outputs_llama/${METHOD}/seed_${SEED}/eval_metrics.json \
            --subset test \
            --device cuda:0
    done
done
```

### 要报告的表格

| Architecture | Method | ROUGE-L ↑ | Mean JSD ↓ |
|-------------|--------|-----------|-----------|
| Qwen 8B→0.6B | Standard KD | x.xx | x.xx |
| Qwen 8B→0.6B | **SaGD** | **x.xx** | **x.xx** |
| LLaMA 8B→1B | Standard KD | x.xx | x.xx |
| LLaMA 8B→1B | **SaGD** | **x.xx** | **x.xx** |

---

## Phase 7: Exp 7 — Benchmark 防御（Appendix）

**目的**: 证明 SaGD 提升 ROUGE-L 的同时没有损害通用能力。
**工具**: `lm-eval-harness`

```bash
pip install lm-eval

# 评测 SaGD student
lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen3-0.6B,peft=outputs/sagd/seed_42/student_final.pt \
    --tasks mmlu,arc_challenge,truthfulqa_mc2 \
    --batch_size 8 \
    --output_path outputs/sagd/seed_42/benchmark/

# 评测 Standard KD student
lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen3-0.6B,peft=outputs/standard_kd/seed_42/student_final.pt \
    --tasks mmlu,arc_challenge,truthfulqa_mc2 \
    --batch_size 8 \
    --output_path outputs/standard_kd/seed_42/benchmark/

# 评测 base student (no distillation)
lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen3-0.6B \
    --tasks mmlu,arc_challenge,truthfulqa_mc2 \
    --batch_size 8 \
    --output_path outputs/base_student_benchmark/
```

> **注意**: 上面的 `peft=` 参数假设你保存的是完整 state_dict，实际使用时需要先加载 checkpoint 到模型再评测。更可靠的方式是写一个小脚本把 student checkpoint 加载后保存为完整的 HuggingFace 格式目录，然后用 `pretrained=path/to/saved_model` 来评测。

### 要报告的表格（Appendix）

| Model | MMLU | ARC-C | TruthfulQA |
|-------|------|-------|------------|
| Qwen3-0.6B (base) | x.xx | x.xx | x.xx |
| + Standard KD | x.xx | x.xx | x.xx |
| + **SaGD** | x.xx | x.xx | x.xx |

---

## 输出目录结构总览

```
outputs/
├── standard_kd/
│   ├── seed_42/
│   │   ├── config.json
│   │   ├── student_epoch{1,2,3}.pt
│   │   ├── student_final.pt
│   │   ├── training_stats.jsonl
│   │   ├── eval_metrics.json         ← ROUGE-L
│   │   ├── saliency_diagnosis.json   ← JSD
│   │   └── saliency_epoch{1,2,3}.json
│   ├── seed_123/ ...
│   └── seed_456/ ...
├── reverse_kl/ ...
├── sagd/ ...
│
outputs_ablation/
├── sagd_loss_only/sagd/seed_{42,123,456}/ ...
├── sagd_reweight_only/sagd/seed_{42,123,456}/ ...
│
outputs_sweep/
├── lambda_{0.01,0.1,0.5,1.0,2.0}/sagd/seed_42/ ...
├── tau_{0.1,0.5,1.0,2.0,5.0}/sagd/seed_42/ ...
├── every_n_{1,3,5,10,20}/sagd/seed_42/ ...
│
outputs_llama/
├── standard_kd/seed_{42,123,456}/ ...
├── sagd/seed_{42,123,456}/ ...
│
data/
├── teacher_saliency.pt        ← Qwen teacher
└── teacher_saliency_llama.pt  ← LLaMA teacher
```

---

## Smoke Test（正式跑之前先验证）

```bash
# 1. 单元测试
pytest tests/ -v

# 2. 200 样本 smoke test
python scripts/precompute_teacher_saliency.py \
    --max_samples 200 --output_path data/test_saliency.pt --device cuda:0

python scripts/train.py \
    --method standard_kd --epochs 1 --max_train_samples 200 \
    --device cuda:0 --skip_eval

python scripts/train.py \
    --method sagd --teacher_saliency_path data/test_saliency.pt \
    --epochs 1 --max_train_samples 200 \
    --device cuda:0 --skip_eval
```

通过后再跑正式实验。

---

## 实验与论文章节的对应

| 论文章节 | 实验 | 回答的问题 | 核心指标 |
|---------|------|-----------|---------|
| §4.2 | Exp 1 | Standard KD 保留 saliency 吗？ | Mean JSD |
| §4.3 | Exp 2 | SaGD vs baselines 谁更好？ | ROUGE-L, Mean JSD |
| §4.4 | Exp 3 | 两个组件各自贡献多少？ | ROUGE-L, Mean JSD |
| §4.5 | Exp 4 | 训练中 saliency 如何变化？ | sal_loss, JSD vs step 曲线 |
| §4.6 | Exp 5 | 哪些样本 saliency 偏差最大？ | per-category JSD, case study |
| §4.7 | Exp 6 | 跨架构能泛化吗？ | ROUGE-L on LLaMA pair |
| Appendix | Exp 7 | 通用能力有损害吗？ | MMLU, ARC-C, TruthfulQA |
