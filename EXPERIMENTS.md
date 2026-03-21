# SaGD 实验指南

本文档对应论文 §4 的全部实验。按顺序执行，后续实验依赖前序实验的输出。

---

## 总览

```
Phase 0   预计算 teacher saliency (SQuAD + Dolly)   1 GPU    ~3h
Phase 1   Exp 1: Saliency 诊断（动机实验）            1 GPU    ~1h     → §4.2
Phase 2   Exp 2: 主实验表 SQuAD（3方法 × 3种子）       4 GPU    ~6h     → §4.3
Phase 3   Exp 3: Evidence Concentration 分析          1 GPU    ~1h     → §4.4
Phase 4   Exp 4: 消融实验（~15 runs）                  4 GPU    ~8h     → §4.5
Phase 5   Exp 5: 训练动态曲线                          1 GPU    ~2h     → §4.6
Phase 6   Exp 6: Dolly 泛化验证                        4 GPU    ~6h     → §4.7
Phase 7   Exp 7: 跨架构泛化（LLaMA）                   1 GPU    ~4h     → §4.8
Phase 8   Exp 8: Benchmark 防御（MMLU 等）             1 GPU    ~2h     → Appendix
```

**硬件**: 4× A100 80GB（Phase 2,4,6 可并行；其他 phase 单卡即可）
**固定超参**: epochs=3, batch_size=8, grad_accum=4, lr=2e-5, max_seq_len=512, T=2.0, fp16=true
**种子**: 42, 123, 456

---

## Phase 0: 预计算 Teacher Saliency（运行一次）

**目的**: Teacher 是冻结的，saliency 只需算一次，缓存到磁盘。需要分别为 SQuAD 和 Dolly 各算一份。

**关键**: 必须与训练使用完全相同的 dataset, data_source, seed, max_seq_len, tokenizer, subset。

### SQuAD（主实验）

```bash
python scripts/precompute_teacher_saliency.py \
    --model_name Qwen/Qwen3-8B \
    --dataset squad \
    --output_path data/teacher_saliency_squad.pt \
    --batch_size 4 --max_seq_len 512 --device cuda:0
```

**输出**: `data/teacher_saliency_squad.pt`（~86K 样本）
**耗时**: ~2 小时

### Dolly（泛化实验）

```bash
python scripts/precompute_teacher_saliency.py \
    --model_name Qwen/Qwen3-8B \
    --dataset dolly \
    --output_path data/teacher_saliency_dolly.pt \
    --batch_size 4 --max_seq_len 512 --device cuda:0
```

**输出**: `data/teacher_saliency_dolly.pt`（~14K 样本）
**耗时**: ~1 小时

---

## Phase 1: Exp 1 — Saliency Divergence 诊断（§4.2）

**论文问题**: "Standard KD 是否保留了 teacher 的 saliency 模式？"
**预期结论**: 不保留。Standard KD student 与 teacher 存在显著的 saliency 偏差。

### Step 1.1: 训练 Standard KD baseline（SQuAD）

```bash
for SEED in 42 123 456; do
    python scripts/train.py \
        --method standard_kd --dataset squad \
        --seed $SEED --output_dir outputs/ \
        --device cuda:0
done
```

### Step 1.2: Saliency 诊断 + Evidence Concentration

```bash
for SEED in 42 123 456; do
    python scripts/diagnose_saliency.py \
        --student_ckpt outputs/standard_kd/seed_${SEED}/student_final.pt \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --dataset squad --subset val --max_samples 500 \
        --output_path outputs/standard_kd/seed_${SEED}/saliency_diagnosis.json \
        --device cuda:0
done
```

**输出**: 每个种子一个 JSON，包含:
```json
{
    "mean_jsd": 0.xxx,
    "std_jsd": 0.xxx,
    "teacher_evidence_concentration": 0.xxx,
    "student_evidence_concentration": 0.xxx,
    "top20_samples": [...]
}
```

### Step 1.3（可选）: 未训练 student 的诊断

```bash
python -c "
from sagd.models import load_student; import torch
student, _ = load_student('Qwen/Qwen3-0.6B', 'cpu')
torch.save(student.state_dict(), 'outputs/pretrained_student.pt')
"

python scripts/diagnose_saliency.py \
    --student_ckpt outputs/pretrained_student.pt \
    --teacher_saliency_path data/teacher_saliency_squad.pt \
    --dataset squad --subset val \
    --output_path outputs/pretrained_saliency_diagnosis.json \
    --device cuda:0
```

### 要报告的数据

| Model | Mean JSD ↓ | Teacher EC | Student EC | EC Gap ↓ |
|-------|-----------|------------|------------|----------|
| Pretrained (no KD) | 较高 | x.xx | 较低 | 较大 |
| Standard KD | 仍较高 | x.xx | 仍较低 | 仍较大 |

---

## Phase 2: Exp 2 — 主实验表 SQuAD（§4.3）

**这是论文最核心的表格。** SQuAD 上的 EM, F1, Evidence Concentration。

### 三个方法 × 三个种子 = 9 runs

```bash
# === Standard KD（已在 Phase 1 训完，补充评测即可）===
for SEED in 42 123 456; do
    python scripts/evaluate.py \
        --student_ckpt outputs/standard_kd/seed_${SEED}/student_final.pt \
        --dataset squad --subset test \
        --output_path outputs/standard_kd/seed_${SEED}/eval_metrics.json \
        --device cuda:0
done

# === Reverse KL ===
for SEED in 42 123 456; do
    python scripts/train.py \
        --method reverse_kl --dataset squad \
        --seed $SEED --output_dir outputs/ \
        --device cuda:0
done

# === SaGD (our method) ===
for SEED in 42 123 456; do
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_sal 0.5 --sagd_every_n_steps 5 --sagd_tau_w 1.0 \
        --seed $SEED --output_dir outputs/ \
        --device cuda:0
done
```

### 评测每个 checkpoint

```bash
for METHOD in standard_kd reverse_kl sagd; do
    for SEED in 42 123 456; do
        # EM / F1 (on test subset)
        python scripts/evaluate.py \
            --student_ckpt outputs/${METHOD}/seed_${SEED}/student_final.pt \
            --dataset squad --subset test \
            --output_path outputs/${METHOD}/seed_${SEED}/eval_metrics.json \
            --device cuda:0

        # Saliency Loyalty + Evidence Concentration (on val subset)
        python scripts/diagnose_saliency.py \
            --student_ckpt outputs/${METHOD}/seed_${SEED}/student_final.pt \
            --teacher_saliency_path data/teacher_saliency_squad.pt \
            --dataset squad --subset val \
            --output_path outputs/${METHOD}/seed_${SEED}/saliency_diagnosis.json \
            --device cuda:0
    done
done
```

### 要报告的表格

| Method | EM ↑ | Token F1 ↑ | Evidence Conc. ↑ | Mean JSD ↓ |
|--------|------|-----------|-------------------|------------|
| Standard KD | x.xx ± x.xx | x.xx ± x.xx | x.xx ± x.xx | x.xx ± x.xx |
| Reverse KL | x.xx ± x.xx | x.xx ± x.xx | x.xx ± x.xx | x.xx ± x.xx |
| **SaGD (ours)** | **x.xx ± x.xx** | **x.xx ± x.xx** | **x.xx ± x.xx** | **x.xx ± x.xx** |

---

## Phase 3: Exp 3 — Evidence Concentration 深度分析（§4.4）

**论文问题**: SaGD 是否让 student 看到了正确的证据？

**这是 SQuAD 数据集带来的核心新贡献**——直接用 answer span 作为 ground truth 验证 saliency quality。

### 可视化数据

从 Phase 2 的 saliency_diagnosis.json 中提取 teacher/student EC 对比：

```bash
python -c "
import json
for method in ['standard_kd', 'sagd']:
    with open(f'outputs/{method}/seed_42/saliency_diagnosis.json') as f:
        d = json.load(f)
    print(f'\n=== {method} ===')
    print(f'Mean JSD: {d[\"mean_jsd\"]:.4f}')
    print(f'Teacher EC: {d[\"teacher_evidence_concentration\"]:.4f}')
    print(f'Student EC: {d[\"student_evidence_concentration\"]:.4f}')
    print(f'EC Gap: {d[\"teacher_evidence_concentration\"] - d[\"student_evidence_concentration\"]:.4f}')
"
```

### 要报告的数据

1. **EC 柱状图**: Teacher vs Standard KD student vs SaGD student 的 evidence concentration
2. **Case study**: 挑 3 个样本，可视化 teacher/student saliency heatmap 叠加 answer span 标注
3. **散点图**: per-sample teacher EC vs student EC，SaGD 的点应更接近 y=x 线

---

## Phase 4: Exp 4 — 消融实验（§4.5）

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
# --- Ablation: saliency loss only ---
for SEED in 42 123 456; do
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_sal 0.5 --sagd_tau_w 100.0 --sagd_every_n_steps 5 \
        --seed $SEED --output_dir outputs_ablation/sagd_loss_only/ \
        --device cuda:0
done

# --- Ablation: reweight only ---
for SEED in 42 123 456; do
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_sal 0.0 --sagd_tau_w 1.0 --sagd_every_n_steps 5 \
        --seed $SEED --output_dir outputs_ablation/sagd_reweight_only/ \
        --device cuda:0
done
```

### 超参敏感性（Appendix）

```bash
# --- λ sweep ---
for LAMBDA in 0.01 0.1 0.5 1.0 2.0; do
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_sal $LAMBDA --sagd_tau_w 1.0 \
        --seed 42 --output_dir outputs_sweep/lambda_${LAMBDA}/ \
        --device cuda:0
done

# --- τ_w sweep ---
for TAU in 0.1 0.5 1.0 2.0 5.0; do
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_sal 0.5 --sagd_tau_w $TAU \
        --seed 42 --output_dir outputs_sweep/tau_${TAU}/ \
        --device cuda:0
done

# --- N (saliency update frequency) sweep ---
for N in 1 3 5 10 20; do
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_sal 0.5 --sagd_tau_w 1.0 --sagd_every_n_steps $N \
        --seed 42 --output_dir outputs_sweep/every_n_${N}/ \
        --device cuda:0
done
```

### 要报告的表格

**消融表**:

| Config | KL | Sal loss | Reweight | EM ↑ | F1 ↑ | EC ↑ | JSD ↓ |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Standard KD | uniform | — | — | x.xx | x.xx | x.xx | x.xx |
| + Sal loss only | uniform | ✓ | — | x.xx | x.xx | x.xx | x.xx |
| + Reweight only | weighted | — | ✓ | x.xx | x.xx | x.xx | x.xx |
| **SaGD (full)** | weighted | ✓ | ✓ | **x.xx** | **x.xx** | **x.xx** | **x.xx** |

---

## Phase 5: Exp 5 — 训练动态（§4.6）

**数据来源**: SaGD 训练的 `training_stats.jsonl`。

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

### 需要额外的 epoch-level 诊断

```bash
for EPOCH in 1 2 3; do
    python scripts/diagnose_saliency.py \
        --student_ckpt outputs/sagd/seed_42/student_epoch${EPOCH}.pt \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --dataset squad --subset val \
        --output_path outputs/sagd/seed_42/saliency_epoch${EPOCH}.json \
        --device cuda:0

    python scripts/diagnose_saliency.py \
        --student_ckpt outputs/standard_kd/seed_42/student_epoch${EPOCH}.pt \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --dataset squad --subset val \
        --output_path outputs/standard_kd/seed_42/saliency_epoch${EPOCH}.json \
        --device cuda:0
done
```

### 要报告的图

1. **Saliency loss vs step**: `sagd/sal_loss` 下降曲线
2. **Evidence Concentration vs epoch**: SaGD vs Standard KD 的 EC 随 epoch 变化
3. **Mean JSD vs epoch**: 对比 JSD 下降

---

## Phase 6: Exp 6 — Dolly 泛化验证（§4.7）

**论文问题**: SaGD 是否泛化到非 extractive QA 任务？

```bash
for METHOD in standard_kd sagd; do
    for SEED in 42 123 456; do
        EXTRA_ARGS=""
        if [ "$METHOD" == "sagd" ]; then
            EXTRA_ARGS="--teacher_saliency_path data/teacher_saliency_dolly.pt \
                        --lambda_sal 0.5 --sagd_every_n_steps 5 --sagd_tau_w 1.0"
        fi

        python scripts/train.py \
            --method $METHOD --dataset dolly \
            --seed $SEED --output_dir outputs_dolly/ \
            $EXTRA_ARGS --device cuda:0
    done
done

# 评测
for METHOD in standard_kd sagd; do
    for SEED in 42 123 456; do
        python scripts/evaluate.py \
            --student_ckpt outputs_dolly/${METHOD}/seed_${SEED}/student_final.pt \
            --dataset dolly --subset test \
            --output_path outputs_dolly/${METHOD}/seed_${SEED}/eval_metrics.json \
            --device cuda:0
    done
done
```

### 要报告的表格

| Dataset | Method | Primary Metric ↑ | Mean JSD ↓ |
|---------|--------|-------------------|------------|
| SQuAD | Standard KD | EM: x.xx, F1: x.xx | x.xx |
| SQuAD | **SaGD** | **EM: x.xx, F1: x.xx** | **x.xx** |
| Dolly | Standard KD | ROUGE-L: x.xx | x.xx |
| Dolly | **SaGD** | **ROUGE-L: x.xx** | **x.xx** |

---

## Phase 7: Exp 7 — 跨架构泛化（§4.8）

**模型对**: LLaMA 3.1-8B → LLaMA 3.1-1B

```bash
# 预计算 LLaMA teacher saliency (SQuAD)
python scripts/precompute_teacher_saliency.py \
    --model_name meta-llama/Llama-3.1-8B \
    --tokenizer_name meta-llama/Llama-3.1-1B \
    --dataset squad \
    --output_path data/teacher_saliency_llama_squad.pt \
    --batch_size 4 --max_seq_len 512 --device cuda:0

# 训练
for METHOD in standard_kd sagd; do
    for SEED in 42 123 456; do
        EXTRA_ARGS=""
        if [ "$METHOD" == "sagd" ]; then
            EXTRA_ARGS="--teacher_saliency_path data/teacher_saliency_llama_squad.pt \
                        --lambda_sal 0.5 --sagd_every_n_steps 5 --sagd_tau_w 1.0"
        fi

        python scripts/train.py \
            --method $METHOD --dataset squad \
            --teacher_model meta-llama/Llama-3.1-8B \
            --student_model meta-llama/Llama-3.1-1B \
            --seed $SEED --output_dir outputs_llama/ \
            $EXTRA_ARGS --device cuda:0
    done
done

# 评测
for METHOD in standard_kd sagd; do
    for SEED in 42 123 456; do
        python scripts/evaluate.py \
            --student_model meta-llama/Llama-3.1-1B \
            --student_ckpt outputs_llama/${METHOD}/seed_${SEED}/student_final.pt \
            --dataset squad --subset test \
            --output_path outputs_llama/${METHOD}/seed_${SEED}/eval_metrics.json \
            --device cuda:0
    done
done
```

### 要报告的表格

| Architecture | Method | EM ↑ | F1 ↑ | EC ↑ |
|-------------|--------|------|------|------|
| Qwen 8B→0.6B | Standard KD | x.xx | x.xx | x.xx |
| Qwen 8B→0.6B | **SaGD** | **x.xx** | **x.xx** | **x.xx** |
| LLaMA 8B→1B | Standard KD | x.xx | x.xx | x.xx |
| LLaMA 8B→1B | **SaGD** | **x.xx** | **x.xx** | **x.xx** |

---

## Phase 8: Exp 8 — Benchmark 防御（Appendix）

**目的**: 证明 SaGD 提升 EM/F1 的同时没有损害通用能力。

```bash
pip install lm-eval

for METHOD in standard_kd sagd; do
    # 先转为 HuggingFace 格式
    python -c "
from sagd.models import load_student; import torch
student, _ = load_student('Qwen/Qwen3-0.6B', 'cpu')
student.load_state_dict(torch.load('outputs/${METHOD}/seed_42/student_final.pt', map_location='cpu', weights_only=True))
student.save_pretrained('outputs/${METHOD}/seed_42/hf_model/')
"
    lm_eval --model hf \
        --model_args pretrained=outputs/${METHOD}/seed_42/hf_model/ \
        --tasks mmlu,arc_challenge,truthfulqa_mc2 \
        --batch_size 8 \
        --output_path outputs/${METHOD}/seed_42/benchmark/
done

# Base student
lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen3-0.6B \
    --tasks mmlu,arc_challenge,truthfulqa_mc2 \
    --batch_size 8 \
    --output_path outputs/base_student_benchmark/
```

---

## 输出目录结构总览

```
data/
├── teacher_saliency_squad.pt      ← SQuAD teacher (primary)
├── teacher_saliency_dolly.pt      ← Dolly teacher (secondary)
└── teacher_saliency_llama_squad.pt ← LLaMA teacher

outputs/                           ← SQuAD 主实验
├── standard_kd/seed_{42,123,456}/
│   ├── config.json, student_final.pt, training_stats.jsonl
│   ├── eval_metrics.json          ← EM, F1, ROUGE-L, PPL
│   └── saliency_diagnosis.json    ← JSD + Evidence Concentration
├── reverse_kl/...
├── sagd/...

outputs_ablation/                  ← SQuAD 消融
├── sagd_loss_only/sagd/seed_{42,123,456}/
├── sagd_reweight_only/sagd/seed_{42,123,456}/

outputs_sweep/                     ← SQuAD 超参 sweep
├── lambda_{0.01,...,2.0}/sagd/seed_42/
├── tau_{0.1,...,5.0}/sagd/seed_42/
├── every_n_{1,...,20}/sagd/seed_42/

outputs_dolly/                     ← Dolly 泛化
├── standard_kd/seed_{42,123,456}/
├── sagd/seed_{42,123,456}/

outputs_llama/                     ← LLaMA 跨架构
├── standard_kd/seed_{42,123,456}/
├── sagd/seed_{42,123,456}/
```

---

## Smoke Test（正式跑之前先验证）

```bash
# 1. 单元测试
pytest tests/ -v

# 2. SQuAD smoke test
python scripts/precompute_teacher_saliency.py \
    --dataset squad --max_samples 200 \
    --output_path data/test_saliency_squad.pt --device cuda:0

python scripts/train.py \
    --method standard_kd --dataset squad \
    --epochs 1 --max_train_samples 200 \
    --device cuda:0 --skip_eval

python scripts/train.py \
    --method sagd --dataset squad \
    --teacher_saliency_path data/test_saliency_squad.pt \
    --epochs 1 --max_train_samples 200 \
    --device cuda:0 --skip_eval
```

---

## 实验与论文章节的对应

| 论文章节 | 实验 | 回答的问题 | 核心指标 |
|---------|------|-----------|---------|
| §4.2 | Exp 1 | Standard KD 保留 saliency 吗？ | Mean JSD, EC |
| §4.3 | Exp 2 | SaGD vs baselines 谁更好？ | EM, F1, EC, JSD |
| §4.4 | Exp 3 | Student 是否看到了正确的证据？ | Evidence Concentration |
| §4.5 | Exp 4 | 两个组件各自贡献多少？ | EM, F1, EC, JSD |
| §4.6 | Exp 5 | 训练中 saliency 如何变化？ | sal_loss, JSD, EC vs step |
| §4.7 | Exp 6 | SQuAD 之外能泛化吗？ | ROUGE-L on Dolly |
| §4.8 | Exp 7 | 跨架构能泛化吗？ | EM, F1 on LLaMA pair |
| Appendix | Exp 8 | 通用能力有损害吗？ | MMLU, ARC-C, TruthfulQA |

---

## 扩展评测系统

### 评测指标一览

| 指标 | 用途 | 数据集 | 备注 |
|------|------|--------|------|
| Exact Match | 答案完全正确 | SQuAD | 主指标 |
| Token F1 | 答案词级重叠 | SQuAD | 主指标 |
| Evidence Concentration | Saliency 在答案 span 上的比例 | SQuAD | 核心新指标 |
| ROUGE-L | 文本重叠度 | Dolly | 泛化指标 |
| Mean JSD | Saliency 忠诚度 | 两者 | 二级指标 |
| BERTScore | 语义相似度 | 两者 | 可选 |
| Perplexity | 语言模型质量 | 两者 | 越低越好 |
| GPT-as-Judge | 人类偏好代理 | 两者 | 需 OpenAI API |

### GPT-as-Judge 对比

```bash
# 预生成 responses
for METHOD in standard_kd sagd; do
    for SEED in 42 123 456; do
        python scripts/generate_responses.py \
            --student_ckpt outputs/${METHOD}/seed_${SEED}/student_final.pt \
            --dataset squad --subset test \
            --output_path outputs/${METHOD}/seed_${SEED}/responses.jsonl \
            --device cuda:0
    done
done

# Pairwise judge
python scripts/gpt_judge.py \
    --responses_a outputs/standard_kd/seed_42/responses.jsonl \
    --responses_b outputs/sagd/seed_42/responses.jsonl \
    --label_a "Standard KD" --label_b "SaGD" \
    --output_path outputs/gpt_judge_std_vs_sagd_seed42.json
```
