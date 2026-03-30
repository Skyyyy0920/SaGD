# SaGD 实验指南

本文档对应论文 §4 的全部实验。按顺序执行，后续实验依赖前序实验的输出。

**方法概要**: SaGD = 噪声 KL（隐式 Jacobian 匹配，position-adaptive）+ Saliency-guided reweighting（DRO）。
超参：`--lambda_noise`（噪声 KL 权重）、`--noise_sigma`（噪声幅度）、`--sagd_tau_w`（DRO 温度）。

**已有 seed=42 单种子结果（Standard KD baseline）**:
- EM: 0.316, Token F1: 0.506, ROUGE-L: 0.497, PPL: 2.48
- Mean JSD: 0.330, Teacher EC: 0.055, Student EC: 0.169

---

## 总览

```
Phase 0   预计算 teacher saliency (SQuAD + Dolly)   1 GPU    ~3h      前置条件
Phase 1   Exp 1: Saliency 诊断（动机实验）            1 GPU    ~2h     → §4.2
Phase 2   Exp 2: 主实验表 SQuAD（3方法 × 3种子）       4 GPU    ~9h     → §4.3
Phase 3   Exp 3: Evidence Concentration 分析          —       ~0h     → §4.4（从 Phase 2 提取）
Phase 4   Exp 4: 消融实验（~9 runs + sweeps）          4 GPU    ~12h    → §4.5
Phase 5   Exp 5: 训练动态曲线                          1 GPU    ~2h     → §4.6
Phase 6   Exp 6: Dolly 泛化验证                        4 GPU    ~6h     → §4.7
Phase 7   Exp 7: 跨架构泛化（LLaMA）                   4 GPU    ~8h     → §4.8
Phase 8   Exp 8: Benchmark 防御（MMLU 等）             1 GPU    ~2h     → Appendix
```

**硬件**: 4× A100 80GB（利用多卡并行加速）
**固定超参**: epochs=3, batch_size=8, grad_accum=4, lr=2e-5, max_seq_len=512, T=2.0, fp16=true
**种子**: 42, 123, 456
**SaGD 默认超参**: λ=0.5, σ=0.1, τ_w=1.0, N=5

---

## Phase 0: 预计算 Teacher Saliency（运行一次）

**目的**: Teacher 是冻结的，saliency 只需算一次，缓存到磁盘。用于 SaGD 训练时的 reweighting + adaptive noise。

**关键**: 必须与训练使用完全相同的 dataset, data_source, seed, max_seq_len, tokenizer, subset。

```bash
# SQuAD（主实验）— 并行跑两个
python scripts/precompute_teacher_saliency.py \
    --model_name Qwen/Qwen3-8B --dataset squad \
    --output_path data/teacher_saliency_squad.pt \
    --batch_size 4 --max_seq_len 512 --device cuda:0 &

# Dolly（泛化实验）
python scripts/precompute_teacher_saliency.py \
    --model_name Qwen/Qwen3-8B --dataset dolly \
    --output_path data/teacher_saliency_dolly.pt \
    --batch_size 4 --max_seq_len 512 --device cuda:1 &

wait
```

**输出**: `data/teacher_saliency_squad.pt`（~86K），`data/teacher_saliency_dolly.pt`（~14K）

---

## Phase 1: Exp 1 — Saliency Divergence 诊断（§4.2）

**论文问题**: "Standard KD 是否保留了 teacher 的 saliency 模式？"

### Step 1.1: 训练 Standard KD baseline（3 种子并行）

```bash
for i in 0 1 2; do
    SEED=(42 123 456)
    GPU=$i
    python scripts/train.py \
        --method standard_kd --dataset squad \
        --seed ${SEED[$i]} --output_dir outputs/ \
        --device cuda:$GPU &
done
wait
```

### Step 1.2: Saliency 诊断 + Evidence Concentration

> **注意**: `diagnose_saliency.py` 直接加载 teacher 模型现算 saliency（不从 train cache 读取），避免 train/val 索引不匹配。需要约 15GB 显存。

```bash
for SEED in 42 123 456; do
    python scripts/diagnose_saliency.py \
        --teacher_model Qwen/Qwen3-8B \
        --student_ckpt outputs/standard_kd/seed_${SEED}/student_final.pt \
        --dataset squad --subset val --max_samples 500 \
        --output_path outputs/standard_kd/seed_${SEED}/saliency_diagnosis.json \
        --device cuda:0
done
```

### Step 1.3: 未训练 student 的诊断（baseline）

```bash
python -c "
from sagd.models import load_student; import torch
student, _ = load_student('Qwen/Qwen3-0.6B', 'cpu')
torch.save(student.state_dict(), 'outputs/pretrained_student.pt')
"

python scripts/diagnose_saliency.py \
    --teacher_model Qwen/Qwen3-8B \
    --student_ckpt outputs/pretrained_student.pt \
    --dataset squad --subset val \
    --output_path outputs/pretrained_saliency_diagnosis.json \
    --device cuda:0
```

### 要报告的数据

| Model | Mean JSD ↓ | Teacher EC | Student EC |
|-------|-----------|------------|------------|
| Pretrained (no KD) | x.xx | 0.055 | x.xx |
| Standard KD | 0.330 | 0.055 | 0.169 |

---

## Phase 2: Exp 2 — 主实验表 SQuAD（§4.3）

**这是论文最核心的表格。**

### 训练（4 卡并行）

```bash
# === Reverse KL（3 种子并行）===
for i in 0 1 2; do
    SEED=(42 123 456)
    python scripts/train.py \
        --method reverse_kl --dataset squad \
        --seed ${SEED[$i]} --output_dir outputs/ \
        --device cuda:$i &
done
wait

# === SaGD（3 种子并行）===
for i in 0 1 2; do
    SEED=(42 123 456)
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_noise 0.5 --noise_sigma 0.1 --sagd_every_n_steps 5 --sagd_tau_w 1.0 \
        --seed ${SEED[$i]} --output_dir outputs/ \
        --device cuda:$i &
done
wait
```

### 评测（所有 checkpoint）

```bash
for METHOD in standard_kd reverse_kl sagd; do
    for SEED in 42 123 456; do
        # EM / F1 (on test subset)
        python scripts/evaluate.py \
            --student_ckpt outputs/${METHOD}/seed_${SEED}/student_final.pt \
            --dataset squad --subset test \
            --output_path outputs/${METHOD}/seed_${SEED}/eval_metrics.json \
            --device cuda:0

        # Saliency Loyalty + EC (on val subset, teacher on-the-fly)
        python scripts/diagnose_saliency.py \
            --teacher_model Qwen/Qwen3-8B \
            --student_ckpt outputs/${METHOD}/seed_${SEED}/student_final.pt \
            --dataset squad --subset val \
            --output_path outputs/${METHOD}/seed_${SEED}/saliency_diagnosis.json \
            --device cuda:0
    done
done
```

### 汇总结果

```bash
python -c "
import json, numpy as np
for method in ['standard_kd', 'reverse_kl', 'sagd']:
    ems, f1s, jsds = [], [], []
    for seed in [42, 123, 456]:
        with open(f'outputs/{method}/seed_{seed}/eval_metrics.json') as f:
            m = json.load(f)
        ems.append(m['exact_match']); f1s.append(m['token_f1'])
        with open(f'outputs/{method}/seed_{seed}/saliency_diagnosis.json') as f:
            d = json.load(f)
        jsds.append(d['mean_jsd'])
    print(f'{method:15s} | EM: {np.mean(ems):.3f}±{np.std(ems):.3f} | F1: {np.mean(f1s):.3f}±{np.std(f1s):.3f} | JSD: {np.mean(jsds):.3f}±{np.std(jsds):.3f}')
"
```

### 要报告的表格

| Method | EM ↑ | Token F1 ↑ | Mean JSD ↓ |
|--------|------|-----------|------------|
| Standard KD | x.xx ± x.xx | x.xx ± x.xx | x.xx ± x.xx |
| Reverse KL | x.xx ± x.xx | x.xx ± x.xx | x.xx ± x.xx |
| **SaGD (ours)** | **x.xx ± x.xx** | **x.xx ± x.xx** | **x.xx ± x.xx** |

---

## Phase 3: Exp 3 — Evidence Concentration 深度分析（§4.4）

**无需额外训练**——从 Phase 2 的 `saliency_diagnosis.json` 提取。

```bash
python -c "
import json
for method in ['standard_kd', 'reverse_kl', 'sagd']:
    tecs, secs = [], []
    for seed in [42, 123, 456]:
        with open(f'outputs/{method}/seed_{seed}/saliency_diagnosis.json') as f:
            d = json.load(f)
        tecs.append(d['teacher_evidence_concentration'])
        secs.append(d['student_evidence_concentration'])
    import numpy as np
    print(f'{method:15s} | Teacher EC: {np.mean(tecs):.4f} | Student EC: {np.mean(secs):.4f}±{np.std(secs):.4f}')
"
```

### 要报告的数据

1. **EC 柱状图**: Teacher vs Standard KD vs SaGD 的 evidence concentration
2. **叙事**: Teacher 低 EC = 全局推理；Standard KD 高 EC = shortcut；SaGD 接近 teacher
3. **Case study**: 挑 3 个样本，可视化 saliency heatmap + answer span

---

## Phase 4: Exp 4 — 消融实验（§4.5）

### 消融配置表

| 配置名 | λ | σ | τ_w | 效果 |
|--------|---|---|-----|------|
| `sagd` (full) | 0.5 | 0.1 | 1.0 | 完整方法 |
| `sagd_noise_only` | 0.5 | 0.1 | 100.0 | τ_w≈∞ → 均匀权重 → 只有 noise KL |
| `sagd_reweight_only` | 0.0 | — | 1.0 | 无 noise KL → 只有 reweighting |

### 运行消融（并行）

```bash
# --- noise KL only ---
for i in 0 1 2; do
    SEED=(42 123 456)
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_noise 0.5 --noise_sigma 0.1 --sagd_tau_w 100.0 --sagd_every_n_steps 5 \
        --seed ${SEED[$i]} --output_dir outputs_ablation/sagd_noise_only/ \
        --device cuda:$i &
done
wait

# --- reweight only ---
for i in 0 1 2; do
    SEED=(42 123 456)
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_noise 0.0 --noise_sigma 0.1 --sagd_tau_w 1.0 --sagd_every_n_steps 5 \
        --seed ${SEED[$i]} --output_dir outputs_ablation/sagd_reweight_only/ \
        --device cuda:$i &
done
wait
```

### 评测消融

```bash
for CONFIG in sagd_noise_only sagd_reweight_only; do
    for SEED in 42 123 456; do
        python scripts/evaluate.py \
            --student_ckpt outputs_ablation/${CONFIG}/sagd/seed_${SEED}/student_final.pt \
            --dataset squad --subset test \
            --output_path outputs_ablation/${CONFIG}/sagd/seed_${SEED}/eval_metrics.json \
            --device cuda:0

        python scripts/diagnose_saliency.py \
            --teacher_model Qwen/Qwen3-8B \
            --student_ckpt outputs_ablation/${CONFIG}/sagd/seed_${SEED}/student_final.pt \
            --dataset squad --subset val \
            --output_path outputs_ablation/${CONFIG}/sagd/seed_${SEED}/saliency_diagnosis.json \
            --device cuda:0
    done
done
```

### 超参敏感性 sweep（单种子 seed=42，4 卡并行）

```bash
# --- λ sweep ---
LAMBDAS=(0.1 0.5 1.0 2.0 5.0)
for i in "${!LAMBDAS[@]}"; do
    GPU=$((i % 4))
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_noise ${LAMBDAS[$i]} --noise_sigma 0.1 --sagd_tau_w 1.0 \
        --seed 42 --output_dir outputs_sweep/lambda_${LAMBDAS[$i]}/ \
        --device cuda:$GPU &
    # 每 4 个等一批
    if (( (i+1) % 4 == 0 )); then wait; fi
done
wait

# --- σ sweep ---
SIGMAS=(0.001 0.005 0.01 0.02 0.05 0.1 0.2 0.5)
for i in "${!SIGMAS[@]}"; do
    GPU=$((i % 4))
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_noise 0.5 --noise_sigma ${SIGMAS[$i]} --sagd_tau_w 1.0 \
        --seed 42 --output_dir outputs_sweep/sigma_${SIGMAS[$i]}/ \
        --device cuda:$GPU &
    if (( (i+1) % 4 == 0 )); then wait; fi
done
wait

# --- τ_w sweep ---
TAUS=(0.1 0.5 1.0 2.0 5.0)
for i in "${!TAUS[@]}"; do
    GPU=$((i % 4))
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_noise 0.5 --noise_sigma 0.1 --sagd_tau_w ${TAUS[$i]} \
        --seed 42 --output_dir outputs_sweep/tau_${TAUS[$i]}/ \
        --device cuda:$GPU &
    if (( (i+1) % 4 == 0 )); then wait; fi
done
wait

# --- N sweep ---
NS=(1 3 5 10 20)
for i in "${!NS[@]}"; do
    GPU=$((i % 4))
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_noise 0.5 --noise_sigma 0.1 --sagd_tau_w 1.0 --sagd_every_n_steps ${NS[$i]} \
        --seed 42 --output_dir outputs_sweep/every_n_${NS[$i]}/ \
        --device cuda:$GPU &
    if (( (i+1) % 4 == 0 )); then wait; fi
done
wait
```

### 评测 sweep

```bash
# λ sweep
for LAMBDA in 0.1 0.5 1.0 2.0 5.0; do
    python scripts/evaluate.py \
        --student_ckpt outputs_sweep/lambda_${LAMBDA}/sagd/seed_42/student_final.pt \
        --dataset squad --subset test \
        --output_path outputs_sweep/lambda_${LAMBDA}/sagd/seed_42/eval_metrics.json \
        --device cuda:0
done

# σ sweep
for SIGMA in 0.001 0.005 0.01 0.02 0.05 0.1 0.2 0.5; do
    python scripts/evaluate.py \
        --student_ckpt outputs_sweep/sigma_${SIGMA}/sagd/seed_42/student_final.pt \
        --dataset squad --subset test \
        --output_path outputs_sweep/sigma_${SIGMA}/sagd/seed_42/eval_metrics.json \
        --device cuda:0
done

# τ_w sweep
for TAU in 0.1 0.5 1.0 2.0 5.0; do
    python scripts/evaluate.py \
        --student_ckpt outputs_sweep/tau_${TAU}/sagd/seed_42/student_final.pt \
        --dataset squad --subset test \
        --output_path outputs_sweep/tau_${TAU}/sagd/seed_42/eval_metrics.json \
        --device cuda:0
done

# N sweep
for N in 1 3 5 10 20; do
    python scripts/evaluate.py \
        --student_ckpt outputs_sweep/every_n_${N}/sagd/seed_42/student_final.pt \
        --dataset squad --subset test \
        --output_path outputs_sweep/every_n_${N}/sagd/seed_42/eval_metrics.json \
        --device cuda:0
done
```

### 要报告的表格

**消融表**:

| Config | Clean KL | Noise KL | Reweight | EM ↑ | F1 ↑ | JSD ↓ |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Standard KD | uniform | — | — | x.xx | x.xx | x.xx |
| + Noise KL only | uniform | ✓ | — | x.xx | x.xx | x.xx |
| + Reweight only | weighted | — | ✓ | x.xx | x.xx | x.xx |
| **SaGD (full)** | weighted | ✓ | ✓ | **x.xx** | **x.xx** | **x.xx** |

**超参敏感性图 (Appendix)**: λ vs EM, σ vs EM, τ_w vs EM, N vs EM 折线图

---

## Phase 5: Exp 5 — 训练动态（§4.6）

**无需额外训练**——从 Phase 2 的 SaGD 训练 log 提取。

```bash
cat outputs/sagd/seed_42/training_stats.jsonl | python -c "
import sys, json
print('step\tloss\tkl_noisy\tkl_clean\tmean_jsd\tmax_weight')
for line in sys.stdin:
    d = json.loads(line)
    if 'sagd/kl_noisy' in d:
        print(f\"{d['step']}\t{d['loss']:.4f}\t{d['sagd/kl_noisy']:.4f}\t{d['sagd/kl_clean']:.4f}\t{d['sagd/mean_jsd']:.4f}\t{d['sagd/max_weight']:.2f}\")
" > outputs/sagd/seed_42/dynamics.tsv
```

### Epoch-level 诊断（需要 teacher 模型）

```bash
for METHOD in sagd standard_kd; do
    for EPOCH in 1 2 3; do
        python scripts/diagnose_saliency.py \
            --teacher_model Qwen/Qwen3-8B \
            --student_ckpt outputs/${METHOD}/seed_42/student_epoch${EPOCH}.pt \
            --dataset squad --subset val \
            --output_path outputs/${METHOD}/seed_42/saliency_epoch${EPOCH}.json \
            --device cuda:0
    done
done
```

### 要报告的图

1. **kl_noisy - kl_clean vs step**（差异 = Jacobian gap 的代理，应该随训练下降）
2. **Mean JSD vs epoch**: SaGD vs Standard KD
3. **Max weight vs step**: DRO reweighting 的动态

---

## Phase 6: Exp 6 — Dolly 泛化验证（§4.7）

```bash
# 训练（并行）
for i in 0 1 2; do
    SEED=(42 123 456)
    # Standard KD
    python scripts/train.py \
        --method standard_kd --dataset dolly \
        --seed ${SEED[$i]} --output_dir outputs_dolly/ \
        --device cuda:$i &
done
wait

for i in 0 1 2; do
    SEED=(42 123 456)
    # SaGD
    python scripts/train.py \
        --method sagd --dataset dolly \
        --teacher_saliency_path data/teacher_saliency_dolly.pt \
        --lambda_noise 0.5 --noise_sigma 0.1 --sagd_every_n_steps 5 --sagd_tau_w 1.0 \
        --seed ${SEED[$i]} --output_dir outputs_dolly/ \
        --device cuda:$i &
done
wait

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
# 预计算 LLaMA teacher saliency
python scripts/precompute_teacher_saliency.py \
    --model_name meta-llama/Llama-3.1-8B \
    --tokenizer_name meta-llama/Llama-3.1-1B \
    --dataset squad \
    --output_path data/teacher_saliency_llama_squad.pt \
    --batch_size 4 --max_seq_len 512 --device cuda:0

# 训练（并行）
for i in 0 1 2; do
    SEED=(42 123 456)
    python scripts/train.py \
        --method standard_kd --dataset squad \
        --teacher_model meta-llama/Llama-3.1-8B --student_model meta-llama/Llama-3.1-1B \
        --seed ${SEED[$i]} --output_dir outputs_llama/ \
        --device cuda:$i &
done
wait

for i in 0 1 2; do
    SEED=(42 123 456)
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_model meta-llama/Llama-3.1-8B --student_model meta-llama/Llama-3.1-1B \
        --teacher_saliency_path data/teacher_saliency_llama_squad.pt \
        --lambda_noise 0.5 --noise_sigma 0.1 --sagd_every_n_steps 5 --sagd_tau_w 1.0 \
        --seed ${SEED[$i]} --output_dir outputs_llama/ \
        --device cuda:$i &
done
wait

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

---

## Phase 8: Exp 8 — Benchmark 防御（Appendix）

```bash
pip install lm-eval

for METHOD in standard_kd sagd; do
    python -c "
from sagd.models import load_student; import torch
student, _ = load_student('Qwen/Qwen3-0.6B', 'cpu')
student.load_state_dict(torch.load('outputs/${METHOD}/seed_42/student_final.pt', map_location='cpu', weights_only=True))
student.save_pretrained('outputs/${METHOD}/seed_42/hf_model/')
"
    lm_eval --model hf \
        --model_args pretrained=outputs/${METHOD}/seed_42/hf_model/ \
        --tasks mmlu,arc_challenge,truthfulqa_mc2 \
        --batch_size 8 --output_path outputs/${METHOD}/seed_42/benchmark/
done

lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen3-0.6B \
    --tasks mmlu,arc_challenge,truthfulqa_mc2 \
    --batch_size 8 --output_path outputs/base_student_benchmark/
```

---

## 输出目录结构

```
data/
├── teacher_saliency_squad.pt
├── teacher_saliency_dolly.pt
└── teacher_saliency_llama_squad.pt

outputs/                           ← SQuAD 主实验 (Phase 1-3, 5)
├── standard_kd/seed_{42,123,456}/
│   ├── config.json, student_final.pt, student_epoch{1,2,3}.pt
│   ├── training_stats.jsonl
│   ├── eval_metrics.json          ← EM, F1, ROUGE-L, PPL
│   └── saliency_diagnosis.json    ← JSD, EC
├── reverse_kl/seed_{42,123,456}/
├── sagd/seed_{42,123,456}/

outputs_ablation/                  ← 消融 (Phase 4)
├── sagd_noise_only/sagd/seed_{42,123,456}/
├── sagd_reweight_only/sagd/seed_{42,123,456}/

outputs_sweep/                     ← 超参 sweep (Phase 4, seed=42 only)
├── lambda_{0.1,...,5.0}/sagd/seed_42/
├── sigma_{0.001,...,0.5}/sagd/seed_42/
├── tau_{0.1,...,5.0}/sagd/seed_42/
├── every_n_{1,...,20}/sagd/seed_42/

outputs_dolly/                     ← Dolly 泛化 (Phase 6)
outputs_llama/                     ← LLaMA 跨架构 (Phase 7)
```

---

## Smoke Test（正式跑之前先验证）

```bash
# 1. 单元测试
pytest tests/ -v

# 2. 快速验证 SaGD 训练不报错
python scripts/train.py \
    --method sagd --dataset squad \
    --teacher_saliency_path data/teacher_saliency_squad.pt \
    --lambda_noise 0.5 --noise_sigma 0.1 \
    --epochs 1 --max_train_samples 200 \
    --device cuda:0
```

---

## 实验与论文章节的对应

| 论文章节 | Phase | 回答的问题 | 核心指标 |
|---------|-------|-----------|---------|
| §4.2 | 1 | Standard KD 保留 saliency 吗？ | Mean JSD, EC |
| §4.3 | 2 | SaGD vs baselines 谁更好？ | EM, F1, JSD |
| §4.4 | 3 | Student 保留了 teacher 的推理模式吗？ | Evidence Concentration |
| §4.5 | 4 | 两个组件各自贡献多少？ | EM, F1, JSD |
| §4.6 | 5 | 训练中 Jacobian gap 如何变化？ | kl_noisy-kl_clean, JSD vs step |
| §4.7 | 6 | SQuAD 之外能泛化吗？ | ROUGE-L on Dolly |
| §4.8 | 7 | 跨架构能泛化吗？ | EM, F1 on LLaMA pair |
| Appendix | 8 | 通用能力有损害吗？ | MMLU, ARC-C, TruthfulQA |

---

## 总计算量估算

| Phase | Runs | GPU-hours (A100) |
|-------|------|-----------------|
| 0 | 2 | 3 |
| 1 | 3 train + 4 diagnose | 12 |
| 2 | 6 train + 18 eval/diagnose | 24 |
| 3 | 0（从 Phase 2 提取） | 0 |
| 4 | 6 ablation + 23 sweep + 29 eval | 100 |
| 5 | 6 diagnose | 3 |
| 6 | 6 train + 6 eval | 24 |
| 7 | 6 train + 6 eval + 1 precompute | 30 |
| 8 | 3 lm-eval | 6 |
| **Total** | | **~200 GPU-hours** |

4× A100 并行约需 **~50 wall-clock hours（~2 天）**。
