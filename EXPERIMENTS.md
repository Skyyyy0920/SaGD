# SaGD 实验指南

本文档对应论文 §4 的全部实验。按顺序执行，后续实验依赖前序实验的输出。

**方法概要**: SaGD = 噪声 KL（隐式 Jacobian 匹配，position-adaptive）+ Saliency-guided reweighting（DRO）。

**确定的超参**: λ=0.5, σ=0.005 (relative to embed norm), τ_w=1.0, N=5

**已有初步结果** (epoch=1, seed=42):
- Standard KD: EM 0.316, F1 0.506
- SaGD σ=0.005: EM 0.328, F1 0.527 (+1.2% EM, +2.1% F1)
- Evidence Concentration: Teacher 0.055, StdKD 0.169, SaGD 0.083

---

## 总览

```
Phase 0   预计算 teacher saliency                     1 GPU    ~3h     前置
Phase 1   主实验表 SQuAD（3方法 × 3种子 × 3epoch）      4 GPU    ~27h    §4.3 核心
Phase 2   Saliency 诊断 + EC 分析                      1 GPU    ~4h     §4.2, §4.4
Phase 3   消融实验（6 runs × 3epoch）                    4 GPU    ~18h    §4.5
Phase 4   超参 sweep（~22 runs × 1epoch）               4 GPU    ~17h    Appendix
Phase 5   训练动态                                      —        ~0h     §4.6（从 Phase 1 提取）
Phase 6   Dolly 泛化验证                                4 GPU    ~6h     §4.7
Phase 7   跨架构 LLaMA                                  4 GPU    ~12h    §4.8
Phase 8   Benchmark 防御                                1 GPU    ~2h     Appendix
```

**总计**: ~90 GPU-hours, 4×A100 并行约 **~24h wall-clock**

**硬件**: 4× A100 80GB
**固定超参**: epochs=3, batch_size=8, grad_accum=4, lr=2e-5, max_seq_len=512, T=2.0, fp16=true
**种子**: 42, 123, 456

---

## Phase 0: 预计算 Teacher Saliency（一次性）

**目的**: Teacher saliency 缓存用于训练时的 reweighting 和 adaptive noise。

**关键**: 必须与训练使用完全相同的 dataset, seed, max_seq_len, tokenizer, subset。

```bash
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# 并行
python scripts/precompute_teacher_saliency.py \
    --model_name Qwen/Qwen3-8B --dataset squad \
    --output_path data/teacher_saliency_squad.pt \
    --batch_size 4 --max_seq_len 512 --device cuda:0 &

python scripts/precompute_teacher_saliency.py \
    --model_name Qwen/Qwen3-8B --dataset dolly \
    --output_path data/teacher_saliency_dolly.pt \
    --batch_size 4 --max_seq_len 512 --device cuda:1 &

wait
echo "Phase 0 done."
```

---

## Phase 1: 主实验表 SQuAD（§4.3）— 最核心

**论文问题**: SaGD vs baselines，谁在 SQuAD extractive QA 上更好？

**指标**: EM（完全匹配）, Token F1（词级 F1）, Mean JSD（saliency 忠诚度）

**为什么 3 epoch**: epoch=1 的 gain 不显著（方差内）。3 epoch 让一阶匹配信号充分积累——理论上残差项 $R_t = O(\|p_T-p_S\|)$ 在后期趋零，$\mathcal{F}_t$ 成为主导。

### 训练（3 方法 × 3 种子 = 9 runs）

```bash
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

for SEED in 42 123 456; do
    # Standard KD
    python scripts/train.py \
        --method standard_kd --dataset squad \
        --seed $SEED --output_dir outputs/ --epochs 3 \
        --device cuda:0 &

    # SaGD
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_noise 0.5 --noise_sigma 0.005 --sagd_every_n_steps 5 --sagd_tau_w 1.0 \
        --seed $SEED --output_dir outputs/ --epochs 3 \
        --device cuda:1 &

    # Reverse KL
    python scripts/train.py \
        --method reverse_kl --dataset squad \
        --seed $SEED --output_dir outputs/ --epochs 3 \
        --device cuda:2 &

    wait  # 每组 seed 等跑完再下一组（防止 OOM）
done
```

### 评测

```bash
for METHOD in standard_kd reverse_kl sagd; do
    for SEED in 42 123 456; do
        python scripts/evaluate.py \
            --student_ckpt outputs/${METHOD}/seed_${SEED}/student_final.pt \
            --dataset squad --subset test \
            --output_path outputs/${METHOD}/seed_${SEED}/eval_metrics.json \
            --device cuda:0
    done
done
```

### 汇总

```bash
python -c "
import json, numpy as np
print(f'{'Method':<15} | {'EM':>12} | {'F1':>12} | {'ROUGE-L':>12} | {'PPL':>6}')
print('-' * 70)
for method in ['standard_kd', 'reverse_kl', 'sagd']:
    ems, f1s, rls, ppls = [], [], [], []
    for seed in [42, 123, 456]:
        try:
            with open(f'outputs/{method}/seed_{seed}/eval_metrics.json') as f:
                m = json.load(f)
            ems.append(m['exact_match']); f1s.append(m['token_f1'])
            rls.append(m['rouge_l_f']); ppls.append(m['perplexity'])
        except: pass
    if ems:
        print(f'{method:<15} | {np.mean(ems):.3f}±{np.std(ems):.3f} | {np.mean(f1s):.3f}±{np.std(f1s):.3f} | {np.mean(rls):.3f}±{np.std(rls):.3f} | {np.mean(ppls):.2f}')
"
```

### 论文表格

| Method | EM ↑ | Token F1 ↑ | Mean JSD ↓ |
|--------|------|-----------|------------|
| Standard KD | x.xx ± x.xx | x.xx ± x.xx | x.xx ± x.xx |
| Reverse KL | x.xx ± x.xx | x.xx ± x.xx | x.xx ± x.xx |
| **SaGD (ours)** | **x.xx ± x.xx** | **x.xx ± x.xx** | **x.xx ± x.xx** |

---

## Phase 2: Saliency 诊断 + EC 分析（§4.2, §4.4）

**目的 1 (§4.2 动机)**: Standard KD 不保留 teacher 的 saliency 模式（JSD 高）。
**目的 2 (§4.4 EC 分析)**: SaGD 让 student 的 EC 更接近 teacher（不是更高，而是更接近——teacher 低 EC = 全局推理，StdKD 高 EC = shortcut，SaGD 接近 teacher = 保留推理模式）。

### 诊断（teacher 现算，不用 cache）

```bash
# Pretrained student（训练前的 baseline）
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

# 所有训练后的 checkpoint
for METHOD in standard_kd reverse_kl sagd; do
    for SEED in 42 123 456; do
        python scripts/diagnose_saliency.py \
            --teacher_model Qwen/Qwen3-8B \
            --student_ckpt outputs/${METHOD}/seed_${SEED}/student_final.pt \
            --dataset squad --subset val --max_samples 500 \
            --output_path outputs/${METHOD}/seed_${SEED}/saliency_diagnosis.json \
            --device cuda:0
    done
done
```

### 汇总 EC

```bash
python -c "
import json, numpy as np
print(f'{'Method':<15} | {'Mean JSD':>12} | {'Teacher EC':>12} | {'Student EC':>12}')
print('-' * 60)
for method in ['standard_kd', 'reverse_kl', 'sagd']:
    jsds, tecs, secs = [], [], []
    for seed in [42, 123, 456]:
        try:
            with open(f'outputs/{method}/seed_{seed}/saliency_diagnosis.json') as f:
                d = json.load(f)
            jsds.append(d['mean_jsd'])
            tecs.append(d['teacher_evidence_concentration'])
            secs.append(d['student_evidence_concentration'])
        except: pass
    if jsds:
        print(f'{method:<15} | {np.mean(jsds):.4f}±{np.std(jsds):.4f} | {np.mean(tecs):.4f} | {np.mean(secs):.4f}±{np.std(secs):.4f}')
"
```

### 论文内容

1. **§4.2 表**: Pretrained vs StdKD vs SaGD 的 Mean JSD
2. **§4.4 EC 柱状图**: Teacher EC vs StdKD Student EC vs SaGD Student EC
3. **§4.4 叙事**: Teacher 用全局 context 推理（低 EC），StdKD 走 shortcut（高 EC），SaGD 保留推理模式（EC 接近 teacher）

---

## Phase 3: 消融实验（§4.5）

**论文问题**: Noise KL 和 reweighting 各自贡献多少？

### 消融配置

| 配置名 | λ | σ | τ_w | 效果 | 理论空间 |
|--------|---|---|-----|------|---------|
| Standard KD | — | — | — | baseline | L² |
| + Noise KL only | 0.5 | 0.005 | 100.0 | τ_w≈∞ → 均匀权重 | W^{1,2} |
| + Reweight only | 0.0 | — | 1.0 | λ=0 → 无 noise KL | L² + DRO |
| **SaGD (full)** | 0.5 | 0.005 | 1.0 | 完整方法 | W^{1,2} + DRO |

### 训练

```bash
# Noise KL only
for SEED in 42 123 456; do
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_noise 0.5 --noise_sigma 0.005 --sagd_tau_w 100.0 --sagd_every_n_steps 5 \
        --seed $SEED --output_dir outputs_ablation/noise_only/ --epochs 3 \
        --device cuda:0
done

# Reweight only
for SEED in 42 123 456; do
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_noise 0.0 --noise_sigma 0.005 --sagd_tau_w 1.0 --sagd_every_n_steps 5 \
        --seed $SEED --output_dir outputs_ablation/reweight_only/ --epochs 3 \
        --device cuda:1
done
```

### 评测

```bash
for CONFIG in noise_only reweight_only; do
    for SEED in 42 123 456; do
        python scripts/evaluate.py \
            --student_ckpt outputs_ablation/${CONFIG}/sagd/seed_${SEED}/student_final.pt \
            --dataset squad --subset test \
            --output_path outputs_ablation/${CONFIG}/sagd/seed_${SEED}/eval_metrics.json \
            --device cuda:0
    done
done
```

### 论文表格

| Config | Noise KL | Reweight | EM ↑ | F1 ↑ | JSD ↓ |
|--------|:---:|:---:|:---:|:---:|:---:|
| Standard KD | — | — | x.xx | x.xx | x.xx |
| + Noise KL only | ✓ | — | x.xx | x.xx | x.xx |
| + Reweight only | — | ✓ | x.xx | x.xx | x.xx |
| **SaGD (full)** | ✓ | ✓ | **x.xx** | **x.xx** | **x.xx** |

---

## Phase 4: 超参 Sweep（Appendix）

**目的**: σ 的 sweet spot 验证 + λ/τ_w/N 的敏感性分析。epoch=1 足够看趋势。

```bash
# σ sweep（最关键）
for SIGMA in 0.001 0.002 0.005 0.01 0.02 0.05; do
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_noise 0.5 --noise_sigma $SIGMA --sagd_tau_w 1.0 --sagd_every_n_steps 5 \
        --seed 42 --output_dir outputs_sweep/sigma_${SIGMA}/ --epochs 1 \
        --device cuda:0
done

# λ sweep
for LAMBDA in 0.1 0.2 0.5 1.0 2.0; do
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_noise $LAMBDA --noise_sigma 0.005 --sagd_tau_w 1.0 --sagd_every_n_steps 5 \
        --seed 42 --output_dir outputs_sweep/lambda_${LAMBDA}/ --epochs 1 \
        --device cuda:1
done

# τ_w sweep
for TAU in 0.1 0.5 1.0 2.0 5.0 100.0; do
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_noise 0.5 --noise_sigma 0.005 --sagd_tau_w $TAU --sagd_every_n_steps 5 \
        --seed 42 --output_dir outputs_sweep/tau_${TAU}/ --epochs 1 \
        --device cuda:2
done

# N sweep
for N in 1 3 5 10 20; do
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path data/teacher_saliency_squad.pt \
        --lambda_noise 0.5 --noise_sigma 0.005 --sagd_tau_w 1.0 --sagd_every_n_steps $N \
        --seed 42 --output_dir outputs_sweep/every_n_${N}/ --epochs 1 \
        --device cuda:3
done
```

**论文**: 四个折线图（σ vs EM, λ vs EM, τ_w vs EM, N vs EM），放 Appendix。

---

## Phase 5: 训练动态（§4.6）

**无需额外训练**——从 Phase 1 的 3-epoch SaGD 训练 log 提取。

```bash
# Step-level dynamics
cat outputs/sagd/seed_42/training_stats.jsonl | python -c "
import sys, json
print('step\tloss\tkl_noisy\tkl_clean\tmean_jsd\tmax_weight')
for line in sys.stdin:
    d = json.loads(line)
    if 'sagd/kl_noisy' in d:
        print(f\"{d['step']}\t{d['loss']:.4f}\t{d['sagd/kl_noisy']:.4f}\t{d['sagd/kl_clean']:.4f}\t{d['sagd/mean_jsd']:.4f}\t{d['sagd/max_weight']:.2f}\")
" > outputs/sagd/seed_42/dynamics.tsv

# Epoch-level JSD + EC（需要 teacher 模型）
for METHOD in sagd standard_kd; do
    for EPOCH in 1 2 3; do
        python scripts/diagnose_saliency.py \
            --teacher_model Qwen/Qwen3-8B \
            --student_ckpt outputs/${METHOD}/seed_42/student_epoch${EPOCH}.pt \
            --dataset squad --subset val --max_samples 500 \
            --output_path outputs/${METHOD}/seed_42/saliency_epoch${EPOCH}.json \
            --device cuda:0
    done
done
```

### 论文图

1. **(kl_noisy - kl_clean) vs step**: Jacobian gap 代理，应随训练下降
2. **Mean JSD vs epoch**: SaGD 的 JSD 下降应快于 Standard KD
3. **Evidence Concentration vs epoch**: SaGD student EC 逐渐接近 teacher

---

## Phase 6: Dolly 泛化（§4.7）

**论文问题**: SaGD 是否泛化到非 extractive QA（instruction-following）任务？

**指标**: ROUGE-L（Dolly 标准）

```bash
for SEED in 42 123 456; do
    python scripts/train.py \
        --method standard_kd --dataset dolly \
        --seed $SEED --output_dir outputs_dolly/ --epochs 3 \
        --device cuda:0

    python scripts/train.py \
        --method sagd --dataset dolly \
        --teacher_saliency_path data/teacher_saliency_dolly.pt \
        --lambda_noise 0.5 --noise_sigma 0.005 --sagd_every_n_steps 5 --sagd_tau_w 1.0 \
        --seed $SEED --output_dir outputs_dolly/ --epochs 3 \
        --device cuda:1
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

### 论文表格

| Dataset | Method | Primary Metric ↑ |
|---------|--------|-------------------|
| SQuAD | Standard KD | EM: x.xx ± x.xx |
| SQuAD | **SaGD** | **EM: x.xx ± x.xx** |
| Dolly | Standard KD | ROUGE-L: x.xx ± x.xx |
| Dolly | **SaGD** | **ROUGE-L: x.xx ± x.xx** |

---

## Phase 7: 跨架构 LLaMA（§4.8）

**论文问题**: SaGD 是否泛化到不同模型架构？

**模型对**: LLaMA 3.1-8B (teacher) → LLaMA 3.1-1B (student)

```bash
# 预计算 LLaMA teacher saliency
python scripts/precompute_teacher_saliency.py \
    --model_name meta-llama/Llama-3.1-8B \
    --tokenizer_name meta-llama/Llama-3.1-1B \
    --dataset squad \
    --output_path data/teacher_saliency_llama_squad.pt \
    --batch_size 4 --max_seq_len 512 --device cuda:0

# 训练
for SEED in 42 123 456; do
    python scripts/train.py \
        --method standard_kd --dataset squad \
        --teacher_model meta-llama/Llama-3.1-8B --student_model meta-llama/Llama-3.1-1B \
        --seed $SEED --output_dir outputs_llama/ --epochs 3 \
        --device cuda:0

    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_model meta-llama/Llama-3.1-8B --student_model meta-llama/Llama-3.1-1B \
        --teacher_saliency_path data/teacher_saliency_llama_squad.pt \
        --lambda_noise 0.5 --noise_sigma 0.005 --sagd_every_n_steps 5 --sagd_tau_w 1.0 \
        --seed $SEED --output_dir outputs_llama/ --epochs 3 \
        --device cuda:1
done
```

### 论文表格

| Architecture | Method | EM ↑ | F1 ↑ |
|-------------|--------|------|------|
| Qwen3 8B→0.6B | Standard KD | x.xx | x.xx |
| Qwen3 8B→0.6B | **SaGD** | **x.xx** | **x.xx** |
| LLaMA 8B→1B | Standard KD | x.xx | x.xx |
| LLaMA 8B→1B | **SaGD** | **x.xx** | **x.xx** |

---

## Phase 8: Benchmark 防御（Appendix）

**目的**: SaGD 没有损害通用能力。

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

## 实验优先级

如果时间有限，按优先级排：

| 优先级 | Phase | 重要性 | 原因 |
|--------|-------|--------|------|
| **P0** | 1 (主实验 3ep) | 必须 | 没有这个表，论文不成立 |
| **P0** | 2 (诊断+EC) | 必须 | 核心 claim 的支撑（saliency 保留） |
| **P1** | 3 (消融) | 必须 | Reviewer 必问"两个组件各自多少" |
| **P1** | 6 (Dolly) | 重要 | 证明不只在 SQuAD 上有效 |
| **P2** | 4 (sweep) | Appendix | σ 的倒 U 型曲线有理论价值 |
| **P2** | 5 (动态) | 有价值 | 展示 Jacobian gap 下降过程 |
| **P3** | 7 (LLaMA) | 加分 | 跨架构泛化 |
| **P3** | 8 (benchmark) | 防御 | Reviewer 可能问但不必须 |

---

## 实验与论文章节的对应

| 论文章节 | Phase | 回答的问题 | 核心指标 |
|---------|-------|-----------|---------|
| §4.2 | 2 | Standard KD 保留 saliency 吗？ | Mean JSD |
| §4.3 | 1 | SaGD vs baselines 谁更好？ | EM, F1 |
| §4.4 | 2 | Student 保留了 teacher 推理模式吗？ | Evidence Concentration |
| §4.5 | 3 | 两个组件各自贡献多少？ | EM, F1 (ablation) |
| §4.6 | 5 | 训练中 Jacobian gap 如何变化？ | kl_noisy-kl_clean vs step |
| §4.7 | 6 | 非 QA 任务能泛化吗？ | ROUGE-L on Dolly |
| §4.8 | 7 | 跨架构能泛化吗？ | EM, F1 on LLaMA |
| Appendix | 4,8 | 超参敏感性 + 通用能力 | sweep 图 + MMLU/ARC |

---

## Smoke Test

```bash
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# 1. 单元测试
pytest tests/ -v

# 2. 快速验证
python scripts/train.py \
    --method sagd --dataset squad \
    --teacher_saliency_path data/teacher_saliency_squad.pt \
    --lambda_noise 0.5 --noise_sigma 0.005 \
    --epochs 1 --max_train_samples 200 \
    --device cuda:0
```
