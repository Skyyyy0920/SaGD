# SaGD 实验指南（对齐 DA-KD 实验设置）

本文档对应论文 §4 的全部实验。实验设置对齐 DA-KD (ICML 2025) 的评测体系，在其基础上加入 SaGD 特有的 saliency 分析。

**方法概要**: SaGD = 噪声 KL（隐式 Jacobian 匹配，position-adaptive）+ Saliency-guided reweighting（DRO）。

**确定的超参**: λ=0.5, σ=0.005 (relative to embed norm), τ_w=1.0, N=5

---

## 总览

```
Phase 0   预计算 teacher saliency                          1 GPU    ~3h      前置
Phase 1   主实验表 1: 指令跟随（Dolly 训练, 5 eval）         4 GPU    ~48h     §4.2 核心
Phase 2   主实验表 2: 任务特定（SAMSum + GSM8K + SQuAD）     4 GPU    ~36h     §4.3 核心
Phase 3   Saliency 诊断 + EC 分析                           1 GPU    ~4h      §4.4
Phase 4   消融实验                                          4 GPU    ~18h     §4.5
Phase 5   超参 sweep                                        4 GPU    ~17h     Appendix
Phase 6   训练动态                                          —        ~0h      §4.6（从 Phase 1 提取）
Phase 7   跨架构 LLaMA                                     4 GPU    ~12h     §4.7（Appendix）
Phase 8   Benchmark 防御                                    1 GPU    ~2h      Appendix
```

**总计**: ~140 GPU-hours, 4×A100 并行约 **~36h wall-clock**

**硬件**: 4× A100 80GB
**种子**: 42, 123, 456, 789, 2024（5 seeds，对齐 DA-KD）

---

## 模型设置

| 角色 | 模型 | 参数量 | HuggingFace ID |
|------|------|--------|----------------|
| Teacher | Qwen3-8B | 8B | `Qwen/Qwen3-8B` |
| Student (large) | Qwen3-1.7B | 1.7B | `Qwen/Qwen3-1.7B` |
| Student (small) | Qwen3-0.6B | 0.6B | `Qwen/Qwen3-0.6B` |

对应 DA-KD 的 Qwen2.5-7B → 1.5B/0.5B 设置。

---

## Baseline 方法

| 方法 | `--method` | 论文 | 关键特点 |
|------|------------|------|---------|
| SFT | `sft` | — | 无蒸馏，直接微调 |
| KD-KL | `standard_kd` | Hinton 2015 | Forward KL(P_T ‖ P_S) |
| KD-RKL | `reverse_kl` | Gu et al. 2024 (MiniLLM) | Reverse KL(P_S ‖ P_T) |
| SeqKD | `seqkd` | Kim & Rush 2016 | Teacher 生成输出 → Student SFT |
| GKD | `gkd` | Agarwal et al. 2023 | 广义 JSD divergence |
| DistiLLM | `distillm` | Ko et al. 2024 | Skew KL divergence |
| DA-KD | `dakd` | He et al. 2025 | BDL loss + DiffUp 数据选择 |
| **SaGD (ours)** | `sagd` | — | Noise KL + Saliency reweighting |

---

## 训练超参（固定）

| 参数 | 值 | 备注 |
|------|-----|------|
| Epochs | 10 | 对齐 DA-KD |
| Batch size | 8 | |
| Gradient accumulation | 4 | 有效 batch = 32 |
| Learning rate | 1e-5 | 对齐 DA-KD (AdamW + cosine) |
| Weight decay | 0.01 | |
| Warmup ratio | 0.03 | |
| Max grad norm | 1.0 | |
| Max sequence length | 512 | |
| KL temperature (T) | 2.0 | |
| fp16 | true | |

### 方法特定超参

| 方法 | 参数 | 值 |
|------|------|-----|
| GKD | `--gkd_beta` | 0.5 |
| DistiLLM | `--distillm_alpha` | 0.5 |
| DA-KD | `--bdl_lambda` | 0.9 |
| SaGD | `--lambda_noise` | 0.5 |
| SaGD | `--noise_sigma` | 0.005 |
| SaGD | `--sagd_tau_w` | 1.0 |
| SaGD | `--sagd_every_n_steps` | 5 |

---

## Phase 0: 预计算 Teacher Saliency（一次性）

**目的**: Teacher saliency 缓存用于 SaGD 训练时的 reweighting 和 adaptive noise。

**关键**: 必须与训练使用完全相同的 dataset, seed, max_seq_len, tokenizer, subset。

```bash
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Dolly（指令跟随训练集）
python scripts/precompute_teacher_saliency.py \
    --model_name Qwen/Qwen3-8B --dataset dolly \
    --output_path data/teacher_saliency_dolly.pt \
    --batch_size 4 --max_seq_len 512 --device cuda:0 &

# SQuAD
python scripts/precompute_teacher_saliency.py \
    --model_name Qwen/Qwen3-8B --dataset squad \
    --output_path data/teacher_saliency_squad.pt \
    --batch_size 4 --max_seq_len 512 --device cuda:1 &

# SAMSum
python scripts/precompute_teacher_saliency.py \
    --model_name Qwen/Qwen3-8B --dataset samsum \
    --output_path data/teacher_saliency_samsum.pt \
    --batch_size 4 --max_seq_len 512 --device cuda:2 &

# GSM8K
python scripts/precompute_teacher_saliency.py \
    --model_name Qwen/Qwen3-8B --dataset gsm8k \
    --output_path data/teacher_saliency_gsm8k.pt \
    --batch_size 4 --max_seq_len 512 --device cuda:3 &

wait
echo "Phase 0 done."
```

---

## Phase 1: 主实验表 1 — 指令跟随（§4.2）

**论文问题**: Task-agnostic instruction following 上各方法对比。

**训练集**: Dolly-15K
**评测集**: DollyEval, SelfInst, Super-Natural, Unnatural, VicunaEval
**指标**: ROUGE-L (每个 eval set + 平均)
**模型对**: Qwen3-8B → Qwen3-1.7B 和 Qwen3-0.6B

### 训练（8 方法 × 2 student × 5 seeds = 80 runs）

```bash
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

for STUDENT in "Qwen/Qwen3-1.7B" "Qwen/Qwen3-0.6B"; do
    STUDENT_TAG=$(echo $STUDENT | sed 's/Qwen\/Qwen3-/qwen3_/')

    for SEED in 42 123 456 789 2024; do
        # SFT
        python scripts/train.py \
            --method sft --dataset dolly \
            --student_model $STUDENT \
            --seed $SEED --output_dir outputs_dolly/${STUDENT_TAG}/ \
            --epochs 10 --lr 1e-5 --skip_eval \
            --device cuda:0 &

        # KD-KL (Standard KD)
        python scripts/train.py \
            --method standard_kd --dataset dolly \
            --student_model $STUDENT \
            --seed $SEED --output_dir outputs_dolly/${STUDENT_TAG}/ \
            --epochs 10 --lr 1e-5 --skip_eval \
            --device cuda:1 &

        # KD-RKL (Reverse KL)
        python scripts/train.py \
            --method reverse_kl --dataset dolly \
            --student_model $STUDENT \
            --seed $SEED --output_dir outputs_dolly/${STUDENT_TAG}/ \
            --epochs 10 --lr 1e-5 --skip_eval \
            --device cuda:2 &

        # SeqKD
        python scripts/train.py \
            --method seqkd --dataset dolly \
            --student_model $STUDENT \
            --seed $SEED --output_dir outputs_dolly/${STUDENT_TAG}/ \
            --epochs 10 --lr 1e-5 --skip_eval \
            --device cuda:3 &

        wait

        # GKD
        python scripts/train.py \
            --method gkd --dataset dolly \
            --student_model $STUDENT --gkd_beta 0.5 \
            --seed $SEED --output_dir outputs_dolly/${STUDENT_TAG}/ \
            --epochs 10 --lr 1e-5 --skip_eval \
            --device cuda:0 &

        # DistiLLM
        python scripts/train.py \
            --method distillm --dataset dolly \
            --student_model $STUDENT --distillm_alpha 0.5 \
            --seed $SEED --output_dir outputs_dolly/${STUDENT_TAG}/ \
            --epochs 10 --lr 1e-5 --skip_eval \
            --device cuda:1 &

        # DA-KD
        python scripts/train.py \
            --method dakd --dataset dolly \
            --student_model $STUDENT --bdl_lambda 0.9 \
            --seed $SEED --output_dir outputs_dolly/${STUDENT_TAG}/ \
            --epochs 10 --lr 1e-5 --skip_eval \
            --device cuda:2 &

        # SaGD (ours)
        python scripts/train.py \
            --method sagd --dataset dolly \
            --student_model $STUDENT \
            --teacher_saliency_path data/teacher_saliency_dolly.pt \
            --lambda_noise 0.5 --noise_sigma 0.005 --sagd_every_n_steps 5 --sagd_tau_w 1.0 \
            --seed $SEED --output_dir outputs_dolly/${STUDENT_TAG}/ \
            --epochs 10 --lr 1e-5 --skip_eval \
            --device cuda:3 &

        wait
    done
done
```

### 评测（5 benchmarks per model）

```bash
for STUDENT in "Qwen/Qwen3-1.7B" "Qwen/Qwen3-0.6B"; do
    STUDENT_TAG=$(echo $STUDENT | sed 's/Qwen\/Qwen3-/qwen3_/')

    for METHOD in sft standard_kd reverse_kl seqkd gkd distillm dakd sagd; do
        for SEED in 42 123 456 789 2024; do
            CKPT="outputs_dolly/${STUDENT_TAG}/${METHOD}/seed_${SEED}/student_final.pt"
            OUT_DIR="outputs_dolly/${STUDENT_TAG}/${METHOD}/seed_${SEED}"

            python scripts/evaluate_benchmarks.py \
                --student_model $STUDENT \
                --student_ckpt $CKPT \
                --output_path ${OUT_DIR}/benchmark_rouge.json \
                --device cuda:0
        done
    done
done
```

### 汇总

```bash
python -c "
import json, numpy as np, os

students = [('qwen3_1.7B', 'Qwen3-1.7B'), ('qwen3_0.6B', 'Qwen3-0.6B')]
methods = ['sft', 'standard_kd', 'reverse_kl', 'seqkd', 'gkd', 'distillm', 'dakd', 'sagd']
benchmarks = ['dolly_eval', 'self_inst', 'super_natural', 'unnatural', 'vicuna_eval']
seeds = [42, 123, 456, 789, 2024]

for tag, name in students:
    print(f'\n=== {name} ===')
    header = f\"{'Method':<15}\" + ''.join(f' | {b:>15}' for b in benchmarks) + ' | Avg.'
    print(header)
    print('-' * len(header))
    for method in methods:
        bench_scores = {b: [] for b in benchmarks}
        for seed in seeds:
            path = f'outputs_dolly/{tag}/{method}/seed_{seed}/benchmark_rouge.json'
            try:
                with open(path) as f:
                    data = json.load(f)
                for b in benchmarks:
                    if b in data:
                        bench_scores[b].append(data[b]['rouge_l_f'])
            except: pass
        row = f'{method:<15}'
        avg_all = []
        for b in benchmarks:
            if bench_scores[b]:
                m = np.mean(bench_scores[b])
                row += f' | {m:>15.2f}'
                avg_all.append(m)
            else:
                row += f' | {\"—\":>15}'
        if avg_all:
            row += f' | {np.mean(avg_all):.2f}'
        print(row)
"
```

### 论文表格（Table 1: Task-Agnostic Instruction Following）

| Model | #Params | Method | DollyEval | SelfInst | Super-Natural | Unnatural | VicunaEval | Avg. |
|-------|---------|--------|-----------|----------|---------------|-----------|------------|------|
| Qwen3 | 8B | Teacher | — | — | — | — | — | — |
| | 1.7B | SFT | | | | | | |
| | | KD-KL | | | | | | |
| | | KD-RKL | | | | | | |
| | | SeqKD | | | | | | |
| | | GKD | | | | | | |
| | | DistiLLM | | | | | | |
| | | DA-KD | | | | | | |
| | | **SaGD** | | | | | | |
| | 0.6B | SFT | | | | | | |
| | | KD-KL | | | | | | |
| | | KD-RKL | | | | | | |
| | | SeqKD | | | | | | |
| | | GKD | | | | | | |
| | | DistiLLM | | | | | | |
| | | DA-KD | | | | | | |
| | | **SaGD** | | | | | | |

---

## Phase 2: 主实验表 2 — 任务特定（§4.3）

**论文问题**: SaGD 在任务特定场景（摘要、数学推理、抽取式 QA）的表现。

**数据集 & 指标**:
- SAMSum: ROUGE-L (文本摘要)
- GSM8K: Zero-shot Accuracy (数学推理)
- SQuAD 2.0: EM, Token F1, PPL (抽取式 QA)

### 训练

```bash
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

for DATASET in samsum gsm8k squad; do
    SALIENCY_PATH="data/teacher_saliency_${DATASET}.pt"

    for SEED in 42 123 456 789 2024; do
        for METHOD in sft standard_kd reverse_kl seqkd gkd distillm dakd sagd; do
            EXTRA_ARGS=""

            case $METHOD in
                gkd) EXTRA_ARGS="--gkd_beta 0.5" ;;
                distillm) EXTRA_ARGS="--distillm_alpha 0.5" ;;
                dakd) EXTRA_ARGS="--bdl_lambda 0.9" ;;
                sagd) EXTRA_ARGS="--teacher_saliency_path $SALIENCY_PATH --lambda_noise 0.5 --noise_sigma 0.005 --sagd_every_n_steps 5 --sagd_tau_w 1.0" ;;
            esac

            # Qwen3-1.7B
            python scripts/train.py \
                --method $METHOD --dataset $DATASET \
                --student_model Qwen/Qwen3-1.7B $EXTRA_ARGS \
                --seed $SEED --output_dir outputs_task/qwen3_1.7B/${DATASET}/ \
                --epochs 10 --lr 1e-5 --skip_eval \
                --device cuda:0 &

            # Qwen3-0.6B
            python scripts/train.py \
                --method $METHOD --dataset $DATASET \
                --student_model Qwen/Qwen3-0.6B $EXTRA_ARGS \
                --seed $SEED --output_dir outputs_task/qwen3_0.6B/${DATASET}/ \
                --epochs 10 --lr 1e-5 --skip_eval \
                --device cuda:1 &

            wait
        done
    done
done
```

### 评测

```bash
for STUDENT in "Qwen/Qwen3-1.7B" "Qwen/Qwen3-0.6B"; do
    STUDENT_TAG=$(echo $STUDENT | sed 's/Qwen\/Qwen3-/qwen3_/')

    for DATASET in samsum gsm8k squad; do
        MAX_NEW=256
        [ "$DATASET" = "squad" ] && MAX_NEW=32

        for METHOD in sft standard_kd reverse_kl seqkd gkd distillm dakd sagd; do
            for SEED in 42 123 456 789 2024; do
                CKPT="outputs_task/${STUDENT_TAG}/${DATASET}/${METHOD}/seed_${SEED}/student_final.pt"
                python scripts/evaluate.py \
                    --student_model $STUDENT \
                    --student_ckpt $CKPT \
                    --dataset $DATASET --subset test \
                    --max_new_tokens $MAX_NEW \
                    --output_path outputs_task/${STUDENT_TAG}/${DATASET}/${METHOD}/seed_${SEED}/eval_metrics.json \
                    --skip_bertscore \
                    --device cuda:0
            done
        done
    done
done
```

### 汇总

```bash
python -c "
import json, numpy as np

students = [('qwen3_1.7B', '1.7B'), ('qwen3_0.6B', '0.6B')]
methods = ['sft', 'standard_kd', 'reverse_kl', 'seqkd', 'gkd', 'distillm', 'dakd', 'sagd']
seeds = [42, 123, 456, 789, 2024]

for tag, name in students:
    print(f'\n=== Qwen3-{name} ===')
    print(f\"{'Method':<15} | {'SAMSum RL':>10} | {'GSM8K Acc':>10} | {'SQuAD EM':>10} | {'SQuAD F1':>10} | {'SQuAD PPL':>10}\")
    print('-' * 80)
    for method in methods:
        samsum_rl, gsm_acc, squad_em, squad_f1, squad_ppl = [], [], [], [], []
        for seed in seeds:
            for ds, lst, key in [
                ('samsum', samsum_rl, 'rouge_l_f'),
                ('gsm8k', gsm_acc, 'gsm8k_accuracy'),
                ('squad', squad_em, 'exact_match'),
                ('squad', squad_f1, 'token_f1'),
                ('squad', squad_ppl, 'perplexity'),
            ]:
                try:
                    with open(f'outputs_task/{tag}/{ds}/{method}/seed_{seed}/eval_metrics.json') as f:
                        m = json.load(f)
                    lst.append(m[key])
                except: pass
        def fmt(lst):
            return f'{np.mean(lst):.3f}' if lst else '—'
        print(f'{method:<15} | {fmt(samsum_rl):>10} | {fmt(gsm_acc):>10} | {fmt(squad_em):>10} | {fmt(squad_f1):>10} | {fmt(squad_ppl):>10}')
"
```

### 论文表格（Table 2: Task-Specific Results）

| Model | Method | SAMSum (ROUGE-L) | GSM8K (Acc) | SQuAD (EM) | SQuAD (F1) | SQuAD (PPL) |
|-------|--------|-----------------|-------------|------------|------------|-------------|
| Qwen3-8B | Teacher | — | — | — | — | — |
| Qwen3-1.7B | SFT | | | | | |
| | KD-KL | | | | | |
| | KD-RKL | | | | | |
| | SeqKD | | | | | |
| | GKD | | | | | |
| | DistiLLM | | | | | |
| | DA-KD | | | | | |
| | **SaGD** | | | | | |
| Qwen3-0.6B | SFT | | | | | |
| | ... | | | | | |
| | **SaGD** | | | | | |

---

## Phase 3: Saliency 诊断 + EC 分析（§4.4）

**目的**: 验证 SaGD 保留 teacher 推理模式（saliency 分布接近 teacher），Standard KD 走 shortcut。

**指标**: Mean JSD（saliency 忠诚度）, Evidence Concentration（SQuAD only）

```bash
# Pretrained student baseline
for STUDENT in "Qwen/Qwen3-1.7B" "Qwen/Qwen3-0.6B"; do
    STUDENT_TAG=$(echo $STUDENT | sed 's/Qwen\/Qwen3-/qwen3_/')
    python -c "
from sagd.models import load_student; import torch
student, _ = load_student('$STUDENT', 'cpu')
torch.save(student.state_dict(), 'outputs_saliency/${STUDENT_TAG}_pretrained.pt')
"
    python scripts/diagnose_saliency.py \
        --teacher_model Qwen/Qwen3-8B \
        --student_model $STUDENT \
        --student_ckpt outputs_saliency/${STUDENT_TAG}_pretrained.pt \
        --dataset squad --subset val --max_samples 500 \
        --output_path outputs_saliency/${STUDENT_TAG}_pretrained_diag.json \
        --device cuda:0
done

# Trained checkpoints
for STUDENT_TAG in qwen3_1.7B qwen3_0.6B; do
    for METHOD in standard_kd reverse_kl sagd dakd; do
        for SEED in 42 123 456; do
            STUDENT_MODEL="Qwen/Qwen3-$(echo $STUDENT_TAG | sed 's/qwen3_//')"
            python scripts/diagnose_saliency.py \
                --teacher_model Qwen/Qwen3-8B \
                --student_model $STUDENT_MODEL \
                --student_ckpt outputs_task/${STUDENT_TAG}/squad/${METHOD}/seed_${SEED}/student_final.pt \
                --dataset squad --subset val --max_samples 500 \
                --output_path outputs_saliency/${STUDENT_TAG}/${METHOD}_seed${SEED}_diag.json \
                --device cuda:0
        done
    done
done
```

### 论文内容

1. **§4.4 表**: Mean JSD comparison (Pretrained → StdKD → DA-KD → SaGD)
2. **§4.4 EC 柱状图**: Teacher EC vs StdKD EC vs DA-KD EC vs SaGD EC
3. **叙事**: Teacher 用全局 context 推理（低 EC），StdKD 走 shortcut（高 EC），SaGD 保留推理模式

---

## Phase 4: 消融实验（§4.5）

**论文问题**: SaGD 的两个组件（Noise KL + Reweighting）各自贡献多少？

### 消融配置

| 配置名 | λ | σ | τ_w | 效果 | 理论空间 |
|--------|---|---|-----|------|---------|
| Standard KD | — | — | — | baseline | L² |
| + Noise KL only | 0.5 | 0.005 | 100.0 | τ_w≈∞ → 均匀权重 | W^{1,2} |
| + Reweight only | 0.0 | — | 1.0 | λ=0 → 无 noise KL | L² + DRO |
| **SaGD (full)** | 0.5 | 0.005 | 1.0 | 完整方法 | W^{1,2} + DRO |

### 训练（在 SQuAD 和 Dolly 上各做）

```bash
for DATASET in squad dolly; do
    SALIENCY_PATH="data/teacher_saliency_${DATASET}.pt"

    for SEED in 42 123 456; do
        # Noise KL only
        python scripts/train.py \
            --method sagd --dataset $DATASET \
            --teacher_saliency_path $SALIENCY_PATH \
            --lambda_noise 0.5 --noise_sigma 0.005 --sagd_tau_w 100.0 --sagd_every_n_steps 5 \
            --seed $SEED --output_dir outputs_ablation/${DATASET}/noise_only/ \
            --epochs 10 --lr 1e-5 --skip_eval \
            --device cuda:0

        # Reweight only
        python scripts/train.py \
            --method sagd --dataset $DATASET \
            --teacher_saliency_path $SALIENCY_PATH \
            --lambda_noise 0.0 --noise_sigma 0.005 --sagd_tau_w 1.0 --sagd_every_n_steps 5 \
            --seed $SEED --output_dir outputs_ablation/${DATASET}/reweight_only/ \
            --epochs 10 --lr 1e-5 --skip_eval \
            --device cuda:1
    done
done
```

### 论文表格

| Config | Noise KL | Reweight | EM ↑ | F1 ↑ | ROUGE-L ↑ | PPL ↓ |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Standard KD | — | — | | | | |
| + Noise KL only | ✓ | — | | | | |
| + Reweight only | — | ✓ | | | | |
| **SaGD (full)** | ✓ | ✓ | | | | |

---

## Phase 5: 超参 Sweep（Appendix）

**目的**: σ 的 sweet spot + λ/τ_w/N 的敏感性分析。

```bash
SALIENCY_PATH="data/teacher_saliency_squad.pt"

# σ sweep
for SIGMA in 0.001 0.002 0.005 0.01 0.02 0.05; do
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path $SALIENCY_PATH \
        --lambda_noise 0.5 --noise_sigma $SIGMA --sagd_tau_w 1.0 --sagd_every_n_steps 5 \
        --seed 42 --output_dir outputs_sweep/sigma_${SIGMA}/ \
        --epochs 3 --lr 1e-5 \
        --device cuda:0
done

# λ sweep
for LAMBDA in 0.1 0.2 0.5 1.0 2.0; do
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path $SALIENCY_PATH \
        --lambda_noise $LAMBDA --noise_sigma 0.005 --sagd_tau_w 1.0 --sagd_every_n_steps 5 \
        --seed 42 --output_dir outputs_sweep/lambda_${LAMBDA}/ \
        --epochs 3 --lr 1e-5 \
        --device cuda:1
done

# τ_w sweep
for TAU in 0.1 0.5 1.0 2.0 5.0 100.0; do
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path $SALIENCY_PATH \
        --lambda_noise 0.5 --noise_sigma 0.005 --sagd_tau_w $TAU --sagd_every_n_steps 5 \
        --seed 42 --output_dir outputs_sweep/tau_${TAU}/ \
        --epochs 3 --lr 1e-5 \
        --device cuda:2
done

# N sweep
for N in 1 3 5 10 20; do
    python scripts/train.py \
        --method sagd --dataset squad \
        --teacher_saliency_path $SALIENCY_PATH \
        --lambda_noise 0.5 --noise_sigma 0.005 --sagd_tau_w 1.0 --sagd_every_n_steps $N \
        --seed 42 --output_dir outputs_sweep/every_n_${N}/ \
        --epochs 3 --lr 1e-5 \
        --device cuda:3
done
```

---

## Phase 6: 训练动态（§4.6）

**无需额外训练** — 从 Phase 2 的 SQuAD 训练 log 提取。

```bash
# Step-level dynamics
cat outputs_task/qwen3_0.6B/squad/sagd/seed_42/training_stats.jsonl | python -c "
import sys, json
print('step\tloss\tkl_noisy\tkl_clean\tmean_jsd\tmax_weight')
for line in sys.stdin:
    d = json.loads(line)
    if 'sagd/kl_noisy' in d:
        print(f\"{d['step']}\t{d['loss']:.4f}\t{d['sagd/kl_noisy']:.4f}\t{d['sagd/kl_clean']:.4f}\t{d['sagd/mean_jsd']:.4f}\t{d['sagd/max_weight']:.2f}\")
" > outputs_dynamics.tsv

# Epoch-level JSD + EC
for METHOD in sagd standard_kd dakd; do
    for EPOCH in 1 3 5 10; do
        python scripts/diagnose_saliency.py \
            --teacher_model Qwen/Qwen3-8B \
            --student_ckpt outputs_task/qwen3_0.6B/squad/${METHOD}/seed_42/student_epoch${EPOCH}.pt \
            --dataset squad --subset val --max_samples 500 \
            --output_path outputs_dynamics/${METHOD}_epoch${EPOCH}.json \
            --device cuda:0
    done
done
```

### 论文图

1. **(kl_noisy - kl_clean) vs step**: Jacobian gap proxy 应随训练下降
2. **Mean JSD vs epoch**: SaGD 下降应快于 Standard KD 和 DA-KD
3. **Evidence Concentration vs epoch**: SaGD student EC 逐渐接近 teacher

---

## Phase 7: 跨架构 LLaMA（Appendix §4.7）

**模型对**: LLaMA 3.1-8B → LLaMA 3.1-1B

```bash
# 预计算
python scripts/precompute_teacher_saliency.py \
    --model_name meta-llama/Llama-3.1-8B \
    --tokenizer_name meta-llama/Llama-3.1-1B \
    --dataset squad \
    --output_path data/teacher_saliency_llama_squad.pt \
    --batch_size 4 --max_seq_len 512 --device cuda:0

# 训练
for SEED in 42 123 456; do
    for METHOD in standard_kd reverse_kl dakd sagd; do
        EXTRA_ARGS=""
        case $METHOD in
            dakd) EXTRA_ARGS="--bdl_lambda 0.9" ;;
            sagd) EXTRA_ARGS="--teacher_saliency_path data/teacher_saliency_llama_squad.pt --lambda_noise 0.5 --noise_sigma 0.005 --sagd_every_n_steps 5 --sagd_tau_w 1.0" ;;
        esac

        python scripts/train.py \
            --method $METHOD --dataset squad \
            --teacher_model meta-llama/Llama-3.1-8B --student_model meta-llama/Llama-3.1-1B \
            $EXTRA_ARGS \
            --seed $SEED --output_dir outputs_llama/ --epochs 10 --lr 1e-5 \
            --device cuda:0
    done
done
```

---

## Phase 8: Benchmark 防御（Appendix）

**目的**: SaGD 没有损害通用能力（MMLU, ARC-Challenge, TruthfulQA）。

```bash
pip install lm-eval

for METHOD in standard_kd sagd dakd; do
    python -c "
from sagd.models import load_student; import torch
student, _ = load_student('Qwen/Qwen3-0.6B', 'cpu')
student.load_state_dict(torch.load('outputs_task/qwen3_0.6B/squad/${METHOD}/seed_42/student_final.pt', map_location='cpu', weights_only=True))
student.save_pretrained('outputs_benchmark/${METHOD}/hf_model/')
"
    lm_eval --model hf \
        --model_args pretrained=outputs_benchmark/${METHOD}/hf_model/ \
        --tasks mmlu,arc_challenge,truthfulqa_mc2 \
        --batch_size 8 --output_path outputs_benchmark/${METHOD}/

done

# Base student (no distillation)
lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen3-0.6B \
    --tasks mmlu,arc_challenge,truthfulqa_mc2 \
    --batch_size 8 --output_path outputs_benchmark/base/
```

---

## 实验优先级

| 优先级 | Phase | 重要性 | 原因 |
|--------|-------|--------|------|
| **P0** | 1 (指令跟随 Table 1) | 必须 | DA-KD 对齐的核心实验 |
| **P0** | 2 (任务特定 Table 2) | 必须 | SAMSum + GSM8K + SQuAD 多任务验证 |
| **P1** | 3 (Saliency + EC) | 必须 | SaGD 核心 claim 的支撑 |
| **P1** | 4 (消融) | 必须 | Reviewer 必问 |
| **P2** | 5 (sweep) | Appendix | σ 的倒 U 型曲线有理论价值 |
| **P2** | 6 (动态) | 有价值 | 展示 Jacobian gap 下降过程 |
| **P3** | 7 (LLaMA) | 加分 | 跨架构泛化 |
| **P3** | 8 (benchmark) | 防御 | Reviewer 可能问但不必须 |

---

## 实验与论文章节的对应

| 论文章节 | Phase | 回答的问题 | 核心指标 |
|---------|-------|-----------|---------|
| §4.1 | — | Setup | — |
| §4.2 | 1 | 指令跟随任务上各方法对比？ | ROUGE-L × 5 benchmarks |
| §4.3 | 2 | 任务特定场景（摘要/推理/QA）？ | ROUGE-L, Acc, EM, F1, PPL |
| §4.4 | 3 | Student 保留 teacher 推理模式吗？ | Mean JSD, Evidence Concentration |
| §4.5 | 4 | 两个组件各自贡献多少？ | EM, F1, ROUGE-L, PPL |
| §4.6 | 6 | 训练中 Jacobian gap 如何变化？ | kl_noisy-kl_clean vs step |
| §4.7 | 7 | 跨架构能泛化吗？ | EM, F1 on LLaMA |
| Appendix | 5,8 | 超参敏感性 + 通用能力 | sweep 图 + MMLU/ARC |

---

## 与 DA-KD (ICML 2025) 的关键差异

| 维度 | DA-KD | SaGD (ours) |
|------|-------|-------------|
| 核心 loss | BDL (混合分布的 KL) | Noise KL (隐式 Jacobian 匹配) |
| 困难样本策略 | DiffUp (按 DDS 筛数据) | Saliency reweighting (DRO) |
| 效率优势 | 更少迭代（数据筛选） | 保持全数据但加权 |
| 理论基础 | 梯度分析（C(x)有界） | Sobolev 范数 + Taylor 展开 |
| 独特指标 | 无 | Evidence Concentration |
| Teacher 利用 | 仅输出分布 | 输出分布 + saliency (一阶信息) |

---

## Smoke Test

```bash
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# 1. 单元测试
pytest tests/ -v

# 2. 快速验证各方法可运行
for METHOD in sft standard_kd reverse_kl seqkd gkd distillm dakd sagd; do
    EXTRA=""
    case $METHOD in
        gkd) EXTRA="--gkd_beta 0.5" ;;
        distillm) EXTRA="--distillm_alpha 0.5" ;;
        dakd) EXTRA="--bdl_lambda 0.9" ;;
        sagd) EXTRA="--teacher_saliency_path data/teacher_saliency_squad.pt --lambda_noise 0.5 --noise_sigma 0.005" ;;
    esac
    echo "=== Testing $METHOD ==="
    python scripts/train.py \
        --method $METHOD --dataset squad \
        $EXTRA \
        --epochs 1 --max_train_samples 50 \
        --device cuda:0 --skip_eval
done
```
