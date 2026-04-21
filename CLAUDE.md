# CLAUDE.md — SaGD

This file is the authoritative guide for AI assistants working on this codebase.
Read it completely before making any changes.

---

## 1. Project Overview

**Research goal**: Knowledge distillation from large to small LLMs.
**Method**: SaGD (Saliency-Guided Knowledge Distillation).
**Paper title**: *Saliency-Guided Knowledge Distillation: A Sobolev Perspective on Teaching Students Where to Look*

### Model pairs
- Primary: Qwen3-8B (teacher) → Qwen3-1.7B (student) and Qwen3-0.6B (student)
- Secondary (cross-architecture): LLaMA 3.1-8B → LLaMA 3.1-1B

### Datasets & evaluation (aligned with DA-KD, ICML 2025)

**Task-Agnostic Instruction Following** (Table 1):
- Training: Dolly-15K (`databricks/databricks-dolly-15k`)
- Evaluation: DollyEval, SelfInst, Super-Natural, Unnatural, VicunaEval
- Metric: ROUGE-L on each eval set + average
- 5 random seeds for statistical significance

**Task-Specific** (Table 2):
- **SAMSum** (`samsum`) — dialogue summarization, ROUGE-L
- **GSM8K** (`openai/gsm8k`) — mathematical reasoning, zero-shot accuracy
- **SQuAD 2.0** (`rajpurkar/squad_v2`) — extractive QA
  - Answerable subset only (~86K train, ~5.9K val)
  - Metrics: EM, Token F1, PPL
  - Saliency metric: Evidence Concentration (fraction of saliency mass on answer span)
  - Answer span token positions tracked for evidence concentration evaluation

**Dolly-15K** data splits:
- shuffle(seed=42), max_seq_len=512
- Train subset: first N-1000 samples (~14K)
- Val subset: next 500 samples
- Test subset: last 500 samples

**Benchmark defense**: MMLU, ARC-Challenge, TruthfulQA (lm-eval-harness, appendix only)

### Environment
- Hardware: 4× A100 80GB
- Python 3.10, PyTorch 2.4, Hugging Face Transformers

---

## 2. Method

### 2.1 Theory

Standard KD minimizes $D_\text{KL}(f_T(x) \| f_S(x))$ at each training point — this is
zero-order (function value) matching in L² sense. By Taylor expansion, the error at a
perturbed input $x + \delta$ ($\|\delta\| \leq \epsilon$) decomposes as:

$$D_\text{KL}(f_T(x+\delta) \| f_S(x+\delta)) \leq \underbrace{D_\text{KL}(f_T(x) \| f_S(x))}_\text{zero-order} + \epsilon \cdot \underbrace{\|J_T(x) - J_S(x)\|_F}_\text{first-order: Jacobian gap} + O(\epsilon^2)$$

Standard KD does not constrain the Jacobian, so even perfect pointwise matching provides
no guarantee in the input neighborhood. SaGD adds first-order matching, upgrading the
approximation quality from L² to Sobolev W^{1,2}.

Since $\epsilon^1 \gg \epsilon^2 \gg \cdots$ for $\epsilon < 1$, first-order is the
highest-ROI additional signal — higher-order terms decay rapidly.

The full Jacobian $J \in \mathbb{R}^{V \times (L \cdot d)}$ is intractable for LLMs.
Instead of explicitly computing and aligning it, we use **noise-based implicit Jacobian
matching** (Srinivas & Fleuret, ICML 2018):

$$\mathbb{E}_\xi[D_\text{KL}(f_T(x+\xi) \| f_S(x+\xi))] = D_\text{KL}(f_T(x) \| f_S(x)) + \sigma^2 \|J_T(x) - J_S(x)\|_F^2 + O(\sigma^4)$$

By computing KL on noise-perturbed embeddings, we implicitly match the **full Jacobian**
without ever computing it. This is an exact equality (not a bound), and requires no
second-order gradients, no flash attention workarounds.

**Note**: The equality above was proven by Srinivas for MSE loss on classification tasks.
For **autoregressive token-level KL** (our setting), we derive an analogous but
non-identical result with an additional residual term:

For sequence-level KL $D(x) = \sum_{t \in \mathcal{R}} D_t(x)$ where
$D_t = \sum_v p_T^v(x,t) \log(p_T^v / p_S^v)$, Taylor expansion gives:

$$\mathbb{E}_\xi[D(x+\xi)] = D(x) + \frac{\sigma^2}{2}\sum_t \left[\mathcal{F}_t(x) + R_t(x)\right] + O(\sigma^4)$$

where $\mathcal{F}_t(x) = \sum_v p_T^v \|\nabla_x \log p_T^v - \nabla_x \log p_S^v\|^2$
is the **Fisher-weighted score matching** term (dominant, captures Jacobian difference),
and $R_t(x) = \sum_v \Delta p_T^v(1+\ell_v) - \sum_v (p_T^v/p_S^v)\Delta p_S^v$ is a
residual satisfying $R_t = O(\|p_T - p_S\|)$ (vanishes as distillation converges).

Derivation: Leibniz rule on $D_t = \sum_v p_T^v \ell_v$ gives
$\Delta D_t = \sum_v [p_T^v \Delta\ell_v + 2\nabla p_T^v \cdot \nabla\ell_v + \ell_v \Delta p_T^v]$.
The cross term $2\nabla p_T^v \cdot \nabla\ell_v$ plus part of $p_T^v \Delta\ell_v$ combine
into the perfect square $\mathcal{F}_t$; remaining terms form $R_t$.

**Neighborhood guarantee** (Theorem): For $\lambda \geq \epsilon^2/\sigma^2$ and $\sigma \geq \epsilon$:

$$\max_i \sup_{\|\delta\| \leq \epsilon} D(x_i+\delta) \leq \max_i \left[D(x_i) + \lambda \mathbb{E}_\xi[D(x_i+\xi)]\right] + O(\epsilon^3)$$

Standard KD ($\lambda=0$): $D(x_i) \to 0$ does not control neighborhood error.
SaGD ($\lambda>0$): loss $\to 0$ requires both $D \to 0$ and $\mathbb{E}_\xi[D(x+\xi)] \to 0$,
the latter implying $\Delta D \to 0$ via the expansion, thus controlling neighborhood error.

**Input saliency** $s_i = \|\partial \log P / \partial e_i\|$ is used separately for the
**reweighting** component (not for the alignment loss). It compresses the Jacobian to
per-position scalars for computing sample difficulty (JSD divergence).

Saliency also guides **position-adaptive noise** allocation:
$\sigma_j = \sigma \cdot \|e\| \cdot \max(|s_{T,j} - s_{S,j}|, \delta) / \overline{\max(|s_T - s_S|, \delta)}$,
where $\sigma$ is a **relative** parameter (fraction of embedding norm $\|e\|$),
concentrating perturbation where teacher/student disagree most. Under minimax optimality
on the per-position Jacobian gap (with linear gap-reduction approximation), this is the
optimal allocation for a fixed total noise budget. The $\delta$ floor ensures every
position receives some noise even when saliency is perfectly aligned. The adaptive ratio
is clamped to [δ, 5×] to prevent numerical instability.

### 2.2 Complete Loss

$$\mathcal{L}_\text{SaGD} = \sum_{i=1}^B w_i \cdot \left[ \underbrace{D_\text{KL}(f_T(x_i) \| f_S(x_i))}_\text{clean KL (zero-order)} + \lambda \cdot \underbrace{D_\text{KL}(f_T(x_i + \xi_i) \| f_S(x_i + \xi_i))}_\text{noise KL (implicit first-order)} \right]$$

where $\xi_i \sim \mathcal{N}(0, \sigma^2 I)$ is Gaussian noise on embeddings.

Sample weights (mean-normalized to 1):
$$w_i = \frac{\exp(\text{JSD}_i / \tau_w)}{\frac{1}{B}\sum_j \exp(\text{JSD}_j / \tau_w)}$$

where $\text{JSD}_i = \text{JSD}(\hat{s}_T^i, \hat{s}_S^i)$, $\hat{s} = \text{softmax}(s/\tau_s)$.

Reweighting is derived from entropy-regularized minimax on sample vulnerability
$c_i = D(x_i) + \frac{\epsilon^2}{2}\sum_t \mathcal{F}_t(x_i)$: minimizing worst-case
neighborhood error with entropy regularization yields $w_i^* = \text{softmax}(c_i/\tau_w)$.
JSD is used as a computationally tractable proxy for $c_i$ (not an equivalence).
$\tau_w$ controls DRO strength: $\tau_w \to 0$ = hard example mining, $\tau_w \to \infty$ = uniform.

**Two components and why both are needed**:
- **Noise KL (implicit Jacobian matching)**: Adds Gaussian noise to embeddings and computes
  KL on the perturbed input. Implicitly matches the full Jacobian $\|J_T - J_S\|_F^2$ —
  no information loss, no second-order derivatives needed.
- **Saliency-guided reweighting**: Concentrates zero-order (KL) optimization on samples where
  teacher/student attend to different input tokens. Corresponds to distributionally robust
  optimization (DRO) — prioritizing samples with the loosest neighborhood error bounds.
- Neither alone achieves both effects: noise KL tightens bounds uniformly, reweighting
  allocates more training budget to the worst-case samples.

### 2.3 Saliency Computation

For a sample with `input_ids`, `attention_mask`, `labels_mask` (0=prompt, 1=response):

```
1. Embed: embed = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
   → Creates a leaf tensor disconnected from model parameters.

2. Forward: logits = model(inputs_embeds=embed, attention_mask=...).logits
   → Runs through the full model, but the computation graph starts at embed.

3. Response log-prob:
   - Shift alignment: logit[j] predicts token[j+1], so use logits[:,:-1] with input_ids[:,1:]
   - Mask: only sum log-probs at response positions (labels_mask[:,1:])
   - response_ll = (token_log_probs * shifted_response_mask).sum()

4. Backward: response_ll.backward()
   → Gradients flow back to embed only (not to model parameters).
   → CRITICAL: must temporarily set all model parameters to requires_grad=False
     before this backward, otherwise gradients accumulate into W_q, W_k, etc.

5. Saliency: saliency = embed.grad.norm(dim=-1)  # (B, L)
   → Mask to keep only prompt positions: multiply by (1-labels_mask) * attention_mask
   → CRITICAL: must include attention_mask to exclude padding positions.
```

**Teacher saliency** is precomputed once (teacher is frozen) and cached to disk.
**Student saliency** is computed every N training steps (for reweighting only).

Saliency is used ONLY for the reweighting component (computing JSD to determine sample
weights). The first-order matching is handled by noise KL, not by saliency alignment.

`compute()`: Non-differentiable. Used for teacher precomputation, diagnosis, and
reweighting signal. Returns detached tensor. No gradients flow to model parameters.

`compute_differentiable()`: Retained for research/analysis but NOT used in training.
The noise KL approach replaces the need for differentiable saliency.

### 2.4 Per-Sample KL

```
1. Compute per-position KL: per_pos = (t_probs * (t_log - s_log)).sum(dim=-1)  # (B, L)
2. Shift alignment: per_pos_shifted = per_pos[:, :-1] with mask = labels_mask[:, 1:]
   → logit[j] predicts token[j+1], so KL at position j uses labels_mask of position j+1
3. Per-sample mean: sum over masked positions, divide by mask count
4. Scale by T²
```

### 2.5 Training Flow

```
Pre-training (once):
  precompute_teacher_saliency.py → data/teacher_saliency_squad.pt

Each training step:
  1. Load batch (input_ids, attention_mask, labels_mask, index)
  2. Teacher forward → t_logits  (under torch.no_grad)
  3. Student forward → s_logits

  if method == "sagd" AND global_step % N == 0:
    4. per_sample_kl = compute_per_sample_kl(t_logits, s_logits, labels_mask)
    5. student_sal = saliency_computer.compute(student, ...)  [non-differentiable]
    6. teacher_sal = get_cached_teacher_saliency(batch["index"])
    7. Compute shared per-position noise scale: σ_j ∝ max(|s_T,j - s_S,j|, δ)
    8. t_embed = teacher.get_input_embeddings()(input_ids).detach()
       s_embed = student.get_input_embeddings()(input_ids).detach()
    9. t_noisy = t_embed + randn_like(t_embed) * σ_j   [same σ_j, independent z]
       s_noisy = s_embed + randn_like(s_embed) * σ_j   [same σ_j, independent z]
    10. Teacher forward on t_noisy → t_logits_noisy  (under torch.no_grad)
    11. Student forward on s_noisy → s_logits_noisy  (differentiable through layers)
    12. per_sample_kl_noisy = compute_per_sample_kl(t_logits_noisy, s_logits_noisy, ...)
    13. jsd = saliency_divergence(teacher_sal, student_sal, labels_mask)
    14. weights = softmax(jsd / τ_w) * B   # mean=1
    15. loss = (weights.detach() * (per_sample_kl + λ * per_sample_kl_noisy)).mean()
  else:
    15. loss = standard_kl_loss(t_logits, s_logits, labels_mask)

  16. loss.backward() → optimizer.step()
```

### 2.6 Teacher Saliency Cache Format

```python
{
    "saliency": List[Tensor],  # each (L_i,) = full sequence length, response positions = 0
    "metadata": {"model": str, "data": str, "dataset": str, "n_samples": int, "max_seq_len": int}
}
```

Cache stores full-sequence-length saliency (prompt + response, response = 0).
Retrieved by dataset index during training, padded/trimmed to batch sequence length.
Must use identical data_source, seed, max_seq_len, tokenizer, dataset type as training.

### 2.7 Evidence Concentration (SQuAD-specific evaluation)

For SQuAD samples with annotated answer spans, evidence concentration measures
what fraction of saliency mass falls on the answer span tokens:

```
evidence_concentration_i = sum(saliency[answer_start : answer_end + 1]) / sum(saliency)
```

- Teacher's EC is moderate-low (teacher distributes saliency across full context for holistic reasoning)
- Standard KD student's EC is high (over-concentrates on answer span — shortcut learning)
- SaGD student's EC should approach teacher's EC (preserves holistic reasoning pattern)

Empirically verified (seed=42, Qwen3-8B→0.6B, SQuAD val):
  Teacher EC: 0.055, Standard KD EC: 0.169, SaGD EC: 0.083

EC measures whether the student preserves the teacher's reasoning pattern.
The goal is NOT "high EC" but "EC close to teacher". Standard KD students learn
shortcuts (over-focusing on answer tokens), while SaGD preserves the teacher's
broader context utilization.

Answer span token mapping: `SquadDataset` maps character offsets from SQuAD annotations
to token positions using `return_offsets_mapping=True` from the fast tokenizer.
Samples where mapping fails (e.g., truncated) have `answer_token_start = -1` and are
excluded from EC computation.

### 2.8 Ablation Theory Correspondence

| Config | KL (zero-order) | Noise KL (first-order) | Reweight | Theoretical space |
|--------|-----------------|------------------------|----------|-------------------|
| Standard KD | uniform | — | — | L² |
| + Noise KL only | uniform | ✓ (λ>0, τ_w=∞) | — | W^{1,2} |
| + Reweight only | weighted | — (λ=0) | ✓ | L² + DRO |
| **SaGD (full)** | weighted | ✓ | ✓ | W^{1,2} + DRO |

### 2.9 Gradient-PCA Data Selection (GPDS)

**Motivation**: SaGD's noise KL and saliency reweighting operate at the loss/sample level
(HOW to train), but treat all training data equally at the dataset level (WHAT to train on).
Inspired by the Epiplexity framework (Finzi et al., 2026) — which distinguishes structural
information (shared learnable patterns) from random information (sample-specific noise) —
we propose selecting training samples that provide the most structurally informative
gradient signal for the student.

**Key insight**: Fine-tuning gradients are empirically low-rank (the LoRA observation).
Different samples push the student's parameters in similar directions when they require
similar corrections. PCA on the gradient matrix reveals the **principal gradient directions**
— the dominant axes along which the student needs to change. Samples that span these
principal directions contain the structural information; samples whose gradients lie in the
residual subspace contribute only idiosyncratic, non-transferable updates.

**Two contributions at different levels**:

| Level | Mechanism | Signal | Theory |
|-------|-----------|--------|--------|
| **Loss (SaGD)** | Noise KL + saliency reweighting | Saliency divergence | Sobolev W^{1,2} + DRO |
| **Data (GPDS)** | Gradient PCA subset selection | Loss gradient direction | Coreset + D-optimal design |

#### 2.9.1 Per-Sample Projected Gradient

For each training sample $x_i$, compute the KL distillation loss gradient w.r.t. student
parameters $\theta$:

$$\mathbf{g}_i = \nabla_\theta D_\text{KL}(f_T(x_i) \| f_S(x_i; \theta)) \in \mathbb{R}^{|\theta|}$$

Since $|\theta|$ is enormous (~600M for Qwen3-0.6B), project via **Count Sketch** (a form
of JL-preserving random projection):

$$\tilde{\mathbf{g}}_i \in \mathbb{R}^d, \quad \tilde{g}_{i,k} = \sum_{j: h(j)=k} s(j) \cdot g_{i,j}$$

where $h: \{1,...,|\theta|\} \to \{1,...,d\}$ is a hash function and $s(j) \in \{-1,+1\}$
is a random sign. $d = 1024$ suffices by JL guarantee.

**Complexity**: O(|θ|) per sample (one pass over gradient), O(d) memory for result.

#### 2.9.2 Principal Gradient Analysis

Stack projected gradients: $\tilde{\mathbf{G}} = [\tilde{\mathbf{g}}_1, ..., \tilde{\mathbf{g}}_n]^T \in \mathbb{R}^{n \times d}$

Center and SVD: $\tilde{\mathbf{G}} - \bar{\tilde{\mathbf{g}}}^T = \mathbf{U\Sigma V}^T$

- $\mathbf{v}_k$: $k$-th **principal gradient direction** in parameter space
- $\sigma_k^2$: gradient variance along this direction
- $u_{ik} \sigma_k$: sample $i$'s contribution to direction $k$
- **Effective rank** $r^*$: smallest $r$ such that top-$r$ PCs explain ≥90% variance

**Low-rank hypothesis** (supported by LoRA): Fine-tuning gradients concentrate in a
low-dimensional subspace ($r^* \ll \min(n, d)$). Different samples push parameters in
highly correlated directions because they share the same teacher-student gap structure.

**Null model**: To distinguish genuine low-rank structure from projection artifacts,
compare against random directions with matched gradient norms. If real spectrum decays
faster than null → gradient structure is genuine.

#### 2.9.3 Subspace-Spanning Selection

Select subset $\mathcal{S}$ of size $K = \lfloor n \cdot \text{ratio} \rfloor$ that covers the
principal gradient subspace.

**D-optimal criterion** (maximizes information about parameter update):
$$\mathcal{S}^* = \arg\max_{|\mathcal{S}|=K} \log\det\left(\sum_{i \in \mathcal{S}} \tilde{\mathbf{g}}_i \tilde{\mathbf{g}}_i^T + \mu \mathbf{I}\right)$$

**Practical approximation** (greedy per-PC quota):
1. Set $\text{quota}_k = K \cdot \sigma_k^2 / \sum_l \sigma_l^2$ for each PC $k = 1,...,r^*$
2. For each PC, select top-quota_k samples by $|u_{ik}| \sigma_k$ (largest projection)
3. Skip already-selected samples (handle overlap across PCs)
4. Fill remaining slots with highest-loss unselected samples

**Coreset guarantee**: If gradient matrix has effective rank $r^*$ with residual
$\sigma_{r^*+1}$, and $\mathcal{S}$ spans the top-$r^*$ subspace, then:
$$\|\mathbf{g}_\mathcal{S} - \mathbf{g}\|^2 \leq \frac{\sum_{k > r^*} \sigma_k^2}{n} + O(r^*/K)$$

First term = truncation error (small if low-rank), second = sampling error.

#### 2.9.4 Connection to Epiplexity

The Epiplexity framework (Finzi et al., 2026) decomposes data information into:
- **Structural** ($S_T$): compressible patterns encoded into model weights → reusable
  circuits → OOD transfer
- **Random** ($H_T$): irreducible unpredictability → doesn't transfer

In gradient PCA terms:
- **Top PCs** = gradient directions shared by many samples = structural updates
  (changing parameters along these directions helps many samples simultaneously)
- **Bottom PCs** = gradient directions unique to few samples = idiosyncratic updates
  (changing parameters along these directions only helps specific samples)

Selecting samples that span the top PCs = selecting samples with high structural
information content, as measured by their gradient's alignment with the principal
subspace.

Prequential coding estimate: $S_T \approx$ area under the loss curve. Samples whose
loss drops fastest during training contribute most to epiplexity. These are precisely
the samples with large projections on the top gradient PCs — their gradient directions
are well-represented in the principal subspace, leading to efficient loss reduction.

#### 2.9.5 Complete Training Pipeline with GPDS

```
Phase 0: Precompute teacher saliency (once)
  → data/teacher_saliency_{dataset}.pt

Phase 1: Warm-up (1 epoch of standard SaGD on full dataset)
  → Student model after initial training

Phase 2: Gradient profiling + selection
  → For each sample: forward (teacher + student) → KL loss → backward → Count Sketch
  → SVD on projected gradient matrix
  → Per-PC quota selection → selected_indices.pt
  Cost: ~1 extra epoch of forward-backward passes (batch_size=1)

Phase 3: Continue SaGD training on selected subset (remaining epochs)
  → Load selected_indices.pt → torch.utils.data.Subset(dataset, indices)
  → Standard SaGD training (noise KL + saliency reweighting) on subset only
  → Optional: re-profile every K epochs as student evolves

Compute budget:
  Warm-up: 1 epoch
  Profiling: ~1 epoch (batch_size=1, no optimization, just gradient collection)
  Subset training (50% data): 4.5 epochs (= 9 remaining × 0.5)
  Total: ~6.5 epochs vs 10 full epochs → 35% compute savings
```

#### 2.9.6 Implementation: Count Sketch Projector

```python
# Pre-generate hash tables (once, deterministic from seed)
for each param_name, numel:
    hash_dim[name] = randint(0, proj_dim, (numel,))   # which bucket
    hash_sign[name] = randint(0, 2, (numel,)) * 2 - 1  # which sign

# Project gradient (after loss.backward())
projected = zeros(proj_dim)  # on GPU
for name, p in model.named_parameters():
    signed_grad = p.grad.view(-1) * hash_sign[name]
    projected.scatter_add_(0, hash_dim[name], signed_grad)
return projected  # (proj_dim,) → move to CPU for storage
```

Hash tables stored on GPU for speed. Memory: ~7GB for 600M-param model (int64 + float32).
Total GPU budget: teacher (16GB fp16) + student (2.4GB fp32) + hash tables (7GB) + activations (~2GB) ≈ 28GB. Fits on A100 80GB.

#### 2.9.7 Negative Finding: Saliency-Space PCA Does NOT Work

We investigated PCA on saliency-weighted embedding differences
$\Delta z_i = \sum_j (\hat{s}_{T,j} - \hat{s}_{S,j}) \cdot e_j$ and found:
- Effective rank (90%): 661 / 1024 — NOT low-rank
- Top-10 PCs explain only 24.1% of variance (vs 68.9% for null model)
- Real spectrum is FLATTER than random (opposite of hypothesis)

**Reason**: Saliency operates in input space. Different samples have different tokens,
so saliency error directions are inherently diverse. Gradient operates in parameter space,
where all samples share the same parameters → gradient directions can be correlated.

This negative finding validates two choices:
1. Soft saliency reweighting (not hard saliency-based selection) for SaGD
2. Gradient space (not saliency space) for data selection

---

## 3. Registered Methods

```python
METHODS = {
    "sft",            # Supervised fine-tuning (no teacher)
    "standard_kd",    # Forward KL (Hinton, 2015)
    "reverse_kl",     # Reverse KL / MiniLLM (Gu et al., 2024)
    "seqkd",          # Sequence-level KD (Kim & Rush, 2016) — SFT on teacher outputs
    "gkd",            # Generalized KD with JSD (Agarwal et al., 2023)
    "distillm",       # DistiLLM with Skew KL (Ko et al., 2024)
    "dakd",           # DA-KD with BDL (He et al., ICML 2025)
    "sagd",           # SaGD (our method)
}
```

---

## 4. Hyperparameters

### Training (fixed)

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size | 8 |
| Gradient accumulation | 4 (effective batch = 32) |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warmup ratio | 0.03 |
| Max grad norm | 1.0 |
| Max sequence length | 512 |
| KL temperature (T) | 2.0 |
| fp16 | true |
| Seeds | [42, 123, 456] |

### SaGD-specific

| Parameter | Symbol | Default | Sensitivity | Sweep |
|-----------|--------|---------|-------------|-------|
| Noise KL weight | λ | 0.5 | High | [0.1, 0.5, 1.0, 2.0, 5.0] |
| Noise std (relative) | σ | 0.01 | High | [0.001, 0.005, 0.01, 0.02, 0.05] |
| Reweighting temperature | τ_w | 1.0 | High | [0.1, 0.5, 1.0, 2.0, 5.0] |
| Saliency normalization temp | τ_s | 2.0 | Low | — |
| SaGD step frequency | N | 5 | Medium | [1, 3, 5, 10, 20] |

### Ablation configs

| Name | λ | σ | τ_w | Effect |
|------|---|---|-----|--------|
| `sagd` | 0.5 | 0.01 | 1.0 | Full method |
| `sagd_noise_only` | 0.5 | 0.01 | 100.0 | τ_w≈∞ → uniform weights → only noise KL |
| `sagd_reweight_only` | 0.0 | — | 1.0 | No noise KL → only reweighting |

---

## 5. Implementation Rules

### 5.1 Saliency computation must not pollute model gradients
`response_ll.backward()` propagates gradients to ALL `requires_grad=True` tensors in the
graph, including model parameters. The canonical pattern is:
```python
# Save and disable all param grads
param_states = {n: p.requires_grad for n, p in model.named_parameters()}
for p in model.parameters(): p.requires_grad_(False)
try:
    # ... embed, forward, backward ...
finally:
    # Restore
    for n, p in model.named_parameters(): p.requires_grad_(param_states[n])
```

### 5.2 Saliency masking must include attention_mask
`(1 - labels_mask)` is 1 for BOTH prompt tokens AND padding tokens.
Correct: `prompt_mask = (1 - labels_mask).float() * attention_mask.float()`

### 5.3 KL and saliency masks must be shifted
`logit[j]` predicts `token[j+1]`. Therefore:
- Per-position KL: use `per_pos[:, :-1]` with `labels_mask[:, 1:]`
- Response log-prob for saliency: use `logits[:, :-1]` with `labels_mask[:, 1:]`

### 5.4 Reweighting weights must be detached
`weights.detach()` before multiplying with per_sample_kl. Gradients must not flow
through the JSD → softmax → weights path.

### 5.5 Saliency output is pre-masked
`SaliencyComputer.compute()` returns saliency with response AND padding positions zeroed.
Downstream functions (alignment loss, divergence) must NOT apply additional masking.

### 5.6 Dataset must return index
Both `InstructionDataset` and `SquadDataset` return `"index": torch.tensor(idx, dtype=torch.long)`.
`collate_fn` stacks it. `SquadDataset` additionally returns `answer_token_start` and
`answer_token_end` (long scalars, -1 if unmapped). `collate_fn` conditionally stacks these.
Non-SaGD methods silently ignore extra fields.

### 5.7 Cache/training alignment
Precompute script must use identical tokenizer, data_source, dataset type, seed,
max_seq_len, and subset as training. For cross-architecture experiments, use
`--tokenizer_name` pointing to the STUDENT model (since training tokenizes with the
student tokenizer). Any mismatch silently corrupts the index→saliency mapping.

### 5.8 Teacher is always frozen
Teacher stays in `eval()` with `torch.no_grad()` throughout. Never modified.

### 5.9 Baseline isolation
When method is not `sagd`, zero SaGD components are initialized. Baselines run
identically as if SaGD code did not exist.

### 5.10 Noise KL uses detached embeddings with noise
The noisy embedding is created from `student.get_input_embeddings()(input_ids)` under
`torch.no_grad()`, then noise is added. The student forward on `noisy_embed` IS
differentiable (gradients flow through the model layers), but the embedding lookup itself
is detached. This is correct: the noise must be fixed (not optimized away), while the
model's response to the noisy input must be differentiable for training.

---

## 6. Related Work

| Paper | Method | Setting | Key difference from SaGD |
|-------|--------|---------|--------------------------|
| AD-KD (Wu et al., ACL 2023) | IG attribution alignment as loss | BERT classification | Encoder-only, MSE, no reweighting, no theory |
| GKD (Wang et al., 2022) | Input gradient MSE alignment | BERT classification | Full gradient vector (requires same d_model), no reweighting, encoder-only |
| GKD (Agarwal et al., 2024) | Generalized JSD + on-policy | LLM generation | Zero-order only, no first-order matching |
| DA-KD (ICML 2025) | Difficulty-adaptive reweighting | LLM generation | Reweights by output KL (zero-order), not saliency (first-order) |
| TSD (2026) | KL on softmax-normalized saliency | Time series | Loses magnitude, no reweighting, not LLM |
| Sobolev Training (NeurIPS 2017) | Full Jacobian matching | Small models | Not KD, Jacobian intractable for LLMs |
| Srinivas & Fleuret (ICML 2018) | Jacobian matching ≈ Gaussian noise | CNN | Complementary theory; we add saliency compression + reweighting |
| Ballout et al. (2024) | Teacher saliency → top-K rationale text | T5 QA | Uses saliency for data augmentation, not as loss or reweighting |

**SaGD's novelty**: (1) noise-based implicit Jacobian matching for decoder-only LLM distillation (Srinivas theory applied to new domain), (2) saliency-based sample reweighting (no prior work uses attribution divergence for this), (3) noise KL + reweighting dual channel, (4) Sobolev/Taylor + DRO theoretical framework, (5) evidence concentration metric for validating saliency alignment.

---

## 7. Experiments

### 7.1 Checklist
```
Phase 0  Precompute teacher saliency (SQuAD)   1 GPU   ~2h
Phase 1  Exp 1: Saliency divergence diagnosis   1 GPU   ~1h     §4.2
Phase 2  Exp 2: Main table SQuAD (3×3)          4 GPU   ~6h     §4.3
Phase 3  Exp 3: Evidence Concentration           1 GPU   ~1h     §4.4
Phase 4  Exp 4: Ablations (~15 runs)             4 GPU   ~8h     §4.5
Phase 5  Exp 5: Training Dynamics                1 GPU   ~2h     §4.6
Phase 6  Exp 6: Dolly generalization             4 GPU   ~6h     §4.7
Phase 7  Exp 7: Cross-arch LLaMA                 1 GPU   ~4h     §4.8
Phase 8  Exp 8: Benchmark defense                1 GPU   ~2h     Appendix
```

### 7.2 Paper structure
```
§1 Introduction
§2 Background
§3 Method
  3.1 KD as Function Approximation: The L² Perspective
  3.2 Beyond Pointwise Matching: Taylor Expansion and Sobolev Norms
  3.3 Saliency as Tractable First-Order Approximation
  3.4 SaGD: Saliency Alignment Loss + Saliency-Guided Reweighting
  3.5 Complete Algorithm
§4 Experiments
  4.1 Setup
  4.2 Motivation: Does Standard KD Preserve Saliency?    ← Exp 1
  4.3 Main Results (SQuAD: EM, F1, EC)                   ← Exp 2
  4.4 Evidence Concentration Analysis                     ← Exp 3
  4.5 Ablation Study                                      ← Exp 4
  4.6 Training Dynamics                                   ← Exp 5
  4.7 Generalization to Instruction-Following (Dolly)     ← Exp 6
  4.8 Cross-Architecture Generalization (LLaMA)           ← Exp 7
§5 Discussion & Limitations
Appendix: Proofs, Benchmark Defense, Hyperparameter Sensitivity, Visualizations
```

---

## 8. Quick Commands
```bash
# Precompute teacher saliency — SQuAD (primary, run once)
python scripts/precompute_teacher_saliency.py \
    --model_name Qwen/Qwen3-8B --dataset squad \
    --output_path data/teacher_saliency_squad.pt \
    --batch_size 4 --max_seq_len 512 --device cuda:0

# Precompute teacher saliency — Dolly (secondary, run once)
python scripts/precompute_teacher_saliency.py \
    --model_name Qwen/Qwen3-8B --dataset dolly \
    --output_path data/teacher_saliency_dolly.pt \
    --batch_size 4 --max_seq_len 512 --device cuda:0

# Smoke test: baseline on SQuAD
python scripts/train.py \
    --method standard_kd --dataset squad \
    --epochs 1 --max_train_samples 200 \
    --device cuda:0 --skip_eval

# Smoke test: SaGD on SQuAD
python scripts/train.py \
    --method sagd --dataset squad \
    --teacher_saliency_path data/teacher_saliency_squad.pt \
    --lambda_noise 0.5 --noise_sigma 0.01 --sagd_every_n_steps 5 \
    --epochs 1 --max_train_samples 200 \
    --device cuda:0 --skip_eval

# Unit tests
pytest tests/ -v

# Saliency diagnosis with evidence concentration (SQuAD)
# Teacher saliency is computed on-the-fly (no cache needed for diagnosis)
python scripts/diagnose_saliency.py \
    --teacher_model Qwen/Qwen3-8B \
    --student_ckpt outputs/standard_kd/seed_42/student_final.pt \
    --dataset squad \
    --output_path outputs/standard_kd/seed_42/saliency_diagnosis.json \
    --device cuda:0
```

---

## 9. What NOT to Do

- **Do not** compute full Jacobians — $O(V \times L \times d)$ is intractable.
- **Do not** use KL for saliency alignment — softmax normalization discards magnitude.
- **Do not** use MSE for saliency alignment — sensitive to teacher/student scale mismatch.
- **Do not** let saliency backward touch model parameter gradients — use param_grad_states save/restore.
- **Do not** mask saliency with only `(1 - labels_mask)` — must also multiply by `attention_mask`.
- **Do not** skip the shift in KL/saliency mask — `logit[j]` predicts `token[j+1]`, use `labels_mask[:, 1:]`.
- **Do not** let reweighting weights carry gradients — always `.detach()`.
- **Do not** assume cache/training data alignment — verify same data_source, dataset type, seed, max_seq_len, tokenizer, subset.
- **Do not** let noise be differentiable w.r.t. the model — embeddings must be detached before adding noise, otherwise the model can learn to cancel the noise.
- **Do not** evaluate on training data — use subset="test" for EM/F1/ROUGE-L, subset="val" for diagnosis.
- **Do not** include unanswerable SQuAD samples — `SquadDataset` filters them out automatically.
- **Do not** use a slow tokenizer with `SquadDataset` — `return_offsets_mapping=True` requires a fast tokenizer for answer span mapping.
