# CLAUDE.md — SaGD

This file is the authoritative guide for AI assistants working on this codebase.
Read it completely before making any changes.

---

## 1. Project Overview

**Research goal**: Knowledge distillation from large to small LLMs.
**Method**: SaGD (Saliency-Guided Knowledge Distillation).
**Paper title**: *Saliency-Guided Knowledge Distillation: A Sobolev Perspective on Teaching Students Where to Look*

### Model pairs
- Primary: Qwen3-8B (teacher) → Qwen3-0.6B (student)
- Secondary (cross-architecture): LLaMA 3.1-8B → LLaMA 3.1-1B

### Datasets & evaluation

**Primary: SQuAD 2.0** (`rajpurkar/squad_v2`) — extractive QA, context-dependent
- Every sample has a context paragraph + question; answer must be extracted from context
- Answerable subset only (unanswerable questions filtered out, ~86K train, ~5.9K val)
- Train: HF `train` split, shuffled(seed=42)
- Val/Test: HF `validation` split, shuffled, split in half (first half=val, second half=test)
- Primary metrics: Exact Match (EM), Token F1 on test split
- Saliency metric: Evidence Concentration (fraction of saliency mass on answer span)
- Secondary metric: Mean JSD (Saliency Loyalty) on val split
- Answer span token positions tracked for evidence concentration evaluation

**Secondary: Dolly-15K** (`databricks/databricks-dolly-15k`) — instruction-following, generalization
- shuffle(seed=42), max_seq_len=512
- Train subset: first N-1000 samples (~14K)
- Val subset: next 500 samples
- Test subset: last 500 samples
- Primary metric: ROUGE-L on test-500
- Used to demonstrate SaGD generalizes beyond extractive QA

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
**Input saliency** compresses it to a per-position scalar:
$$s_i = \left\|\frac{\partial \log P(\text{response})}{\partial e_i}\right\|$$
By Cauchy-Schwarz, $\|s_T - s_S\|^2 \leq \|J_T - J_S\|_F^2$ — saliency distance is a
principled lower bound of Jacobian distance, not an arbitrary approximation.

### 2.2 Complete Loss

$$\mathcal{L}_\text{SaGD} = \underbrace{\sum_{i=1}^B w_i \cdot D_\text{KL}(f_T(x_i) \| f_S(x_i))}_\text{weighted KL (zero-order + DRO)} + \lambda \cdot \underbrace{\frac{1}{B}\sum_{i=1}^B (1 - \cos(s_T^i, s_S^i))}_\text{saliency alignment loss (first-order)}$$

Sample weights (mean-normalized to 1):
$$w_i = \frac{\exp(\text{JSD}_i / \tau_w)}{\frac{1}{B}\sum_j \exp(\text{JSD}_j / \tau_w)}$$

where $\text{JSD}_i = \text{JSD}(\hat{s}_T^i, \hat{s}_S^i)$, $\hat{s} = \text{softmax}(s/\tau_s)$.

**Two components and why both are needed**:
- **Saliency alignment loss**: Directly shrinks the first-order gap → tightens neighborhood
  error bounds for all samples. Uses cosine distance on raw (unnormalized) saliency vectors.
- **Saliency-guided reweighting**: Concentrates zero-order (KL) optimization on samples where
  teacher/student attend to different input tokens. Corresponds to distributionally robust
  optimization (DRO) — prioritizing samples with the loosest neighborhood error bounds.
- Neither alone achieves both effects.

**Why cosine** (not KL or MSE):
- KL requires softmax normalization → discards magnitude information.
- MSE is sensitive to absolute scale differences between teacher/student (different model sizes).
- Cosine captures "which positions matter" independent of scale.

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
**Student saliency** is computed every N training steps.

**Differentiable vs non-differentiable saliency**:
- `compute()`: Non-differentiable. Used for teacher precomputation, diagnosis, and
  reweighting signal. Returns detached tensor. No gradients flow to model parameters.
- `compute_differentiable()`: Differentiable via `torch.autograd.grad(create_graph=True)`.
  Used ONLY for student saliency in the alignment loss path. Returns tensor with gradient
  graph intact so that sal_loss.backward() propagates to student parameters.

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
Pre-training (once, ~1h):
  precompute_teacher_saliency.py → data/teacher_saliency.pt

Each training step:
  1. Load batch (input_ids, attention_mask, labels_mask, index)
  2. Teacher forward → t_logits  (under torch.no_grad)
  3. Student forward → s_logits

  if method == "sagd" AND global_step % N == 0:
    4. per_sample_kl = compute_per_sample_kl(t_logits, s_logits, labels_mask)
    5. student_sal = saliency_computer.compute(student, ...)
    6. teacher_sal = get_cached_teacher_saliency(batch["index"])
    7. sal_loss = saliency_alignment_loss(teacher_sal, student_sal, labels_mask)
    8. jsd = saliency_divergence(teacher_sal, student_sal, labels_mask)
    9. weights = softmax(jsd / τ_w) * B   # mean=1
    10. loss = (weights.detach() * per_sample_kl).mean() + λ * sal_loss
  else:
    10. loss = standard_kl_loss(t_logits, s_logits, labels_mask)

  11. loss.backward() → optimizer.step()
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

- Teacher's EC should be high (teacher "looks at" the evidence)
- SaGD student's EC should approach teacher's EC
- Standard KD student's EC should be lower (doesn't preserve where to look)

This directly validates the core claim: SaGD teaches students WHERE to look, not just
WHAT to output. Unlike Mean JSD (which measures distribution divergence), EC has
ground-truth: the answer span IS the evidence the model should attend to.

Answer span token mapping: `SquadDataset` maps character offsets from SQuAD annotations
to token positions using `return_offsets_mapping=True` from the fast tokenizer.
Samples where mapping fails (e.g., truncated) have `answer_token_start = -1` and are
excluded from EC computation.

### 2.8 Ablation Theory Correspondence

| Config | KL (zero-order) | Sal loss (first-order) | Reweight | Theoretical space |
|--------|-----------------|------------------------|----------|-------------------|
| Standard KD | uniform | — | — | L² |
| + Sal loss only | uniform | ✓ | — | W^{1,2} |
| + Reweight only | weighted | — | ✓ | L² + DRO |
| **SaGD (full)** | weighted | ✓ | ✓ | W^{1,2} + DRO |

---

## 3. Registered Methods

```python
METHODS = {
    "standard_kd",    # Forward KL baseline
    "reverse_kl",     # Reverse KL baseline
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
| Saliency loss weight | λ | 0.5 | High | [0.01, 0.1, 0.5, 1.0, 2.0] |
| Reweighting temperature | τ_w | 1.0 | High | [0.1, 0.5, 1.0, 2.0, 5.0] |
| Saliency normalization temp | τ_s | 2.0 | Low | — |
| Saliency update frequency | N | 5 | Medium | [1, 3, 5, 10, 20] |

### Ablation configs

| Name | λ | τ_w | Effect |
|------|---|-----|--------|
| `sagd` | 0.5 | 1.0 | Full method |
| `sagd_loss_only` | 0.5 | 100.0 | τ_w≈∞ → uniform weights → only saliency loss |
| `sagd_reweight_only` | 0.0 | 1.0 | No saliency loss → only reweighting |

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

### 5.10 Saliency alignment loss requires differentiable saliency
The alignment loss must use `compute_differentiable()` for the student. Using `compute()`
(which returns detached tensors) makes the loss a constant that does not affect training.
This is a critical correctness requirement.

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

**SaGD's novelty**: (1) decoder-only LLM instruction distillation, (2) saliency-based sample reweighting (no prior work uses attribution divergence for this), (3) cosine alignment (not KL/MSE), (4) loss + reweighting dual channel, (5) Sobolev/Taylor + DRO theoretical framework.

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
    --lambda_sal 0.5 --sagd_every_n_steps 5 \
    --epochs 1 --max_train_samples 200 \
    --device cuda:0 --skip_eval

# Unit tests
pytest tests/ -v

# Saliency diagnosis with evidence concentration (SQuAD)
python scripts/diagnose_saliency.py \
    --student_ckpt outputs/standard_kd/seed_42/student_final.pt \
    --teacher_saliency_path data/teacher_saliency_squad.pt \
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
- **Do not** use `compute()` for student saliency in the alignment loss — it returns detached tensors, making the loss term a no-op. Use `compute_differentiable()`.
- **Do not** evaluate on training data — use subset="test" for EM/F1/ROUGE-L, subset="val" for diagnosis.
- **Do not** include unanswerable SQuAD samples — `SquadDataset` filters them out automatically.
- **Do not** use a slow tokenizer with `SquadDataset` — `return_offsets_mapping=True` requires a fast tokenizer for answer span mapping.
