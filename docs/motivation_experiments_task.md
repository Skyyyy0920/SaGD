# Task: Motivation Experiments for SaKD Paper

**Status:** Ready to start. Paper writing has been restructured (Phase 1 done) to remove the PC-curriculum and reorganize the narrative around motivation, method, experiments. This task supplies the missing motivation experiments.

**Owner:** Independent agent (a separate Claude Code session will execute this).

**Deadline:** Before the final paper draft is locked.

---

## 1. Background

After advisor discussion, the SaKD paper now follows a single-thread narrative:

| Block | Content |
|-------|---------|
| Motivation | Empirical evidence that pointwise output matching leaves large neighborhood gaps between teacher and student, even when output agreement is exact. |
| Method | SaKD = noise-injected KL (implicit Jacobian matching) + saliency-divergence DRO reweighting. |
| Experiments | 7-baseline ROUGE-L SOTA + Evidence Concentration diagnostic + cross-architecture validation. |

The motivation block is currently a placeholder. The advisor explicitly asked for **two** empirical figures that justify why SaKD is needed:

> "motivation 需要一些实验验证，比如说一些 pointwise matching 但是邻域差距很大的 failure mode，然后在两个模型上邻域差异分布的分布情况，这两个正好可以 motivate 我们去设计方法，saliency-aware 就是为了 motivation 解决问题的一个手段。"

Translation: motivation needs experimental verification: (1) some failure modes where pointwise matching agrees but the neighborhood gap is large, and (2) the distribution of neighborhood differences between the two models. These two pieces are exactly what motivates the method design.

---

## 2. Goal

Produce two motivation figures plus a numerical summary, ready to drop into Section 1 (Introduction). Per the user's editorial decision, motivation evidence is woven into the intro, not given its own section.

### M1: Pointwise-matching failure mode

Demonstrate that some samples exhibit small teacher-student output KL at the training point but large divergence under small input perturbations.

**Visualization:** Scatter plot.

| Axis | Quantity |
|------|----------|
| X | $D(x_i) := D_\text{KL}(f_T(x_i) \\| f_S(x_i))$, teacher-student KL at the training point |
| Y | $\Delta D(x_i, \delta) := \mathbb{E}_\xi[D(x_i + \xi)] - D(x_i)$, average neighborhood degradation under Gaussian perturbation $\xi \sim \mathcal{N}(0, \sigma^2 I)$ on input embeddings |

Highlight the failure-mode cluster: samples where `X is small` but `Y is large`. Annotate 1-2 representative samples; for each, optionally show the saliency heatmap to illustrate the shortcut (similar to the existing intro figure on Eiffel Tower QA).

**Sub-deliverable:** `motivation_M1.pdf` showing the scatter and 1-2 highlighted samples.

### M2: Distribution of teacher-student neighborhood gap

Demonstrate that, across the training set, the Standard-KD student exhibits systematically larger neighborhood gaps than the teacher does internally.

**Visualization:** Histograms or kernel density estimates (KDE), 2-3 distributions overlaid.

| Distribution | Quantity |
|--------------|----------|
| Teacher self-consistency (control) | $D_\text{KL}(f_T(x_i) \\| f_T(x_i + \xi))$ for each $x_i$ |
| Standard-KD student | $D_\text{KL}(f_T(x_i) \\| f_S^{\text{KD}}(x_i + \xi))$ for each $x_i$ |
| (Optional, if SaKD checkpoint exists) SaKD student | $D_\text{KL}(f_T(x_i) \\| f_S^{\text{SaKD}}(x_i + \xi))$ for each $x_i$ |

Standard-KD's distribution should have a heavier right tail than the teacher's self-consistency control. The figure makes the gap visible.

**Sub-deliverable:** `motivation_M2.pdf` with overlaid distributions and clear labels.

### Optional: numerical summary

| Method | $D(x)$ at train pt (mean) | $\mathbb{E}_\xi[D(x+\xi)]$ (mean) | 95th percentile |
|--------|---------------------------|------------------------------------|-----------------|
| Standard KD | ... | ... | ... |
| SaKD (if available) | ... | ... | ... |

---

## 3. Inputs

| Resource | Path / source |
|----------|---------------|
| Teacher model | `Qwen/Qwen3-8B` (HuggingFace, already cached) |
| Standard-KD student checkpoint | `outputs/standard_kd/seed_42/student_final.pt` |
| SaKD student checkpoint (if available) | `outputs/sagd/seed_42/student_final.pt` (may not exist yet; if not, M2 with two distributions only is acceptable) |
| Dataset | Dolly-15K validation split (`MiniLLM/dolly/raw.jsonl`, 500 samples), primary; or SQuAD 2.0 val if M1 saliency examples need answer-span context |
| Existing utility | `scripts/diagnose_saliency.py` and `scripts/precompute_teacher_saliency.py` for saliency mechanics; reuse these patterns. |
| Existing trainer | `src/sagd/trainer.py` for embedding-perturbation conventions (see `Trainer._adaptive_noise_scale` for $\sigma_j$ formula; for motivation we use **isotropic** noise, not adaptive). |

### Hardware
- 1x A100 80GB sufficient (teacher Qwen3-8B in fp16 plus student Qwen3-0.6B in fp32).
- Estimated wall-clock: M1 ~30 min (500 samples, 1 noise sample), M2 ~1 hour (500 samples, 5 noise samples, 2-3 models).

---

## 4. Implementation plan

Create one script: `scripts/motivation_experiments.py`. It should support both experiments behind subcommands.

```bash
# M1: failure-mode scatter
python scripts/motivation_experiments.py m1 \
    --teacher Qwen/Qwen3-8B \
    --student outputs/standard_kd/seed_42/student_final.pt \
    --dataset MiniLLM/dolly/raw.jsonl \
    --n_samples 500 \
    --noise_sigma 0.01 \
    --noise_repeats 5 \
    --output_pdf writing/NeurIPS26-SaGD/sources/motivation_M1.pdf \
    --output_json outputs/motivation/m1_data.json

# M2: distribution comparison
python scripts/motivation_experiments.py m2 \
    --teacher Qwen/Qwen3-8B \
    --students outputs/standard_kd/seed_42/student_final.pt outputs/sagd/seed_42/student_final.pt \
    --student_labels "Standard KD" "SaKD" \
    --dataset MiniLLM/dolly/raw.jsonl \
    --n_samples 500 \
    --noise_sigma 0.01 \
    --noise_repeats 5 \
    --output_pdf writing/NeurIPS26-SaGD/sources/motivation_M2.pdf \
    --output_json outputs/motivation/m2_data.json
```

### Computation specifics

For each sample $x_i$ in the eval subset:

1. **Embed input:** `e = model.get_input_embeddings()(input_ids).detach()`.
2. **Compute clean KL** $D(x_i)$:
   - `t_logits = teacher(inputs_embeds=e_T, ...).logits` (under `torch.no_grad()`)
   - `s_logits = student(inputs_embeds=e_S, ...).logits` (under `torch.no_grad()`)
   - `D_clean = forward_kl(t_logits, s_logits, labels_mask)` over response positions only.
3. **Compute neighborhood KL** $\mathbb{E}_\xi[D(x_i + \xi)]$:
   - For `noise_repeats` independent noise draws $\xi \sim \mathcal{N}(0, \sigma^2 I)$ at the same scale on both models' embeddings:
     - `t_noisy = teacher(inputs_embeds=e_T + xi_T).logits`
     - `s_noisy = student(inputs_embeds=e_S + xi_S).logits`
     - `D_noisy_k = forward_kl(t_noisy, s_noisy, labels_mask)`.
   - Average: `D_noisy_mean = mean(D_noisy_k for k in repeats)`.
4. **Record** `(D_clean, D_noisy_mean - D_clean, sample_idx)` to JSON.

For M2, the same per-sample computation; but produce a histogram of `D_noisy_mean` values across all samples per model.

### Plotting

Use `matplotlib`:
- M1: `ax.scatter(x, y, alpha=0.3, s=10)`; highlight failure cluster with red color where `D_clean < threshold_low` and `delta_D > threshold_high`. Add titles, axis labels, and a "failure-mode" annotation arrow.
- M2: `seaborn.kdeplot` or `ax.hist(values, bins=50, density=True, alpha=0.5, label=...)` for each model. Use distinct colors and a legend.

NeurIPS-style aesthetic:
- White background, sans-serif font, no gridlines.
- Tight layout, axis labels ~10pt, title ~11pt.
- Save as PDF (vector graphics) at `tight_layout` to fit `0.7 \linewidth` in single column.

### Outputs to commit / publish

| File | Destination | Format |
|------|-------------|--------|
| `motivation_M1.pdf` | `writing/NeurIPS26-SaGD/sources/` (Overleaf repo) | PDF |
| `motivation_M2.pdf` | `writing/NeurIPS26-SaGD/sources/` (Overleaf repo) | PDF |
| `m1_data.json`, `m2_data.json` | `outputs/motivation/` (main repo, gitignored or kept) | JSON for reproducibility |
| `scripts/motivation_experiments.py` | main repo `scripts/` | Python script |

---

## 5. Acceptance criteria

1. `scripts/motivation_experiments.py` runs end-to-end on Dolly-15K val 500 samples within ~2 hours wall-clock on 1x A100.
2. M1 scatter plot shows a visible cluster of samples with small `D(x)` and large `delta_D`. At least 5% of samples should fall in this regime.
3. M2 histograms show that the Standard-KD student's distribution has a clearly heavier right tail than the teacher self-consistency control. Numerical: 95th percentile of Standard-KD distribution should be at least 1.5x the teacher's.
4. Both PDFs are vector graphics and fit at `0.7 \linewidth` in NeurIPS single-column layout (~3.85 inches wide).
5. JSON sidecars (`m1_data.json`, `m2_data.json`) record raw per-sample numbers so the user can re-plot or include in supplementary tables.
6. Script accepts both Standard-KD and SaKD checkpoints if SaKD is available; gracefully degrades to teacher-vs-Standard-KD if SaKD is missing.

---

## 6. Prompt for the new agent

Paste the following block to the new Claude Code session (it is self-contained; the agent has no access to this conversation's history):

> ---
>
> Please complete the SaKD paper's motivation experiments described in
> `docs/motivation_experiments_task.md` at the project root
> (`W:\Beyond Output Mimicry Preserving Internal Mechanisms in Knowledge Distillation via Contrastive Residual Alignment\SaGD\`).
>
> Read that file first; it contains full context (background, goal,
> inputs, implementation plan, acceptance criteria).
>
> Concretely:
> 1. Implement `scripts/motivation_experiments.py` with two subcommands `m1` and `m2`.
>    Reuse the saliency / embedding-perturbation patterns from
>    `scripts/diagnose_saliency.py` and `src/sagd/trainer.py`. Use isotropic
>    Gaussian noise on input embeddings (sigma = 0.01 of embedding norm by
>    default), compute teacher-student forward-KL on response positions,
>    and aggregate over 5 noise samples.
> 2. Run both subcommands on Dolly-15K val (first 500 samples) with the
>    Qwen3-8B teacher and the Standard-KD student checkpoint at
>    `outputs/standard_kd/seed_42/student_final.pt`. If
>    `outputs/sagd/seed_42/student_final.pt` exists, include it in M2.
> 3. Produce `motivation_M1.pdf` and `motivation_M2.pdf` in
>    `writing/NeurIPS26-SaGD/sources/` (NeurIPS-style: white background,
>    sans-serif, no gridlines, tight_layout, vector PDF).
> 4. Save raw per-sample numbers to `outputs/motivation/m1_data.json` and
>    `m2_data.json`.
> 5. Verify the acceptance criteria (Section 5 of the task doc):
>    - M1 must show a visible failure-mode cluster (>=5% of samples).
>    - M2 must show that the Standard-KD distribution's 95th percentile is
>      >=1.5x the teacher self-consistency control.
> 6. Print a summary report when done: how many samples processed, which
>    acceptance criteria were met, paths of the four output files. Do
>    NOT push to Overleaf. The user will review and push manually.
>
> Do not modify the LaTeX paper itself. Only generate the PDFs and the
> Python script.
>
> If a checkpoint is missing or a script fails to run, stop and report
> the issue rather than fabricating results.
>
> ---

---

## 7. Hand-off checklist (for the user)

After the new agent finishes:

- [ ] Inspect `motivation_M1.pdf` and `motivation_M2.pdf` visually.
- [ ] Read summary report; verify acceptance criteria 1-6.
- [ ] If satisfied, push the two PDFs to Overleaf:
  ```
  cd writing/NeurIPS26-SaGD
  git add sources/motivation_M1.pdf sources/motivation_M2.pdf
  git commit -m "Add motivation experiments M1 and M2 PDFs"
  git push origin master
  ```
- [ ] Add `\includegraphics` references to the LaTeX inside Section 1 (Introduction), with caption descriptions. Do not create a separate motivation section.
- [ ] Update Table 2 ablation prose if the new motivation evidence changes the framing of "noise KL captures the gap".
- [ ] Consider whether the `effective rank 7 of 1024` evidence (now removed from the paper) should be replaced with the new neighborhood-gap distribution as the empirical motivation.
