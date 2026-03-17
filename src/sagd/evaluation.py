"""Evaluation metrics for instruction-following: ROUGE-L, BERTScore, Perplexity."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from sagd.data import InstructionDataset


# ---------------------------------------------------------------------------
# 1. Shared response generation
# ---------------------------------------------------------------------------

def generate_responses(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    dataset: InstructionDataset,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    device: str = "cuda",
) -> list[dict[str, str]]:
    """Generate responses for every sample in *dataset*.

    Returns a list of dicts, each with keys:
        - ``"index"``: int — dataset index
        - ``"instruction"``: str — prompt text
        - ``"reference"``: str — ground-truth response
        - ``"generated"``: str — model-generated response
        - ``"category"``: str — Dolly category
    """
    model.eval()
    results: list[dict[str, str]] = []

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Generating"):
            batch_indices = list(range(i, min(i + batch_size, len(dataset))))
            prompts = []
            metas = []

            for idx in batch_indices:
                item = dataset[idx]
                input_ids = item["input_ids"]        # (L,)
                labels_mask = item["labels_mask"]     # (L,)

                prompt_len = (labels_mask == 0).sum().item()
                prompt_ids = input_ids[:prompt_len]
                prompts.append(prompt_ids)

                meta = dataset.get_metadata(idx)
                metas.append(meta)

            # Left-pad for causal LM generation
            max_prompt_len = max(p.size(0) for p in prompts)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

            input_ids_batch = torch.full(
                (len(prompts), max_prompt_len), pad_id, dtype=torch.long,
            )
            attention_mask_batch = torch.zeros(
                (len(prompts), max_prompt_len), dtype=torch.long,
            )

            for j, p in enumerate(prompts):
                pad_len = max_prompt_len - p.size(0)
                input_ids_batch[j, pad_len:] = p
                attention_mask_batch[j, pad_len:] = 1

            input_ids_batch = input_ids_batch.to(device)
            attention_mask_batch = attention_mask_batch.to(device)

            outputs = model.generate(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )

            for j, idx in enumerate(batch_indices):
                gen_ids = outputs[j, max_prompt_len:]
                generated = tokenizer.decode(gen_ids, skip_special_tokens=True)
                results.append({
                    "index": idx,
                    "instruction": metas[j]["instruction"],
                    "reference": metas[j]["response"],
                    "generated": generated,
                    "category": metas[j]["category"],
                })

    return results


def save_responses(responses: list[dict], path: str | Path) -> None:
    """Save generated responses to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in responses:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_responses(path: str | Path) -> list[dict]:
    """Load generated responses from a JSONL file."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


# ---------------------------------------------------------------------------
# 2. ROUGE-L
# ---------------------------------------------------------------------------

def compute_rouge(responses: list[dict]) -> dict[str, float]:
    """Compute ROUGE-L from pre-generated responses.

    Args:
        responses: list of dicts with ``"reference"`` and ``"generated"`` keys.

    Returns:
        ``{"rouge_l_f": float, "rouge_l_p": float, "rouge_l_r": float}``
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for r in responses:
        score = scorer.score(r["reference"], r["generated"])
        scores.append({
            "rouge_l_f": score["rougeL"].fmeasure,
            "rouge_l_p": score["rougeL"].precision,
            "rouge_l_r": score["rougeL"].recall,
        })
    n = max(len(scores), 1)
    return {
        "rouge_l_f": sum(s["rouge_l_f"] for s in scores) / n,
        "rouge_l_p": sum(s["rouge_l_p"] for s in scores) / n,
        "rouge_l_r": sum(s["rouge_l_r"] for s in scores) / n,
    }


# ---------------------------------------------------------------------------
# 3. BERTScore
# ---------------------------------------------------------------------------

def compute_bertscore(
    responses: list[dict],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    device: str = "cuda",
    batch_size: int = 32,
) -> dict[str, float]:
    """Compute BERTScore F1/P/R from pre-generated responses.

    Requires the ``bert-score`` package (``pip install bert-score``).

    Args:
        responses: list of dicts with ``"reference"`` and ``"generated"`` keys.
        model_type: BERTScore encoder model.
        device: torch device string.
        batch_size: BERTScore internal batch size.

    Returns:
        ``{"bertscore_f": float, "bertscore_p": float, "bertscore_r": float}``
    """
    from bert_score import score as bert_score  # lazy import — optional dep

    refs = [r["reference"] for r in responses]
    hyps = [r["generated"] for r in responses]

    P, R, F = bert_score(
        hyps, refs,
        model_type=model_type,
        device=device,
        batch_size=batch_size,
        verbose=True,
    )

    return {
        "bertscore_f": F.mean().item(),
        "bertscore_p": P.mean().item(),
        "bertscore_r": R.mean().item(),
    }


# ---------------------------------------------------------------------------
# 4. Perplexity
# ---------------------------------------------------------------------------

def compute_perplexity(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    dataset: InstructionDataset,
    batch_size: int = 8,
    device: str = "cuda",
) -> dict[str, float]:
    """Compute perplexity on response tokens.

    For each sample, computes cross-entropy loss on response positions
    (labels_mask=1) using teacher-forced decoding, then exponentiates.

    Returns:
        ``{"perplexity": float, "avg_loss": float}``
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Perplexity"):
            batch_indices = list(range(i, min(i + batch_size, len(dataset))))
            items = [dataset[idx] for idx in batch_indices]

            # Manual collation with padding
            max_len = max(item["input_ids"].size(0) for item in items)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

            input_ids = torch.full((len(items), max_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros(len(items), max_len, dtype=torch.long)
            labels_mask = torch.zeros(len(items), max_len, dtype=torch.long)

            for j, item in enumerate(items):
                L = item["input_ids"].size(0)
                input_ids[j, :L] = item["input_ids"]
                attention_mask[j, :L] = item["attention_mask"]
                labels_mask[j, :L] = item["labels_mask"]

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels_mask = labels_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B, L, V)

            # Shift: logit[j] predicts token[j+1]
            shift_logits = logits[:, :-1, :].contiguous()   # (B, L-1, V)
            shift_labels = input_ids[:, 1:].contiguous()     # (B, L-1)
            shift_mask = labels_mask[:, 1:].float()          # (B, L-1)

            # Per-token cross-entropy
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(
                dim=-1, index=shift_labels.unsqueeze(-1),
            ).squeeze(-1)  # (B, L-1)

            # Masked sum
            masked_nll = -(token_log_probs * shift_mask)
            total_loss += masked_nll.sum().item()
            total_tokens += shift_mask.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "perplexity": torch.exp(torch.tensor(avg_loss)).item(),
        "avg_loss": avg_loss,
    }


# ---------------------------------------------------------------------------
# 5. Combined evaluation
# ---------------------------------------------------------------------------

def evaluate_all(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    dataset: InstructionDataset,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    device: str = "cuda",
    skip_bertscore: bool = False,
    bertscore_model: str = "microsoft/deberta-xlarge-mnli",
) -> dict[str, float]:
    """Run all metrics: ROUGE-L, BERTScore (optional), Perplexity.

    Generates responses once, then reuses them across ROUGE-L and BERTScore.
    Perplexity is computed separately (teacher-forced, no generation needed).

    Returns dict with all metric keys merged.
    """
    # 1. Generate responses once
    responses = generate_responses(
        model, tokenizer, dataset,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        device=device,
    )

    # 2. ROUGE-L
    metrics = compute_rouge(responses)

    # 3. BERTScore (optional)
    if not skip_bertscore:
        try:
            bs_metrics = compute_bertscore(
                responses, model_type=bertscore_model, device=device,
            )
            metrics.update(bs_metrics)
        except ImportError:
            print("WARNING: bert-score not installed, skipping BERTScore.")

    # 4. Perplexity
    ppl_metrics = compute_perplexity(
        model, tokenizer, dataset,
        batch_size=batch_size, device=device,
    )
    metrics.update(ppl_metrics)

    return metrics


# ---------------------------------------------------------------------------
# 6. Legacy API (backward-compatible)
# ---------------------------------------------------------------------------

def evaluate_rouge(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    dataset: InstructionDataset,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    device: str = "cuda",
) -> dict[str, float]:
    """Legacy API: generate + ROUGE-L in one call.

    Kept for backward compatibility with existing scripts and tests.
    """
    responses = generate_responses(
        model, tokenizer, dataset,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        device=device,
    )
    return compute_rouge(responses)
