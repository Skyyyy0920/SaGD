"""ROUGE-L evaluation for instruction-following."""

from __future__ import annotations

import torch
import torch.nn as nn
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from sagd.data import InstructionDataset


def evaluate_rouge(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    dataset: InstructionDataset,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    device: str = "cuda",
) -> dict[str, float]:
    """Generate responses and compute ROUGE-L.

    For each sample:
      1. Extract prompt portion (labels_mask=0) of input_ids
      2. model.generate(prompt, max_new_tokens=max_new_tokens)
      3. Decode generated text
      4. Compare with ground truth response via rouge-score

    Returns:
        {"rouge_l_f": float, "rouge_l_p": float, "rouge_l_r": float}
    """
    model.eval()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    all_scores: list[dict[str, float]] = []

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
            batch_indices = range(i, min(i + batch_size, len(dataset)))
            prompts = []
            references = []

            for idx in batch_indices:
                item = dataset[idx]
                input_ids = item["input_ids"]       # (L,)
                labels_mask = item["labels_mask"]    # (L,)

                # Extract prompt tokens (labels_mask == 0)
                prompt_len = (labels_mask == 0).sum().item()
                prompt_ids = input_ids[:prompt_len]  # (P,)
                prompts.append(prompt_ids)

                meta = dataset.get_metadata(idx)
                references.append(meta["response"])

            # Pad prompts for batched generation
            max_prompt_len = max(p.size(0) for p in prompts)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

            input_ids_batch = torch.full(
                (len(prompts), max_prompt_len), pad_id, dtype=torch.long
            )
            attention_mask_batch = torch.zeros(
                (len(prompts), max_prompt_len), dtype=torch.long
            )

            # Left-pad for causal LM generation
            for j, p in enumerate(prompts):
                pad_len = max_prompt_len - p.size(0)
                input_ids_batch[j, pad_len:] = p
                attention_mask_batch[j, pad_len:] = 1

            input_ids_batch = input_ids_batch.to(device)      # (B, P_max)
            attention_mask_batch = attention_mask_batch.to(device)  # (B, P_max)

            outputs = model.generate(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )  # (B, P_max + G)

            # Decode only the generated part
            for j in range(len(prompts)):
                gen_ids = outputs[j, max_prompt_len:]  # (G,)
                generated = tokenizer.decode(gen_ids, skip_special_tokens=True)
                score = scorer.score(references[j], generated)
                all_scores.append({
                    "rouge_l_f": score["rougeL"].fmeasure,
                    "rouge_l_p": score["rougeL"].precision,
                    "rouge_l_r": score["rougeL"].recall,
                })

    # Average across all samples
    n = len(all_scores)
    return {
        "rouge_l_f": sum(s["rouge_l_f"] for s in all_scores) / max(n, 1),
        "rouge_l_p": sum(s["rouge_l_p"] for s in all_scores) / max(n, 1),
        "rouge_l_r": sum(s["rouge_l_r"] for s in all_scores) / max(n, 1),
    }
