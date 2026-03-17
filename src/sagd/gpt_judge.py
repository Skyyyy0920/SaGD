"""GPT-as-Judge pairwise evaluation with position debiasing.

Compares two model outputs (A vs B) for each instruction and asks
GPT-4o-mini to pick the better one. To remove position bias, each pair
is judged twice with A/B swapped, and the verdict is aggregated.

Usage (as library):
    from sagd.gpt_judge import GPTJudge
    judge = GPTJudge(api_key="sk-...")
    results = judge.judge_pairwise(responses_a, responses_b)

Usage (as CLI):
    python scripts/gpt_judge.py \
        --responses_a outputs/standard_kd/responses.jsonl \
        --responses_b outputs/sagd/responses.jsonl \
        --output_path outputs/gpt_judge_results.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from tqdm import tqdm


SYSTEM_PROMPT = """\
You are an expert judge evaluating the quality of AI-generated responses.
You will be given an instruction and two responses (Response A and Response B).

Evaluate which response is better based on:
1. **Accuracy**: Is the information correct and relevant?
2. **Completeness**: Does it fully address the instruction?
3. **Clarity**: Is it well-written and easy to understand?
4. **Helpfulness**: Is it genuinely useful to the user?

Respond with EXACTLY one of:
- "A" if Response A is clearly better
- "B" if Response B is clearly better
- "TIE" if they are roughly equal in quality

Output ONLY the verdict (A, B, or TIE), nothing else."""

USER_TEMPLATE = """\
### Instruction
{instruction}

### Response A
{response_a}

### Response B
{response_b}

### Verdict"""


class GPTJudge:
    """Pairwise GPT-as-Judge with position debiasing.

    Each pair is judged twice (A-first, B-first) to mitigate position bias.
    A verdict counts only when both orderings agree. Disagreements are TIEs.

    Args:
        api_key: OpenAI API key. If ``None``, reads ``OPENAI_API_KEY`` env var.
        model: OpenAI model to use for judging.
        max_tokens: Max tokens for the judge response.
        temperature: Sampling temperature (0 = deterministic).
        max_retries: Number of retries per API call on failure.
        retry_delay: Seconds between retries.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 8,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        import openai  # lazy import — optional dependency

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if api_key is None:
            import os
            api_key = os.environ.get("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=api_key)

    # ------------------------------------------------------------------
    # Single-call helpers
    # ------------------------------------------------------------------

    def _call_judge(self, instruction: str, response_a: str, response_b: str) -> str:
        """Single judge call. Returns 'A', 'B', or 'TIE'."""
        user_msg = USER_TEMPLATE.format(
            instruction=instruction,
            response_a=response_a,
            response_b=response_b,
        )

        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                verdict = completion.choices[0].message.content.strip().upper()

                if verdict in ("A", "B", "TIE"):
                    return verdict
                # Unexpected output → treat as TIE
                return "TIE"

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"WARNING: API call failed after {self.max_retries} retries: {e}")
                    return "TIE"

        return "TIE"

    def _judge_one_pair(
        self, instruction: str, response_a: str, response_b: str,
    ) -> dict:
        """Judge one pair with position debiasing (A-first + B-first).

        Returns dict with keys:
            - ``"verdict_ab"``: verdict when A is shown first
            - ``"verdict_ba"``: verdict when B is shown first
            - ``"final_verdict"``: debiased verdict
        """
        # Round 1: A first, B second
        v_ab = self._call_judge(instruction, response_a, response_b)

        # Round 2: B first, A second (swap labels)
        v_ba_raw = self._call_judge(instruction, response_b, response_a)
        # Translate: if the judge said "A" when B was shown first, that means B is better
        v_ba = {"A": "B", "B": "A", "TIE": "TIE"}.get(v_ba_raw, "TIE")

        # Aggregate: only count if both orderings agree
        if v_ab == v_ba:
            final = v_ab
        else:
            final = "TIE"

        return {
            "verdict_ab": v_ab,
            "verdict_ba": v_ba,
            "final_verdict": final,
        }

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def judge_pairwise(
        self,
        responses_a: list[dict],
        responses_b: list[dict],
        label_a: str = "Model_A",
        label_b: str = "Model_B",
    ) -> dict:
        """Run pairwise GPT-as-Judge on two sets of responses.

        Both ``responses_a`` and ``responses_b`` must be lists of dicts
        with at least ``"instruction"`` and ``"generated"`` keys, in the
        same order (aligned by index).

        Returns:
            dict with keys:
                - ``"win_rate_a"``: fraction where A wins
                - ``"win_rate_b"``: fraction where B wins
                - ``"tie_rate"``: fraction of ties
                - ``"n_samples"``: total samples judged
                - ``"per_sample"``: list of per-sample verdict dicts
                - ``"label_a"``: label for model A
                - ``"label_b"``: label for model B
        """
        assert len(responses_a) == len(responses_b), (
            f"Response lists must have same length, got {len(responses_a)} vs {len(responses_b)}"
        )

        per_sample = []
        wins_a = 0
        wins_b = 0
        ties = 0

        for ra, rb in tqdm(
            zip(responses_a, responses_b),
            total=len(responses_a),
            desc="GPT Judging",
        ):
            instruction = ra["instruction"]
            verdict_info = self._judge_one_pair(
                instruction, ra["generated"], rb["generated"],
            )

            verdict_info["index"] = ra.get("index", -1)
            verdict_info["instruction_preview"] = instruction[:100]
            per_sample.append(verdict_info)

            final = verdict_info["final_verdict"]
            if final == "A":
                wins_a += 1
            elif final == "B":
                wins_b += 1
            else:
                ties += 1

        n = len(responses_a)
        return {
            "label_a": label_a,
            "label_b": label_b,
            "win_rate_a": wins_a / max(n, 1),
            "win_rate_b": wins_b / max(n, 1),
            "tie_rate": ties / max(n, 1),
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "n_samples": n,
            "per_sample": per_sample,
        }


def save_judge_results(results: dict, path: str | Path) -> None:
    """Save judge results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
