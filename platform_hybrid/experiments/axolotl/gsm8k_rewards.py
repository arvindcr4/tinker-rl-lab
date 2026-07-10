"""
GSM8K reward functions and transforms for Axolotl GRPO.

Binary correctness reward matching the Tinker GRPO experiment:
  1.0 if final answer is correct, 0.0 otherwise.
"""

import re
from typing import Optional


question_suffix = " Provide a numerical answer without units, written inside \\boxed{}."


def _extract_boxed(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} format."""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    # Fallback: last number in text
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    return numbers[-1] if numbers else None


def _normalize_number(s: str) -> Optional[float]:
    """Parse a string as a number, handling commas."""
    try:
        return float(s.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def gsm8k_correctness_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Binary reward: 1.0 if correct, 0.0 otherwise.

    Expects kwargs to contain 'answer' (gold numeric answers).
    Matches Tinker's scoring exactly.
    """
    gold_answers = kwargs.get("answer", [])
    rewards = []

    for completion, gold in zip(completions, gold_answers):
        # Strip think tags if present (Qwen3 thinking mode)
        answer_text = completion.split("</think>")[-1] if "</think>" in completion else completion

        predicted_str = _extract_boxed(answer_text)
        predicted = _normalize_number(predicted_str) if predicted_str else None
        expected = _normalize_number(str(gold))

        if predicted is not None and expected is not None:
            rewards.append(1.0 if abs(predicted - expected) < 0.01 else 0.0)
        else:
            rewards.append(0.0)

    return rewards


def gsm8k_grpo_transform(cfg, *args, **kwargs):
    """Transform GSM8K dataset for GRPO: extract prompt + gold answer label."""

    def transform_fn(example, tokenizer=None):
        gold = example["answer"].split("####")[-1].strip().replace(",", "")
        return {
            "prompt": [
                {"role": "user", "content": example["question"] + question_suffix},
            ],
            "answer": gold,
        }

    return transform_fn, {"remove_columns": ["question"]}
