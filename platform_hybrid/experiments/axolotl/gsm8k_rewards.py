"""
GSM8K reward functions and transforms for Axolotl GRPO.

Binary correctness reward matching the Tinker GRPO experiment:
  1.0 if final answer is correct, 0.0 otherwise.
"""

import sys as _contract_sys
from pathlib import Path as _ContractPath

if len(_ContractPath(__file__).resolve().parents) > 3:
    _contract_sys.path.insert(0, str(_ContractPath(__file__).resolve().parents[3]))
from platform_hybrid.experiments.framework_contracts import (
    score_batch,
)


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


def gsm8k_correctness_reward(completions, answer=None, **kwargs):
    """One versioned score per string or conversational completion."""
    return score_batch(completions, answer)


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
