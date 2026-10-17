"""Numeric correctness and prompt-group diagnostics, protocol trl-gsm8k-v2.

This is an outcome grader, not a process-reward model. Historical keyword-reward
runs are not comparable and must retain their historical protocol identity.
"""

from __future__ import annotations

import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from fractions import Fraction

PROTOCOL_VERSION = "trl-gsm8k-numeric-v2"
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?")


def completion_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and isinstance(value[-1], dict):
        return str(value[-1].get("content", ""))
    raise ValueError("Expected a text or chat completion")


def numeric_answer(value):
    text = str(value).strip()
    boxes = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxes:
        text = boxes[-1].strip()
    elif "####" in text:
        text = text.rsplit("####", 1)[-1].strip()
    text = text.replace(",", "")
    if not _NUMBER.fullmatch(text):
        return None
    try:
        return Fraction(text) if "/" in text else Fraction(Decimal(text))
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return None


def correctness_rewards(completions, answers):
    if answers is None or len(completions) != len(answers):
        raise ValueError("One gold answer is required for every completion")
    rewards = []
    for completion, answer in zip(completions, answers):
        gold = numeric_answer(answer)
        if gold is None:
            raise ValueError("Gold answer does not match the versioned numeric contract")
        rewards.append(float(numeric_answer(completion_text(completion)) == gold))
    return rewards


def group_diagnostics(rewards, prompt_ids, group_size, epsilon=1e-8):
    if prompt_ids is None or len(prompt_ids) != len(rewards) or not rewards or group_size < 2:
        raise ValueError("Verified prompt IDs and a positive generation group are required")
    groups = defaultdict(list)
    for prompt_id, reward in zip(prompt_ids, rewards):
        if not isinstance(prompt_id, (str, int)) or isinstance(prompt_id, bool):
            raise ValueError("Invalid prompt ID")
        groups[prompt_id].append(float(reward))
    if any(len(group) != group_size for group in groups.values()):
        raise ValueError("Prompt groups must each contain exactly num_generations rewards")
    variances = []
    for group in groups.values():
        mean = sum(group) / len(group)
        variances.append(sum((r - mean) ** 2 for r in group) / len(group))
    return {
        "reward_variance": sum(variances) / len(variances),
        "advantage_collapse_rate": sum(v <= epsilon for v in variances) / len(variances),
    }
