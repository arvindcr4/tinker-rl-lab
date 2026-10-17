"""Pinned local OpenRLHF reward_func(queries, prompts, labels) callback.

Exactly one numeric boxed answer in the completion is accepted. Multiple boxes,
malformed numbers and truncated boxes score zero, including conflicting boxes.
"""
from decimal import Decimal, InvalidOperation
import re


def parse_number(text):
    text = text.strip()
    if "," in text and not re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d*)?", text):
        return None
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text.replace(",", "")):
        return None
    try:
        value = Decimal(text.strip().replace(",", ""))
        return value if value.is_finite() else None
    except InvalidOperation:
        return None


def score_completion(completion, label):
    truth = parse_number(str(label))
    if truth is None:
        raise ValueError("Invalid ground-truth label")
    answers = re.findall(r"\\boxed\{([^{}]*)\}", completion)
    if completion.count("\\boxed") != 1 or len(answers) != 1:
        return 0.0
    answer = parse_number(answers[0])
    return float(answer is not None and answer == truth)


def reward_func(queries, prompts, labels):
    import torch
    if not (len(queries) == len(prompts) == len(labels)):
        raise ValueError("Reward batch lengths differ")
    scores = []
    for query, prompt, label in zip(queries, prompts, labels):
        if not query.startswith(prompt):
            raise ValueError("Decoded query does not start with its prompt")
        scores.append(score_completion(query[len(prompt):], label))
    return torch.tensor(scores, dtype=torch.float32)


def smoke_test():
    examples = [r"\boxed{2}", r"\boxed{3}", r"\boxed{x}", r"\boxed{2", r"\boxed{2} \boxed{3}"]
    actual = reward_func(["prompt" + e for e in examples], ["prompt"] * 5, ["2"] * 5)
    if actual.tolist() != [1, 0, 0, 0, 0]:
        raise RuntimeError("Reward callback smoke test failed")
