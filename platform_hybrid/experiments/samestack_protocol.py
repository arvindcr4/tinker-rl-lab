"""Versioned, platform-neutral execution helpers; importing never launches a run."""
from __future__ import annotations

import hashlib
import json
import random
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

PROTOCOL_ID = "samestack-gsm8k-cot-v4-policy-exact-eos-grouping"
GRADING_SPEC_ID = "boxed-exact-rational-v2"
ANALYSIS_PLAN_ID = "samestack-v3-seed-paired-analysis-v1"
PROTOCOL_SPEC = {
    "protocol_id": PROTOCOL_ID, "grading_spec_id": GRADING_SPEC_ID,
    "analysis_plan_id": ANALYSIS_PLAN_ID,
    "generation": {"temperature": 1.0, "top_p": 1.0, "top_k": 0,
                   "repetition_penalty": 1.0, "num_beams": 1},
    "generation_defaults": "fresh-configs-on-wrapper-and-underlying-model",
    "mask": "include-first-eos-exclude-after-first-eos-v1",
    "schedule": "seed-shared-step-prefix-independent-warmup-sha256-52bit-v1",
    "ppo_variant": "prompt-end detached linear terminal-reward baseline; no GAE",
    "numeric_decorations": "one currency prefix/wrapper, percent suffix (coefficient), or second/seconds suffix",
    "numeric_grouping": "complete numeric atoms; commas only between three-digit integer groups",
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


PROTOCOL_HASH = canonical_hash(PROTOCOL_SPEC)
_NUMBER = re.compile(r"[+-]?(?:(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
_BOX = re.compile(r"\\boxed\s*\{")
_LATEX_FRACTION = re.compile(r"([+-]?)\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}\Z")


def exact_number(value):
    """Exact GSM8K numeric coefficient with a narrow full-string decoration grammar.

    One of: leading dollar/escaped-dollar or matched dollar wrapper, trailing
    percent/escaped-percent, or trailing second/seconds (plain or LaTeX text).
    Percent means the numeric coefficient, not division by 100. Other units or
    prose are rejected; fraction operands cannot carry decorations.
    """
    if not isinstance(value, str) or len(value) > 4096:
        return None
    value = re.sub(r"\\[!,;: ]", "", value).strip()
    decorations = 0
    if value.startswith("$") and value.endswith("$") and len(value) >= 2:
        value = value[1:-1].strip()
        decorations += 1
    elif value.startswith((r"\$", "$")):
        value = value[2 if value.startswith(r"\$") else 1:].strip()
        decorations += 1
    percent = re.search(r"(?:\\%|%)\s*\Z", value)
    if percent:
        value = value[:percent.start()].strip()
        decorations += 1
    unit = re.search(r"(?:\\text\{\s*(?:second|seconds)\s*\}|\s+(?:second|seconds))\s*\Z", value)
    if unit:
        value = value[:unit.start()].strip()
        decorations += 1
    if decorations > 1:
        return None
    return _numeric_core(value)


def _numeric_core(value):
    match = _LATEX_FRACTION.fullmatch(value)
    if match:
        numerator, denominator = _numeric_core(match[2].strip()), _numeric_core(match[3].strip())
        if numerator is None or denominator is None or denominator == 0:
            return None
        return (-1 if match[1] == "-" else 1) * numerator / denominator
    if value.count("/") == 1:
        numerator, denominator = (_numeric_core(v.strip()) for v in value.split("/"))
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator
    # Validate each complete atom before removing integer-group separators.
    # A substring replacement could turn 42.0,000 or 1e0,000 into valid values.
    if not _NUMBER.fullmatch(value):
        return None
    try:
        number = Decimal(value.replace(",", ""))
        if not number.is_finite() or abs(number.adjusted()) > 10000:
            return None
        return Fraction(number)
    except (InvalidOperation, ValueError, OverflowError, ZeroDivisionError):
        return None


def last_boxed_content(text):
    """Last valid exact boxed command with immediately following balanced braces."""
    found = None
    for match in _BOX.finditer(text):
        start, depth = match.end(), 1
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    found = text[start:index]
                    break
    return found


def boxed_reward(text, gold):
    boxed = last_boxed_content(text)
    predicted = exact_number(boxed) if boxed is not None else None
    expected = exact_number(gold)
    return float(predicted is not None and expected is not None and predicted == expected)


def resolved_token_ids(tokenizer, model):
    """Resolve stops independently of inherited sampling defaults."""
    candidates = []
    for source in (tokenizer, model.config, model.generation_config):
        value = getattr(source, "eos_token_id", None)
        candidates.extend(value if isinstance(value, (list, tuple)) else [value])
    vocab = tokenizer.get_vocab()
    if "<|im_end|>" in vocab:
        candidates.append(vocab["<|im_end|>"])
    eos = sorted({int(i) for i in candidates if isinstance(i, int) and i >= 0})
    if not eos:
        raise ValueError("The protocol requires at least one resolved EOS token")
    pad = tokenizer.pad_token_id
    if pad is None:
        tokenizer.pad_token_id = eos[0]
        pad = eos[0]
    return eos, int(pad)


def generation_configs(eos_ids, pad_id, max_new_tokens):
    """Fresh configs: no inherited top-k, repetition or other logits processors."""
    from transformers import GenerationConfig
    common = dict(PROTOCOL_SPEC["generation"], eos_token_id=list(eos_ids),
                  pad_token_id=int(pad_id), max_new_tokens=int(max_new_tokens), use_cache=True,
                  min_length=0, min_new_tokens=0, typical_p=1.0, epsilon_cutoff=0.0,
                  eta_cutoff=0.0, no_repeat_ngram_size=0, num_return_sequences=1,
                  renormalize_logits=False, remove_invalid_values=False)
    return (GenerationConfig(**common, do_sample=True),
            GenerationConfig(**common, do_sample=False))


def clear_inherited_generation_defaults(model, train_config):
    """Transformers 5 merges None fields from model.generation_config.

    Replace that source with our fresh config too, so optional processors such
    as suppress_tokens/forced_eos cannot leak back in during generate().
    """
    import copy
    model.generation_config = copy.deepcopy(train_config)
    # PEFT delegates generate() to its underlying pretrained model. Set both
    # explicitly rather than relying on a wrapper's attribute-forwarding rules.
    if callable(getattr(model, "get_base_model", None)):
        model.get_base_model().generation_config = copy.deepcopy(train_config)


def completion_mask(completion_ids, eos_ids):
    """Include first stop even for pad==EOS; exclude all subsequent padding."""
    import torch
    stop = torch.zeros_like(completion_ids, dtype=torch.bool)
    for eos in eos_ids:
        stop |= completion_ids.eq(eos)
    return (stop.long().cumsum(-1) - stop.long()).eq(0)


def completion_mask_rows(rows, eos_ids):
    stop_ids, output = set(eos_ids), []
    for row in rows:
        stopped, mask = False, []
        for token in row:
            mask.append(not stopped)
            stopped = stopped or token in stop_ids
        output.append(mask)
    return output


def position_ids_from_attention(attention_mask):
    positions = attention_mask.long().cumsum(-1) - 1
    return positions.masked_fill(attention_mask.eq(0), 1)


def stream_seed(seed, stream):
    # 52 bits stay exact in both Python and JavaScript JSON consumers.
    return int(hashlib.sha256(f"{PROTOCOL_ID}:{seed}:{stream}".encode()).hexdigest()[:13], 16)


def prompt_schedules(seed, train_size, n_steps, warm_batches, prompts_per_step, max_prompts=64):
    """Shared per-seed policy step prefixes; independent warmup RNG and prompts."""
    if train_size < 1 or not 1 <= prompts_per_step <= max_prompts:
        raise ValueError("Invalid prompt schedule dimensions")
    def rows(stream, count, width):
        rng = random.Random(stream_seed(seed, stream))
        pool, cursor, result = [], 0, []
        for _ in range(count):
            row = []
            for _ in range(width):
                if cursor == len(pool):
                    pool = list(range(train_size)); rng.shuffle(pool); cursor = 0
                row.append(pool[cursor]); cursor += 1
            result.append(row)
        return result
    return {"policy": [row[:prompts_per_step] for row in rows("policy-prompts", n_steps, max_prompts)],
            "critic_warmup": rows("critic-warmup-prompts", warm_batches, prompts_per_step),
            "policy_sampling_seed": stream_seed(seed, "policy-sampling"),
            "warmup_sampling_seed": stream_seed(seed, "critic-warmup-sampling")}


def budget_accounting(n_steps, n_gen, group, warm_batches):
    policy_prompts, warm_prompts = n_steps * (n_gen // group), warm_batches * (n_gen // group)
    return {"policy_prompt_presentations": policy_prompts,
            "critic_warmup_prompt_presentations": warm_prompts,
            "total_prompt_presentations": policy_prompts + warm_prompts,
            "policy_generations": n_steps * n_gen,
            "critic_warmup_generations": warm_batches * n_gen,
            "total_generations": (n_steps + warm_batches) * n_gen,
            "total_exposure_matched": False,
            "matching_scope": "grpo_g8 and ppo_8x8 policy-update prompt schedule only"}


def strict_json_dump(value, handle, **kwargs):
    json.dump(value, handle, allow_nan=False, **kwargs)


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()
