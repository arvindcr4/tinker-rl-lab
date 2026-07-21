"""Executable silent reward-channel regimes grounded in the frozen E1 parser contract."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

E1_SOURCE_SHA256 = "986811e3e78fe86ffcbede4a98599ada167ff1975b3341eef391eb2b2e7fe8c6"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def verify_e1_source() -> Path:
    path = _repo_root() / "zvf-program/colab-experiments/e1_grpo_confirmatory.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != E1_SOURCE_SHA256:
        raise RuntimeError(f"E1 reward source drifted: {digest}")
    return path


def parse_marked_integer(text: str, *, marker: str) -> int | None:
    """Configurable analogue of the frozen E1 final-marker parser."""
    if marker not in text:
        return None
    tail = text.rsplit(marker, 1)[1]
    match = re.search(r"-?\d[\d,]*", tail)
    if not match:
        return None
    try:
        return int(match.group(0).replace(",", ""))
    except ValueError:
        return None


def exact_reward(completion: str, gold: str, *, marker: str) -> float:
    prediction = parse_marked_integer(completion, marker=marker)
    target = parse_marked_integer(gold, marker=marker)
    return float(prediction is not None and prediction == target)


def regime_receipt(group_size: int = 8) -> dict[str, object]:
    """Return matched primary outcomes and one same-path calibration outcome."""
    if group_size < 2:
        raise ValueError("group_size must be at least two")
    verify_e1_source()
    gold = "reasoning\n#### 42"
    primary_completions = [f"attempt {index}\n#### {index}" for index in range(group_size)]
    calibration = "known-correct control\n#### 42"
    clean_marker, broken_marker = "####", "FINAL:"
    clean_primary = [exact_reward(text, gold, marker=clean_marker) for text in primary_completions]
    broken_primary = [exact_reward(text, gold, marker=broken_marker) for text in primary_completions]
    telemetry = {
        "runtime_error": None,
        "latency_bucket": "normal",
        "reward_code_version": "matched",
    }
    return {
        "source": {
            "path": "zvf-program/colab-experiments/e1_grpo_confirmatory.py",
            "sha256": E1_SOURCE_SHA256,
            "functions": ["parse_marked_integer", "gsm8k_exact_reward"],
        },
        "clean_hard": {
            "marker": clean_marker,
            "primary_rewards": clean_primary,
            "calibration_reward": exact_reward(calibration, gold, marker=clean_marker),
            "telemetry": telemetry,
        },
        "silent_marker_mismatch": {
            "marker": broken_marker,
            "primary_rewards": broken_primary,
            "calibration_reward": exact_reward(calibration, gold, marker=broken_marker),
            "telemetry": telemetry,
        },
    }
