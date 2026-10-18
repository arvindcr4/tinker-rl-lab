"""Executable silent reward-channel regimes grounded in the frozen E1 parser contract."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

HISTORICAL_E1_SOURCE_SHA256 = "986811e3e78fe86ffcbede4a98599ada167ff1975b3341eef391eb2b2e7fe8c6"

# Candidate source adds execution-identity receipt fields; historical pin stays immutable.
E1_SOURCE_SHA256 = "0933f1f15e4e666aaa738423ca8e788c8ec8d7bd9bb5e62df43b180f0c423d45"
ANALOGUE_PROTOCOL_ID = "reward-channel-witness-v2-execution-identity"


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
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 2:
        raise ValueError("group_size must be at least two")
    verify_e1_source()
    gold = "reasoning\n#### 42"
    primary_completions = [f"attempt {index}\n#### {-index - 1}" for index in range(group_size)]
    calibration = "known-correct control\n#### 42"
    clean_marker, broken_marker = "####", "FINAL:"
    clean_primary = [exact_reward(text, gold, marker=clean_marker) for text in primary_completions]
    broken_primary = [
        exact_reward(text, gold, marker=broken_marker) for text in primary_completions
    ]
    if clean_primary != broken_primary or clean_primary != [0.0] * group_size:
        raise RuntimeError("conditional witness requires matched all-zero rewards")
    telemetry = {
        "runtime_error": None,
        "latency_bucket": "normal",
        "reward_code_version": "matched",
    }
    return {
        "protocol_id": ANALOGUE_PROTOCOL_ID,
        "historical_source_sha256": HISTORICAL_E1_SOURCE_SHA256,
        "claim_scope": "conditional action ambiguity after all-zero rewards; local analogue only",
        "production_calibration_validated": False,
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
