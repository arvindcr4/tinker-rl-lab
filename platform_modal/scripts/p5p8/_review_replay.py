"""Conditional replay inference, version review19-v2. Not training-seed inference."""

import hashlib
import random
import statistics


def stable_seed(value):
    return int.from_bytes(hashlib.sha256(repr(value).encode()).digest()[:8], "big")


def ratio_bootstrap(numerators, denominators, B=2000, seed=42):
    """Ratio of sums in every paired replicate, with unavailable zero support."""
    if len(numerators) != len(denominators) or not denominators:
        raise ValueError("nonempty paired arrays required")
    total = sum(denominators)
    if total <= 0:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    n = len(denominators)
    draws = []
    for _ in range(B):
        idx = rng.choices(range(n), k=n)
        den = sum(denominators[i] for i in idx)
        if den > 0:
            draws.append(sum(numerators[i] for i in idx) / den)
    draws.sort()
    return (
        sum(numerators) / total,
        draws[int(0.025 * len(draws))],
        draws[min(len(draws) - 1, int(0.975 * len(draws)))],
    )


def group_gap_bootstrap(fired, not_fired, B=2000, seed=42):
    """Conditional two-group bootstrap; recompute each group mean."""
    if not fired or not not_fired:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    draws = sorted(
        statistics.mean(rng.choices(fired, k=len(fired)))
        - statistics.mean(rng.choices(not_fired, k=len(not_fired)))
        for _ in range(B)
    )
    return (
        statistics.mean(fired) - statistics.mean(not_fired),
        draws[int(0.025 * B)],
        draws[min(B - 1, int(0.975 * B))],
    )


def dominance(lo_ret, hi_ret, lo_cost, hi_cost):
    if lo_ret > 0 and hi_cost < 0:
        return "C4"
    if hi_ret < 0 and lo_cost > 0:
        return "vs_C"
    return "UNRESOLVED_OR_TRADEOFF"
