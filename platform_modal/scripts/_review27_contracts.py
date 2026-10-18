"""Measurement contracts for review27-v2. Pure, offline, deterministic."""

import math
import hashlib


def exact_iso_g(p, target):
    """Smallest integer G>=1 reaching IID contrast yield; inf if unattainable."""
    if not all(math.isfinite(v) for v in (p, target)) or not 0 <= p <= 1 or not 0 <= target <= 1:
        raise ValueError("p and target must be finite probabilities")
    if target == 0:
        return 1
    if p in (0, 1) or target == 1:
        return math.inf

    def reaches(g):
        return p**g + (1 - p) ** g <= 1 - target

    upper = 2
    while not reaches(upper):
        upper *= 2
    lower = 1
    while lower < upper:
        middle = (lower + upper) // 2
        if reaches(middle):
            upper = middle
        else:
            lower = middle + 1
    return lower


def stable_seed(master, identity):
    return int.from_bytes(
        hashlib.sha256(f"review27-v2:{master}:{identity}".encode()).digest()[:8], "big"
    )


def validated_groups(groups):
    groups = [list(row) for row in groups]
    if not groups:
        raise ValueError("no observed groups")
    for row in groups:
        if len(row) < 2:
            raise ValueError("within-group variance needs at least two rewards")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in row
        ):
            raise ValueError("malformed or nonfinite group reward")
    return groups
