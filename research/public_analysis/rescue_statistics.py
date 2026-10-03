"""Public derivative: extracted scientific functions; operational code removed.

Selected definitions are unchanged from the retained source identified in
PROVENANCE.json. This module alone cannot authenticate empirical evidence.
"""
from __future__ import annotations

import math

def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    if not 0 <= k <= n:
        raise ValueError("Invalid binomial counts")
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, center - half), min(1.0, center + half)]


def binomial_summary(k: int, n: int) -> dict:
    return {
        "rescued": k, "initially_homogeneous": n,
        "fraction": k / n if n else None,
        "wilson_95": wilson(k, n),
        "zero_events_exact_upper_one_sided_95": 1 - 0.05 ** (1 / n) if n and k == 0 else None,
    }


def logbeta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def rescue_prediction(k: int, n: int, m: int, prior: float) -> float:
    """Mixedness of combined original+fresh draws, given homogeneous original."""
    if k not in (0, n):
        raise ValueError("Prediction conditions on a homogeneous initial group")
    a, b = k + prior, n - k + prior
    unchanged = math.exp(logbeta(a, b + m) - logbeta(a, b)) if k == 0 else math.exp(logbeta(a + m, b) - logbeta(a, b))
    return 1 - unchanged
