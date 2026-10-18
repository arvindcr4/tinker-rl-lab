"""Candidate rescue statistics v2; historical definitions remain hash-preserved."""
from __future__ import annotations
import math
from . import rescue_statistics_historical_v1 as historical

PROTOCOL_VERSION = "rescue-statistics-v2-candidate"

def binomial_counts(k, n):
    if type(k) is not int or type(n) is not int or not 0 <= k <= n:
        raise ValueError("Invalid integer binomial counts")

def wilson(k: int, n: int, z: float = 1.959963984540054):
    binomial_counts(k, n)
    if type(z) not in (int, float) or not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    return historical.wilson(k, n, z)

def binomial_summary(k: int, n: int):
    binomial_counts(k, n)
    return {"rescued": k, "initially_homogeneous": n,
            "fraction": k / n if n else None, "wilson_95": wilson(k, n),
            "zero_events_exact_upper_one_sided_95": 1 - .05 ** (1 / n) if n and k == 0 else None}

logbeta = historical.logbeta

def rescue_prediction(k: int, n: int, m: int, prior: float):
    binomial_counts(k, n)
    if n <= 0 or type(m) is not int or m < 0:
        raise ValueError("positive initial size and nonnegative integer fresh draws required")
    if type(prior) not in (float, int) or not math.isfinite(prior) or prior <= 0:
        raise ValueError("finite positive prior required")
    return historical.rescue_prediction(k, n, m, prior)
