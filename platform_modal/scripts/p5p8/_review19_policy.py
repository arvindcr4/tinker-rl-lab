"""Shared corrected-review19-v4 controller definitions; model replay only."""
import math

POLICY_VERSION = "corrected-review19-v4"


def high_zvf_fire(z, tau):
    return z >= tau


def adaptive_target(zvf_obs, G_obs=8, G_candidates=(16, 32, 64), floor=0.50):
    """Iter119 policy: smallest candidate strictly below max(floor, z/2).

    The largest candidate is a fallback action, not a feasibility certificate.
    """
    if not math.isfinite(zvf_obs) or not 0 <= zvf_obs <= 1:
        raise ValueError("ZVF must be finite and in [0, 1]")
    if zvf_obs <= floor:
        return G_obs
    lo, hi = 0.0, 0.5
    for _ in range(80):
        mid = (lo + hi) / 2
        if mid**G_obs + (1-mid)**G_obs > zvf_obs:
            lo = mid
        else:
            hi = mid
    prob = 0.0 if zvf_obs == 1 else (lo + hi) / 2
    target = max(floor, 0.5 * zvf_obs)
    return next((g for g in G_candidates if prob**g + (1-prob)**g < target), G_candidates[-1])
