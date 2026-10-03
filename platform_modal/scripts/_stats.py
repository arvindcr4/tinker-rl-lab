"""Shared statistics primitives for the platform_modal analysis scripts.

Every helper reproduces, bit for bit, the copy it replaced in the p5/p6/p7/synth
scripts. Callers keep their own z literal, n == 0 handling, tuple order and
bootstrap size/seed; nothing here reads module globals or picks defaults.

Usage from a script one level below (``p5p8/``)::

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from _stats import wilson
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from statistics import fmean, pstdev


def pow_root(v):
    """``v ** 0.5``: not always equal to ``math.sqrt`` in the last ULP."""
    return v**0.5


def wilson_centre_half(p, n, z, root=math.sqrt):
    """Unclipped Wilson score centre and half-width for proportion ``p`` over ``n > 0``."""
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * root(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre, half


def wilson_p(p, n, z, root=math.sqrt):
    """Wilson score interval ``(lo, hi)`` clipped to [0, 1]; ``n > 0``."""
    centre, half = wilson_centre_half(p, n, z, root)
    return max(0.0, centre - half), min(1.0, centre + half)


def wilson(k, n, z, root=math.sqrt):
    """``(p, lo, hi)`` Wilson score interval for ``k`` successes out of ``n > 0``."""
    p = k / n
    return (p, *wilson_p(p, n, z, root))


def wilson_p_factored(p, n, z):
    """Wilson ``(lo, hi)`` with the half-width written as sqrt((p(1-p) + z^2/4n) / n).

    Algebraically equal to ``wilson_p`` but differs from it in the last ULP, so the
    scripts that used this form keep it.
    """
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def log_beta(a, b):
    """log B(a, b) = lgamma(a) + lgamma(b) - lgamma(a + b)."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def cohens_d_pstdev_pooled(a, b):
    """Cohen's d with population-stdev pooling; nan below n=2 or for zero spread."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = fmean(a), fmean(b)
    sa, sb = pstdev(a), pstdev(b)
    sp = math.sqrt(((len(a) - 1) * sa * sa + (len(b) - 1) * sb * sb) / (len(a) + len(b) - 2))
    return (ma - mb) / sp if sp > 1e-12 else float("nan")


def bootstrap_ci_mean_pct(values, B, alpha, seed):
    """Percentile bootstrap CI on the mean: ``(mean, lo, hi, n)``; nan-tuple if empty."""
    if not values:
        return float("nan"), float("nan"), float("nan"), 0
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(B):
        s = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo = means[int(B * alpha / 2)]
    hi = means[int(B * (1 - alpha / 2))]
    return sum(values) / n, lo, hi, n


def bootstrap_ci_statmean(values, boot, seed):
    """95% bootstrap CI via ``statistics.mean``: ``(mean, lo, hi)``; zeros if empty."""
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    pts = []
    for _ in range(boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        pts.append(statistics.mean(sample))
    pts.sort()
    return (statistics.mean(values), pts[int(0.025 * boot)], pts[int(0.975 * boot)])


def bootstrap_ci_rng(values, n_boot, alpha, rng):
    """Bootstrap CI on the mean drawing from ``rng``: ``(mean, lo, hi)``; zeros if n < 2."""
    n = len(values)
    if n < 2:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_boot):
        s = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return sum(values) / n, lo, hi


def paired_boot_pct(dv, dg, n_boot, seed):
    """Paired bootstrap on ``dv - dg``: ``(delta, lo, hi, n)``.

    Percentile indices are ``[int(0.025 * n_boot), int(0.975 * n_boot) - 1]``.
    """
    d = [a - b for a, b in zip(dv, dg)]
    rng = random.Random(seed)
    n = len(d)
    means = []
    for _ in range(n_boot):
        s = [d[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return sum(d) / n, lo, hi, n


def paired_step_bootstrap(rows, fn, b, seed):
    """Resample steps with replacement, keeping every row of a picked step together.

    Returns the non-None, non-nan values of ``fn(sample)`` over ``b`` resamples.
    """
    rng = random.Random(seed)
    by_step = defaultdict(list)
    for r in rows:
        by_step[r["step"]].append(r)
    steps = sorted(by_step.keys())
    n_steps = len(steps)
    out = []
    for _ in range(b):
        pick = [rng.choice(steps) for _ in range(n_steps)]
        sample = []
        for s in pick:
            sample.extend(by_step[s])
        v = fn(sample)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            out.append(v)
    return out
