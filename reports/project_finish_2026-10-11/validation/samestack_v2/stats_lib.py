"""Pure seed-level statistics. Sign flips assume independent pairs and symmetric
exchangeable signs, not randomized treatment assignment. Five nonzero pairs
have minimum two-sided exact p = 2 / 2**5 = .0625.
"""

import itertools
import math
import numbers

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from scipy.integrate import quad


def differences(d):
    a = np.asarray(d, dtype=float)
    if a.ndim != 1 or a.size < 2:
        raise ValueError("at least two paired differences are required")
    if not np.isfinite(a).all():
        raise ValueError("finite paired differences are required")
    return a


def _probability(value, name, *, inclusive=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite probability")
    if not (0 <= value <= 1 if inclusive else 0 < value < 1):
        raise ValueError(f"{name} is outside its probability range")
    return float(value)


def signflip(d):
    d = differences(d)
    if len(d) > 20:
        raise ValueError("exact sign-flip enumeration supports at most 20 pairs")
    # Scale-invariant statistic avoids overflow for large finite differences.
    scale = float(np.max(np.abs(d)))
    if scale:
        d = d / scale
    obs = abs(float(d.sum()))
    tol = np.finfo(float).eps * max(float(np.abs(d).sum()), np.finfo(float).tiny) * len(d) * 4
    hits = sum(
        abs(float((np.array(s) * d).sum())) >= obs - tol
        for s in itertools.product([1, -1], repeat=len(d))
    )
    return hits / 2 ** len(d)


def boot_ci(d, B=20000, seed=0):
    d = differences(d)
    if isinstance(B, bool) or not isinstance(B, numbers.Integral) or B < 1:
        raise ValueError("B must be a positive integer")
    rng = np.random.default_rng(seed)
    scale = float(np.max(np.abs(d)))
    normalized = d / scale if scale else d
    means = normalized[rng.integers(0, len(d), (B, len(d)))].mean(1) * scale
    return np.percentile(means, [2.5, 97.5]).tolist()


def paired(a, b, margin=0.02):
    """Paired seed inference; 90% CI corresponds to alpha=.05 TOST.

    The margin is a sensitivity parameter, not a justified equivalence claim.
    """
    if (
        isinstance(margin, bool)
        or not isinstance(margin, numbers.Real)
        or not math.isfinite(margin)
        or margin <= 0
    ):
        raise ValueError("margin must be finite and positive")
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        raise ValueError("paired requires at least two shared seeds")
    av, bv = np.asarray([a[s] for s in seeds], float), np.asarray([b[s] for s in seeds], float)
    if not np.isfinite(av).all() or not np.isfinite(bv).all():
        raise ValueError("finite paired differences are required")
    d = differences(av - bv)
    n = len(d)
    scale = float(np.max(np.abs(d)))
    normalized = d / scale if scale else d
    md, sd = (
        (float(d[0]), 0.0)
        if np.all(d == d[0])
        else (float(normalized.mean() * scale), float(normalized.std(ddof=1) * scale))
    )
    se = sd / math.sqrt(n)
    if not math.isfinite(md) or not math.isfinite(sd):
        raise ValueError("paired summary overflowed; rescale the finite inputs")
    if sd == 0:
        p_t = 1.0 if md == 0 else 0.0
        p_lo = 0.0 if md > -margin else 0.5 if md == -margin else 1.0
        p_hi = 0.0 if md < margin else 0.5 if md == margin else 1.0
        ci95, ci90 = [md, md], [md, md]
        t = 0.0 if md == 0 else None
    else:
        t = md / se
        p_t = float(2 * stats.t.sf(abs(t), n - 1))
        p_lo = float(stats.t.sf((md + margin) / se, n - 1))
        p_hi = float(stats.t.cdf((md - margin) / se, n - 1))
        ci95 = [md - stats.t.ppf(0.975, n - 1) * se, md + stats.t.ppf(0.975, n - 1) * se]
        ci90 = [md - stats.t.ppf(0.95, n - 1) * se, md + stats.t.ppf(0.95, n - 1) * se]
    if not all(math.isfinite(value) for value in ci95 + ci90):
        raise ValueError("paired confidence interval overflowed; rescale inputs")
    directional_p = (
        float(stats.t.cdf((md - 0.05) / se, n - 1))
        if se
        else (0.0 if md < 0.05 else 0.5 if md == 0.05 else 1.0)
    )
    upper = float(md + stats.t.ppf(0.95, n - 1) * se)
    return {
        "difference_ci_95": list(map(float, ci95)),
        "equivalence_ci_90": list(map(float, ci90)),
        "equivalence_margin": float(margin),
        "positive_5pp_test": {
            "null_hypothesis": "mean >= +0.05",
            "alternative_hypothesis": "mean < +0.05",
            "p_value": directional_p,
            "upper_bound_95": upper,
            "reject_at_0_05": directional_p < 0.05,
        },
        "n": n,
        "seeds": seeds,
        "diffs": d.tolist(),
        "mean": md,
        "sd": sd,
        "unmatched_a_seeds": sorted(set(a) - set(b)),
        "unmatched_b_seeds": sorted(set(b) - set(a)),
        "t": t,
        "zero_variance": sd == 0,
        "t_ci": list(map(float, ci95)),
        "tost_ci90": list(map(float, ci90)),
        "boot_ci": boot_ci(d),
        "p_t": p_t,
        "p_signflip": signflip(d),
        "tost_margin": float(margin),
        "tost_p": max(p_lo, p_hi),
        "tost_lower_p": p_lo,
        "tost_upper_p": p_hi,
        "within_margin_unadjusted": bool(ci90[0] > -margin and ci90[1] < margin),
    }


def holm(ps):
    """Holm adjustment; callers must supply the entire fixed family."""
    checked = {key: _probability(p, f"p[{key}]", inclusive=True) for key, p in ps.items()}
    order = sorted(checked, key=checked.get)
    out, running = {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * checked[key]))
        out[key] = running
    return out


def mcnemar_exact(b, c):
    if any(isinstance(x, bool) or not isinstance(x, numbers.Integral) or x < 0 for x in (b, c)):
        raise ValueError("discordant counts must be nonnegative integers")
    return float(stats.binomtest(b, b + c).pvalue) if b + c else 1.0


def _power_inputs(sd, n, alpha):
    if isinstance(sd, bool) or not isinstance(sd, numbers.Real) or not math.isfinite(sd) or sd <= 0:
        raise ValueError("sd must be finite and strictly positive")
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 2:
        raise ValueError("n must be an integer >= 2")
    _probability(alpha, "alpha")


def paired_t_power(sd, n, effect, alpha=0.05):
    """Unadjusted paired-t power, including BOTH noncentral-t tails."""
    _power_inputs(sd, n, alpha)
    if (
        isinstance(effect, bool)
        or not isinstance(effect, numbers.Real)
        or not math.isfinite(effect)
    ):
        raise ValueError("effect must be finite")
    df = n - 1
    critical = stats.t.ppf(1 - alpha / 2, df)
    noncentrality = (abs(effect) / sd) * math.sqrt(n)
    if not math.isfinite(noncentrality):
        raise ValueError("noncentrality overflowed; rescale inputs")
    upper = float(stats.nct.sf(critical, df, noncentrality))
    lower = float(stats.nct.cdf(-critical, df, noncentrality))
    # Some SciPy versions produce NaN for the tiny lower tail at df=1,
    # positive noncentrality. Integrate the defining normal/chi mixture,
    # rather than dropping the second tail or failing the n=2 root bracket.
    if not math.isfinite(lower):
        lower = quad(
            lambda radius: (
                stats.norm.cdf(-critical * radius / math.sqrt(df) - noncentrality)
                * stats.chi.pdf(radius, df)
            ),
            0,
            math.inf,
            epsabs=1e-14,
        )[0]
    if not math.isfinite(upper):
        upper = quad(
            lambda radius: (
                stats.norm.sf(critical * radius / math.sqrt(df) - noncentrality)
                * stats.chi.pdf(radius, df)
            ),
            0,
            math.inf,
            epsabs=1e-12,
        )[0]
    result = upper + lower
    if not math.isfinite(result):
        raise ValueError("noncentral-t power calculation was nonfinite")
    return min(1.0, max(0.0, result))


def mde(sd, n, power=0.8, alpha=0.05):
    """Pilot SD sensitivity for one t test; NOT family/TOST design power."""
    _power_inputs(sd, n, alpha)
    _probability(power, "power")
    if power <= alpha:
        raise ValueError("target power must exceed alpha")
    high = float(sd)
    while paired_t_power(sd, n, high, alpha) < power:
        high *= 2
        if not math.isfinite(high):
            raise ValueError("could not bracket a finite MDE")
    root = brentq(
        lambda ratio: paired_t_power(sd, n, ratio * sd, alpha) - power, 0.0, high / sd, xtol=1e-12
    )
    return float(root * sd)


def n_needed(sd, effect, power=0.8, alpha=0.05, max_n=100000):
    _power_inputs(sd, 2, alpha)
    _probability(power, "power")
    if power <= alpha:
        raise ValueError("target power must exceed alpha")
    if (
        isinstance(effect, bool)
        or not isinstance(effect, numbers.Real)
        or not math.isfinite(effect)
        or effect <= 0
    ):
        raise ValueError("effect must be finite and strictly positive")
    if isinstance(max_n, bool) or not isinstance(max_n, numbers.Integral) or max_n < 2:
        raise ValueError("max_n must be an integer >= 2")
    if paired_t_power(sd, 2, effect, alpha) >= power:
        return 2
    low, high = 2, min(4, max_n)
    while paired_t_power(sd, high, effect, alpha) < power:
        if high == max_n:
            raise ValueError(f"required sample size exceeds max_n={max_n}")
        low, high = high, min(2 * high, max_n)
    while high - low > 1:
        middle = (high + low) // 2
        if paired_t_power(sd, middle, effect, alpha) >= power:
            high = middle
        else:
            low = middle
    return high
