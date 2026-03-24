"""
Statistical analysis utilities for GRPO training results.

Each logged training step produces a mean reward = k/n where:
  n = batch_size = 128 (8 groups × 16 completions)
  k = number of correct completions out of 128

Tests implemented
-----------------
bootstrap_ci         Bootstrap 95% CI for a list of proportions.
two_prop_ztest       Two-proportion z-test: did accuracy change between two steps?
spearman_trend       Spearman rank-correlation of reward vs step (monotonic trend test).
mannwhitney          Mann-Whitney U: compare two reward sequences non-parametrically.
cohen_d              Cohen's d effect size between two reward windows.
chow_test            Chow structural-break F-test at a given candidate breakpoint.
find_phase_transition Exhaustive search for the breakpoint maximising Chow F.
run_full_analysis    Run all tests for a single experiment and return a results dict.
print_report         Pretty-print the results dict.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _reconstruct_binary(mean_reward: float, n: int = 128) -> List[int]:
    """Convert a step mean reward to n binary {0,1} values."""
    k = round(mean_reward * n)
    return [1] * k + [0] * (n - k)


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float], ddof: int = 1) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - ddof)
    return math.sqrt(var)


# ---------------------------------------------------------------------------
# 1. Bootstrap 95% CI on step-level means
# ---------------------------------------------------------------------------

def bootstrap_ci(
    step_rewards: List[float],
    n_per_step: int = 128,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap CI for the *overall mean* across all provided step rewards.

    Parameters
    ----------
    step_rewards : list of per-step mean rewards (each is k/n_per_step)
    n_per_step   : number of binary samples per step (default 128)

    Returns
    -------
    (lower, mean, upper)
    """
    rng = random.Random(seed)
    # Pool all binary observations
    pool: List[int] = []
    for r in step_rewards:
        pool.extend(_reconstruct_binary(r, n_per_step))

    n = len(pool)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = [pool[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    alpha = 1 - confidence
    lo = boot_means[int(alpha / 2 * n_bootstrap)]
    hi = boot_means[int((1 - alpha / 2) * n_bootstrap)]
    return lo, _mean(pool), hi


def bootstrap_ci_per_step(
    step_rewards: List[float],
    n_per_step: int = 128,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> List[Tuple[float, float, float]]:
    """Return (lower, mean, upper) for each individual step."""
    return [
        bootstrap_ci([r], n_per_step, n_bootstrap, confidence, seed + i)
        for i, r in enumerate(step_rewards)
    ]


# ---------------------------------------------------------------------------
# 2. Two-proportion z-test
# ---------------------------------------------------------------------------

def two_prop_ztest(
    p1: float,
    p2: float,
    n1: int = 128,
    n2: int = 128,
) -> Tuple[float, float]:
    """
    Two-proportion z-test: H0: p1 == p2 (two-sided).

    Returns (z_statistic, p_value).
    Directional (one-sided, H1: p2 > p1): halve p_value.
    """
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("inf"), 0.0
    z = (p2 - p1) / se
    # Approximate p-value from standard normal (two-sided)
    p_value = 2 * (1 - _standard_normal_cdf(abs(z)))
    return z, p_value


def _standard_normal_cdf(z: float) -> float:
    """Approximation of Phi(z) accurate to ~1e-7."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ---------------------------------------------------------------------------
# 3. Spearman rank-correlation (trend test)
# ---------------------------------------------------------------------------

def spearman_trend(
    steps: List[int],
    rewards: List[float],
) -> Tuple[float, float]:
    """
    Spearman rank-correlation of rewards vs steps.
    H0: rho = 0 (no monotonic trend).

    Returns (rho, p_value).
    """
    n = len(rewards)
    assert n == len(steps), "steps and rewards must have the same length"

    def _ranks(xs):
        sorted_idx = sorted(range(len(xs)), key=lambda i: xs[i])
        ranks = [0.0] * len(xs)
        for rank, idx in enumerate(sorted_idx, 1):
            ranks[idx] = float(rank)
        return ranks

    r_steps = _ranks(steps)
    r_rewards = _ranks(rewards)

    # Pearson on ranks
    mean_s = _mean(r_steps)
    mean_r = _mean(r_rewards)
    num = sum((r_steps[i] - mean_s) * (r_rewards[i] - mean_r) for i in range(n))
    den = math.sqrt(
        sum((r_steps[i] - mean_s) ** 2 for i in range(n))
        * sum((r_rewards[i] - mean_r) ** 2 for i in range(n))
    )
    rho = num / den if den != 0 else 0.0

    # t-statistic for H0: rho=0
    if abs(rho) >= 1.0:
        return rho, 0.0
    t = rho * math.sqrt((n - 2) / (1 - rho ** 2))
    # Two-tailed p using t-distribution approximation (via normal for large n)
    p_value = 2 * (1 - _standard_normal_cdf(abs(t)))
    return rho, p_value


# ---------------------------------------------------------------------------
# 4. Mann-Whitney U
# ---------------------------------------------------------------------------

def mannwhitney(
    group_a: List[float],
    group_b: List[float],
) -> Tuple[float, float, str]:
    """
    Mann-Whitney U test: H0: distributions of A and B are equal.

    Returns (U_statistic, p_value_approx, conclusion).
    Uses normal approximation (valid for n >= 10).
    """
    na, nb = len(group_a), len(group_b)
    # Count U_A = number of (a, b) pairs where a > b
    u_a = sum(1 for a in group_a for b in group_b if a > b) + \
          sum(0.5 for a in group_a for b in group_b if a == b)
    u_b = na * nb - u_a
    u = min(u_a, u_b)

    # Normal approximation
    mu_u = na * nb / 2
    sigma_u = math.sqrt(na * nb * (na + nb + 1) / 12)
    z = (u - mu_u) / sigma_u if sigma_u > 0 else 0.0
    p_value = 2 * (1 - _standard_normal_cdf(abs(z)))

    conclusion = "SIGNIFICANT (p < 0.05)" if p_value < 0.05 else "not significant"
    return u, p_value, conclusion


# ---------------------------------------------------------------------------
# 5. Cohen's d
# ---------------------------------------------------------------------------

def cohen_d(group1: List[float], group2: List[float]) -> float:
    """
    Cohen's d = (mean2 - mean1) / pooled_std.
    Convention: group2 is the later/higher group.
    Magnitude: <0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.
    """
    n1, n2 = len(group1), len(group2)
    m1, m2 = _mean(group1), _mean(group2)
    s1, s2 = _std(group1), _std(group2)
    pooled = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    return (m2 - m1) / pooled if pooled > 0 else float("inf")


# ---------------------------------------------------------------------------
# 6. Chow structural-break test
# ---------------------------------------------------------------------------

def _ols_sse(xs: List[float], ys: List[float]) -> float:
    """SSE from simple OLS regression of ys on xs."""
    n = len(xs)
    if n < 2:
        return 0.0
    x_m, y_m = _mean(xs), _mean(ys)
    ss_xx = sum((x - x_m) ** 2 for x in xs)
    if ss_xx == 0:
        return sum((y - y_m) ** 2 for y in ys)
    b1 = sum((xs[i] - x_m) * (ys[i] - y_m) for i in range(n)) / ss_xx
    b0 = y_m - b1 * x_m
    return sum((ys[i] - (b0 + b1 * xs[i])) ** 2 for i in range(n))


def chow_test(
    steps: List[int],
    rewards: List[float],
    breakpoint: int,
) -> Tuple[float, float]:
    """
    Chow F-test for a structural break at `breakpoint` step index.

    H0: no structural break (single regression fits both segments).
    Returns (F_statistic, p_value_approx).
    """
    n = len(steps)
    assert 3 <= breakpoint <= n - 3, "breakpoint must leave at least 3 points on each side"

    s1_x, s1_y = steps[:breakpoint], rewards[:breakpoint]
    s2_x, s2_y = steps[breakpoint:], rewards[breakpoint:]

    sse_full = _ols_sse(steps, rewards)
    sse_1 = _ols_sse(s1_x, s1_y)
    sse_2 = _ols_sse(s2_x, s2_y)
    sse_restricted = sse_1 + sse_2

    k = 2  # number of parameters per segment (intercept + slope)
    df_num = k
    df_den = n - 2 * k
    if df_den <= 0 or sse_restricted == 0:
        return 0.0, 1.0

    f_stat = ((sse_full - sse_restricted) / df_num) / (sse_restricted / df_den)

    # p-value from F-distribution approximation
    p_value = _f_pvalue(f_stat, df_num, df_den)
    return f_stat, p_value


def _f_pvalue(f: float, df1: int, df2: int) -> float:
    """
    Approximate p-value for F(df1, df2) using Wilson-Hilferty normal approximation.
    Accurate enough for df1 in {2,3} and df2 > 20.
    """
    if f <= 0:
        return 1.0
    # Transform to chi-squared / df1, use normal approximation
    x = df1 * f / (df1 * f + df2)
    # Beta incomplete function approximation via normal
    a, b = df1 / 2, df2 / 2
    # Use a simple log-space approximation
    mu = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    if var == 0:
        return 0.0 if x > mu else 1.0
    z = (x - mu) / math.sqrt(var)
    return 2 * (1 - _standard_normal_cdf(abs(z)))


def find_phase_transition(
    steps: List[int],
    rewards: List[float],
    min_segment: int = 5,
) -> Dict:
    """
    Exhaustive search for the breakpoint with maximum Chow F-statistic.

    Returns dict with keys:
      breakpoint_step  – the step index (0-based) of the phase transition
      f_statistic      – F value at that breakpoint
      p_value          – corresponding p-value
      pre_mean         – mean reward before breakpoint
      post_mean        – mean reward after breakpoint
    """
    n = len(steps)
    best = {"breakpoint_step": None, "f_statistic": -1.0, "p_value": 1.0}

    for bp in range(min_segment, n - min_segment):
        try:
            f, p = chow_test(steps, rewards, bp)
            if f > best["f_statistic"]:
                best = {
                    "breakpoint_step": steps[bp],
                    "breakpoint_index": bp,
                    "f_statistic": f,
                    "p_value": p,
                    "pre_mean": _mean(rewards[:bp]),
                    "post_mean": _mean(rewards[bp:]),
                }
        except AssertionError:
            continue

    return best


# ---------------------------------------------------------------------------
# 7. Full analysis runner
# ---------------------------------------------------------------------------

def run_full_analysis(
    model_name: str,
    steps: List[int],
    rewards: List[float],
    n_per_step: int = 128,
    early_cutoff: Optional[int] = None,
) -> Dict:
    """
    Run all statistical tests for one experiment.

    Parameters
    ----------
    model_name   : label for reporting
    steps        : list of step indices (0-based integers)
    rewards      : list of per-step mean rewards
    n_per_step   : binary samples per step (default 128)
    early_cutoff : step index separating 'early' and 'late' phase
                   (auto-detected via Chow test if None)

    Returns a dict of all results.
    """
    n = len(steps)
    results: Dict = {"model": model_name, "n_steps": n, "n_per_step": n_per_step}

    # -- Basic stats --
    results["initial_reward"] = rewards[0]
    results["final_reward"] = rewards[-1]
    results["mean_first5"] = _mean(rewards[:5])
    results["mean_last5"] = _mean(rewards[-5:])
    results["peak_reward"] = max(rewards)

    # -- Bootstrap CI on initial performance (first 5 steps) --
    lo, mu, hi = bootstrap_ci(rewards[:5], n_per_step)
    results["initial_ci"] = (lo, mu, hi)

    # -- Bootstrap CI on final performance (last 10 steps) --
    lo, mu, hi = bootstrap_ci(rewards[-10:], n_per_step)
    results["final_ci"] = (lo, mu, hi)

    # -- Two-proportion z-test: step 0 vs step 49 --
    z, p = two_prop_ztest(rewards[0], rewards[-1], n_per_step, n_per_step)
    results["ztest_step0_vs_final"] = {"z": z, "p": p,
                                        "significant": p < 0.05}

    # -- Two-proportion z-test: first5 mean vs last5 mean --
    z2, p2 = two_prop_ztest(
        _mean(rewards[:5]), _mean(rewards[-5:]), n_per_step * 5, n_per_step * 5
    )
    results["ztest_first5_vs_last5"] = {"z": z2, "p": p2, "significant": p2 < 0.05}

    # -- Spearman trend --
    rho, p_rho = spearman_trend(list(steps), rewards)
    results["spearman"] = {"rho": rho, "p": p_rho,
                            "significant": p_rho < 0.05}

    # -- Phase transition (Chow test) --
    pt = find_phase_transition(list(steps), rewards)
    results["phase_transition"] = pt

    # -- Split at phase transition for Cohen's d --
    bp = pt.get("breakpoint_index", n // 2)
    if early_cutoff is not None:
        bp = early_cutoff
    early = rewards[:bp]
    late = rewards[bp:]
    results["phase_split_index"] = bp
    results["cohen_d_early_vs_late"] = cohen_d(early, late)

    # -- Mann-Whitney: early vs late --
    u, p_mw, concl = mannwhitney(early, late)
    results["mannwhitney_early_vs_late"] = {"U": u, "p": p_mw, "conclusion": concl}

    return results


def compare_models(
    results_a: Dict,
    results_b: Dict,
    window: int = 10,
) -> Dict:
    """
    Run Mann-Whitney U and Cohen's d comparing the final `window` steps
    of two models.
    """
    # We only have the full results dict, so we need the raw rewards.
    # This function should be called after run_full_analysis with raw data.
    raise NotImplementedError("Call mannwhitney() directly with reward slices.")


# ---------------------------------------------------------------------------
# 8. Pretty printer
# ---------------------------------------------------------------------------

def print_report(results: Dict) -> None:
    m = results["model"]
    print(f"\n{'='*60}")
    print(f"  Statistical Report: {m}")
    print(f"{'='*60}")
    print(f"  Steps: {results['n_steps']}  |  Samples/step: {results['n_per_step']}")
    print(f"  Initial reward : {results['initial_reward']:.4f}")
    print(f"  Final reward   : {results['final_reward']:.4f}")
    print()

    lo, mu, hi = results["initial_ci"]
    print(f"  Bootstrap CI (first 5 steps)  : {mu:.4f}  [{lo:.4f}, {hi:.4f}]  95% CI")
    lo, mu, hi = results["final_ci"]
    print(f"  Bootstrap CI (last 10 steps)  : {mu:.4f}  [{lo:.4f}, {hi:.4f}]  95% CI")
    print()

    zt = results["ztest_step0_vs_final"]
    print(f"  Two-prop z-test (step 0 vs final)")
    print(f"    z = {zt['z']:.3f},  p = {zt['p']:.2e}  {'***' if zt['p'] < 0.001 else '*' if zt['p'] < 0.05 else 'ns'}")

    zt2 = results["ztest_first5_vs_last5"]
    print(f"  Two-prop z-test (first-5 vs last-5 means)")
    print(f"    z = {zt2['z']:.3f},  p = {zt2['p']:.2e}  {'***' if zt2['p'] < 0.001 else '*' if zt2['p'] < 0.05 else 'ns'}")
    print()

    sp = results["spearman"]
    print(f"  Spearman trend (reward vs step)")
    print(f"    rho = {sp['rho']:.4f},  p = {sp['p']:.2e}  {'***' if sp['p'] < 0.001 else 'ns'}")
    print()

    pt = results["phase_transition"]
    if pt.get("breakpoint_step") is not None:
        print(f"  Chow breakpoint (phase transition)")
        print(f"    Best breakpoint: step {pt['breakpoint_step']}")
        print(f"    F = {pt['f_statistic']:.2f},  p = {pt['p_value']:.2e}")
        print(f"    Pre-break mean:  {pt['pre_mean']:.4f}")
        print(f"    Post-break mean: {pt['post_mean']:.4f}")
    print()

    d = results["cohen_d_early_vs_late"]
    mag = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small"
    print(f"  Cohen's d (early vs late phase)  : {d:.3f}  ({mag} effect)")

    mw = results["mannwhitney_early_vs_late"]
    print(f"  Mann-Whitney U (early vs late)    : U={mw['U']:.0f},  p={mw['p']:.2e}  {mw['conclusion']}")
    print()
