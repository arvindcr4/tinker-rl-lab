"""Recompute P10's analytic examples; finite checks supplement, not replace, proofs.

Run: python3 tools/review_ledger/P10_zvf_theory_verify.py (requires SciPy).
No model calls, training data, or empirical performance claims are involved.
"""

import json
import math
import sys
from fractions import Fraction as F
from pathlib import Path

import scipy
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "platform_hybrid/experiments/results/review_2026-10-18/P10_zvf_theory/math_checks.json"


def h(p, k):
    return p**k + (1 - p) ** k


def main():
    checks = {}
    # Enumerate every binomial outcome, rather than using the claimed identities.
    points = [F(0), F(1, 100), F(1, 4), F(1, 2), F(3, 4), F(99, 100), F(1)]
    variance_rows = []
    for p in points:
        for g in range(2, 17):
            probabilities = [math.comb(g, c) * p**c * (1 - p) ** (g - c) for c in range(g + 1)]
            variance = sum(
                F(c * (g - c), g * (g - 1)) * probability
                for c, probability in enumerate(probabilities)
            )
            mixed = sum(probabilities[1:-1])
            proxy = mixed * p * (1 - p) / g
            target = (p * (1 - p)) ** 2
            checks[f"variance_p={p}_G={g}"] = variance == p * (1 - p)
            checks[f"proxy_p={p}_G={g}"] = (
                proxy == target if g in (2, 3) or p in (0, 1) else proxy < target
            )
            variance_rows.append({"p": str(p), "G": g, "variance": str(variance), "J": str(proxy)})
    checks["variance_per_rollout_no_tie"] = F(1, 4) / 2 > F(1, 4) / 3

    # Count exact rational tail probabilities; logs are only a cross-check.
    quantiles = []
    for theta, delta, k in [(F(443, 500), F(1, 10), 8), (F(1, 2), F(1, 8), 2)]:
        n = 1
        while theta**n > delta:
            n += 1
        continuous = k * math.log(1 / float(delta)) / -math.log(float(theta))
        checks[f"quantile_theta={theta}"] = (
            n == math.ceil(continuous / k) and theta ** (n - 1) > delta >= theta**n
        )
        quantiles.append(
            {
                "theta": str(theta),
                "delta": str(delta),
                "K": k,
                "continuous": continuous,
                "discrete": k * n,
                "previous_group_failure": str(theta ** (n - 1)),
                "quantile_failure": str(theta**n),
            }
        )
    checks["between_continuous_and_discrete_fails"] = quantiles[0]["continuous"] < 159 < quantiles[
        0
    ]["discrete"] and F(443, 500) ** (159 // 8) > F(1, 10)

    # Exact sum over all possible diagnostic counts (no Monte Carlo/bootstrap).
    # Coverage is checked on a finite theta grid; the manuscript gives the proof.
    alpha = 0.05
    coverage = []
    for m in (1, 8, 32, 64):
        lower = [0.0] + [float(beta.ppf(alpha, x, m - x + 1)) for x in range(1, m + 1)]
        upper = [float(beta.ppf(1 - alpha, x + 1, m - x)) for x in range(m)] + [1.0]
        checks[f"CP_endpoints_m={m}"] = (
            lower[0] == 0
            and upper[-1] == 1
            and math.isclose(lower[-1], alpha ** (1 / m), rel_tol=1e-12)
            and math.isclose(upper[0], 1 - alpha ** (1 / m), rel_tol=1e-12)
        )
        for theta in [0, 0.001] + [i / 100 for i in range(1, 100)] + [0.999, 1]:
            pmf = [math.comb(m, x) * theta**x * (1 - theta) ** (m - x) for x in range(m + 1)]
            lc = sum(prob for prob, bound in zip(pmf, lower, strict=True) if bound <= theta)
            uc = sum(prob for prob, bound in zip(pmf, upper, strict=True) if theta <= bound)
            checks[f"CP_coverage_m={m}_theta={theta}"] = min(lc, uc) >= 1 - alpha - 1e-12
            coverage.append({"m": m, "theta": theta, "lower_coverage": lc, "upper_coverage": uc})

    # A lower bound can fail to trigger even when the next budget is insufficient.
    # X=0 is possible at theta=0.75 and produces L=0, hence a one-group lower floor.
    checks["lower_floor_is_not_sufficiency"] = (1 - F(3, 4)) ** 8 > 0 and F(3, 4) > F(1, 10)
    theta2, theta8 = (h(F(0), k) + h(F(1, 2), k) for k in (2, 8))
    theta2, theta8 = theta2 / 2, theta8 / 2
    fixed_budget = {
        "B": 8,
        "theta2": str(theta2),
        "theta8": str(theta8),
        "failure_G2": str(theta2**4),
        "failure_G8": str(theta8),
    }
    checks["fixed_budget_escalation_worsens"] = theta2**4 == F(81, 256) < theta8 == F(129, 256)
    # Same theta_2, different theta_4: the scalar rate is generally insufficient.
    a2, a4 = h(F(1, 4), 2), h(F(1, 4), 4)
    b2, b4 = (F(3, 4) * h(F(1, 2), k) + F(1, 4) * h(F(0), k) for k in (2, 4))
    checks["cross_size_not_identified"] = a2 == b2 and a4 != b4
    checks["reciprocal_not_logarithmic_divergence"] = math.isclose(
        (1 - 0.999999) / -math.log(0.999999), 1, rel_tol=1e-6
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"P10 mathematical checks failed: {failed}")
    result = {
        "kind": "analytic examples and finite-grid verification; not empirical training evidence",
        "python": sys.version.split()[0],
        "scipy": scipy.__version__,
        "checks_passed": len(checks),
        "quantiles": quantiles,
        "fixed_budget": fixed_budget,
        "cross_size": {"theta2": str(a2), "theta4_point_mass": str(a4), "theta4_mixture": str(b4)},
        "minimum_CP_coverage_on_grid": min(
            min(r["lower_coverage"], r["upper_coverage"]) for r in coverage
        ),
        "variance_and_proxy_enumeration": variance_rows,
        "coverage_grid": coverage,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(checks)} mathematical checks passed; wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
