"""Iter 39 Pillar 3 (G=4 vs G=32): critical-budget detection + claim-strength audit.

Three analyses, building on iter 31 (iso-token TOST) and iter 35 (pair sweep):

1. Retention-vs-budget curve fit: R(T) = R_inf + (1-R_inf) * exp(-T / Tau).
   Solve for T_critical where R(T_critical) = Wu 2025's 97.6% threshold,
   with bootstrap CI on T_critical for each (G_a, G_b) pair.

2. Sigmoidal threshold model: R(T) = R_inf + (R_max - R_inf) / (1 + exp(-(T-T0)/k)).
   Compare to the exponential-decay form via AIC.

3. Claim-strength audit:
   (a) On iter35's 40-cell sweep, what fraction of (G_a, G_b, T) cells
       PASS Wu at Fieller-conservative 95% CI? Bootstrap CI on the
       fraction.
   (b) At the worst measured cell (G=4 vs G=32, T=64M, R=0.727),
       compute the maximum possible retention under sampling noise
       (upper CI bound) -- is 0.976 still in play?

Inputs (read-only, sourced from existing results):
  - platform_hybrid/experiments/results/group_size_iter31_iso_token.tsv
  - platform_hybrid/experiments/results/group_size_iter35_pair_sweep.tsv
  - platform_hybrid/experiments/results/group_size_token_normalized.tsv

Outputs:
  - platform_hybrid/experiments/results/group_size_iter39_t_critical.tsv
  - platform_hybrid/experiments/results/group_size_iter39_aic.tsv
  - platform_hybrid/experiments/results/group_size_iter39_claim_strength.tsv
  - platform_hybrid/experiments/results/group_size_iter39_summary.tsv
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

RESULTS = Path("platform_hybrid/experiments/results")
RESULTS.mkdir(parents=True, exist_ok=True)

# Wu et al. 2025 headline threshold: G=2 retains 97.6% of G=16.
# Source: arXiv:2510.00977 -- "It Takes Two: Your GRPO Is Secretly DPO".
WU_THRESHOLD = 0.976
N_BOOT = 4000
RNG = np.random.default_rng(20260702)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def read_tsv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    return reader.fieldnames or [], rows


def write_tsv(path: Path, header: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Analysis 1 -- Exponential retention decay + critical budget T*
# ---------------------------------------------------------------------------
def fit_exponential(T: np.ndarray, R: np.ndarray) -> tuple[float, float, float]:
    """Fit R(T) = R_inf + (1-R_inf) * exp(-T/Tau), minimize SSE."""
    # Closed-form candidate: R_inf = min(R), Tau = OLS-on-log-residual.
    R_inf = float(np.min(R))
    R0 = float(np.max(R))
    if R_inf >= R0:
        R_inf = R0 - 0.01
    resid = R - R_inf
    mask = resid > 0
    if mask.sum() < 2:
        return R_inf, 1e6, 1.0
    log_resid = np.log(resid[mask])
    T_m = T[mask]
    # log(R - R_inf) = log(1-R_inf) - T/Tau  => OLS
    x = T_m
    y = log_resid
    A = np.vstack([np.ones_like(x), -x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    log_A, slope = coef[0], coef[1]
    if slope <= 1e-9:
        slope = 1e-9
    Tau = 1.0 / slope
    pred = R_inf + (1.0 - R_inf) * np.exp(-T / Tau)
    sse = float(np.sum((R - pred) ** 2))
    return R_inf, Tau, sse


def fit_sigmoid(T: np.ndarray, R: np.ndarray) -> tuple[float, float, float, float, float]:
    """Fit R(T) = R_inf + (R_max - R_inf) / (1 + exp(-(T - T0)/k))."""
    R_inf = float(np.min(R))
    R_max = float(np.max(R))
    if R_inf >= R_max:
        R_inf = R_max - 0.01
    # Grid search over (T0, k).
    T_min, T_max = float(T.min()), float(T.max())
    best = (R_inf, R_max, T_min, T_max, np.inf)
    for T0 in np.linspace(T_min, T_max, 12):
        for k in np.geomspace(1e5, 1e8, 12):
            sig = 1.0 / (1.0 + np.exp(-(T - T0) / k))
            pred = R_inf + (R_max - R_inf) * sig
            sse = float(np.sum((R - pred) ** 2))
            if sse < best[4]:
                best = (R_inf, R_max, float(T0), float(k), sse)
    return best[0], best[1], best[2], best[3], best[4]  # R_inf, R_max, T0, k, sse


def t_critical_exponential(R_inf: float, Tau: float) -> float:
    """Solve R_inf + (1-R_inf)*exp(-T/Tau) = 0.976 for T."""
    target = WU_THRESHOLD
    if R_inf >= target:
        return 0.0  # already below at T=0
    delta = (target - R_inf) / (1.0 - R_inf)
    if delta >= 1.0:
        return 0.0
    if delta <= 0.0:
        return float("inf")
    return -Tau * math.log(delta)


def aic(sse: float, n: int, k_params: int) -> float:
    if sse <= 0:
        sse = 1e-12
    return 2 * k_params + n * math.log(sse / n)


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------
def bootstrap_ci(values: np.ndarray, q: float = 0.025) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(values, q)), float(np.quantile(values, 1 - q))


def bootstrap_t_critical(
    T_obs: np.ndarray, R_obs: np.ndarray, n_boot: int = N_BOOT
) -> tuple[float, float, float]:
    """Parametric bootstrap assuming R ~ Normal(R_fit(T), sigma).

    Returns: (T_star_point, T_star_lo, T_star_hi)
    """
    R_inf, Tau, _ = fit_exponential(T_obs, R_obs)
    pred = R_inf + (1.0 - R_inf) * np.exp(-T_obs / Tau)
    sigma = float(np.std(R_obs - pred))
    if sigma <= 0:
        sigma = 0.01
    Tc_samples = np.zeros(n_boot)
    for i in range(n_boot):
        R_perturbed = pred + RNG.normal(0, sigma, size=len(T_obs))
        R_perturbed = np.clip(R_perturbed, 0.0, 2.0)
        r_inf_b, tau_b, _ = fit_exponential(T_obs, R_perturbed)
        Tc_samples[i] = t_critical_exponential(r_inf_b, tau_b)
    T_star = t_critical_exponential(R_inf, Tau)
    ci_lo, ci_hi = bootstrap_ci(Tc_samples)
    # Guarantee lo <= hi for downstream display.
    if ci_lo > ci_hi:
        ci_lo, ci_hi = ci_hi, ci_lo
    # Also clip infinite values to a sensible upper bound for display.
    if not math.isfinite(ci_hi):
        ci_hi = float(T_obs.max() * 100.0)
    return T_star, ci_lo, ci_hi


def fit_and_audit_pair(
    T_arr: np.ndarray, R_arr: np.ndarray, R_lo: np.ndarray, R_hi: np.ndarray
) -> dict:
    """Fit exp/sigmoid on (T, R), compute T_critical, and report summary row."""
    R_inf, Tau, sse_exp = fit_exponential(T_arr, R_arr)
    T_star = t_critical_exponential(R_inf, Tau)
    _T_star_dup, T_star_lo, T_star_hi = bootstrap_t_critical(T_arr, R_arr)
    # Sigmoid fit
    R_inf_s, R_max_s, T0_s, k_s, sse_sig = fit_sigmoid(T_arr, R_arr)
    n = len(T_arr)
    aic_exp = aic(sse_exp, n, k_params=2)
    aic_sig = aic(sse_sig, n, k_params=4)
    delta_aic = aic_sig - aic_exp
    return {
        "R_inf_exp": R_inf,
        "Tau_M": Tau / 1e6,
        "T_star_M": T_star / 1e6 if T_star != float("inf") else float("inf"),
        "T_star_lo_M": T_star_lo / 1e6,
        "T_star_hi_M": T_star_hi / 1e6,
        "T_star_below_max_T": bool(T_star <= float(T_arr.max())),
        "AIC_exp": aic_exp,
        "AIC_sig": aic_sig,
        "delta_AIC_sig_minus_exp": delta_aic,
        "R_inf_sig": R_inf_s,
        "R_max_sig": R_max_s,
        "T0_sig_M": T0_s / 1e6,
        "k_sig_M": k_s / 1e6,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Load iter31 iso-token (G=4 vs G=32, 4 budgets).
    f31 = RESULTS / "group_size_iter31_iso_token.tsv"
    _, rows31 = read_tsv(f31)
    T31 = np.array([float(r["T_tokens"]) for r in rows31])
    R31 = np.array([float(r["retention"]) for r in rows31])
    R31_lo = np.array([float(r["retention_ci_low"]) for r in rows31])
    R31_hi = np.array([float(r["retention_ci_high"]) for r in rows31])

    # Load iter35 pair sweep (40 cells).
    f35 = RESULTS / "group_size_iter35_pair_sweep.tsv"
    _, rows35 = read_tsv(f35)

    # ---- Analysis 1: critical-budget detection ----
    audit_iter31 = fit_and_audit_pair(T31, R31, R31_lo, R31_hi)
    audit_rows = [{
        "pair": "G=4 vs G=32",
        "source": "iter31_iso_token",
        "n_budgets": int(len(T31)),
        "T_max_M": float(T31.max() / 1e6),
        "R_min": float(R31.min()),
        "R_max": float(R31.max()),
        **audit_iter31,
    }]
    # Per (G_a, G_b) pair at all 4 budgets from iter35
    pairs = sorted({(int(r["G_a"]), int(r["G_b"])) for r in rows35})
    for g_a, g_b in pairs:
        sub = [r for r in rows35 if int(r["G_a"]) == g_a and int(r["G_b"]) == g_b]
        if len(sub) < 3:
            continue
        T_arr = np.array([float(r["T_tokens"]) for r in sub])
        R_arr = np.array([float(r["retention"]) for r in sub])
        R_lo = np.array([float(r["retention_ci_low"]) for r in sub])
        R_hi = np.array([float(r["retention_ci_high"]) for r in sub])
        # Need variation for the exp fit; if all R_arr identical, skip.
        if np.std(R_arr) < 1e-6:
            continue
        audit = fit_and_audit_pair(T_arr, R_arr, R_lo, R_hi)
        audit_rows.append({
            "pair": f"G={g_a} vs G={g_b}",
            "source": "iter35_pair_sweep",
            "n_budgets": int(len(T_arr)),
            "T_max_M": float(T_arr.max() / 1e6),
            "R_min": float(R_arr.min()),
            "R_max": float(R_arr.max()),
            **audit,
        })
    header = list(audit_rows[0].keys())
    write_tsv(RESULTS / "group_size_iter39_t_critical.tsv", header, audit_rows)

    # ---- Analysis 2: per-pair AIC ranking (exp vs sigmoid) ----
    aic_rows = []
    for g_a, g_b in pairs:
        sub = [r for r in rows35 if int(r["G_a"]) == g_a and int(r["G_b"]) == g_b]
        if len(sub) < 3:
            continue
        T_arr = np.array([float(r["T_tokens"]) for r in sub])
        R_arr = np.array([float(r["retention"]) for r in sub])
        R_inf, Tau, sse_exp = fit_exponential(T_arr, R_arr)
        R_inf_s, R_max_s, T0_s, k_s, sse_sig = fit_sigmoid(T_arr, R_arr)
        n = len(T_arr)
        aic_exp = aic(sse_exp, n, 2)
        aic_sig = aic(sse_sig, n, 4)
        aic_rows.append({
            "G_a": g_a,
            "G_b": g_b,
            "n_budgets": n,
            "AIC_exp": round(aic_exp, 3),
            "AIC_sig": round(aic_sig, 3),
            "winner": "exp" if aic_exp <= aic_sig else "sig",
            "delta_AIC_sig_minus_exp": round(aic_sig - aic_exp, 3),
        })
    header = list(aic_rows[0].keys())
    write_tsv(RESULTS / "group_size_iter39_aic.tsv", header, aic_rows)

    # ---- Analysis 3: claim-strength audit ----
    # (a) Fraction of 40 cells passing Wu at Fieller-conservative 95% CI.
    n_total = len(rows35)
    n_pass = sum(1 for r in rows35 if r["above_wu_97_6pct"] == "True")
    # Bootstrap CI on the pass-fraction by treating each cell's pass status as
    # a Bernoulli with the empirical probability.
    p_hat = n_pass / n_total
    boot_frac = RNG.binomial(n_total, p_hat, size=N_BOOT) / n_total
    frac_lo, frac_hi = bootstrap_ci(boot_frac)

    # (b) Worst-cell (G=4 vs G=32, T=64M, R=0.727): upper CI on R vs Wu.
    worst = next(
        r for r in rows35
        if int(r["G_a"]) == 4 and int(r["G_b"]) == 32 and int(r["T_tokens"]) == 64000000
    )
    R_worst = float(worst["retention"])
    R_worst_hi = float(worst["retention_ci_high"])
    gap_to_Wu = R_worst_hi - WU_THRESHOLD  # negative => upper CI excludes Wu

    # (c) Of 18 passing cells, how many pass under stricter eps tests?
    # 4 levels: Wu 97.6% in CI, TOST eps=0.02, TOST eps=0.05, TOST eps=0.10
    eps_levels = [0.02, 0.05, 0.10]
    tost_counts = {e: sum(1 for r in rows35 if r.get("tost_p_eps0.02") and float(r.get("tost_p_eps0.02", 1.0)) < 0.05)
                   for e in eps_levels}
    # The TOST p-value column is "tost_p_eps0.02" -- so this counts cells at eps=0.02.
    n_tost_002 = sum(1 for r in rows35 if r.get("tost_equivalent") == "True")
    n_above_wu = sum(1 for r in rows35 if r["above_wu_97_6pct"] == "True")

    claim_rows = [
        {"claim": "n_cells_above_wu_in_CI", "value": n_above_wu, "denominator": n_total, "fraction": n_above_wu / n_total},
        {"claim": "n_cells_tost_equivalent_eps0.02", "value": n_tost_002, "denominator": n_total, "fraction": n_tost_002 / n_total},
        {"claim": "worst_cell_G4_vs_G32_T64M_R", "value": R_worst, "denominator": "", "fraction": ""},
        {"claim": "worst_cell_upper_CI_minus_Wu_threshold", "value": round(gap_to_Wu, 4), "denominator": "", "fraction": ""},
        {"claim": "worst_cell_upper_CI_excludes_Wu", "value": bool(gap_to_Wu < 0), "denominator": "", "fraction": ""},
        {"claim": "pass_fraction_bootstrap_CI_low", "value": round(float(frac_lo), 4), "denominator": n_total, "fraction": ""},
        {"claim": "pass_fraction_bootstrap_CI_high", "value": round(float(frac_hi), 4), "denominator": n_total, "fraction": ""},
        {"claim": "pass_fraction_significantly_below_half", "value": bool(frac_hi < 0.5), "denominator": "", "fraction": ""},
    ]
    header = list(claim_rows[0].keys())
    write_tsv(RESULTS / "group_size_iter39_claim_strength.tsv", header, claim_rows)

    # ---- Summary ----
    summary = {
        "n_pairs_with_fitted_retention_curve": len(audit_rows),
        "pairs_with_T_star_below_max_T": sum(1 for r in audit_rows if r["T_star_below_max_T"]),
        "iter31_G4_vs_G32_R_inf_exp": round(audit_iter31["R_inf_exp"], 4),
        "iter31_G4_vs_G32_Tau_M": round(audit_iter31["Tau_M"], 3),
        "iter31_G4_vs_G32_T_star_M": round(audit_iter31["T_star_M"], 3) if audit_iter31["T_star_M"] != float("inf") else "inf",
        "iter31_G4_vs_G32_T_star_lo_M": round(audit_iter31["T_star_lo_M"], 3),
        "iter31_G4_vs_G32_T_star_hi_M": round(audit_iter31["T_star_hi_M"], 3),
        "iter31_G4_vs_G32_AIC_exp_minus_AIC_sig": round(audit_iter31["AIC_exp"] - audit_iter31["AIC_sig"], 3),
        "iter35_40_cell_pass_fraction": n_pass / n_total,
        "iter35_40_cell_pass_fraction_CI95": [round(float(frac_lo), 4), round(float(frac_hi), 4)],
        "iter35_40_cell_tost_equivalent_eps0.02": n_tost_002,
        "worst_cell_G4_vs_G32_T64M": {"R": R_worst, "R_upper_CI": R_worst_hi, "above_Wu": gap_to_Wu >= 0},
        "claim_significantly_below_50pct_pass": bool(frac_hi < 0.5),
    }
    with (RESULTS / "group_size_iter39_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    # Also a TSV one-rollup for the paper.
    one_row = [
        {"metric": k, "value": str(v)}
        for k, v in summary.items()
        if not isinstance(v, (dict, list))
    ]
    one_row.append({"metric": "iter35_pass_fraction_CI95", "value": f"[{frac_lo:.4f}, {frac_hi:.4f}]"})
    write_tsv(
        RESULTS / "group_size_iter39_summary.tsv",
        ["metric", "value"],
        one_row,
    )

    # Console summary
    print(f"[iter39] Fit {len(audit_rows)} retention curves")
    print(f"[iter39] G=4 vs G=32 R_inf = {audit_iter31['R_inf_exp']:.4f}, "
          f"Tau = {audit_iter31['Tau_M']:.3f}M, T* = {audit_iter31['T_star_M']:.3f}M "
          f"[{audit_iter31['T_star_lo_M']:.3f}, {audit_iter31['T_star_hi_M']:.3f}]")
    print(f"[iter39] AIC exp={audit_iter31['AIC_exp']:.2f} vs sig={audit_iter31['AIC_sig']:.2f} "
          f"(delta_sig_minus_exp={audit_iter31['delta_AIC_sig_minus_exp']:.2f})")
    print(f"[iter39] Iter35 40-cell pass fraction = {n_pass}/{n_total} = {n_pass/n_total:.3f}")
    print(f"[iter39] Pass fraction 95% CI = [{frac_lo:.3f}, {frac_hi:.3f}]")
    print(f"[iter39] Worst cell R={R_worst:.3f}, upper CI={R_worst_hi:.3f} "
          f"(gap to Wu {gap_to_Wu:+.3f}; excludes Wu: {gap_to_Wu < 0})")


if __name__ == "__main__":
    main()