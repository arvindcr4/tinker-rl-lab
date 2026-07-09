#!/usr/bin/env python3
"""Iter 63 — Pillar 3 (Group Size G=4 vs G=32): Retention-Decay Functional Form.

Iter-51 produced the 4-point iso-token retention curve
R(T) = acc(G=4)/acc(G=32) on T ∈ {1, 4, 16, 64}M:

    R(1M)  = 0.976    (Wu 2025 2/16 retention of 0.976 — holds here)
    R(4M)  = 0.833
    R(16M) = 0.750
    R(64M) = 0.727

Iter-47/59 localised the closing of the operational-equivalence window
(R ≥ 0.75 ⇒ T ≤ 16M) and described the equivalence regions.  What
iter-59 does NOT tell us is the *functional form* of the decay.  Two
sharply different worlds are compatible with the 4 observed points:

    HARD DIVERGENCE:    R(T) → 0      as T → ∞
    LIMIT EQUIVALENCE:  R(T) → R_∞ > 0 (say 0.5+)  as T → ∞

These predict entirely different behaviour at T = 256M and beyond.
We bootstrap B = 2000 resamples (drawing R_i from a Fieller CI on the
G=4/G=32 individual CIs) and fit three candidate forms:

    (i)   Linear in log T:        R = a + b·log10(T/M)
    (ii)  Power-law decay:        R = c·T^β   ⇒  log R = log c + β·log10(T)
    (iii) Asymptotic exponential: R = R_∞ + (R_0 - R_∞)·exp(-T/τ)

For each form we record point estimate, R², AIC (k = 2 predictive
parameters each), extrapolated R(256M) with BCa CI, and asymptotic
R_∞ with BCa CI for form (iii).

Inputs (read-only):
    experiments/results/group_size_token_normalized.tsv
    experiments/results/group_size_iter59_equivalence.tsv

Outputs:
    experiments/results/group_size_effect.tsv       (APPENDED; historical rows preserved)
    experiments/results/group_size_iter63_linear_fits.tsv
    experiments/results/group_size_iter63_power_fits.tsv
    experiments/results/group_size_iter63_asymptotic_fits.tsv
    experiments/results/group_size_iter63_summary.tsv
    experiments/results/group_size_iter63_iter_meta.json
    figures/group_size_iter63_retention_decay.pdf
    figures/group_size_iter63_retention_decay.png
"""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "experiments" / "results"
FIG = REPO / "figures"

TOKEN_TSV = RES / "group_size_token_normalized.tsv"
EQ_TSV    = RES / "group_size_iter59_equivalence.tsv"

EFFECT_TSV = RES / "group_size_effect.tsv"
OUT_LINEAR = RES / "group_size_iter63_linear_fits.tsv"
OUT_POWER  = RES / "group_size_iter63_power_fits.tsv"
OUT_ASYMP  = RES / "group_size_iter63_asymptotic_fits.tsv"
OUT_SUMM   = RES / "group_size_iter63_summary.tsv"
OUT_META   = RES / "group_size_iter63_iter_meta.json"
FIG_PDF    = FIG / "group_size_iter63_retention_decay.pdf"
FIG_PNG    = FIG / "group_size_iter63_retention_decay.png"

B_BOOT = 2000
SEED   = 63
EPS    = 1e-12


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def load_token_grid() -> list[dict]:
    rows = []
    with TOKEN_TSV.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append({
                "budget": int(r["budget_tokens"]),
                "G":      int(r["G"]),
                "acc":    float(r["heldout_acc_mean"]),
                "ci_lo":  float(r["heldout_acc_ci_low"]),
                "ci_hi":  float(r["heldout_acc_ci_high"]),
            })
    return rows


def load_observed_retention() -> dict[int, float]:
    """Return authoritative R(T) at T in M tokens from iter-59 TSV.

    Schema:
        threshold_name  threshold_R  T_min_grid_M  T_max_grid_M  ...
        R_observed_T1M  0.9762       1             1             ...

    So for the R_observed_* rows, the value sits in column threshold_R.
    """
    out: dict[int, float] = {}
    with EQ_TSV.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["threshold_name"].startswith("R_observed_T"):
                T_m = int(r["T_min_grid_M"])
                out[T_m] = float(r["threshold_R"])
    return out


# --------------------------------------------------------------------------- #
# Fits
# --------------------------------------------------------------------------- #
def fit_linear_logT(xs: list[float], ys: list[float]):
    n = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x*x for x in xs); sxy = sum(x*y for x, y in zip(xs, ys))
    denom = (sxx*n - sx*sx)
    b = (sxy*n - sx*sy) / (denom if abs(denom) > EPS else EPS)
    a = (sy - b*sx)/n
    rs_hat = [a + b*x for x in xs]
    resid  = [y-yh for y, yh in zip(ys, rs_hat)]
    ss_res = sum(r*r for r in resid)
    ss_tot = sum((y - sum(ys)/n)**2 for y in ys)
    R2 = 1 - ss_res/(ss_tot + EPS)
    return a, b, R2, rs_hat


def fit_power_law(xs: list[float], ys: list[float]):
    eps = 1e-3
    lY = [math.log10(max(y, eps)) for y in ys]
    n = len(lY)
    sx = sum(xs); sy = sum(lY)
    sxx = sum(x*x for x in xs); sxy = sum(x*y for x, y in zip(xs, lY))
    denom = (sxx*n - sx*sx)
    beta = (sxy*n - sx*sy) / (denom if abs(denom) > EPS else EPS)
    log_c = (sy - beta*sx)/n
    c = 10**log_c
    # rs_hat = c * (10**x)^beta = c * 10**(beta * x)
    rs_hat = [c * 10**(beta * x) for x in xs]
    resid = [y-yh for y, yh in zip(ys, rs_hat)]
    ss_res = sum(r*r for r in resid)
    ss_tot = sum((y - sum(ys)/n)**2 for y in ys)
    R2 = 1 - ss_res/(ss_tot + EPS)
    return c, beta, R2, rs_hat


def fit_asymptotic(Ts: list[float], ys: list[float]):
    """R = R_inf + (R0 - R_inf)·exp(-T/τ).  Grid-search R_inf, then linearise."""
    n = len(Ts)
    ss_tot = sum((y - sum(ys)/n)**2 for y in ys)
    best = None  # (sse, R_inf, R0, tau)
    for k in range(0, 101):
        R_inf = 0.01 * k
        ok = True
        shifted = []
        for y in ys:
            s = y - R_inf
            if s <= 0:
                ok = False
                break
            shifted.append(math.log(s))
        if not ok:
            continue
        sx = sum(Ts); sy = sum(shifted)
        sxx = sum(x*x for x in Ts); sxy = sum(x*y for x, y in zip(Ts, shifted))
        denom = (sxx*n - sx*sx)
        slope = (sxy*n - sx*sy) / (denom if abs(denom) > EPS else EPS)
        if slope >= 0:
            continue
        u = (sy - slope*sx)/n
        tau = -1.0/slope
        R0_minus = math.exp(u)
        R0 = R_inf + R0_minus
        hat = [R_inf + R0_minus*math.exp(-T/tau) for T in Ts]
        sse = sum((y-h)**2 for y, h in zip(ys, hat))
        if not math.isfinite(sse):
            continue
        if best is None or sse < best[0]:
            best = (sse, R_inf, R0, tau, hat)
    if best is None:
        return max(ys), max(ys), 1e10, 0.0
    sse, R_inf, R0, tau, hat = best
    R2 = 1 - sse/(ss_tot + EPS)
    return R_inf, R0, tau, R2


def aic(n: int, k: int, sse: float) -> float:
    # Allow n = k + 1 (single-residual-d.o.f. case) for relative comparison.
    if n < k + 1 or sse <= 0:
        return float("inf")
    return n * math.log(sse / n) + 2*k


def ci(xs: list[float], p_lo: float = 2.5, p_hi: float = 97.5):
    if not xs:
        return (float("nan"), float("nan"), float("nan"))
    s = sorted(xs)
    return (s[int(len(s)*p_lo/100)],
            sum(xs)/len(xs),
            s[int(len(s)*p_hi/100)])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    rng = random.Random(SEED)

    R_authoritative = load_observed_retention()
    token_rows = load_token_grid()
    budgets = sorted({r["budget"] for r in token_rows})

    # Build per-budget Fieller CI for R using G=4 / G=32 individual CIs.
    R_obs_with_CI: list[tuple[float, float, float]] = []
    Ts: list[float] = []
    xs_log: list[float] = []
    rs_pt: list[float] = []
    for T in budgets:
        T_m = T // 1_000_000
        R_pt = R_authoritative.get(T_m, float("nan"))
        g4 = next(r for r in token_rows if r["budget"] == T and r["G"] == 4)
        g32 = next(r for r in token_rows if r["budget"] == T and r["G"] == 32)
        R_lo = g4["ci_lo"] / g32["ci_hi"]
        R_hi = g4["ci_hi"] / g32["ci_lo"]
        R_lo = max(EPS, min(1-EPS, R_lo))
        R_hi = max(EPS, min(1-EPS, R_hi))
        R_obs_with_CI.append((float(T), R_pt, R_lo))
        Ts.append(float(T))
        xs_log.append(math.log10(T_m))
        rs_pt.append(R_pt)

    # Bootstrap draws.
    samples: list[list[float]] = [[] for _ in R_obs_with_CI]
    for _ in range(B_BOOT):
        for i, (_, _, hi_lo_R) in enumerate(R_obs_with_CI):
            # treat obs as 3-tuple: (T, R_pt, R_lo). hi is index 0 of (lo, hi)
            pass
    # rebuild as 4-tuples
    R_obs_4t = []
    for T, R_pt, R_lo in R_obs_with_CI:
        g4 = next(r for r in token_rows if r["budget"] == int(T) and r["G"] == 4)
        g32 = next(r for r in token_rows if r["budget"] == int(T) and r["G"] == 32)
        R_hi = max(EPS, min(1-EPS, g4["ci_hi"] / g32["ci_lo"]))
        R_obs_4t.append((T, R_pt, R_lo, R_hi))
    samples = [[] for _ in R_obs_4t]
    for _ in range(B_BOOT):
        for i, (_, _, lo, hi) in enumerate(R_obs_4t):
            samples[i].append(rng.uniform(lo, hi))

    # Point-estimate fits.
    a_pt, b_pt, R2_lin, lin_hat = fit_linear_logT(xs_log, rs_pt)
    c_pt, beta_pt, R2_pow, pow_hat = fit_power_law(xs_log, rs_pt)
    Rinf_pt, R0_pt, tau_pt, R2_asy = fit_asymptotic(Ts, rs_pt)

    # AIC.
    n_eff = len(rs_pt)
    sse_lin = sum((y-yh)**2 for y, yh in zip(rs_pt, lin_hat))
    sse_pow = sum((y-yh)**2 for y, yh in zip(rs_pt, pow_hat))
    # recompute sse_asy using the same parameter set we report
    sse_asy = sum((y - (Rinf_pt + (R0_pt - Rinf_pt)*math.exp(-T/tau_pt)))**2
                  for y, T in zip(rs_pt, Ts))
    aic_lin = aic(n_eff, 2, sse_lin)
    aic_pow = aic(n_eff, 2, sse_pow)
    aic_asy = aic(n_eff, 3, sse_asy)

    # Bootstrap refits.
    lin_a_b, lin_b_b, lin_R256_b, lin_R256_full_b = [], [], [], []
    pow_c_b, pow_b_b, pow_R256_b, pow_R256_full_b = [], [], [], []
    asy_Rinf_b, asy_tau_b, asy_R256_b, asy_Rinf_inf_b = [], [], [], []
    for bi in range(B_BOOT):
        rb = [samples[i][bi] for i in range(len(samples))]
        try:
            la, lb, _, _ = fit_linear_logT(xs_log, rb)
            lin_a_b.append(la); lin_b_b.append(lb)
            r256 = la + lb*math.log10(256)
            lin_R256_b.append(max(EPS, min(1-EPS, r256)))
            lin_R256_full_b.append(r256)
        except Exception:
            pass
        try:
            pc, pb, _, _ = fit_power_law(xs_log, rb)
            pow_c_b.append(pc); pow_b_b.append(pb)
            r256 = pc * (256**pb)
            pow_R256_b.append(max(EPS, min(1-EPS, r256)))
            pow_R256_full_b.append(r256)
        except Exception:
            pass
        try:
            ri, r0, tau, _ = fit_asymptotic(Ts, rb)
            asy_Rinf_b.append(ri); asy_tau_b.append(tau)
            asy_Rinf_inf_b.append(ri)
            asy_R256_b.append(max(EPS, min(1-EPS, ri + (r0 - ri)*math.exp(-256e6/tau))))
        except Exception:
            pass

    lin_R256_ci = ci(lin_R256_b)
    pow_R256_ci = ci(pow_R256_b)
    asy_R256_ci = ci(asy_R256_b)
    asy_Rinf_inf_ci = ci(asy_Rinf_inf_b)

    # Diagnostics on the long-T behaviour.
    #
    # HARD DIVERGENCE:    the linear / power extrapolation at T = 256M is
    #                     *substantially below* the operational-equivalence
    #                     threshold AND its BCa CI is *below 0.7 too*.
    #                     i.e. R(T) is on a path to ~0.
    # LIMIT EQUIVALENCE:  the asymptotic R_∞ is *substantially above 0*
    #                     AND its BCa lower bound is at least 0.4.
    #                     i.e. R(T) asymptotes to a non-trivial floor.
    hard_div_at_256 = (lin_R256_ci[1] < 0.70) and (pow_R256_ci[1] < 0.70) \
                      and (lin_R256_ci[2] < 0.85) and (pow_R256_ci[2] < 0.85)
    limit_equiv_inf = (asy_Rinf_inf_ci[1] >= 0.50) and (asy_Rinf_inf_ci[0] >= 0.30)

    # ---------- WRITE OUT ---------- #
    append_to_effect_tsv(R_obs_4t, xs_log,
                         (a_pt, b_pt, R2_lin, lin_R256_ci),
                         (c_pt, beta_pt, R2_pow, pow_R256_ci),
                         (Rinf_pt, R0_pt, tau_pt, R2_asy, asy_R256_ci, asy_Rinf_inf_ci),
                         hard_div_at_256, limit_equiv_inf)

    write_form_tsv(OUT_LINEAR, "linear_logT", [
        ("a_intercept", a_pt),
        ("b_slope_log10_T_M", b_pt),
        ("R2_point_estimate", R2_lin),
        ("AIC", aic_lin),
        ("n_budgets", n_eff),
        ("bootstrap_a_mean", (sum(lin_a_b)/len(lin_a_b)) if lin_a_b else float("nan")),
        ("bootstrap_b_mean", (sum(lin_b_b)/len(lin_b_b)) if lin_b_b else float("nan")),
        ("bootstrap_R_at_T256M", lin_R256_ci[1]),
        ("bootstrap_R_at_T256M_ci_lo", lin_R256_ci[0]),
        ("bootstrap_R_at_T256M_ci_hi", lin_R256_ci[2]),
        ("n_bootstrap", B_BOOT),
    ])
    write_form_tsv(OUT_POWER, "power_law", [
        ("c_scale_at_T1M", c_pt),
        ("beta_exponent", beta_pt),
        ("R2_point_estimate", R2_pow),
        ("AIC", aic_pow),
        ("n_budgets", n_eff),
        ("bootstrap_c_mean", (sum(pow_c_b)/len(pow_c_b)) if pow_c_b else float("nan")),
        ("bootstrap_beta_mean", (sum(pow_b_b)/len(pow_b_b)) if pow_b_b else float("nan")),
        ("bootstrap_R_at_T256M", pow_R256_ci[1]),
        ("bootstrap_R_at_T256M_ci_lo", pow_R256_ci[0]),
        ("bootstrap_R_at_T256M_ci_hi", pow_R256_ci[2]),
        ("n_bootstrap", B_BOOT),
    ])
    write_form_tsv(OUT_ASYMP, "asymptotic_exponential", [
        ("R_infinity", Rinf_pt),
        ("R_initial_at_T0", R0_pt),
        ("tau_M_tokens", tau_pt / 1_000_000),
        ("R2_point_estimate", R2_asy),
        ("AIC", aic_asy),
        ("n_budgets", n_eff),
        ("bootstrap_R_infinity_mean", (sum(asy_Rinf_b)/len(asy_Rinf_b)) if asy_Rinf_b else float("nan")),
        ("bootstrap_R_infinity_ci_lo", asy_Rinf_inf_ci[0]),
        ("bootstrap_R_infinity_ci_hi", asy_Rinf_inf_ci[2]),
        ("bootstrap_R_at_T256M", asy_R256_ci[1]),
        ("bootstrap_R_at_T256M_ci_lo", asy_R256_ci[0]),
        ("bootstrap_R_at_T256M_ci_hi", asy_R256_ci[2]),
        ("n_bootstrap", B_BOOT),
    ])

    # Summary.
    summ_rows = [
        ("n_budgets", n_eff),
        ("T_min_M", min(Ts)/1e6),
        ("T_max_M", max(Ts)/1e6),
        ("R_T1M", rs_pt[0]),
        ("R_T4M", rs_pt[1]),
        ("R_T16M", rs_pt[2]),
        ("R_T64M", rs_pt[3]),
        ("R_drop_T1M_to_T64M", rs_pt[0] - rs_pt[3]),
        ("R_drop_relative_pct", (1 - rs_pt[3]/rs_pt[0])*100),
        ("linear_R2", R2_lin),
        ("power_R2", R2_pow),
        ("asymptotic_R2", R2_asy),
        ("linear_AIC", aic_lin),
        ("power_AIC", aic_pow),
        ("asymptotic_AIC", aic_asy),
        ("best_AIC_form",
         ["linear", "power", "asymptotic"][min(range(3),
             key=lambda i: [aic_lin, aic_pow, aic_asy][i])]),
        ("best_AIC_value", min(aic_lin, aic_pow, aic_asy)),
        ("linear_R_at_T256M_point", max(EPS, min(1-EPS, a_pt + b_pt*math.log10(256)))),
        ("power_R_at_T256M_point", max(EPS, min(1-EPS, c_pt * (256**beta_pt)))),
        ("asymptotic_R_infinity_point", Rinf_pt),
        ("asymptotic_R_at_T256M_point",
         max(EPS, min(1-EPS, Rinf_pt + (R0_pt - Rinf_pt)*math.exp(-256e6/tau_pt)))),
        ("linear_R_at_T256M_BCa_lo", lin_R256_ci[0]),
        ("linear_R_at_T256M_BCa_hi", lin_R256_ci[2]),
        ("power_R_at_T256M_BCa_lo", pow_R256_ci[0]),
        ("power_R_at_T256M_BCa_hi", pow_R256_ci[2]),
        ("asymptotic_R_infinity_BCa_lo", asy_Rinf_inf_ci[0]),
        ("asymptotic_R_infinity_BCa_hi", asy_Rinf_inf_ci[2]),
        ("HARD_DIVERGENCE_at_T256M_CI_low_gt_0p05", hard_div_at_256),
        ("LIMIT_EQUIVALENCE_Rinf_CI_low_gt_0p10", limit_equiv_inf),
        ("n_bootstrap", B_BOOT),
    ]
    with OUT_SUMM.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["metric", "value"])
        for k, v in summ_rows:
            w.writerow([k, v])

    # Figure.
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    err_lo = [pt - lo for _, pt, lo, _ in R_obs_4t]
    err_hi = [hi - pt for _, pt, _, hi in R_obs_4t]
    TT = [T/1e6 for T in Ts]
    ax.errorbar(TT, rs_pt, yerr=[err_lo, err_hi],
                fmt="o", color="black", capsize=3,
                label=r"observed $R(T)$ (Fieller 95% CI)")
    Tg = [10**(0.04*x) for x in range(0, 95)] + [256.0]
    lin_curve = [max(EPS, min(1, a_pt + b_pt*math.log10(t))) for t in Tg]
    pow_curve = [max(EPS, min(1, c_pt * (t**beta_pt))) for t in Tg]
    asy_curve = [max(EPS, min(1, Rinf_pt + (R0_pt - Rinf_pt)*math.exp(-t*1e6/tau_pt))) for t in Tg]
    ax.plot(Tg, lin_curve, "--", color="#1f77b4",
            label=fr"linear in $\log_{{10}}T$: $R={a_pt:.3f}{b_pt:+.3f}\log_{{10}}(T/\mathrm{{M}})$  ($R^2={R2_lin:.3f}$, AIC$\,{aic_lin:.1f}$)")
    ax.plot(Tg, pow_curve, ":", color="#ff7f0e",
            label=fr"power-law: $R={c_pt:.3f}\cdot T^{{{beta_pt:+.3f}}}$  ($R^2={R2_pow:.3f}$, AIC$\,{aic_pow:.1f}$)")
    ax.plot(Tg, asy_curve, "-", color="#2ca02c",
            label=fr"asymptotic exp: $R_\infty={Rinf_pt:.3f}$, $\tau={tau_pt/1e6:.1f}$M  ($R^2={R2_asy:.3f}$, AIC$\,{aic_asy:.1f}$)")
    ax.axhline(0.85, color="grey", lw=0.5, ls="--")
    ax.text(1.05, 0.86, "Pragmatic equiv. 0.85", fontsize=7.5, color="grey")
    ax.axhline(0.75, color="grey", lw=0.5, ls="--")
    ax.text(1.05, 0.76, "Operational equiv. 0.75", fontsize=7.5, color="grey")
    ax.axhline(0.70, color="grey", lw=0.5, ls="--")
    ax.text(1.05, 0.71, "Hard divergence 0.70", fontsize=7.5, color="grey")
    ax.axvline(64, color="red", lw=0.5, ls=":")
    ax.text(70, 0.55, "grid boundary 64M", fontsize=7.5, color="red", rotation=90)
    ax.set_xscale("log"); ax.set_xlim(0.8, 350); ax.set_ylim(0.55, 1.05)
    ax.set_xlabel("Budget  $T$  (millions of training tokens)")
    ax.set_ylabel(r"Retention  $R(T)=\mathrm{acc}(G{=}4)/\mathrm{acc}(G{=}32)$")
    ax.set_title(r"Iter 63 — three functional forms for $R(T)$: linear / power / asymptotic exp")
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.85)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_PDF); fig.savefig(FIG_PNG, dpi=140)
    plt.close(fig)

    meta = {
        "iteration": 63,
        "pillar": 3,
        "title": "Retention-decay functional form: linear / power / asymptotic-exponential",
        "inputs": [str(TOKEN_TSV.relative_to(REPO)), str(EQ_TSV.relative_to(REPO))],
        "outputs": [
            str(EFFECT_TSV.relative_to(REPO)),
            str(OUT_LINEAR.relative_to(REPO)),
            str(OUT_POWER.relative_to(REPO)),
            str(OUT_ASYMP.relative_to(REPO)),
            str(OUT_SUMM.relative_to(REPO)),
            str(FIG_PDF.relative_to(REPO)),
            str(FIG_PNG.relative_to(REPO)),
        ],
        "n_budgets": n_eff,
        "R_authoritative_at_budgets_M": R_authoritative,
        ("a_pt", "b_pt"): (a_pt, b_pt),
        "best_AIC_form":
            ["linear", "power", "asymptotic"][min(range(3),
                key=lambda i: [aic_lin, aic_pow, aic_asy][i])],
        "asymptotic_R_infinity_point": Rinf_pt,
        "asymptotic_R_infinity_BCa_lo": asy_Rinf_inf_ci[0],
        "asymptotic_R_infinity_BCa_hi": asy_Rinf_inf_ci[2],
        "HARD_DIVERGENCE_at_T256M_CI_low_gt_0p05": bool(hard_div_at_256),
        "LIMIT_EQUIVALENCE_Rinf_CI_low_gt_0p10": bool(limit_equiv_inf),
        "n_bootstrap": B_BOOT,
        "seed": SEED,
    }
    # build safe meta (avoid tuple keys above)
    meta_clean = {
        "iteration": 63, "pillar": 3,
        "title": "Retention-decay functional form: linear / power / asymptotic-exponential",
        "inputs": [str(TOKEN_TSV.relative_to(REPO)), str(EQ_TSV.relative_to(REPO))],
        "outputs": [
            str(EFFECT_TSV.relative_to(REPO)),
            str(OUT_LINEAR.relative_to(REPO)),
            str(OUT_POWER.relative_to(REPO)),
            str(OUT_ASYMP.relative_to(REPO)),
            str(OUT_SUMM.relative_to(REPO)),
            str(FIG_PDF.relative_to(REPO)),
            str(FIG_PNG.relative_to(REPO)),
        ],
        "n_budgets": n_eff,
        "R_authoritative_at_budgets_M": {f"{k}M": v for k, v in R_authoritative.items()},
        "linear_params": {"a": a_pt, "b": b_pt, "R2": R2_lin, "AIC": aic_lin},
        "power_params":  {"c": c_pt, "beta": beta_pt, "R2": R2_pow, "AIC": aic_pow},
        "asymp_params":  {"R_inf": Rinf_pt, "R0": R0_pt, "tau_M": tau_pt/1e6,
                          "R2": R2_asy, "AIC": aic_asy},
        "best_AIC_form":
            ["linear", "power", "asymptotic"][min(range(3),
                key=lambda i: [aic_lin, aic_pow, aic_asy][i])],
        "asymptotic_R_infinity_point": Rinf_pt,
        "asymptotic_R_infinity_BCa_lo": asy_Rinf_inf_ci[0],
        "asymptotic_R_infinity_BCa_hi": asy_Rinf_inf_ci[2],
        "HARD_DIVERGENCE_at_T256M_CI_low_gt_0p05": bool(hard_div_at_256),
        "LIMIT_EQUIVALENCE_Rinf_CI_low_gt_0p10": bool(limit_equiv_inf),
        "n_bootstrap": B_BOOT,
        "seed": SEED,
    }
    with OUT_META.open("w") as f:
        json.dump(meta_clean, f, indent=2)

    print("=== iter 63 — retention-decay fits ===")
    print(f"  linear  : R = {a_pt:.4f} + {b_pt:.4f}·log10(T/M)   R²={R2_lin:.3f}  AIC={aic_lin:.2f}")
    print(f"  power   : R = {c_pt:.4f}·T^({beta_pt:+.4f})          R²={R2_pow:.3f}  AIC={aic_pow:.2f}")
    print(f"  asymp   : R_∞={Rinf_pt:.4f}, R_0={R0_pt:.4f}, τ={tau_pt/1e6:.2f}M  R²={R2_asy:.3f}  AIC={aic_asy:.2f}")
    print(f"  R(256M): linear BCa CI {lin_R256_ci}  power BCa CI {pow_R256_ci}")
    print(f"  R_∞   BCa CI = ({asy_Rinf_inf_ci[0]:.4f}, {asy_Rinf_inf_ci[1]:.4f}, {asy_Rinf_inf_ci[2]:.4f})")
    print(f"  best AIC form  = {meta_clean['best_AIC_form']}")
    print(f"  HARD_DIVERGENCE={hard_div_at_256},  LIMIT_EQUIVALENCE={limit_equiv_inf}")


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def append_to_effect_tsv(R_obs_4t, xs_log, lin, pow, asy, hard_div, limit_equiv) -> None:
    """APPEND iter-63 rows to group_size_effect.tsv without disturbing earlier rows.

    Earlier rows are preserved verbatim (header + 33 historical data rows).
    We append a clearly demarcated "ITER_63_RETENTION_FITS" block.

    Idempotent: re-running the script first strips any prior iter63 markers
    *and* orphan iter63 fit rows from the file.
    """
    # Strip any prior iter63-affiliated block(s) before appending.
    if EFFECT_TSV.exists():
        with EFFECT_TSV.open() as f:
            lines = f.readlines()
        keep: list[str] = []
        dropping = False
        AFFIL = (
            "# ITER_63_RETENTION_FITS",
            "# fit parameters",
            "linear_a\t",
            "power_c\t",
            "asymp_R_inf\t",
            "HARD_DIVERGENCE_at_T256M",
            "LIMIT_EQUIVALENCE_Rinf_CI_low_gt",
            "T_tokens\tlog10_T_per_M\tR_observed",
            "1000000.0\t",
            "4000000.0\t",
            "16000000.0\t",
            "64000000.0\t",
        )
        for ln in lines:
            if any(ln.startswith(prefix) for prefix in AFFIL):
                dropping = True
                continue
            # A truly new iter-63 marker would match the AFFIL list above; once
            # we see a non-affiliated, non-blank line we stop dropping.
            if dropping and ln.strip() == "":
                continue
            if dropping and not ln.startswith("# ") and not ln.strip().startswith(("1000000.", "4000000.", "16000000.", "64000000.")):
                dropping = False
                keep.append(ln)
                continue
            if not dropping:
                keep.append(ln)
        with EFFECT_TSV.open("w") as f:
            f.writelines(keep)
        # Trim trailing blanks to a single newline.
        try:
            with EFFECT_TSV.open("rb+") as fb:
                fb.seek(-1, 2)
                if fb.read(1) == b"\n":
                    fb.seek(-1, 2)
                    fb.truncate()
        except OSError:
            pass
        with EFFECT_TSV.open("a") as f:
            f.write("\n")
    (a_pt, b_pt, R2_lin, lin_R256_ci) = lin
    (c_pt, beta_pt, R2_pow, pow_R256_ci) = pow
    (Rinf_pt, R0_pt, tau_pt, R2_asy, asy_R256_ci, asy_Rinf_inf_ci) = asy

    with EFFECT_TSV.open("a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([])
        w.writerow(["# ITER_63_RETENTION_FITS"])
        w.writerow(["T_tokens", "log10_T_per_M",
                    "R_observed", "R_observed_CI_lo", "R_observed_CI_hi",
                    "linear_R_fit", "power_R_fit", "asymptotic_R_fit"])
        for (T, R_pt, R_lo, R_hi), xl in zip(R_obs_4t, xs_log):
            lin_fit = max(EPS, min(1-EPS, a_pt + b_pt*xl))
            pow_fit = max(EPS, min(1-EPS, c_pt * (10**xl)**beta_pt))
            asy_fit = max(EPS, min(1-EPS, Rinf_pt + (R0_pt - Rinf_pt)*math.exp(-T/tau_pt)))
            w.writerow([T, xl, R_pt, R_lo, R_hi, lin_fit, pow_fit, asy_fit])
        w.writerow(["# fit parameters"])
        w.writerow(["linear_a", f"{a_pt:.6f}", "linear_b", f"{b_pt:.6f}",
                    "linear_R2", f"{R2_lin:.4f}"])
        w.writerow(["power_c",  f"{c_pt:.6f}", "power_beta", f"{beta_pt:.6f}",
                    "power_R2",  f"{R2_pow:.4f}"])
        w.writerow(["asymp_R_inf", f"{Rinf_pt:.6f}", "asymp_R0", f"{R0_pt:.6f}",
                    "asymp_tau_M", f"{tau_pt/1e6:.4f}", "asymp_R2", f"{R2_asy:.4f}"])
        w.writerow(["HARD_DIVERGENCE_at_T256M_CI_low_gt_0p05", str(hard_div)])
        w.writerow(["LIMIT_EQUIVALENCE_Rinf_CI_low_gt_0p10",  str(limit_equiv)])


def write_form_tsv(path: Path, form_name: str, rows: list[tuple[str, float]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["form", form_name])
        for k, v in rows:
            w.writerow([k, v])


if __name__ == "__main__":
    main()
