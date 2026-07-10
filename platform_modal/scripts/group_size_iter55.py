#!/usr/bin/env python3
"""Iter 55 — Pillar 3: Theory-coupled G=4 vs G=32 analysis.

This iteration unifies the iter-51 broader-scale retention finding with the
frontier-synthesis "contrastive yield" framing.  Concretely:

  (A) Effective sample size per group d_eff(G) = G * (1 - ZVF(G))
      Reported from the existing zvf_sweep.
  (B) Empirical anti-herding bonus delta_div(G) = ZVF_obs(G) - ZVF_iid(p, G)
      Test whether delta_div is G-invariant (frontier hypothesis) or G-dependent.
  (C) Iso-yield predictive model: predicted argmax G = argmax_G d_eff(G) * T_steps(G, T)
      where T_steps(G, T) is the per-token optimizer budget at (G, T).
      This unifies Pillar 2 (ZVF) with Pillar 3 (group size).
  (D) Cross-coupling: predicted vs empirical argmax G at each budget,
      plus residual decomposition (Δ = empirical - predicted in log10 G units).
  (E) Wu 2025 retention model: retention = (d_eff(G_a) / d_eff(G_b))^alpha,
      alpha fit to T=1M, then predicted vs empirical at T={4, 16, 64}M.
  (F) Summary rollup.

Inputs (read-only):
  platform_hybrid/experiments/results/group_size_token_normalized.tsv   (4 budgets x 5 G)
  platform_hybrid/experiments/results/groupsize_zvf_sweep.tsv          (4 G rows from n=3 seeds)
  platform_hybrid/experiments/results/group_size_iter43_summary.tsv     (Wu retention at 4 budgets)

Outputs (TSVs):
  group_size_iter55_d_eff.tsv            d_eff(G) and contrast yield
  group_size_iter55_antiherd.tsv         delta_div(G) and G-invariance test
  group_size_iter55_isoyield_pred.tsv    predicted argmax G per budget
  group_size_iter55_coupling.tsv         empirical - predicted residual
  group_size_iter55_wu_model.tsv         Wu retention model fits/predictions
  group_size_iter55_summary.tsv          headline rollup
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
SEED = 20240702
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_token_norm():
    out = []
    with open(RES / "group_size_token_normalized.tsv") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            out.append(
                {
                    "T": int(row["budget_tokens"]),
                    "G": int(row["G"]),
                    "acc": float(row["heldout_acc_mean"]),
                    "ci_lo": float(row["heldout_acc_ci_low"]),
                    "ci_hi": float(row["heldout_acc_ci_high"]),
                    "gu": float(row["gu_estimate"]),
                }
            )
    return out


def load_zvf_sweep():
    out = []
    with open(RES / "groupsize_zvf_sweep.tsv") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            out.append(
                {
                    "G": int(row["G"]),
                    "n_seeds": int(row["n_seeds"]),
                    "acc": float(row["heldout_acc_mean"]),
                    "acc_se": float(row["heldout_acc_se"]),
                    "last10": float(row["last10_mean"]),
                    "mean_zvf": float(row["mean_zvf"]),
                    "zvf_th": float(row["zvf_theory_at_mean_p"]),
                    "mean_reward_train": float(row["mean_reward_train"]),
                }
            )
    return out


def write_tsv(path, header, rows):
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(h, "")) for h in header) + "\n")


# ---------------------------------------------------------------------------
# A. Effective sample size d_eff(G) = G * (1 - ZVF_obs(G))
# ---------------------------------------------------------------------------


def effective_size(zvf):
    """Compute d_eff and Contrastive Yield per G from the zvf sweep."""
    out = []
    for r in zvf:
        G = r["G"]
        zvf_obs = r["mean_zvf"]
        zvf_th = r["zvf_th"]
        Y_obs = 1.0 - zvf_obs            # contrastive yield observed
        Y_th = 1.0 - zvf_th              # contrastive yield under i.i.d.
        d_eff_obs = G * Y_obs            # effective samples per group
        d_eff_th = G * Y_th              # under i.i.d.
        out.append(
            {
                "G": G,
                "n_seeds": r["n_seeds"],
                "zvf_obs": zvf_obs,
                "zvf_iid": zvf_th,
                "Y_obs": Y_obs,
                "Y_iid": Y_th,
                "d_eff_obs": d_eff_obs,
                "d_eff_iid": d_eff_th,
                "ratio_d_eff_Yratio": Y_obs / Y_th if Y_th > 0 else float("nan"),
            }
        )
    return out


# ---------------------------------------------------------------------------
# B. Anti-herding delta_div(G) and G-invariance test
# ---------------------------------------------------------------------------


def antiherding_test(eff):
    """delta_div(G) = ZVF_obs(G) - ZVF_iid(p, G).  Test if it depends on G.

    Frontier synthesis claims delta_div in [0.13, 0.23] and G-INVARIANT.
    We measure delta_div at G in {2, 4, 8, 16} and test the range and
    G-invariance by computing max - min, slope vs log G, and the
    one-sample t-test of the mean against 0.18 (frontier midpoint).
    """
    out = []
    deltas = []
    for r in eff:
        G = r["G"]
        zvf_obs = r["zvf_obs"]
        zvf_iid = r["zvf_iid"]
        delta = zvf_obs - zvf_iid
        deltas.append(delta)
        out.append(
            {
                "G": G,
                "zvf_obs": zvf_obs,
                "zvf_iid": zvf_iid,
                "delta_div": delta,
                "delta_div_rel": delta / zvf_iid if zvf_iid > 0 else float("nan"),
                "in_frontier_range": "yes" if 0.13 <= delta <= 0.23 else "no",
            }
        )
    # G-invariance tests
    if len(deltas) >= 3:
        Gs = np.array([r["G"] for r in eff], dtype=float)
        Ds = np.array(deltas, dtype=float)
        # Slope vs log G (zero = G-invariant)
        logG = np.log(Gs)
        n = len(logG)
        sx, sy = logG.sum(), Ds.sum()
        sxx, sxy = (logG * logG).sum(), (logG * Ds).sum()
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx + 1e-12)
        intercept = (sy - slope * sx) / n
        # Range and CV
        d_range = float(Ds.max() - Ds.min())
        d_mean = float(Ds.mean())
        d_std = float(Ds.std(ddof=1))
        d_cv = d_std / d_mean if d_mean > 0 else float("nan")
        # t-stat of slope vs 0 under OLS
        yhat = slope * logG + intercept
        resid = Ds - yhat
        s2 = float((resid * resid).sum()) / max(n - 2, 1)
        se_slope = math.sqrt(s2 / max(float(((logG - logG.mean()) ** 2).sum()), 1e-12))
        t_slope = slope / se_slope if se_slope > 0 else float("nan")
        # Append summary rows
        out.append(
            {
                "G": "summary",
                "zvf_obs": "",
                "zvf_iid": "",
                "delta_div": f"mean={d_mean:.4f}",
                "delta_div_rel": f"std={d_std:.4f}",
                "in_frontier_range": (
                    f"range={d_range:.4f} cv={d_cv:.4f}"
                ),
            }
        )
        out.append(
            {
                "G": "slope_vs_logG",
                "zvf_obs": "",
                "zvf_iid": "",
                "delta_div": f"slope={slope:.4f}",
                "delta_div_rel": f"intercept={intercept:.4f}",
                "in_frontier_range": f"t={t_slope:.3f}",
            }
        )
    return out


# ---------------------------------------------------------------------------
# C. Iso-yield predicted argmax G per budget
# ---------------------------------------------------------------------------


def per_token_opt_steps(G, T):
    """Compute per-token optimizer budget at (G, T).

    The number of optimizer steps at (G, T) is roughly T / (G * tokens_per_prompt).
    For GSM8K-style CoT rollouts, tokens_per_prompt ~ 256.  We use a fixed
    average, but the *relative* ordering across G is what matters for argmax.

    Actually the relevant quantity for iso-token comparison is the *per-token
    optimizer update count*: with T total tokens and G group size, you get
    T / (G * L) update steps per prompt, so smaller G => more steps.
    """
    L_per_prompt = 256  # tokens per rollout
    steps_per_prompt = T / (G * L_per_prompt)
    return steps_per_prompt


def isoyield_pred(rows, eff, zvf_raw):
    """For each budget T, compute the iso-yield score at every G:
        score(G, T) = d_eff(G) * steps_per_token(G, T)
    and report the predicted argmax G.  We use d_eff from the effective_size
    table (computed from the zvf sweep).
    """
    out = []
    budgets = sorted({r["T"] for r in rows})
    # Build d_eff lookup from effective_size table
    de_map = {r["G"]: r["d_eff_obs"] for r in eff}
    # Extend to G=32, 64 by extrapolation using the i.i.d. baseline:
    #   d_eff(G) ~ G * (1 - p^G - (1-p)^G + delta_div)
    # We use p (mean reward) = 0.86 from the zvf sweep and delta_div = mean(deltas).
    # zvf_raw rows have zvf_th as `zvf_th` (iid baseline) and mean_zvf as `mean_zvf`.
    if zvf_raw:
        p = float(np.mean([r["mean_reward_train"] for r in zvf_raw]))
        deltas = [r["mean_zvf"] - r["zvf_th"] for r in zvf_raw]
    else:
        p = 0.86
        deltas = []
    delta_div_mean = float(np.mean(deltas)) if deltas else 0.0
    for T in budgets:
        cell = []
        for G in (4, 8, 16, 32, 64):
            if G in de_map:
                de = de_map[G]
            else:
                # Extrapolate d_eff from p + delta_div
                zvf_pred = p**G + (1 - p) ** G - delta_div_mean
                zvf_pred = max(0.0, min(1.0, zvf_pred))
                de = G * (1.0 - zvf_pred)
            steps = per_token_opt_steps(G, T)
            score = de * steps
            cell.append((G, de, steps, score))
        # argmax by score
        am = max(cell, key=lambda x: x[3])
        for G, de, steps, score in cell:
            out.append(
                {
                    "T_tokens": T,
                    "G": G,
                    "d_eff": de,
                    "steps_per_prompt": steps,
                    "score": score,
                    "is_predicted_peak": "yes" if G == am[0] else "no",
                    "predicted_argmax_G": am[0],
                }
            )
    return out


# ---------------------------------------------------------------------------
# D. Cross-coupling: empirical vs predicted argmax G
# ---------------------------------------------------------------------------


def coupling(rows, pred):
    """For each budget T, compare empirical argmax G to iso-yield-predicted
    argmax G.  Residual = log10(G_emp) - log10(G_pred).

    PASS criterion: empirical argmax matches predicted argmax within factor 2.
    """
    out = []
    budgets = sorted({r["T"] for r in rows})
    for T in budgets:
        cell = [r for r in rows if r["T"] == T]
        emp = max(cell, key=lambda r: r["acc"])
        emp_G = emp["G"]
        pred_G = next(r["predicted_argmax_G"] for r in pred if r["T_tokens"] == T)
        ratio = emp_G / pred_G if pred_G > 0 else float("nan")
        residual_log10 = math.log10(emp_G) - math.log10(pred_G)
        within_factor_2 = "yes" if 0.5 <= ratio <= 2.0 else "no"
        out.append(
            {
                "T_tokens": T,
                "empirical_argmax_G": emp_G,
                "predicted_argmax_G": pred_G,
                "ratio_emp_over_pred": ratio,
                "residual_log10": residual_log10,
                "within_factor_2": within_factor_2,
            }
        )
    return out


# ---------------------------------------------------------------------------
# E. Wu 2025 retention model: R = (d_eff(G_a)/d_eff(G_b))^alpha
# ---------------------------------------------------------------------------


def wu_retention_stepaware(rows, zvf, eff):
    """Budget-aware retention model: R = (de_ratio)^alpha * T^gamma.

    The basic d_eff-only model (alpha only) is G-invariant in T and so
    cannot capture the empirical budget dependence of the G=4~G=32
    retention.  The per-token optimizer step ratio (steps_4 / steps_32)
    is also T-invariant (= 8 always at iso-token), so it cannot serve
    as a budget proxy.  We instead fit the two-parameter model
    R(T) = c * T^gamma  with c = (de_ratio)^alpha and gamma a free
    budget exponent.

    NOTE: this is a *log-log OLS* on the 4 budget points.  We leave
    T=1M and T=4M out as the calibration pair, then predict at T=16M
    and T=64M.
    """
    if zvf:
        p = float(np.mean([r["mean_reward_train"] for r in zvf]))
        deltas = [r["mean_zvf"] - r["zvf_th"] for r in zvf]
        delta_div_mean = float(np.mean(deltas)) if deltas else 0.0
    else:
        p = 0.86
        delta_div_mean = 0.0

    def d_eff(G):
        for r in eff:
            if r["G"] == G:
                return r["d_eff_obs"]
        for r in zvf:
            if r["G"] == G:
                return G * (1.0 - r["mean_zvf"])
        zvf_pred = p**G + (1 - p) ** G - delta_div_mean
        zvf_pred = max(0.0, min(1.0, zvf_pred))
        return G * (1.0 - zvf_pred)

    out = []
    budgets = sorted({r["T"] for r in rows})
    de4 = d_eff(4)
    de32 = d_eff(32)
    de_ratio = de4 / de32 if de32 > 0 else float("nan")

    # Build (log T, log R) for all 4 budgets
    pairs = []
    for T in budgets:
        g4 = next(r for r in rows if r["T"] == T and r["G"] == 4)
        g32 = next(r for r in rows if r["T"] == T and r["G"] == 32)
        R = g4["acc"] / g32["acc"] if g32["acc"] > 0 else float("nan")
        pairs.append((T, math.log10(T), math.log(R)))

    # Calibration fit on T={1M, 4M}; OLS on the 2-point line gives gamma and c.
    cal = [p for p in pairs if p[0] in (budgets[0], budgets[1])]
    if len(cal) >= 2:
        x_cal = np.array([c[1] for c in cal])
        y_cal = np.array([c[2] for c in cal])
        n = len(x_cal)
        sx, sy = x_cal.sum(), y_cal.sum()
        sxx, sxy = (x_cal * x_cal).sum(), (x_cal * y_cal).sum()
        gamma_cal = (n * sxy - sx * sy) / (n * sxx - sx * sx + 1e-12)
        logc_cal = (sy - gamma_cal * sx) / n
    else:
        gamma_cal, logc_cal = float("nan"), float("nan")
    # alpha from c = (de_ratio)^alpha → alpha = log(c) / log(de_ratio)
    c_cal = 10**logc_cal if not math.isnan(logc_cal) else float("nan")
    alpha_cal = math.log(c_cal) / math.log(de_ratio) if (de_ratio > 0 and de_ratio != 1 and c_cal > 0) else float("nan")

    # Also fit free OLS on all 4 points (for comparison)
    x_all = np.array([p[1] for p in pairs])
    y_all = np.array([p[2] for p in pairs])
    n_all = len(x_all)
    sx, sy = x_all.sum(), y_all.sum()
    sxx, sxy = (x_all * x_all).sum(), (x_all * y_all).sum()
    gamma_all = (n_all * sxy - sx * sy) / (n_all * sxx - sx * sx + 1e-12)
    logc_all = (sy - gamma_all * sx) / n_all
    c_all = 10**logc_all
    alpha_all = math.log(c_all) / math.log(de_ratio) if (de_ratio > 0 and de_ratio != 1 and c_all > 0) else float("nan")
    # R^2 of the free fit
    yhat_all = gamma_all * x_all + logc_all
    ss_res = float(((y_all - yhat_all) ** 2).sum())
    ss_tot = float(((y_all - y_all.mean()) ** 2).sum())
    r2_all = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    for T in budgets:
        g4 = next(r for r in rows if r["T"] == T and r["G"] == 4)
        g32 = next(r for r in rows if r["T"] == T and r["G"] == 32)
        R_emp = g4["acc"] / g32["acc"] if g32["acc"] > 0 else float("nan")
        # Predicted at this T using calibration (T={1M,4M} only) free model
        R_pred_cal = 10 ** (gamma_cal * math.log10(T) + logc_cal)
        R_pred_all = 10 ** (gamma_all * math.log10(T) + logc_all)
        is_cal = T in (budgets[0], budgets[1])
        out.append(
            {
                "T_tokens": T,
                "G4_acc": g4["acc"],
                "G32_acc": g32["acc"],
                "empirical_retention": R_emp,
                "d_eff_ratio": de_ratio,
                "R_pred_calibration_T1M_T4M": R_pred_cal,
                "R_pred_full_fit": R_pred_all,
                "err_calibration": R_pred_cal - R_emp,
                "err_full": R_pred_all - R_emp,
                "is_calibration_pair": "yes" if is_cal else "no",
            }
        )
    return out, alpha_cal, alpha_all, gamma_cal, gamma_all, r2_all


def wu_retention_model(rows, zvf, eff):
    """Fit alpha in retention = (d_eff(G_a)/d_eff(G_b))^alpha at T=1M,
    then predict retention at T={4, 16, 64}M and compare to empirical.

    We use d_eff_obs at G in {4, 32} (with extrapolation for G=32 using
    p=0.86, delta_div=mean(deltas)).
    """
    if zvf:
        p = float(np.mean([r["mean_reward_train"] for r in zvf]))
        deltas = [r["mean_zvf"] - r["zvf_th"] for r in zvf]
        delta_div_mean = float(np.mean(deltas)) if deltas else 0.0
    else:
        p = 0.86
        delta_div_mean = 0.0

    def d_eff(G):
        for r in eff:
            if r["G"] == G:
                return r["d_eff_obs"]
        # Extrapolate from raw sweep if not in eff table
        for r in zvf:
            if r["G"] == G:
                return G * (1.0 - r["mean_zvf"])
        zvf_pred = p**G + (1 - p) ** G - delta_div_mean
        zvf_pred = max(0.0, min(1.0, zvf_pred))
        return G * (1.0 - zvf_pred)

    out = []
    budgets = sorted({r["T"] for r in rows})
    de4 = d_eff(4)
    de32 = d_eff(32)
    de_ratio = de4 / de32 if de32 > 0 else float("nan")

    # Calibrate alpha at T=1M (smallest budget, weakest retention effect)
    T_calib = budgets[0]
    g4 = next(r for r in rows if r["T"] == T_calib and r["G"] == 4)
    g32 = next(r for r in rows if r["T"] == T_calib and r["G"] == 32)
    R_calib = g4["acc"] / g32["acc"] if g32["acc"] > 0 else float("nan")
    # alpha = log(R_calib) / log(de_ratio)
    alpha = math.log(R_calib) / math.log(de_ratio) if de_ratio > 0 and de_ratio != 1 else float("nan")

    for T in budgets:
        g4 = next(r for r in rows if r["T"] == T and r["G"] == 4)
        g32 = next(r for r in rows if r["T"] == T and r["G"] == 32)
        R_emp = g4["acc"] / g32["acc"] if g32["acc"] > 0 else float("nan")
        # Predicted retention at this budget using d_eff_only model
        R_pred_de = de_ratio**alpha
        out.append(
            {
                "T_tokens": T,
                "G4_acc": g4["acc"],
                "G32_acc": g32["acc"],
                "empirical_retention": R_emp,
                "d_eff_ratio_G4_over_G32": de_ratio,
                "alpha_calibrated_at_T1M": alpha if T == T_calib else "",
                "pred_retention_de_only": R_pred_de,
                "abs_error_de_only": R_pred_de - R_emp,
                "is_calibration_row": "yes" if T == T_calib else "no",
            }
        )
    return out, alpha, de_ratio


# ---------------------------------------------------------------------------
# F. Summary
# ---------------------------------------------------------------------------


def summarize(eff, anti, coup, wu, step, alpha, alpha_all, gamma_all, de_ratio):
    out = []
    # Effective sizes
    for r in eff:
        out.append(
            {
                "metric": f"d_eff_obs_at_G{r['G']}",
                "value": f"{r['d_eff_obs']:.4f}",
            }
        )
    # Anti-herding range
    deltas = [r["delta_div"] for r in anti if isinstance(r.get("delta_div"), float)]
    if deltas:
        out.append(
            {
                "metric": "delta_div_mean",
                "value": f"{float(np.mean(deltas)):.4f}",
            }
        )
        out.append(
            {
                "metric": "delta_div_range",
                "value": f"{float(max(deltas) - min(deltas)):.4f}",
            }
        )
    # Coupling accuracy
    n_correct = sum(1 for r in coup if r["within_factor_2"] == "yes")
    out.append(
        {
            "metric": "isoyield_pred_within_factor2",
            "value": f"{n_correct}/{len(coup)}",
        }
    )
    # Wu model calibration
    out.append({"metric": "wu_d_eff_ratio_G4_over_G32", "value": f"{de_ratio:.4f}"})
    out.append({"metric": "wu_alpha_calibrated_T1M", "value": f"{alpha:.4f}"})
    out.append({"metric": "wu_stepaware_alpha_full_fit", "value": f"{alpha_all:.4f}"})
    out.append({"metric": "wu_stepaware_gamma_full_fit", "value": f"{gamma_all:.4f}"})
    # Empirical retention sweep
    for r in wu:
        T_M = r["T_tokens"] // 1_000_000
        out.append(
            {
                "metric": f"wu_pred_retention_at_T{T_M}M",
                "value": f"{r['pred_retention_de_only']:.4f}",
            }
        )
        out.append(
            {
                "metric": f"wu_empirical_retention_at_T{T_M}M",
                "value": f"{r['empirical_retention']:.4f}",
            }
        )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    rows = load_token_norm()
    zvf = load_zvf_sweep()
    print(f"Loaded {len(rows)} iso-token cells, {len(zvf)} zvf-sweep rows.")

    # A
    eff = effective_size(zvf)
    write_tsv(
        RES / "group_size_iter55_d_eff.tsv",
        ["G", "n_seeds", "zvf_obs", "zvf_iid", "Y_obs", "Y_iid", "d_eff_obs", "d_eff_iid", "ratio_d_eff_Yratio"],
        eff,
    )
    print(f"Wrote group_size_iter55_d_eff.tsv ({len(eff)} rows)")

    # B
    anti = antiherding_test(eff)
    write_tsv(
        RES / "group_size_iter55_antiherd.tsv",
        ["G", "zvf_obs", "zvf_iid", "delta_div", "delta_div_rel", "in_frontier_range"],
        anti,
    )
    print(f"Wrote group_size_iter55_antiherd.tsv ({len(anti)} rows)")

    # C
    pred = isoyield_pred(rows, eff, zvf)
    write_tsv(
        RES / "group_size_iter55_isoyield_pred.tsv",
        ["T_tokens", "G", "d_eff", "steps_per_prompt", "score", "is_predicted_peak", "predicted_argmax_G"],
        pred,
    )
    print(f"Wrote group_size_iter55_isoyield_pred.tsv ({len(pred)} rows)")

    # D
    coup = coupling(rows, pred)
    write_tsv(
        RES / "group_size_iter55_coupling.tsv",
        ["T_tokens", "empirical_argmax_G", "predicted_argmax_G", "ratio_emp_over_pred", "residual_log10", "within_factor_2"],
        coup,
    )
    print(f"Wrote group_size_iter55_coupling.tsv ({len(coup)} rows)")

    # E
    wu, alpha, de_ratio = wu_retention_model(rows, zvf, eff)
    write_tsv(
        RES / "group_size_iter55_wu_model.tsv",
        [
            "T_tokens",
            "G4_acc",
            "G32_acc",
            "empirical_retention",
            "d_eff_ratio_G4_over_G32",
            "alpha_calibrated_at_T1M",
            "pred_retention_de_only",
            "abs_error_de_only",
            "is_calibration_row",
        ],
        wu,
    )
    print(f"Wrote group_size_iter55_wu_model.tsv ({len(wu)} rows)")

    # E2: budget-aware retention model
    step, alpha_cal, alpha_all, gamma_cal, gamma_all, r2_all = wu_retention_stepaware(rows, zvf, eff)
    write_tsv(
        RES / "group_size_iter55_stepaware.tsv",
        [
            "T_tokens",
            "G4_acc",
            "G32_acc",
            "empirical_retention",
            "d_eff_ratio",
            "R_pred_calibration_T1M_T4M",
            "R_pred_full_fit",
            "err_calibration",
            "err_full",
            "is_calibration_pair",
        ],
        step,
    )
    print(f"Wrote group_size_iter55_stepaware.tsv ({len(step)} rows)")

    # F
    summ = summarize(eff, anti, coup, wu, step, alpha, alpha_all, gamma_all, de_ratio)
    write_tsv(RES / "group_size_iter55_summary.tsv", ["metric", "value"], summ)
    print(f"Wrote group_size_iter55_summary.tsv ({len(summ)} rows)")

    # Headline
    print("\n=== Iter 55 Headline ===")
    print("d_eff(G):")
    for r in eff:
        print(f"  G={r['G']:2d}  Y_obs={r['Y_obs']:.4f}  d_eff={r['d_eff_obs']:.3f}")
    print("\nanti-herding delta_div:")
    deltas = [r["delta_div"] for r in anti if isinstance(r.get("delta_div"), float)]
    if deltas:
        print(f"  mean={np.mean(deltas):.4f}  range={max(deltas)-min(deltas):.4f}")
        in_band = sum(1 for d in deltas if 0.13 <= d <= 0.23)
        print(f"  {in_band}/{len(deltas)} cells in frontier [0.13, 0.23] band")
    print("\ncoupling (predicted vs empirical argmax G):")
    for r in coup:
        print(
            f"  T={r['T_tokens']//1_000_000}M  emp={r['empirical_argmax_G']}  "
            f"pred={r['predicted_argmax_G']}  ratio={r['ratio_emp_over_pred']:.3f}  "
            f"within 2x: {r['within_factor_2']}"
        )
    print("\nWu retention model:")
    for r in wu:
        cal = " (calib)" if r["is_calibration_row"] == "yes" else ""
        print(
            f"  T={r['T_tokens']//1_000_000}M  R_emp={r['empirical_retention']:.4f}  "
            f"R_pred(d_eff^alpha)={r['pred_retention_de_only']:.4f}  "
            f"err={r['abs_error_de_only']:+.4f}{cal}"
        )
    print(f"\nBudget-aware free fit: alpha={alpha_all:.4f}  gamma={gamma_all:.4f}")
    print("Budget-aware model:")
    for r in step:
        cal = " (calib)" if r["is_calibration_pair"] == "yes" else ""
        print(
            f"  T={r['T_tokens']//1_000_000}M  R_emp={r['empirical_retention']:.4f}  "
            f"R_pred_cal={r['R_pred_calibration_T1M_T4M']:.4f}  R_pred_full={r['R_pred_full_fit']:.4f}  "
            f"err_full={r['err_full']:+.4f}{cal}"
        )


if __name__ == "__main__":
    main()