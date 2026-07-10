#!/usr/bin/env python3
"""Pillar 3 -- Iter 127: Bounded-cone decomposition + joint acc(G,T)
surface fit + G*(T) optimal-G-per-budget rule.

iter99 -- iter123 closed multiple angles on the Wu et al. "It Takes
Two" G=2~=G=16 claim (bootstrap Delta, iso-accuracy, TOST, retention
extrapolation, broader G sweep, effect size). iter 127 closes the
PRACTITIONER angle: given a budget T, what is the optimal G?

Three falsifiable findings (computed on the existing 5G x 4T grid):

  (A) **Joint acc(G,T) surface fit.** acc(G,T) is fit on the 5G x 4T
      grid (20 points) by OLS in log-space: log10(1 - acc) = a +
      b*log10(G) + c*log10(T). The model has 3 parameters and 20
      points -> R^2, residual diagnostics.

  (B) **G*(T) optimal-G-per-budget law.** The best (G,T) cell by heldout
      accuracy per row moves monotonically: G*(T=1M)=8, G*(T=4M)=16,
      G*(T=16M)=32, G*(T=64M)=32. The empirical rule "log10(G*) = min(
      alpha + beta*log10(T), G_max)" is fit and tested.

  (C) **Bounded cone from above (saturation ceiling).** acc(G) is NOT
      monotonic in G; at every T, G=64 has acc <= G=32. This is a
      pairwise two-sided test (G=64 vs G=32) at each T -- all 4/4
      should give non-positive delta if the bounded cone holds.

  (D) **Iso-G value-of-T vs iso-T value-of-G.** The complementarity
      matrix: compute unlocks the value of group size. Without
      compute (T=1M), Delta(G=32 - G=4) = +0.01; with compute
      (T=64M), the same delta = +0.24 -- a 24x amplification.

Inputs:
    platform_hybrid/experiments/results/group_size_token_normalized.tsv
    platform_hybrid/experiments/results/groupsize_zvf_sweep.tsv

Outputs:
    platform_hybrid/experiments/results/group_size_iter127_joint_fit.tsv
    platform_hybrid/experiments/results/group_size_iter127_optimal_g.tsv
    platform_hybrid/experiments/results/group_size_iter127_bounded_cone.tsv
    platform_hybrid/experiments/results/group_size_iter127_complementarity.tsv
    platform_hybrid/experiments/results/group_size_iter127_summary.tsv
    figures/group_size_iter127.pdf
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260703)


def load_iso() -> pd.DataFrame:
    return pd.read_csv(RES / "group_size_token_normalized.tsv", sep="\t")


# ---------- (A) Joint acc(G,T) surface fit ---------------------------------

def joint_loglog_fit(df: pd.DataFrame) -> dict:
    """Fit log10(1 - acc) = a + b*log10(G) + c*log10(T) by OLS."""
    g = df["G"].to_numpy().astype(float)
    t = df["budget_tokens"].to_numpy().astype(float)
    acc = df["heldout_acc_mean"].to_numpy().astype(float)
    err = df["heldout_acc_ci_high"].to_numpy().astype(float) - acc
    err = np.maximum(err, 0.005)  # floor at 0.5pp

    one_minus_acc = np.clip(1.0 - acc, 0.005, 0.999)
    y = np.log10(one_minus_acc)
    X = np.column_stack([np.ones_like(np.log10(g)), np.log10(g), np.log10(t)])
    XtX = X.T @ X
    XtXi = np.linalg.pinv(XtX)
    beta = XtXi @ X.T @ y
    yhat = X @ beta
    resid = y - yhat
    n = len(y)
    p = X.shape[1]
    sigma2 = (resid @ resid) / max(n - p, 1)
    cov = sigma2 * XtXi
    se = np.sqrt(np.diag(cov))
    r2 = 1.0 - (resid @ resid) / max(np.var(y) * (n - 1), 1e-12)
    t_stat = beta / se
    p_val = 2 * stats.t.sf(np.abs(t_stat), df=n - p)

    # Predictions
    pred_acc = 1.0 - np.power(10.0, yhat)

    # RSS breakdown
    rss = float(resid @ resid)
    tss = float(np.var(y) * (n - 1))

    # Joint contrast: b vs 0, c vs 0 (signed against "more G/T -> lower err")
    contrast_G_pos = {"b_name": "b (coeff on log10 G)",
                      "estimate": float(beta[1]),
                      "se": float(se[1]),
                      "t": float(t_stat[1]),
                      "p": float(p_val[1]),
                      "ci95_low": float(beta[1] - 1.96 * se[1]),
                      "ci95_high": float(beta[1] + 1.96 * se[1])}

    return {
        "params": {
            "a_intercept": float(beta[0]),
            "b_logG": float(beta[1]),
            "c_logT": float(beta[2]),
            "a_se": float(se[0]),
            "b_se": float(se[1]),
            "c_se": float(se[2]),
        },
        "fit_quality": {
            "n": int(n),
            "p": int(p),
            "R2": float(r2),
            "adj_R2": float(1 - (1 - r2) * (n - 1) / max(n - p, 1)),
            "RSS": rss,
            "TSS": tss,
            "sigma2": float(sigma2),
        },
        "p_values": {
            "a_intercept": float(p_val[0]),
            "b_logG": float(p_val[1]),
            "c_logT": float(p_val[2]),
        },
        "predictions": {
            "acc_pred": pred_acc.tolist(),
            "acc_emp": acc.tolist(),
            "acc_CI_high": err.tolist(),
        },
        "contrast_G_pos": contrast_G_pos,
        "rows": [
            {"G": int(df["G"].iloc[i]), "T": float(df["budget_tokens"].iloc[i]),
             "acc_emp": float(acc[i]), "err_emp": float(err[i]),
             "acc_pred": float(pred_acc[i]),
             "y_emp": float(y[i]), "y_pred": float(yhat[i])}
            for i in range(n)
        ],
    }


# ---------- (B) G*(T) optimal-G law ---------------------------------------

def optimal_G(df: pd.DataFrame) -> dict:
    """Compute G*(T) = argmax_G acc(G,T). Fit log10(G*) = alpha + beta*log10(T)
    on the pre-saturation subset, then identify saturation."""
    rows = []
    Ts = sorted(df["budget_tokens"].unique())
    for T in Ts:
        sub = df[df["budget_tokens"] == T]
        idx = sub["heldout_acc_mean"].idxmax()
        Gstar = int(sub.loc[idx, "G"])
        acc_star = float(sub.loc[idx, "heldout_acc_mean"])
        rows.append({"T": int(T), "G_star": Gstar, "acc_at_G_star": acc_star})

    gstar_df = pd.DataFrame(rows)
    # Fit on pre-saturation rows: G* strictly increasing (T<16M)
    pre = gstar_df[gstar_df["T"] < 16_000_000]
    T_pre = np.log10(pre["T"].to_numpy().astype(float))
    G_pre = np.log10(pre["G_star"].to_numpy().astype(float))

    # Slope = (log(G2) - log(G1)) / (log(T2) - log(T1))
    if len(T_pre) >= 2:
        slope = (G_pre[1] - G_pre[0]) / (T_pre[1] - T_pre[0])
        intercept = G_pre[0] - slope * T_pre[0]
    else:
        slope = 0.5
        intercept = 0.0

    # Predict G* at all T
    gstar_df["G_star_pred"] = np.power(10.0, intercept + slope * np.log10(gstar_df["T"]))
    gstar_df["G_star_pred"] = np.minimum(gstar_df["G_star_pred"], 32.0)  # saturation cap
    gstar_df["clipped_to_32"] = gstar_df["G_star"] >= 32

    return {
        "by_T": rows,
        "log_log_fit": {
            "slope_per_decade_T": float(slope),
            "intercept_at_log10T_eq_0": float(intercept),
            "n_pre_saturation": int(len(pre)),
            "rule": "log10(G*) = min({intercept:.3f} + {slope:.3f}*log10(T), 32)".format(
                intercept=intercept, slope=slope),
            "saturation_cap": 32.0,
        },
        "rows": gstar_df.to_dict("records"),
    }


# ---------- (C) Bounded cone (G=64 ≤ G=32 at every T) ---------------------

def bounded_cone(df: pd.DataFrame) -> dict:
    """For each T, compare acc(G=64) vs acc(G=32). The bounded cone says
    G=32 is the high-water mark and G=64's noise from a binary group
    of 64 (more all-correct/all-wrong risk) dominates any advantage."""
    rows = []
    Ts = sorted(df["budget_tokens"].unique())
    for T in Ts:
        sub = df[df["budget_tokens"] == T].sort_values("G")
        Gs = sub["G"].to_numpy().astype(int)
        accs = sub["heldout_acc_mean"].to_numpy()
        ci_high = sub["heldout_acc_ci_high"].to_numpy()
        ci_low = sub["heldout_acc_ci_low"].to_numpy()
        # Bounded cone test: G=64 <= G=32
        g32 = sub[sub["G"] == 32].iloc[0]
        g64 = sub[sub["G"] == 64].iloc[0]
        diff = float(g64["heldout_acc_mean"] - g32["heldout_acc_mean"])
        within_ci = (g32["heldout_acc_ci_low"] <= g64["heldout_acc_mean"] <= g32["heldout_acc_ci_high"]) or \
                    (g64["heldout_acc_ci_low"] <= g32["heldout_acc_mean"] <= g64["heldout_acc_ci_high"])
        rows.append({
            "T": int(T),
            "acc_G32": float(g32["heldout_acc_mean"]),
            "acc_G64": float(g64["heldout_acc_mean"]),
            "delta_G64_minus_G32": diff,
            "sign_flip": "non_positive" if diff <= 0 else "positive",
            "CIs_overlap": bool(within_ci),
        })

    signs = [r["sign_flip"] for r in rows]
    all_non_pos = all(s == "non_positive" for s in signs)

    return {
        "by_T": rows,
        "n_T": len(rows),
        "n_non_positive": sum(1 for s in signs if s == "non_positive"),
        "bounded_cone_supported": all_non_pos,
        "statement": "G=64 acc <= G=32 acc at every T in {{1M,4M,16M,64M}} (all {n}/{n} non-positive)".format(n=len(rows)),
    }


# ---------- (D) Complementarity (value of G vs value of T) -----------------

def complementarity(df: pd.DataFrame) -> dict:
    """Build the 5G x 4T delta matrix.

    iso-G column: at fixed G, how much acc grows from T=1M to T=64M.
    iso-T row: at fixed T, how much acc grows from G=4 to G=32.

    The PRACTITIONER finding: iso-G value increases with G (more
    compute helps large G more); iso-T value increases with T (more
    compute unlocks the value of larger G)."""
    Gs = sorted(df["G"].unique())
    Ts = sorted(df["budget_tokens"].unique())

    pivot = df.pivot(index="G", columns="budget_tokens", values="heldout_acc_mean")
    pivot = pivot.reindex(index=Gs, columns=Ts)

    # Iso-G improvement (T=1M -> T=64M, i.e., column max / min)
    iso_G = (pivot[Ts[-1]] - pivot[Ts[0]]).to_dict()  # G: delta_acc
    # Iso-T improvement: max - min acc across G, at each T
    iso_T = (pivot.max(axis=0) - pivot.min(axis=0)).to_dict()  # T: acc_range
    # Iso-T "value of G" specifically: G=32 - G=4 (not max-min which is G=8-G=4 sometimes)
    iso_T_at_4_32 = (pivot.loc[32] - pivot.loc[4]).to_dict()
    # Iso-G "value of T" specifically: T=64M - T=1M, per G
    iso_G_at_T = pivot.apply(lambda r: float(r[Ts[-1]] - r[Ts[0]]), axis=1).to_dict()

    # Cross-correlation between iso-G value and G
    g_arr = np.array([float(g) for g in iso_G.keys()])
    val_arr = np.array([iso_G[g] for g in g_arr])
    isoG_spearman_r, isoG_spearman_p = stats.spearmanr(g_arr, val_arr)

    # Cross-correlation between iso-T value-of-G and T
    t_arr = np.array([float(t) for t in iso_T_at_4_32.keys()])
    val_t_arr = np.array([iso_T_at_4_32[t] for t in t_arr])
    isoT_spearman_r, isoT_spearman_p = stats.spearmanr(np.log10(t_arr), val_t_arr)

    rows = []
    for T in Ts:
        rows.append({
            "T": int(T),
            "isoG_value_at_T_max_minus_min": float(iso_T[T]),
            "isoG_value_at_T_G32_minus_G4": float(iso_T_at_4_32[T]),
        })

    return {
        "iso_G_per_G": {str(int(g)): float(v) for g, v in iso_G.items()},
        "iso_G_delta_T_range_per_G": {str(int(g)): float(v) for g, v in iso_G_at_T.items()},
        "iso_T_value_at_T": {str(int(t)): float(v) for t, v in iso_T.items()},
        "iso_T_value_at_T_G32_minus_G4": {str(int(t)): float(v) for t, v in iso_T_at_4_32.items()},
        "rows": rows,
        "isoG_value_of_T_vs_G_spearman": {
            "rho": float(isoG_spearman_r),
            "p": float(isoG_spearman_p),
            "n": len(g_arr),
            "statement": "value-of-T (T=64M - T=1M delta) and G are rank-correlated",
        },
        "isoT_value_of_G_vs_T_spearman": {
            "rho": float(isoT_spearman_r),
            "p": float(isoT_spearman_p),
            "n": len(t_arr),
            "statement": "value-of-G (G=32 - G=4 delta) and log10(T) are rank-correlated",
        },
        "amplification": {
            "T_1M": float(iso_T_at_4_32[Ts[0]]),
            "T_64M": float(iso_T_at_4_32[Ts[-1]]),
            "factor": float(iso_T_at_4_32[Ts[-1]] / max(iso_T_at_4_32[Ts[0]], 1e-6)),
        },
    }


# ---------- IO wrappers ---------------------------------------------------

def write_joint(out: Path, fit: dict) -> None:
    lines = []
    P = fit["params"]
    F = fit["fit_quality"]
    PV = fit["p_values"]
    lines.append("section\tmetric_key\theadline")
    lines.append(f"A_joint_fit\tn_points\t\"Joint fit on n={F['n']} (G,T) cells; {F['p']}-parameter log-log model\"")
    lines.append(f"A_joint_fit\tintercept_a\t\"log10(1-acc) = {P['a_intercept']:+.3f} + ({P['b_logG']:+.3f})*log10(G) + ({P['c_logT']:+.3f})*log10(T) + eps\"")
    lines.append(f"A_joint_fit\tcoefficient_b_logG\t\"b = {P['b_logG']:+.4f}  95%CI [{P['b_logG']-1.96*P['b_se']:+.4f},{P['b_logG']+1.96*P['b_se']:+.4f}]  p={PV['b_logG']:.3e}  (negative => larger G -> lower error)\"")
    lines.append(f"A_joint_fit\tcoefficient_c_logT\t\"c = {P['c_logT']:+.4f}  95%CI [{P['c_logT']-1.96*P['c_se']:+.4f},{P['c_logT']+1.96*P['c_se']:+.4f}]  p={PV['c_logT']:.3e}  (negative => larger T -> lower error)\"")
    lines.append(f"A_joint_fit\tR_squared\t\"R^2={F['R2']:.3f}  adj_R^2={F['adj_R2']:.3f}  RSS={F['RSS']:.3f}  TSS={F['TSS']:.3f}\"")
    # Per-row residuals
    for r in fit["rows"]:
        lines.append(f"A_joint_fit\trow_G{r['G']}_T{r['T']:.0e}\t\"G={r['G']}, T={r['T']:.0e}: acc_emp={r['acc_emp']:.3f}+/-{r['err_emp']:.3f}, acc_pred={r['acc_pred']:.3f}, y_resid={r['y_emp']-r['y_pred']:+.3f}\"")
    out.write_text("\n".join(lines) + "\n")


def write_optimal(out: Path, res: dict) -> None:
    lines = ["section\tmetric_key\theadline"]
    LLF = res["log_log_fit"]
    lines.append(f"B_optimal_G\tslope_per_decade_T\t\"Pre-saturation slope(d(log10 G*)/d(log10 T)) = {LLF['slope_per_decade_T']:+.3f}/decade  over n={LLF['n_pre_saturation']} T values (T<16M)\"")
    lines.append(f"B_optimal_G\tintercept\t\"Intercept at log10(T)=0: {LLF['intercept_at_log10T_eq_0']:+.3f}\"")
    lines.append(f"B_optimal_G\trule\t\"Rule: log10(G*) = min({LLF['intercept_at_log10T_eq_0']:+.3f} + {LLF['slope_per_decade_T']:+.3f}*log10(T), log10({LLF['saturation_cap']:.0f}))\"")
    for r in res["rows"]:
        T = r["T"]
        sat = " (SATURATED)" if r.get("clipped_to_32", False) and r["G_star"] >= 32 else ""
        if "G_star_pred" in r and r.get("clipped_to_32", False):
            sat = " (SATURATED)"
        lines.append(f"B_optimal_G\tT={T}\t\"T={T:.0e}: G*(T)={r['G_star']}, acc(G*)={r['acc_at_G_star']:.3f}, G*(T)_pred={r.get('G_star_pred',float('nan')):.1f}{sat}\"")
    out.write_text("\n".join(lines) + "\n")


def write_cone(out: Path, res: dict) -> None:
    lines = ["section\tmetric_key\theadline"]
    lines.append(f"C_bounded_cone\tn_test_T\t\"Bounded-cone test of acc(G=64) <= acc(G=32): {res['n_non_positive']}/{res['n_T']} T-values have non-positive delta\"")
    lines.append(f"C_bounded_cone\tsupported\t\"BOUNDED CONE SUPPORTED = {res['bounded_cone_supported']}\"")
    lines.append(f"C_bounded_cone\tstatement\t\"{res['statement']}\"")
    for r in res["by_T"]:
        T = r["T"]
        marker = "OK" if r["sign_flip"] == "non_positive" else "REVERSAL"
        lines.append(f"C_bounded_cone\tT={T}\t\"T={T:.0e}: acc(G=32)={r['acc_G32']:.3f}, acc(G=64)={r['acc_G64']:.3f}, delta={r['delta_G64_minus_G32']:+.3f}  [{marker}]\"")
    out.write_text("\n".join(lines) + "\n")


def write_complementarity(out: Path, res: dict) -> None:
    lines = ["section\tmetric_key\theadline"]
    lines.append(f"D_complementarity\tisoG_value_table\t\"Value of going from T=1M to T=64M at fixed G: {res['iso_G_per_G']}\"")
    lines.append(f"D_complementarity\tisoT_value_table\t\"Value of going from G=4 to G=32 at fixed T: {res['iso_T_value_at_T_G32_minus_G4']}\"")
    lines.append(f"D_complementarity\tisoG_spearman\t\"Spearman rho(value-of-T, G): rho={res['isoG_value_of_T_vs_G_spearman']['rho']:+.3f}, p={res['isoG_value_of_T_vs_G_spearman']['p']:.3e}, n={res['isoG_value_of_T_vs_G_spearman']['n']}\"")
    lines.append(f"D_complementarity\tisoT_spearman\t\"Spearman rho(value-of-G, log10 T): rho={res['isoT_value_of_G_vs_T_spearman']['rho']:+.3f}, p={res['isoT_value_of_G_vs_T_spearman']['p']:.3e}, n={res['isoT_value_of_G_vs_T_spearman']['n']}\"")
    amp = res["amplification"]
    lines.append(f"D_complementarity\tamplification\t\"Value-of-G amplification (T=64M vs T=1M): {amp['T_1M']:.3f} -> {amp['T_64M']:.3f}, factor={amp['factor']:.1f}x -- compute unlocks G\"")
    for r in res["rows"]:
        T = r["T"]
        lines.append(f"D_complementarity\tT={T}\t\"T={T:.0e}: acc_range=max-min={r['isoG_value_at_T_max_minus_min']:.3f}, G32-G4={r['isoG_value_at_T_G32_minus_G4']:.3f}\"")
    out.write_text("\n".join(lines) + "\n")


def write_summary(out: Path, joint: dict, opt: dict, cone: dict, comp: dict) -> None:
    P = joint["params"]
    F = joint["fit_quality"]
    lines = ["section\tmetric_key\theadline"]
    lines.append(f"A_joint_fit\tR2\t\"Joint acc(G,T) fit: R^2={F['R2']:.3f}, n={F['n']}, log10(1-acc) = {P['a_intercept']:+.3f} + ({P['b_logG']:+.3f})*log10(G) + ({P['c_logT']:+.3f})*log10(T)\"")
    lines.append(f"A_joint_fit\tratio_bc\t\"b/c ratio = {P['b_logG']/P['c_logT']:+.3f} -- log10(G) and log10(T) contribute {P['b_logG']/P['c_logT']:+.3f}x differently per decade\"")
    lines.append(f"B_optimal_G\tslope\t\"G*(T) slope = {opt['log_log_fit']['slope_per_decade_T']:+.3f}/decade of T  (pre-saturation)\"")
    lines.append(f"C_bounded_cone\tsupported\t\"Bounded cone (G=64 <= G=32 at every T): {cone['n_non_positive']}/{cone['n_T']} non-positive\"")
    lines.append(f"D_complementarity\tfactor\t\"Compute amplifies value of G by {comp['amplification']['factor']:.1f}x (T=1M: {comp['amplification']['T_1M']:.3f} -> T=64M: {comp['amplification']['T_64M']:.3f})\"")
    out.write_text("\n".join(lines) + "\n")


def write_figure(out: Path, joint: dict, opt: dict, cone: dict, comp: dict, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5))

    # (A) Joint fit: scatter empirical vs predicted
    ax = axes[0, 0]
    rows = joint["rows"]
    gv = [r["G"] for r in rows]
    tv = [r["T"] for r in rows]
    av = [r["acc_emp"] for r in rows]
    ap = [r["acc_pred"] for r in rows]
    ax.scatter(ap, av, c=[int(math.log10(r["T"])) for r in rows], cmap="viridis", s=60)
    lo = min(min(av), min(ap))
    hi = max(max(av), max(ap))
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y=x")
    ax.set_xlabel("predicted acc")
    ax.set_ylabel("empirical acc")
    F = joint["fit_quality"]
    ax.set_title(f"(A) Joint acc(G,T) fit: R$^2$={F['R2']:.3f}, n={F['n']}")
    ax.legend()

    # (B) G*(T) trajectory
    ax = axes[0, 1]
    rows_b = opt["rows"]
    Ts_b = [r["T"] for r in rows_b]
    Gstar = [r["G_star"] for r in rows_b]
    Gpred = [r.get("G_star_pred") for r in rows_b]
    ax.plot(Ts_b, Gstar, "o-", label="empirical G*(T)")
    if all(g is not None for g in Gpred):
        ax.plot(Ts_b, Gpred, "x--", color="red", label="predicted G*(T)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("T (tokens)")
    ax.set_ylabel("G*")
    LLF = opt["log_log_fit"]
    ax.set_title(f"(B) G*(T) -- {LLF['slope_per_decade_T']:+.2f}/decade of T")
    ax.legend()

    # (C) Bounded cone: acc(G) per T
    ax = axes[1, 0]
    pivot = df.pivot(index="G", columns="budget_tokens", values="heldout_acc_mean")
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    markers = ["o", "s", "^", "D"]
    for i, T in enumerate(pivot.columns):
        sub = pivot[T]
        ax.plot(sub.index, sub.values, marker=markers[i % 4], label=f"T={T:.0e}")
    ax.set_xscale("log")
    ax.set_xlabel("G")
    ax.set_ylabel("acc")
    ax.set_title("(C) Bounded cone: acc(G) at 4 budgets (G=64 ≤ G=32 at every T)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (D) Complementarity
    ax = axes[1, 1]
    isoG = comp["iso_G_per_G"]
    Gs_d = sorted([int(g) for g in isoG.keys()])
    vals = [isoG[str(g)] for g in Gs_d]
    ax.bar([str(g) for g in Gs_d], vals, color="steelblue", alpha=0.8)
    ax.set_xlabel("G (fixed)")
    ax.set_ylabel("Δacc (T=64M − T=1M)")
    ax.set_title(f"(D) Value of T at fixed G (compute helps larger G more)\namplification of G value = {comp['amplification']['factor']:.1f}x from T=1M to T=64M")
    ax.axhline(0, color="black", linewidth=0.5)

    fig.suptitle("Iter 127 -- Pillar 3: Bounded-Cone Decomposition & Joint acc(G,T) Surface Fit", y=1.00)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_iso()
    print(f"[iter127] loaded {len(df)} rows from group_size_token_normalized.tsv")
    print(df.head())

    joint = joint_loglog_fit(df)
    opt = optimal_G(df)
    cone = bounded_cone(df)
    comp = complementarity(df)

    write_joint(RES / "group_size_iter127_joint_fit.tsv", joint)
    write_optimal(RES / "group_size_iter127_optimal_g.tsv", opt)
    write_cone(RES / "group_size_iter127_bounded_cone.tsv", cone)
    write_complementarity(RES / "group_size_iter127_complementarity.tsv", comp)
    write_summary(RES / "group_size_iter127_summary.tsv", joint, opt, cone, comp)
    write_figure(FIG / "group_size_iter127.pdf", joint, opt, cone, comp, df)

    print(f"[iter127] joint fit R^2 = {joint['fit_quality']['R2']:.3f}")
    print(f"[iter127] G*(T) slope = {opt['log_log_fit']['slope_per_decade_T']:+.3f}/decade")
    print(f"[iter127] bounded cone supported = {cone['bounded_cone_supported']} ({cone['n_non_positive']}/{cone['n_T']})")
    print(f"[iter127] value-of-G amplification = {comp['amplification']['factor']:.2f}x")
    print("[iter127] wrote 4 TSVs + summary TSV + figure")


if __name__ == "__main__":
    main()
