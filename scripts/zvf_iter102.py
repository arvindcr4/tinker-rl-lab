#!/usr/bin/env python3
"""
zvf_iter102.py - Pillar 2 (ZVF): Predictive Calibration vs AERO/RL-ZVP.

Frontier synthesis (Round 2): "ZVF is observed signal availability, not difficulty.
ZVF = E_x [p_x^G + (1-p_x)^G]. The clean decomposition is
    ZVF_obs = ZVF_iid + Delta_G      (Delta_G = herding/anti-herding residual)."

iter98 materialised that decomposition as the over-dispersion ratio rho =
ZVF_emp / max(ZVF_iid, eps). iter102 takes the NEXT step: predict ZVF from
prompt-difficulty alone and ask whether each library runs at, above, or below
that prediction. The "calibration gap" Delta_G = ZVF_emp - ZVF_pred is the
operational version of "by how much is the library herding/anti-herding
beyond what difficulty forces".

Three fresh analyses on REAL measured data:

  1. PER-(LIBRARY, EXPERIMENT) CALIBRATION GAP
     For every row in zvf_summary.tsv (9 variance-mitigation libraries + 6
     cross-experiment families), compute
         p        = mean_reward (last10 proxy; fall back to peak)
         G        = group_size (default 8 if unknown)
         ZVF_iid  = p^G + (1-p)^G             (predicted, i.i.d. null)
         Delta    = ZVF_emp - ZVF_iid          (calibration gap; herding residual)
         rho_pred = ZVF_emp / max(ZVF_iid,eps)(over-dispersion ratio)
     Aggregate per-library mean Delta + 95% bootstrap CI over rows.

  2. AERO-vs-GRPO CALIBRATION TEST
     AERO/RL-ZVP (Le et al. 2025, arXiv:2509.21880) claims it detects zero-
     variance prompts and reshapes the advantage signal to mitigate them.
     The DIRECT empirical prediction: AERO should run at LOWER Delta than
     GRPO (it removes the herding pressure that creates ZVF in excess of
     i.i.d.). Test this on the variance_mitigation.tsv 5-seed runs.
     Also test the cross-experiment row (aero in zvf_by_library.tsv).

  3. CALIBRATION GAP vs FAILURE CORRELATION
     The headline diagnostic question for iter102: does the calibration gap
     Delta predict training failure (Nemotron-120B collapse, tool-use 0%)
     BETTER than raw ZVF?  Compute Pearson/Spearman rho of Delta vs
     collapse_label across all rows in zvf_summary.tsv, with bootstrap CIs.
     Compare to the iter94 raw-ZVF correlations (rho_Pearson=0.62 / rho_Spear=0.56).

Outputs:
    experiments/results/zvf_iter102_calibration.tsv   (one row per library+experiment)
    experiments/results/zvf_iter102_aero_test.tsv     (AERO vs GRPO paired)
    experiments/results/zvf_iter102_failure_corr.tsv  (Delta vs collapse)
    experiments/results/zvf_summary.tsv               (RE-EMIT, +iter102 columns)
    figures/zvf_vs_failure.pdf                        (RE-EMIT, 5-panel)
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

EPS = 1e-6
G_DEFAULT = 8


def _safe_div(a: float, b: float) -> float:
    return a / b if abs(b) > EPS else float("nan")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    if denom < EPS:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = math.sqrt(float((xm ** 2).sum() * (ym ** 2).sum()))
    if denom < EPS:
        return float("nan")
    return float((xm * ym).sum() / denom)


def _auc_rank(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    sum_ranks_pos = ranks[pos].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _bootstrap_ci(x: np.ndarray, y: np.ndarray, fn, B: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        try:
            boots[b] = fn(x[idx], y[idx])
        except Exception:
            boots[b] = float("nan")
    boots = boots[~np.isnan(boots)]
    if len(boots) < 10:
        return float("nan"), float("nan"), float("nan")
    return (float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 50)),
            float(np.percentile(boots, 97.5)))


def _write_tsv(path: pathlib.Path, rows, header_comment: str | None = None) -> None:
    with path.open("w") as f:
        if header_comment:
            for line in header_comment.splitlines():
                f.write(f"# {line}\n")
        if not rows:
            f.write("(empty)\n")
            return
        cols = list(rows[0].keys())
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


def _load_summary():
    """Read zvf_summary.tsv as a list of dicts (the canonical Pillar 2 row set)."""
    rows = []
    with (RES / "zvf_summary.tsv").open() as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    header = lines[0].rstrip("\n").split("\t")
    for line in lines[1:]:
        fields = line.rstrip("\n").split("\t")
        rows.append(dict(zip(header, fields)))
    return rows


def _load_by_library():
    """Read zvf_by_library.tsv (the cross-library aggregator)."""
    rows = []
    with (RES / "zvf_by_library.tsv").open() as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    header = lines[0].rstrip("\n").split("\t")
    for line in lines[1:]:
        fields = line.rstrip("\n").split("\t")
        rows.append(dict(zip(header, fields)))
    return rows


def _load_variance_mitigation_per_step():
    """Load per-step variance_mitigation.tsv; aggregate per (method, seed)."""
    rows = []
    with (RES / "variance_mitigation.tsv").open() as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    header = lines[0].rstrip("\n").split("\t")
    for line in lines[1:]:
        fields = line.rstrip("\n").split("\t")
        rows.append(dict(zip(header, fields)))
    # aggregate
    agg = {}
    for r in rows:
        key = (r["method"], r["seed"])
        try:
            zvf = float(r["zvf"])
            rwd = float(r["reward_mean"])
        except (KeyError, ValueError):
            continue
        agg.setdefault(key, {"method": r["method"], "seed": r["seed"],
                             "zvf_vals": [], "rwd_vals": []})
        agg[key]["zvf_vals"].append(zvf)
        agg[key]["rwd_vals"].append(rwd)
    out = []
    for k, v in agg.items():
        out.append(dict(
            method=v["method"], seed=v["seed"],
            mean_zvf=float(np.mean(v["zvf_vals"])),
            mean_reward=float(np.mean(v["rwd_vals"])),
            n_steps=len(v["zvf_vals"]),
        ))
    return out


# ---------------------------------------------------------------------------
# 1. PER-(LIBRARY, EXPERIMENT) CALIBRATION GAP
# ---------------------------------------------------------------------------

def calibration_table(by_lib_rows, summary_rows):
    """For each row in zvf_by_library, compute ZVF_iid prediction and Delta."""
    out = []
    for r in by_lib_rows:
        try:
            zvf_emp = float(r["mean_zvf"])
            last10 = float(r["mean_last10"])
            peak = float(r["mean_peak"])
        except (KeyError, ValueError):
            continue
        if math.isnan(zvf_emp) or math.isnan(last10):
            continue
        # Use last10 as the difficulty proxy; fall back to peak.
        # IMPORTANT: if last10 is exactly 0 (true collapse / 0% reward), p=0 and
        # ZVF_iid = 1.0 by definition; do NOT fall back to 0.5 (that creates an
        # artificial Delta = 1 - 2*0.5^G ~ 1 in collapse cases).
        if last10 >= 0:
            p = last10
        elif peak >= 0:
            p = peak
        else:
            continue  # skip rows with no usable difficulty
        # group size: arithmetic_groupsize uses 4, gsm8k_real uses 8, others 8
        lib = r["library"]
        if "arithmetic_groupsize" in lib:
            G = 4
        elif "gsm8k_real" in lib:
            G = 8
        elif "tool_use" in lib:
            G = 8
        elif "samestack" in lib:
            G = 4
        elif "drgrpo" in lib:
            G = 4
        else:
            G = G_DEFAULT
        zvf_iid = p ** G + (1.0 - p) ** G
        delta = zvf_emp - zvf_iid
        rho_pred = _safe_div(zvf_emp, max(zvf_iid, EPS))
        out.append(dict(
            library=lib, model=r["model"], G=G,
            p_difficulty=round(p, 4),
            zvf_emp=round(zvf_emp, 4),
            zvf_iid_pred=round(zvf_iid, 6),
            delta_calibration=round(delta, 4),
            rho_overdispersion=round(rho_pred, 4),
            collapse_rate=r.get("collapse_rate", ""),
            converged_rate=r.get("converged_rate", ""),
            evidence_path=r.get("evidence_path", ""),
        ))
    return out


def per_library_delta_summary(cal_rows):
    """Group calibration rows by library; mean Delta + bootstrap CI over rows."""
    libs = {}
    for r in cal_rows:
        libs.setdefault(r["library"], []).append(r)
    out = []
    for lib, rows in libs.items():
        deltas = np.array([r["delta_calibration"] for r in rows], dtype=float)
        rhos = np.array([r["rho_overdispersion"] for r in rows], dtype=float)
        zvf_iids = np.array([r["zvf_iid_pred"] for r in rows], dtype=float)
        zvf_emps = np.array([r["zvf_emp"] for r in rows], dtype=float)
        d_lo, d_med, d_hi = _bootstrap_ci(deltas, deltas, lambda x, y: float(x.mean()), B=1000, seed=lib.__hash__() & 0xFFFF)
        r_lo, r_med, r_hi = _bootstrap_ci(rhos, rhos, lambda x, y: float(x.mean()), B=1000, seed=lib.__hash__() & 0xFFFF)
        out.append(dict(
            library=lib,
            n_rows=len(rows),
            delta_mean=round(float(deltas.mean()), 4),
            delta_median=round(float(np.median(deltas)), 4),
            delta_ci_lo=round(d_lo, 4),
            delta_ci_hi=round(d_hi, 4),
            rho_mean=round(float(rhos.mean()), 4),
            rho_ci_lo=round(r_lo, 4),
            rho_ci_hi=round(r_hi, 4),
            zvf_iid_mean=round(float(zvf_iids.mean()), 4),
            zvf_emp_mean=round(float(zvf_emps.mean()), 4),
        ))
    # Sort by absolute delta (most herding libraries first)
    out.sort(key=lambda r: -abs(r["delta_mean"]))
    return out


# ---------------------------------------------------------------------------
# 2. AERO-vs-GRPO CALIBRATION TEST
# ---------------------------------------------------------------------------

def aero_vs_grpo_paired(vm_per_seed):
    """For each (method, seed) in variance_mitigation, compute predicted
    ZVF_iid = mean_reward^8 + (1-mean_reward)^8 (assume G=8 throughout
    the variance_mitigation runs) and the calibration Delta.
    Then paired-difference test: does AERO run at lower Delta than GRPO?
    """
    # aggregate to per-method
    methods = {}
    for r in vm_per_seed:
        methods.setdefault(r["method"], []).append(r)
    out_rows = []
    for method, runs in sorted(methods.items()):
        deltas = []
        rhos = []
        zvf_emps = []
        zvf_iids = []
        for r in runs:
            p = r["mean_reward"]
            G = G_DEFAULT
            zvf_iid = p ** G + (1.0 - p) ** G
            delta = r["mean_zvf"] - zvf_iid
            rho = _safe_div(r["mean_zvf"], max(zvf_iid, EPS))
            deltas.append(delta); rhos.append(rho)
            zvf_emps.append(r["mean_zvf"]); zvf_iids.append(zvf_iid)
        out_rows.append(dict(
            method=method,
            n_seeds=len(runs),
            zvf_emp_mean=round(float(np.mean(zvf_emps)), 4),
            zvf_iid_pred_mean=round(float(np.mean(zvf_iids)), 4),
            delta_mean=round(float(np.mean(deltas)), 4),
            rho_mean=round(float(np.mean(rhos)), 4),
            raw_zvf_above_iid_pct=round(float(np.mean(np.array(zvf_emps) > np.array(zvf_iids))) * 100, 1),
        ))
    # Paired AERO vs GRPO comparison: per-seed delta difference
    aero_runs = {r["seed"]: r for r in methods.get("aero", [])}
    grpo_runs = {r["seed"]: r for r in methods.get("grpo", [])}
    paired = []
    for seed in sorted(set(aero_runs) & set(grpo_runs)):
        a = aero_runs[seed]; g = grpo_runs[seed]
        a_p = a["mean_reward"]; g_p = g["mean_reward"]
        a_zvf_iid = a_p ** G_DEFAULT + (1.0 - a_p) ** G_DEFAULT
        g_zvf_iid = g_p ** G_DEFAULT + (1.0 - g_p) ** G_DEFAULT
        a_delta = a["mean_zvf"] - a_zvf_iid
        g_delta = g["mean_zvf"] - g_zvf_iid
        paired.append(dict(
            seed=seed,
            aero_zvf=a["mean_zvf"], aero_delta=round(a_delta, 4),
            grpo_zvf=g["mean_zvf"], grpo_delta=round(g_delta, 4),
            delta_aero_minus_grpo=round(a_delta - g_delta, 4),
            aero_below_grpo=a_delta < g_delta,
        ))
    if paired:
        diffs = np.array([r["delta_aero_minus_grpo"] for r in paired], dtype=float)
        aero_below = int(np.sum(diffs < 0))
        out_rows.append(dict(
            method="__PAIRED_AERO_VS_GRPO__",
            n_seeds=len(paired),
            zvf_emp_mean="",
            zvf_iid_pred_mean="",
            delta_mean=round(float(diffs.mean()), 4),
            rho_mean="",
            raw_zvf_above_iid_pct=f"{aero_below}/{len(paired)} below",
        ))
    return out_rows, paired


# ---------------------------------------------------------------------------
# 3. CALIBRATION GAP vs FAILURE CORRELATION
# ---------------------------------------------------------------------------

def failure_correlation(cal_rows):
    """Correlate Delta and raw ZVF with collapse indicators across all rows."""
    deltas = []
    zvf_emps = []
    collapse_labels = []
    converged_labels = []
    libs = []
    for r in cal_rows:
        try:
            cr = float(r["collapse_rate"])
            cv = float(r["converged_rate"]) if r["converged_rate"] not in ("", "NA") else float("nan")
        except (KeyError, ValueError):
            continue
        deltas.append(r["delta_calibration"])
        zvf_emps.append(r["zvf_emp"])
        collapse_labels.append(1 if cr > 0 else 0)
        libs.append(r["library"])
    if len(deltas) < 3:
        return []
    d = np.array(deltas, dtype=float)
    z = np.array(zvf_emps, dtype=float)
    y = np.array(collapse_labels, dtype=float)
    out = []
    # Pearson + Spearman for Delta
    pearson_d = _pearson(d, y)
    spear_d = _spearman(d, y)
    lo_d, med_d, hi_d = _bootstrap_ci(d, y, _pearson, B=2000, seed=1)
    out.append(dict(
        predictor="delta_calibration", target="is_collapse",
        n=len(d),
        pearson=round(pearson_d, 4),
        pearson_ci_lo=round(lo_d, 4),
        pearson_ci_hi=round(hi_d, 4),
        spearman=round(spear_d, 4),
        method="Pearson + 95% bootstrap CI",
    ))
    # Pearson + Spearman for raw ZVF (for direct comparison)
    pearson_z = _pearson(z, y)
    spear_z = _spearman(z, y)
    lo_z, med_z, hi_z = _bootstrap_ci(z, y, _pearson, B=2000, seed=2)
    out.append(dict(
        predictor="zvf_emp", target="is_collapse",
        n=len(z),
        pearson=round(pearson_z, 4),
        pearson_ci_lo=round(lo_z, 4),
        pearson_ci_hi=round(hi_z, 4),
        spearman=round(spear_z, 4),
        method="Pearson + 95% bootstrap CI",
    ))
    # AUC: continuous Delta score -> is_collapse
    auc_d = _auc_rank(y, d)
    auc_z = _auc_rank(y, z)
    out.append(dict(
        predictor="AUC(delta_calibration -> is_collapse)",
        target="binary",
        n=len(d),
        pearson=round(auc_d, 4),
        pearson_ci_lo="",
        pearson_ci_hi="",
        spearman="",
        method="Rank-AUC",
    ))
    out.append(dict(
        predictor="AUC(zvf_emp -> is_collapse)",
        target="binary",
        n=len(z),
        pearson=round(auc_z, 4),
        pearson_ci_lo="",
        pearson_ci_hi="",
        spearman="",
        method="Rank-AUC",
    ))
    return out


# ---------------------------------------------------------------------------
# RE-EMIT zvf_summary.tsv with iter102 columns
# ---------------------------------------------------------------------------

def reemit_summary(per_lib_delta, cal_corr, aero_paired, summary_rows):
    """Append iter102 calibration columns onto the existing dashboard."""
    by_lib = {r["library"]: r for r in per_lib_delta}
    out = []
    for row in summary_rows:
        r2 = dict(row)
        # summary_rows keyed by 'method' for variance_mitigation rows; those
        # don't have a 'library' column. Use the first column as the lookup
        # key for the variance_mitigation subset, and 'library' elsewhere.
        key = row.get("method", row.get("library", ""))
        delta = by_lib.get(key, {})
        r2["iter102_delta_calibration_mean"] = delta.get("delta_mean", "")
        r2["iter102_delta_calibration_ci_lo"] = delta.get("delta_ci_lo", "")
        r2["iter102_delta_calibration_ci_hi"] = delta.get("delta_ci_hi", "")
        r2["iter102_rho_overdispersion_mean"] = delta.get("rho_mean", "")
        r2["iter102_zvf_iid_pred_mean"] = delta.get("zvf_iid_mean", "")
        out.append(r2)
    # also writea paired delta correlation onto the LAST row as headline
    if cal_corr:
        last = out[-1]
        for cr in cal_corr:
            if cr["predictor"] == "delta_calibration":
                last["iter102_delta_vs_collapse_pearson"] = cr["pearson"]
                last["iter102_delta_vs_collapse_ci_lo"] = cr["pearson_ci_lo"]
                last["iter102_delta_vs_collapse_ci_hi"] = cr["pearson_ci_hi"]
                last["iter102_delta_vs_collapse_spearman"] = cr["spearman"]
            if cr["predictor"] == "AUC(delta_calibration -> is_collapse)":
                last["iter102_delta_collapse_auc"] = cr["pearson"]
    if aero_paired:
        last = out[-1]
        for r in aero_paired:
            if r.get("method") == "__PAIRED_AERO_VS_GRPO__":
                last["iter102_aero_minus_grpo_delta"] = r["delta_mean"]
                last["iter102_aero_below_grpo_count"] = r["raw_zvf_above_iid_pct"]
    return out


# ---------------------------------------------------------------------------
# FIGURE (5-panel: calibration vs AERO vs failure)
# ---------------------------------------------------------------------------

def make_figure(cal_rows, per_lib_delta, aero_paired, aero_paired_detail, cal_corr, out_pdf, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Panel 1: calibration curve predicted vs empirical
    ax = axes[0, 0]
    pred = np.array([r["zvf_iid_pred"] for r in cal_rows], dtype=float)
    emp = np.array([r["zvf_emp"] for r in cal_rows], dtype=float)
    ax.scatter(pred, emp, s=50, alpha=0.7, color="steelblue", edgecolor="black")
    lo = min(0, float(min(pred.min(), emp.min())))
    hi = max(1.0, float(max(pred.max(), emp.max())))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x (i.i.d. null)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$\mathrm{ZVF}_{\mathrm{iid}} = p^{G} + (1-p)^{G}$")
    ax.set_ylabel(r"$\mathrm{ZVF}_{\mathrm{emp}}$")
    ax.set_title("Calibration: predicted vs empirical ZVF")
    ax.legend(fontsize=8)

    # Panel 2: per-library Delta bar (signed, with CI)
    ax = axes[0, 1]
    libs = [r["library"] for r in per_lib_delta]
    deltas = [r["delta_mean"] for r in per_lib_delta]
    lo_ci = [r["delta_ci_lo"] for r in per_lib_delta]
    hi_ci = [r["delta_ci_hi"] for r in per_lib_delta]
    colors = ["tab:red" if d > 0 else "tab:blue" for d in deltas]
    y_pos = np.arange(len(libs))
    ax.barh(y_pos, deltas, color=colors, edgecolor="black", alpha=0.75)
    for i, (l, h) in enumerate(zip(lo_ci, hi_ci)):
        if not (math.isnan(l) or math.isnan(h)):
            ax.plot([l, h], [i, i], "k-", lw=1.5)
    ax.axvline(0.0, color="k", ls="-", lw=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(libs, fontsize=8)
    ax.set_xlabel(r"$\Delta = \mathrm{ZVF}_{\mathrm{emp}} - \mathrm{ZVF}_{\mathrm{iid}}$")
    ax.set_title("Per-library calibration gap")
    ax.invert_yaxis()

    # Panel 3: AERO vs GRPO paired delta scatter
    ax = axes[0, 2]
    if aero_paired_detail:
        seeds = [r["seed"] for r in aero_paired_detail]
        aero_d = [r["aero_delta"] for r in aero_paired_detail]
        grpo_d = [r["grpo_delta"] for r in aero_paired_detail]
        ax.scatter(grpo_d, aero_d, s=80, c="tab:purple", edgecolor="black")
        for s, gd, ad in zip(seeds, grpo_d, aero_d):
            ax.annotate(f"s={s}", (gd, ad), fontsize=7, xytext=(3, 3), textcoords="offset points")
        lo_b = min(min(grpo_d), min(aero_d)) - 0.05
        hi_b = max(max(grpo_d), max(aero_d)) + 0.05
        ax.plot([lo_b, hi_b], [lo_b, hi_b], "k--", lw=1, label="y=x (no advantage)")
        ax.set_xlim(lo_b, hi_b); ax.set_ylim(lo_b, hi_b)
        ax.set_xlabel(r"GRPO $\Delta$")
        ax.set_ylabel(r"AERO $\Delta$")
        ax.set_title("AERO vs GRPO paired calibration")
        ax.legend(fontsize=8)

    # Panel 4: Delta vs collapse indicator
    ax = axes[1, 0]
    deltas_d = []
    collapse_d = []
    libs_d = []
    for r in cal_rows:
        try:
            cr = float(r["collapse_rate"])
        except (KeyError, ValueError):
            continue
        deltas_d.append(r["delta_calibration"])
        collapse_d.append(1 if cr > 0 else 0)
        libs_d.append(r["library"])
    collapse_d = np.array(collapse_d, dtype=float)
    if deltas_d:
        d_arr = np.array(deltas_d, dtype=float)
        ax.scatter(d_arr, collapse_d + np.random.RandomState(0).uniform(-0.05, 0.05, size=len(d_arr)),
                   s=80, alpha=0.7, c="darkorange", edgecolor="black")
        for x, y, l in zip(d_arr, collapse_d, libs_d):
            ax.annotate(l, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(r"$\Delta$ (calibration gap)")
    ax.set_ylabel("collapse indicator (jittered)")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["no", "yes"])
    ax.set_title("Calibration gap vs collapse (all libraries)")

    # Panel 5: ZVF_iid prediction density vs empirical density
    ax = axes[1, 1]
    if len(pred) > 0 and len(emp) > 0:
        bins = np.linspace(0, 1.0, 16)
        ax.hist(pred, bins=bins, alpha=0.5, label="predicted i.i.d.", color="steelblue", density=True)
        ax.hist(emp, bins=bins, alpha=0.5, label="empirical", color="tab:red", density=True)
        ax.set_xlabel("ZVF")
        ax.set_ylabel("density")
        ax.set_title("Predicted vs empirical ZVF distribution")
        ax.legend(fontsize=8)

    # Panel 6: Pearson Delta vs Pearson ZVF correlation comparison
    ax = axes[1, 2]
    pearsons = []
    targets = []
    for cr in cal_corr:
        if cr["predictor"] in ("delta_calibration", "zvf_emp") and cr["target"] == "is_collapse":
            pearsons.append((cr["predictor"], cr["pearson"],
                             cr.get("pearson_ci_lo", float("nan")),
                             cr.get("pearson_ci_hi", float("nan"))))
    if pearsons:
        names = [p[0] for p in pearsons]
        vals = [p[1] for p in pearsons]
        lo = [p[2] for p in pearsons]
        hi = [p[3] for p in pearsons]
        x = np.arange(len(names))
        ax.bar(x, vals, color=["tab:red", "steelblue"], edgecolor="black", alpha=0.75)
        for i, (l, h) in enumerate(zip(lo, hi)):
            if not (math.isnan(l) or math.isnan(h)):
                ax.plot([i, i], [l, h], "k-", lw=2)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8, rotation=15)
        ax.set_ylabel(r"Pearson $\rho$ with $\mathrm{is\_collapse}$")
        ax.set_title("Predictor comparison (lower panel)")

    fig.suptitle("Iter 102 ZVF Predictive Calibration vs AERO", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("zvf_iter102 starting (predictive calibration vs AERO)", flush=True)
    by_lib = _load_by_library()
    summary = _load_summary()
    vm_per_seed = _load_variance_mitigation_per_step()
    print(f"  loaded {len(by_lib)} by_library rows, {len(summary)} summary rows, "
          f"{len(vm_per_seed)} variance_mitigation (method,seed) rows", flush=True)

    # 1. per-row calibration
    cal_rows = calibration_table(by_lib, summary)
    print(f"  computed calibration for {len(cal_rows)} rows", flush=True)
    per_lib = per_library_delta_summary(cal_rows)
    print(f"  aggregated {len(per_lib)} libraries", flush=True)

    # 2. AERO vs GRPO paired
    aero_per_lib, aero_paired_detail = aero_vs_grpo_paired(vm_per_seed)
    print(f"  AERO vs GRPO: {len(aero_per_lib)} methods, {len(aero_paired_detail)} paired seeds", flush=True)

# 3. failure correlation
    cal_corr = failure_correlation(cal_rows)
    print(f"  {len(cal_corr)} calibration-vs-failure correlations computed", flush=True)

    # Write outputs
    _write_tsv(RES / "zvf_iter102_calibration.tsv", cal_rows,
               header_comment="zvf_iter102_calibration.tsv - per-(library,experiment) calibration gap\n"
                              "Delta = ZVF_emp - (p^G+(1-p)^G), i.e. herding/anti-herding residual over the i.i.d. null.\n"
                              "Source: scripts/zvf_iter102.py")
    _write_tsv(RES / "zvf_iter102_per_library.tsv", per_lib,
               header_comment="zvf_iter102_per_library.tsv - aggregated calibration gap per library\n"
                              "with 95% bootstrap CIs. Source: scripts/zvf_iter102.py")
    _write_tsv(RES / "zvf_iter102_aero_test.tsv", aero_per_lib,
               header_comment="zvf_iter102_aero_test.tsv - per-method mean ZVF, predicted ZVF_iid,\n"
                              "and calibration gap on variance_mitigation runs (G=8 assumed).\n"
                              "Last row is the paired AERO-vs-GRPO summary. Source: scripts/zvf_iter102.py")
    _write_tsv(RES / "zvf_iter102_aero_paired.tsv", aero_paired_detail,
               header_comment="zvf_iter102_aero_paired.tsv - per-seed AERO vs GRPO paired calibration delta\n"
                              "from variance_mitigation.tsv. Source: scripts/zvf_iter102.py")
    _write_tsv(RES / "zvf_iter102_failure_corr.tsv", cal_corr,
               header_comment="zvf_iter102_failure_corr.tsv - correlation of calibration gap (and raw ZVF)\n"
                              "with is_collapse across all cross-library rows. B=2000 bootstrap CIs.\n"
                              "Source: scripts/zvf_iter102.py")

    # Re-emit zvf_summary.tsv
    new_summary = reemit_summary(per_lib, cal_corr, aero_per_lib, summary)
    with (RES / "zvf_summary.tsv").open("w") as f:
        f.write("# zvf_summary.tsv - Pillar 2 headline dashboard (iter102: +calibration columns).\n")
        f.write("# iter94 base columns + iter98 rho + iter102 delta_calibration, rho_overdispersion,\n"
                "# delta_vs_collapse correlations and AERO-vs-GRPO paired summary. Source: scripts/zvf_iter102.py\n")
        cols = list(new_summary[0].keys())
        f.write("\t".join(cols) + "\n")
        for r in new_summary:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    # Figure
    make_figure(cal_rows, per_lib, aero_per_lib, aero_paired_detail, cal_corr,
                FIG / "zvf_vs_failure.pdf", FIG / "zvf_vs_failure.png")
    print("  wrote figures/zvf_vs_failure.{pdf,png}", flush=True)

    # Headline console print
    print("\n=== iter102 headline ===", flush=True)
    print("Per-library calibration gap (top 5 by |Delta|):", flush=True)
    for r in per_lib[:5]:
        print(f"  {r['library']:25s}  Delta={r['delta_mean']:+.4f}  "
              f"CI=[{r['delta_ci_lo']:+.4f},{r['delta_ci_hi']:+.4f}]  "
              f"rho={r['rho_mean']:.3f}", flush=True)
    print("\nAERO-vs-GRPO paired Delta (variance_mitigation G=8):", flush=True)
    for r in aero_per_lib:
        if r["method"] in ("aero", "grpo", "__PAIRED_AERO_VS_GRPO__"):
            print(f"  {r['method']:25s}  delta={r['delta_mean']}", flush=True)
    print("\nFailure correlation (Pearson vs is_collapse):", flush=True)
    for cr in cal_corr:
        if cr["predictor"] in ("delta_calibration", "zvf_emp"):
            print(f"  {cr['predictor']:25s}  rho={cr['pearson']:+.4f}  "
                  f"CI=[{cr['pearson_ci_lo']:+.4f},{cr['pearson_ci_hi']:+.4f}]", flush=True)


if __name__ == "__main__":
    main()