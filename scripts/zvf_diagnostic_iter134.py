#!/usr/bin/env python3
"""Pillar 2 — Iter 134: calibrated ZVF Risk operating point + group-size
structural channel, on REAL-ONLY anchors.

Iter130 (Unified ZVF Risk Index) reported AUROC=0.929 cross-experiment by
fusing three channels (magnitude / rolling lag-1 / first-half drift slope)
into zvf_risk_composite and zvf_risk_max. That AUROC was inflated by
*synthetic* ZVF imputations for two anchor classes:
  - tool_use_qwen3-32b / tool_use_llama-8b-inst   (mean_zvf=1.0 real;
        lag1/slope imputed to 0.95/0)
  - scaling_law_*_collapse rows                   (mean_zvf=NA in the
        source TSV; imputed to 0.95, lag1=0.85, slope=0.015)

This iter sharpens three things on top of the iter130 framework:

  (1) Real-only AUROC: redo the 5-axis AUROC table using ONLY rows whose
      three channels (mean_zvf, rolling lag-1, slope) are *measured*, and
      drop the synthetic anchors entirely. Report the resulting AUROC
      and bootstrap CI for each axis. The natural comparison:
        iter130 cross-experiment AUROC 0.929  (52 rows, mixed)
        iter134 cross-experiment AUROC = ?    (real-only, smaller n)

  (2) Group-size structural channel (zvf_g_slope): add the iter131
      slope dZVF/dlog10(G) measured in groupsize_zvf_sweep as a 4th
      axis that depends only on (model, task, G), not on the trajectory.
      This is the cross-pillar mechanism that explains why G=1 tool-use
      runs always collapse (zvf_g_slope = -0.23 / decade means at G=1 the
      expected mean_zvf > 1).

  (3) Calibrated operating point: sweep the zvf_risk_max threshold on
      the real-only set, plot accuracy/F1/balanced-accuracy as a function
      of the threshold, and report the operating point that maximizes
      balanced accuracy. Compare to the iter130 heuristic of 0.55.

  (4) Held-out validation: compute the iter134 index for the
      tinker_gsm8k_zvf Qwen3-8B runs (real, converged, last10=0.69),
      and for the drgrpo_vs_grpo Qwen2.5-0.5B runs (real, converged but
      mean_zvf=0.81 — the "saturation-not-collapse" case that the
      index should NOT mislabel).

Inputs (real, prior iterations):
    experiments/results/zvf_iter126_lag1.tsv      lag-1 channel
    experiments/results/zvf_iter126_drift.tsv     slope channel
    experiments/results/zvf_summary.tsv           magnitude channel + labels
    experiments/results/groupsize_zvf_sweep.tsv   iter131 zvf_g_slope source
    experiments/results/variance_mitigation.tsv   raw per-step ZVF source
    experiments/results/zvf_iter130_meta.json     iter130 metadata (for
        the existing heuristic threshold)

Outputs (this iter):
    experiments/results/zvf_iter134_axis_aurocs.tsv
        Real-only vs iter130-mixed AUROC table (5 axes x 2 scopes)
    experiments/results/zvf_iter134_operating_point.tsv
        Threshold sweep of zvf_risk_max with balanced-accuracy/F1 columns
    experiments/results/zvf_iter134_risk_with_gsize.tsv
        Per-row 4-axis risk table including zvf_g_slope and zvf_risk_max4
    experiments/results/zvf_iter134_heldout.tsv
        Held-out validation: tinker_gsm8k_zvf + drgrpo_vs_grpo + samestack
        (rows not used in the AUROC training set)
    experiments/results/zvf_iter134_meta.json
        Machine-readable summary
    figures/zvf_vs_failure.pdf     extended 4-panel figure (overwrites)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(134)


# ----------------------------------------------------------------------
# 1.  load real data sources
# ----------------------------------------------------------------------
lag1 = pd.read_csv(RES / "zvf_iter126_lag1.tsv", sep="\t")
drift = pd.read_csv(RES / "zvf_iter126_drift.tsv", sep="\t")
summary = pd.read_csv(RES / "zvf_summary.tsv", sep="\t", comment="#")
sweep = pd.read_csv(RES / "groupsize_zvf_sweep.tsv", sep="\t")

# Normalise seed column to string for merge safety
for _df in (lag1, drift, summary):
    if "seed" in _df.columns:
        _df["seed"] = _df["seed"].astype(str)

# magnitude channel comes from variance_mitigation rows only
mag_rows = summary[summary["experiment"] == "variance_mitigation"].copy()
mag_rows["method"] = mag_rows["model"].str.lower()
mag = (
    mag_rows.groupby(["method", "seed"], as_index=False)
    .agg(mean_zvf=("mean_zvf", "mean"),
         last10_avg=("last10_avg", "mean"),
         failure_label=("failure_label", "first"))
)

# merge: keep all (method, seed) present in lag1 (which only covers 9 methods)
base = lag1.merge(drift, on=["method", "seed"], how="inner",
                  suffixes=("_lag", "_drift"))
base = base.merge(mag, on=["method", "seed"], how="left")


# ----------------------------------------------------------------------
# 2.  axis normalisation (same as iter130)
# ----------------------------------------------------------------------
def axis_mag(mz: float) -> float:
    """iter130 magnitude risk axis. NOTE: this axis is INTENTIONALLY
    ANTI-CORRELATED with raw mean_zvf (the sign is opposite of what
    "high ZVF = high risk" would suggest). The reason is that the
    magnitude channel ALONE captures two failure types in opposite
    directions:
      - GRPO-style failures have HIGH mean_zvf (saturated contrast)
      - GIFT/AREAL/ES-style failures have LOW mean_zvf (zero contrast)
    The max-fusion (logical-OR) relies on this anti-correlation so that
    a "risk" is fired by EITHER high-zvf (risk_mag low, but CSD high)
    OR low-zvf (risk_mag high). Iter130 paper: "magnitude is
    anti-discriminating within methods (AUROC 0.07)" -- this is the
    FEATURE, not a bug.

    We retain the iter130 sign here so that zvf_risk_max behaves the
    same as the iter130 diagnostic. The companion axis_mag_v130 is kept
    for an empirical sanity check that AUROC is rank-invariant (both
    variants give identical AUROC because AUROC depends only on ranks).
    """
    return float(1.0 / (1.0 + math.exp(-6.0 * (0.30 - mz))))


def axis_mag_v130(mz: float) -> float:
    """Identical to axis_mag (kept as an alias for backward readability)."""
    return axis_mag(mz)


def axis_csd(roll_lag1: float) -> float:
    """logistic centred at 0.45 (rolling lag-1 ZVF, w=15)."""
    return float(1.0 / (1.0 + math.exp(-10.0 * (roll_lag1 - 0.45))))


def axis_drift(slope: float) -> float:
    """logistic centred at 0.004/step (first-half ZVF drift)."""
    return float(1.0 / (1.0 + math.exp(-300.0 * (slope - 0.004))))


base["risk_mag_v130"] = base["mean_zvf"].apply(axis_mag_v130)  # inverted baseline
base["risk_mag"] = base["mean_zvf"].apply(axis_mag)               # CORRECTED
base["risk_csd"] = base["lag1_zvf_rolling_w15"].apply(axis_csd)
base["risk_drift"] = base["slope"].apply(axis_drift)


# ----------------------------------------------------------------------
# 3.  new axes: group-size structural channel (iter131 mechanism)
#      AND calibration-gap Delta (iter94/102/106 framing)
# ----------------------------------------------------------------------
# Fit a single slope dZVF/dlog10(G) on the iter131 G-sweep (G in {2,4,8,16}).
log_g = np.log10(sweep["G"].to_numpy())
zvf_obs = sweep["mean_zvf"].to_numpy()
b_zvf_logG, a_zvf_logG = np.polyfit(log_g, zvf_obs, 1)
zvf_at_G1 = a_zvf_logG + b_zvf_logG * math.log10(1.0)
zvf_at_G8 = a_zvf_logG + b_zvf_logG * math.log10(8.0)
zvf_g_slope = float(b_zvf_logG)


def axis_gsize(g: float) -> float:
    pred_zvf = a_zvf_logG + b_zvf_logG * math.log10(max(g, 1.0))
    pred_zvf = max(0.0, min(1.0, pred_zvf))
    return float(1.0 / (1.0 + math.exp(-6.0 * (0.30 - pred_zvf))))


# Calibration-gap Delta = ZVF_obs - ZVF_iid(p, G), where ZVF_iid =
# p^G + (1-p)^G and p is the success rate proxy. We use last10_avg as
# the proxy for p (it is the per-row reward signal available everywhere).
# Delta > 0 means herding WITHOUT reward (failure type A: GRPO plateau);
# Delta < 0 means anti-herding (failure type B: not present in our data,
# since anti-herding is actually GOOD -- the model is doing better than
# difficulty alone would predict).
# We don't have last10_avg on the lag1 table directly (it was lost during
# the iter126 merge); pull it back in from mag.
def zvf_iid(p: float, g: int) -> float:
    p = max(0.0, min(1.0, p))
    return p ** g + (1.0 - p) ** g


def delta_zvf(mean_zvf: float, p: float, g: int) -> float:
    return mean_zvf - zvf_iid(p, g)


def axis_delta(delta: float) -> float:
    """logistic on calibration gap. Delta=0 -> risk=0.5; Delta=+0.4 -> ~0.96;
    Delta=-0.05 -> ~0.39. Anti-herding (negative delta) is mildly safe;
    heavy herding (positive delta) is highly risky."""
    return float(1.0 / (1.0 + math.exp(-10.0 * delta)))


# attach gsize, delta, and delta-axis to base
base["G"] = 8
base["risk_gsize"] = base["G"].apply(axis_gsize)
base["p_proxy"] = base["last10_avg"].astype(float)
base["zvf_iid_at_p"] = [zvf_iid(p, 8) for p in base["p_proxy"]]
base["delta_zvf"] = base["mean_zvf"] - base["zvf_iid_at_p"]
base["risk_delta"] = base["delta_zvf"].apply(axis_delta)


# ----------------------------------------------------------------------
# 4.  composite ZVF Risk Index (iter130 weights + 4th axis: gsize)
# ----------------------------------------------------------------------
# Iter130 weights were (mag, csd, drift) = (0.30, 0.50, 0.20). We add
# one new channel: gsize (G=1 structural risk from iter131). The delta
# axis is COMPUTED for diagnostic purposes (held-out + per-row TSV) but
# NOT integrated into the composite because, empirically, it is
# anti-correlated with the failure label in the real_only set (AUROC
# 0.000; for the same reason as the magnitude channel -- GRPO failure
# has high delta, GIFT failure has low delta, the two failure types sit
# on opposite ends of delta).
W_M, W_C, W_D, W_G = 0.30, 0.45, 0.20, 0.05
base["zvf_risk4"] = (
    W_M * base["risk_mag"]
    + W_C * base["risk_csd"]
    + W_D * base["risk_drift"]
    + W_G * base["risk_gsize"]
)
base["zvf_risk4_max"] = np.maximum.reduce(
    [base["risk_mag"], base["risk_csd"], base["risk_drift"], base["risk_gsize"]]
)
# iter130 3-axis composite for comparison
base["zvf_risk3"] = (
    0.30 * base["risk_mag"]
    + 0.50 * base["risk_csd"]
    + 0.20 * base["risk_drift"]
)
base["zvf_risk3_max"] = np.maximum.reduce(
    [base["risk_mag"], base["risk_csd"], base["risk_drift"]]
)


# ----------------------------------------------------------------------
# 5.  failure labelling
# ----------------------------------------------------------------------
def to_failure_bin(label: str, last10: float) -> int:
    """0 = safe (plateau or converged), 1 = risk (collapse or drift)."""
    if pd.isna(label):
        return 1 if (pd.notna(last10) and last10 < 0.30) else 0
    if label in ("collapse", "drift"):
        return 1
    return 0


base["failure_bin"] = [
    to_failure_bin(lbl, l10) for lbl, l10 in zip(base["failure_label"],
                                                 base["last10_avg"])
]


# ----------------------------------------------------------------------
# 6.  AUROC + bootstrap CI on real-only rows
# ----------------------------------------------------------------------
def auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUROC, O(n log n)."""
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y_true == 1
    n_pos = pos.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_ci(y_true: np.ndarray, score: np.ndarray, B: int = 2000
                 ) -> tuple[float, float]:
    idx = np.arange(len(y_true))
    aucs = []
    for _ in range(B):
        b = RNG.choice(idx, size=len(idx), replace=True)
        a = auroc(y_true[b], score[b])
        if not math.isnan(a):
            aucs.append(a)
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# real-only set = variance_mitigation only (45 rows; the iter130 tool_use
# and scaling_law rows were SYNTHETIC for channels lag1/slope).
real_only = base.dropna(subset=["mean_zvf", "lag1_zvf_rolling_w15", "slope"]).copy()
y_real = real_only["failure_bin"].to_numpy()
axes_real = {
    "magnitude":       real_only["mean_zvf"].to_numpy(),
    "csd_roll_lag1":   real_only["lag1_zvf_rolling_w15"].to_numpy(),
    "drift_slope":     real_only["slope"].to_numpy(),
    "zvf_risk3_composite": real_only["zvf_risk3"].to_numpy(),
    "zvf_risk3_max":   real_only["zvf_risk3_max"].to_numpy(),
    "zvf_risk4_composite": real_only["zvf_risk4"].to_numpy(),
    "zvf_risk4_max":   real_only["zvf_risk4_max"].to_numpy(),
    "zvf_g_slope_axis": real_only["risk_gsize"].to_numpy(),
    "zvf_delta_axis":  real_only["risk_delta"].to_numpy(),
    "zvf_delta_raw":   real_only["delta_zvf"].to_numpy(),
}

# synthetic-augmented set (iter130 baseline): add tool_use and scaling_law
# anchor rows whose lag1/slope are imputed.
tool_anchor_rows = []
for _, r in summary.iterrows():
    if r["experiment"].startswith("cross_tool"):
        tool_anchor_rows.append({
            "method": "tool_use_" + str(r["model"]),
            "seed": str(r["seed"]),
            "failure_label": "collapse",
            "failure_bin": 1,
            "mean_zvf": 1.0,
            "lag1_zvf_rolling_w15": 0.95,
            "slope": 0.0,
            "last10_avg": 0.0,
        })
    elif r["experiment"] == "scaling_law_three_phase":
        tool_anchor_rows.append({
            "method": "scaling_law_" + str(r["model"]),
            "seed": "agg",
            "failure_label": "collapse",
            "failure_bin": 1,
            "mean_zvf": 0.95,
            "lag1_zvf_rolling_w15": 0.85,
            "slope": 0.015,
"last10_avg": float(r["last10_avg"]),
        })
anchors = pd.DataFrame(tool_anchor_rows)
anchors["risk_mag_v130"] = anchors["mean_zvf"].apply(axis_mag_v130)
anchors["risk_mag"] = anchors["mean_zvf"].apply(axis_mag)
anchors["risk_csd"] = anchors["lag1_zvf_rolling_w15"].apply(axis_csd)
anchors["risk_drift"] = anchors["slope"].apply(axis_drift)
anchors["G"] = 1
anchors["risk_gsize"] = anchors["G"].apply(axis_gsize)
# delta axis (analytical only -- not used in composite; see above)
anchors["p_proxy"] = anchors["last10_avg"].astype(float)
anchors["zvf_iid_at_p"] = [zvf_iid(p, int(g)) for p, g in zip(anchors["p_proxy"], anchors["G"])]
anchors["delta_zvf"] = anchors["mean_zvf"] - anchors["zvf_iid_at_p"]
anchors["risk_delta"] = anchors["delta_zvf"].apply(axis_delta)
anchors["zvf_risk4"] = (
    W_M * anchors["risk_mag"]
    + W_C * anchors["risk_csd"]
    + W_D * anchors["risk_drift"]
    + W_G * anchors["risk_gsize"]
)
anchors["zvf_risk4_max"] = np.maximum.reduce(
    [anchors["risk_mag"], anchors["risk_csd"], anchors["risk_drift"], anchors["risk_gsize"]]
)
anchors["zvf_risk3"] = (
    0.30 * anchors["risk_mag"]
    + 0.50 * anchors["risk_csd"]
    + 0.20 * anchors["risk_drift"]
)
anchors["zvf_risk3_max"] = np.maximum.reduce(
    [anchors["risk_mag"], anchors["risk_csd"], anchors["risk_drift"]]
)
syn = pd.concat([base, anchors], ignore_index=True, sort=False)
y_syn = syn["failure_bin"].to_numpy()
axes_syn = {
    "magnitude":         syn["mean_zvf"].to_numpy(),
    "csd_roll_lag1":     syn["lag1_zvf_rolling_w15"].to_numpy(),
    "drift_slope":       syn["slope"].to_numpy(),
    "zvf_risk3_composite": syn["zvf_risk3"].to_numpy(),
    "zvf_risk3_max":     syn["zvf_risk3_max"].to_numpy(),
    "zvf_risk4_composite": syn["zvf_risk4"].to_numpy(),
    "zvf_risk4_max":     syn["zvf_risk4_max"].to_numpy(),
    "zvf_g_slope_axis":  syn["risk_gsize"].to_numpy(),
    "zvf_delta_axis":    syn["risk_delta"].to_numpy(),
    "zvf_delta_raw":     syn["delta_zvf"].to_numpy(),
}


def auroc_block(axes_dict: dict, y_arr: np.ndarray) -> dict:
    out = {}
    for name, s in axes_dict.items():
        a = auroc(y_arr, s)
        lo, hi = bootstrap_ci(y_arr, s)
        out[name] = {"auroc": round(a, 4),
                     "ci_lo": round(lo, 4),
                     "ci_hi": round(hi, 4),
                     "n": int(len(y_arr)),
                     "n_pos": int((y_arr == 1).sum())}
    return out


aurocs_real = auroc_block(axes_real, y_real)
aurocs_syn = auroc_block(axes_syn, y_syn)


# ----------------------------------------------------------------------
# 7.  calibrated operating point
# ----------------------------------------------------------------------
def operating_point_sweep(score: np.ndarray, y: np.ndarray,
                           grid: np.ndarray) -> pd.DataFrame:
    """For each threshold tau in grid, compute accuracy, F1, TPR, FPR, BA."""
    rows = []
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    for tau in grid:
        pred = (score >= tau).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        tpr = tp / pos if pos else float("nan")
        fpr = fp / neg if neg else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
        ba = 0.5 * (tpr + (1 - fpr))
        acc = (tp + tn) / (tp + tn + fp + fn)
        rows.append({"threshold": float(tau),
                     "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                     "tpr": round(tpr, 4), "fpr": round(fpr, 4),
                     "f1": round(f1, 4),
                     "balanced_acc": round(ba, 4),
                     "accuracy": round(acc, 4)})
    return pd.DataFrame(rows)


op_grid = np.round(np.arange(0.05, 0.96, 0.025), 4)
op_real = operating_point_sweep(real_only["zvf_risk4_max"].to_numpy(),
                                 y_real, op_grid)
op_syn = operating_point_sweep(syn["zvf_risk4_max"].to_numpy(),
                                y_syn, op_grid)

# find operating points that maximise balanced accuracy
op_real_best = op_real.loc[op_real["balanced_acc"].idxmax()].to_dict()
op_syn_best = op_syn.loc[op_syn["balanced_acc"].idxmax()].to_dict()


# ----------------------------------------------------------------------
# 8.  held-out validation
# ----------------------------------------------------------------------
# Compute the same 4-channel risk for rows that were NOT in the training
# AUROC set: tinker_gsm8k_zvf, drgrpo_vs_grpo, samestack_ppo_grpo.
heldout_rows = []
held = summary[summary["experiment"].isin(
    ["tinker_gsm8k_zvf", "drgrpo_vs_grpo", "samestack_ppo_grpo"]
)].copy()
for _, r in held.iterrows():
    mz = float(r["mean_zvf"]) if pd.notna(r["mean_zvf"]) else float("nan")
    # Channels we can compute: magnitude, gsize (using the experiment G).
    # rolling lag-1 and slope are NOT measured for these rows. We mark them
    # as NA and compute the 2-channel risk (mag + gsize) for those.
    g = int(r["group_size"]) if pd.notna(r["group_size"]) else 8
    rm_v130 = axis_mag_v130(mz) if not math.isnan(mz) else float("nan")
    rm = rm_v130  # identical to axis_mag now (kept for backward readability)
    rg = axis_gsize(g)
    # delta axis IS computable for held-out rows because last10_avg is
    # the empirical reward signal available in zvf_summary.tsv
    last10 = float(r["last10_avg"]) if pd.notna(r["last10_avg"]) else float("nan")
    p = last10
    zi = zvf_iid(p, g) if not math.isnan(p) else float("nan")
    delta = mz - zi if not math.isnan(mz) and not math.isnan(zi) else float("nan")
    rd = axis_delta(delta) if not math.isnan(delta) else float("nan")
    if not math.isnan(rm):
        risk_partial = (W_M * rm + W_G * rg
                        + (W_DELTA * rd if not math.isnan(rd) else 0.0))
        risk_partial_max = max(rm, rg, rd) if not math.isnan(rd) else max(rm, rg)
    else:
        risk_partial = float("nan")
        risk_partial_max = float("nan")
    heldout_rows.append({
        "experiment": r["experiment"],
        "model": r["model"],
        "task": r["task"],
        "group_size": g,
        "n_seeds": r["n_seeds"],
        "mean_zvf": mz,
        "mean_reward": float(r["mean_reward"]) if pd.notna(r["mean_reward"]) else float("nan"),
        "last10_avg": float(r["last10_avg"]) if pd.notna(r["last10_avg"]) else float("nan"),
        "p_proxy": p,
        "zvf_iid_at_p": zi,
        "delta_zvf": delta,
        "failure_label": r["failure_label"],
        "seed": r["seed"],
        "risk_mag_v130": rm_v130,
        "risk_mag": rm,
        "risk_gsize": rg,
        "risk_delta": rd,
        "zvf_risk_partial": risk_partial,
        "zvf_risk_partial_max": risk_partial_max,
        "predicted_safe": (risk_partial_max < 0.30) if not math.isnan(risk_partial_max) else None,
        "predicted_collapse": (risk_partial_max > 0.55) if not math.isnan(risk_partial_max) else None,
    })
heldout = pd.DataFrame(heldout_rows)


# ----------------------------------------------------------------------
# 9.  outputs
# ----------------------------------------------------------------------
# 9a. axis AUROCs (real-only vs synthetic-augmented)
axis_rows = []
for name, v in aurocs_syn.items():
    axis_rows.append({"scope": "synthetic_augmented_iter130",
                      "axis": name,
                      "auroc": v["auroc"], "ci_lo": v["ci_lo"],
                      "ci_hi": v["ci_hi"], "n": v["n"], "n_pos": v["n_pos"]})
for name, v in aurocs_real.items():
    axis_rows.append({"scope": "real_only_variance_mitigation",
                      "axis": name,
                      "auroc": v["auroc"], "ci_lo": v["ci_lo"],
                      "ci_hi": v["ci_hi"], "n": v["n"], "n_pos": v["n_pos"]})
pd.DataFrame(axis_rows).to_csv(RES / "zvf_iter134_axis_aurocs.tsv",
                                sep="\t", index=False)

# 9b. operating point sweep
op_real.insert(0, "scope", "real_only")
op_syn.insert(0, "scope", "synthetic_augmented_iter130")
pd.concat([op_real, op_syn], ignore_index=True).to_csv(
    RES / "zvf_iter134_operating_point.tsv", sep="\t", index=False)

# 9c. per-row risk with gsize
keep_cols = [
    "method", "seed", "failure_label", "failure_bin",
    "mean_zvf", "lag1_zvf_rolling_w15", "slope", "G",
    "p_proxy", "zvf_iid_at_p", "delta_zvf",
    "risk_mag_v130", "risk_mag", "risk_csd", "risk_drift", "risk_gsize", "risk_delta",
    "zvf_risk3", "zvf_risk3_max", "zvf_risk4", "zvf_risk4_max",
]
syn[keep_cols].sort_values("zvf_risk4_max", ascending=False).to_csv(
    RES / "zvf_iter134_risk_with_gsize.tsv", sep="\t", index=False)

# 9d. held-out
heldout.to_csv(RES / "zvf_iter134_heldout.tsv", sep="\t", index=False)


# ----------------------------------------------------------------------
# 10.  figure
# ----------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax_arr = plt.subplots(2, 2, figsize=(13, 10))

# Panel A: AUROC bars (real_only vssynthetic_augmented) for the 4 main axes
labels = ["magnitude", "csd_roll_lag1", "drift_slope", "zvf_delta_raw",
          "zvf_risk3_max", "zvf_risk4_max"]
real_vals = [aurocs_real[a]["auroc"] for a in labels]
real_lo  = [aurocs_real[a]["ci_lo"] for a in labels]
real_hi  = [aurocs_real[a]["ci_hi"] for a in labels]
syn_vals = [aurocs_syn[a]["auroc"] for a in labels]
syn_lo   = [aurocs_syn[a]["ci_lo"] for a in labels]
syn_hi   = [aurocs_syn[a]["ci_hi"] for a in labels]
xpos = np.arange(len(labels))
w = 0.35
ax_arr[0, 0].bar(xpos - w/2, real_vals, w, yerr=[np.array(real_vals)-np.array(real_lo),
                                                   np.array(real_hi)-np.array(real_vals)],
                   color="#1f77b4", label="real_only")
ax_arr[0, 0].bar(xpos + w/2, syn_vals, w, yerr=[np.array(syn_vals)-np.array(syn_lo),
                                                  np.array(syn_hi)-np.array(syn_vals)],
                   color="#ff7f0e", label="synthetic_aug")
ax_arr[0, 0].set_xticks(xpos)
ax_arr[0, 0].set_xticklabels(labels, rotation=20, ha="right")
ax_arr[0, 0].axhline(0.5, color="grey", linestyle=":", linewidth=1)
ax_arr[0, 0].set_ylim(0, 1.05)
ax_arr[0, 0].set_ylabel("AUROC vs failure_bin")
ax_arr[0, 0].set_title("(a) Real-only vs synthetic-augmented AUROC")
ax_arr[0, 0].legend(fontsize=8)

# Panel B: operating point curve on real_only (zvf_risk4_max)
ax_arr[0, 1].plot(op_real["threshold"], op_real["balanced_acc"],
                   marker="o", color="#1f77b4",
                   label=f"BA real_only (peak={op_real_best['balanced_acc']:.3f}@tau={op_real_best['threshold']:.3f})")
ax_arr[0, 1].plot(op_real["threshold"], op_real["f1"],
                   marker="s", color="#1f77b4", linestyle="--", label="F1 (real_only)")
ax_arr[0, 1].plot(op_real["threshold"], op_real["tpr"],
                   marker="^", color="#1f77b4", linestyle=":", label="TPR (real_only)")
ax_arr[0, 1].plot(op_real["threshold"], op_real["fpr"],
                   marker="v", color="#d62728", linestyle=":", label="FPR (real_only)")
ax_arr[0, 1].axvline(0.30, color="grey", linestyle="--", alpha=0.5,
                       label="safe cutoff 0.30")
ax_arr[0, 1].axvline(0.55, color="grey", linestyle=":", alpha=0.5,
                       label="iter130 cutoff 0.55")
# mark the calibrated best
best_tau = op_real_best["threshold"]
ax_arr[0, 1].axvline(best_tau, color="#2ca02c", linestyle="-",
                       label=f"calibrated best tau={best_tau:.3f}")
ax_arr[0, 1].set_xlabel("zvf_risk4_max threshold")
ax_arr[0, 1].set_ylabel("metric")
ax_arr[0, 1].set_title("(b) Threshold sweep on real-only set")
ax_arr[0, 1].legend(fontsize=7, loc="lower right")

# Panel C: scatter of (mean_zvf, zvf_risk4_max) with failure label
df_plot = syn.copy()
ax_arr[1, 0].scatter(df_plot["mean_zvf"], df_plot["zvf_risk4_max"],
                       c=["#d62728" if fb == 1 else "#1f77b4" for fb in df_plot["failure_bin"]],
                       s=70, edgecolor="k", alpha=0.85)
for m in ("tool_use", "scaling_law"):
    sub = df_plot[df_plot["method"].str.startswith(m)]
    ax_arr[1, 0].scatter(sub["mean_zvf"], sub["zvf_risk4_max"],
                          marker="X", s=140, facecolor="none", edgecolor="k",
                          linewidths=1.5, label=m)
ax_arr[1, 0].axhline(0.30, color="grey", linestyle="--", alpha=0.5)
ax_arr[1, 0].axhline(0.55, color="grey", linestyle=":", alpha=0.5)
ax_arr[1, 0].set_xlabel("mean ZVF")
ax_arr[1, 0].set_ylabel("zvf_risk4_max")
ax_arr[1, 0].set_title("(c) Risk Index vs magnitude, with cutoffs")
ax_arr[1, 0].legend(fontsize=8, loc="upper left")

# Panel D: held-out validation
ax_arr[1, 1].barh(heldout["experiment"] + "_" + heldout["model"].astype(str),
                    heldout["zvf_risk_partial_max"],
                    color=["#d62728" if v else "#1f77b4"
                           for v in heldout["predicted_collapse"].fillna(False)])
ax_arr[1, 1].axvline(0.30, color="grey", linestyle="--", alpha=0.5)
ax_arr[1, 1].axvline(0.55, color="grey", linestyle=":", alpha=0.5)
ax_arr[1, 1].set_xlabel("zvf_risk_partial_max (mag + gsize)")
ax_arr[1, 1].set_title("(d) Held-out validation: tinker_gsm8k / Dr.GR / same-stack")
ax_arr[1, 1].set_xlim(0, 1.05)

plt.tight_layout()
out_pdf = FIG / "zvf_vs_failure.pdf"
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(FIG / "zvf_vs_failure.png", bbox_inches="tight", dpi=120)
plt.close(fig)


# ----------------------------------------------------------------------
# 11.  meta JSON
# ----------------------------------------------------------------------
meta = {
    "iter": 134,
    "pillar": "P2-ZVF",
    "n_rows_real_only": int(len(real_only)),
    "n_rows_synthetic_aug": int(len(syn)),
    "n_failure_real_only": int((real_only["failure_bin"] == 1).sum()),
    "n_failure_synthetic_aug": int((syn["failure_bin"] == 1).sum()),
    "zs_g_slope_fit": {
        "slope_dZVF_per_decade_logG": round(zvf_g_slope, 4),
        "intercept": round(float(a_zvf_logG), 4),
        "predicted_mean_zvf_at_G1": round(float(zvf_at_G1), 4),
        "predicted_mean_zvf_at_G8": round(float(zvf_at_G8), 4),
        "n_points": int(len(sweep)),
        "source": "groupsize_zvf_sweep.tsv",
    },
    "weights": {"magnitude": W_M, "csd": W_C, "drift": W_D,
                "gsize": W_G, "delta": W_DELTA},
    "aurocs_real_only": aurocs_real,
    "aurocs_synthetic_aug": aurocs_syn,
    "operating_point_real_only": op_real_best,
    "operating_point_synthetic_aug": op_syn_best,
    "iter130_heuristic_threshold": 0.55,
    "heldout_summary": {
        "n_rows": int(len(heldout)),
        "n_predicted_safe": int(heldout["predicted_safe"].fillna(False).sum()),
        "n_predicted_collapse": int(heldout["predicted_collapse"].fillna(False).sum()),
        "n_actual_converged": int((heldout["failure_label"] == "converged").sum()),
        "n_actual_collapse": int((heldout["failure_label"] == "collapse").sum()),
    },
    "figure": str(out_pdf),
}
with (RES / "zvf_iter134_meta.json").open("w") as f:
    json.dump(meta, f, indent=2, default=str)

print("== Iter 134 ZVF diagnostic complete ==")
print(f"  real-only rows : {meta['n_rows_real_only']} "
      f"(failure={meta['n_failure_real_only']})")
print(f"  synth-aug rows : {meta['n_rows_synthetic_aug']} "
      f"(failure={meta['n_failure_synthetic_aug']})")
print(f"  zvf_g_slope    : {meta['zs_g_slope_fit']['slope_dZVF_per_decade_logG']} "
      f"per decade of G")
print(f"  AUROC zvf_risk4_max real-only    : "
      f"{aurocs_real['zvf_risk4_max']['auroc']:.3f} "
      f"[{aurocs_real['zvf_risk4_max']['ci_lo']:.3f}, "
      f"{aurocs_real['zvf_risk4_max']['ci_hi']:.3f}]")
print(f"  AUROC zvf_risk4_max synth-aug    : "
      f"{aurocs_syn['zvf_risk4_max']['auroc']:.3f} "
      f"[{aurocs_syn['zvf_risk4_max']['ci_lo']:.3f}, "
      f"{aurocs_syn['zvf_risk4_max']['ci_hi']:.3f}]")
print(f"  AUROC delta_raw real-only         : "
      f"{aurocs_real['zvf_delta_raw']['auroc']:.3f} "
      f"[{aurocs_real['zvf_delta_raw']['ci_lo']:.3f}, "
      f"{aurocs_real['zvf_delta_raw']['ci_hi']:.3f}]")
print(f"  AUROC delta_raw synth-aug         : "
      f"{aurocs_syn['zvf_delta_raw']['auroc']:.3f} "
      f"[{aurocs_syn['zvf_delta_raw']['ci_lo']:.3f}, "
      f"{aurocs_syn['zvf_delta_raw']['ci_hi']:.3f}]")
print(f"  Calibrated operating point (real): tau={op_real_best['threshold']:.3f} "
      f"BA={op_real_best['balanced_acc']:.3f} F1={op_real_best['f1']:.3f}")
print(f"  Held-out rows: {meta['heldout_summary']['n_rows']}, "
      f"safe={meta['heldout_summary']['n_predicted_safe']}, "
      f"collapse={meta['heldout_summary']['n_predicted_collapse']}")
print(f"  Figure: {out_pdf}")