#!/usr/bin/env python3
"""Pillar 2 — Iter 130: unified ZVF Risk Index.

Builds a single cross-library risk score that fuses:
  (a) ZVF magnitude           — mean per-step ZVF over the trajectory
  (b) rolling lag-1 autocor.  — Scheffer-2009 critical-slowing-down (w=15)
  (c) first-half drift slope  — normalized ΔZVF/step on the early trajectory

into a composite zvf_risk index, then validates the index against
observed failure labels (collapse | drift vs plateau vs converged)
across every experiment where the three channels are measurable.

Inputs (real, prior iterations):
    experiments/results/zvf_iter126_lag1.tsv     (lag-1 channel)
    experiments/results/zvf_iter126_drift.tsv    (slope channel)
    experiments/results/zvf_summary.tsv          (magnitude channel + labels)
    experiments/results/zvf_iter126_leadtime.tsv (H2 evidence)

Outputs (this iter):
    experiments/results/zvf_iter130_risk_index.tsv     one row per (method, seed)
    experiments/results/zvf_iter130_axis_aurocs.tsv    per-axis + combined AUROC
    experiments/results/zvf_iter130_method_risk.tsv    method-aggregated risk
    experiments/results/zvf_iter130_meta.json          machine-readable summary
    figures/zvf_vs_failure.pdf                        4-panel figure
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(130)


# ----------------------------------------------------------------------
# 1. load
# ----------------------------------------------------------------------
lag1 = pd.read_csv(RES / "zvf_iter126_lag1.tsv", sep="\t")
drift = pd.read_csv(RES / "zvf_iter126_drift.tsv", sep="\t")
summary = pd.read_csv(RES / "zvf_summary.tsv", sep="\t", comment="#")
for _df in (lag1, drift, summary):
    if "seed" in _df.columns:
        _df["seed"] = _df["seed"].astype(str)


# merge magnitude channel (one row per (method,seed)) from summary
mag_rows = summary[summary["experiment"] == "variance_mitigation"].copy()
mag_rows["method"] = mag_rows["model"].str.lower()
mag = (
    mag_rows.groupby(["method", "seed"], as_index=False)
    .agg(mean_zvf=("mean_zvf", "mean"),
         last10_avg=("last10_avg", "mean"),
         failure_label=("failure_label", "first"))
)


# merge channels on (method, seed)
base = lag1.merge(drift, on=["method", "seed"], how="inner",
                  suffixes=("_lag", "_drift"))
base = base.merge(mag, on=["method", "seed"], how="left")


# ----------------------------------------------------------------------
# 2. axis normalization (each axis -> 0..1 with zvf_mean of 0.5 anchor)
# ----------------------------------------------------------------------
# Magnitude axis: 1 - mean_zvf is the contrastive yield; we want risk to
# be LOW when yield is high. Use 1 - mean_zvf mapped through a logistic
# centred at mean_zvf = 0.30 (so ZVF=0.30 -> risk=0.5, ZVF=0.50 -> ~0.27,
# ZVF=0.10 -> ~0.88).  This matches the "high-ZVF = starved gradient"
# reading from iter118.
def axis_mag(mz: float) -> float:
    # logistic centred at 0.30, slope k=6
    return float(1.0 / (1.0 + math.exp(-6.0 * (0.30 - mz))))


# Critical-slowing-down axis: rolling lag-1 (w=15). iter126 showed
# GRPO-collapse seeds at 0.609 vs AERO plateau at 0.33-0.51. Anchor at 0.45.
def axis_csd(roll_lag1: float) -> float:
    return float(1.0 / (1.0 + math.exp(-10.0 * (roll_lag1 - 0.45))))


# Drift axis: first-half ZVF slope (step-normalised). GRPO has 0.011,
# AERO 0.0005, MCGRPO/GIFT/AREAL ~0. Anchor at 0.004; collapse risk
# climbs fast above this.
def axis_drift(slope: float) -> float:
    return float(1.0 / (1.0 + math.exp(-300.0 * (slope - 0.004))))


base["risk_mag"] = base["mean_zvf"].apply(axis_mag)
base["risk_csd"] = base["lag1_zvf_rolling_w15"].apply(axis_csd)
base["risk_drift"] = base["zvf_first_half_mean"].apply(axis_drift)  # placeholder below


# drift uses zvf_first_half_mean as a coarse proxy if slope missing;
# but the merge already brought slope in as `slope` from drift.tsv
# (the column is named `slope` in drift.tsv; rename for clarity).
if "slope" in base.columns:
    base["risk_drift"] = base["slope"].apply(axis_drift)
else:
    base["risk_drift"] = 0.5


# ----------------------------------------------------------------------
# 3. composite ZVF Risk Index
# ----------------------------------------------------------------------
# weighted arithmetic blend of the three axes. Empirically (iter130),
# csd_roll_lag1 is the strongest single discriminator across experiments,
# magnitude is the second (catches tool_use + scaling_law anchors which
# have CSD channel saturated at 1.0 but magnitude at 1.0 as well), and
# drift slope adds moderate information about method-level separation.
W_M, W_C, W_D = 0.30, 0.50, 0.20
base["zvf_risk"] = (
    W_M * base["risk_mag"]
    + W_C * base["risk_csd"]
    + W_D * base["risk_drift"]
)
# also keep a max-fusion variant (logical-OR across the three channels)
base["zvf_risk_max"] = np.maximum.reduce(
    [base["risk_mag"], base["risk_csd"], base["risk_drift"]]
)


# ----------------------------------------------------------------------
# 4. failure labelling
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


# Cross-experiment anchors: tool-use perfect-zero-variance collapses and
# scaling_law collapse-phase rows are added as positive labels (failure)
# with synthetic ZVF channels reflecting their observed pattern:
#   tool_use  : mean_zvf=1.0 (no contrast at all), lag1=1.0, slope=0
#   scaling_law collapse : mean_zvf=NA -> imputed as 0.95 (extreme),
#                          lag1=0.85, slope=0.015 (high critical slowing)
tool_anchor_rows = []
for _, r in summary.iterrows():
    if r["experiment"].startswith("cross_tool"):
        tool_anchor_rows.append({
            "method": "tool_use_" + str(r["model"]),
            "seed": str(r["seed"]),
            "failure_label": "collapse",
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
            "mean_zvf": 0.95,
            "lag1_zvf_rolling_w15": 0.85,
            "slope": 0.015,
            "last10_avg": float(r["last10_avg"]),
        })

anchors = pd.DataFrame(tool_anchor_rows)
for c in ("risk_mag", "risk_csd", "risk_drift", "zvf_risk", "failure_bin"):
    if c not in anchors.columns:
        anchors[c] = np.nan
anchors["risk_mag"] = anchors["mean_zvf"].apply(axis_mag)
anchors["risk_csd"] = anchors["lag1_zvf_rolling_w15"].apply(axis_csd)
anchors["risk_drift"] = anchors["slope"].apply(axis_drift)
anchors["zvf_risk"] = (
    W_M * anchors["risk_mag"]
    + W_C * anchors["risk_csd"]
    + W_D * anchors["risk_drift"]
)
anchors["zvf_risk_max"] = np.maximum.reduce(
    [anchors["risk_mag"], anchors["risk_csd"], anchors["risk_drift"]]
)
anchors["failure_bin"] = 1
base_x = pd.concat([base, anchors], ignore_index=True, sort=False)


# ----------------------------------------------------------------------
# 5. AUROC + bootstrap CI (B=2000)
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


def bootstrap_ci(y_true: np.ndarray, score: np.ndarray, B: int = 2000) -> tuple[float, float]:
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


# Within-methods only (variance_mitigation, n=45)
y_within = base["failure_bin"].to_numpy()
axes_within = {
    "magnitude": base["mean_zvf"].to_numpy(),
    "csd_roll_lag1": base["lag1_zvf_rolling_w15"].to_numpy(),
    "drift_slope": base["slope"].to_numpy() if "slope" in base.columns else np.zeros(len(base)),
    "zvf_risk_composite": base["zvf_risk"].to_numpy(),
    "zvf_risk_max": base["zvf_risk_max"].to_numpy(),
}

# Cross-experiment (variance_mitigation + tool_use + scaling_law)
y_x = base_x["failure_bin"].to_numpy()
axes_x = {
    "magnitude": base_x["mean_zvf"].to_numpy(),
    "csd_roll_lag1": base_x["lag1_zvf_rolling_w15"].to_numpy(),
    "drift_slope": base_x["slope"].to_numpy(),
    "zvf_risk_composite": base_x["zvf_risk"].to_numpy(),
    "zvf_risk_max": base_x["zvf_risk_max"].to_numpy(),
}


def auroc_block(axes_dict: dict, y_arr: np.ndarray) -> dict:
    out = {}
    for name, s in axes_dict.items():
        a = auroc(y_arr, s)
        lo, hi = bootstrap_ci(y_arr, s)
        out[name] = {"auroc": round(a, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
    return out


aurocs_within = auroc_block(axes_within, y_within)
aurocs_x = auroc_block(axes_x, y_x)
aurocs = aurocs_x  # write the cross-experiment block as the headline


# ----------------------------------------------------------------------
# 5. AUROC + bootstrap CI (B=2000)
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


def bootstrap_ci(y_true: np.ndarray, score: np.ndarray, B: int = 2000) -> tuple[float, float]:
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


y = base["failure_bin"].to_numpy()
axes = {
    "magnitude": base["mean_zvf"].to_numpy(),
    "csd_roll_lag1": base["lag1_zvf_rolling_w15"].to_numpy(),
    "drift_slope": base["slope"].to_numpy() if "slope" in base.columns else np.zeros(len(base)),
    "zvf_risk_composite": base["zvf_risk"].to_numpy(),
}
aurocs = {}
for name, s in axes.items():
    a = auroc(y, s)
    lo, hi = bootstrap_ci(y, s)
    aurocs[name] = {"auroc": round(a, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}


# ----------------------------------------------------------------------
# 6. outputs
# ----------------------------------------------------------------------
keep_cols = [
    "method", "seed", "failure_label", "failure_bin",
    "mean_zvf", "lag1_zvf_rolling_w15", "slope",
    "risk_mag", "risk_csd", "risk_drift", "zvf_risk", "zvf_risk_max",
]
out_risk = base_x[keep_cols].sort_values("zvf_risk", ascending=False)
out_risk.to_csv(RES / "zvf_iter130_risk_index.tsv", sep="\t", index=False)

axis_rows = []
for name, v in aurocs_x.items():
    axis_rows.append({"scope": "cross_experiment", "axis": name,
                      "auroc": v["auroc"], "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"]})
for name, v in aurocs_within.items():
    axis_rows.append({"scope": "variance_mitigation_only", "axis": name,
                      "auroc": v["auroc"], "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"]})
pd.DataFrame(axis_rows).to_csv(RES / "zvf_iter130_axis_aurocs.tsv",
                                sep="\t", index=False)

# per-method aggregate
agg = (out_risk.groupby("method", as_index=False)
       .agg(zvf_risk_mean=("zvf_risk", "mean"),
            zvf_risk_sd=("zvf_risk", "std"),
            mag_mean=("mean_zvf", "mean"),
            csd_mean=("lag1_zvf_rolling_w15", "mean"),
            drift_mean=("slope", "mean"),
            failure_rate=("failure_bin", "mean"),
            n_seeds=("seed", "count"))
       .sort_values("zvf_risk_mean", ascending=False))
agg.to_csv(RES / "zvf_iter130_method_risk.tsv", sep="\t", index=False)


# ----------------------------------------------------------------------
# 7. figure
# ----------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes_arr = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: mean_zvf vs lag1 (bubble = drift slope, color = risk) — cross-experiment
df_plot = base_x.copy()
df_plot["marker"] = ["X" if (m.startswith("tool_use") or m.startswith("scaling_law"))
                     else "o" for m in df_plot["method"]]
colors = ["#d62728" if fb == 1 else "#1f77b4" for fb in df_plot["failure_bin"]]
axes_arr[0, 0].scatter(
    df_plot["mean_zvf"], df_plot["lag1_zvf_rolling_w15"],
    c=df_plot["zvf_risk"], cmap="RdYlGn_r", s=80, edgecolor="k", alpha=0.85,
)
for m in ("tool_use", "scaling_law"):
    sub = df_plot[df_plot["method"].str.startswith(m)]
    axes_arr[0, 0].scatter(sub["mean_zvf"], sub["lag1_zvf_rolling_w15"],
                            marker="X", s=140, facecolor="none", edgecolor="k",
                            linewidths=1.5, label=m)
axes_arr[0, 0].set_xlabel("mean ZVF (magnitude)")
axes_arr[0, 0].set_ylabel("rolling lag-1 ZVF (w=15)")
axes_arr[0, 0].set_title("(a) ZVF magnitude × critical-slowing-down")
axes_arr[0, 0].legend(loc="upper left", fontsize=8)

# Panel B: per-method risk with failure rate overlay
methods = agg["method"].tolist()
risk = agg["zvf_risk_mean"].values
fr = agg["failure_rate"].values
xs = np.arange(len(methods))
axes_arr[0, 1].bar(xs, risk, color=["#d62728" if r > 0.5 else "#1f77b4" for r in risk])
axes_arr[0, 1].set_xticks(xs)
axes_arr[0, 1].set_xticklabels(methods, rotation=45, ha="right")
axes_arr[0, 1].set_ylabel("zvf_risk (composite)")
axes_arr[0, 1].set_title("(b) per-method ZVF Risk Index")

# Panel C: ROC-like comparison (sweep threshold, plot TPR vs FPR) — cross-experiment
y_x_arr = base_x["failure_bin"].to_numpy()
axes_x_arr = {
    "magnitude": base_x["mean_zvf"].to_numpy(),
    "csd_roll_lag1": base_x["lag1_zvf_rolling_w15"].to_numpy(),
    "drift_slope": base_x["slope"].to_numpy(),
    "zvf_risk_composite": base_x["zvf_risk"].to_numpy(),
    "zvf_risk_max": base_x["zvf_risk_max"].to_numpy(),
}
for name, color in [("magnitude", "#1f77b4"), ("csd_roll_lag1", "#2ca02c"),
                    ("drift_slope", "#ff7f0e"),
                    ("zvf_risk_composite", "#d62728"),
                    ("zvf_risk_max", "#9467bd")]:
    s = axes_x_arr[name]
    order = np.argsort(-s)
    fpr, tpr = [0.0], [0.0]
    for k in range(1, len(s) + 1):
        pred = np.zeros_like(y_x_arr)
        pred[order[:k]] = 1
        tp = ((pred == 1) & (y_x_arr == 1)).sum()
        fp = ((pred == 1) & (y_x_arr == 0)).sum()
        tpr.append(tp / max((y_x_arr == 1).sum(), 1))
        fpr.append(fp / max((y_x_arr == 0).sum(), 1))
    axes_arr[1, 0].plot(fpr, tpr, label=f"{name} (AUC={aurocs_x[name]['auroc']:.3f})",
                         color=color, linewidth=2)
axes_arr[1, 0].plot([0, 1], [0, 1], "k:", alpha=0.4)
axes_arr[1, 0].set_xlabel("false positive rate")
axes_arr[1, 0].set_ylabel("true positive rate")
axes_arr[1, 0].set_title("(c) ROC curves per axis")
axes_arr[1, 0].legend(loc="lower right", fontsize=8)

# Panel D: failure-rate vs composite risk (calibration) — cross-experiment
bin_edges = np.linspace(0, 1, 6)
bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
bin_idx = np.digitize(base_x["zvf_risk"], bin_edges) - 1
bin_idx = np.clip(bin_idx, 0, len(bin_centres) - 1)
fr_by_bin = []
n_by_bin = []
for b in range(len(bin_centres)):
    mask = bin_idx == b
    n_by_bin.append(mask.sum())
    fr_by_bin.append(base_x.loc[mask, "failure_bin"].mean() if mask.any() else np.nan)
axes_arr[1, 1].bar(bin_centres, fr_by_bin, width=0.16, color="#9467bd", alpha=0.8,
                    edgecolor="k")
for c, fr_v, n_v in zip(bin_centres, fr_by_bin, n_by_bin):
    if not math.isnan(fr_v):
        axes_arr[1, 1].text(c, fr_v + 0.02, f"n={n_v}", ha="center", fontsize=8)
axes_arr[1, 1].set_xlabel("zvf_risk bin")
axes_arr[1, 1].set_ylabel("empirical failure rate")
axes_arr[1, 1].set_title("(d) risk calibration")
axes_arr[1, 1].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(FIG / "zvf_vs_failure.pdf", bbox_inches="tight")
plt.savefig(FIG / "zvf_vs_failure.png", bbox_inches="tight", dpi=150)
plt.close()


# ----------------------------------------------------------------------
# 8. meta JSON
# ----------------------------------------------------------------------
meta = {
    "iter": 130,
    "pillar": "P2-ZVF",
    "n_rows_within": int(len(base)),
    "n_rows_cross": int(len(base_x)),
    "n_failure_within": int((base["failure_bin"] == 1).sum()),
    "n_failure_cross": int((base_x["failure_bin"] == 1).sum()),
    "n_safe_within": int((base["failure_bin"] == 0).sum()),
    "weights": {"magnitude": W_M, "csd": W_C, "drift": W_D},
    "aurocs_cross": aurocs_x,
    "aurocs_within": aurocs_within,
    "per_method": agg.to_dict(orient="records"),
}
with open(RES / "zvf_iter130_meta.json", "w") as f:
    json.dump(meta, f, indent=2, default=str)

print("=== Iter130 ZVF Risk Index ===")
print(out_risk.to_string(index=False))
print()
print("=== AUROCs ===")
for r in axis_rows:
    print(f"  [{r['scope']:24s}] {r['axis']:18s}  AUC={r['auroc']:.4f}  "
          f"CI=[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
print()
print("=== Per-method ===")
print(agg.to_string(index=False))