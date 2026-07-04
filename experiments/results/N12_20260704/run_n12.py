#!/usr/bin/env python3
"""N12 — add a 4th length-coupling risk channel (rho(dZ,dL)) to the iter130 max-fusion index.

Inputs (archived, zero new training):
    inputs/zvf_iter130_risk_index.tsv       iter130 3-channel n=52 panel
    inputs/length_bias_iter136_step_coupling.tsv  real per-run rho(dZ,dL) from Dr.GRPO panels
    inputs/drgrpo_vs_grpo.json              real per-step length traces (arithmetic_easy)
    inputs/drgrpo_gsm8k_cot_full.json       real per-step length traces (gsm8k_cot)

Because the full n=52 panel does not ship per-step length traces as a single file,
the script:
  1. Computes the new rho(dZ,dL) channel from the archived real length traces.
  2. Applies method-level length-coupling priors to the n=52 iter130 panel
     (grpo gets its own mean, variance-mitigation methods get the Dr.GRPO proxy).
  3. Adds the channel to both max-fusion (logical-OR) and a 4-channel weighted composite.
  4. Re-runs AUROC + bootstrap CI and produces figures/TSV/JSON.

Outputs:
    n12_risk_index.tsv
    n12_axis_aurocs.tsv
    n12_method_risk.tsv
    n12_real_length_traces_rho.tsv
    n12_meta.json
    figures/*.png + figures/*.pdf
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
INP = ROOT / "inputs"
OUT = ROOT
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260704)

# ---------------------------------------------------------------------------
# Logistic axis transforms (same anchors as iter130, plus new length channel)
# ---------------------------------------------------------------------------
def axis_mag(mz: float) -> float:
    return float(1.0 / (1.0 + math.exp(-6.0 * (0.30 - mz))))


def axis_csd(roll_lag1: float) -> float:
    return float(1.0 / (1.0 + math.exp(-10.0 * (roll_lag1 - 0.45))))


def axis_drift(slope: float) -> float:
    return float(1.0 / (1.0 + math.exp(-300.0 * (slope - 0.004))))


def axis_len(rho: float, anchor: float = 0.15, k: float = 10.0) -> float:
    """Length-coupling risk: stronger |rho(dZ,dL)| -> higher risk."""
    if math.isnan(rho):
        return 0.0
    return float(1.0 / (1.0 + math.exp(-k * (abs(rho) - anchor))))


# ---------------------------------------------------------------------------
# AUROC + bootstrap CI
# ---------------------------------------------------------------------------
def auroc(y_true: np.ndarray, score: np.ndarray) -> float:
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


def auroc_block(axes_dict: dict, y_arr: np.ndarray) -> dict:
    out = {}
    for name, s in axes_dict.items():
        a = auroc(y_arr, s)
        lo, hi = bootstrap_ci(y_arr, s)
        out[name] = {"auroc": round(a, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
    return out


# ---------------------------------------------------------------------------
# 1. Load iter130 n=52 panel
# ---------------------------------------------------------------------------
risk = pd.read_csv(INP / "zvf_iter130_risk_index.tsv", sep="\t")
# Ensure seed is string for consistent merging
risk["seed"] = risk["seed"].astype(str)

# ---------------------------------------------------------------------------
# 2. Compute rho(dZ,dL) from real per-step length traces
# ---------------------------------------------------------------------------
real_rho_rows: list[dict[str, Any]] = []
for json_path in [INP / "drgrpo_vs_grpo.json", INP / "drgrpo_gsm8k_cot_full.json"]:
    with open(json_path) as f:
        data = json.load(f)
    for run in data.get("runs", []):
        step_log = run.get("step_log", [])
        if len(step_log) < 4:
            continue
        df = pd.DataFrame(step_log)
        zvf = df["zvf"].to_numpy()
        length = df["mean_comp_len"].to_numpy()
        d_z = np.diff(zvf)
        d_l = np.diff(length)
        rho, pval = stats.spearmanr(d_z, d_l)
        real_rho_rows.append({
            "task": run.get("experiment", ""),
            "method": run["algo"],
            "seed": str(run["seed"]),
            "model": run.get("model", ""),
            "n_steps": len(step_log),
            "rho_dZ_dL": float(rho),
            "rho_pval": float(pval),
            "abs_rho_dZ_dL": float(abs(rho)),
        })

real_rho = pd.DataFrame(real_rho_rows)
real_rho.to_csv(OUT / "n12_real_length_traces_rho.tsv", sep="\t", index=False)

# ---------------------------------------------------------------------------
# 3. Method-level length-coupling priors
# ---------------------------------------------------------------------------
# iter136 step-coupling file gives the same rho values; use it to set method-level priors
coup = pd.read_csv(INP / "length_bias_iter136_step_coupling.tsv", sep="\t")
grpo_abs = float(coup[coup["algo"] == "grpo"]["rho_dZ_dL"].abs().mean())
drgrpo_abs = float(coup[coup["algo"] == "dr_grpo"]["rho_dZ_dL"].abs().mean())

# Variance-mitigation methods are closer to a length-decoupled (Dr.GRPO-like) profile
mitigated_methods = {"aero", "cppo", "ngrpo", "scafgrpo", "mcgrpo", "gift", "areal", "es"}


def length_prior(method: str) -> float:
    if method == "grpo":
        return grpo_abs
    if method in mitigated_methods:
        return drgrpo_abs
    return float("nan")  # anchors have no length-coupling prior


risk["rho_dZ_dL"] = risk["method"].apply(length_prior)
risk["length_channel_source"] = risk["method"].apply(
    lambda m: "grpo_real" if m == "grpo" else ("mitigated_proxy" if m in mitigated_methods else "none")
)
risk["risk_len"] = risk["rho_dZ_dL"].apply(axis_len)

# ---------------------------------------------------------------------------
# 4. Build 4-channel indices
# ---------------------------------------------------------------------------
# Max-fusion: logical OR over all 4 channels (missing length contributes 0)
risk["zvf_risk_max_4ch"] = risk[["risk_mag", "risk_csd", "risk_drift", "risk_len"]].max(axis=1)

# Weighted 4-channel composite: keep iter130 emphasis, add modest length weight
W_M, W_C, W_D, W_L = 0.25, 0.45, 0.15, 0.15
risk["zvf_risk_4ch"] = (
    W_M * risk["risk_mag"]
    + W_C * risk["risk_csd"]
    + W_D * risk["risk_drift"]
    + W_L * risk["risk_len"]
)

# ---------------------------------------------------------------------------
# 5. AUROC comparison
# ---------------------------------------------------------------------------
y_within = risk[~risk["method"].str.startswith(("tool_use", "scaling_law"))]["failure_bin"].to_numpy()
y_cross = risk["failure_bin"].to_numpy()

# Within-methods panel (n=45 variance_mitigation seeds)
within_base = risk[~risk["method"].str.startswith(("tool_use", "scaling_law"))]
axes_within = {
    "magnitude": within_base["mean_zvf"].to_numpy(),
    "csd_roll_lag1": within_base["lag1_zvf_rolling_w15"].to_numpy(),
    "drift_slope": within_base["slope"].to_numpy(),
    "zvf_risk_max_3ch": within_base["zvf_risk_max"].to_numpy(),
    "risk_len": within_base["risk_len"].to_numpy(),
    "zvf_risk_max_4ch": within_base["zvf_risk_max_4ch"].to_numpy(),
    "zvf_risk_4ch": within_base["zvf_risk_4ch"].to_numpy(),
}
aurocs_within = auroc_block(axes_within, y_within)

# Cross-experiment panel (n=52)
axes_cross = {
    "magnitude": risk["mean_zvf"].to_numpy(),
    "csd_roll_lag1": risk["lag1_zvf_rolling_w15"].to_numpy(),
    "drift_slope": risk["slope"].to_numpy(),
    "zvf_risk_max_3ch": risk["zvf_risk_max"].to_numpy(),
    "risk_len": risk["risk_len"].to_numpy(),
    "zvf_risk_max_4ch": risk["zvf_risk_max_4ch"].to_numpy(),
    "zvf_risk_4ch": risk["zvf_risk_4ch"].to_numpy(),
}
aurocs_cross = auroc_block(axes_cross, y_cross)

axis_rows = []
for name, v in aurocs_cross.items():
    axis_rows.append({"scope": "cross_experiment", "axis": name,
                      "auroc": v["auroc"], "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"]})
for name, v in aurocs_within.items():
    axis_rows.append({"scope": "variance_mitigation_only", "axis": name,
                      "auroc": v["auroc"], "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"]})
pd.DataFrame(axis_rows).to_csv(OUT / "n12_axis_aurocs.tsv", sep="\t", index=False)

# ---------------------------------------------------------------------------
# 6. Per-method aggregate
# ---------------------------------------------------------------------------
agg = (risk.groupby("method", as_index=False)
       .agg(zvf_risk_mean=("zvf_risk", "mean"),
            zvf_risk_max_mean=("zvf_risk_max", "mean"),
            zvf_risk_4ch_mean=("zvf_risk_4ch", "mean"),
            zvf_risk_max_4ch_mean=("zvf_risk_max_4ch", "mean"),
            risk_len_mean=("risk_len", "mean"),
            rho_dZ_dL_mean=("rho_dZ_dL", "mean"),
            mag_mean=("mean_zvf", "mean"),
            csd_mean=("lag1_zvf_rolling_w15", "mean"),
            drift_mean=("slope", "mean"),
            failure_rate=("failure_bin", "mean"),
            n_seeds=("seed", "count"))
       .sort_values("zvf_risk_max_4ch_mean", ascending=False))
agg.to_csv(OUT / "n12_method_risk.tsv", sep="\t", index=False)

# ---------------------------------------------------------------------------
# 7. Outputs
# ---------------------------------------------------------------------------
keep_cols = [
    "method", "seed", "failure_label", "failure_bin",
    "mean_zvf", "lag1_zvf_rolling_w15", "slope",
    "risk_mag", "risk_csd", "risk_drift", "risk_len",
    "rho_dZ_dL", "length_channel_source",
    "zvf_risk", "zvf_risk_max", "zvf_risk_4ch", "zvf_risk_max_4ch",
]
risk_out = risk[keep_cols].sort_values("zvf_risk_max_4ch", ascending=False)
risk_out.to_csv(OUT / "n12_risk_index.tsv", sep="\t", index=False)

# ---------------------------------------------------------------------------
# 8. Figures
# ---------------------------------------------------------------------------
def savefig(name: str) -> None:
    plt.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close()

# --- Figure 1: AUROC comparison bar chart ---
fig, ax = plt.subplots(figsize=(10, 5))
comp = [
    ("3ch max-fusion", aurocs_cross["zvf_risk_max_3ch"]),
    ("4ch max-fusion", aurocs_cross["zvf_risk_max_4ch"]),
    ("length-only", aurocs_cross["risk_len"]),
    ("4ch weighted", aurocs_cross["zvf_risk_4ch"]),
]
names = [c[0] for c in comp]
vals = [c[1]["auroc"] for c in comp]
err_lo = [c[1]["auroc"] - c[1]["ci_lo"] for c in comp]
err_hi = [c[1]["ci_hi"] - c[1]["auroc"] for c in comp]
xs = np.arange(len(names))
bars = ax.bar(xs, vals, color=["#1f77b4", "#9467bd", "#2ca02c", "#d62728"], edgecolor="k")
ax.errorbar(xs, vals, yerr=[err_lo, err_hi], fmt="none", color="k", capsize=4)
ax.axhline(0.5, color="k", linestyle="--", alpha=0.4)
ax.set_xticks(xs)
ax.set_xticklabels(names, rotation=15, ha="right")
ax.set_ylabel("AUROC")
ax.set_ylim(0, 1.05)
ax.set_title("N12: cross-experiment AUROC — adding rho(dZ,dL) channel")
for bar, v, n in zip(bars, vals, names):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.3f}", ha="center", fontsize=9)
savefig("n12_auroc_comparison")

# --- Figure 2: scatter of existing 3ch max-fusion vs new 4ch max-fusion ---
fig, ax = plt.subplots(figsize=(7, 6))
colors = ["#d62728" if fb == 1 else "#1f77b4" for fb in risk["failure_bin"]]
ax.scatter(risk["zvf_risk_max"], risk["zvf_risk_max_4ch"], c=colors, s=80, edgecolor="k", alpha=0.8)
ax.plot([0, 1], [0, 1], "k:", alpha=0.4)
ax.set_xlabel("3-channel max-fusion risk")
ax.set_ylabel("4-channel max-fusion risk")
ax.set_title("N12: 3ch vs 4ch max-fusion (red = failure)")
from matplotlib.lines import Line2D
legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markeredgecolor="k", label="failure"),
          Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markeredgecolor="k", label="safe")]
ax.legend(handles=legend, loc="lower right")
savefig("n12_risk_scatter")

# --- Figure 3: ROC curves ---
fig, ax = plt.subplots(figsize=(7, 6))
y_x = risk["failure_bin"].to_numpy()
roc_axes = {
    "3ch max-fusion": risk["zvf_risk_max"].to_numpy(),
    "4ch max-fusion": risk["zvf_risk_max_4ch"].to_numpy(),
    "4ch weighted": risk["zvf_risk_4ch"].to_numpy(),
    "length-only": risk["risk_len"].to_numpy(),
}
palette = {"3ch max-fusion": "#1f77b4", "4ch max-fusion": "#9467bd",
           "4ch weighted": "#d62728", "length-only": "#2ca02c"}
for name, s in roc_axes.items():
    order = np.argsort(-s)
    fpr, tpr = [0.0], [0.0]
    for k in range(1, len(s) + 1):
        pred = np.zeros_like(y_x)
        pred[order[:k]] = 1
        tp = ((pred == 1) & (y_x == 1)).sum()
        fp = ((pred == 1) & (y_x == 0)).sum()
        tpr.append(tp / max((y_x == 1).sum(), 1))
        fpr.append(fp / max((y_x == 0).sum(), 1))
    auc = aurocs_cross["zvf_risk_max_3ch"]["auroc"] if name == "3ch max-fusion" else (
          aurocs_cross["zvf_risk_max_4ch"]["auroc"] if name == "4ch max-fusion" else (
          aurocs_cross["zvf_risk_4ch"]["auroc"] if name == "4ch weighted" else aurocs_cross["risk_len"]["auroc"]))
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=palette[name], linewidth=2)
ax.plot([0, 1], [0, 1], "k:", alpha=0.4)
ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate")
ax.set_title("N12: ROC curves on n=52 panel")
ax.legend(loc="lower right", fontsize=8)
savefig("n12_roc_curves")

# ---------------------------------------------------------------------------
# 9. Meta JSON
# ---------------------------------------------------------------------------
meta = {
    "experiment": "N12",
    "date": "2026-07-04",
    "n_rows_cross": int(len(risk)),
    "n_rows_within": int(len(within_base)),
    "n_failure_cross": int((risk["failure_bin"] == 1).sum()),
    "n_failure_within": int((within_base["failure_bin"] == 1).sum()),
    "weights_4channel": {"magnitude": W_M, "csd": W_C, "drift": W_D, "length_coupling": W_L},
    "length_channel_priors": {
        "grpo": round(grpo_abs, 4),
        "mitigated_methods_proxy": round(drgrpo_abs, 4),
        "source_file": "length_bias_iter136_step_coupling.tsv",
        "note": "Per-step length traces for the full n=52 panel are not archived as a single file; method-level priors were derived from the real Dr.GRPO length-trace panel (iter136).",
    },
    "aurocs_cross": aurocs_cross,
    "aurocs_within": aurocs_within,
    "per_method": agg.to_dict(orient="records"),
}
with open(OUT / "n12_meta.json", "w") as f:
    json.dump(meta, f, indent=2, default=str)

# ---------------------------------------------------------------------------
# 10. Print summary
# ---------------------------------------------------------------------------
print("=== N12: 4-channel ZVF Risk Index ===")
print(f"n=52 rows: {len(risk)}  (failures={meta['n_failure_cross']})")
print(f"n=45 within-method rows: {meta['n_rows_within']}  (failures={meta['n_failure_within']})")
print()
print("New 4-channel weights:", meta["weights_4channel"])
print()
print("=== Cross-experiment AUROCs ===")
for r in axis_rows:
    if r["scope"] == "cross_experiment":
        print(f"  {r['axis']:22s}  AUC={r['auroc']:.4f}  CI=[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
print()
print("=== Within-method AUROCs ===")
for r in axis_rows:
    if r["scope"] == "variance_mitigation_only":
        print(f"  {r['axis']:22s}  AUC={r['auroc']:.4f}  CI=[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
print()
print("=== Per-method (top 10 by 4ch max-fusion) ===")
print(agg.head(10).to_string(index=False))
print()
print("Outputs:")
for p in ["n12_risk_index.tsv", "n12_axis_aurocs.tsv", "n12_method_risk.tsv",
          "n12_real_length_traces_rho.tsv", "n12_meta.json"]:
    print(f"  {OUT / p}")
print(f"  {FIG}")
