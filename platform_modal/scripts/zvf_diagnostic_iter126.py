"""Iter 126: ZVF Trajectory Dynamics as Early-Warning Signal (EWS) for GRPO collapse.

Angles tested (all three falsifiable):
  H1 - Critical slowing down:  GRPO seeds that eventually collapse have higher
        lag-1 autocorrelation in their pre-event ZVF trajectory than GRPO seeds
        that do not collapse.  (window = first 60 steps of each seed).
  H2 - Threshold-crossing lead time:  For ZVF > tau thresholds {0.4, 0.5, 0.6,
        0.7}, the empirical lead-time (steps from first crossing to collapse
        onset) is non-negative and tracks the structural failure signature.
  H3 - Method-level ZVF-drift ceiling:  ZVF growth rate (slope of mean ZVF over
        the first half of training) separates collapse-prone methods from safe
        methods.

Inputs:
  platform_hybrid/experiments/results/variance_mitigation.tsv   (9 methods x 5 seeds x ~100 steps)
Outputs:
  platform_hybrid/experiments/results/zvf_iter126_lag1.tsv
  platform_hybrid/experiments/results/zvf_iter126_leadtime.tsv
  platform_hybrid/experiments/results/zvf_iter126_drift.tsv
  platform_hybrid/experiments/results/zvf_iter126_meta.json
  figures/zvf_iter126_ews.{pdf,png}
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiments" / "results" / "variance_mitigation.tsv"
OUT = ROOT / "experiments" / "results"
FIG = ROOT / "figures"


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def load_trajectories() -> dict[tuple[str, str], list[tuple[int, float, float, float, int]]]:
    """Load (method, seed) -> sorted list of (step, zvf, reward, heldout, collapse)."""
    raw: dict[tuple[str, str], dict[int, tuple[float, float, float, int]]] = defaultdict(dict)
    with open(SRC) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            key = (r["method"], r["seed"])
            step = int(r["step"])
            raw[key][step] = (
                float(r["zvf"]),
                float(r["reward_mean"]),
                float(r["heldout_acc"]),
                int(r["collapse"]),
            )
    out = {}
    for k, m in raw.items():
        steps = sorted(m)
        out[k] = [(s, *m[s]) for s in steps]
    return out


def collapse_onset(tr: list[tuple[int, float, float, float, int]]) -> int | None:
    """Return first step where collapse=1, or None if no collapse in trace."""
    for s, _z, _r, _h, c in tr:
        if c == 1:
            return s
    return None


def lag1(x: np.ndarray) -> float:
    """Lag-1 Pearson autocorrelation. Returns NaN if too short."""
    if len(x) < 5:
        return float("nan")
    a = x[:-1]
    b = x[1:]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rolling_lag1(x: np.ndarray, win: int = 15) -> float:
    """Mean lag-1 autocorrelation across all rolling windows of length `win`."""
    if len(x) < win + 1:
        return float("nan")
    vals = []
    for i in range(0, len(x) - win):
        vals.append(lag1(x[i : i + win + 1]))
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def first_crossing(x: np.ndarray, tau: float) -> int | None:
    """Index of first element >= tau, else None."""
    for i, v in enumerate(x):
        if v >= tau:
            return i
    return None


def write_tsv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


# --------------------------------------------------------------------------- #
# H1 — Critical Slowing Down (CSD) on GRPO seeds
# --------------------------------------------------------------------------- #
def h1_lag1(trs: dict) -> tuple[list[list], dict]:
    """Per-seed lag-1 autocorrelation over the pre-event window."""
    rows = []
    pre_grp_collapse = []
    pre_grp_safe = []
    for (method, seed), tr in trs.items():
        onset = collapse_onset(tr)
        # pre-event window: steps 0..(onset-1) for collapse seeds;
        # full trajectory for non-collapse seeds (cap at 100)
        if onset is not None:
            pre = [t for t in tr if t[0] < onset]
            label = "collapse"
        else:
            pre = tr[: min(len(tr), 100)]
            label = "no_collapse"
        if len(pre) < 6:
            continue
        z = np.array([t[1] for t in pre])
        r = np.array([t[2] for t in pre])
        rho_z_lag1 = lag1(z)
        rho_z_roll = rolling_lag1(z, win=15)
        rho_r_lag1 = lag1(r)
        rows.append([
            method,
            seed,
            label,
            onset if onset is not None else "NA",
            len(pre),
            f"{rho_z_lag1:.4f}" if not math.isnan(rho_z_lag1) else "NA",
            f"{rho_z_roll:.4f}" if not math.isnan(rho_z_roll) else "NA",
            f"{rho_r_lag1:.4f}" if not math.isnan(rho_r_lag1) else "NA",
            f"{z.mean():.4f}",
            f"{z.std():.4f}",
        ])
        if method == "grpo":
            (pre_grp_collapse if label == "collapse" else pre_grp_safe).append(
                (rho_z_lag1, rho_z_roll, rho_r_lag1, z.mean(), z.std())
            )
    meta = {}
    if pre_grp_collapse and pre_grp_safe:
        meta["grpo_collapse_n"] = len(pre_grp_collapse)
        meta["grpo_safe_n"] = len(pre_grp_safe)
        meta["grpo_collapse_lag1_mean"] = float(np.nanmean([v[0] for v in pre_grp_collapse]))
        meta["grpo_safe_lag1_mean"] = float(np.nanmean([v[0] for v in pre_grp_safe]))
        meta["grpo_collapse_roll_mean"] = float(np.nanmean([v[1] for v in pre_grp_collapse]))
        meta["grpo_safe_roll_mean"] = float(np.nanmean([v[1] for v in pre_grp_safe]))
        meta["grpo_collapse_zmean"] = float(np.nanmean([v[3] for v in pre_grp_collapse]))
        meta["grpo_safe_zmean"] = float(np.nanmean([v[3] for v in pre_grp_safe]))
        # effect-size on lag-1 (pooled SD across both groups)
        c_arr = np.array([v[0] for v in pre_grp_collapse])
        s_arr = np.array([v[0] for v in pre_grp_safe])
        if c_arr.std() + s_arr.std() > 0:
            pooled = math.sqrt(
                (c_arr.var(ddof=1 if len(c_arr) > 1 else 0) + s_arr.var(ddof=1 if len(s_arr) > 1 else 0)) / 2
            )
            meta["grpo_lag1_cohen_d"] = float((c_arr.mean() - s_arr.mean()) / pooled) if pooled > 0 else float("nan")
        else:
            meta["grpo_lag1_cohen_d"] = float("nan")
    return rows, meta


# --------------------------------------------------------------------------- #
# H2 — Threshold-crossing lead time
# --------------------------------------------------------------------------- #
def h2_leadtime(trs: dict, taus=(0.4, 0.5, 0.6, 0.7)) -> tuple[list[list], dict]:
    """For each (method, seed, tau), record the first crossing step and lead time."""
    rows = []
    grp_crossings: dict[float, list[int]] = {tau: [] for tau in taus}
    grp_crossings_by_method: dict[str, dict[float, list[int]]] = defaultdict(lambda: {tau: [] for tau in taus})
    for (method, seed), tr in trs.items():
        z = np.array([t[1] for t in tr])
        steps = np.array([t[0] for t in tr])
        onset = collapse_onset(tr)
        for tau in taus:
            idx = first_crossing(z, tau)
            if idx is None:
                rows.append([method, seed, tau, "never", "NA", "NA", f"{z.mean():.4f}"])
                continue
            cross_step = int(steps[idx])
            if onset is not None:
                lead = onset - cross_step
                rows.append(
                    [method, seed, tau, cross_step, onset, lead, f"{z.mean():.4f}"]
                )
                if method == "grpo" and onset > 0:
                    grp_crossings[tau].append(lead)
                    grp_crossings_by_method[method][tau].append(lead)
            else:
                rows.append([method, seed, tau, cross_step, "NA", "NA", f"{z.mean():.4f}"])
    meta = {"tau_leadtime_grpo": {}}
    for tau in taus:
        leads = grp_crossings[tau]
        meta["tau_leadtime_grpo"][f"tau_{tau}"] = {
            "n": len(leads),
            "mean": float(np.mean(leads)) if leads else None,
            "min": int(min(leads)) if leads else None,
            "max": int(max(leads)) if leads else None,
            "all_non_negative": bool(all(l >= 0 for l in leads)) if leads else None,
        }
    return rows, meta


# --------------------------------------------------------------------------- #
# H3 — ZVF growth-rate / drift ceiling
# --------------------------------------------------------------------------- #
def h3_drift(trs: dict) -> tuple[list[list], dict]:
    """Linear slope of mean ZVF over the first half of each (method, seed) trajectory."""
    rows = []
    slopes_by_method: dict[str, list[float]] = defaultdict(list)
    for (method, seed), tr in trs.items():
        if len(tr) < 10:
            continue
        z = np.array([t[1] for t in tr])
        steps = np.array([t[0] for t in tr], dtype=float)
        half = len(tr) // 2
        a, b = np.polyfit(steps[:half], z[:half], 1)
        z_first = z[:half].mean()
        z_last = z[half:].mean()
        slope_norm = a / max(z.max(), 1e-6)
        rows.append([method, seed, half, f"{a:.6f}", f"{b:.4f}", f"{z_first:.4f}",
                     f"{z_last:.4f}", f"{slope_norm:.4f}", z.max()])
        slopes_by_method[method].append(a)
    meta = {}
    for m, sl in slopes_by_method.items():
        meta[m] = {
            "n_seeds": len(sl),
            "mean_slope": float(np.mean(sl)),
            "max_slope": float(np.max(sl)),
            "min_slope": float(np.min(sl)),
            "slope_sd": float(np.std(sl)),
        }
    return rows, meta


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(trs: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    methods_to_plot = ["grpo", "aero", "cppo", "areal"]
    cmap = plt.get_cmap("tab10")
    for k, m in enumerate(methods_to_plot):
        color = cmap(k)
        for (method, seed), tr in trs.items():
            if method != m:
                continue
            z = [t[1] for t in tr]
            s = [t[0] for t in tr]
            ls = "-" if collapse_onset(tr) is not None else ":"
            alpha = 0.85 if collapse_onset(tr) is not None else 0.55
            axes[0, 0].plot(s, z, color=color, ls=ls, alpha=alpha, lw=1.0)
        # add mean line on top
        zs = []
        ss = []
        for (method, seed), tr in trs.items():
            if method == m:
                zs.append([t[1] for t in tr])
                ss.append([t[0] for t in tr])
        if zs:
            min_len = min(len(z) for z in zs)
            zs_arr = np.array([z[:min_len] for z in zs])
            ss_arr = np.array([s[:min_len] for s in ss])
            axes[0, 0].plot(ss_arr.mean(0), zs_arr.mean(0), color=color, lw=2.4,
                            label=f"{m} (mean)", zorder=5)
    axes[0, 0].axhline(0.5, color="grey", ls="--", lw=0.8, label="ZVF=0.5")
    axes[0, 0].axhline(0.7, color="grey", ls=":", lw=0.8, label="ZVF=0.7")
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].set_xlabel("Training step")
    axes[0, 0].set_ylabel("ZVF")
    axes[0, 0].set_title("(a) ZVF trajectory: collapse (solid) vs safe (dotted)")
    axes[0, 0].legend(fontsize=7, loc="upper left")

    # panel b: lag-1 autocorrelation bar
    collapse_lags = []
    safe_lags = []
    for (method, seed), tr in trs.items():
        if method != "grpo":
            continue
        onset = collapse_onset(tr)
        pre = [t for t in tr if (onset is None or t[0] < onset)][:100]
        z = np.array([t[1] for t in pre])
        rho = lag1(z)
        if math.isnan(rho):
            continue
        if onset is not None:
            collapse_lags.append(rho)
        else:
            safe_lags.append(rho)
    means = [np.mean(collapse_lags) if collapse_lags else 0,
             np.mean(safe_lags) if safe_lags else 0]
    ses = [np.std(collapse_lags, ddof=1) / math.sqrt(len(collapse_lags)) if len(collapse_lags) > 1 else 0,
           np.std(safe_lags, ddof=1) / math.sqrt(len(safe_lags)) if len(safe_lags) > 1 else 0]
    axes[0, 1].bar(["GRPO\ncollapse\n(n=3)", "GRPO\nno collapse\n(n=2)"], means, yerr=ses,
                   color=["#d62728", "#2ca02c"], alpha=0.75, capsize=4)
    axes[0, 1].set_ylabel("lag-1 autocorrelation (pre-event window)")
    axes[0, 1].set_title("(b) H1: Critical Slowing Down on GRPO")
    axes[0, 1].set_ylim(-0.1, 1.0)

    # panel c: lead time per threshold for GRPO collapse seeds
    taus = [0.4, 0.5, 0.6, 0.7]
    leads_by_tau: dict[float, list[int]] = {tau: [] for tau in taus}
    for (method, seed), tr in trs.items():
        if method != "grpo":
            continue
        onset = collapse_onset(tr)
        if onset is None:
            continue
        z = np.array([t[1] for t in tr])
        steps = np.array([t[0] for t in tr])
        for tau in taus:
            idx = first_crossing(z, tau)
            if idx is not None:
                leads_by_tau[tau].append(int(onset - steps[idx]))
    width = 0.18
    xpos = np.arange(len(taus))
    for j, seed in enumerate([0, 1, 2]):
        seed_leads = []
        for tau in taus:
            if j < len(leads_by_tau[tau]):
                seed_leads.append(leads_by_tau[tau][j])
            else:
                seed_leads.append(0)
        axes[1, 0].bar(xpos + (j - 1) * width, seed_leads, width,
                       label=f"seed={seed}")
    axes[1, 0].set_xticks(xpos)
    axes[1, 0].set_xticklabels([f"tau={t}" for t in taus])
    axes[1, 0].set_ylabel("lead time (steps)")
    axes[1, 0].set_title("(c) H2: Threshold-crossing lead time (GRPO collapse)")
    axes[1, 0].axhline(0, color="black", lw=0.5)
    axes[1, 0].legend(fontsize=8)

    # panel d: ZVF drift slopes by method
    slopes_by_method: dict[str, list[float]] = defaultdict(list)
    for (method, seed), tr in trs.items():
        if len(tr) < 10:
            continue
        z = np.array([t[1] for t in tr])
        steps = np.array([t[0] for t in tr], dtype=float)
        half = len(tr) // 2
        a, _ = np.polyfit(steps[:half], z[:half], 1)
        slopes_by_method[method].append(a)
    methods = sorted(slopes_by_method.keys())
    means_d = [np.mean(slopes_by_method[m]) for m in methods]
    ses_d = [np.std(slopes_by_method[m], ddof=1) / math.sqrt(len(slopes_by_method[m]))
             if len(slopes_by_method[m]) > 1 else 0 for m in methods]
    colors_d = ["#d62728" if m == "grpo" else "#1f77b4" for m in methods]
    axes[1, 1].bar(methods, means_d, yerr=ses_d, color=colors_d, alpha=0.85, capsize=4)
    axes[1, 1].axhline(0, color="black", lw=0.5)
    axes[1, 1].set_ylabel("ZVF slope (first-half training)")
    axes[1, 1].set_title("(d) H3: ZVF drift ceiling by method")
    axes[1, 1].tick_params(axis="x", rotation=30)

    fig.suptitle("Iter 126: ZVF trajectory dynamics & EWS for GRPO collapse", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    trs = load_trajectories()
    print(f"Loaded {len(trs)} (method, seed) trajectories")

    rows_h1, meta_h1 = h1_lag1(trs)
    write_tsv(
        OUT / "zvf_iter126_lag1.tsv",
        ["method", "seed", "label", "collapse_onset_step", "pre_window_n",
         "lag1_zvf", "lag1_zvf_rolling_w15", "lag1_reward",
         "zvf_mean", "zvf_std"],
        rows_h1,
    )

    rows_h2, meta_h2 = h2_leadtime(trs)
    write_tsv(
        OUT / "zvf_iter126_leadtime.tsv",
        ["method", "seed", "tau", "first_crossing_step", "collapse_onset_step",
         "lead_time", "zvf_mean"],
        rows_h2,
    )

    rows_h3, meta_h3 = h3_drift(trs)
    write_tsv(
        OUT / "zvf_iter126_drift.tsv",
        ["method", "seed", "half_n", "slope", "intercept",
         "zvf_first_half_mean", "zvf_second_half_mean",
         "slope_normalised", "zvf_max"],
        rows_h3,
    )

    meta = {
        "iter": 126,
        "pillar": "P2-ZVF",
        "n_trajectories": len(trs),
        "h1_lag1": meta_h1,
        "h2_leadtime": meta_h2,
        "h3_drift": meta_h3,
    }
    with open(OUT / "zvf_iter126_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("H1 (lag-1, GRPO):", json.dumps(meta_h1, indent=2))
    print("H2 (lead-time):", json.dumps(meta_h2, indent=2))
    print("H3 (drift):", json.dumps(meta_h3, indent=2))

    make_figure(trs, FIG / "zvf_iter126_ews.pdf")


if __name__ == "__main__":
    main()