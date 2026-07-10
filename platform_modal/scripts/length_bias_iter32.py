#!/usr/bin/env python3
"""
length_bias_iter32.py — Temporal lag cross-correlation analysis of L vs R.

GOAL.  Earlier iters (8, 12, 16, 20, 24, 28) decomposed the length trajectory
along (A) aggregate trend, (B) within-step ZVF coupling, (C) within-step
reward coupling.  iter28 concluded that the within-step rho(L,R) is consistently
negative on every (task, algo) cell -- the verbosity-trap does NOT operate
at the within-step gradient level.

This iter asks the *time-axis* question that within-step coupling cannot
answer: does the length trajectory LEAD or LAG the reward trajectory?

Null hypothesis: lag-0 dominates (no feedback timing).
Verbosity-trap timing hypothesis: positive rho at positive k (length drops
  *before* reward rises -- L leads R).
Anti-trap / correction-channel hypothesis: positive rho at negative k
  (reward rises *before* length drops -- R leads L).

We compute the lag-k Spearman correlation between L(t) and R(t+k)
for k in [-3, 3].  Per (task, algo, seed) we report the lag with the
*peak* |rho| and the sign of the dominant lag.  Then the per-(task, algo)
summary tests which hypothesis wins across seeds.

Sister metric: length-stability CV_L = std(L)/mean(L) per run, as an
operational "Dr.GRPO effect on length variance" number.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, permutation_test

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "figures"

GSM_FILE = RESULTS / "drgrpo_gsm8k_cot_full.json"
ARITH_FILE = RESULTS / "drgrpo_vs_grpo.json"

LAGS = [-3, -2, -1, 0, 1, 2, 3]


def _load_step_log(path: Path):
    """Return list of (task, algo, seed, step, L, R, ZVF) tuples."""
    with open(path) as f:
        d = json.load(f)
    out = []
    for r in d.get("runs", []):
        algo = r.get("algo", "?")
        seed = r.get("seed", -1)
        exp = r.get("experiment", "?")
        if "gsm8k" in exp:
            task = "gsm8k_cot"
        elif "drgrpo_vs_grpo" in exp:
            task = "arithmetic_easy"
        else:
            task = exp
        for s in r.get("step_log", []):
            out.append((task, algo, seed, s["step"],
                        s["mean_comp_len"], s["mean_reward"], s["zvf"]))
    return out


def _lag_col(k: int) -> str:
    return f"rho_k{k:+d}"


def per_run_lag_profile(rows) -> pd.DataFrame:
    """One row per (task, algo, seed) with lag-k Spearman rho(L_t, R_{t+k})."""
    df = pd.DataFrame(rows, columns=["task", "algo", "seed", "step", "L", "R", "ZVF"])
    out = []
    for (task, algo, seed), g in df.groupby(["task", "algo", "seed"]):
        g = g.sort_values("step")
        L = g["L"].values
        R = g["R"].values
        n = len(L)
        if n < 8:
            continue
        row = {"task": task, "algo": algo, "seed": int(seed), "n_steps": n}
        # CV_L
        row["CV_L"] = float(np.std(L) / max(np.mean(L), 1e-9))
        row["mean_L"] = float(np.mean(L))
        row["mean_R"] = float(np.mean(R))
        # Lag correlations
        for k in LAGS:
            if k >= 0:
                x = L[: n - k]
                y = R[k:]
            else:
                kk = -k
                x = L[kk:]
                y = R[: n - kk]
            if len(x) < 5:
                row[_lag_col(k)] = np.nan
                row[f"p_k{k:+d}"] = np.nan
                continue
            r, p = spearmanr(x, y)
            row[_lag_col(k)] = float(r)
            row[f"p_k{k:+d}"] = float(p)
        # Find dominant lag (max |rho|)
        rs = {k: abs(row[_lag_col(k)]) for k in LAGS if not np.isnan(row[_lag_col(k)])}
        if not rs:
            continue
        dom_k = max(rs, key=rs.get)
        row["dominant_lag"] = int(dom_k)
        row["dominant_absrho"] = float(rs[dom_k])
        row["rho_at_dominant"] = float(row[_lag_col(dom_k)])
        out.append(row)
    return pd.DataFrame(out).sort_values(["task", "algo", "seed"]).reset_index(drop=True)


def lag_summary(profile: pd.DataFrame) -> pd.DataFrame:
    """Per (task, algo) cell: sign of dominant lag, fraction-of-seeds in each sign regime."""
    rows = []
    for (task, algo), g in profile.groupby(["task", "algo"]):
        rec = {"task": task, "algo": algo, "n_seeds": int(len(g))}
        # Distribution of dominant_lag
        for k in LAGS:
            rec[f"n_dom_k{k:+d}"] = int((g["dominant_lag"] == k).sum())
        # Aggregate sign of rho_at_dominant
        rec["mean_rho_at_dom"] = float(g["rho_at_dominant"].mean())
        rec["mean_dom_absrho"] = float(g["dominant_absrho"].mean())
        # Sign of rho at k=0
        rec["rho_k0_mean"] = float(g[_lag_col(0)].mean())
        # Lag-lead sign: fraction of seeds with positive lag (L leads) vs negative
        n_pos = int((g["dominant_lag"] > 0).sum())
        n_neg = int((g["dominant_lag"] < 0).sum())
        n_zero = int((g["dominant_lag"] == 0).sum())
        rec["n_dom_pos"] = n_pos
        rec["n_dom_neg"] = n_neg
        rec["n_dom_zero"] = n_zero
        # Fraction of seeds with positive rho at lag=+1 and lag=-1
        rec["rho_k+1_mean"] = float(g[_lag_col(1)].mean())
        rec["rho_k-1_mean"] = float(g[_lag_col(-1)].mean())
        # Paired bootstrap on (rho_k+1 - rho_k-1)
        if len(g) >= 3:
            diffs = (g[_lag_col(1)] - g[_lag_col(-1)]).dropna().values
            if len(diffs) >= 3:
                rng = np.random.default_rng(20260702)
                boots = []
                for _ in range(2000):
                    idx = rng.integers(0, len(diffs), len(diffs))
                    boots.append(np.mean(diffs[idx]))
                boots = np.array(boots)
                rec["rho_k1_minus_km1_mean"] = float(np.mean(diffs))
                lo, hi = np.percentile(boots, [2.5, 97.5])
                rec["rho_k1_minus_km1_lo"] = float(lo)
                rec["rho_k1_minus_km1_hi"] = float(hi)
                rec["rho_k1_minus_km1_p_le0"] = float(np.mean(boots <= 0))
            else:
                rec["rho_k1_minus_km1_mean"] = np.nan
                rec["rho_k1_minus_km1_lo"] = np.nan
                rec["rho_k1_minus_km1_hi"] = np.nan
                rec["rho_k1_minus_km1_p_le0"] = np.nan
        else:
            rec["rho_k1_minus_km1_mean"] = np.nan
            rec["rho_k1_minus_km1_lo"] = np.nan
            rec["rho_k1_minus_km1_hi"] = np.nan
            rec["rho_k1_minus_km1_p_le0"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["task", "algo"]).reset_index(drop=True)


def drift_alignment(profile: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap corr(mean_L_per_run, dominant_lag) and corr(mean_L, mean_R).

    Cross-seed prediction: cells with higher mean length at the start should
    have length more strongly lead reward change (positive lag).
    """
    rows = []
    for (task, algo), g in profile.groupby(["task", "algo"]):
        rec = {"task": task, "algo": algo, "n_seeds": int(len(g))}
        if len(g) < 3:
            rows.append(rec)
            continue
        # Pearson corr across seeds
        for lag in [-1, 0, 1]:
            x = g["dominant_lag"].astype(float).values
            y = g[_lag_col(lag)].astype(float).values
            if len(x) >= 3:
                r, p = pearsonr(x, y)
                rec[f"corr_dominantlag_vs_rho_k{lag:+d}"] = float(r)
                rec[f"p_corr_dominantlag_vs_rho_k{lag:+d}"] = float(p)
        # corr L-axis vs rho_k+1
        for tgt in [_lag_col(1), _lag_col(0), _lag_col(-1)]:
            x = g["mean_L"].values
            y = g[tgt].values
            r, p = pearsonr(x, y)
            rec[f"corr_meanL_vs_{tgt}"] = float(r)
            rec[f"p_corr_meanL_vs_{tgt}"] = float(p)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["task", "algo"]).reset_index(drop=True)


def main():
    rows = _load_step_log(GSM_FILE) + _load_step_log(ARITH_FILE)
    print(f"loaded {len(rows)} step records")
    profile = per_run_lag_profile(rows)
    print(f"per-run profile rows: {len(profile)}")
    summary = lag_summary(profile)
    drift = drift_alignment(profile)
    out_profile = RESULTS / "length_bias_iter32_lag_profile.tsv"
    out_summary = RESULTS / "length_bias_iter32_lag_summary.tsv"
    out_drift = RESULTS / "length_bias_iter32_drift_alignment.tsv"
    profile.to_csv(out_profile, sep="\t", index=False)
    summary.to_csv(out_summary, sep="\t", index=False)
    drift.to_csv(out_drift, sep="\t", index=False)
    print(f"wrote {out_profile}")
    print(f"wrote {out_summary}")
    print(f"wrote {out_drift}")
    print()
    print("=== summary ===")
    print(summary.to_string(index=False))
    print()
    print("=== drift ===")
    print(drift.to_string(index=False))
    print()
    print("=== dominant_lag distribution ===")
    print(profile.groupby(["task", "algo"])["dominant_lag"].agg(lambda x: dict(x.value_counts())).to_string())
    return profile, summary, drift


if __name__ == "__main__":
    main()
