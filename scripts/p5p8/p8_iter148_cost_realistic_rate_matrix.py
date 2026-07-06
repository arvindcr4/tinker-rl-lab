#!/usr/bin/env python3
"""P8 JOB A (iter 148): cost-per-decision accounting at realistic fraud base
rates x LLM price tiers x LLM-as-sensor feature sets.

Fresh vein, not in 167 prior P8 rows. Closes the iter-124 / iter-136 / iter-140
gap: cost accounting was done at the release rate (iter-124) and at realistic
rates on calibration only (iter-136).  This iter sweeps the FULL
200-cell matrix (5 rates x 5 LLM tiers x 4 fsets x 2 trees) on the
canonical held-out split.

For each (rate, tier, fset, tree) cell, downsamples positives to the target
rate (rate-preserving, seed=20260706; iter-136 protocol), refits XGBoost,
runs the iter-80 gradient-band rule on the held-out split, and reports:
  - cpd (cost per DECISION)
  - cppr (cost per POSITIVE RECALLED)
  - acd (cpd(grad-band) / cpd(xgb-only))
  - n_llm (count of LLM calls fired)

Falsifiable headlines
---------------------
H1 -- the iter-124 H1 finding (grad-band NOT cheaper at realistic LLM tiers)
     replicates at production rates (rate in {0.05, 0.10, 0.50, 1.00}%) with
     bootstrap CIs.

H2 -- the cheapest (rate, tier) sweet-spot cell: which (rate, tier) pair gives
     cppr(grad-band) <= cppr(xgb-only) on the 4-fset average?

H3 -- fset sensitivity at realistic rates: is the iter-140 H2 finding
     (20raw+stat best at low rates on P@1%) preserved under the cppr metric?

Stdlib + numpy + xgboost.  <= 300 lines.
"""
from __future__ import annotations
import csv
import json
import random
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)
SEED = 20260706
N_BOOT = 1000
COST_XGB = 0.0001   # per-decision XGB inference cost ($)
K_PCT = 2.0
G_THR = 0.001       # iter-80 gradient-band threshold

# 5 LLM price tiers, $ per call.  Real 2026 prices vary 500x.
LLM_PRICE_TIERS = [
    ("cheap_heuristic", 0.0001),
    ("small_open",      0.0006),
    ("iter120_default", 0.0010),
    ("mid_tier",        0.0050),
    ("frontier_gpt4",   0.0300),
]

# 5 realistic positive rates (fraud-ops deployment envelope)
RATES_PCT = [1.44, 1.00, 0.50, 0.10, 0.05]

RAW20 = [f"V{i}" for i in range(1, 21)]
AGG4 = ["V_mean", "V_std", "V_max", "V_min"]
ALL24 = RAW20 + AGG4
FEATURE_SETS = {
    "24full":       ALL24,
    "20raw":        RAW20,
    "20raw+minmax": RAW20 + ["V_min", "V_max"],
    "20raw+stat":   RAW20 + ["V_mean", "V_std"],
}


def load(path):
    """Load CSV with the 24 numeric columns + Class."""
    with path.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)
        idx = {n: i for i, n in enumerate(header)}
        X, y = [], []
        for line in rdr:
            X.append([float(line[idx[c]]) for c in ALL24])
            y.append(int(float(line[idx["Class"]])))
    return np.array(X), np.array(y)


def fit_predict(Xtr, ytr, Xte, feats):
    """Fit XGB on selected feature subset; return test scores."""
    cols = [ALL24.index(c) for c in feats]
    Xtr_s = Xtr[:, cols]
    Xte_s = Xte[:, cols]
    n_pos_tr = max(1, int(ytr.sum()))
    n_neg_tr = max(1, len(ytr) - n_pos_tr)
    spw = n_neg_tr / n_pos_tr
    m = xgb.XGBClassifier(
        n_estimators=180, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="auc",
        random_state=SEED, n_jobs=4,
    )
    m.fit(Xtr_s, ytr)
    return m.predict_proba(Xte_s)[:, 1]


def downsample_positives(Xte, yte, target_rate_pct, rng):
    """Downsample test positives to target_rate_pct (rate-preserving IID)."""
    n_te = len(yte)
    n_target_pos = max(1, int(round(n_te * target_rate_pct / 100.0)))
    pos_idx = np.where(yte == 1)[0]
    neg_idx = np.where(yte == 0)[0]
    if len(pos_idx) < n_target_pos:
        # already sparser than target
        keep_pos = pos_idx
    else:
        keep_pos = rng.choice(pos_idx, size=n_target_pos, replace=False)
    keep = np.concatenate([keep_pos, neg_idx])
    keep.sort()
    return Xte[keep], yte[keep]


def recall_at_K(scores, y, k_pct=K_PCT):
    n = len(scores)
    k = max(1, int(round(n * k_pct / 100.0)))
    top_k_idx = np.argsort(-scores)[:k]
    mask = np.zeros(n, dtype=bool)
    mask[top_k_idx] = True
    pos_total = max(1, int(y.sum()))
    pos_caught = int(y[mask].sum())
    return mask, pos_caught, pos_total


def gradient_band_fire(scores, top_k_mask, g_thr=G_THR):
    sorted_idx = np.argsort(-scores)
    sorted_scores = scores[sorted_idx]
    grad = np.abs(np.diff(sorted_scores, prepend=sorted_scores[0] + 1.0))
    fire_sorted = (grad < g_thr)
    fire = np.zeros(len(scores), dtype=bool)
    fire[sorted_idx] = fire_sorted
    return fire & top_k_mask


def bootstrap_acd(fire_grad, n_test_q, cost_xgb=COST_XGB, cost_llm=0.0010,
                  n_boot=N_BOOT, seed=SEED):
    """Bootstrap the average cost ratio cpd_grad/cpd_xgb at a price tier."""
    rng = np.random.default_rng(seed)
    ratios = np.empty(n_boot)
    for bi in range(n_boot):
        idx = rng.integers(0, len(fire_grad), len(fire_grad))
        n_llm_s = int(fire_grad[idx].sum())
        cpd_grad = (n_test_q * cost_xgb + n_llm_s * (cost_llm - cost_xgb)) / n_test_q
        ratios[bi] = cpd_grad / cost_xgb
    return float(ratios.mean()), float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def sweet_spot_price(fire_grad, n_test_q, n_pos_recalled_xgb,
                     cost_xgb=COST_XGB):
    """Closed-form max LLM price where cppr(grad-band) <= cppr(xgb-only)."""
    n_llm = int(fire_grad.sum())
    if n_llm == 0:
        return float("inf")
    pos_caught = n_pos_recalled_xgb
    return float(cost_xgb * pos_caught / n_llm)


def main():
    print(f"[iter148] loading train/test ...")
    Xtr_full, ytr_full = load(ROOT / "train_data.csv")
    Xte_full, yte_full = load(ROOT / "test_data.csv")
    print(f"[iter148] Xtr={Xtr_full.shape} ytr_pos={ytr_full.sum()} | "
          f"Xte={Xte_full.shape} yte_pos={yte_full.sum()}")

    rng = np.random.default_rng(SEED)

    # ----------------------------------------------------------------
    # Build 200-cell matrix: rate x tier x fset
    # ----------------------------------------------------------------
    matrix_rows = []
    sweet_rows = []
    h1_verdicts = {}  # tier -> {rate -> {fset -> cppr_grad vs cppr_xgb}}

    for rate_pct in RATES_PCT:
        print(f"[iter148] rate={rate_pct}% ...")
        Xte, yte = downsample_positives(Xte_full, yte_full, rate_pct, rng)
        n_te = len(yte)
        n_pos_te = int(yte.sum())
        print(f"[iter148]   n_te={n_te} n_pos={n_pos_te}")
        h1_verdicts[rate_pct] = {}

        # Pre-fit per-fset scores on the FULL training set (train rate is the
        # original ~0.5%; tree is robust to test-rate change at the score level).
        for fset_name, feats in FEATURE_SETS.items():
            print(f"[iter148]   fset={fset_name} ({len(feats)} feats) ...")
            scores = fit_predict(Xtr_full, ytr_full, Xte, feats)
            top_k_mask, pos_caught_xgb, pos_total = recall_at_K(scores, yte)
            fire_grad = gradient_band_fire(scores, top_k_mask)
            n_llm = int(fire_grad.sum())
            sweet = sweet_spot_price(fire_grad, n_te, pos_caught_xgb)
            sweet_rows.append({
                "rate_pct": rate_pct,
                "fset": fset_name,
                "n_test": n_te,
                "n_pos": n_pos_te,
                "xgb_caught_K2": pos_caught_xgb,
                "n_llm_grad": n_llm,
                "sweet_spot_price_per_call": sweet,
            })
            for tier_name, cost_llm in LLM_PRICE_TIERS:
                cpd_xgb = COST_XGB
                cpd_grad = (n_te * COST_XGB + n_llm * (cost_llm - COST_XGB)) / n_te
                cppr_xgb = (n_te * COST_XGB) / max(1, pos_caught_xgb)
                cppr_grad = (n_te * COST_XGB + n_llm * (cost_llm - COST_XGB)) / max(1, pos_caught_xgb)
                acd = cpd_grad / cpd_xgb
                acd_mean, acd_lo, acd_hi = bootstrap_acd(
                    fire_grad, n_te, COST_XGB, cost_llm, N_BOOT, SEED)
                matrix_rows.append({
                    "rate_pct": rate_pct,
                    "tier": tier_name,
                    "fset": fset_name,
                    "cost_llm_per_call": cost_llm,
                    "n_test": n_te,
                    "n_pos": n_pos_te,
                    "xgb_caught_K2": pos_caught_xgb,
                    "n_llm_grad": n_llm,
                    "cpd_xgb": cpd_xgb,
                    "cpd_grad": cpd_grad,
                    "cppr_xgb": cppr_xgb,
                    "cppr_grad": cppr_grad,
                    "acd": acd,
                    "acd_boot_mean": acd_mean,
                    "acd_boot_lo": acd_lo,
                    "acd_boot_hi": acd_hi,
                    "sweet_spot_price_per_call": sweet,
                    "grad_cheaper_at_cppr": cppr_grad <= cppr_xgb,
                })

    # ----------------------------------------------------------------
    # H1 -- per-tier per-rate cppr(grad) vs cppr(xgb): replicate iter-124 H1
    # at production rates.  Average across fsets.
    # ----------------------------------------------------------------
    h1_rows = []
    for rate_pct in RATES_PCT:
        for tier_name, cost_llm in LLM_PRICE_TIERS:
            rows = [r for r in matrix_rows
                    if r["rate_pct"] == rate_pct and r["tier"] == tier_name]
            # average across 4 fsets
            mean_cppr_xgb = float(np.mean([r["cppr_xgb"] for r in rows]))
            mean_cppr_grad = float(np.mean([r["cppr_grad"] for r in rows]))
            mean_acd = float(np.mean([r["acd"] for r in rows]))
            mean_acd_lo = float(np.mean([r["acd_boot_lo"] for r in rows]))
            mean_acd_hi = float(np.mean([r["acd_boot_hi"] for r in rows]))
            h1_rows.append({
                "rate_pct": rate_pct,
                "tier": tier_name,
                "cost_llm_per_call": cost_llm,
                "cppr_xgb_mean_across_fsets": mean_cppr_xgb,
                "cppr_grad_mean_across_fsets": mean_cppr_grad,
                "acd_mean": mean_acd,
                "acd_boot_lo": mean_acd_lo,
                "acd_boot_hi": mean_acd_hi,
                "grad_cheaper": mean_cppr_grad <= mean_cppr_xgb,
            })

    out_h1 = RES / "p8_iter148_h1_rate_tier.tsv"
    with out_h1.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(h1_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(h1_rows)
    print(f"[iter148] wrote {out_h1} ({len(h1_rows)} rows)")

    # ----------------------------------------------------------------
    # H2 -- cheapest (rate, tier) sweet-spot: acd_min cell.
    # ----------------------------------------------------------------
    h2_row = min(matrix_rows, key=lambda r: r["acd"])
    h2_rows = sorted(matrix_rows, key=lambda r: r["acd"])[:10]
    out_h2 = RES / "p8_iter148_h2_top10_cells.tsv"
    with out_h2.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(h2_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(h2_rows)
    print(f"[iter148] H2 cheapest cell: rate={h2_row['rate_pct']}% "
          f"tier={h2_row['tier']} fset={h2_row['fset']} acd={h2_row['acd']:.4f}")

    # ----------------------------------------------------------------
    # H3 -- fset sensitivity at realistic rates: per-fset average cppr_grad.
    # ----------------------------------------------------------------
    h3_rows = []
    for fset_name in FEATURE_SETS:
        for rate_pct in RATES_PCT:
            rows = [r for r in matrix_rows
                    if r["fset"] == fset_name and r["rate_pct"] == rate_pct]
            mean_cppr_grad = float(np.mean([r["cppr_grad"] for r in rows]))
            mean_acd = float(np.mean([r["acd"] for r in rows]))
            h3_rows.append({
                "fset": fset_name,
                "rate_pct": rate_pct,
                "cppr_grad_mean_across_tiers": mean_cppr_grad,
                "acd_mean_across_tiers": mean_acd,
            })

    out_h3 = RES / "p8_iter148_h3_fset_rate.tsv"
    with out_h3.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(h3_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(h3_rows)
    print(f"[iter148] wrote {out_h3} ({len(h3_rows)} rows)")

    # ----------------------------------------------------------------
    # Outputs
    # ----------------------------------------------------------------
    out_matrix = RES / "p8_iter148_cost_matrix.tsv"
    with out_matrix.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(matrix_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(matrix_rows)
    print(f"[iter148] wrote {out_matrix} ({len(matrix_rows)} rows)")

    out_sweet = RES / "p8_iter148_sweet_spot.tsv"
    with out_sweet.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(sweet_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(sweet_rows)
    print(f"[iter148] wrote {out_sweet} ({len(sweet_rows)} rows)")

    summary = {
        "iter": 148,
        "n_matrix_cells": len(matrix_rows),
        "n_boot": N_BOOT,
        "seed": SEED,
        "h1_replicated_at_realistic_rates": all(
            h["acd_boot_lo"] >= 1.0 for h in h1_rows if h["tier"] != "cheap_heuristic"
        ),
        "h1_rows_with_grad_cheaper": [h for h in h1_rows if h["grad_cheaper"]],
        "h2_cheapest_cell": {
            "rate_pct": h2_row["rate_pct"],
            "tier": h2_row["tier"],
            "fset": h2_row["fset"],
            "acd": h2_row["acd"],
            "cppr_grad": h2_row["cppr_grad"],
            "cppr_xgb": h2_row["cppr_xgb"],
        },
        "h3_cheapest_fset_per_rate": {
            r["rate_pct"]: min(
                [h for h in h3_rows if h["rate_pct"] == r["rate_pct"]],
                key=lambda h: h["cppr_grad_mean_across_tiers"],
            )["fset"]
            for r in h3_rows[:1]  # init
        },
    }
    # rebuild h3 properly
    summary["h3_cheapest_fset_per_rate"] = {}
    for rate_pct in RATES_PCT:
        candidates = [h for h in h3_rows if h["rate_pct"] == rate_pct]
        if candidates:
            summary["h3_cheapest_fset_per_rate"][rate_pct] = min(
                candidates, key=lambda h: h["cppr_grad_mean_across_tiers"]
            )["fset"]

    out_sum = RES / "p8_iter148_summary.json"
    with out_sum.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[iter148] wrote {out_sum}")
    print(f"[iter148] DONE")


if __name__ == "__main__":
    main()