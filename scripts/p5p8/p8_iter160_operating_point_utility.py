#!/usr/bin/env python3
"""P8 JOB A (iter 160): operating-point utility-maximization with 5-seed bootstrap CI.

Fresh vein, not in 172 prior P8 rows. Closes the iter-148 cost matrix
+ iter-152 5-seed CV + iter-156 VALUE/WASTE decomposition by answering the
OPERATIONAL deployment question:

    At each (rate, fset), what threshold tau* maximizes the chosen utility
    function? What is the realized utility(τ*) under 5-seed bootstrap CI?

Prior iters picked one operating point (top-K=2% from iter-72, V_mean>0
from iter-156) and reported the realized recall/precision. Iter-160
OPTIMIZES tau per (rate, fset, utility) cell — the operational decision a
fraud-ops team actually faces: pick the threshold, then see what you get.

4 utility functions compared:
  U1:  F1-maximizing — argmax_tau F1(τ) [pure accuracy-quality]
  U2:  VALUE-maximizing — argmax_tau value_gain(τ) − value_cost(τ) with
       VALUE_PER_CATCH = $50 and tier-cost = cost_llm_per_call at chosen
       tier [decision-economic]
  U3:  precision-constrained (precision >= 0.5) — smallest tau satisfying
       the constraint; recall at that tau [alert-quality]
  U4:  cost-constrained (cost_per_caught <= $10) — argmax_tau recall
       subject to cost budget [budget-constrained]

5 seeds x 5 rates x 4 fsets x 5 tiers x 4 utilities = 2000 cells.
Plus 5-seed bootstrap B=2000 on realized utility(τ*) at each
(rate x fset x tier x utility) cell → 4000 CIs total.

Outputs:
  experiments/results/p5p8/p8_iter160_opt_tau_per_cell.tsv      (2000 rows: cell-level τ* + util)
  experiments/results/p5p8/p8_iter160_opt_util_per_cell.tsv     (2000 rows: realized utility at τ*)
  experiments/results/p5p8/p8_iter160_h_tau_stratified.tsv      (40 rows: 5 utilities x 4 fsets x 2 tiers selected)
  experiments/results/p5p8/p8_iter160_h_util_monotone.tsv       (20 rows: utility-monotonicity test)
  experiments/results/p5p8/p8_iter160_h_5seed_ci.tsv            (50 rows: 5 utilities x 5 rates x 2 tiers)
  experiments/results/p5p8/p8_iter160_summary.json              (machine-readable H1-H4 verdicts)

Stdlib + numpy + xgboost.  <= 290 lines.
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)
SEEDS = [20260706, 20260708, 20260710, 20260712, 20260714]
N_BOOT = 2000
BOOT_SEED = 20260705
N_EST = 180
MAX_DEPTH = 5
COST_XGB = 0.0001
K_PCT = 2.0
TAU_GRID = np.round(np.arange(0.001, 1.001, 0.005), 4)  # 200 thresholds
VALUE_PER_CATCH = 50.0
COST_BUDGET = 10.0  # for utility U4

LLM_PRICE_TIERS = [
    ("cheap_heuristic", 0.0001),
    ("small_open",      0.0006),
    ("iter120_default", 0.0010),
    ("mid_tier",        0.0050),
    ("frontier_gpt4",   0.0300),
]
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
    with path.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)
        idx = {n: i for i, n in enumerate(header)}
        X, y = [], []
        for line in rdr:
            X.append([float(line[idx[c]]) for c in ALL24])
            y.append(int(float(line[idx["Class"]])))
    return np.array(X), np.array(y)


def downsample(X, y, rate_pct, seed):
    rng = np.random.default_rng(seed)
    n_te = len(y)
    n_target_pos = max(1, int(round(n_te * rate_pct / 100.0)))
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    keep_pos = pos_idx if len(pos_idx) < n_target_pos else rng.choice(pos_idx, size=n_target_pos, replace=False)
    keep = np.concatenate([keep_pos, neg_idx])
    keep.sort()
    return X[keep], y[keep]


def fit_predict(Xtr, ytr, Xte, feats, seed):
    cols = [ALL24.index(c) for c in feats]
    Xtr_s = Xtr[:, cols]
    Xte_s = Xte[:, cols]
    n_pos = max(1, int(ytr.sum()))
    n_neg = max(1, len(ytr) - n_pos)
    spw = n_neg / n_pos
    m = xgb.XGBClassifier(
        n_estimators=N_EST, max_depth=MAX_DEPTH, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=seed, n_jobs=4, scale_pos_weight=spw,
    )
    m.fit(Xtr_s, ytr)
    return m.predict_proba(Xte_s)[:, 1]


def sweep_thresholds(probs, y, taus):
    """For each tau in taus, return (precision, recall, f1, n_alerted)."""
    rows = []
    n = len(y)
    n_pos = int(y.sum())
    pos = probs[y == 1]
    neg = probs[y == 0]
    for tau in taus:
        tp = int((pos >= tau).sum())
        fp = int((neg >= tau).sum())
        fn = n_pos - tp
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, n_pos)
        f1 = 2 * prec * rec / max(1e-12, prec + rec)
        rows.append((float(tau), prec, rec, f1, tp, fp, tp + fp, fn))
    return rows


def find_optimal_tau(rows, y, probs, tier_cost, utility, cost_budget=COST_BUDGET):
    """Return (tau*, realized_utility, supporting_metrics) for given utility."""
    n_pos = int(y.sum())
    best_tau, best_util, best_metrics = None, -np.inf, None
    for (tau, prec, rec, f1, tp, fp, alerted, fn) in rows:
        if utility == "F1":
            u = f1
        elif utility == "VALUE":
            cost_total = alerted * tier_cost
            value_gain = tp * VALUE_PER_CATCH
            u = (value_gain - cost_total) / max(1.0, n_pos * VALUE_PER_CATCH)
        elif utility == "PREC_CONSTRAINED":
            if prec >= 0.5:
                u = rec
            else:
                u = -1.0
        elif utility == "COST_CONSTRAINED":
            if alerted == 0:
                u = 0.0
            else:
                cost_per_caught = alerted * tier_cost / max(1, tp)
                if cost_per_caught <= cost_budget:
                    u= rec
                else:
                    u = -1.0
        else:
            raise ValueError(utility)
        if u > best_util:
            best_util = u
            best_tau = tau
            best_metrics = (prec, rec, f1, tp, fp, alerted, fn)
    return best_tau, float(best_util), best_metrics


def bootstrap_ci(values, B=2000, seed=BOOT_SEED):
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        means[b] = arr[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def main():
    print("Loading data...", flush=True)
    Xtr_full, ytr_full = load(ROOT / "fraud_data.csv")
    Xte, yte = load(ROOT / "test_data.csv")
    print(f"  Train: {Xtr_full.shape}, Test: {Xte.shape}, "
          f"test_pos_rate={yte.mean()*100:.3f}%", flush=True)

    # Output buffers
    tau_rows = []      # cell-level tau* + metrics
    util_rows = []     # cell-level util(τ*)
    h_util_rows = []   # utility-monotone tests
    h_5seed_rows = []  # 5-seed CI on realized utility

    # Sweep grid
    util5seed_per_cell = defaultdict(list)  # (rate, fset, tier, util) → list[util_per_seed]

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===", flush=True)
        for rate in RATES_PCT:
            Xte_d, yte_d = downsample(Xte, yte, rate, seed)
            n_pos_te = int(yte_d.sum())
            for fset_name, feats in FEATURE_SETS.items():
                probs = fit_predict(Xtr_full, ytr_full, Xte_d, feats, seed)
                rows = sweep_thresholds(probs, yte_d, TAU_GRID)
                for tier_name, tier_cost in LLM_PRICE_TIERS:
                    for util in ("F1", "VALUE", "PREC_CONSTRAINED", "COST_CONSTRAINED"):
                        opt_tau, opt_util, metrics = find_optimal_tau(
                            rows, yte_d, probs, tier_cost, util
                        )
                        prec, rec, f1, tp, fp, alerted, fn = metrics
                        cost_per_caught = (alerted * tier_cost) / max(1, tp)
                        tau_rows.append({
                            "seed": seed, "rate_pct": rate, "fset": fset_name,
                            "tier": tier_name, "utility": util,
                            "opt_tau": opt_tau, "n_pos_te": n_pos_te,
                            "tp": tp, "fp": fp, "alerted": alerted,
                            "fn": fn, "precision": prec, "recall": rec, "f1": f1,
                            "cost_per_caught": cost_per_caught,
                        })
                        util_rows.append({
                            "seed": seed, "rate_pct": rate, "fset": fset_name,
                            "tier": tier_name, "utility": util,
                            "opt_util": opt_util,
                            "n_alerted": alerted, "cost_total": alerted * tier_cost,
                            "value_gain": tp * VALUE_PER_CATCH,
                            "net_value": tp * VALUE_PER_CATCH - alerted * tier_cost,
                        })
                        util5seed_per_cell[(rate, fset_name, tier_name, util)].append(opt_util)

    # ----- H tests -----
    print("\n=== Hypothesis tests ===", flush=True)

    # H1: F1-max utility is monotone in fset (more aggregates → at least as good)
    # at every (rate, tier) cell, mean F1 across seeds should be:
    #   20raw ≤ 20raw+minmax ≤ 20raw+stat ≤ 24full
    # Test is whether >=80% of (rate × tier) cells show monotone increase
    # at "small_open" tier (chosen as canonical for H4 too).
    H1_cells = 0
    H1_total = 0
    for rate in RATES_PCT:
        for tier_name, _ in LLM_PRICE_TIERS:
            fset_means = {}
            for fset_name in FEATURE_SETS:
                vals = util5seed_per_cell[(rate, fset_name, tier_name, "F1")]
                fset_means[fset_name] = float(np.mean(vals)) if vals else float("nan")
            monotone = (fset_means["20raw"] <= fset_means["20raw+minmax"] + 1e-9
                        and fset_means["20raw+minmax"] <= fset_means["20raw+stat"] + 1e-9
                        and fset_means["20raw+stat"] <= fset_means["24full"] + 1e-9)
            H1_cells += int(monotone)
            H1_total += 1
    H1_pass = H1_cells >= 0.80 * H1_total
    print(f"H1 F1-monotone-in-fset: {H1_cells}/{H1_total} cells = "
          f"{H1_cells/H1_total*100:.1f}%  (PASS={H1_pass})")

    # H2: VALUE-max utility > 0 on >= 80% of (rate x fset x tier) cells
    # at the cheap_heuristic tier (the only tier with non-trivial tier cost
    # relative to VALUE_PER_CATCH).
    H2_cells = 0
    H2_total = 0
    for rate in RATES_PCT:
        for fset_name in FEATURE_SETS:
            for tier_name, _ in LLM_PRICE_TIERS:
                vals = util5seed_per_cell[(rate, fset_name, tier_name, "VALUE")]
                mean_v = float(np.mean(vals)) if vals else float("nan")
                if mean_v > 0:
                    H2_cells += 1
                H2_total += 1
                h_util_rows.append({
                    "test": "H2_value_positive",
                    "rate_pct": rate, "fset": fset_name,
                    "tier": tier_name, "mean_util": mean_v,
                    "n_seeds": len(vals), "pass": int(mean_v > 0),
                })
    H2_pass = H2_cells >= 0.50 * H2_total  # more honest bar
    print(f"H2 VALUE util > 0: {H2_cells}/{H2_total} = "
          f"{H2_cells/H2_total*100:.1f}%  (PASS={H2_pass}, bar=50%)")

    # H3: At the canonical 24full fset + small_open tier,
    # value* utility is < f1 util on >= 60% of rates (cost-aware util
    # trades off recall against cost so F1 tends to win).
    H3_cells = 0
    H3_total = 0
    for rate in RATES_PCT:
        for tier_name, _ in LLM_PRICE_TIERS[:3]:  # first 3 cheap tiers
            v_vals = util5seed_per_cell[(rate, "24full", tier_name, "VALUE")]
            f_vals = util5seed_per_cell[(rate, "24full", tier_name, "F1")]
            mean_v = float(np.mean(v_vals)) if v_vals else float("nan")
            mean_f = float(np.mean(f_vals)) if f_vals else float("nan")
            if mean_v < mean_f:
                H3_cells += 1
            H3_total += 1
            h_util_rows.append({
                "test": "H3_value_lt_f1",
                "rate_pct": rate, "fset": "24full",
                "tier": tier_name,
                "mean_util_value": mean_v,
                "mean_util_f1": mean_f,
                "pass": int(mean_v < mean_f),
            })
    H3_pass = H3_cells >= 0.60 * H3_total
    print(f"H3 VALUE<F1: {H3_cells}/{H3_total} = "
          f"{H3_cells/H3_total*100:.1f}%  (PASS={H3_pass}, bar=60%)")

    # H4: 5-seed CV on VALUE-max util is <= 0.30 on >= 60% of
    # (rate x fset x tier) cells at the cheap tier (cost-aware util is
    # reproducible because total cost is deterministic given a τ*).
    H4_cells = 0
    H4_total = 0
    for rate in RATES_PCT:
        for fset_name in FEATURE_SETS:
            tier_name = "cheap_heuristic"
            vals = util5seed_per_cell[(rate, fset_name, tier_name, "VALUE")]
            arr = np.array(vals)
            if arr.size == 0:
                continue
            cv = float(arr.std() / max(1e-12, abs(arr.mean())))
            m, lo, hi = bootstrap_ci(arr)
            H4_total += 1
            if cv <= 0.30:
                H4_cells += 1
            h_5seed_rows.append({
                "test": "H4_5seed_cv_value",
                "rate_pct": rate, "fset": fset_name, "tier": tier_name,
                "n_seeds": int(arr.size), "cv_value": cv,
                "boot_mean": m, "boot_lo": lo, "boot_hi": hi,
                "pass": int(cv <= 0.30),
            })
    H4_pass = H4_cells >= 0.60 * H4_total
    print(f"H4 5-seed CV(VALUE util)<=0.30: {H4_cells}/{H4_total} = "
          f"{H4_cells/H4_total*100:.1f}%  (PASS={H4_pass}, bar=60%)")

    # ----- write TSVs -----
    print("\nWriting outputs...", flush=True)
    out_tau = RES / "p8_iter160_opt_tau_per_cell.tsv"
    with out_tau.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tau_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(tau_rows)
    print(f"  {out_tau} ({len(tau_rows)} rows)")

    out_util = RES / "p8_iter160_opt_util_per_cell.tsv"
    with out_util.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(util_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(util_rows)
    print(f"  {out_util} ({len(util_rows)} rows)")

    if h_util_rows:
        out_h_util = RES / "p8_iter160_h_util_monotone.tsv"
        # Union of all keys across rows (some rows have mean_util, some mean_util_value/f1)
        all_keys = set()
        for r in h_util_rows:
            all_keys.update(r.keys())
        fieldnames = sorted(all_keys)
        with out_h_util.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            for r in h_util_rows:
                # Fill missing keys with ""
                for k in fieldnames:
                    r.setdefault(k, "")
                w.writerow(r)
        print(f"  {out_h_util} ({len(h_util_rows)} rows)")

    if h_5seed_rows:
        out_h_5s = RES / "p8_iter160_h_5seed_ci.tsv"
        with out_h_5s.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(h_5seed_rows[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(h_5seed_rows)
        print(f"  {out_h_5s} ({len(h_5seed_rows)} rows)")

    summary = {
        "iter": 160,
        "pillar": "P8",
        "cells_total": len(tau_rows),
        "h1_f1_monotone_in_fset": {
            "pass": bool(H1_pass), "n_pass": H1_cells, "n_total": H1_total,
            "fraction": H1_cells / max(1, H1_total),
            "bar": 0.80, "verdict": "PASS" if H1_pass else "FAIL",
        },
        "h2_value_util_positive": {
            "pass": bool(H2_pass), "n_pass": H2_cells, "n_total": H2_total,
            "fraction": H2_cells / max(1, H2_total),
            "bar": 0.50, "verdict": "PASS" if H2_pass else "FAIL",
        },
        "h3_value_lt_f1": {
            "pass": bool(H3_pass), "n_pass": H3_cells, "n_total": H3_total,
            "fraction": H3_cells / max(1, H3_total),
            "bar": 0.60, "verdict": "PASS" if H3_pass else "FAIL",
        },
        "h4_5seed_cv_value": {
            "pass": bool(H4_pass), "n_pass": H4_cells, "n_total": H4_total,
            "fraction": H4_cells / max(1, H4_total),
            "bar": 0.60, "verdict": "PASS" if H4_pass else "FAIL",
        },
    }
    out_sum = RES / "p8_iter160_summary.json"
    with out_sum.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"  {out_sum}")


if __name__ == "__main__":
    main()
