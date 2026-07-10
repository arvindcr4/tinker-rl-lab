#!/usr/bin/env python3
"""P8 JOB A (iter 152): cost-matrix seed-stability audit — CORRECTED.

Bug fix vs the first attempt: n_iter152's smaller XGB (n_est=100, depth=4)
gives uniformly high scores, so G_THR=0.001 fires on every sample.
This script matches the iter-148 XGB parameters exactly so the XGB-derived
n_llm_grad matches the iter-148 baseline; the only variance is the
**downsample seed**.

Speedup: 3 seeds x 5 rates x 4 fsets = 60 XGB fits total (one fit per
seed-rate-fset, then sweep 5 cost tiers analytically on the resulting
caughtt/n_llm_grad counts). Each tier-cost is just a multiplication, no
refit.

Stdlib + numpy + xgboost. <= 280 LoC.
"""
from __future__ import annotations
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)

# 3 downsample seeds. XGB random_state is fixed (matches iter-148).
DOWNSAMPLE_SEEDS = [20260706, 20260708, 20260710]
# Use the iter-148 EXACT XGB hyperparams so the n_llm_grad counts match.
N_EST = 180
MAX_DEPTH = 5
# Use a different random_state for XGB so we'd capture XGB-stability too,
# but reproduce iter-148 to get comparable numbers.
XGB_RANDOM_STATE = 20260706  # MATCH iter-148
COST_XGB = 0.0001
K_PCT = 2.0
G_THR = 0.001  # iter-80 threshold (also matches iter-148)

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


def downsample_positives_iter148(X, y, rate_pct, seed):
    """Match iter-148 EXACT: n_target_pos = round(n_te * rate_pct / 100)
    using default_rng(seed)."""
    rng = np.random.default_rng(seed)
    n_te = len(y)
    n_target_pos = max(1, int(round(n_te * rate_pct / 100.0)))
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) < n_target_pos:
        keep_pos = pos_idx
    else:
        keep_pos = rng.choice(pos_idx, size=n_target_pos, replace=False)
    keep = np.concatenate([keep_pos, neg_idx])
    keep.sort()
    return X[keep], y[keep]


def fit_predict(Xtr, ytr, Xte, feats, random_state):
    cols = [ALL24.index(c) for c in feats]
    Xtr_s = Xtr[:, cols]
    Xte_s = Xte[:, cols]
    n_pos_tr = max(1, int(ytr.sum()))
    n_neg_tr = max(1, len(ytr) - n_pos_tr)
    spw = n_neg_tr / n_pos_tr
    m = xgb.XGBClassifier(
        n_estimators=N_EST, max_depth=MAX_DEPTH, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="auc",
        random_state=random_state, n_jobs=4,
    )
    m.fit(Xtr_s, ytr)
    return m.predict_proba(Xte_s)[:, 1]


def gradient_band_fire(scores, top_k_mask, g_thr=G_THR):
    """iter-148 EXACT: fire where the sorted-score gradient is < g_thr,
    intersected with top-K mask."""
    sorted_idx = np.argsort(-scores)
    sorted_scores = scores[sorted_idx]
    grad = np.abs(np.diff(sorted_scores, prepend=sorted_scores[0] + 1.0))
    fire_sorted = (grad < g_thr)
    fire = np.zeros(len(scores), dtype=bool)
    fire[sorted_idx] = fire_sorted
    return fire & top_k_mask


def compute_cell(cost_llm, n_test, caught_xgb, n_llm_grad):
    """Match iter-148 cpd_grad/cpd_xgb formulas exactly."""
    cpd_xgb = COST_XGB
    cpd_grad = (n_test * COST_XGB + n_llm_grad * (cost_llm - COST_XGB)) / n_test
    cppr_xgb = cpd_xgb * n_test / max(1, caught_xgb)
    cppr_grad = cpd_grad * n_test / max(1, caught_xgb)
    acd = cpd_grad / max(1e-12, cpd_xgb)
    return cpd_xgb, cpd_grad, cppr_xgb, cppr_grad, acd


def main():
    t0 = time.time()
    Xtr, ytr = load(ROOT / "fraud_data.csv")
    Xte_full, yte_full = load(ROOT / "test_data.csv")
    print(f"[load] n_train = {len(Xtr)} pos = {int(ytr.sum())} | "
          f"n_test_full = {len(Xte_full)} pos = {int(yte_full.sum())}", flush=True)

    # Step 1: fit XGB ONCE per (seed, rate, fset); sweep all 5 tiers analytically.
    n_fits = len(DOWNSAMPLE_SEEDS) * len(RATES_PCT) * len(FEATURE_SETS)
    fit_done = 0
    cache = {}  # (seed, rate_pct, fset) -> (n_test, n_pos, caught_xgb, n_llm_grad)
    for seed in DOWNSAMPLE_SEEDS:
        for rate_pct in RATES_PCT:
            # Downsample the TEST set (not train) per iter-148 protocol.
            Xte, yte = downsample_positives(Xte_full, yte_full, rate_pct, seed)
            for fset_name, feats in FEATURE_SETS.items():
                scores = fit_predict(Xtr, ytr, Xte, feats, XGB_RANDOM_STATE)
                k = max(1, int(round(K_PCT / 100.0 * len(Xte))))
                top = np.argsort(-scores)[:k]
                top_k_mask = np.zeros(len(scores), dtype=bool)
                top_k_mask[top] = True
                caught = int(yte[top].sum())
                fire = gradient_band_fire(scores, top_k_mask, G_THR)
                n_llm = int(fire.sum())
                cache[(seed, rate_pct, fset_name)] = (
                    len(Xte), int(yte.sum()), caught, n_llm,
                )
                fit_done += 1
                if fit_done % 5 == 0:
                    print(f"[fit] {fit_done}/{n_fits} "
                          f"elapsed {time.time()-t0:.1f}s", flush=True)
    print(f"[fits done] {n_fits} fits in {time.time()-t0:.1f}s", flush=True)

    # Step 2: build (seed, rate, tier, fset) row per cell, 300 cells total.
    rows = []
    for seed in DOWNSAMPLE_SEEDS:
        for rate_pct in RATES_PCT:
            for tier_name, cost_llm in LLM_PRICE_TIERS:
                for fset_name in FEATURE_SETS:
                    n_test, n_pos, caught_xgb, n_llm_grad = cache[
                        (seed, rate_pct, fset_name)]
                    cpd_xgb, cpd_grad, cppr_xgb, cppr_grad, acd = compute_cell(
                        cost_llm, n_test, caught_xgb, n_llm_grad)
                    rows.append(dict(
                        seed=seed, rate_pct=rate_pct, tier=tier_name,
                        cost_llm=cost_llm, fset=fset_name,
                        n_test=n_test, n_pos=n_pos,
                        n_llm_grad=n_llm_grad, caught_xgb=caught_xgb,
                        k_top=max(1, int(round(K_PCT / 100.0 * n_test))),
                        cpd_xgb=cpd_xgb, cpd_grad=cpd_grad,
                        cppr_xgb=cppr_xgb, cppr_grad=cppr_grad, acd=acd,
                    ))

    fieldnames = ["seed", "rate_pct", "tier", "cost_llm", "fset",
                  "n_test", "n_pos", "n_llm_grad", "caught_xgb", "k_top",
                  "cpd_xgb", "cpd_grad", "cppr_xgb", "cppr_grad", "acd"]
    out_cell = RES / "p8_iter152_acd_5seed_per_cell.tsv"
    with out_cell.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"[save] {out_cell}", flush=True)

    # Per (rate, tier) CV across 3 seeds
    by_rt = defaultdict(list)
    for r in rows:
        by_rt[(r["rate_pct"], r["tier"])].append(r["acd"])
    rt_rows = []
    for (rate_pct, tier), vals in sorted(by_rt.items()):
        arr = np.array(vals)
        cv = float(arr.std() / max(1e-12, arr.mean())) if len(vals) > 1 else 0.0
        rt_rows.append(dict(
            rate_pct=rate_pct, tier=tier, n=len(vals),
            acd_mean=float(arr.mean()), acd_std=float(arr.std()),
            acd_min=float(arr.min()), acd_max=float(arr.max()),
            acd_cv=cv,
        ))
    out_rt = RES / "p8_iter152_acd_5seed_cv.tsv"
    with out_rt.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rt_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rt_rows)
    print(f"[save] {out_rt}", flush=True)

# Cheapest per seed
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append(r)
    cheapest_rows = []
    for seed in sorted(by_seed.keys()):
        cells = by_seed[seed]
        best = min(cells, key=lambda c: (c["acd"], c["rate_pct"]))
        cheapest_rows.append(dict(
            seed=seed, rate_pct=best["rate_pct"], tier=best["tier"],
            fset=best["fset"], acd=best["acd"],
            cppr_grad=best["cppr_grad"], cppr_xgb=best["cppr_xgb"],
        ))
    out_ch = RES / "p8_iter152_cheapest_cell.tsv"
    with out_ch.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(cheapest_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(cheapest_rows)
    print(f"[save] {out_ch}", flush=True)

    # Rate-monotone per (tier, fset, seed). iter-148 H4 was reported at the
    # per-tier average over fsets, so we test at this aggregation layer
    # too. Aggregate per-cell acd by (seed, tier, rate), then check if the
    # resulting per-(seed, tier) sequence over rates is monotonically
    # non-increasing as rate drops.
    rm_agg_per_seed = defaultdict(lambda: defaultdict(list))  # seed -> rate -> [acds]
    for r in rows:
        rm_agg_per_seed[r["seed"]][r["rate_pct"]].append(r["acd"])

    rm_agg = []
    for seed in DOWNSAMPLE_SEEDS:
        for tier_name in [t for t, _ in LLM_PRICE_TIERS]:
            # for each (tier, seed), get tier-mean acd per rate
            rate_means = []
            for rate_pct in RATES_PCT:
                # find cells matching (seed, rate, tier) across all fsets
                cells = [r for r in rows if r["seed"] == seed
                         and r["tier"] == tier_name
                         and r["rate_pct"] == rate_pct]
                mean_acd = sum(c["acd"] for c in cells) / max(1, len(cells))
                rate_means.append(mean_acd)
            # monotone: as rate DROPS (1.44 -> 0.05), acd INCREASES (or stays).
            # Iterate rates from largest (1.44) to smallest (0.05).
            # iter-148 H4 was "frontier_gpt4 acd INCREASES as rate DROPS",
            # so rate_means[i] (lower rate) >= rate_means[i-1] (higher rate)
            # for monotone INCREASING-as-rate-drops.
            monotone = all(rate_means[i] >= rate_means[i - 1] - 1e-6
                           for i in range(1, len(rate_means)))
            rm_agg.append(dict(
                seed=seed, tier=tier_name,
                acd_144=rate_means[0], acd_100=rate_means[1],
                acd_050=rate_means[2], acd_010=rate_means[3],
                acd_005=rate_means[4],
                monotone_increasing_as_rate_drops=monotone,
            ))

    out_rm = RES / "p8_iter152_rate_monotone.tsv"
    with out_rm.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rm_agg[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rm_agg)
    print(f"[save] {out_rm}", flush=True)

    # Verdicts
    h1 = {}
    for seed in DOWNSAMPLE_SEEDS:
        ch_cells = [r for r in rows if r["seed"] == seed
                    and r["tier"] == "cheap_heuristic"]
        all_tied = all(abs(r["acd"] - 1.0) < 1e-6 for r in ch_cells)
        h1[seed] = all_tied
    h1_pass = all(h1.values())

    h2_pass = all(cr["tier"] == "cheap_heuristic" for cr in cheapest_rows)
    per_seed_cheapest_tier = {cr["seed"]: cr["tier"] for cr in cheapest_rows}
    per_seed_cheapest_rate = {cr["seed"]: cr["rate_pct"] for cr in cheapest_rows}

    h3_per_seed = {rm["seed"]: rm["monotone_increasing_as_rate_drops"]
                   for rm in rm_agg if rm["tier"] == "frontier_gpt4"}
    # Test DIRECTION (H3 relaxed): does acd at 0.05% exceed acd at 1.44%
    # in the per-tier average? iter-148 H4 endpoint comparison.
    h3_dir = {}
    for rm in rm_agg:
        if rm["tier"] != "frontier_gpt4":
            continue
        h3_dir[rm["seed"]] = rm["acd_005"] > rm["acd_144"]
    h3_dir_pass = sum(1 for v in h3_dir.values() if v) >= 2
    h3_strict_monotone_pass = sum(1 for v in h3_per_seed.values() if v) >= 2

    cv_by_tier = defaultdict(list)
    for rr in rt_rows:
        if rr["tier"] != "cheap_heuristic":
            cv_by_tier[rr["tier"]].append(rr["acd_cv"])
    h4_per_tier = {t: float(np.mean(v)) for t, v in cv_by_tier.items()}
    h4_pass = all(v <= 0.10 for v in h4_per_tier.values())

    summary = dict(
        iter=152,
        n_cells_per_seed=len(FEATURE_SETS) * len(LLM_PRICE_TIERS) * len(RATES_PCT),
        n_seeds=len(DOWNSAMPLE_SEEDS),
        n_total_cells=len(rows),
        downsample_seeds=DOWNSAMPLE_SEEDS,
        xgb_random_state=XGB_RANDOM_STATE,
        xgb_n_estimators=N_EST, xgb_max_depth=MAX_DEPTH,
        runtime_seconds=time.time() - t0,
        h1_pass_cheap_heuristic_tied_at_all_seeds=h1_pass,
        h1_per_seed_tied=h1,
        h2_pass_cheapest_is_cheap_heuristic=h2_pass,
        h2_per_seed_cheapest_tier=per_seed_cheapest_tier,
        h2_per_seed_cheapest_rate=per_seed_cheapest_rate,
        h3_pass_frontier_gpt4_endpoint_direction=h3_dir_pass,
        h3_strict_monotone_pass=h3_strict_monotone_pass,
        h3_per_seed_endpoint_direction=h3_dir,
        h3_per_seed_monotone=h3_per_seed,
        h4_pass_per_tier_mean_cv_leq_010=h4_pass,
        h4_per_tier_mean_cv=h4_per_tier,
        cheapest_cell_3seed=cheapest_rows,
        rate_monotone_per_seed_tier=rm_agg,
    )
    out_sum = RES / "p8_iter152_summary.json"
    with out_sum.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[save] {out_sum}", flush=True)

    print("\n=== H1: cheap_heuristic tied at all seeds ===")
    print(f"  H1 PASS = {h1_pass}; per-seed: {h1}")
    print(f"\n=== H2: cheapest cell at cheap_heuristic ===")
    print(f"  H2 PASS = {h2_pass}; per-seed cheapest: tier={per_seed_cheapest_tier}, rate={per_seed_cheapest_rate}")
    print(f"\n=== H3: frontier_gpt4 acd endpoint-direction (0.05% > 1.44%) majority ===")
    print(f"  H3 PASS = {h3_dir_pass}; per-seed direction: {h3_dir}; per-seed strict-monotone: {h3_per_seed}")
    print(f"\n=== H4: per-tier mean CV(acd) <= 0.10 ===")
    print(f"  H4 PASS = {h4_pass}; per-tier mean CV: {h4_per_tier}")
    print(f"\n=== runtime ===")
    print(f"  total = {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
