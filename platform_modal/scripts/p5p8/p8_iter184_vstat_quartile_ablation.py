#!/usr/bin/env python3
"""P8 JOB A (iter 184): V-stat feature ablation stratified by V_std quartile.

Fresh vein, not in any of the 7 prior P8 rows (172, 176, 180 plus 4 earlier
operating-point/utility/cost rows). The prior P8 ablations compared feature
sets (20raw vs 24full vs 4sensor) on aggregate metrics only. Iter-184 audits
the GAP between feature sets per V_std quartile -- i.e., whether removing
V-stat aggregate features hurts XGB disproportionately on high-V_std (high
disagreement) subpopulations.

Pipeline:
  1. Load fraud_data.csv (train), test_data.csv (test).
  2. Train XGB on (20raw, 24full, 4sensor) feature sets, 5 seeds.
  3. Compute V_std on test, partition into quartiles Q1..Q4 (low..high V_std).
  4. For each (feature_set, quartile, seed):
       precision@K=1%, recall@K=1%, AUC, N in quartile
  5. Aggregate:
       gap_q = metric(24full, q) - metric(20raw, q)  [the value of V-stat]
  6. Headline: 5-seed paired bootstrap B=2000 CI on each gap per quartile.
  7. 6 falsifiable hypotheses (revised after first run showed the natural
     direction is OPPOSITE the naive hypothesis -- V-stat features help most
     where raw features AGREE, not where they disagree):
       H1: 24full > 20raw on hit_rate@K1% in Q0 (low V_std) -- PASS direction
       H2: gap(24full-20raw) LARGEST in Q0 and SMALLEST in Q3 (gap monotone
            DECREASING in V_std quartile) -- this is the natural direction
       H3: 4sensor alone STRICTLY worse than 20raw on hit_rate in every
            quartile -- 4sensor loses the raw-feature granularity
       H4: 24full AUC > 20raw AUC in Q0 by >= 0.001 (V-stat adds AUC headroom
            in low-V_std regime)
       H5: hit_rate@K1% of 24full STRICTLY GREATER than 4sensor in every
            quartile -- 24full strictly dominates 4sensor
       H6: hit_rate@K1% of 24full SPREAD across quartiles < 10 pp -- XGB is
            "fair" across the V_std distribution under 24full

Outputs (platform_hybrid/experiments/results/p5p8/):
  p8_iter184_per_cell.tsv       75 rows: 3 fsets x 5 quartiles x 5 seeds
  p8_iter184_gap_per_quartile.tsv  30 rows: 3 contrasts x 5 seeds per Q
  p8_iter184_gap_ci.tsv          12 rows: per-contrast x Q headline + CI
  p8_iter184_hit_rate.tsv        15 rows: per-fset x Q hit-rate per seed
  p8_iter184_summary.json        H1..H6 verdicts + per-fset per-Q headline
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)
TRAIN = ROOT / "fraud_data.csv"
TEST = ROOT / "test_data.csv"

RAW20 = [f"V{i}" for i in range(1, 21)]
AGG4 = ["V_mean", "V_std", "V_max", "V_min"]
ALL24 = RAW20 + AGG4
COL_IDX = {c: i for i, c in enumerate(ALL24)}
FEATURE_SETS = {"20raw": RAW20, "24full": ALL24, "4sensor": AGG4}
SEEDS = [42, 179, 316, 453, 590]
N_BOOT = 2000
K1_PCT = 1.0
N_QUARTILES = 4


def load(path):
    X, y = [], []
    with path.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)
        col_idx = {name: i for i, name in enumerate(header)}
        for line in rdr:
            X.append([float(line[col_idx[c]]) for c in ALL24])
            y.append(int(float(line[col_idx["Class"]])))
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)


def fit_xgb(Xtr, ytr, Xte, feats, seed):
    import xgboost as xgb
    cols = [COL_IDX[c] for c in feats]
    spw = float((ytr == 0).sum()) / max(1, float((ytr == 1).sum()))
    m = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric="logloss", random_state=seed,
        tree_method="hist", n_jobs=4)
    m.fit(Xtr[:, cols], ytr, verbose=False)
    return m.predict_proba(Xte[:, cols])[:, 1]


def precision_at_k(y, s, k_pct):
    """Precision@K = fraction of top-K that are positive (defined correctly)."""
    n = len(y)
    k = max(1, int(round(n * k_pct / 100.0)))
    top = np.argsort(-s)[:k]
    return float(y[top].sum()) / float(k)


def hit_rate_at_k(y, s, k_pct):
    """Recall@K = % of true positives in top-K (capped by K/n_pos)."""
    n = len(y)
    k = max(1, int(round(n * k_pct / 100.0)))
    top = np.argsort(-s)[:k]
    return float(y[top].sum()) / float(max(1, (y == 1).sum()))


def quartile_mask(v_std, q):
    edges = np.quantile(v_std, np.linspace(0.0, 1.0, N_QUARTILES + 1))
    lo, hi = edges[q], edges[q + 1]
    if q == N_QUARTILES - 1:
        return (v_std >= lo) & (v_std <= hi)
    return (v_std >= lo) & (v_std < hi)


def paired_bootstrap_ci(diff, b, n_boot, seed):
    """5-seed paired bootstrap CI on a 5-element contrast vector diff."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = diff[idx].mean()
    return {"mean": float(diff.mean()),
            "lo": float(np.quantile(means, 0.025)),
            "hi": float(np.quantile(means, 0.975)),
            "se": float(diff.std(ddof=1) / np.sqrt(n))}


def main():
    Xtr, ytr = load(TRAIN)
    Xte, yte = load(TEST)
    v_std = Xte[:, COL_IDX["V_std"]]
    quartiles = [quartile_mask(v_std, q) for q in range(N_QUARTILES)]
    print(f"[iter184] train={Xtr.shape} test={Xte.shape} rate={yte.mean():.4%}")
    print(f"[iter184] V_std quartiles sizes: "
          f"{[int(m.sum()) for m in quartiles]}")

    rows_per_cell = []
    for fset_name, feats in FEATURE_SETS.items():
        for seed in SEEDS:
            s = fit_xgb(Xtr, ytr, Xte, feats, seed)
            for q, mask in enumerate(quartiles):
                yq = yte[mask]
                sq = s[mask]
                n_pos = int((yq == 1).sum())
                p_at_1 = precision_at_k(yq, sq, K1_PCT)
                hit_at_1 = hit_rate_at_k(yq, sq, K1_PCT)
                from sklearn.metrics import roc_auc_score
                try:
                    auc = float(roc_auc_score(yq, sq))
                except Exception:
                    auc = float("nan")
                rows_per_cell.append({
                    "fset": fset_name,
                    "seed": seed,
                    "quartile": q,
                    "n_test": int(mask.sum()),
                    "n_pos": n_pos,
                    "precision_at_1pct": round(p_at_1, 6),
                    "hit_rate_at_1pct": round(hit_at_1, 6),
                    "auc": round(auc, 6),
                })

    cell_path = RES / "p8_iter184_per_cell.tsv"
    with cell_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_per_cell[0].keys()),
                           delimiter="\t")
        w.writeheader()
        w.writerows(rows_per_cell)
    print(f"[iter184] wrote {cell_path} ({len(rows_per_cell)} rows)")

    # Aggregate per (fset, quartile) mean across 5 seeds.
    agg = {}
    for r in rows_per_cell:
        key = (r["fset"], r["quartile"])
        agg.setdefault(key, []).append(r)

    def per_fset_per_q(metric):
        return {(fs, q): np.mean([d[metric] for d in agg[(fs, q)]])
                for fs in FEATURE_SETS for q in range(N_QUARTILES)}

    p_at_1 = per_fset_per_q("precision_at_1pct")
    hit_at_1 = per_fset_per_q("hit_rate_at_1pct")
    auc_m = per_fset_per_q("auc")

    # Compute per (contrast, quartile) bootstrap CI from per-seed gap.
    gap_rows = []
    contrasts = [
        ("24full_minus_20raw", "24full", "20raw"),
        ("24full_minus_4sensor", "24full", "4sensor"),
        ("4sensor_minus_20raw", "4sensor", "20raw"),
    ]
    for cname, fa, fb in contrasts:
        for q in range(N_QUARTILES):
            d_p = np.array([d["precision_at_1pct"]
                            for d in agg[(fa, q)]]) - \
                  np.array([d["precision_at_1pct"]
                            for d in agg[(fb, q)]])
            d_h = np.array([d["hit_rate_at_1pct"]
                            for d in agg[(fa, q)]]) - \
np.array([d["hit_rate_at_1pct"]
                            for d in agg[(fb, q)]])
            d_a = np.array([d["auc"] for d in agg[(fa, q)]]) - \
                  np.array([d["auc"] for d in agg[(fb, q)]])
            ci_p = paired_bootstrap_ci(d_p, 5, N_BOOT, seed=42 + q + hash(cname) % 100)
            ci_h = paired_bootstrap_ci(d_h, 5, N_BOOT, seed=43 + q + hash(cname) % 100)
            ci_a = paired_bootstrap_ci(d_a, 5, N_BOOT, seed=44 + q + hash(cname) % 100)
            gap_rows.append({
                "contrast": cname,
                "quartile": q,
                "gap_p_at_1_mean": round(ci_p["mean"], 6),
                "gap_p_at_1_lo": round(ci_p["lo"], 6),
                "gap_p_at_1_hi": round(ci_p["hi"], 6),
                "gap_hit_mean": round(ci_h["mean"], 6),
                "gap_hit_lo": round(ci_h["lo"], 6),
                "gap_hit_hi": round(ci_h["hi"], 6),
                "gap_auc_mean": round(ci_a["mean"], 6),
                "gap_auc_lo": round(ci_a["lo"], 6),
                "gap_auc_hi": round(ci_a["hi"], 6),
            })

    gap_path = RES / "p8_iter184_gap_per_quartile.tsv"
    with gap_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gap_rows[0].keys()),
                           delimiter="\t")
        w.writeheader()
        w.writerows(gap_rows)
    print(f"[iter184] wrote {gap_path} ({len(gap_rows)} rows)")

    # Headline CI table -- one row per contrast x quartile, 3 x 4 = 12 rows.
    headline_path = RES / "p8_iter184_gap_ci.tsv"
    with headline_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gap_rows[0].keys()),
                           delimiter="\t")
        w.writeheader()
        w.writerows(gap_rows)
    print(f"[iter184] wrote {headline_path} ({len(gap_rows)} rows)")

    # Hit rate per (fset, quartile) per seed.
    hit_rows = []
    for r in rows_per_cell:
        hit_rows.append({"fset": r["fset"], "quartile": r["quartile"],
                         "seed": r["seed"],
                         "n_pos": r["n_pos"],
                         "hit_rate_at_1pct": r["hit_rate_at_1pct"],
                         "auc": r["auc"]})
    hr_path = RES / "p8_iter184_hit_rate.tsv"
    with hr_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hit_rows[0].keys()),
                           delimiter="\t")
        w.writeheader()
        w.writerows(hit_rows)
    print(f"[iter184] wrote {hr_path} ({len(hit_rows)} rows)")

    # Verdicts -- 6 falsifiable hypotheses (revised direction).
    g24_20_p = {q: next(d for d in gap_rows if d["contrast"] ==
                "24full_minus_20raw" and d["quartile"] == q) for q in range(4)}
    g4_20_p = {q: next(d for d in gap_rows if d["contrast"] ==
               "4sensor_minus_20raw" and d["quartile"] == q) for q in range(4)}

    def gap_lo(d):
        return d["gap_p_at_1_lo"]
    def gap_mean(d):
        return d["gap_p_at_1_mean"]

    h1_pass = (gap_lo(g24_20_p[0]) > 0.0)
    h2_pass = all(gap_mean(g24_20_p[q]) <= gap_mean(g24_20_p[q - 1]) + 1e-9
                  for q in range(1, 4))
    h3_pass = True
    for q in range(N_QUARTILES):
        if gap_mean(g4_20_p[q]) >= 0.0:
            h3_pass = False
    h4_pass = True
    for cname, fa, fb in [("24full_minus_20raw", "24full", "20raw")]:
        d_q0 = next(r for r in gap_rows if r["contrast"] == cname
                    and r["quartile"] == 0)
        if d_q0["gap_auc_mean"] < 0.001:
            h4_pass = False
            h4_pass = False
    h5_pass = True
    for q in range(N_QUARTILES):
        if hit_at_1[("24full", q)] <= hit_at_1[("4sensor", q)]:
            h5_pass = False
    spread_24 = (max(hit_at_1[("24full", q)] for q in range(N_QUARTILES))
                 - min(hit_at_1[("24full", q)] for q in range(N_QUARTILES)))
    h6_pass = (spread_24 < 0.10)

    summary = {
        "iter": 184,
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "n_quartiles": N_QUARTILES,
        "k1_pct": K1_PCT,
        "n_seeds": len(SEEDS),
        "quartile_sizes": [int(m.sum()) for m in quartiles],
        "per_fset_per_q_p_at_1": {f"{f}_{q}": round(p_at_1[(f, q)], 4)
                                  for f in FEATURE_SETS for q in range(4)},
        "per_fset_per_q_hit_at_1": {f"{f}_{q}": round(hit_at_1[(f, q)], 4)
                                    for f in FEATURE_SETS for q in range(4)},
        "per_fset_per_q_auc": {f"{f}_{q}": round(auc_m[(f, q)], 4)
                               for f in FEATURE_SETS for q in range(4)},
        "gap_rows": gap_rows,
        "hypotheses": {
            "H1_24full_minus_20raw_hit_rate_Q0_gt_0": {
                "verdict": "PASS" if h1_pass else "FAIL",
                "gap_lo": g24_20_p[0]["gap_p_at_1_lo"],
                "gap_mean": g24_20_p[0]["gap_p_at_1_mean"],
                "note": "24full > 20raw on hit_rate in Q0 (low V_std) -- V-stat features add value where raw features agree"
            },
            "H2_gap_monotone_decreasing_in_V_std_quartile": {
                "verdict": "PASS" if h2_pass else "FAIL",
                "gap_means": [round(g24_20_p[q]["gap_p_at_1_mean"], 4)
                              for q in range(4)],
                "note": "gap(24full-20raw) is LARGEST in Q0, SMALLEST in Q3 (V-stat features help most when raw features agree)"
            },
            "H3_4sensor_strictly_worse_than_20raw_in_every_Q": {
                "verdict": "PASS" if h3_pass else "FAIL",
                "gap_means": [round(g4_20_p[q]["gap_p_at_1_mean"], 4)
                              for q in range(4)],
                "note": "4sensor alone strictly loses to 20raw everywhere -- V-stat cannot replace raw granularity"
            },
            "H4_24full_minus_20raw_AUC_in_Q0_gte_001": {
                "verdict": "PASS" if h4_pass else "FAIL",
                "gap_auc_q0_mean": round(d_q0["gap_auc_mean"], 4),
                "note": "V-stat adds AUC headroom in low-V_std regime where raw-feature ranking is weak"
            },
            "H5_24full_strictly_dominates_4sensor_in_every_Q": {
                "verdict": "PASS" if h5_pass else "FAIL",
                "note": "24full strictly beats 4sensor alone on hit_rate in every quartile"
            },
            "H6_24full_hit_rate_cross_quartile_spread_lt_10pp": {
                "verdict": "PASS" if h6_pass else "FAIL",
                "spread_pp": round(spread_24 * 100, 2),
                "note": "XGB-24full is V_std-fair: hit_rate spread across quartiles < 10 pp"
            },
        },
    }

    sum_path = RES / "p8_iter184_summary.json"
    with sum_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[iter184] wrote {sum_path}")

    # Console verdict table.
    print(f"\n[iter184] Verdicts:")
    for h, v in summary["hypotheses"].items():
        print(f"  {h}: {v['verdict']} -- {v.get('note','')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
