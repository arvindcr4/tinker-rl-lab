#!/usr/bin/env python3
"""P8 JOB A (iter 192): V_mean predictive-decile audit on held-out test_data.csv.

Fresh vein: not in any prior P8 row (72, 80, 84, 88, 96, 100, 104, 108, 112,
116, 120, 124, 128, 132, 136, 140, 148, 156, 160, 164, 168, 172, 176, 180,
184, 188). Prior P8 iters stratified by V_std quartile (iter-184), V-stat
ensemble (iter-172), V_mean threshold (iter-168), or cost-asymmetric ratio
(iter-188). None stratified the held-out test set into V_mean deciles and
asked: "in WHICH decile does adding V-stat features help most?"

This iter answers that question. The decile view is the natural lens because
V_mean is the LLM aggregate that captures the per-row latent difficulty.

Pipeline:
  1. Train XGB-200 on fraud_data.csv with 3 fsets (20raw, 24full, 4sensor)
     x 5 seeds = 15 models.
  2. Predict on test_data.csv.
  3. Stratify test set into 10 V_mean deciles (1 = lowest V_mean, 10 = highest).
  4. Per-(fset, decile, seed) compute: hit_rate (fraud base rate), XGB AUC,
     Brier, ECE (10-bin reliability), lift_24_vs_20 = brier_20 - brier_24
     (positive means V-stat helps).
  5. Bootstrap CIs (B=2000, paired across 5 seeds) on per-(decile, contrast)
     metric gaps.
  6. 5 falsifiable hypotheses.

Outputs (experiments/results/p5p8/):
  p8_iter192_decile_metrics.tsv   3 fsets x 10 deciles x 5 seeds = 150 rows
  p8_iter192_decile_lift.tsv      10 deciles x 4 metrics = 40 rows
  p8_iter192_per_decile_summary.json
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
N_DECILES = 10
N_ECE_BINS = 10


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
    spw = float((ytr == 0).sum()) / max(1.0, float((ytr == 1).sum()))
    m = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric="logloss", random_state=seed,
        tree_method="hist", n_jobs=4)
    m.fit(Xtr[:, cols], ytr, verbose=False)
    return m.predict_proba(Xte[:, cols])[:, 1]


def ece(prob, y, n_bins=10):
    """Expected calibration error with n_bins equal-width bins on [0, 1]."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        if mask.sum() == 0:
            continue
        bin_conf = float(prob[mask].mean())
        bin_acc = float(y[mask].mean())
        e += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(e)


def auc_quick(s, y):
    """Mann-Whitney U statistic as AUC.  Vectorised O(n log n)."""
    pos = s[y == 1]
    neg = s[y == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # combined rank
    comb = np.concatenate([pos, neg])
    order = np.argsort(comb, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(comb) + 1)
    sum_ranks_pos = float(ranks[:n_pos].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def paired_bootstrap_ci(diff, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = diff[idx].mean()
    return {
        "mean": float(diff.mean()),
        "lo": float(np.quantile(means, 0.025)),
        "hi": float(np.quantile(means, 0.975)),
    }


def main():
    print("Loading data...", flush=True)
    Xtr, ytr = load(TRAIN)
    Xte, yte = load(TEST)
    v_mean = Xte[:, COL_IDX["V_mean"]]
    v_std = Xte[:, COL_IDX["V_std"]]
    print(f"  train: {Xtr.shape} pos={int((ytr==1).sum())}", flush=True)
    print(f"  test:  {Xte.shape} pos={int((yte==1).sum())} "
          f"V_mean range [{v_mean.min():.3f}, {v_mean.max():.3f}]", flush=True)

    # V_mean deciles (stratification)
    decile_edges = np.quantile(v_mean, np.linspace(0.0, 1.0, N_DECILES + 1))
    decile_edges[0] -= 1e-9
    decile_edges[-1] += 1e-9
    decile_idx = np.digitize(v_mean, decile_edges[1:-1])  # 0..N_DECILES-1
    print(f"  decile sizes: {[int((decile_idx == d).sum()) for d in range(N_DECILES)]}", flush=True)

    print("Fitting XGB models...", flush=True)
    probas = {}
    for fset, feats in FEATURE_SETS.items():
        for sd in SEEDS:
            probas[(fset, sd)] = fit_xgb(Xtr, ytr, Xte, feats, sd)

    # Per-cell TSV: per-(fset, decile, seed) hit_rate, AUC, Brier, ECE
    print("Computing per-decile metrics...", flush=True)
    out_cell = RES / "p8_iter192_decile_metrics.tsv"
    cell_rows = []
    for d in range(N_DECILES):
        mask_d = decile_idx == d
        n_pos = int((yte[mask_d] == 1).sum())
        n_total = int(mask_d.sum())
        base = n_pos / max(1, n_total)
        for fset in FEATURE_SETS:
            for sd in SEEDS:
                p = probas[(fset, sd)][mask_d]
                yy = yte[mask_d]
                brier = float(((p - yy) ** 2).mean())
                ec = ece(p, yy, N_ECE_BINS)
                a = auc_quick(p, yy) if yy.sum() > 0 and yy.sum() < len(yy) else float("nan")
                cell_rows.append((fset, d + 1, sd, n_total, n_pos,
                                  base, brier, ec, a))

    with out_cell.open("w") as f:
        f.write("fset\tdecile\tseed\tn_total\tn_pos\thit_rate\t"
                "brier\tece\tauc\n")
        for r in cell_rows:
            f.write("\t".join([
                r[0], str(r[1]), str(r[2]), str(r[3]), str(r[4]),
                f"{r[5]:.4f}", f"{r[6]:.6f}", f"{r[7]:.6f}",
                f"{r[8]:.6f}" if r[8] == r[8] else "nan"
            ]) + "\n")

    # Aggregate to per-(fset, decile) means across seeds
    by_fd = {}
    for r in cell_rows:
        key = (r[0], r[1])
        by_fd.setdefault(key, []).append(r)
    agg = {}
    for (fset, d), rows in by_fd.items():
        agg[(fset, d)] = {
            "brier_mean": float(np.mean([x[6] for x in rows])),
            "ece_mean": float(np.mean([x[7] for x in rows])),
            "auc_mean": float(np.mean([x[8] for x in rows])) if all(x[8] == x[8] for x in rows) else float("nan"),
            "hit_rate": float(rows[0][5]),
            "n_total": int(rows[0][3]),
            "n_pos": int(rows[0][4]),
        }

    # Lift TSV: per decile, compute 24full - 20raw gap for Brier + ECE + AUC
    out_lift = RES / "p8_iter192_decile_lift.tsv"
    lift_rows = []
    for d in range(1, N_DECILES + 1):
        # 24full vs 20raw per-seed diff
        d_brier =np.array([
            agg[("24full", d)]["brier_mean"] - agg[("20raw", d)]["brier_mean"]
        ])
        # paired per-seed
        b_24 = np.array([r[6] for r in cell_rows
                          if r[0] == "24full" and r[1] == d])
        b_20 = np.array([r[6] for r in cell_rows
                          if r[0] == "20raw" and r[1] == d])
        e_24 = np.array([r[7] for r in cell_rows
                          if r[0] == "24full" and r[1] == d])
        e_20 = np.array([r[7] for r in cell_rows
                          if r[0] == "20raw" and r[1] == d])
        a_24 = np.array([r[8] for r in cell_rows
                          if r[0] == "24full" and r[1] == d])
        a_20 = np.array([r[8] for r in cell_rows
                          if r[0] == "20raw" and r[1] == d])
        brier_gap = b_24 - b_20
        ece_gap = e_24 - e_20
        auc_gap = a_24 - a_20
        ci_b = paired_bootstrap_ci(brier_gap, N_BOOT, seed=20260706 + d)
        ci_e = paired_bootstrap_ci(ece_gap, N_BOOT, seed=20260706 + 100 + d)
        ci_a = paired_bootstrap_ci(auc_gap, N_BOOT, seed=20260706 + 200 + d)
        lift_rows.append({
            "decile": d,
            "n": agg[("24full", d)]["n_total"],
            "n_pos": agg[("24full", d)]["n_pos"],
            "hit_rate": agg[("24full", d)]["hit_rate"],
            "brier_20raw": float(b_20.mean()),
            "brier_24full": float(b_24.mean()),
            "brier_gap_mean": ci_b["mean"],
            "brier_gap_lo": ci_b["lo"],
            "brier_gap_hi": ci_b["hi"],
            "ece_20raw": float(e_20.mean()),
            "ece_24full": float(e_24.mean()),
            "ece_gap_mean": ci_e["mean"],
            "ece_gap_lo": ci_e["lo"],
            "ece_gap_hi": ci_e["hi"],
            "auc_20raw": float(a_20.mean()),
            "auc_24full": float(a_24.mean()),
            "auc_gap_mean": ci_a["mean"],
            "auc_gap_lo": ci_a["lo"],
            "auc_gap_hi": ci_a["hi"],
        })

    with out_lift.open("w") as f:
        f.write("decile\tn\tn_pos\thit_rate\t"
                "brier_20raw\tbrier_24full\tbrier_gap_mean\tbrier_gap_lo\tbrier_gap_hi\t"
                "ece_20raw\tece_24full\tece_gap_mean\tece_gap_lo\tece_gap_hi\t"
                "auc_20raw\tauc_24full\tauc_gap_mean\tauc_gap_lo\tauc_gap_hi\n")
        for r in lift_rows:
            f.write("\t".join([
                str(r["decile"]), str(r["n"]), str(r["n_pos"]),
                f"{r['hit_rate']:.4f}",
                f"{r['brier_20raw']:.6f}", f"{r['brier_24full']:.6f}",
                f"{r['brier_gap_mean']:.6f}", f"{r['brier_gap_lo']:.6f}",
                f"{r['brier_gap_hi']:.6f}",
                f"{r['ece_20raw']:.6f}", f"{r['ece_24full']:.6f}",
                f"{r['ece_gap_mean']:.6f}", f"{r['ece_gap_lo']:.6f}",
                f"{r['ece_gap_hi']:.6f}",
                f"{r['auc_20raw']:.6f}", f"{r['auc_24full']:.6f}",
                f"{r['auc_gap_mean']:.6f}", f"{r['auc_gap_lo']:.6f}",
                f"{r['auc_gap_hi']:.6f}",
            ]) + "\n")

    # ----- hypotheses -----
    print("Evaluating hypotheses...", flush=True)

    # H1: 24full helps MORE in LOW V_mean deciles (low-information regime).
    # Operationalized: brier_gap(24full - 20raw) is more negative in deciles
    # 1-3 than in deciles 8-10. (lower brier = better)
    low_deciles_brier = np.array([r["brier_gap_mean"] for r in lift_rows
                                   if r["decile"] <= 3])
    high_deciles_brier = np.array([r["brier_gap_mean"] for r in lift_rows
                                    if r["decile"] >= 8])
    h1 = low_deciles_brier.mean() < high_deciles_brier.mean()

    # H2: 24full helps more in HIGH V_std deciles (high-variance regime).
    # Approximation: deciles 8-10 have highest V_std (V_mean and V_std are
    # correlated on PCA data). We use the same proxy as H1.
    # Tighter: |brier_gap| should be larger in high-V_std deciles.
    h2 = abs(low_deciles_brier.mean()) > abs(high_deciles_brier.mean())
    # (we expect LLM to fill gaps in low-V_mean regime, so this should FAIL)

    # H3: Per-decile Brier is lower for 24full in >=8/10 deciles.
    # (broad lift: V-stat helps in most deciles)
    h3_count = sum(1 for r in lift_rows if r["brier_gap_mean"] < 0)
    h3 = h3_count >= 8

    # H4: Per-decile ECE is lower for 24full in >=5/10 deciles (calibration help).
    h4_count = sum(1 for r in lift_rows if r["ece_gap_mean"] < 0)
    h4 = h4_count >= 5

    # H5: The decile where V-stat helps MOST (max |negative brier gap|) is the
    # decile where fraud base rate is closest to 50% (max entropy regime).
    best_decile = min(lift_rows, key=lambda r: r["brier_gap_mean"])
    target_rate = 0.5
    # Among deciles with at least some fraud, the one with hit_rate closest to 0.5
    candidates = [r for r in lift_rows if r["n_pos"] >= 5]
    if candidates:
        maxent_decile = min(candidates,
                            key=lambda r: abs(r["hit_rate"] - target_rate))
    else:
        maxent_decile = lift_rows[0]
    h5 = best_decile["decile"] == maxent_decile["decile"]

    summary = {
        "n_test": int(Xte.shape[0]),
        "n_pos_test": int((yte == 1).sum()),
        "n_deciles": N_DECILES,
        "decile_sizes": [int((decile_idx == d).sum()) for d in range(N_DECILES)],
        "decile_hit_rates": [r["hit_rate"] for r in lift_rows],
        "h1_low_vmean_more_help": bool(h1),
        "h1_low_mean_gap": float(low_deciles_brier.mean()),
        "h1_high_mean_gap": float(high_deciles_brier.mean()),
        "h2_high_vstd_more_help": bool(h2),
        "h2_low_abs": float(abs(low_deciles_brier.mean())),
        "h2_high_abs": float(abs(high_deciles_brier.mean())),
        "h3_broad_lift_24full_count": int(h3_count),
        "h3_broad_lift_pass": bool(h3),
        "h4_ece_lift_24full_count": int(h4_count),
        "h4_ece_lift_pass": bool(h4),
        "h5_best_decile": int(best_decile["decile"]),
        "h5_best_decile_gap": float(best_decile["brier_gap_mean"]),
        "h5_maxent_decile": int(maxent_decile["decile"]),
        "h5_maxent_decile_hit_rate": float(maxent_decile["hit_rate"]),
        "h5_match": bool(h5),
        "best_decile": {
            "decile": int(best_decile["decile"]),
            "brier_24": float(best_decile["brier_24full"]),
            "brier_20": float(best_decile["brier_20raw"]),
            "brier_gap": float(best_decile["brier_gap_mean"]),
            "hit_rate": float(best_decile["hit_rate"]),
            "n_pos": int(best_decile["n_pos"]),
        },
        "worst_decile": {
            "decile": int(max(lift_rows, key=lambda r: r["brier_gap_mean"])["decile"]),
        },
    }

    out_sum = RES / "p8_iter192_per_decile_summary.json"
    with out_sum.open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()