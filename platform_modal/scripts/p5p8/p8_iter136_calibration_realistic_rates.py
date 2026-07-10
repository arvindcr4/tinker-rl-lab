#!/usr/bin/env python3
"""P8 JOB A (iter 136): calibration audit at realistic (downsampled) positive rates.

Fresh vein (not in 151 prior P8 rows). Prior P8 calibration work audited ECE
at the released 1.44% positive rate (iter-99 cohort calibration parity) and
the 4 positive rates of iter-17 PR-AUC (which focused on PR-AUC + top-K
*operating* metrics, NOT calibration). This iter pivots the calibration
audit DOWN to realistic fraud base rates (1.00%, 0.50%, 0.10%, 0.05%)
to test the production-frequency robustness of the iter-99 worst-cohort ECE
finding (every cell has ECE > 0.10; worst-cohort ECE on XGB-20raw Amount Q0
= 0.176). Operational hypothesis: per-cohort calibration gap may either
AMPLIFY at lower positive rates (smaller N_per_cohort-stratum, more binomial
sampling variance on per-cohort obs_rate) or COMPRESS (rare positives are
concentrated in fewer cohorts). Falsifiable.

Backbones (matching iter-99 / iter-104 / iter-124 / iter-132):
  XGB-20raw  : V1..V20 only (no LLM-sensor aggregates)
  XGB-24full : V1..V20 + V_mean + V_std + V_max + V_min

Cohorts (matching iter-99):
  V_mean quintile    (the LLM-sensor aggregate cohort)
  Amount quintile    (synthesized = rank(V_std) per iter-99 convention)
  Time tertile       (synthesized = rank(V_max) per iter-99 convention)

Calibration methods (matching iter-104):
  raw               (uncalibrated baseline)
  iso_per_cohort    (5-fold OOF PAVA isotonic per cohort)

Outputs (mirrors iter-99 / iter-104 / iter-132):
  experiments/results/p5p8/p8_iter136_cal_realistic.tsv
  experiments/results/p5p8/p8_iter136_cal_realistic_summary.json
  experiments/results/p5p8/p8_iter136_worst_ece_curve.tsv
"""

from __future__ import annotations
import csv
import json
import os
import random
from collections import defaultdict

DATA_DIR = "/home/claude/tinker-rl-lab-minimax"
OUT_DIR = f"{DATA_DIR}/experiments/results/p5p8"
TRAIN = f"{DATA_DIR}/fraud_data.csv"
TEST = f"{DATA_DIR}/test_data.csv"

SEED = 20260705
B = 400
N_BIN = 10
QUANTILES = 5
TERTILES = 3
TARGET_RATES = (1.44, 1.00, 0.50, 0.10, 0.05)

random.seed(SEED)


def load(path):
    feats, labels = [], []
    with open(path) as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            try:
                klass = int(r["Class"])
            except (KeyError, ValueError):
                continue
            vals = [float(r[f"V{i}"]) for i in range(1, 21)]
            vals.extend([float(r["V_mean"]), float(r["V_std"]), float(r["V_max"]), float(r["V_min"])])
            feats.append(vals)
            labels.append(klass)
    return feats, labels


def ece(probs, labels, n_bin=N_BIN):
    """10-bin uniform ECE."""
    bins = [[] for _ in range(n_bin)]
    for p, y in zip(probs, labels):
        b = min(int(p * n_bin), n_bin - 1)
        bins[b].append((p, y))
    e = 0.0
    n = len(probs)
    for b in bins:
        if not b:
            continue
        avg_p = sum(p for p, _ in b) / len(b)
        obs = sum(y for _, y in b) / len(b)
        e += abs(obs - avg_p) * len(b) / max(1, n)
    return e


def isotonic_pava(xs, ys):
    """Pool Adjacent Violators Algorithm."""
    n = len(xs)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: xs[i])
    blocks = []
    for i in order:
        x = xs[i]
        y = ys[i]
        if blocks and blocks[-1][2] == x:
            blocks[-1][0] += y
            blocks[-1][1] += 1
            blocks[-1][3] = x
        else:
            blocks.append([y, 1, x, x])
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(blocks) - 1:
            mean_i = blocks[i][0] / blocks[i][1]
            mean_ip1 = blocks[i + 1][0] / blocks[i + 1][1]
            if mean_i > mean_ip1:
                blocks[i][0] += blocks[i + 1][0]
                blocks[i][1] += blocks[i + 1][1]
                blocks[i][3] = blocks[i + 1][3]
                del blocks[i + 1]
                changed = True
                if i > 0:
                    i -= 1
                continue
            i += 1
    return sorted([(b[2], b[3], b[0] / b[1]) for b in blocks])


def apply_isotonic(bps, p):
    if not bps:
        return p
    if p < bps[0][0]:
        return bps[0][2]
    if p >= bps[-1][1]:
        return bps[-1][2]
    for lo, hi, v in bps:
        if lo <= p < hi:
            return v
    return bps[-1][2]


def make_quantiles(values, q):
    s = sorted(values)
    n = len(s)
    return [s[int(round(i * n / q))] for i in range(1, q)]


def assign_cohort(v, cuts):
    for i, c in enumerate(cuts):
        if v < c:
            return i
    return len(cuts)


def downsample_mask(labels, target_rate_pct, rng):
    """Return list of indices keeping all negatives and a random subsample of positives."""
    if abs(target_rate_pct - 1.44) < 1e-6:
        return list(range(len(labels)))
    pos = [i for i, y in enumerate(labels) if y == 1]
    neg = [i for i, y in enumerate(labels) if y == 0]
    n_pos_target = int(round(target_rate_pct / 100.0 * len(neg) / (1.0 - target_rate_pct / 100.0)))
    n_pos_target = min(n_pos_target, len(pos))
    tmp = list(pos)
    rng.shuffle(tmp)
    keep = set(tmp[:n_pos_target])
    return [i for i in range(len(labels)) if labels[i] == 0 or i in keep]


def main():
    print("[1/4] loading data")
    train_feats, train_y = load(TRAIN)
    test_feats, test_y = load(TEST)

    import numpy as np
    X_tr = np.asarray(train_feats, dtype=np.float64)
    X_tr_20 = X_tr[:, :20]
    y_tr = np.asarray(train_y, dtype=np.int32)
    X_te_20 = np.asarray(test_feats, dtype=np.float64)[:, :20]
    X_te_24 = np.asarray(test_feats, dtype=np.float64)
    y_te = np.asarray(test_y, dtype=np.int32)
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    spw = max(1, int(round(n_neg / max(1, n_pos))))

    print(f"  train: {len(y_tr)} (pos={n_pos}); test: {len(y_te)} (pos={int(y_te.sum())})")
    print(f"  scale_pos_weight = {spw}")

    print("[2/4] fitting XGB trees")
    import xgboost as xgb

    def fit(X, seed):
        m = xgb.XGBClassifier(
            n_estimators=180, max_depth=5, learning_rate=0.1,
            scale_pos_weight=spw, random_state=seed,
            tree_method="hist", verbosity=0, eval_metric="logloss",
        )
        m.fit(X, y_tr)
        return m

    rng = random.Random(SEED)
    m20 = fit(X_tr_20, SEED)
    m24 = fit(X_tr, SEED + 1)

    p20_full = m20.predict_proba(X_te_20)[:, 1]
    p24_full = m24.predict_proba(X_te_24)[:, 1]

    # Synthesize Amount/Time per iter-99 convention: Amount = rank(V_std),
    # Time = rank(V_max). The corpus has no native Amount/Time column.
    test_meta = []
    with open(TEST) as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            test_meta.append({
                "V_mean": float(row["V_mean"]),
                "V_std": float(row["V_std"]),
                "V_max": float(row["V_max"]),
                "V_min": float(row["V_min"]),
                "Amount": 0.0, "Time": 0.0,
            })
    ranks_std = sorted(range(len(test_meta)), key=lambda i: test_meta[i]["V_std"])
    ranks_max = sorted(range(len(test_meta)), key=lambda i: test_meta[i]["V_max"])
    for r, i in enumerate(ranks_std):
        test_meta[i]["Amount"] = (r + 1) / len(test_meta) * 1000.0
    for r, i in enumerate(ranks_max):
        test_meta[i]["Time"] = (r + 1) / len(test_meta) * 86400.0

    print("[3/4] downsampling test + per-cell calibration")

    main_rows = []
    for rate in TARGET_RATES:
        mask = downsample_mask(test_y, rate, rng)
        labels_m = [int(test_y[i]) for i in mask]
        probs20_m = [float(p20_full[i]) for i in mask]
        probs24_m = [float(p24_full[i]) for i in mask]
        vmean_m = [test_meta[i]["V_mean"] for i in mask]
        amt_m = [test_meta[i]["Amount"] for i in mask]
        time_m = [test_meta[i]["Time"] for i in mask]
        n_total = len(labels_m)
        n_pos_m = sum(labels_m)
        obs_rate = (n_pos_m / max(1, n_total)) * 100.0

        vmean_cuts = make_quantiles(vmean_m, QUANTILES)
        amt_cuts = make_quantiles(amt_m, QUANTILES)
        time_cuts = make_quantiles(time_m, TERTILES)

        def cohort_for(axis, i):
            if axis == "v_mean":
                return assign_cohort(vmean_m[i], vmean_cuts)
            if axis == "amount":
                return assign_cohort(amt_m[i], amt_cuts)
            if axis == "time":
                return assign_cohort(time_m[i], time_cuts)

        for tree_name, probs in [("XGB-20raw", probs20_m), ("XGB-24full", probs24_m)]:
            for cohort_axis in ["v_mean", "amount", "time"]:
                # Group indices by cohort
                cg = defaultdict(list)
                for i in range(n_total):
                    cg[cohort_for(cohort_axis, i)].append(i)

                # OOF per-cohort isotonic
                iso = [0.0] * n_total
                for c, idxs in cg.items():
                    if len(idxs) < 30:
                        for k in idxs:
                            iso[k] = probs[k]
                        continue
                    fold_size = max(1, len(idxs) // 5)
                    for fold in range(5):
                        start = fold * fold_size
                        end = (fold + 1) * fold_size if fold < 4 else len(idxs)
                        val = idxs[start:end]
                        tr_ = idxs[:start] + idxs[end:]
                        bps = isotonic_pava([probs[i] for i in tr_], [labels_m[i] for i in tr_])
                        for k in val:
                            iso[k] = apply_isotonic(bps, probs[k])

                for cal_name, cal_probs in [("raw", probs), ("iso_per_cohort", iso)]:
                    worst_ece = 0.0
                    sum_ece = 0.0
                    n_strata_used = 0
                    for c, idxs in cg.items():
                        pp = [cal_probs[i] for i in idxs]
                        yy = [labels_m[i] for i in idxs]
                        if len(pp) < 5:
                            continue
                        e = ece(pp, yy)
                        sum_ece += e
                        n_strata_used += 1
                        if e > worst_ece:
                            worst_ece = e
                    mean_cell = sum_ece / max(1, n_strata_used)
                    g_ece = ece(cal_probs, labels_m)
                    brier = sum((cal_probs[i] - labels_m[i]) ** 2 for i in range(n_total)) / max(1, n_total)
                    main_rows.append({
                        "rate_pct": rate, "obs_rate_pct": round(obs_rate, 4),
                        "tree": tree_name, "cohort": cohort_axis,
                        "calibration": cal_name, "n_total": n_total, "n_pos": n_pos_m,
                        "n_strata_used": n_strata_used,
                        "global_ece": round(g_ece, 6),
                        "worst_cell_ece": round(worst_ece, 6),
                        "mean_cell_ece": round(mean_cell, 6),
                        "brier": round(brier, 6),
                    })

    print(f"[4/4] writing {len(main_rows)} rows")
    fields = ["rate_pct", "obs_rate_pct", "tree", "cohort", "calibration",
              "n_total", "n_pos", "n_strata_used", "global_ece",
              "worst_cell_ece", "mean_cell_ece", "brier"]
    with open(f"{OUT_DIR}/p8_iter136_cal_realistic.tsv", "w") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for row in main_rows:
            w.writerow(row)

    # Worst-ECE curve (aggregate across cohorts, by rate × tree × cal)
    by_key = defaultdict(lambda: {"max_worst": 0.0, "sum_global": 0.0, "n": 0})
    for r in main_rows:
        k = (r["rate_pct"], r["tree"], r["calibration"])
        by_key[k]["max_worst"] = max(by_key[k]["max_worst"], r["worst_cell_ece"])
        by_key[k]["sum_global"] += r["global_ece"]
        by_key[k]["n"] += 1

    with open(f"{OUT_DIR}/p8_iter136_worst_ece_curve.tsv", "w") as fh:
        w = csv.DictWriter(fh, fieldnames=["rate_pct", "obs_rate_pct", "tree", "calibration",
                                            "max_worst_ece_across_cohorts", "mean_global_ece"], delimiter="\t")
        w.writeheader()
        for (rate, tree, cal), v in sorted(by_key.items()):
            obs_rate = next((r["obs_rate_pct"] for r in main_rows
                              if r["rate_pct"] == rate and r["tree"] == tree
                              and r["calibration"] == cal), None)
            w.writerow({
                "rate_pct": rate, "obs_rate_pct": obs_rate,
                "tree": tree, "calibration": cal,
                "max_worst_ece_across_cohorts": round(v["max_worst"], 6),
                "mean_global_ece": round(v["sum_global"] / max(1, v["n"]), 6),
            })

    summary = {
        "iter": 136,
        "backbones": ["XGB-20raw", "XGB-24full"],
        "cohorts": ["v_mean", "amount", "time"],
        "cal_methods": ["raw", "iso_per_cohort"],
        "rates_tested_pct": list(TARGET_RATES),
        "n_rows": len(main_rows),
        "fraud_pos_train": n_pos,
        "fraud_pos_test_release": int(y_te.sum()),
        "test_n": len(y_te),
        "scale_pos_weight": spw,
        "rng_seed": SEED,
    }
    with open(f"{OUT_DIR}/p8_iter136_cal_realistic_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
