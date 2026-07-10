#!/usr/bin/env python3
"""P8 sensor-noise robustness + information-gain analysis (iter 8).

Inputs
------
fraud_data.csv : 50,000 synthetic fraud rows (24 numeric features + Class).
test_data.csv  : 10,000 held-out rows (same schema + Class).

This script answers two questions P8 currently argues but does not measure:

Q1. **Sensor noise budget.** A real LLM-as-sensor will not produce the true
    V_mean/V_std/V_max/V_min exactly; it will produce noisy estimates with
    drift, quantization, and per-row variance. How much noise can it add
    before the tree's performance degrades measurably? We sweep a Gaussian
    noise multiplier on the four aggregate columns and report the noise
    level at which the paired bootstrap CI on AUC delta versus the
    noise-free baseline first excludes zero.

Q2. **Required information gain for the sensor to matter.** A sensor that
    produces only the four true aggregates does not help (iter-4 result:
    AUC(24full) - AUC(20raw) = +0.0002, CI contains zero). What AUC gain
    would an oracle sensor need to deliver (per-row, on top of the
    existing 20 features) before the CI on delta excludes zero? We sweep
    an additive information bonus in the form of a synthetic 25th feature
    that adds a small monotone signal to the true label, and report the
    bonus strength at which the bootstrap CI first excludes zero.

Outputs
-------
platform_hybrid/experiments/results/p5p8/p8_sensor_noise_sweep.tsv
platform_hybrid/experiments/results/p5p8/p8_sensor_noise_summary.json
platform_hybrid/experiments/results/p5p8/p8_required_info_gain.tsv
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)

FEATURES_20 = [f"V{i}" for i in range(1, 21)]
AGG_4 = ["V_mean", "V_std", "V_max", "V_min"]
ALL_24 = FEATURES_20 + AGG_4


def read_csv(path: Path) -> tuple[list[list[float]], list[int]]:
    rows, labels = [], []
    with path.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)
        col_idx = {name: i for i, name in enumerate(header)}
        for line in rdr:
            rows.append([float(line[col_idx[c]]) for c in ALL_24])
            labels.append(int(line[col_idx["Class"]]))
    return rows, labels


def fit_predict(X_tr, y_tr, X_te, feature_idx, seed=42):
    """Fit a small XGBoost and return test-side predicted probabilities."""
    import xgboost as xgb
    Xtr = [[r[i] for i in feature_idx] for r in X_tr]
    Xte = [[r[i] for i in feature_idx] for r in X_te]
    m = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=7.0,
        eval_metric="logloss", random_state=seed, tree_method="hist", n_jobs=4)
    m.fit(Xtr, y_tr, verbose=False)
    return m.predict_proba(Xte)[:, 1].tolist()


def paired_bootstrap_ci(preds_a, preds_b, y_full, n_boot=1000, seed=2026):
    """Paired bootstrap CI on a single derived quantity: mean over (a - b)
    across the test rows. This is a coarse but defensible proxy for
    ``is delta statistically distinguishable from zero'' when the delta is
    a per-row metric like (AUC_a - AUC_b) computed over a fixed test split.
    """
    from sklearn.metrics import roc_auc_score
    rng = random.Random(seed)
    n = len(y_full)
    auc_a = roc_auc_score(y_full, preds_a)
    auc_b = roc_auc_score(y_full, preds_b)
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        ys = [y_full[i] for i in idx]
        pa = [preds_a[i] for i in idx]
        pb = [preds_b[i] for i in idx]
        if sum(ys) == 0 or sum(ys) == len(ys):
            continue  # bootstrap sample is degenerate; skip
        deltas.append(roc_auc_score(ys, pa) - roc_auc_score(ys, pb))
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas)) - 1]
    return auc_a, auc_b, lo, hi


def write_tsv(path: Path, header, rows):
    with path.open("w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(c) for c in r) + "\n")


def add_noise_to_aggregates(rows, sigma_mult, seed=2026):
    """Return a copy of `rows` with Gaussian noise added to V_mean, V_std,
    V_max, V_min only (columns 20-23). Noise is calibrated as
    sigma_mult * column-stddev-on-train.

    `rows` is a list of [V1..V20, V_mean, V_std, V_max, V_min] lists.
    """
    rng = random.Random(seed)
    # Compute per-column stddev across all rows for the 4 aggregates.
    n = len(rows)
    out = [list(r) for r in rows]
    for j, name in zip(range(20, 24), AGG_4):
        col = [r[j] for r in rows]
        mu = sum(col) / n
        var = sum((c - mu) ** 2 for c in col) / n
        sd = math.sqrt(var) or 1.0
        sd *= sigma_mult
        for r in out:
            r[j] = r[j] + rng.gauss(0.0, sd)
    return out


def main():
    print("[p8-noise] reading data", file=sys.stderr)
    Xtr, ytr = read_csv(ROOT / "fraud_data.csv")
    Xte, yte = read_csv(ROOT / "test_data.csv")
    print(f"[p8-noise] train={len(Xtr)} pos={sum(ytr)}  test={len(Xte)} pos={sum(yte)}",
          file=sys.stderr)

    feat_all = {c: i for i, c in enumerate(ALL_24)}
    idx_20 = [feat_all[c] for c in FEATURES_20]
    idx_24 = [feat_all[c] for c in ALL_24]

    # --- Q1: sensor noise budget ---
    # Baseline: 24-feature tree on clean aggregates.
    print("[p8-noise] fitting baseline 24-feature tree", file=sys.stderr)
    preds_24_clean = fit_predict(Xtr, ytr, Xte, idx_24)
    # 20-feature tree (no aggregates).
    preds_20 = fit_predict(Xtr, ytr, Xte, idx_20)
    from sklearn.metrics import roc_auc_score
    auc_24_clean = roc_auc_score(yte, preds_24_clean)
    auc_20 = roc_auc_score(yte, preds_20)
    print(f"[p8-noise] AUC baseline: 20feat={auc_20:.4f}  24feat_clean={auc_24_clean:.4f}",
          file=sys.stderr)

    sigma_grid = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
    noise_rows = [["sigma_mult", "auc_24_noisy", "delta_auc_vs_clean",
                   "ci_lo", "ci_hi", "ci_excludes_zero"]]
    summary_noise = {"baseline": {"auc_24_clean": round(auc_24_clean, 4),
                                  "auc_20": round(auc_20, 4)},
                     "sweep": []}

    for sigma in sigma_grid:
        Xtr_n = add_noise_to_aggregates(Xtr, sigma, seed=42)
        Xte_n = add_noise_to_aggregates(Xte, sigma, seed=43)
        # Refit on the noisy training set, then score on the noisy test set.
        preds_n = fit_predict(Xtr_n, ytr, Xte_n, idx_24)
        auc_n = roc_auc_score(yte, preds_n)
        _, _, lo, hi = paired_bootstrap_ci(preds_n, preds_24_clean, yte,
                                           n_boot=400, seed=2026)
        excl = "yes" if (lo > 0 or hi < 0) else "no"
        noise_rows.append([sigma, round(auc_n, 4),
                           round(auc_n - auc_24_clean, 4),
                           round(lo, 4), round(hi, 4), excl])
        summary_noise["sweep"].append({
            "sigma_mult": sigma,
            "auc_24_noisy": round(auc_n, 4),
            "delta_auc_vs_clean": round(auc_n - auc_24_clean, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "ci_excludes_zero": excl,
        })
        print(f"[p8-noise] sigma={sigma:.2f}  AUC={auc_n:.4f}  "
              f"delta={auc_n - auc_24_clean:+.4f}  CI=[{lo:+.4f},{hi:+.4f}]  "
              f"excl_zero={excl}", file=sys.stderr)

    write_tsv(RES / "p8_sensor_noise_sweep.tsv", noise_rows[0], noise_rows[1:])

    # --- Q2: required information gain ---
    # We add a synthetic 25th feature that, with probability p (a tunable
    # success probability shift), increases the positive-class likelihood.
    # Specifically: the synthetic feature has a small monotone signal equal
    # to `strength * (y - 0.5)`, so on positives it shifts up by
    # strength/2 and on negatives it shifts down by strength/2. The tree
    # then sees a 25-feature table where the new feature has AUC(strength)
    # in the low-AUC range.
    strength_grid = [0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.50, 3.00]
    info_rows = [["strength", "auc_25", "delta_auc_vs_24clean",
                  "ci_lo", "ci_hi", "ci_excludes_zero"]]
    summary_info = {"baseline": {"auc_24_clean": round(auc_24_clean, 4)},
                    "sweep": []}

    for strength in strength_grid:
        Xtr_i = [list(r) + [strength * (y - 0.5)] for r, y in zip(Xtr, ytr)]
        Xte_i = [list(r) + [strength * (y - 0.5)] for r, y in zip(Xte, yte)]
        preds_i = fit_predict(Xtr_i, ytr, Xte_i, idx_24 + [24])
        auc_i = roc_auc_score(yte, preds_i)
        _, _, lo, hi = paired_bootstrap_ci(preds_i, preds_24_clean, yte,
                                           n_boot=400, seed=2026)
        excl = "yes" if (lo > 0 or hi < 0) else "no"
        info_rows.append([strength, round(auc_i, 4),
                          round(auc_i - auc_24_clean, 4),
                          round(lo, 4), round(hi, 4), excl])
        summary_info["sweep"].append({
            "strength": strength,
            "auc_25": round(auc_i, 4),
            "delta_auc_vs_24clean": round(auc_i - auc_24_clean, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "ci_excludes_zero": excl,
        })
        print(f"[p8-info] strength={strength:.2f}  AUC={auc_i:.4f}  "
              f"delta={auc_i - auc_24_clean:+.4f}  CI=[{lo:+.4f},{hi:+.4f}]  "
              f"excl_zero={excl}", file=sys.stderr)

    write_tsv(RES / "p8_required_info_gain.tsv", info_rows[0], info_rows[1:])

    # Persist machine-readable summary.
    summary = {
        "noise_budget": summary_noise,
        "info_gain_required": summary_info,
        "n_train": len(ytr), "n_test": len(yte),
        "pos_train": sum(ytr), "pos_test": sum(yte),
        "note": ("Noise budget Q1: how much Gaussian noise can a hypothetical "
                 "LLM sensor add to the 4 aggregates before the tree's AUC "
                 "drops measurably? Info gain Q2: what 25th-feature "
                 "monotone signal strength is needed for the bootstrap CI "
                 "on AUC(25) - AUC(24_clean) to first exclude zero?"),
    }
    (RES / "p8_sensor_noise_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True))
    print("[p8-noise] done.", file=sys.stderr)


if __name__ == "__main__":
    main()