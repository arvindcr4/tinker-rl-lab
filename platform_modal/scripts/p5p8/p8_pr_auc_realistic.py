#!/usr/bin/env python3
"""P8 PR-AUC at realistic fraud ratios (iter 12).

The iter-4 calibration paper measures ROC-AUC at the released positive
rate (144 / 10_000 = 1.44%). ROC-AUC is informative but misleadingly
optimistic when the positive class is very rare; precision-recall AUC
(PR-AUC) and top-K metrics (precision@K, recall@K) are the standard
operating-point metrics for fraud. This iter downsamples positives in
the released test split to realistic base rates (0.05% / 0.10% / 0.50%
/ 1.00% / 1.44%-release) and measures PR-AUC, precision-at-top-1%, and
recall-at-top-1% on the three tree variants of iter-4 (XGB-20raw,
XGB-24full, XGB-4sensor).

Inputs
------
fraud_data.csv : 50,000 synthetic fraud rows (24 numeric features + Class).
test_data.csv  : 10,000 held-out rows (same schema + Class).

Jobs
----
J1 Load + clean + train all three trees on the released 50,000 training split.
J2 Predict on the released test split at five positive rates
    (release / 1.0% / 0.50% / 0.10% / 0.05%) by *down-sampling positives*.
    This is the standard fraud stress test: rank the model on a
    natural-rate stream and ask how high PR-AUC and top-1% precision
    remain.
J3 Compute PR-AUC + precision@top1% + recall@top1% per
    (model, positive-rate) cell.
J4 Paired bootstrap CIs (n=400) on the deltas
    delta_pr_auc(X_full, X_base) and delta_p_at_1pct(X_full, X_base).

Outputs
-------
platform_hybrid/experiments/results/p5p8/p8_pr_auc_realistic.tsv
platform_hybrid/experiments/results/p5p8/p8_pr_auc_realistic.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "fraud_data.csv"
TEST = ROOT / "test_data.csv"
OUT_DIR = ROOT / "experiments" / "results" / "p5p8"
RANDOM_SEED = 42
BOOT_SEED = 2026
N_BOOT = 400

V20 = [f"V{i}" for i in range(1, 21)]
V_AGG = ["V_mean", "V_std", "V_max", "V_min"]


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Mirror p8_calibration_cis: train = fraud_data.csv, test = test_data.csv."""
    df_tr = pd.read_csv(TRAIN)
    df_te = pd.read_csv(TEST)
    feat_20_tr = df_tr[V20].to_numpy(dtype=np.float64)
    feat_20_te = df_te[V20].to_numpy(dtype=np.float64)
    feat_full_te = df_te[V20 + V_AGG].to_numpy(dtype=np.float64)
    feat_4_te = df_te[V_AGG].to_numpy(dtype=np.float64)
    y_tr = df_tr["Class"].to_numpy(dtype=np.int32)
    y_te = df_te["Class"].to_numpy(dtype=np.int32)
    return (
        feat_20_tr,
        y_tr,
        np.column_stack([feat_20_te, df_te[V_AGG].to_numpy(dtype=np.float64)]) if False else feat_full_te,
        y_te,
        [feat_20_te, feat_full_te, feat_4_te],
    )


def fit_tree(X_tr: np.ndarray, y_tr: np.ndarray, seed: int = RANDOM_SEED) -> xgb.XGBClassifier:
    """Match the released XGBoost config from p8_calibration_cis.py."""
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=7,
        eval_metric="logloss",
        random_state=seed,
        tree_method="hist",
        verbosity=0,
    )
    clf.fit(X_tr, y_tr)
    return clf


def downsample_positives(
    y: np.ndarray, target_pos_rate: float, rng: np.random.Generator
) -> np.ndarray:
    """Return a mask that keeps all negatives and a subsample of positives
    so that the resulting positive rate is approximately target_pos_rate.
    """
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    n_neg = neg_idx.size
    target_pos = int(round(target_pos_rate * n_neg / max(1e-9, 1.0 - target_pos_rate)))
    target_pos = min(target_pos, pos_idx.size)
    if target_pos <= 0:
        return np.zeros_like(y, dtype=bool)
    keep_pos = rng.choice(pos_idx, size=target_pos, replace=False)
    mask = np.zeros_like(y, dtype=bool)
    mask[neg_idx] = True
    mask[keep_pos] = True
    return mask


def pr_metrics(y: np.ndarray, p: np.ndarray, top_frac: float = 0.01) -> dict:
    """Standard PR metrics: PR-AUC, precision-at-K and recall-at-K where K
    is the top top_frac fraction of scores (sorted descending)."""
    out = {
        "n_test": int(y.size),
        "n_pos": int(y.sum()),
        "pos_rate": float(y.mean()) if y.size else 0.0,
        "pr_auc": float(average_precision_score(y, p)) if y.sum() > 0 else float("nan"),
    }
    n_top = max(1, int(round(top_frac * y.size)))
    order = np.argsort(-p, kind="stable")
    top_y = y[order][:n_top]
    out["top_frac"] = float(top_frac)
    out["n_top"] = int(n_top)
    out["p_at_top"] = float(top_y.mean())
    out["r_at_top"] = float(top_y.sum() / max(1, y.sum())) if y.sum() else 0.0
    return out


def paired_bootstrap_diff(
    y_full: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    metric,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> tuple[float, float, float]:
    """Paired bootstrap on metric(y, p_a) - metric(y, p_b).
    metric must take (y, p) and return a scalar.
    """
    rng = np.random.default_rng(seed)
    n = y_full.size
    diffs = np.empty(n_boot, dtype=np.float64)
    base_a = metric(y_full, p_a)
    base_b = metric(y_full, p_b)
    base = base_a - base_b
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[b] = metric(y_full[idx], p_a[idx]) - metric(y_full[idx], p_b[idx])
    return float(base), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X20_tr, y_tr, X24_te, y_te, [X20_te, _, X4_te] = load_data()
    X24_tr = np.column_stack([X20_tr, pd.read_csv(TRAIN)[V_AGG].to_numpy(dtype=np.float64)])
    rng_down = np.random.default_rng(RANDOM_SEED)
    rows: list[dict] = []
    boot_seed = BOOT_SEED
    target_rates = [0.0144, 0.0100, 0.0050, 0.0010, 0.0005]
    rate_labels = ["release", "1.00%", "0.50%", "0.10%", "0.05%"]
    trees = {
        "XGB-20raw": fit_tree(X20_tr, y_tr),
        "XGB-24full": fit_tree(X24_tr, y_tr),
        "XGB-4sensor": fit_tree(
            pd.read_csv(TRAIN)[V_AGG].to_numpy(dtype=np.float64), y_tr
        ),
    }
    feature_sets = {"XGB-20raw": X20_te, "XGB-24full": X24_te, "XGB-4sensor": X4_te}
    for tgt, lbl in zip(target_rates, rate_labels):
        mask = downsample_positives(y_te, tgt, rng_down)
        y_sub = y_te[mask]
        for name, X in feature_sets.items():
            p_all = trees[name].predict_proba(X)[:, 1]
            p = p_all[mask]
            met = pr_metrics(y_sub, p)
            rows.append(
                {
                    "model": name,
                    "rate_label": lbl,
                    "target_rate": tgt,
                    **met,
                }
            )
    # Paired bootstrap CIs for XGB-24full - XGB-20raw at each rate.
    boot_rows = []
    for tgt, lbl in zip(target_rates, rate_labels):
        mask = downsample_positives(y_te, tgt, rng_down)
        y_sub = y_te[mask]
        p_full = trees["XGB-24full"].predict_proba(feature_sets["XGB-24full"])[:, 1][mask]
        p_raw = trees["XGB-20raw"].predict_proba(feature_sets["XGB-20raw"])[:, 1][mask]
        p_4 = trees["XGB-4sensor"].predict_proba(feature_sets["XGB-4sensor"])[:, 1][mask]
        for a, b in [("XGB-24full", "XGB-20raw"), ("XGB-24full", "XGB-4sensor"), ("XGB-20raw", "XGB-4sensor")]:
            pa = {"XGB-24full": p_full, "XGB-20raw": p_raw, "XGB-4sensor": p_4}[a]
            pb = {"XGB-24full": p_full, "XGB-20raw": p_raw, "XGB-4sensor": p_4}[b]
            for mname, mfn in [
                ("pr_auc", average_precision_score),
                (
                    "p_at_top",
                    lambda yy, pp: float(
                        yy[np.argsort(-pp, kind="stable")][
                            : max(1, int(round(0.01 * yy.size)))
                        ].mean()
                    ),
                ),
            ]:
                base, lo, hi = paired_bootstrap_diff(
                    y_sub, pa, pb, mfn, n_boot=N_BOOT, seed=boot_seed
                )
                boot_seed += 1
                boot_rows.append(
                    {
                        "rate_label": lbl,
                        "target_rate": tgt,
                        "a_minus_b": f"{a}-{b}",
                        "metric": mname,
                        "point_diff": base,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "ci_excludes_zero": bool(lo > 0 or hi < 0),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "p8_pr_auc_realistic.tsv", sep="\t", index=False)
    pd.DataFrame(boot_rows).to_csv(OUT_DIR / "p8_pr_auc_boot.tsv", sep="\t", index=False)
    summary = {
        "rates": list(zip(rate_labels, target_rates)),
        "per_cell": rows,
        "bootstrap": boot_rows,
    }
    with open(OUT_DIR / "p8_pr_auc_realistic.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    # Console sanity banner.
    print("P8 PR-AUC at realistic ratios")
    for r in rows:
        print(
            f"  {r['model']:>11s} rate={r['rate_label']:>8s} pos={r['n_pos']:>5d} "
            f"pr_auc={r['pr_auc']:.4f} p@top1%={r['p_at_top']*100:.2f}% r@top1%={r['r_at_top']*100:.2f}%"
        )


if __name__ == "__main__":
    main()
