"""Shared feature lists, loaders, and model helpers for the P8 fraud-detector scripts.

Every helper here is a verbatim move of a copy that was duplicated across the
``p8_*`` / ``synth_*`` scripts; behaviour is unchanged. Helpers take any former
script-level setting (seed, budget, paths, feature map) as an explicit argument.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))  # platform_modal/scripts, for _paths
from _paths import REPO_ROOT  # noqa: E402

RAW20 = [f"V{i}" for i in range(1, 21)]
AGG4 = ["V_mean", "V_std", "V_max", "V_min"]
ALL24 = RAW20 + AGG4
COL_IDX = {c: i for i, c in enumerate(ALL24)}
TRAIN = REPO_ROOT / "fraud_data.csv"
TEST = REPO_ROOT / "test_data.csv"


def load(path):
    """Read the 24 features and Class label as plain numpy arrays."""
    with path.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)
        idx = {n: i for i, n in enumerate(header)}
        X, y = [], []
        for line in rdr:
            X.append([float(line[idx[c]]) for c in ALL24])
            y.append(int(float(line[idx["Class"]])))
    return np.array(X), np.array(y)


def load_v2(path):
    """Like ``load`` but with explicit float64 / int32 dtypes."""
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
    """Class-weighted hist XGB on the ``feats`` columns of ALL24-ordered arrays."""
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


def fit_tree(X_tr, y_tr, seed):
    import xgboost as xgb

    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="auc",
        tree_method="hist", random_state=seed, n_jobs=4,
    )
    clf.fit(X_tr, y_tr)
    return clf


def load_split(train_path, test_path, feats, tree_seed):
    """Return dict of model -> test scores and y_te."""
    import pandas as pd

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    y_tr = train["Class"].to_numpy(np.int32)
    y_te = test["Class"].to_numpy(np.int32)
    scores = {}
    for name, cols in feats.items():
        clf = fit_tree(train[cols].to_numpy(np.float64), y_tr, tree_seed)
        scores[name] = clf.predict_proba(test[cols].to_numpy(np.float64))[:, 1]
    return scores, y_te


def auc_p8(scores, y):
    pos = scores[y == 1]; neg = scores[y == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0: return 0.5
    comb = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(comb)) + 1
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def recall_at_K(scores, y, k_pct):
    """Top-K mask and recall on positives."""
    n = len(scores)
    k = max(1, int(round(n * k_pct / 100.0)))
    top_k_idx = np.argsort(-scores)[:k]
    mask = np.zeros(n, dtype=bool)
    mask[top_k_idx] = True
    pos_total = max(1, int(y.sum()))
    pos_caught = int(y[mask].sum())
    return mask, pos_caught, pos_total


def downsample_positives(Xte, yte, target_rate_pct, rng):
    n_te = len(yte)
    n_target_pos = max(1, int(round(n_te * target_rate_pct / 100.0)))
    pos_idx = np.where(yte == 1)[0]
    neg_idx = np.where(yte == 0)[0]
    keep_pos = pos_idx if len(pos_idx) < n_target_pos else rng.choice(
        pos_idx, size=n_target_pos, replace=False)
    keep = np.concatenate([keep_pos, neg_idx])
    keep.sort()
    return Xte[keep], yte[keep]


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
