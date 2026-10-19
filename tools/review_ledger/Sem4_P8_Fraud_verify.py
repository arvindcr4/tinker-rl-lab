"""Verify the Sem4_P8_Fraud corrected descendant's retained numbers against the released CSVs.

Usage: python3 tools/review_ledger/Sem4_P8_Fraud_verify.py <dir containing fraud_data.csv,train_data.csv,test_data.csv>
CSVs are gitignored; they are produced by platform_local/train_xgboost.py (make_classification seed 42).
Checks: row/positive counts, test subset of fraud_data (leakage of the historical calibration pipeline),
train/test disjointness, aggregate columns are deterministic row functions, and a CPU refit of the
platform_local/train_xgboost.py configuration on train_data.csv -> test_data.csv.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# scikit-learn does not distribute stubs or a py.typed marker.
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score  # type: ignore[import-untyped]

d = Path(sys.argv[1])
out = (
    Path(__file__).resolve().parents[2]
    / "platform_hybrid/experiments/results/review_2026-10-18/Sem4_P8_Fraud"
)
out.mkdir(parents=True, exist_ok=True)
names = ("fraud_data.csv", "train_data.csv", "test_data.csv")
full, tr, te = (pd.read_csv(d / f) for f in names)


COLS = sorted(full.columns)  # fraud_data.csv puts Class before the aggregates; splits put it last


def key(df):
    return set(map(tuple, df[COLS].round(10).itertuples(index=False, name=None)))


kf, ktr, kte = key(full), key(tr), key(te)
raw = [f"V{i}" for i in range(1, 21)]
agg_ok = all(
    np.allclose(full[c], f(full[raw]))
    for c, f in [
        ("V_mean", lambda x: x.mean(1)),
        ("V_std", lambda x: x.std(1)),
        ("V_max", lambda x: x.max(1)),
        ("V_min", lambda x: x.min(1)),
    ]
)
res = {
    "sha256": {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in names},
    "n_full": len(full),
    "pos_full": int(full.Class.sum()),
    "n_train": len(tr),
    "pos_train": int(tr.Class.sum()),
    "n_test": len(te),
    "pos_test": int(te.Class.sum()),
    "test_rows_in_full": len(kte & kf),
    "train_rows_in_full": len(ktr & kf),
    "train_test_overlap": len(ktr & kte),
    "aggregates_deterministic_of_V1_V20": bool(agg_ok),
    "constant_negative_accuracy_test": float(1 - te.Class.mean()),
}
try:
    # XGBoost is an optional CPU-refit dependency, absent from core/dev installs.
    import xgboost as xgb

    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="auc",
    )
    clf.fit(tr.drop(columns="Class"), tr.Class)
    p = clf.predict_proba(te.drop(columns="Class"))[:, 1]
    yh = (p >= 0.5).astype(int)
    res["refit_disjoint"] = {
        "xgboost_version": xgb.__version__,
        "auc": float(roc_auc_score(te.Class, p)),
        "precision": float(precision_score(te.Class, yh)),
        "recall": float(recall_score(te.Class, yh)),
        "f1": float(f1_score(te.Class, yh)),
        "alerts_at_0.5": int(yh.sum()),
    }
except ImportError:
    res["refit_disjoint"] = "xgboost unavailable"
(out / "verify.json").write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res, indent=2))
