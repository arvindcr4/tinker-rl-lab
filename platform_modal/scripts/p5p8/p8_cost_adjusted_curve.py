#!/usr/bin/env python3
"""P8 cost-adjusted operating curve (iter 16).

The iter-4 calibration paper measures ROC-AUC + accuracy at the released
positive rate (1.44%). The iter-12 PR-AUC table measures PR-AUC + top-1%
precision at five positive rates. This iter closes the operational loop:
at a fixed top-K% review budget (the analyst queue), what is the
dollar cost AND the precision/recall recovered under each deployment
mode?

Four deployment modes are compared on the released 10,000-row test split:

  M1 XGB-20raw    : tree on the 20 raw V-features only, no LLM.
  M2 XGB-24full   : tree on the 20 raw + 4 hand-engineered aggregates.
                    Treats the aggregates as a free oracle LLM sensor.
  M3 Hybrid-10%   : XGB-20raw for the bottom 90% (no LLM cost);
                    XGB-24full for the top 10% (LLM sensor paid per row).
                    Realistic fraud-ops posture: triage the suspicious
                    tail with the LLM-augmented scorer, pay nothing on
                    the long safe tail.
  M4 Hybrid-1%    : same as M3 but top 1% only.

The per-row cost figures are taken from
platform_hybrid/experiments/results/p5p8/p8_cost_accounting.tsv (iter 4):
  XGBoost scorer      $0.0001 / row  (10k inference ~ $1)
  LLM sensor async    $0.0035 / row  (Qwen3.5-4B SFT, 120 in + 5 out tokens)
  Analyst review       $0.50 / row  (standard call-centre unit cost for
                                     a fraud analyst reviewing an alert)

The headline metric is **TP-recovered per dollar**, the number of true
positives the deployment recovers for each dollar of combined model +
review spend. We compute it at six review budgets (top 0.1%, 0.5%, 1%,
2%, 5%, 10%) and pair-bootstrap a 95% CI on the TP/$ delta between
deployment modes.

Jobs
----
J1 Load + train the three tree variants on the released 50k split.
J2 Predict on the released 10k held-out split.
J3 For each (mode, budget), simulate the analyst queue and compute
    TP / FP / FN, precision, recall, dollars spent, TP per dollar.
J4 Paired bootstrap CIs (n=400) on the TP/$ delta between M2/M1, M3/M1,
    M4/M1 at each budget.
J5 Write per-row operating-curve TSV (long format) and summary JSON.

Outputs
-------
platform_hybrid/experiments/results/p5p8/p8_cost_adjusted_curve.tsv
platform_hybrid/experiments/results/p5p8/p8_cost_adjusted_boot.tsv
platform_hybrid/experiments/results/p5p8/p8_cost_adjusted_summary.json
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "fraud_data.csv"
TEST = ROOT / "test_data.csv"
OUT_DIR = ROOT / "experiments" / "results" / "p5p8"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42
BOOT_SEED = 2026
N_BOOT = 400

V20 = [f"V{i}" for i in range(1, 21)]
V_AGG = ["V_mean", "V_std", "V_max", "V_min"]

# Cost model (USD)
COST_XGB_PER_ROW = 0.0001   # 10_000 inference ~ $1 batch
COST_LLM_PER_ROW = 0.0035   # async sensor, Qwen3.5-4B SFT, ~125 tokens
COST_REVIEW_PER_ROW = 0.50  # fraud analyst unit cost

# Review budgets as fractions of stream
BUDGETS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
BUDGET_LABELS = ["0.1%", "0.5%", "1%", "2%", "5%", "10%"]


def fit_tree(X_tr: np.ndarray, y_tr: np.ndarray, seed: int = RANDOM_SEED) -> xgb.XGBClassifier:
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=seed,
        n_jobs=4,
    )
    clf.fit(X_tr, y_tr, verbose=False)
    return clf


def load_and_train() -> tuple[dict, np.ndarray]:
    df_tr = pd.read_csv(TRAIN)
    df_te = pd.read_csv(TEST)

    X_tr_20 = df_tr[V20].to_numpy(dtype=np.float64)
    X_tr_24 = df_tr[V20 + V_AGG].to_numpy(dtype=np.float64)
    X_tr_4 = df_tr[V_AGG].to_numpy(dtype=np.float64)
    y_tr = df_tr["Class"].to_numpy(dtype=np.int32)

    X_te_20 = df_te[V20].to_numpy(dtype=np.float64)
    X_te_24 = df_te[V20 + V_AGG].to_numpy(dtype=np.float64)
    X_te_4 = df_te[V_AGG].to_numpy(dtype=np.float64)
    y_te = df_te["Class"].to_numpy(dtype=np.int32)

    trees = {
        "XGB-20raw": fit_tree(X_tr_20, y_tr),
        "XGB-24full": fit_tree(X_tr_24, y_tr),
        "XGB-4sensor": fit_tree(X_tr_4, y_tr),
    }
    scores = {
        "XGB-20raw": trees["XGB-20raw"].predict_proba(X_te_20)[:, 1],
        "XGB-24full": trees["XGB-24full"].predict_proba(X_te_24)[:, 1],
        "XGB-4sensor": trees["XGB-4sensor"].predict_proba(X_te_4)[:, 1],
    }
    return scores, y_te


def op_point(scores: np.ndarray, y: np.ndarray, top_frac: float) -> dict:
    """Apply a top-K% review budget; return TP/FP/FN, precision, recall, dollars."""
    n = len(y)
    k = max(1, int(round(top_frac * n)))
    order = np.argsort(-scores, kind="stable")
    review_idx = order[:k]
    y_pred_top = np.zeros(n, dtype=np.int32)
    y_pred_top[review_idx] = 1
    tp = int(((y_pred_top == 1) & (y == 1)).sum())
    fp = int(((y_pred_top == 1) & (y == 0)).sum())
    fn = int(((y_pred_top == 0) & (y == 1)).sum())
    tn = int(((y_pred_top == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    cost_review = k * COST_REVIEW_PER_ROW
    cost_model = n * COST_XGB_PER_ROW
    return {
        "n_reviewed": k,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "cost_review_usd": cost_review,
        "cost_model_usd": cost_model,
        "cost_total_usd": cost_review + cost_model,
        "tp_per_dollar": tp / (cost_review + cost_model),
    }


def hybrid_op_point(
    scores_base: np.ndarray, scores_full: np.ndarray, y: np.ndarray,
    top_frac: float, llm_coverage: float,
) -> dict:
    """M3/M4 hybrid: bottom (1-llm_coverage) scored by base tree only,
    top llm_coverage scored by full tree (LLM sensor paid)."""
    n = len(y)
    k_review = max(1, int(round(top_frac * n)))
    k_llm = max(1, int(round(llm_coverage * n)))
    # Stage 1: score every row by base
    order = np.argsort(-scores_base, kind="stable")
    review_idx = order[:k_review]
    llm_idx = order[:k_llm]
    y_pred_top = np.zeros(n, dtype=np.int32)
    y_pred_top[review_idx] = 1
    tp = int(((y_pred_top == 1) & (y == 1)).sum())
    fp = int(((y_pred_top == 1) & (y == 0)).sum())
    fn = int(((y_pred_top == 0) & (y == 1)).sum())
    tn = int(((y_pred_top == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    cost_review = k_review * COST_REVIEW_PER_ROW
    cost_model = n * COST_XGB_PER_ROW
    cost_llm = k_llm * COST_LLM_PER_ROW
    cost_total = cost_review + cost_model + cost_llm
    return {
        "n_reviewed": k_review,
        "n_llm_scored": k_llm,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "cost_review_usd": cost_review,
        "cost_model_usd": cost_model,
        "cost_llm_usd": cost_llm,
        "cost_total_usd": cost_total,
        "tp_per_dollar": tp / cost_total,
    }


def bootstrap_ci_delta_tp_per_dollar(
    y: np.ndarray,
    fn_factory,
    fn_base_factory,
    budget: float,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> tuple[float, float, float, float, float]:
    """Paired bootstrap on (TP_per_dollar_X_full - TP_per_dollar_X_base).

    fn_factory(b_idx) and fn_base_factory(b_idx) return scores arrays for the
    bootstrap resample. We re-rank within each bootstrap to keep the
    same top-K fractionunder resampling."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot, dtype=np.float64)
    base_tps = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s_full = fn_factory(idx)
        s_base = fn_base_factory(idx)
        op_full = op_point(s_full, y[idx], budget)
        op_base = op_point(s_base, y[idx], budget)
        deltas[b] = op_full["tp_per_dollar"] - op_base["tp_per_dollar"]
        base_tps[b] = op_base["tp_per_dollar"]
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(deltas.mean()), float(lo), float(hi), float(base_tps.mean()), float(np.std(deltas))


def main() -> None:
    scores, y = load_and_train()
    print(f"[p8_cost_adjusted] loaded: n_test={len(y)}, pos_test={int(y.sum())}")

    rows = []
    # M1: XGB-20raw baseline
    for b, blab in zip(BUDGETS, BUDGET_LABELS):
        op = op_point(scores["XGB-20raw"], y, b)
        op["mode"] = "M1_XGB-20raw"
        op["budget_label"] = blab
        op["budget"] = b
        rows.append(op)
    # M2: XGB-24full (oracle LLM-as-sensor)
    for b, blab in zip(BUDGETS, BUDGET_LABELS):
        op = op_point(scores["XGB-24full"], y, b)
        op["mode"] = "M2_XGB-24full_oracle"
        op["budget_label"] = blab
        op["budget"] = b
        rows.append(op)
    # M3: Hybrid with LLM coverage = 10%
    for b, blab in zip(BUDGETS, BUDGET_LABELS):
        op = hybrid_op_point(scores["XGB-20raw"], scores["XGB-24full"], y, b, llm_coverage=0.10)
        op["mode"] = "M3_Hybrid10"
        op["budget_label"] = blab
        op["budget"] = b
        rows.append(op)
    # M4: Hybrid with LLM coverage = 1%
    for b, blab in zip(BUDGETS, BUDGET_LABELS):
        op = hybrid_op_point(scores["XGB-20raw"], scores["XGB-24full"], y, b, llm_coverage=0.01)
        op["mode"] = "M4_Hybrid01"
        op["budget_label"] = blab
        op["budget"] = b
        rows.append(op)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "p8_cost_adjusted_curve.tsv", sep="\t", index=False)
    print(f"[p8_cost_adjusted] wrote {len(df)} operating rows -> p8_cost_adjusted_curve.tsv")

    # Bootstrap CIs on TP/$ deltas
    boot_rows = []
    for b, blab in zip(BUDGETS, BUDGET_LABELS):
        # M2 - M1
        fn_full = lambda idx, s=scores["XGB-24full"]: s[idx]
        fn_base = lambda idx, s=scores["XGB-20raw"]: s[idx]
        mean_d, lo, hi, base_mean, base_std = bootstrap_ci_delta_tp_per_dollar(
            y, fn_full, fn_base, b, n_boot=N_BOOT, seed=BOOT_SEED,
        )
        boot_rows.append({
            "comparison": "M2_minus_M1",
            "budget": b, "budget_label": blab,
            "delta_tp_per_dollar_mean": mean_d,
            "ci_lo": lo, "ci_hi": hi,
            "base_tp_per_dollar_mean": base_mean,
            "base_tp_per_dollar_std": base_std,
            "excl_zero": "yes" if (lo > 0 or hi < 0) else "no",
        })
        # M3 - M1
        # Hybrid's TP/$ is on a different cost base; compare TP directly
        # but on a TP-per-total-cost basis. We replicate the same bootstrap
        # on the hybrid op_point.
        def hybrid_op(idx, _s_base=scores["XGB-20raw"], _s_full=scores["XGB-24full"], _b=b):
            return hybrid_op_point(_s_base[idx], _s_full[idx], y[idx], _b, llm_coverage=0.10)["tp_per_dollar"]
        def base_op(idx, _s=scores["XGB-20raw"], _b=b):
            return op_point(_s[idx], y[idx], _b)["tp_per_dollar"]
        rng = np.random.default_rng(BOOT_SEED)
        n = len(y)
        deltas = np.empty(N_BOOT)
        for k in range(N_BOOT):
            idx = rng.integers(0, n, size=n)
            deltas[k] = hybrid_op(idx) - base_op(idx)
        m, lo3, hi3 = float(deltas.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
        boot_rows.append({
            "comparison": "M3_minus_M1",
            "budget": b, "budget_label": blab,
            "delta_tp_per_dollar_mean": m,
            "ci_lo": lo3, "ci_hi": hi3,
            "base_tp_per_dollar_mean": float(np.mean([base_op(np.array([i])) for i in range(min(n, 200))])),
            "base_tp_per_dollar_std": float(np.std([base_op(np.array([i])) for i in range(min(n, 200))])),
            "excl_zero": "yes" if (lo3 > 0 or hi3 < 0) else "no",
        })
        # M4 - M1
        def hybrid_op1(idx, _s_base=scores["XGB-20raw"], _s_full=scores["XGB-24full"], _b=b):
            return hybrid_op_point(_s_base[idx], _s_full[idx], y[idx], _b, llm_coverage=0.01)["tp_per_dollar"]
        rng = np.random.default_rng(BOOT_SEED + 1)
        deltas = np.empty(N_BOOT)
        for k in range(N_BOOT):
            idx = rng.integers(0, n, size=n)
            deltas[k] = hybrid_op1(idx) - base_op(idx)
        m4, lo4, hi4 = float(deltas.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
        boot_rows.append({
            "comparison": "M4_minus_M1",
            "budget": b, "budget_label": blab,
            "delta_tp_per_dollar_mean": m4,
            "ci_lo": lo4, "ci_hi": hi4,
            "base_tp_per_dollar_mean": float(np.mean([base_op(np.array([i])) for i in range(min(n, 200))])),
            "base_tp_per_dollar_std": float(np.std([base_op(np.array([i])) for i in range(min(n, 200))])),
            "excl_zero": "yes" if (lo4 > 0 or hi4 < 0) else "no",
        })
    dfb = pd.DataFrame(boot_rows)
    dfb.to_csv(OUT_DIR / "p8_cost_adjusted_boot.tsv", sep="\t", index=False)
    print(f"[p8_cost_adjusted] wrote {len(dfb)} bootstrap rows -> p8_cost_adjusted_boot.tsv")

    # Summary JSON
    summary = {
        "n_test": int(len(y)),
        "pos_test": int(y.sum()),
        "cost_model": {
            "xgb_per_row_usd": COST_XGB_PER_ROW,
            "llm_per_row_usd": COST_LLM_PER_ROW,
            "review_per_row_usd": COST_REVIEW_PER_ROW,
        },
        "by_mode_budget": [],
    }
    for _, r in df.iterrows():
        summary["by_mode_budget"].append({
            "mode": r["mode"], "budget_label": r["budget_label"], "budget": float(r["budget"]),
            "tp": int(r["tp"]), "fp": int(r["fp"]), "fn": int(r["fn"]),
            "precision": float(r["precision"]), "recall": float(r["recall"]),
            "cost_total_usd": float(r["cost_total_usd"]),
            "tp_per_dollar": float(r["tp_per_dollar"]),
        })
    with (OUT_DIR / "p8_cost_adjusted_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[p8_cost_adjusted] wrote summary -> p8_cost_adjusted_summary.json")

    # Headline print
    print("\n=== Headline: TP per dollar by mode and budget ===")
    pivot = df.pivot_table(index="mode", columns="budget_label", values="tp_per_dollar")
    pivot = pivot.reindex(BUDGET_LABELS, axis=1)
    print(pivot.to_string(float_format=lambda x: f"{x:8.4f}"))
    print("\n=== Headline: paired-bootstrap CI on TP/$ deltas ===")
    print(dfb.to_string(index=False, float_format=lambda x: f"{x:10.4f}"))


if __name__ == "__main__":
    main()