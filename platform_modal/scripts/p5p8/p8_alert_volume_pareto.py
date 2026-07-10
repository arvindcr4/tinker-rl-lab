#!/usr/bin/env python3
"""P8 JOB A (iter 56): Alert-volume-constrained Pareto frontier with paired bootstrap CIs.

iter-52 (#62) measured ABSOLUTE regret vs the oracle (cost-axis);
iter-58 (#58) measured τ*(train) → τ̂(test) transfer gap under class-prior shift;
iter-12 (#17) measured recall@top-1% at downsampled fraud rates;
iter-59 measured $/fraud_caught on the (C_inv × L) grid;
none of these answers the operational question fraud-ops lead ACTUALLY asks:

  "Given I can review K% of the stream (a fixed alert-volume budget from the
   staffing model), which tree gives me the highest recall, and is the
   gap statistically detectable?"

This is the budget-K axis: it is the dual of iter-28's cost-optimal threshold
(which optimizes a cost function). It is also the missing link between the
iter-12 recall@top-1% point (which is K=1%) and the iter-58 transfer-gap
(which varies K but reports Δ-cost).

For each model ∈ {XGB-20raw, XGB-24full, XGB-4sensor}:
  Fit on fraud_data.csv (held-out split per iter-28), score on test_data.csv.
  For each K ∈ {0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 5.00} (review budget %):
    threshold τ_K = top-K%-th score cutoff
    compute (recall, precision, F1, $/dec, $/fraud_caught) at τ_K
    paired bootstrap B=400 on the (recall, F1) gaps vs each other tree
    95% percentile CI on the gap

Outputs
-------
platform_hybrid/experiments/results/p5p8/p8_alert_volume.tsv               (21 cells: 7 K × 3 trees)
platform_hybrid/experiments/results/p5p8/p8_alert_volume_boot.tsv          (42 paired-bootstrap rows: 7 K × 3 tree-pairs)
platform_hybrid/experiments/results/p5p8/p8_alert_volume_summary.json
platform_hybrid/experiments/results/p5p8/figures/p8_alert_volume.{png,pdf}

Stdlib + numpy + pandas + xgboost + sklearn + matplotlib. <=300 lines.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score

# ------------------------------------------------------------------ paths
ROOT = Path("/home/claude/tinker-rl-lab-minimax")
OUT = ROOT / "experiments" / "results" / "p5p8"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# Same operational costs as iter-48 / iter-52
C_INV = 0.50            # USD per alert (analyst review cost)
C_SENSE = 0.0035        # USD per row scored by LLM-as-sensor (24full, 4sensor only)
LOSS_GRID = [5, 25, 100, 250, 1000]   # L values to scan $/dec over

K_BUDGETS = [0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 5.00]   # review budget as % of stream
N_BOOT = 400
SEED = 20260704
RNG = np.random.default_rng(SEED)

RAW_FEATURES = [f"V{i}" for i in range(1, 21)]
SENSOR_FEATURES = ["V_mean", "V_std", "V_max", "V_min"]


# ------------------------------------------------------------------ data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(ROOT / "fraud_data.csv")
    test = pd.read_csv(ROOT / "test_data.csv")
    return train, test


def fit_and_score(model_kind: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Fit one tree and return the score for the test set (higher = more fraud-like)."""
    if model_kind == "20raw":
        feats = RAW_FEATURES
    elif model_kind == "24full":
        feats = RAW_FEATURES + SENSOR_FEATURES
    elif model_kind == "4sensor":
        feats = SENSOR_FEATURES
    else:
        raise ValueError(model_kind)

    X_tr, y_tr = train[feats].values, train["Class"].values
    X_te, y_te = test[feats].values, test["Class"].values
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    scale = n_neg / max(n_pos, 1)

    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale,
        eval_metric="logloss",
        random_state=SEED,
        verbosity=0,
    )
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1], y_te


def metrics_at_topk(scores: np.ndarray, y: np.ndarray, k_pct: float) -> dict:
    """Score the top-k%-ranked predictions (k_pct is fraction of stream alerted)."""
    n = len(scores)
    k = max(1, int(round(n * k_pct / 100.0)))
    # Top-k by descending score
    topk_idx = np.argpartition(-scores, k - 1)[:k]
    yhat = np.zeros(n, dtype=int)
    yhat[topk_idx] = 1
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    n_pos = max(int(y.sum()), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / n_pos
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_pos": int(y.sum()),
        "alert_rate": k / n,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def compute_dollar_at_k(m: dict, model_kind: str) -> tuple[float, float]:
    """$/dec and $/fraud_caught at this operating point (for canonical L=$100)."""
    L = 100.0
    n = int(round(m["tp"] / max(m["alert_rate"], 1e-9)))
    cost = m["fp"] * C_INV + m["fn"] * L
    c_sense = (C_SENSE if model_kind in ("24full", "4sensor") else 0.0) * n
    total = cost + c_sense
    per_dec = total / n
    per_caught = total / max(m["tp"], 1)
    return per_dec, per_caught


# ------------------------------------------------------------------ bootstrap
def paired_recall_gap_ci(
    scores_a: np.ndarray, scores_b: np.ndarray, y: np.ndarray, k_pct: float,
    n_boot: int, seed: int,
) -> tuple[float, float, float, float, float]:
    """Paired bootstrap of (recall_a, recall_b, F1_a, F1_b, gap) at fixed K.

    Returns (mean_recall_a, mean_recall_b, mean_f1_a, mean_f1_b, mean_gap, ci_low, ci_high).
    """
    n = len(y)
    k = max(1, int(round(n * k_pct / 100.0)))
    rng = np.random.default_rng(seed)
    rec_a = np.empty(n_boot)
    rec_b = np.empty(n_boot)
    f1_a = np.empty(n_boot)
    f1_b = np.empty(n_boot)
    n_pos = int(y.sum())
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        sa = scores_a[idx]
        sb = scores_b[idx]
        # Top-k by descending score within the bootstrap resample
        topa = np.argpartition(-sa, min(k - 1, len(sa) - 1))[:k]
        topb = np.argpartition(-sb, min(k - 1, len(sb) - 1))[:k]
        yhat_a = np.zeros_like(y_b)
        yhat_b = np.zeros_like(y_b)
        yhat_a[topa] = 1
        yhat_b[topb] = 1
        pos_b = max(int(y_b.sum()), 1)
        tpa = int(((yhat_a == 1) & (y_b == 1)).sum())
        tpb = int(((yhat_b == 1) & (y_b == 1)).sum())
        fpa = int(((yhat_a == 1) & (y_b == 0)).sum())
        fpb = int(((yhat_b == 1) & (y_b == 0)).sum())
        rec_a[b] = tpa / pos_b
        rec_b[b] = tpb / pos_b
        pa = tpa / max(tpa + fpa, 1)
        pb = tpb / max(tpb + fpb, 1)
        f1_a[b] = 2 * pa * rec_a[b] / max(pa + rec_a[b], 1e-9)
        f1_b[b] = 2 * pb * rec_b[b] / max(pb + rec_b[b], 1e-9)
    gap = rec_a - rec_b
    return (
        float(rec_a.mean()), float(rec_b.mean()),
        float(f1_a.mean()), float(f1_b.mean()),
        float(gap.mean()), float(np.percentile(gap, 2.5)), float(np.percentile(gap, 97.5)),
    )


# ------------------------------------------------------------------ main
def main() -> None:
    print("[p8_alert_volume_pareto] starting")
    train, test = load_data()
    print(f"  train: {len(train)} rows, {train['Class'].sum()} positives ({train['Class'].mean()*100:.3f}%)")
    print(f"  test:  {len(test)} rows, {test['Class'].sum()} positives ({test['Class'].mean()*100:.3f}%)")

    # Fit each tree once
    print("  fitting trees...")
    scores = {}
    for m in ("20raw", "24full", "4sensor"):
        s, y = fit_and_score(m, train, test)
        scores[m] = s
        # Sanity print
        for K in [0.5, 1.0]:
            met = metrics_at_topk(s, y, K)
            print(f"    {m} @ K={K}%: recall={met['recall']:.4f} prec={met['precision']:.4f} F1={met['f1']:.4f} tp={met['tp']} fp={met['fp']}")
    y = test["Class"].values

    # Per-cell metrics table
    rows = []
    for K in K_BUDGETS:
        for m in ("20raw", "24full", "4sensor"):
            met = metrics_at_topk(scores[m], y, K)
            dpd, dpc = compute_dollar_at_k(met, m)
            rows.append({
                "model": m,
                "k_pct": K,
                "alert_rate": met["alert_rate"],
                "tp": met["tp"],
                "fp": met["fp"],
                "fn": met["fn"],
                "n_pos": met["n_pos"],
                "precision": met["precision"],
                "recall": met["recall"],
                "f1": met["f1"],
                "usd_per_dec": dpd,
                "usd_per_caught": dpc,
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "p8_alert_volume.tsv", sep="\t", index=False)
    print(f"  wrote {OUT / 'p8_alert_volume.tsv'} ({len(df)} rows)")

    # Paired-bootstrap CI on (24full - 20raw), (24full - 4sensor), (20raw - 4sensor) at each K
    pairs = [
        ("24full", "20raw"),    # sensor increment
        ("20raw", "4sensor"),   # raw vs sensor-only surrogate
        ("24full", "4sensor"),  # full vs sensor-only
    ]
    boot_rows = []
    for K in K_BUDGETS:
        for a, b in pairs:
            ma, mb, fa, fb, gap_mean, ci_lo, ci_hi = paired_recall_gap_ci(
                scores[a], scores[b], y, K, n_boot=N_BOOT,
                seed=SEED + int(K * 100) + len(a) * 13,
            )
            excl_zero = "YES" if (ci_lo > 0 or ci_hi < 0) else "no"
            winner = a if ci_lo > 0 else (b if ci_hi < 0 else "TIE")
            boot_rows.append({
                "K_pct": K,
                "model_a": a,
                "model_b": b,
                "recall_a": ma,
                "recall_b": mb,
                "f1_a": fa,
                "f1_b": fb,
                "delta_recall": gap_mean,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "ci_excludes_zero": excl_zero,
                "winner": winner,
            })
            print(
                f"  K={K}% {a}−{b}: Δ-recall={gap_mean:+.4f} "
                f"CI=[{ci_lo:+.4f},{ci_hi:+.4f}] excl_zero={excl_zero} winner={winner}"
            )
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(OUT / "p8_alert_volume_boot.tsv", sep="\t", index=False)
    print(f"  wrote {OUT / 'p8_alert_volume_boot.tsv'} ({len(boot_df)} rows)")

    # Summary JSON
    summary = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_pos_test": int(y.sum()),
        "pos_rate_test": float(y.mean()),
        "K_budgets_pct": K_BUDGETS,
        "c_inv_usd": C_INV,
        "c_sense_usd_per_row": C_SENSE,
        "n_boot": N_BOOT,
        "seed": SEED,
        "headline": {
            "operating_point_K_0.5_pct": {
                "20raw": metrics_at_topk(scores["20raw"], y, 0.5),
                "24full": metrics_at_topk(scores["24full"], y, 0.5),
                "4sensor": metrics_at_topk(scores["4sensor"], y, 0.5),
            },
            "operating_point_K_1_pct": {
                "20raw": metrics_at_topk(scores["20raw"], y, 1.0),
                "24full": metrics_at_topk(scores["24full"], y, 1.0),
                "4sensor": metrics_at_topk(scores["4sensor"], y, 1.0),
            },
        },
        "dominance_switch_K_pct": _dominance_switch(boot_df),
        "total_bootstrap_cells_excl_zero": int((boot_df["ci_excludes_zero"] == "YES").sum()),
        "winners_by_K": _winners_table(boot_df),
    }
    with open(OUT / "p8_alert_volume_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"  wrote {OUT / 'p8_alert_volume_summary.json'}")

    # Plot
    _plot_pareto(df, boot_df, scores, y)


def _dominance_switch(boot_df: pd.DataFrame) -> dict:
    """Find the K at which 24full starts strictly dominating 20raw on recall gap."""
    sub = boot_df[(boot_df["model_a"] == "24full") & (boot_df["model_b"] == "20raw")].sort_values("K_pct")
    first_strictly_dominant = sub[sub["ci_excludes_zero"] == "YES"]
    if len(first_strictly_dominant) == 0:
        return {"24full_strictly_dominates_20raw_at_no_K": True}
    first_row = first_strictly_dominant.iloc[0]
    return {
        "first_strict_K_pct": float(first_row["K_pct"]),
        "first_strict_delta": float(first_row["delta_recall"]),
        "first_strict_ci_lo": float(first_row["ci_lo"]),
        "first_strict_ci_hi": float(first_row["ci_hi"]),
    }


def _winners_table(boot_df: pd.DataFrame) -> dict:
    out = {}
    for K in K_BUDGETS:
        sub = boot_df[boot_df["K_pct"] == K]
        out[f"{K}%"] = list(zip(sub["model_a"].tolist(), sub["winner"].tolist()))
    return out


def _plot_pareto(df: pd.DataFrame, boot_df: pd.DataFrame, scores: dict, y: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: recall vs K
    ax = axes[0]
    for m, color in [("20raw", "#1f77b4"), ("24full", "#d62728"), ("4sensor", "#7f7f7f")]:
        sub = df[df["model"] == m].sort_values("k_pct")
        ax.plot(sub["k_pct"], sub["recall"], marker="o", color=color, label=m)
    ax.set_xscale("log")
    ax.set_xlabel("Review budget K (%)")
    ax.set_ylabel("Recall")
    ax.set_title("Recall vs alert-volume budget")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: F1 vs K
    ax = axes[1]
    for m, color in [("20raw", "#1f77b4"), ("24full", "#d62728"), ("4sensor", "#7f7f7f")]:
        sub = df[df["model"] == m].sort_values("k_pct")
        ax.plot(sub["k_pct"], sub["f1"], marker="o", color=color, label=m)
    ax.set_xscale("log")
    ax.set_xlabel("Review budget K (%)")
    ax.set_ylabel("F1")
    ax.set_title("F1 vs alert-volume budget")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: paired bootstrap Δ-recall(24full-20raw) with CI bars
    ax = axes[2]
    sub = boot_df[(boot_df["model_a"] == "24full") & (boot_df["model_b"] == "20raw")].sort_values("K_pct")
    xs = sub["K_pct"].values
    ys = sub["delta_recall"].values
    yerr_lo = ys - sub["ci_lo"].values
    yerr_hi = sub["ci_hi"].values - ys
    ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], marker="o", color="#d62728", capsize=3)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Review budget K (%)")
    ax.set_ylabel("Δ-recall (24full − 20raw)")
    ax.set_title("Paired bootstrap CI on recall gap\n24full vs 20raw")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG / "p8_alert_volume.png", dpi=150)
    plt.savefig(FIG / "p8_alert_volume.pdf")
    plt.close()
    print(f"  wrote {FIG / 'p8_alert_volume.png'}")


if __name__ == "__main__":
    main()