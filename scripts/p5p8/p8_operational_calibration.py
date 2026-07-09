#!/usr/bin/env python3
"""P8 JOB A (iter 60): Operational calibration gap at alert-volume budgets.

iter-24 #31 measured GLOBAL reliability diagrams (over all 10k test rows,
decile-binned) and reported max absolute calibration drift per model;
iter-28 #35 measured the cost-optimal THRESHOLD; iter-56 #66 measured the
alert-volume-constrained Pareto frontier on RECALL. None answers the
operational question fraud-ops leads actually ask on a daily basis:

  "If my analysts only look at the top-K alerts (because that's all I can
   staff), is the model's predicted-probability AMONG those alerts
   calibrated to the OBSERVED positive rate?"

This iter closes that gap. For each K ∈ {0.25, 0.50, 1.00, 2.00, 5.00}%
(every K the iter-56 budget-sweep also visits), and each tree
∈ {XGB-20raw, XGB-24full, XGB-4sensor}, we compute:

  - mean_predicted_topK: mean predicted probability among the top-K alerts
  - observed_pos_rate_topK: actual positive rate among the top-K alerts
  - calibration_gap = mean_predicted_topK - observed_pos_rate_topK
  - brier_topK: mean squared error among the top-K alerts (predicted - actual)^2
  - ece_topK: weighted absolute calibration gap in 10 prediction-bins within top-K

then run a paired bootstrap (B=400, percentile, seed 20260704) on
(delta_gap, delta_brier) between every pair of trees, separately at each K.

This is the operationally-meaningful calibration metric: an alert-volume
K=1% means only the top-100 alerts are reviewed, and the calibration gap
in THOSE 100 alerts is what determines whether the model's predicted
probability can be trusted as a triage signal.

Outputs
-------
experiments/results/p5p8/p8_operational_calibration.tsv           (15 rows: 5 K × 3 trees)
experiments/results/p5p8/p8_operational_calibration_boot.tsv      (30 paired-bootstrap rows: 5 K × 3 pairs)
experiments/results/p5p8/p8_operational_calibration_summary.json
experiments/results/p5p8/figures/p8_operational_calibration.{png,pdf}

Stdlib + numpy + pandas + xgboost + matplotlib. <=290 lines.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

# ------------------------------------------------------------------ paths
ROOT = Path("/home/claude/tinker-rl-lab-minimax")
OUT = ROOT / "experiments" / "results" / "p5p8"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SEED = 20260704
N_BOOT = 400
K_BUDGETS = [0.25, 0.50, 1.00, 2.00, 5.00]   # alert-volume budgets (% of stream)
RAW_FEATURES = [f"V{i}" for i in range(1, 21)]
SENSOR_FEATURES = ["V_mean", "V_std", "V_max", "V_min"]
TREE_KWARGS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    eval_metric="logloss", random_state=SEED, verbosity=0,
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(ROOT / "fraud_data.csv")
    test = pd.read_csv(ROOT / "test_data.csv")
    return train, test


def fit_and_score(model_kind: str, train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fit one tree; return (test_scores, test_y)."""
    feats = RAW_FEATURES if model_kind == "20raw" else (
        RAW_FEATURES + SENSOR_FEATURES if model_kind == "24full" else SENSOR_FEATURES
    )
    X_tr, y_tr = train[feats].values, train["Class"].values
    X_te, y_te = test[feats].values, test["Class"].values
    n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
    clf = xgb.XGBClassifier(
        **TREE_KWARGS, scale_pos_weight=n_neg / max(n_pos, 1)
    )
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1], y_te


def topk_metrics(scores: np.ndarray, y: np.ndarray, k_pct: float) -> dict:
    """Compute the 5 operational-calibration metrics on the top-K alerts."""
    n = len(scores)
    k = max(1, int(round(n * k_pct / 100.0)))
    # argpartition is O(n); safer than argsort for n=10k
    topk_idx = np.argpartition(-scores, k - 1)[:k]
    s_top = scores[topk_idx]
    y_top = y[topk_idx]
    n_pos_top = int(y_top.sum())
    obs_pos_rate = n_pos_top / k
    mean_pred = float(s_top.mean())
    gap = mean_pred - obs_pos_rate
    brier_topk = float(((s_top - y_top) ** 2).mean())
    # ECE_topK with 10 quantile bins over the top-K predictions
    if k >= 10:
        qs = np.quantile(s_top, np.linspace(0.0, 1.0, 11))
        qs[0] -= 1e-9; qs[-1] += 1e-9
        ece = 0.0
        for i in range(10):
            mask = (s_top >= qs[i]) & (s_top < qs[i + 1]) if i < 9 else (s_top >= qs[i]) & (s_top <= qs[i + 1])
            n_bin = int(mask.sum())
            if n_bin == 0:
                continue
            mean_pred_bin = float(s_top[mask].mean())
            obs_pos_bin = float(y_top[mask].mean())
            ece += (n_bin / k) * abs(mean_pred_bin - obs_pos_bin)
    else:
        ece = abs(gap)
    return {
        "k": k,
        "k_pct": k_pct,
        "n_top": k,
        "n_pos_top": n_pos_top,
        "obs_pos_rate": obs_pos_rate,
        "mean_pred": mean_pred,
        "calibration_gap": gap,
        "abs_calibration_gap": abs(gap),
        "brier_topk": brier_topk,
        "ece_topk": float(ece),
    }


def paired_calibration_bootstrap(
    scores_a: np.ndarray, scores_b: np.ndarray, y: np.ndarray,
    k_pct: float, n_boot: int, seed: int,
) -> tuple[float, float, float, float, float, float, float]:
    """Paired bootstrap of the calibration gap (a - b) at fixed K.

    Returns (gap_a_mean, gap_b_mean, brier_a_mean, brier_b_mean,
             gap_delta_mean, gap_ci_lo, gap_ci_high).
    """
    n = len(y)
    k = max(1, int(round(n * k_pct / 100.0)))
    rng = np.random.default_rng(seed)
    gap_a = np.empty(n_boot)
    gap_b = np.empty(n_boot)
    brier_a = np.empty(n_boot)
    brier_b = np.empty(n_boot)
    for j in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sa = scores_a[idx]; sb = scores_b[idx]; yb = y[idx]
        # Top-k within the resample
        ta = np.argpartition(-sa, min(k - 1, len(sa) - 1))[:k]
        tb = np.argpartition(-sb, min(k - 1, len(sb) - 1))[:k]
        sa_t = sa[ta]; yb_a = yb[ta]
        sb_t = sb[tb]; yb_b = yb[tb]
        gap_a[j] = float(sa_t.mean() - yb_a.mean())
        gap_b[j] = float(sb_t.mean() - yb_b.mean())
        brier_a[j] = float(((sa_t - yb_a) ** 2).mean())
        brier_b[j] = float(((sb_t - yb_b) ** 2).mean())
    delta = gap_a - gap_b
    return (
        float(gap_a.mean()), float(gap_b.mean()),
        float(brier_a.mean()), float(brier_b.mean()),
        float(delta.mean()),
        float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5)),
    )


def main() -> None:
    print("[p8_operational_calibration] starting")
    train, test = load_data()
    print(f"  train: {len(train)} rows, {int(train['Class'].sum())} positives")
    print(f"  test:  {len(test)} rows, {int(test['Class'].sum())} positives")

    # Fit each tree once
    scores: dict[str, np.ndarray] = {}
    for kind in ("20raw", "24full", "4sensor"):
        s, y = fit_and_score(kind, train, test)
        scores[kind] = s
        print(f"  fitted {kind}: AUC proxy at top-1% = {topk_metrics(s, y, 1.00)['obs_pos_rate']:.4f}")

    # PART A: per-K, per-tree operational-calibration table
    rows = []
    for k_pct in K_BUDGETS:
        for kind in ("20raw", "24full", "4sensor"):
            m = topk_metrics(scores[kind], y, k_pct)
            rows.append({"K_pct": k_pct, "model": f"XGB-{kind}", **m})
    df_a = pd.DataFrame(rows)
    df_a.to_csv(OUT / "p8_operational_calibration.tsv", sep="\t", index=False)
    print(f"[p8_operational_calibration] wrote {OUT / 'p8_operational_calibration.tsv'} ({len(df_a)} rows)")

    # PART B: paired bootstrap CIs on the (gap_a - gap_b) calibration gap
    boot_rows = []
    pair_defs = [
        ("24full", "20raw", "XGB-24full", "XGB-20raw"),
        ("20raw", "4sensor", "XGB-20raw", "XGB-4sensor"),
        ("24full", "4sensor", "XGB-24full", "XGB-4sensor"),
    ]
    for k_pct in K_BUDGETS:
        for ka, kb, na, nb in pair_defs:
            ga, gb, bra, brb, dmean, dlo, dhi = paired_calibration_bootstrap(
                scores[ka], scores[kb], y, k_pct, N_BOOT, SEED + int(k_pct * 100),
            )
            excl_zero = "YES" if (dlo > 0 or dhi < 0) else "no"
            direction = "pos" if dmean > 0 else "neg"
            boot_rows.append({
                "K_pct": k_pct, "pair": f"{na}-{nb}",
                "gap_a": ga, "gap_b": gb,
                "brier_a": bra, "brier_b": brb,
                "delta_gap": dmean, "ci_lo": dlo, "ci_hi": dhi,
                "excludes_zero": excl_zero, "direction": direction,
            })
    df_b = pd.DataFrame(boot_rows)
    df_b.to_csv(OUT / "p8_operational_calibration_boot.tsv", sep="\t", index=False)
    print(f"[p8_operational_calibration] wrote {OUT / 'p8_operational_calibration_boot.tsv'} ({len(df_b)} rows)")

    # PART C: headline summary
    headline = {
        "n_test": int(len(y)),
        "n_pos_test": int(y.sum()),
        "k_budgets_pct": K_BUDGETS,
        "models": {
            f"XGB-{kind}": {
                "auc_proxy_at_k1": topk_metrics(scores[kind], y, 1.0)["obs_pos_rate"],
            } for kind in ("20raw", "24full", "4sensor")
        },
        "headline": {
            "description": "per-K operational calibration gap; signed delta = (gap_a - gap_b)",
            "rows_a": int(len(df_a)),
            "rows_boot": int(len(df_b)),
            "n_boot": N_BOOT,
            "seed": SEED,
        },
        "excludes_zero_counts": {
            pair: int((df_b[df_b["pair"].str.contains(pair)]["excludes_zero"] == "YES").sum())
            for pair in ["24full-20raw", "20raw-4sensor", "24full-4sensor"]
        },
    }
    with (OUT / "p8_operational_calibration_summary.json").open("w") as fp:
        json.dump(headline, fp, indent=2)
    print(f"[p8_operational_calibration] wrote {OUT / 'p8_operational_calibration_summary.json'}")

    # Figure: calibration gap vs K, three lines
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_gap = axes[0]
    ax_brier = axes[1]
    for kind, marker, color in (("20raw", "o", "tab:blue"), ("24full", "s", "tab:orange"), ("4sensor", "^", "tab:red")):
        sub = df_a[df_a["model"] == f"XGB-{kind}"]
        ax_gap.plot(sub["K_pct"], sub["calibration_gap"], marker=marker, color=color, label=f"XGB-{kind}")
        ax_brier.plot(sub["K_pct"], sub["brier_topk"], marker=marker, color=color, label=f"XGB-{kind}")
    ax_gap.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_gap.set_xscale("log"); ax_gap.set_xlabel("Alert-volume K (% of stream, log scale)")
    ax_gap.set_ylabel("calibration gap (mean_pred - obs_pos_rate)")
    ax_gap.set_title("Operational calibration gap at alert-volume K")
    ax_gap.legend(fontsize=8); ax_gap.grid(True, alpha=0.3)
    ax_brier.set_xscale("log"); ax_brier.set_xlabel("Alert-volume K (% of stream, log scale)")
    ax_brier.set_ylabel("Brier score among top-K alerts")
    ax_brier.set_title("Top-K Brier at alert-volume K")
    ax_brier.legend(fontsize=8); ax_brier.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "p8_operational_calibration.png", dpi=130)
    fig.savefig(FIG / "p8_operational_calibration.pdf")
    plt.close(fig)
    print(f"[p8_operational_calibration] wrote figure {FIG / 'p8_operational_calibration.png'}")

    # PART D: print headline
    print("\n[p8_operational_calibration] headline:")
    for k_pct in K_BUDGETS:
        sub = df_b[df_b["K_pct"] == k_pct]
        for _, r in sub.iterrows():
            print(f"  K={k_pct:5.2f}%  {r['pair']:35s}  delta_gap={r['delta_gap']:+.4f}  CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  excl0={r['excludes_zero']}")
    print("[p8_operational_calibration] done")


if __name__ == "__main__":
    main()