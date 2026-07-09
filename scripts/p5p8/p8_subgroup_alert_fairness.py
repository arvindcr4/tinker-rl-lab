#!/usr/bin/env python3
"""P8 JOB A (iter 64): subgroup-stratified alert-distribution fairness.

Fresh vein (not in the prior 75 P8 rows). Iter-28 (#35) measured global
cost-optimal threshold; iter-49 (#55) measured threshold transfer under
prevalence shift; iter-52 measured absolute regret; iter-56 (#66) measured
alert-volume Pareto on the FULL test stream; iter-60 (#70) measured
operational calibration gap.

None of them asks the empirically-loaded fairness question fraud-ops
leads raise after seeing a global K=2% alert budget: "is the global
budget distributed proportionally to the per-bucket positive rate, or is
one bucket getting starved?"

This iter closes that gap. For two stratification axes (V_mean quintile —
the LLM-as-sensor aggregate itself; V14 quintile — the top-importance raw
feature), and each tree ∈ {XGB-20raw, XGB-24full, XGB-4sensor}, we measure
on the test split:

  Per-bin at global K=2% (matching iter-56 dominance switch):
    - n_b  : number of rows in the bin
    - p_b  : positive rate in the bin
    - n_alert_b : number of top-2% alerts falling in the bin (lift-share)
    - alert_rate_b = n_alert_b / n_b   (fraction of bin alerted)
    - recall_b   = TPs_in_bin / positives_in_bin  (positive coverage)
    - precision_b = TPs_in_bin / n_alert_b
    - lift_b    = (precision_b) / p_base   (above/below average)

  Per-tree across-bin heterogeneity:
    - Gini(n_alert_b / sum(alert)) : alert-distribution concentration
    - Std of (recall_b)            : positive-coverage dispersion
    - Std of (precision_b)         : precision dispersion
    - Sum of |alert_rate_b - p_b|  : L1 alerting-bias from positive-density

  Paired bootstrap (B=400, percentile, seed 20260704) on each metric.

Hypotheses tested:
  H1: XGB-24full has LOWER alert-distribution Gini than XGB-20raw (the
      sensor's population-aggregate features reduce per-subgroup bias).
  H2: XGB-4sensor has the HIGHEST Gini (sensor alone is rank-poor on
      subgroups — it captures only the LLM aggregate).
  H3: The same pattern appears on the V14-quintile axis.

Outputs (5 files in experiments/results/p5p8/):
  p8_subgroup_fairness.tsv         (30 rows: 2 strata × 5 bins × 3 trees)
  p8_subgroup_fairness_boot.tsv    (24 paired-bootstrap rows)
  p8_subgroup_fairness_summary.json
  figures/p8_subgroup_fairness.{png,pdf}

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
K_GLOBAL_PCT = 2.0
N_BINS = 5
RAW_FEATURES = [f"V{i}" for i in range(1, 21)]
SENSOR_FEATURES = ["V_mean", "V_std", "V_max", "V_min"]
STRATA = {
    "V_mean": ("V_mean",),                     # LLM-as-sensor aggregate
    "V14":    ("V14",),                         # top-importance raw feature
}
TREE_KWARGS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    eval_metric="logloss", random_state=SEED, verbosity=0,
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(ROOT / "fraud_data.csv")
    test = pd.read_csv(ROOT / "test_data.csv")
    return train, test


def fit_and_score(model_kind: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    feats = (RAW_FEATURES if model_kind == "20raw"
             else (RAW_FEATURES + SENSOR_FEATURES if model_kind == "24full"
                   else SENSOR_FEATURES))
    X_tr, y_tr = train[feats].values, train["Class"].values
    X_te = test[feats].values
    n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
    clf = xgb.XGBClassifier(
        **TREE_KWARGS, scale_pos_weight=n_neg / max(n_pos, 1),
    )
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative array (0 = perfectly equal)."""
    arr = np.asarray(values, dtype=float)
    if arr.sum() <= 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(arr) / (n * arr.sum()))


def stratified_metrics(
    scores: np.ndarray, y: np.ndarray, stratum: np.ndarray,
    k_pct: float, n_bins: int,
) -> list[dict]:
    """For one tree, compute per-bin alert+fairness metrics under global K=2%."""
    n = len(scores)
    k = max(1, int(round(n * k_pct / 100.0)))
    # Top-K global indices
    top_idx = np.argpartition(-scores, k - 1)[:k]
    alerted_mask = np.zeros(n, dtype=bool)
    alerted_mask[top_idx] = True
    # Bin stratum into n_bins quintiles (by quantile)
    bins = np.quantile(stratum, np.linspace(0.0, 1.0, n_bins + 1))
    bins[0] -= 1e-9; bins[-1] += 1e-9
    bin_id = np.digitize(stratum, bins[1:-1])  # 0..n_bins-1
    rows = []
    for b in range(n_bins):
        mask = bin_id == b
        n_b = int(mask.sum())
        n_pos_b = int(y[mask].sum())
        p_b = (n_pos_b / n_b) if n_b else 0.0
        n_alert_b = int((mask & alerted_mask).sum())
        alert_rate_b = (n_alert_b / n_b) if n_b else 0.0
        recall_b = ((y[mask] & alerted_mask[mask]).sum() / n_pos_b) if n_pos_b else 0.0
        precision_b = ((y[mask] & alerted_mask[mask]).sum() / n_alert_b) if n_alert_b else 0.0
        # lift = precision_b / p_base_test (overall positive rate, not p_b)
        p_base = float(y.sum()) / n
        lift_b = (precision_b / p_base) if (precision_b and p_base) else 0.0
        rows.append({
            "bin": b,
            "n_bin": n_b,
            "n_pos_bin": n_pos_b,
            "p_bin": p_b,
            "n_alert_bin": n_alert_b,
            "alert_rate_bin": alert_rate_b,
            "recall_bin": recall_b,
            "precision_bin": precision_b,
            "lift_bin": lift_b,
        })
    return rows


def heterogeneity_summary(per_bin: list[dict]) -> dict:
    """Aggregate per-bin rows into tree-level heterogeneity scalars."""
    arr_alerts = np.array([r["n_alert_bin"] for r in per_bin], dtype=float)
    arr_recall = np.array([r["recall_bin"] for r in per_bin], dtype=float)
    arr_prec = np.array([r["precision_bin"] for r in per_bin], dtype=float)
    arr_prate = np.array([r["p_bin"] for r in per_bin], dtype=float)
    arr_alert_rate = np.array([r["alert_rate_bin"] for r in per_bin], dtype=float)
    total_alerts = arr_alerts.sum()
    actual_pos_rate = float(np.nansum([
        r["n_pos_bin"] for r in per_bin
    ]) / max(1, np.nansum([r["n_bin"] for r in per_bin])))
    # Expected alert rate per bin if perfectly proportional to n_bin
    n_bins = len(per_bin)
    expected_alert_rate = np.array([
        sum(r["n_bin"] for r in per_bin) / (
            sum(rr["n_bin"] for rr in per_bin) * n_bins) for _ in range(n_bins)
    ])
    # L1 alerting-bias from population baseline (uniform across bins)
    l1_bias = float(np.sum(np.abs(arr_alert_rate - expected_alert_rate)))
    return {
        "gini_alerts": gini(arr_alerts),
        "gini_recall": gini(arr_recall),
        "std_recall": float(np.std(arr_recall, ddof=0)) if n_bins > 1 else 0.0,
        "std_precision": float(np.std(arr_prec, ddof=0)) if n_bins > 1 else 0.0,
        "l1_alert_bias_vs_uniform": l1_bias,
    }


def paired_heterogeneity_bootstrap(
    scores_a: np.ndarray, scores_b: np.ndarray,
    y: np.ndarray, stratum: np.ndarray,
    k_pct: float, n_bins: int, n_boot: int, seed: int,
) -> list[dict]:
    """Paired bootstrap on the heterogeneity-summary scalars (4 metrics).

    Returns a list of dicts (one per metric) with mean / CI / excludes_zero.
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    metrics = ("gini_alerts", "std_recall", "std_precision", "l1_alert_bias_vs_uniform")
    arr_a = {m: np.empty(n_boot) for m in metrics}
    arr_b = {m: np.empty(n_boot) for m in metrics}
    for j in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sa = scores_a[idx]; sb = scores_b[idx]
        ya = y[idx]; stratb = stratum[idx]
        per_a = stratified_metrics(sa, ya, stratb, k_pct, n_bins)
        per_b = stratified_metrics(sb, ya, stratb, k_pct, n_bins)
        sum_a = heterogeneity_summary(per_a)
        sum_b = heterogeneity_summary(per_b)
        for m in metrics:
            arr_a[m][j] = sum_a[m]
            arr_b[m][j] = sum_b[m]
    out = []
    for m in metrics:
        delta = arr_a[m] - arr_b[m]
        dmean = float(delta.mean())
        dlo = float(np.percentile(delta, 2.5))
        dhi = float(np.percentile(delta, 97.5))
        excl = "YES" if (dlo > 0 or dhi < 0) else "no"
        out.append({
            "metric": m,
            "mean_a": float(arr_a[m].mean()),
            "mean_b": float(arr_b[m].mean()),
            "delta": dmean,
            "ci_lo": dlo,
            "ci_hi": dhi,
            "excludes_zero": excl,
        })
    return out


def main() -> None:
    print("[p8_subgroup_fairness] starting")
    train, test = load_data()
    print(f"  train: {len(train)} rows, {int(train['Class'].sum())} positives")
    print(f"  test:  {len(test)} rows, {int(test['Class'].sum())} positives")

    # Fit each tree once
    scores: dict[str, np.ndarray] = {}
    y = test["Class"].values
    for kind in ("20raw", "24full", "4sensor"):
        scores[kind] = fit_and_score(kind, train, test)
        print(f"  fitted {kind}: AUC proxy at top-1% recall = "
              f"{stratified_metrics(scores[kind], y, test['V_mean'].values, 1.0, 5)[0]['recall_bin']:.4f}")

    # PART A: per-strata × per-bin × per-tree table
    rows = []
    for strat_label, (col,) in STRATA.items():
        stratum = test[col].values
        for kind in ("20raw", "24full", "4sensor"):
            per_bin = stratified_metrics(scores[kind], y, stratum, K_GLOBAL_PCT, N_BINS)
            for r in per_bin:
                rows.append({
                    "stratum": strat_label,
                    "model": f"XGB-{kind}",
                    **r,
                })
            # also add the heterogeneity-scalar row
            hs = heterogeneity_summary(per_bin)
            rows.append({
                "stratum": strat_label,
                "model": f"XGB-{kind}",
                "bin": -1, "n_bin": -1, "n_pos_bin": -1, "p_bin": -1.0,
                "n_alert_bin": -1, "alert_rate_bin": -1.0, "recall_bin": -1.0,
                "precision_bin": -1.0, "lift_bin": -1.0,
                "gini_alerts": hs["gini_alerts"],
                "std_recall": hs["std_recall"],
                "std_precision": hs["std_precision"],
                "l1_alert_bias_vs_uniform": hs["l1_alert_bias_vs_uniform"],
            })
    df_a = pd.DataFrame(rows)
    df_a.to_csv(OUT / "p8_subgroup_fairness.tsv", sep="\t", index=False)
    print(f"[p8_subgroup_fairness] wrote {OUT / 'p8_subgroup_fairness.tsv'} ({len(df_a)} rows)")

    # PART B: paired bootstrap on heterogeneity scalars, two strata × two pairs
    boot_rows = []
    pair_defs = [
        ("24full", "20raw", "XGB-24full", "XGB-20raw"),
        ("20raw", "4sensor", "XGB-20raw", "XGB-4sensor"),
        ("24full", "4sensor", "XGB-24full", "XGB-4sensor"),
    ]
    for strat_label, (col,) in STRATA.items():
        stratum = test[col].values
        for ka, kb, na, nb in pair_defs:
            rows_b = paired_heterogeneity_bootstrap(
                scores[ka], scores[kb], y, stratum,
                K_GLOBAL_PCT, N_BINS, N_BOOT, SEED + hash((strat_label, ka, kb)) % 100000,
            )
            for rb in rows_b:
                boot_rows.append({
                    "stratum": strat_label,
                    "pair": f"{na}-{nb}",
                    **rb,
                })
    df_b = pd.DataFrame(boot_rows)
    df_b.to_csv(OUT / "p8_subgroup_fairness_boot.tsv", sep="\t", index=False)
    print(f"[p8_subgroup_fairness] wrote {OUT / 'p8_subgroup_fairness_boot.tsv'} ({len(df_b)} rows)")

    # PART C: summary
    headline = {
        "k_global_pct": K_GLOBAL_PCT,
        "n_bins": N_BINS,
        "n_boot": N_BOOT,
        "seed": SEED,
        "rows_a": int(len(df_a)),
        "rows_boot": int(len(df_b)),
        "excludes_zero_counts": {
            pair: int((df_b[df_b["pair"] == pair]["excludes_zero"] == "YES").sum())
            for _, _, _, pair in [(k, k, n, f"{n}-{m}") for k, m, n, _ in pair_defs]
        },
        "headline": "Tree-level per-stratum alert-distribution heterogeneity",
    }
    with (OUT / "p8_subgroup_fairness_summary.json").open("w") as fp:
        json.dump(headline, fp, indent=2)
    print(f"[p8_subgroup_fairness] wrote {OUT / 'p8_subgroup_fairness_summary.json'}")

    # Figure: Gini vs stratum (V_mean, V14), three bars each, both pairs
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for ax, stratum in zip(axes, STRATA.keys()):
        sub = df_b[df_b["stratum"] == stratum]
        pair_names = sub["pair"].unique()
        metric_to_use = "gini_alerts"
        widths = np.arange(len(pair_names))
        means = []; los = []; his = []
        for pn in pair_names:
            sel = sub[(sub["pair"] == pn) & (sub["metric"] == metric_to_use)]
            means.append(sel["delta"].iloc[0])
            los.append(sel["delta"].iloc[0] - sel["ci_lo"].iloc[0])
            his.append(sel["ci_hi"].iloc[0] - sel["delta"].iloc[0])
        ax.bar(widths, means, yerr=[los, his], capsize=5, color=["#5a8", "#a55", "#57c"][:len(pair_names)])
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(widths)
        ax.set_xticklabels(pair_names, rotation=20, ha="right", fontsize=8)
        ax.set_title(f"Δ-Gini(alerts) at K={K_GLOBAL_PCT}% on {stratum} quintile", fontsize=10)
        ax.set_ylabel("(a − b) Gini(alerts)")
    fig.tight_layout()
    fig.savefig(FIG / "p8_subgroup_fairness.png", dpi=130)
    fig.savefig(FIG / "p8_subgroup_fairness.pdf")
    plt.close(fig)
    print(f"[p8_subgroup_fairness] wrote figures/p8_subgroup_fairness.{{png,pdf}}")
    print("[p8_subgroup_fairness] done")


if __name__ == "__main__":
    main()
