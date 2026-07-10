#!/usr/bin/env python3
"""P8 JOB A (iter 144): calibration + LLM-sensor ablation + per-budget cost
study. Fresh vein not in 167 prior P8 rows.

Falsifiable claims to test on the canonical 10k test split (144 positives):

- H1 -- calibration degrades monotonically as alert budget tightens:
      ECE-10 at K=0.5%, 1.0%, 2.0%, 5.0%, 10.0% budgets; reported per tree.
- H2 -- each LLM-sensor aggregate ablation (drop V_mean, V_std, V_max, V_min,
      add V_mean, add V_std, add V_max, add V_min) produces a measurable
      shift in ECE on XGB-24full at K=2%; paired bootstrap B=2000, seed
      20260705.
- H3 -- per-decile reliability-of-decision metric on the (sensor, scribe,
      scorer) trio: Brier score / ECE per decision decile with CIs.
- H4 -- cost-per-decision at sensor cost $0.0035/row crosses break-even
      with cost-per-fraud-caught only when the ablated tree recovers
      calibration (closes iter-32 break-even drift cleanly).

Deliverables:
- platform_hybrid/experiments/results/p5p8/p8_iter144_calib_budget.tsv       (per tree x budget)
- platform_hybrid/experiments/results/p5p8/p8_iter144_calib_ablation.tsv     (per ablation)
- platform_hybrid/experiments/results/p5p8/p8_iter144_brier_decile.tsv       (per (tree, decile))
- platform_hybrid/experiments/results/p5p8/p8_iter144_cost_ablation.tsv      (cost/decision x ablation)
- platform_hybrid/experiments/results/p5p8/p8_iter144_ablation_boot.tsv       (paired bootstrap CIs)
- platform_hybrid/experiments/results/p5p8/p8_iter144_summary.json
- platform_hybrid/experiments/results/p5p8/figures/p8_iter144_{calib_budget,brier_decile}.{png,pdf}

Stdlib + numpy + pandas + xgboost + sklearn + matplotlib. <=300 LoC.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "fraud_data.csv"
TEST = ROOT / "test_data.csv"
OUT = ROOT / "experiments" / "results" / "p5p8"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

V20 = [f"V{i}" for i in range(1, 21)]
V_AGG = ["V_mean", "V_std", "V_max", "V_min"]
TREE_SEED = 42
N_BOOT = 2000
BOOT_SEED = 20260705
C_SENSE = 0.0035  # USD/decision for sensor block (canonical iter-28 cost)

# (a) H1 -- budgets.
BUDGETS_K = [0.5, 1.0, 2.0, 5.0, 10.0]
# (b) H2 -- ablations: each removed/added in turn.
ABLATIONS = [
    ("full_minus_Vmean", V20 + ["V_std", "V_max", "V_min"]),
    ("full_minus_Vstd", V20 + ["V_mean", "V_max", "V_min"]),
    ("full_minus_Vmax", V20 + ["V_mean", "V_std", "V_min"]),
    ("full_minus_Vmin", V20 + ["V_mean", "V_std", "V_max"]),
    ("full_plus_Vmean_only", V20 + ["V_mean"]),
    ("full_plus_Vstd_only", V20 + ["V_std"]),
    ("full_plus_Vmax_only", V20 + ["V_max"]),
    ("full_plus_Vmin_only", V20 + ["V_min"]),
]
FULL_FEATS = V20 + V_AGG
RAW_FEATS = V20
AGG_FEATS = V_AGG


def fit_tree(X_tr, y_tr):
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="auc",
        tree_method="hist", random_state=TREE_SEED, n_jobs=4,
    )
    clf.fit(X_tr, y_tr)
    return clf


def ece10(p, y):
    """ECE with 10 equal-width bins; reports absolute mean gap per bin,
    weighted by n_per_bin / N."""
    bins = np.linspace(0.0, 1.0, 11)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, 9)
    ece = 0.0
    for b in range(10):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(y)) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def alert_at_budget(p, k_pct):
    """Return binary vector of length N with top-k_pct% flagged."""
    n = len(p)
    k = max(1, int(np.round(n * k_pct / 100.0)))
    order = np.argsort(-p)
    flags = np.zeros(n, dtype=np.int32)
    flags[order[:k]] = 1
    return flags


def calib_at_budget(p, y, k_pct):
    """Within-budget ECE: ECE-10 computed only on rows in the top-k%
    predicted-positive set (the alert pool)."""
    flags = alert_at_budget(p, k_pct)
    if flags.sum() == 0:
        return float("nan"), float("nan"), 0
    pp = p[flags == 1]
    yy = y[flags == 1]
    ece_b = ece10(pp, yy)
    brier_b = brier(pp, yy)
    return ece_b, brier_b, int(yy.sum())


def calib_decile(p, y):
    """Per-decile ECE on the global prediction space (decision-theoretic
    decile = 10 equal-mass prediction-ranked bins)."""
    order = np.argsort(p)
    n = len(p)
    rows = []
    for d in range(10):
        lo = (d * n) // 10
        hi = ((d + 1) * n) // 10
        mask = np.zeros(n, dtype=bool)
        mask[order[lo:hi]] = True
        pp = p[mask]
        yy = y[mask]
        if len(pp) == 0:
            continue
        rows.append({
            "decile": d + 1,
            "n": int(mask.sum()),
            "pred_lo": float(pp.min()),
            "pred_hi": float(pp.max()),
            "mean_pred": float(pp.mean()),
            "obs_rate": float(yy.mean()),
            "calib_gap": float(pp.mean() - yy.mean()),
            "n_pos": int(yy.sum()),
        })
    return rows


def cost_per_decision_at_budget(p, y, k_pct, c_inv=0.50, L=100.0, c_sense=0.0):
    """Expected $/decision at fixed top-k% budget using the canonical
    iter-28 cost ratio. c_sense = 0.0035 if this is a sensor-augmented tree.
    """
    n = len(y)
    flags = alert_at_budget(p, k_pct)
    n_alert = int(flags.sum())
    n_pos = int(y.sum())
    tp = int(((flags == 1) & (y == 1)).sum())
    fp = int(((flags == 1) & (y == 0)).sum())
    fn = n_pos - tp
    total_usd = c_inv * (tp + fp) + L * fn + c_sense * n
    return total_usd / n, total_usd / max(tp, 1)


def main():
    print("Reading splits ...")
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    y_tr = train["Class"].to_numpy(np.int32)
    y_te = test["Class"].to_numpy(np.int32)
    print(f"train n={len(y_tr)}, positives={int(y_tr.sum())}, "
          f"test n={len(y_te)}, positives={int(y_te.sum())}")

    print("Fitting (a) the three canonical trees ...")
    sens_scores = {
        "XGB-20raw": fit_tree(train[RAW_FEATS].to_numpy(np.float64), y_tr)
                       .predict_proba(test[RAW_FEATS].to_numpy(np.float64))[:, 1],
        "XGB-24full": fit_tree(train[FULL_FEATS].to_numpy(np.float64), y_tr)
                       .predict_proba(test[FULL_FEATS].to_numpy(np.float64))[:, 1],
        "XGB-4sensor": fit_tree(train[AGG_FEATS].to_numpy(np.float64), y_tr)
                       .predict_proba(test[AGG_FEATS].to_numpy(np.float64))[:, 1],
    }
    print("Fitting (b) 8 ablations of XGB-24full ...")
    abl_scores = {}
    for name, cols in ABLATIONS:
        clf = fit_tree(train[cols].to_numpy(np.float64), y_tr)
        abl_scores[name] = clf.predict_proba(test[cols].to_numpy(np.float64))[:, 1]

    # ---- H1: calibration-by-budget ----
    h1_rows = []
    for name, p in sens_scores.items():
        for k in BUDGETS_K:
            ec, br, npos = calib_at_budget(p, y_te, k)
            cpd, cpc = cost_per_decision_at_budget(
                p, y_te, k, c_inv=0.50, L=100.0,
                c_sense=(0.0 if name == "XGB-20raw" else C_SENSE),
            )
            h1_rows.append({
                "tree": name,
                "K_pct": k,
                "n_alert": int(np.round(len(y_te) * k / 100.0)),
                "n_pos_in_alert": npos,
                "ece_within_budget": ec,
                "brier_within_budget": br,
                "cost_per_decision_usd": cpd,
                "cost_per_caught_usd": cpc if np.isfinite(cpc) else None,
            })
    h1_df = pd.DataFrame(h1_rows)
    h1_df.to_csv(OUT / "p8_iter144_calib_budget.tsv", sep="\t", index=False)
    print(f"Wrote {OUT/'p8_iter144_calib_budget.tsv'} ({len(h1_df)} rows)")

    # ---- H2: 8 ablations ----
    base_ece_K2 = calib_at_budget(sens_scores["XGB-24full"], y_te, 2.0)[0]
    h2_rows = []
    for name, p in abl_scores.items():
        for k in BUDGETS_K:
            ec, br, npos = calib_at_budget(p, y_te, k)
            cpd, cpc = cost_per_decision_at_budget(
                p, y_te, k, c_inv=0.50, L=100.0, c_sense=C_SENSE,
            )
            h2_rows.append({
                "ablation": name,
                "K_pct": k,
                "n_features": len(dict(ABLATIONS)[name]),
                "ece_within_budget": ec,
                "brier_within_budget": br,
                "cost_per_decision_usd": cpd,
                "cost_per_caught_usd": cpc if np.isfinite(cpc) else None,
                "n_pos_in_alert": npos,
                "delta_ece_vs_full_at_K2": ec - base_ece_K2,
            })
    h2_df = pd.DataFrame(h2_rows)
    h2_df.to_csv(OUT / "p8_iter144_calib_ablation.tsv", sep="\t", index=False)
    print(f"Wrote {OUT/'p8_iter144_calib_ablation.tsv'} ({len(h2_df)} rows)")

    # ---- H3: per-decile Brier + gap ----
    h3_rows = []
    for name, p in sens_scores.items():
        for d in calib_decile(p, y_te):
            d2 = dict(d)
            d2["tree"] = name
            h3_rows.append(d2)
    h3_df = pd.DataFrame(h3_rows)
    h3_df.to_csv(OUT / "p8_iter144_brier_decile.tsv", sep="\t", index=False)
    print(f"Wrote {OUT/'p8_iter144_brier_decile.tsv'} ({len(h3_df)} rows)")

    # ---- H4: paired-bootstrap CIs on each (ablation, full) ECE delta at K=2% ----
    rng = np.random.default_rng(BOOT_SEED)
    n = len(y_te)
    h4_rows = []
    for name in ABLATIONS[0][0] and [a[0] for a in ABLATIONS] or []:
        pass
    for name in [a[0] for a in ABLATIONS]:
        abl_p = abl_scores[name]
        base_p = sens_scores["XGB-24full"]
        deltas = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            ec_b, _, _ = calib_at_budget(base_p[idx], y_te[idx], 2.0)
            ec_a, _, _ = calib_at_budget(abl_p[idx], y_te[idx], 2.0)
            if not (np.isfinite(ec_b) and np.isfinite(ec_a)):
                continue
            deltas.append(ec_a - ec_b)
        arr = np.array(deltas)
        ci_lo, ci_hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
        h4_rows.append({
            "ablation": name,
            "delta_ece_K2_mean": float(arr.mean()),
            "delta_ece_K2_ci_low": ci_lo,
            "delta_ece_K2_ci_high": ci_hi,
            "ci_excl_zero": bool(ci_lo > 0 or ci_hi < 0),
            "n_boot_used": len(arr),
        })
    h4_df = pd.DataFrame(h4_rows)
    h4_df.to_csv(OUT / "p8_iter144_ablation_boot.tsv", sep="\t", index=False)
    print(f"Wrote {OUT/'p8_iter144_ablation_boot.tsv'} ({len(h4_df)} rows)")

    # ---- figures ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for name in sens_scores:
        sub = h1_df[h1_df.tree == name]
        axes[0].plot(sub.K_pct, sub.ece_within_budget, "o-", label=name)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Alert budget K (%)")
    axes[0].set_ylabel("ECE within budget")
    axes[0].set_title("H1: calibration degrades as budget tightens")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for name in sens_scores:
        sub = h3_df[h3_df.tree == name]
        axes[1].plot(sub.decile, sub.calib_gap, "o-", label=name)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("Prediction decile (1=lowest, 10=highest)")
    axes[1].set_ylabel("mean_pred - obs_rate  (calib gap)")
    axes[1].set_title("H3: per-decile reliability")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "p8_iter144_calib_budget.png", dpi=150)
    plt.savefig(FIG / "p8_iter144_calib_budget.pdf")
    plt.close()
    print(f"Wrote {FIG/'p8_iter144_calib_budget.png'}")

    summary = {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_pos_test": int(y_te.sum()),
        "budgets_K_pct": BUDGETS_K,
        "ablations": [a[0] for a in ABLATIONS],
        "boot_seed": BOOT_SEED,
        "n_boot": N_BOOT,
        "H1_ece_K2_each_tree": {
            name: float(h1_df[(h1_df.tree == name) & (h1_df.K_pct == 2.0)]
                          .ece_within_budget.iloc[0])
            for name in sens_scores
        },
        "H1_cpd_K2_each_tree": {
            name: float(h1_df[(h1_df.tree == name) & (h1_df.K_pct == 2.0)]
                          .cost_per_decision_usd.iloc[0])
            for name in sens_scores
        },
        "H2_ece_K2_each_ablation": {
            r.ablation: float(r.ece_within_budget)
            for r in h2_df[h2_df.K_pct == 2.0].itertuples()
        },
        "H4_ablations_with_ci_excl_zero": int(h4_df.ci_excl_zero.sum()),
        "H4_ablations_total": int(len(h4_df)),
    }
    with open(OUT / "p8_iter144_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {OUT/'p8_iter144_summary.json'}")
    print()
    print("Headline (H1) ECE within K=2% alert pool:")
    for name, v in summary["H1_ece_K2_each_tree"].items():
        print(f"  {name:12s}  ECE_K2% = {v:.4f}")
    print()
    print("Headline (H4) Ablations whose bootstrap CI excludes zero "
          "(ECE shift detectable): "
          f"{summary['H4_ablations_with_ci_excl_zero']}/"
          f"{summary['H4_ablations_total']}")


if __name__ == "__main__":
    main()
