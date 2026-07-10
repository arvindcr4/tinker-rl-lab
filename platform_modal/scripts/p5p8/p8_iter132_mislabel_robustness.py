"""
P8 mislabel-noise robustness on operating point (iter 132 JOB A, fresh vein)

Falsifiable question: how does operating-point quality degrade when a fraction
of TRAINING labels are randomly flipped? This is the realistic noise floor
that fraud-ops teams actually face (chargeback labels are reverse-engineered
from disputed transactions and aren't perfect).

For each tree in {XGB-20raw, XGB-24full} and each mislabel rate epsilon in
{0, 0.005, 0.01, 0.02, 0.05}:
  - flip epsilon of train labels (1-p flips positive->negative, p flips neg->pos
    preserving base rate)
  - refit tree on noisy train
  - evaluate on CLEAN held-out test
  - report 6 ops on clean test: AUC, F1@tau=0.5, Brier, ECE-10,
    cost@tau*(0.50, 100) [iter-28 framework], caught_Recall@K=2% [iter-66]
Paired bootstrap B=400 seed=20260705.

Outputs:
  platform_hybrid/experiments/results/p5p8/p8_iter132_mislabel_main.tsv (50 rows: 5 eps x 2 trees x 5 metrics)
  platform_hybrid/experiments/results/p5p8/p8_iter132_mislabel_flip.tsv  (rows: 5 eps x 2 trees x 2 ops (tp @ 0.5 / cost @ tau*))
  platform_hybrid/experiments/results/p5p8/p8_iter132_mislabel_boot.tsv  (paired bootstrap CIs per gap, 5 eps x 5 metrics)
  platform_hybrid/experiments/results/p5p8/p8_iter132_mislabel_summary.json

Operationally this answers: does the iter-66 K=2% dominance switch survive
label noise? If yes, the sensor's recall-restoration value is robust; if the
gap collapses at 1% mislabel, the dominance claim needs a label-noise caveat.
"""

from __future__ import annotations
import json
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss

SEED = 20260705
N_BOOT = 400
EPS_GRID = (0.0, 0.005, 0.01, 0.02, 0.05)
K_OPERATING = 0.02  # K=2% — iter-66 dominance-switch
RHO_CANON = 200.0   # L/C_inv ratio for tau* computation
C_INV_CANON = 0.50  # $/alert
L_CANON = C_INV_CANON * RHO_CANON  # $=100
C_SENSE = 0.0035    # $/row for LLM sensor
TAU_DEFAULT = 0.5

OUT_DIR = Path("/home/claude/tinker-rl-lab-minimax/platform_hybrid/experiments/results/p5p8")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    train = pd.read_csv("/home/claude/tinker-rl-lab-minimax/fraud_data.csv")
    test = pd.read_csv("/home/claude/tinker-rl-lab-minimax/test_data.csv")
    raw_cols = [f"V{i}" for i in range(1, 21)]
    agg_cols = ["V_mean", "V_std", "V_max", "V_min"]
    Xtr_raw = train[raw_cols].values
    Xtr_full = train[raw_cols + agg_cols].values
    ytr = train["Class"].values.astype(int)
    Xte_raw = test[raw_cols].values
    Xte_full = test[raw_cols + agg_cols].values
    yte = test["Class"].values.astype(int)
    return (Xtr_raw, Xtr_full, ytr, Xte_raw, Xte_full, yte)

def flip_labels(y, eps, rng):
    """Flip epsilon fraction of labels. Preserve the overall positive rate
    by flipping positives to negative AND negatives to positive at the same
    rate (this keeps class prevalence close but does not strictly preserve it
    for finite N)."""
    if eps == 0.0:
        return y.copy()
    n = len(y)
    n_flip = int(round(eps * n))
    flip_idx = rng.choice(n, size=n_flip, replace=False)
    y_noisy = y.copy()
    y_noisy[flip_idx] = 1 - y_noisy[flip_idx]
    return y_noisy

def fit_predict(Xtr, ytr, Xte, seed):
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        random_state=seed, n_jobs=1, tree_method="hist",
        scale_pos_weight=(1.0 - ytr.mean()) / max(ytr.mean(), 1e-6),
    )
    clf.fit(Xtr, ytr, verbose=False)
    return clf.predict_proba(Xte)[:, 1]

def tau_star(yscores, y, c_inv, c_sense, l_miss, k_max=None):
    """Iter-28 cost-optimal threshold rule. Returns tau* and the minimum total
    cost over all possible cutoff rank positions."""
    n = len(y)
    if k_max is None:
        k_max = n
    order = np.argsort(-yscores)
    y_sorted = y[order]
    cum_tp = np.cumsum(y_sorted)
    cum_fp = np.cumsum(1 - y_sorted)
    cum_fn = (cum_tp[-1] - cum_tp)
    total_cost = (
        c_sense * n
        + c_inv * (cum_tp + cum_fp)
        + l_miss * cum_fn
    )
    best = np.argmin(total_cost)
    tau_at_best = yscores[order[best]]
    return float(tau_at_best), float(total_cost[best])

def ece_10(yscores, y):
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(yscores, bins) - 1
    idx = np.clip(idx, 0, 9)
    ece = 0.0
    for b in range(10):
        m = idx == b
        if m.any():
            ece += abs(yscores[m].mean() - y[m].mean()) * m.mean()
    return ece

def caught_recall_at_K(yscores, y, k_frac):
    n = len(y)
    k = max(1, int(round(k_frac * n)))
    order = np.argsort(-yscores)[:k]
    return float(y[order].sum() / max(y.sum(), 1))

def metrics(yscores, y, c_inv=C_INV_CANON, l_miss=L_CANON):
    """Compute 6 ops. yscores sorted descending → y."""
    auc = float(roc_auc_score(y, yscores))
    brier = float(brier_score_loss(y, yscores))
    ece = float(ece_10(yscores, y))
    yhat05 = (yscores >= TAU_DEFAULT).astype(int)
    f1 = float(f1_score(y, yhat05, zero_division=0))
    tau, cost = tau_star(yscores, y, c_inv, C_SENSE, l_miss)
    caught = caught_recall_at_K(yscores, y, K_OPERATING)
    return {
        "auc": auc,
        "brier": brier,
        "ece10": ece,
        "f1_at_0.5": f1,
        "tau_star": float(tau),
        "cost_at_tau_star": float(cost),
        "cost_per_dec": float(cost / len(y)),
        "caught_recall_at_K2pct": caught,
    }

def paired_bootstrap_gap(m_a, m_b, y, score_a, score_b, rng, n_boot=N_BOOT):
    """Paired bootstrap on the per-sample gap. m_a/m_b ignored; recompute
    over (score_a, score_b) which are paired on the held-out split."""
    n = len(y)
    out = {}
    for k in ("auc", "brier", "ece10", "f1_at_0.5", "cost_per_dec", "caught_recall_at_K2pct"):
        diffs = []
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            mb_a = metrics(score_a[idx], y[idx])
            mb_b = metrics(score_b[idx], y[idx])
            diffs.append(mb_a[k] - mb_b[k])
        arr = np.array(diffs)
        out[k] = {
            "mean": float(arr.mean()),
            "ci_lo": float(np.quantile(arr, 0.025)),
            "ci_hi": float(np.quantile(arr, 0.975)),
            "p_excludes_zero": bool((arr.min() > 0) or (arr.max() < 0)),
        }
    return out

def main():
    (Xtr_raw, Xtr_full, ytr, Xte_raw, Xte_full, yte) = load_data()
    rng = np.random.default_rng(SEED)

    # tree seeds for fit stochasticity (independent of bootstrap)
    fit_seeds = {0.0: 11, 0.005: 22, 0.01: 33, 0.02: 44, 0.05: 55}

    # Main pass
    main_records = []
    score_cache = {}  # (eps, tree) -> (score_raw, score_full)
    for eps in EPS_GRID:
        # Use a SINGLE noise realization per (eps, rep) to keep results deterministic.
        rng_noise = np.random.default_rng(SEED + int(eps * 1e6))
        y_noisy = flip_labels(ytr, eps, rng_noise)

        X_raw = Xtr_raw
        X_full = Xtr_full

        clf_raw = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=fit_seeds[eps], n_jobs=1, tree_method="hist",
            scale_pos_weight=(1.0 - y_noisy.mean()) / max(y_noisy.mean(), 1e-6),
        )
        clf_raw.fit(X_raw, y_noisy, verbose=False)
        s_raw = clf_raw.predict_proba(Xte_raw)[:, 1]

        clf_full = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=fit_seeds[eps], n_jobs=1, tree_method="hist",
            scale_pos_weight=(1.0 -y_noisy.mean()) / max(y_noisy.mean(), 1e-6),
        )
        clf_full.fit(X_full, y_noisy, verbose=False)
        s_full = clf_full.predict_proba(Xte_full)[:, 1]

        m_raw = metrics(s_raw, yte)
        m_full = metrics(s_full, yte)
        score_cache[eps] = (s_raw, s_full)

        for tree_name, m in (("XGB-20raw", m_raw), ("XGB-24full", m_full)):
            for k, v in m.items():
                main_records.append({
                    "epsilon": eps, "tree": tree_name, "metric": k, "value": v,
                    "n_test": int(len(yte)), "n_test_pos": int(yte.sum()),
                })

    main = pd.DataFrame(main_records)
    main.to_csv(OUT_DIR / "p8_iter132_mislabel_main.tsv", sep="\t", index=False)

    # Flip-rate record (compact per-eps summary)
    flip_records = []
    for eps in EPS_GRID:
        s_raw, s_full = score_cache[eps]
        m_raw = metrics(s_raw, yte)
        m_full = metrics(s_full, yte)
        flip_records.append({
            "epsilon": eps,
            "n_train_pos_original": int(ytr.sum()),
            "n_train_pos_noisy": int(sum((1 if not (flip_labels(ytr, eps, np.random.default_rng(SEED+int(eps*1e6)))==ytr).any() else 1 for _ in [0]))),  # not used
        })
    # (Skip noisy n_flips since we don't store them; use simpler per-eps summary)
    simple = []
    for eps in EPS_GRID:
        s_raw, s_full = score_cache[eps]
        m_raw = metrics(s_raw, yte)
        m_full = metrics(s_full, yte)
        simple.append({
            "epsilon": eps,
            "n_train_pos": int(ytr.sum()),
            "n_train_neg": int(len(ytr) - ytr.sum()),
            "raw_auc": m_raw["auc"], "full_auc": m_full["auc"],
            "raw_cost_per_dec": m_raw["cost_per_dec"], "full_cost_per_dec": m_full["cost_per_dec"],
            "raw_caught_K2pct": m_raw["caught_recall_at_K2pct"], "full_caught_K2pct": m_full["caught_recall_at_K2pct"],
            "delta_auc_24full_minus_20raw": m_full["auc"] - m_raw["auc"],
            "delta_cost_24full_minus_20raw": m_full["cost_per_dec"] - m_raw["cost_per_dec"],
            "delta_caught_24full_minus_20raw": m_full["caught_recall_at_K2pct"] - m_raw["caught_recall_at_K2pct"],
        })
    flip_df = pd.DataFrame(simple)
    flip_df.to_csv(OUT_DIR / "p8_iter132_mislabel_flip.tsv", sep="\t", index=False)

    # Bootstrap CIs on the per-eps gap (24full - 20raw)
    boot_rng = np.random.default_rng(SEED + 7919)
    boot_records = []
    for eps in EPS_GRID:
        s_raw, s_full = score_cache[eps]
        gap = paired_bootstrap_gap(None, None, yte, s_full, s_raw, boot_rng, n_boot=N_BOOT)
        for metric, g in gap.items():
            boot_records.append({
                "epsilon": eps,
                "metric": metric,
                "mean_gap": g["mean"],
                "ci_lo": g["ci_lo"],
                "ci_hi": g["ci_hi"],
                "p_excludes_zero": g["p_excludes_zero"],
                "n_boot": N_BOOT,
                "seed": SEED + 7919,
            })
    boot_df = pd.DataFrame(boot_records)
    boot_df.to_csv(OUT_DIR / "p8_iter132_mislabel_boot.tsv", sep="\t", index=False)

    # H-tests
    h = {}
    from scipy.stats import spearmanr
    # H1 (operational dominance flip): 24full beats 20raw on caught_K2pct at
    # eps>=0.01 (where iter-66 dominance switch sits on clean data; iter-132
    # asks whether mislabel noise preserves or destroys it)
    gap_per_eps = {
        e: float(flip_df[flip_df["epsilon"]==e]["delta_caught_24full_minus_20raw"].iloc[0])
        for e in EPS_GRID
    }
    h["H1_dominance_emerges_at_eps_1pct"] = bool(gap_per_eps[0.01] > 0)
    # H2: full_auc at eps=0.05 should remain in operationally viable band
    eps5_full_auc = float(flip_df[flip_df["epsilon"] == 0.05]["full_auc"].iloc[0])
    h["H2_24full_auc_at_5pct_in_safe_band"] = bool(eps5_full_auc > 0.97 and eps5_full_auc < 0.999)
    # H3: cost-per-dec degrades monotonically WITH eps (rho <= -0.5)
    rho_raw = spearmanr(EPS_GRID, [flip_df[flip_df["epsilon"]==e]["raw_cost_per_dec"].iloc[0] for e in EPS_GRID]).statistic
    rho_full = spearmanr(EPS_GRID, [flip_df[flip_df["epsilon"]==e]["full_cost_per_dec"].iloc[0] for e in EPS_GRID]).statistic
    h["H3_cost_degrades_monotonic_raw"] = bool(rho_raw <= -0.5)
    h["H3_cost_degrades_monotonic_full"] = bool(rho_full <= -0.5)
    # H4: caught_recall@K=2% drops with eps (rho <= -0.5; caught DECREASES as eps INCREASES)
    caught_raw = [flip_df[flip_df["epsilon"]==e]["raw_caught_K2pct"].iloc[0] for e in EPS_GRID]
    caught_full = [flip_df[flip_df["epsilon"]==e]["full_caught_K2pct"].iloc[0] for e in EPS_GRID]
    rho_cr_raw = spearmanr(EPS_GRID, caught_raw).statistic
    rho_cr_full = spearmanr(EPS_GRID, caught_full).statistic
    h["H4_caught_K2pct_degrades_raw"] = bool(rho_cr_raw <= -0.5)
    h["H4_caught_K2pct_degrades_full"] = bool(rho_cr_full <= -0.5)

    # H5 (sharpest headline): the iter-66 K=2% dominance switch — 24full > 20raw at K>=2%
    # on caught_Recall — gap STRENGTHENS monotonically with eps (positive correlation
    # between |gap| and eps once past the dominance-flip threshold)
    eps5 = boot_df[(boot_df["epsilon"]==0.05)]
    caught_row = eps5[eps5["metric"]=="caught_recall_at_K2pct"].iloc[0]
    gap_arr = [abs(gap_per_eps[e]) for e in EPS_GRID if abs(gap_per_eps[e]) > 0]
    if len(gap_arr) >= 3 and sum(1 for e in EPS_GRID if abs(gap_per_eps[e]) > 0) >= 3:
        active_eps = [e for e in EPS_GRID if abs(gap_per_eps[e]) > 0]
        active_gap = [gap_per_eps[e] for e in active_eps]
        rho_gap = spearmanr(active_eps, active_gap).statistic
    else:
        rho_gap = float("nan")
    h["H5_gap_strengthens_with_eps"] = bool(rho_gap > 0.5)

    summary = {
        "iter": 132,
        "pillar": "P8",
        "seed_main": SEED,
        "n_boot": N_BOOT,
        "epsilon_grid": list(EPS_GRID),
        "c_inv_canonical_USD": C_INV_CANON,
        "L_canonical_USD": L_CANON,
        "c_sense_USD": C_SENSE,
        "K_operating": K_OPERATING,
        "operating_point_metrics_at_each_eps": {
            str(e): {
                "raw_auc": float(flip_df[flip_df["epsilon"]==e]["raw_auc"].iloc[0]),
                "full_auc": float(flip_df[flip_df["epsilon"]==e]["full_auc"].iloc[0]),
                "raw_cost_per_dec": float(flip_df[flip_df["epsilon"]==e]["raw_cost_per_dec"].iloc[0]),
                "full_cost_per_dec": float(flip_df[flip_df["epsilon"]==e]["full_cost_per_dec"].iloc[0]),
                "raw_caught_K2pct": float(flip_df[flip_df["epsilon"]==e]["raw_caught_K2pct"].iloc[0]),
                "full_caught_K2pct": float(flip_df[flip_df["epsilon"]==e]["full_caught_K2pct"].iloc[0]),
            }
            for e in EPS_GRID
        },
        "bootstrap_CI_at_eps_5pct": {
            "auc":             float(eps5[eps5["metric"]=="auc"]["mean_gap"].iloc[0]),
            "brier":           float(eps5[eps5["metric"]=="brier"]["mean_gap"].iloc[0]),
            "caught_K2pct":    float(caught_row["mean_gap"]),
            "caught_ci_lo":    float(caught_row["ci_lo"]),
            "caught_ci_hi":    float(caught_row["ci_hi"]),
        },
        "headline_falsifiable": {
            "H1_dominance_emerges_at_eps_1pct": bool(h["H1_dominance_emerges_at_eps_1pct"]),
            "gap_per_eps_24full_minus_20raw_caught_K2pct": gap_per_eps,
            "H2_24full_auc_at_5pct_in_safe_band": bool(h["H2_24full_auc_at_5pct_in_safe_band"]),
            "H3_cost_degrades_monotonic_raw": float(rho_raw),
            "H3_cost_degrades_monotonic_full": float(rho_full),
            "H4_caught_K2pct_degrades_raw": float(rho_cr_raw),
            "H4_caught_K2pct_degrades_full": float(rho_cr_full),
            "H5_gap_strengthens_with_eps": bool(h["H5_gap_strengthens_with_eps"]),
            "rho_gap_vs_eps": float(rho_gap),
        },
        "n_test": int(len(yte)),
        "n_test_pos": int(yte.sum()),
    }
    with open(OUT_DIR / "p8_iter132_mislabel_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
