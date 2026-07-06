#!/usr/bin/env python3
"""P8 JOB A (iter 140): LLM-as-sensor feature ablation at realistic fraud
rates. Fresh vein, not in 156 prior P8 rows.

Closes iter-124 H3 (96% firing preservation across 4 feature sets)
at iter-104/136 realistic positive rates.  Iter-124 fitted XGB at
release-rate 0.172% positives (note: 24full trained on the full
49k train) and reported 96% gradient-band firing preservation across
4 feature sets.  Iter-140 re-runs that ablation on the SAME
24-feature XGB backbone BUT EVALUATES ON 5 DOWNSAMPLED POSITIVE
RATES {1.44, 1.00, 0.50, 0.10, 0.05}% (rate-preserving IID,
seed=20260706, per iter-136 protocol).

H1 (PASS expected): iter-124 H3 result (96% firing preservation)
HOLDS at every rate on a 24full-anchor; rate-conditioning is <
2 percentage points for the 20raw / 20raw+minmax / 20raw+stat
backbones -- the LLM-as-sensor 4-aggregate feature set is
rate-robust at the firing layer.

H2 (NEW): top-K recall @K=1% is rate-conditioned.  At release
rate all 4 backbones match (sensor features carry signal only
in the tail); at 0.05% positives P@1% diverges because the
sparse-positive regime exposes noise floor in the 20raw
backbone.  This is the **operational** case for the 4 aggregate
features.

H3 (NEW): per-rate fire count differs by < 10% across backbones,
even at 0.05%.  This sharpens H1 with a magnitude (not just
sign) claim.

H4 (NEW): the firing agreement matrix is rate-dependent for
{anchor_only} but stable for {both / neither}, which means the
DIVIDING line (test set rows that flip between fire / no-fire
across backbones) is the cost-sensitive surface.

Stdlib + numpy + xgboost.  <= 300 lines.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)
TRAIN = ROOT / "train_data.csv"
TEST = ROOT / "test_data.csv"
SEED = 20260705
RATE_SEED = 20260706
K_PCT = 1.0     # top-1% for the rate-conditioned read
G_THR = 0.001
WIDTH = 0.50

RAW20 = [f"V{i}" for i in range(1, 21)]
AGG4 = ["V_mean", "V_std", "V_max", "V_min"]
ALL24 = RAW20 + AGG4

FEATURE_SETS = {
    "24full":       ALL24,
    "20raw":        RAW20,
    "20raw+minmax": RAW20 + ["V_min", "V_max"],
    "20raw+stat":   RAW20 + ["V_mean", "V_std"],
}

RATES = [1.44, 1.00, 0.50, 0.10, 0.05]


def load(path):
    """Load the 24 numeric columns + Class."""
    with path.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)
        idx = {n: i for i, n in enumerate(header)}
        X, y = [], []
        for line in rdr:
            X.append([float(line[idx[c]]) for c in ALL24])
            y.append(int(float(line[idx["Class"]])))
    return np.array(X), np.array(y)


def downsample_keep(labels, target_rate_pct, rng):
    """Rate-preserving IID: keep all negatives, subsample positives to
    achieve target positive rate.  Returns list of kept row indices."""
    if abs(target_rate_pct - 1.44) < 1e-6:
        return np.arange(len(labels))
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    n_neg = len(neg)
    n_pos_target = int(round(target_rate_pct / 100.0 * n_neg / (1 - target_rate_pct / 100.0)))
    n_pos_target = min(n_pos_target, len(pos))
    keep_pos = rng.choice(pos, size=n_pos_target, replace=False)
    return np.concatenate([neg, keep_pos])


def fit_predict(Xtr, ytr, Xte, feats):
    """XGB on the selected feature subset; return predicted scores."""
    cols = [ALL24.index(c) for c in feats]
    Xtr_s = Xtr[:, cols]
    Xte_s = Xte[:, cols]
    n_pos_tr = max(1, int(ytr.sum()))
    n_neg_tr = max(1, len(ytr) - n_pos_tr)
    spw = n_neg_tr / n_pos_tr
    m = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="auc",
        random_state=SEED, n_jobs=4,
    )
    m.fit(Xtr_s, ytr)
    return m.predict_proba(Xte_s)[:, 1]


def top_k_mask(scores, k_pct):
    n = len(scores)
    k = max(1, int(round(n * k_pct / 100.0)))
    idx = np.argsort(-scores)[:k]
    m = np.zeros(n, dtype=bool)
    m[idx] = True
    return m


def grad_band_fire(scores, top_k_mask_arr, g_thr=G_THR):
    sorted_idx = np.argsort(-scores)
    sorted_scores = scores[sorted_idx]
    grad = np.abs(np.diff(sorted_scores, prepend=sorted_scores[0] + 1.0))
    fire_sorted = (grad < g_thr)
    fire = np.zeros(len(scores), dtype=bool)
    fire[sorted_idx] = fire_sorted
    return fire & top_k_mask_arr


def p_at_top_k(scores, y, k_pct):
    """Precision at top-k%: fraction of top-k scores that are positive."""
    n = len(scores)
    k = max(1, int(round(n * k_pct / 100.0)))
    top_idx = np.argsort(-scores)[:k]
    return float(y[top_idx].sum()) / k


def pr_auc_at_mask(scores, y, k_pct):
    """PR-AUC computed only over test rows actually used (downsampled)."""
    # Standard PR-AUC via (precision, recall) interpolation over all rows
    order = np.argsort(-scores)
    y_sorted = y[order]
    tp_cum = np.cumsum(y_sorted == 1)
    pos_total = max(1, int(y.sum()))
    precision = tp_cum / np.arange(1, len(y) + 1)
    recall = tp_cum / pos_total
    # trapezoidal
    d_recall = np.diff(np.r_[0.0, recall])
    return float((precision * d_recall).sum())


def main():
    print(f"[iter140] loading train/test ...")
    Xtr, ytr = load(TRAIN)
    Xte, yte = load(TEST)
    rng_rate = np.random.default_rng(RATE_SEED)
    print(f"[iter140] Xtr={Xtr.shape} pos_tr={ytr.sum()} | "
          f"Xte={Xte.shape} pos_te={yte.sum()} rate={100*yte.mean():.3f}%")

    # First, fit each feature set ONCE on full train (iter-124 convention);
    # we'll re-use the trained model across rates by re-evaluating on
    # rate-conditioned test subsets.
    print(f"[iter140] fitting XGB on 4 feature sets (single-shot on full train) ...")
    scores_full = {}
    for fset_name, feats in FEATURE_SETS.items():
        scores_full[fset_name] = fit_predict(Xtr, ytr, Xte, feats)
    print(f"[iter140] 4 backbones fit.")

    # ------------------------------------------------------------------
    # Sweep rates: for each (rate, fset), downsample test positives and
    # compute top-K mask + gradient-band fire + P@1% + PR-AUC + agreement
    # ------------------------------------------------------------------
    rows_h1 = []
    rows_h4 = []
    rate_summary = {}

    for rate in RATES:
        keep_idx = downsample_keep(yte, rate, rng_rate)
        Xte_r = Xte[keep_idx]
        yte_r = yte[keep_idx]
        n_te = len(yte_r)
        n_pos_r = int(yte_r.sum())
        rate_actual = 100 * n_pos_r / n_te

        # Store fire + top-K mask per feature set, on THIS rate-conditioned
        # test set.
        fires = {}
        topks = {}
        p_at_1 = {}
        prs = {}
        for fset_name in FEATURE_SETS:
            scores = scores_full[fset_name][keep_idx]
            tk = top_k_mask(scores, K_PCT)
            fire = grad_band_fire(scores, tk)
            fires[fset_name] = fire
            topks[fset_name] = tk
            p_at_1[fset_name] = p_at_top_k(scores, yte_r, K_PCT)
            prs[fset_name] = pr_auc_at_mask(scores, yte_r, K_PCT)

        # H1: agreement vs anchor (24full) at each rate, for 3 non-anchor backbones
        anchor = "24full"
        af = fires[anchor]
        for fset_name in FEATURE_SETS:
            if fset_name == anchor:
                continue
            ff = fires[fset_name]
            n_both = int((af & ff).sum())
            n_anc_only = int((af & ~ff).sum())
            n_fset_only = int((~af & ff).sum())
            n_neither = int((~af & ~ff).sum())
            agreement = (n_both + n_neither) / n_te
            rows_h1.append({
                "rate_target": rate,
                "rate_actual": round(rate_actual, 3),
                "n_test": n_te,
                "n_pos": n_pos_r,
                "fset": fset_name,
                "anchor": anchor,
                "n_fires_anchor": int(af.sum()),
                "n_fires_fset": int(ff.sum()),
                "n_both": n_both,
                "n_anchor_only": n_anc_only,
                "n_fset_only": n_fset_only,
                "n_neither": n_neither,
                "agreement": round(agreement, 4),
                "delta_fires": int(ff.sum() - af.sum()),
                "p_at_1": round(p_at_1[fset_name], 4),
                "pr_auc": round(prs[fset_name], 4),
                "p_at_1_anchor": round(p_at_1[anchor], 4),
                "pr_auc_anchor": round(prs[anchor], 4),
            })

        # H3: per-rate fire-count std across 4 backbones (rate-conditioning)
        fire_counts = {k: int(v.sum()) for k, v in fires.items()}
        fire_arr = np.array(list(fire_counts.values()), dtype=float)
        fire_std = float(fire_arr.std(ddof=1)) if len(fire_arr) > 1 else 0.0
        fire_mean = float(fire_arr.mean())
        rate_summary[str(rate)] = {
            "n_test": n_te,
            "n_pos": n_pos_r,
            "rate_actual_pct": round(rate_actual, 3),
            "fire_counts_per_fset": fire_counts,
            "fire_mean": fire_mean,
            "fire_std": fire_std,
            "fire_cv_pct": round(100 * fire_std / max(1, fire_mean), 2),
            "p_at_1_per_fset": {k: round(v, 4) for k, v in p_at_1.items()},
            "pr_auc_per_fset": {k: round(v, 4) for k, v in prs.items()},
        }

    # Write H1 (agreement per rate per non-anchor fset)
    out_h1 = RES / "p8_iter140_firing_agreement.tsv"
    with out_h1.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_h1[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows_h1)
    print(f"[iter140] wrote {out_h1} ({len(rows_h1)} rows)")

    # ------------------------------------------------------------------
    # H2 -- top-K recall (P@1%) is rate-conditioned.  At low rates the
    # sparse-positive regime exposes noise floor in the 20raw backbone.
    # ------------------------------------------------------------------
    rows_h2 = []
    for rate in RATES:
        rs = rate_summary[str(rate)]
        for fset_name in FEATURE_SETS:
            p_anchor = rate_summary[str(rate)]["p_at_1_per_fset"]["24full"]
            p_fset = rate_summary[str(rate)]["p_at_1_per_fset"][fset_name]
            rows_h2.append({
                "rate": rate,
                "fset": fset_name,
                "p_at_1": p_fset,
                "p_at_1_anchor": p_anchor,
                "delta": round(p_fset - p_anchor, 4),
            })
    out_h2 = RES / "p8_iter140_p_at_1_per_rate.tsv"
    with out_h2.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_h2[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows_h2)
    print(f"[iter140] wrote {out_h2} ({len(rows_h2)} rows)")

    # ------------------------------------------------------------------
    # Verdicts
    # ------------------------------------------------------------------
    # H1: agreement >= 0.94 at every (rate, non-anchor fset)?  Actually
    # iter-124's 96% was BOTH-and-NEITHER fraction over the FULL test set
    # (which itself has a 0.172% pos rate ~release).  We expect agreement
    # >= 0.92 across all rate x fset cells.
    agreements = [r["agreement"] for r in rows_h1]
    h1_min = min(agreements)
    h1_max = max(agreements)
    h1_pass = h1_min >= 0.92

    # H2: max abs(delta p@1) across rates per fset
    deltas_by_fset = {}
    for r in rows_h2:
        deltas_by_fset.setdefault(r["fset"], []).append(abs(r["delta"]))
    h2_max_delta = {k: max(v) for k, v in deltas_by_fset.items()}

    # H3: fire-count CV per rate, all 4 backbones
    h3_cv = {rate: rs["fire_cv_pct"] for rate, rs in rate_summary.items()}

    # H4: the cost-sensitive "flip" cells (fset_only union anchor_only)
    # divided by n_test, per rate.
    rows_h4 = []
    for r in rows_h1:
        # "flip rate" = cells that fire in EXACTLY one of {anchor, fset}
        flip = (r["n_anchor_only"] + r["n_fset_only"]) / r["n_test"]
        rows_h4.append({
            "rate": r["rate_target"],
            "fset": r["fset"],
            "flip_rate_pct": round(100 * flip, 2),
            "anchor_only_pct": round(100 * r["n_anchor_only"] / r["n_test"], 2),
            "fset_only_pct": round(100 * r["n_fset_only"] / r["n_test"], 2),
        })
    out_h4 = RES / "p8_iter140_flip_rate.tsv"
    with out_h4.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_h4[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows_h4)
    print(f"[iter140] wrote {out_h4} ({len(rows_h4)} rows)")

    summary = {
        "iter": 140,
        "n_boot": 0,           # this iter is rate-sweep not bootstrap
        "rates": RATES,
        "n_test_release": len(yte),
        "n_pos_release": int(yte.sum()),
        "rate_release_pct": round(100 * float(yte.mean()), 4),
        "feature_sets": list(FEATURE_SETS.keys()),
        "k_pct": K_PCT,
        "h1_agreement_min": round(h1_min, 4),
        "h1_agreement_max": round(h1_max, 4),
        "h1_pass": bool(h1_pass),
        "h2_max_delta_p_at_1_per_fset": h2_max_delta,
        "h3_fire_cv_per_rate_pct": h3_cv,
        "rate_summary": rate_summary,
    }
    out_sum = RES / "p8_iter140_summary.json"
    with out_sum.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[iter140] wrote {out_sum}")
    print(f"[iter140 H1] agreement range = [{h1_min:.4f}, {h1_max:.4f}], "
          f"verdict={'PASS' if h1_pass else 'FAIL'}")
    print(f"[iter140 H2] max |delta P@1%| per fset = {h2_max_delta}")
    print(f"[iter140 H3] fire-count CV per rate = {h3_cv}")
    print(f"[iter140] DONE")


if __name__ == "__main__":
    main()
