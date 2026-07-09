#!/usr/bin/env python3
"""P8 V-stat ensemble precision-restoration ablation (iter 172 JOB A).

Fresh vein, not in 179 prior P8 rows. Closes iter-168's operational
recommendation (4): EXTEND the sensor with a learned precision-restoration
layer (joint V_mean/V_std/V_max/V_min classifier) -- the structural
precision ceiling may be a *single-feature* ceiling, not a dataset-level
ceiling.

Approach
--------
For each (seed, rate, fset) cell, fit XGB on `fraud_data.csv` and compute
XGB scores on `test_data.csv` (top-K=2% scoring as the canonical XGB
baseline). Then for every XGB-missed row, train a SECOND-stage classifier
on the same train set restricted to the four aggregate features
(V_mean, V_std, V_max, V_min) -- a logistic regression on standardised
features with high scale_pos_weight -- and predict P_fraud for the
XGB-missed rows. Sweep a probability threshold on this joint-V classifier
and measure esc_prec / value_rate at each.

Hypotheses
----------
H1 (DECISIVE): the joint V-stat classifier raises esc_prec above 5%
     on >= 25% of (seed x rate x fset) cells at some threshold
     (vs iter-168's 0.0% structural ceiling).
H2: the Pareto frontier (esc_prec >= 0.10 AND value_rate >= 0.30)
     exists on at least one (seed x rate x fset x tau) cell when
     using the joint classifier (vs iter-168's 0 cells).
H3: joint classifier esc_prec strictly exceeds single-V_mean classifier
     esc_prec at matched operating points on >= 50% of cells
     (positive monotonic lift from feature aggregation).
H4: joint classifier value_rate remains >= 0.30 at the threshold where
     esc_prec first crosses 5% on >= 25% of cells (precision lift
     without collapsing recall).

Stdlib + numpy + xgboost + sklearn.linear_model.  <= 300 lines.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)
SEEDS = [20260706, 20260708, 20260710, 20260712, 20260714]
K_PCT = 2.0
VALUE_PER_CATCH = 50.0

RAW20 = [f"V{i}" for i in range(1, 21)]
AGG4 = ["V_mean", "V_std", "V_max", "V_min"]
ALL24 = RAW20 + AGG4
FEATURE_SETS = {
    "24full": ALL24,
    "20raw":  RAW20,
    "20raw+minmax": RAW20 + ["V_min", "V_max"],
    "20raw+stat":   RAW20 + ["V_mean", "V_std"],
}

RATES_PCT = [1.44, 1.00, 0.50, 0.10, 0.05]
TAUS_JOINT = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def load(path):
    with path.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)
        idx = {n: i for i, n in enumerate(header)}
        X, y = [], []
        for line in rdr:
            X.append([float(line[idx[c]]) for c in ALL24])
            y.append(int(float(line[idx["Class"]])))
    return np.array(X), np.array(y)


def fit_xgb(Xtr, ytr, Xte, feats, seed):
    cols = [ALL24.index(c) for c in feats]
    n_pos = max(1, int(ytr.sum()))
    n_neg = max(1, len(ytr) - n_pos)
    spw = n_neg / n_pos
    m = xgb.XGBClassifier(
        n_estimators=180, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="auc",
        random_state=seed, n_jobs=4,
    )
    m.fit(Xtr[:, cols], ytr)
    return m.predict_proba(Xte[:, cols])[:, 1]


def fit_joint_vstat_lr(Xtr, ytr):
    """Logistic regression on (V_mean, V_std, V_max, V_min) -> is_fraud."""
    cols = [ALL24.index(c) for c in AGG4]
    Xtr_v = Xtr[:, cols]
    sc = StandardScaler().fit(Xtr_v)
    Xtr_v_std = sc.transform(Xtr_v)
    n_pos = max(1, int(ytr.sum()))
    n_neg = max(1, len(ytr) - n_pos)
    spw = n_neg / n_pos
    clf = LogisticRegression(
        C=1.0, max_iter=200, solver="liblinear",
        class_weight={0: 1.0, 1: float(spw)},
        random_state=0,
    )
    clf.fit(Xtr_v_std, ytr)
    return clf, sc


def predict_joint(clf, sc, Xte):
    cols = [ALL24.index(c) for c in AGG4]
    return clf.predict_proba(sc.transform(Xte[:, cols]))[:, 1]


def fit_vmean_lr(Xtr, ytr):
    """Single-feature logistic regression on V_mean only -> is_fraud."""
    cols = [ALL24.index("V_mean")]
    Xtr_v = Xtr[:, cols]
    sc = StandardScaler().fit(Xtr_v)
    Xtr_v_std = sc.transform(Xtr_v)
    n_pos = max(1, int(ytr.sum()))
    n_neg = max(1, len(ytr) - n_pos)
    spw = n_neg / n_pos
    clf = LogisticRegression(
        C=1.0, max_iter=200, solver="liblinear",
        class_weight={0: 1.0, 1: float(spw)},
        random_state=0,
    )
    clf.fit(Xtr_v_std, ytr)
    return clf, sc


def predict_vmean(clf, sc, Xte):
    cols = [ALL24.index("V_mean")]
    return clf.predict_proba(sc.transform(Xte[:, cols]))[:, 1]


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


def main():
    print(f"[iter172] loading train/test ...")
    Xtr, ytr = load(ROOT / "train_data.csv")
    Xte_full, yte_full = load(ROOT / "test_data.csv")
    print(f"[iter172] Xtr={Xtr.shape} ytr_pos={ytr.sum()} | "
          f"Xte={Xte_full.shape} yte_pos={yte_full.sum()}")

    # Train joint and single-V_mean LRs once on the full train set.
    joint_clf, joint_sc = fit_joint_vstat_lr(Xtr, ytr)
    vmean_clf, vmean_sc = fit_vmean_lr(Xtr, ytr)
    joint_train_p = predict_joint(joint_clf, joint_sc, Xtr)
    vmean_train_p = predict_vmean(vmean_clf, vmean_sc, Xtr)
    print(f"[iter172] train joint AUC-equivalent (pos_rate at thr=0.5): "
          f"{(joint_train_p[ytr==1] > 0.5).mean():.4f} vs neg "
          f"{(joint_train_p[ytr==0] > 0.5).mean():.4f}")
    print(f"[iter172] train vmean pos_rate at thr=0.5: "
          f"{(vmean_train_p[ytr==1] > 0.5).mean():.4f} vs neg "
          f"{(vmean_train_p[ytr==0] > 0.5).mean():.4f}")

    matrix_rows = []
    pareto_rows = []
    h3_rows = []

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for rate_pct in RATES_PCT:
            Xte, yte = downsample_positives(Xte_full, yte_full, rate_pct, rng)
            n_te = len(yte)
            k = max(1, int(round(n_te * K_PCT / 100.0)))

            joint_p = predict_joint(joint_clf, joint_sc, Xte)
            vmean_p = predict_vmean(vmean_clf, vmean_sc, Xte)

            for fset_name, feats in FEATURE_SETS.items():
                scores = fit_xgb(Xtr, ytr, Xte, feats, seed)
                top_k_idx = np.argsort(-scores)[:k]
                xgb_fire = np.zeros(n_te, dtype=bool)
                xgb_fire[top_k_idx] = True
                n_xgb_missed_pos = int(np.sum(~xgb_fire & (yte == 1)))
                xgb_missed_mask = ~xgb_fire

                # ---- joint classifier sweep ----
                for tau in TAUS_JOINT:
                    joint_fire = joint_p > tau
                    llm_fire = xgb_missed_mask & joint_fire
                    n_lift = int(np.sum(llm_fire & (yte == 1)))
                    n_waste = int(np.sum(llm_fire & (yte == 0)))
                    n_llm_only = int(np.sum(llm_fire))
                    value_rate = n_lift / max(1, n_xgb_missed_pos)
                    esc_prec = n_lift / max(1, n_lift + n_waste)
                    matrix_rows.append({
                        "seed": seed, "rate_pct": rate_pct, "fset": fset_name,
                        "tau": tau, "clf": "joint_vstat",
                        "n_test": n_te, "n_pos": int(yte.sum()),
                        "n_xgb_missed_pos": n_xgb_missed_pos,
                        "n_lift": n_lift, "n_waste": n_waste,
                        "n_llm_only": n_llm_only,
                        "value_rate": value_rate, "esc_prec": esc_prec,
                    })

                # ---- single-V_mean classifier sweep (matched operating) ----
                for tau in TAUS_JOINT:
                    vmean_fire = vmean_p > tau
                    llm_fire = xgb_missed_mask & vmean_fire
                    n_lift = int(np.sum(llm_fire & (yte == 1)))
                    n_waste = int(np.sum(llm_fire & (yte == 0)))
                    n_llm_only = int(np.sum(llm_fire))
                    value_rate = n_lift / max(1, n_xgb_missed_pos)
                    esc_prec = n_lift / max(1, n_lift + n_waste)
                    matrix_rows.append({
                        "seed": seed, "rate_pct": rate_pct, "fset": fset_name,
                        "tau": tau, "clf": "vmean_only",
                        "n_test": n_te, "n_pos": int(yte.sum()),
                        "n_xgb_missed_pos": n_xgb_missed_pos,
                        "n_lift": n_lift, "n_waste": n_waste,
                        "n_llm_only": n_llm_only,
                        "value_rate": value_rate, "esc_prec": esc_prec,
                    })

                # Pareto frontier check at each (seed, rate, fset, clf, tau)
                for tau in TAUS_JOINT:
                    for clf_name in ("joint_vstat", "vmean_only"):
                        row = next((r for r in matrix_rows
                                    if r["seed"] == seed
                                    and r["rate_pct"] == rate_pct
                                    and r["fset"] == fset_name
                                    and r["tau"] == tau
                                    and r["clf"] == clf_name), None)
                        if row is None:
                            continue
                        pareto_rows.append({
                            "seed": seed, "rate_pct": rate_pct,
                            "fset": fset_name, "tau": tau,
                            "clf": clf_name,
                            "value_rate": row["value_rate"],
                            "esc_prec": row["esc_prec"],
                            "n_lift": row["n_lift"],
                            "n_waste": row["n_waste"],
                            "pareto_ok": (row["esc_prec"] >= 0.10
                                          and row["value_rate"] >= 0.30),
                        })

                # H3: joint strictly > vmean-only at matched (tau) cell
                joint_vrs = [r["value_rate"] for r in matrix_rows
                             if r["seed"] == seed and r["rate_pct"] == rate_pct
                             and r["fset"] == fset_name
                             and r["clf"] == "joint_vstat"]
                vmean_vrs = [r["value_rate"] for r in matrix_rows
                             if r["seed"] == seed and r["rate_pct"] == rate_pct
                             and r["fset"] == fset_name
                             and r["clf"] == "vmean_only"]
                joint_prec = [r["esc_prec"] for r in matrix_rows
                              if r["seed"] == seed and r["rate_pct"] == rate_pct
                              and r["fset"] == fset_name
                              and r["clf"] == "joint_vstat"]
                vmean_prec = [r["esc_prec"] for r in matrix_rows
                              if r["seed"] == seed and r["rate_pct"] == rate_pct
                              and r["fset"] == fset_name
                              and r["clf"] == "vmean_only"]
                if len(joint_vrs) == len(TAUS_JOINT) and len(vmean_vrs) == len(TAUS_JOINT):
                    h3_rows.append({
                        "seed": seed, "rate_pct": rate_pct, "fset": fset_name,
                        "joint_mean_vr": float(np.mean(joint_vrs)),
                        "vmean_mean_vr": float(np.mean(vmean_vrs)),
                        "joint_mean_prec": float(np.mean(joint_prec)),
                        "vmean_mean_prec": float(np.mean(vmean_prec)),
                        "joint_strict_gt_vmean_prec": float(np.mean(
                            [j > v for j, v in zip(joint_prec, vmean_prec)])),
                        "joint_strict_gt_vmean_vr": float(np.mean(
                            [j > v for j, v in zip(joint_vrs, vmean_vrs)])),
                    })

    # H1: joint esc_prec > 0.05 on >= 25% of (seed x rate x fset) cells at any tau
    joint_cells = [r for r in matrix_rows if r["clf"] == "joint_vstat"]
    h1_pass_cells = set()
    for cell_key in {(r["seed"], r["rate_pct"], r["fset"]) for r in joint_cells}:
        cell_rows = [r for r in joint_cells
                     if (r["seed"], r["rate_pct"], r["fset"]) == cell_key]
        if any(r["esc_prec"] >= 0.05 for r in cell_rows):
            h1_pass_cells.add(cell_key)
    h1_total = len({(r["seed"], r["rate_pct"], r["fset"]) for r in joint_cells})
    h1_pass_rate = len(h1_pass_cells) / max(1, h1_total)
    h1_pass = h1_pass_rate >= 0.25
    print(f"[iter172] H1: {len(h1_pass_cells)}/{h1_total} = {h1_pass_rate:.3f} "
          f"cells achieve esc_prec >= 0.05 at some joint-tau; PASS={h1_pass}")

    # H2: Pareto frontier exists on at least one joint-cell
    joint_pareto = [p for p in pareto_rows if p["clf"] == "joint_vstat"]
    n_pareto = sum(1 for p in joint_pareto if p["pareto_ok"])
    h2_pass = n_pareto >= 1
    print(f"[iter172] H2: {n_pareto}/{len(joint_pareto)} joint pareto_ok cells; "
          f"PASS={h2_pass}")
    if n_pareto >= 1:
        best = max((p for p in joint_pareto if p["pareto_ok"]),
                   key=lambda p: p["value_rate"])
        print(f"[iter172]   best joint pareto cell: seed={best['seed']} "
              f"rate={best['rate_pct']} fset={best['fset']} "
              f"tau={best['tau']} vr={best['value_rate']:.4f} "
              f"prec={best['esc_prec']:.4f}")

    # H3: joint precision strictly > vmean precision on >= 50% of cells (averaged across taus)
    if h3_rows:
        h3_pass_rate = sum(1 for r in h3_rows
                           if r["joint_strict_gt_vmean_prec"] >= 0.5) / max(1, len(h3_rows))
        h3_pass = h3_pass_rate >= 0.50
        print(f"[iter172] H3: {sum(1 for r in h3_rows if r['joint_strict_gt_vmean_prec'] >= 0.5)}/"
              f"{len(h3_rows)} = {h3_pass_rate:.3f} cells with joint>50% of taus"
              f" beating vmean on precision; PASS={h3_pass}")

    # H4: at the joint-tau where esc_prec first crosses 5%, value_rate remains >= 0.30 on >= 25%
    h4_pass_cells = 0
    h4_total_cells = 0
    for cell_key in {(r["seed"], r["rate_pct"], r["fset"]) for r in joint_cells}:
        cell_rows = sorted(
            [r for r in joint_cells
             if (r["seed"], r["rate_pct"], r["fset"]) == cell_key],
            key=lambda r: r["tau"])
        # First tau at which esc_prec >= 0.05
        first_5pct = next((r for r in cell_rows if r["esc_prec"] >= 0.05), None)
        if first_5pct is not None:
            h4_total_cells += 1
            if first_5pct["value_rate"] >= 0.30:
                h4_pass_cells += 1
    h4_pass_rate = h4_pass_cells / max(1, h4_total_cells)
    h4_pass = h4_pass_rate >= 0.25
    print(f"[iter172] H4: {h4_pass_cells}/{h4_total_cells} = {h4_pass_rate:.3f} "
          f"cells with value_rate>=0.30 at first-5%-tau; PASS={h4_pass}")

    # Output files
    out_matrix = RES / "p8_iter172_threshold_matrix.tsv"
    fields = list(matrix_rows[0].keys())
    with out_matrix.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(matrix_rows)
    print(f"[iter172] wrote {out_matrix} ({len(matrix_rows)} rows)")

    out_pareto = RES / "p8_iter172_pareto_cells.tsv"
    fields2 = list(pareto_rows[0].keys())
    with out_pareto.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields2, delimiter="\t")
        w.writeheader()
        w.writerows(pareto_rows)
    print(f"[iter172] wrote {out_pareto} ({len(pareto_rows)} rows)")

    out_h3 = RES / "p8_iter172_joint_vs_vmean.tsv"
    fields3 = list(h3_rows[0].keys())
    with out_h3.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields3, delimiter="\t")
        w.writeheader()
        w.writerows(h3_rows)
    print(f"[iter172] wrote {out_h3} ({len(h3_rows)} rows)")

    summary = {
        "iter": 172,
        "job": "P8 V-stat ensemble precision-restoration ablation",
        "n_seeds": len(SEEDS),
        "n_taus_joint": len(TAUS_JOINT),
        "n_total_cells": len(matrix_rows),
        "h1_pass": h1_pass,
        "h1_pass_rate": h1_pass_rate,
        "h1_n_pass_cells": len(h1_pass_cells),
        "h1_n_total_cells": h1_total,
        "h2_pass": h2_pass,
        "h2_n_pareto_cells": n_pareto,
        "h3_pass": h3_pass,
        "h3_pass_rate": h3_pass_rate,
        "h4_pass": h4_pass,
        "h4_pass_rate": h4_pass_rate,
    }
    out_sum = RES / "p8_iter172_summary.json"
    with out_sum.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[iter172] wrote {out_sum}")
    print(f"[iter172] DONE")


if __name__ == "__main__":
    main()