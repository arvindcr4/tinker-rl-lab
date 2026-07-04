#!/usr/bin/env python3
"""P5 MIN-REPORT field predictive-sufficiency ("load-bearing") test.

Operationalizes "Report the Stack, Not the Label": how much of the variance in
per-cell telemetry (ZVF, mean_reward) can be predicted from the DISCLOSED stack
fields, and what is the marginal predictive regret of OMITTING each field?

Method (stdlib + numpy + sklearn):
  * Predictors = MIN-REPORT stack fields present in every mega manifest/cell:
    model_family, task_slice, G (log2), temperature. seed is a nuisance control.
  * Model = RandomForestRegressor (captures field interactions; min_samples_leaf
    guards the 98-row regime). Out-of-fold (OOF) predictions via K-fold CV,
    averaged over several fold seeds for stability.
  * Full-model OOF R^2, leave-one-field-out OOF R^2 -> dR^2 = regret of omission.
  * Paired cluster (over-cell) bootstrap CIs on R^2 and every dR^2.
  * Label-only baseline: all cells carry the SAME sampling label -> R^2 == 0 by
    construction (predict-the-mean). The stack, not the label, is predictive.

Outputs: experiments/results/p5p8/p5_field_sufficiency{.tsv,_summary.json}
"""
import csv, json, math, os, time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CELLS = os.path.join(ROOT, "experiments/results/mega_20260704/cells.tsv")
OUTDIR = os.path.join(ROOT, "experiments/results/p5p8")
os.makedirs(OUTDIR, exist_ok=True)

TARGETS = ["zvf", "mean_reward"]
# Disclosed MIN-REPORT stack fields used as predictors + kind.
FIELDS = ["model_family", "task_slice", "G", "temperature"]
NUISANCE = "seed"
CAT = {"model_family", "task_slice"}          # one-hot
NUM = {"G", "temperature", "seed"}            # numeric (G -> log2)
N_BOOT = 2000
FOLD_SEEDS = list(range(8))
N_SPLITS = 8
RNG = np.random.default_rng(20260704)


def load():
    rows = list(csv.DictReader(open(CELLS), delimiter="\t"))
    return rows


def encode(rows, fields):
    """Build design matrix over the given field subset."""
    cols = []
    names = []
    for f in fields:
        if f in CAT:
            vals = sorted(set(r[f] for r in rows))
            # drop-first to avoid collinearity (RF is fine either way)
            for v in vals[1:]:
                cols.append([1.0 if r[f] == v else 0.0 for r in rows])
                names.append(f"{f}={v}")
        else:
            if f == "G":
                cols.append([math.log2(float(r[f])) for r in rows])
                names.append("log2G")
            else:
                cols.append([float(r[f]) for r in rows])
                names.append(f)
    if not cols:
        return np.zeros((len(rows), 0)), names
    return np.array(cols, dtype=float).T, names


def oof_predict(X, y):
    """Averaged out-of-fold predictions over several fold seeds."""
    n = len(y)
    if X.shape[1] == 0:  # no predictors -> mean model
        return np.full(n, y.mean())
    acc = np.zeros(n)
    for fs in FOLD_SEEDS:
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=fs)
        pred = np.zeros(n)
        for tr, te in kf.split(X):
            m = RandomForestRegressor(
                n_estimators=200, min_samples_leaf=3,
                max_features=0.8, random_state=fs, n_jobs=1)
            m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        acc += pred
    return acc / len(FOLD_SEEDS)


def r2(y, pred, idx=None):
    if idx is not None:
        y, pred = y[idx], pred[idx]
    ybar = y.mean()
    ss_tot = np.sum((y - ybar) ** 2)
    if ss_tot == 0:
        return float("nan")
    ss_res = np.sum((y - pred) ** 2)
    return 1.0 - ss_res / ss_tot


def boot_ci(fn, n, reps=N_BOOT):
    stats = np.empty(reps)
    for b in range(reps):
        idx = RNG.integers(0, n, n)
        stats[b] = fn(idx)
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return float(np.nanmean(stats)), float(lo), float(hi)


def main():
    rows = load()
    n = len(rows)
    tsv_rows = []
    summary = {"n_cells": n, "n_boot": N_BOOT, "fold_seeds": len(FOLD_SEEDS),
               "n_splits": N_SPLITS, "targets": {}}

    for tgt in TARGETS:
        y = np.array([float(r[tgt]) for r in rows])
        # Full model over all disclosed stack fields.
        Xfull, _ = encode(rows, FIELDS)
        oof_full = oof_predict(Xfull, y)
        r2_full = r2(y, oof_full)
        m, lo, hi = boot_ci(lambda idx: r2(y, oof_full, idx), n)
        tsv_rows.append(dict(target=tgt, model="full_stack",
                             field_omitted="(none)", r2=round(r2_full, 4),
                             dr2=0.0, dr2_lo="", dr2_hi="",
                             r2_ci_lo=round(lo, 4), r2_ci_hi=round(hi, 4)))
        tgt_sum = {"r2_full": round(r2_full, 4),
                   "r2_full_ci": [round(lo, 4), round(hi, 4)],
                   "fields": {}}

        # Leave-one-field-out.
        oof_abl = {}
        for f in FIELDS:
            sub = [x for x in FIELDS if x != f]
            Xa, _ = encode(rows, sub)
            oof_abl[f] = oof_predict(Xa, y)
            r2_a = r2(y, oof_abl[f])
            dr2 = r2_full - r2_a
            # paired bootstrap on dR^2
            dm, dlo, dhi = boot_ci(
                lambda idx: r2(y, oof_full, idx) - r2(y, oof_abl[f], idx), n)
            tsv_rows.append(dict(target=tgt, model=f"drop_{f}",
                                 field_omitted=f, r2=round(r2_a, 4),
                                 dr2=round(dr2, 4), dr2_lo=round(dlo, 4),
                                 dr2_hi=round(dhi, 4), r2_ci_lo="", r2_ci_hi=""))
            tgt_sum["fields"][f] = {
                "r2_ablated": round(r2_a, 4), "dr2": round(dr2, 4),
                "dr2_ci": [round(dlo, 4), round(dhi, 4)],
                "load_bearing": bool(dlo > 0)}

        # Nuisance control: add seed on top of full stack.
        Xseed, _ = encode(rows, FIELDS + [NUISANCE])
        oof_seed = oof_predict(Xseed, y)
        r2_seed = r2(y, oof_seed)
        dm, dlo, dhi = boot_ci(
            lambda idx: r2(y, oof_seed, idx) - r2(y, oof_full, idx), n)
        tsv_rows.append(dict(target=tgt, model="full_stack+seed",
                             field_omitted="(+seed)", r2=round(r2_seed, 4),
                             dr2=round(r2_seed - r2_full, 4), dr2_lo=round(dlo, 4),
                             dr2_hi=round(dhi, 4), r2_ci_lo="", r2_ci_hi=""))
        tgt_sum["seed_control"] = {
            "r2_full+seed": round(r2_seed, 4),
            "dr2_add_seed": round(r2_seed - r2_full, 4),
            "dr2_add_seed_ci": [round(dlo, 4), round(dhi, 4)]}

        # Label-only baseline (same label all cells -> mean model -> R^2 = 0).
        oof_lbl = oof_predict(np.zeros((n, 0)), y)
        tgt_sum["label_only_r2"] = round(r2(y, oof_lbl), 4)
        tsv_rows.append(dict(target=tgt, model="label_only",
                             field_omitted="(all stack)", r2=round(r2(y, oof_lbl), 4),
                             dr2=round(r2_full - r2(y, oof_lbl), 4), dr2_lo="",
                             dr2_hi="", r2_ci_lo="", r2_ci_hi=""))
        summary["targets"][tgt] = tgt_sum

    # Write outputs.
    cols = ["target", "model", "field_omitted", "r2", "dr2", "dr2_lo",
            "dr2_hi", "r2_ci_lo", "r2_ci_hi"]
    with open(os.path.join(OUTDIR, "p5_field_sufficiency.tsv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(tsv_rows)
    summary["generated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(summary, open(os.path.join(OUTDIR, "p5_field_sufficiency_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
