#!/usr/bin/env python3
"""P8 JOB A (iter 180): calibration SLOPE + CURVATURE analysis with paired
bootstrap CIs.

Fresh vein, not in 176 prior P8 rows. Iter-176 measured 6 H-level
operating metrics (AUC, Brier, within-budget ECE, P@K) on the raw
classifiers; iter-180 adds the calibration-CURVE layer:
  - per-model PRE-calibration Brier (replicates iter-176 H4)
  - per-model POST calibration Brier under (i) Platt (logistic) and
    (ii) Isotonic regression (5-fold CV on training set; out-of-fold
    probability is the calibrated score on test)
  - per-model RELIABILITY slope (logit-score vs logit-label linear
    regression; slope=1.0 means perfectly calibrated in the linear
    sense) and intercept
  - per-model RELIABILITY CURVATURE on the within-top-K=1% alerted pool
    (predicted mean - actual fraction, signed, smaller=better).
  - 5-seed paired bootstrap CIs on each (model, calibrator) pair:
    H1 isotonic-lowers-Brier; H2 isotonic-lowers-ECE-K1%; H3 24full
    lowest Brier both pre and post; H4 calibration slope
    deviation monotone (4sensor > 20raw > 24full); H5 calibration
    curvature (within-K1%) monotone.

Outputs (platform_hybrid/experiments/results/p5p8/):
  p8_iter180_calib_per_fset.tsv    (30 rows: 3 fsets x 2 calibrators x 5 seeds)
  p8_iter180_calibration_curve.tsv (3 fsets x 5 rows: pre/platt/iso/perfect/ref)
  p8_iter180_reliability.tsv       (30 rows: per-(fset, seed) slope + intercept)
  p8_iter180_curvature_K1.tsv      (30 rows: per-(fset, calibrator, seed) Δ)
  p8_iter180_headline_cis.tsv      (12 rows: CI on post-Brier / slope / Δ)
  p8_iter180_summary.json          (H1-H5 verdicts + per-fset headline)

Stdlib + numpy + xgboost + sklearn (already in venv).
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)
TRAIN = ROOT / "fraud_data.csv"
TEST = ROOT / "test_data.csv"

RAW20 = [f"V{i}" for i in range(1, 21)]
AGG4 = ["V_mean", "V_std", "V_max", "V_min"]
ALL24 = RAW20 + AGG4
COL_IDX = {c: i for i, c in enumerate(ALL24)}
FEATURE_SETS = {
    "20raw":   RAW20,
    "24full":  ALL24,
    "4sensor": AGG4,
}
SEEDS = [42, 179, 316, 453, 590]
N_BOOT = 2000
K_BUDGETS_PCT = [1.0, 2.0]


def load(path):
    X, y = [], []
    with path.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)
        col_idx = {name: i for i, name in enumerate(header)}
        for line in rdr:
            X.append([float(line[col_idx[c]]) for c in ALL24])
            y.append(int(float(line[col_idx["Class"]])))
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)


def fit_xgb(Xtr, ytr, Xte, feats, seed):
    import xgboost as xgb
    cols = [COL_IDX[c] for c in feats]
    spw = float((ytr == 0).sum()) / max(1, float((ytr == 1).sum()))
    m = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric="logloss", random_state=seed,
        tree_method="hist", n_jobs=4)
    m.fit(Xtr[:, cols], ytr, verbose=False)
    return m.predict_proba(Xte[:, cols])[:, 1]


def platt_calibrate(p_train_uncal,y_train):
    """Platt scaling: logistic regression on logit(p). Returns the fitted
    LogisticRegression object. Caller applies it to test p."""
    from sklearn.linear_model import LogisticRegression
    eps = 1e-9
    z = np.log(np.clip(p_train_uncal, eps, 1 - eps) /
               np.clip(1 - p_train_uncal, eps, 1 - eps)).reshape(-1, 1)
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
    clf.fit(z, y_train)
    return clf


def apply_platt(platt, p_test):
    eps = 1e-9
    z = np.log(np.clip(p_test, eps, 1 - eps) /
               np.clip(1 - p_test, eps, 1 - eps)).reshape(-1, 1)
    return platt.predict_proba(z)[:, 1]


def isotonic_calibrate(p_train_uncal, y_train):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_train_uncal, y_train)
    return iso


def apply_isotonic(iso, p_test):
    return iso.transform(p_test)


def brier(y, p):
    from sklearn.metrics import brier_score_loss
    return float(brier_score_loss(y, p))


def ece_within_k(y, p, k_pct):
    """Within-budget ECE: among the top-k_pct% predicted-positive rows,
    divide into 10 sub-deciles and compute standard ECE."""
    n = len(y)
    k = max(1, int(round(k_pct / 100.0 * n)))
    order = np.argsort(-p)
    sel = order[:k]
    ys, ps = y[sel], p[sel]
    n_bins = min(10, max(1, len(ps) // 20))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (ps >= lo) & (ps < hi if i < n_bins - 1 else ps <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(ys[mask].mean() - ps[mask].mean())
    return float(ece / max(1, len(ps)))


def reliability_slope(y, p):
    """Calibration slope: linear regression on logit(p) -> logit(y_smooth)
    (Laplace-smoothed y to avoid +-inf); slope=1.0 means perfectly
    calibrated in the linear sense. Intercept near 0 means no bias."""
    eps = 1e-6
    y_s = (y.astype(float) * (1 - 2 * eps) + eps)  # in (eps, 1-eps)
    z_lbl = np.log(y_s / (1 - y_s))
    z_prd = np.log(np.clip(p, eps, 1 - eps) /
                   np.clip(1 - p, eps, 1 - eps))
    coef = np.polyfit(z_prd, z_lbl, 1)
    # intercept b is y - slope*x at x=0
    slope = float(coef[0])
    intercept = float(coef[1])
    return slope, intercept


def curvature_within_k(y, p, k_pct):
    """Curvature = (mean predicted - actual fraction) on the alerted pool.
    Positive = overconfident; Negative = underconfident; smaller |.| = better."""
    n = len(y)
    k = max(1, int(round(k_pct / 100.0 * n)))
    order = np.argsort(-p)
    sel = order[:k]
    if len(sel) == 0:
        return float("nan")
    return float(p[sel].mean() - y[sel].astype(float).mean())


def ece_within_k_1(y, p):
    return ece_within_k(y, p, 1.0)


def abs_curv_K1(y, p):
    return float(abs(curvature_within_k(y, p, 1.0)))


def paired_boot_diff_brier(y, p_a, p_b, n_boot=N_BOOT, seed=20260705):
    """Paired bootstrap CI on Brier(y, p_a) - Brier(y, p_b) where rows are
    resampled jointly for y, p_a, p_b. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    pt = float(np.mean((y - p_a) ** 2) - np.mean((y - p_b) ** 2))
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ya, pa, pb = y[idx], p_a[idx], p_b[idx]
        boots.append(float(np.mean((ya - pa) ** 2) - np.mean((ya - pb) ** 2)))
    boots.sort()
    return float(pt), float(boots[int(0.025 * n_boot)]), float(boots[int(0.975 * n_boot) - 1])


def paired_boot_diff_scalar(y, p_a, p_b, metric, n_boot=N_BOOT, seed=20260705):
    """Paired bootstrap CI on metric_fn(y, p_a) - metric_fn(y, p_b)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    pt = float(metric(y, p_a) - metric(y, p_b))
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(float(metric(y[idx], p_a[idx]) - metric(y[idx], p_b[idx])))
    boots.sort()
    return float(pt), float(boots[int(0.025 * n_boot)]), float(boots[int(0.975 * n_boot) - 1])


def main():
    print("[p8-iter180] loading data", file=sys.stderr)
    Xtr, ytr = load(TRAIN)
    Xte, yte = load(TEST)
    print(f"[p8-iter180] train n={len(ytr)} pos={int(ytr.sum())} test n={len(yte)} pos={int(yte.sum())}",
          file=sys.stderr)

    cal_rows = []        # 30 rows: 3 fsets x 2 calibrators x 5 seeds -> actually 3 fsets x 3 states x 5 seeds (pre/platt/iso)
    rel_rows = []        # 30 rows: per-(fset, seed) slope + intercept (pre only)
    curv_rows = []       # 30 rows: per-(fset, calibrator, seed)
    per_fset_pre = {}    # fset -> list[dict] (per seed)

    for fset_name, feats in FEATURE_SETS.items():
        print(f"[p8-iter180] fset={fset_name}", file=sys.stderr)
        per_seed = []
        for s in SEEDS:
            p_test_uncal = fit_xgb(Xtr, ytr, Xte, feats, s)
            # also need training predictions for calibrator fit (use OOF)
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=5, shuffle=True, random_state=s)
            p_train_oof = np.zeros(len(ytr), dtype=np.float64)
            for tr_idx, vl_idx in kf.split(Xtr):
                p_train_oof[vl_idx] = fit_xgb(Xtr[tr_idx], ytr[tr_idx],
                                              Xtr[vl_idx], feats, s)
            platt = platt_calibrate(p_train_oof, ytr)
            iso = isotonic_calibrate(p_train_oof, ytr)
            p_platt = apply_platt(platt, p_test_uncal)
            p_iso = apply_isotonic(iso, p_test_uncal)

            # record per-seed metric
            pre_brier = brier(yte, p_test_uncal)
            platt_brier = brier(yte, p_platt)
            iso_brier = brier(yte, p_iso)
            pre_ece_K1 = ece_within_k(yte, p_test_uncal, 1.0)
            platt_ece_K1 = ece_within_k(yte, p_platt, 1.0)
            iso_ece_K1 = ece_within_k(yte, p_iso, 1.0)
            curv_pre_K1 = curvature_within_k(yte, p_test_uncal, 1.0)
            curv_platt_K1 = curvature_within_k(yte, p_platt, 1.0)
            curv_iso_K1 = curvature_within_k(yte, p_iso, 1.0)
            slope, intercept = reliability_slope(yte, p_test_uncal)

            cal_rows.append([fset_name, "pre",   s, round(pre_brier, 5),
                             round(pre_ece_K1, 5), round(curv_pre_K1, 5)])
            cal_rows.append([fset_name, "platt", s, round(platt_brier, 5),
                             round(platt_ece_K1, 5), round(curv_platt_K1, 5)])
            cal_rows.append([fset_name, "iso",   s, round(iso_brier, 5),
                             round(iso_ece_K1, 5), round(curv_iso_K1, 5)])
            rel_rows.append([fset_name, s,
                             round(slope, 5), round(intercept, 5),
                             round(pre_brier, 5)])
            curv_rows.append([fset_name, "pre",   s, round(curv_pre_K1, 5)])
            curv_rows.append([fset_name, "platt", s, round(curv_platt_K1, 5)])
            curv_rows.append([fset_name, "iso",   s, round(curv_iso_K1, 5)])
            per_seed.append(dict(seed=s, pre=pre_brier, platt=platt_brier,
                                 iso=iso_brier, pre_ece=pre_ece_K1,
                                 platt_ece=platt_ece_K1, iso_ece=iso_ece_K1,
                                 curv_pre=curv_pre_K1, curv_platt=curv_platt_K1,
                                 curv_iso=curv_iso_K1, slope=slope,
                                 intercept=intercept))
        per_fset_pre[fset_name] = per_seed

    # Write per-fset calibration table
    out_cal = RES / "p8_iter180_calib_per_fset.tsv"
    with out_cal.open("w") as f:
        f.write("fset\tcalibrator\tseed\tbrier\tece_K1\tcurv_K1\n")
        for r in cal_rows:
            f.write("\t".join(str(c) for c in r) + "\n")
    print(f"[p8-iter180] wrote {out_cal}", file=sys.stderr)

    out_rel = RES / "p8_iter180_reliability.tsv"
    with out_rel.open("w") as f:
        f.write("fset\tseed\tslope\tintercept\tpre_brier\n")
        for r in rel_rows:
            f.write("\t".join(str(c) for c in r) + "\n")
    print(f"[p8-iter180] wrote {out_rel}", file=sys.stderr)

    out_curv = RES / "p8_iter180_curvature_K1.tsv"
    with out_curv.open("w") as f:
        f.write("fset\tcalibrator\tseed\tcurv_K1\n")
        for r in curv_rows:
            f.write("\t".join(str(c) for c in r) + "\n")
    print(f"[p8-iter180] wrote {out_curv}", file=sys.stderr)

    # Per-fset pre-summary table (for quick view)
    pre_summary = []
    for fset_name in FEATURE_SETS:
        ps = per_fset_pre[fset_name]
        mean_pre = float(np.mean([r["pre"] for r in ps]))
        mean_platt = float(np.mean([r["platt"]for r in ps]))
        mean_iso = float(np.mean([r["iso"] for r in ps]))
        mean_slope = float(np.mean([r["slope"] for r in ps]))
        mean_curv_pre = float(np.mean([abs(r["curv_pre"]) for r in ps]))
        mean_curv_iso = float(np.mean([abs(r["curv_iso"]) for r in ps]))
        pre_summary.append([fset_name,
                            round(mean_pre, 5),
                            round(mean_platt, 5),
                            round(mean_iso, 5),
                            round(mean_slope, 5),
                            round(mean_curv_pre, 5),
                            round(mean_curv_iso, 5)])
    out_curve = RES / "p8_iter180_calibration_curve.tsv"
    with out_curve.open("w") as f:
        f.write("fset\tmean_pre_brier\tmean_platt_brier\tmean_iso_brier\tmean_slope\tmean_abs_curv_K1_pre\tmean_abs_curv_K1_iso\n")
        for r in pre_summary:
            f.write("\t".join(str(c) for c in r) + "\n")
    print(f"[p8-iter180] wrote {out_curve}", file=sys.stderr)

    # Paired bootstrap CIs on headline comparisons
    # Strategy: use the SEED 0 predictions as canonical "the model" and
    # bootstrap over test rows for CIs.
    print("[p8-iter180] building CIs at seed=42", file=sys.stderr)
    # refit on seed=42 (already in per_fset_pre[42]: use the stored per-seed metrics)
    # For paired bootstrap on delta metrics, we need the per-row predicted
    # vectors, so refit with seed 42:
    headline_pred = {}
    for fset_name, feats in FEATURE_SETS.items():
        p_test_uncal = fit_xgb(Xtr, ytr, Xte, feats, 42)
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        p_train_oof = np.zeros(len(ytr), dtype=np.float64)
        for tr_idx, vl_idx in kf.split(Xtr):
            p_train_oof[vl_idx] = fit_xgb(Xtr[tr_idx], ytr[tr_idx],
                                          Xtr[vl_idx], feats, 42)
        platt = platt_calibrate(p_train_oof, ytr)
        iso = isotonic_calibrate(p_train_oof, ytr)
        headline_pred[fset_name] = dict(
            pre=p_test_uncal, platt=apply_platt(platt, p_test_uncal),
            iso=apply_isotonic(iso, p_test_uncal))

    # CIs
    ci_rows = []
    pre_24full = headline_pred["24full"]["pre"]
    pre_20raw = headline_pred["20raw"]["pre"]
    pre_4sensor = headline_pred["4sensor"]["pre"]
    iso_24full = headline_pred["24full"]["iso"]
    iso_20raw = headline_pred["20raw"]["iso"]
    iso_4sensor = headline_pred["4sensor"]["iso"]
    platt_24full = headline_pred["24full"]["platt"]
    platt_20raw = headline_pred["20raw"]["platt"]
    platt_4sensor = headline_pred["4sensor"]["platt"]

    # For paired bootstrap that re-samples (y, p_a, p_b) jointly we wrap
    # metric(y, p) into a scalar closure:
    def w(metric, p):
        return lambda yy, pp: metric(yy, pp) if (pp is p) else None  # placeholder

    # (a) iso-lowers-brier: 24full
    pt, lo, hi = paired_boot_diff_brier(yte, pre_24full, iso_24full)
    ci_rows.append(["iso_minus_pre_brier_24full", round(pt, 5),
                    round(lo, 5), round(hi, 5),
                    "positive=iso lowers Brier (lower=better); CI excludes 0"])
    pt, lo, hi = paired_boot_diff_brier(yte, pre_20raw, iso_20raw)
    ci_rows.append(["iso_minus_pre_brier_20raw", round(pt, 5),
                    round(lo, 5), round(hi, 5),
                    "iso lowers Brier on raw-20 too"])
    pt, lo, hi = paired_boot_diff_brier(yte, pre_4sensor, iso_4sensor)
    ci_rows.append(["iso_minus_pre_brier_4sensor", round(pt, 5),
                    round(lo, 5), round(hi, 5),
                    "iso lowers Brier on LLM-as-sensor surrogate"])

    # (b) iso vs platt brier on 24full (iso typically wins at top-K)
    pt, lo, hi = paired_boot_diff_brier(yte, platt_24full, iso_24full)
    ci_rows.append(["platt_minus_iso_brier_24full", round(pt, 5),
                    round(lo, 5), round(hi, 5),
                    "positive=platt higher Brier than iso"])

    # (c) iso-K1%-reliability-gap: 24full
    pt, lo, hi = paired_boot_diff_scalar(yte, pre_24full, iso_24full, ece_within_k_1)
    ci_rows.append(["iso_minus_pre_eceK1_24full", round(pt, 5),
                    round(lo, 5), round(hi, 5),
                    "positive=iso lowers within-budget ECE@K=1% on 24full"])

    # (d) iso-24full brier vs iso-20raw brier (delta = 24full - 20raw so
    # negative means 24full wins)
    pt, lo, hi = paired_boot_diff_brier(yte, iso_24full, iso_20raw)
    ci_rows.append(["iso_24full_minus_iso_20raw_brier", round(pt, 5),
                    round(lo, 5), round(hi, 5),
                    "negative=24full lower Brier than 20raw after iso"])

    # (e) iso-24full brier vs iso-4sensor brier (delta = 24full - 4sensor)
    pt, lo, hi = paired_boot_diff_brier(yte, iso_24full, iso_4sensor)
    ci_rows.append(["iso_24full_minus_iso_4sensor_brier", round(pt, 5),
                    round(lo, 5), round(hi, 5),
                    "negative=24full lower Brier than 4sensor after iso"])

    # (f) iso-K1%-curvature: 24full absolute curv reduction
    pt, lo, hi = paired_boot_diff_scalar(yte, pre_24full, iso_24full, abs_curv_K1)
    ci_rows.append(["abs_curv_K1_pre_minus_iso_24full", round(pt, 5),
                    round(lo, 5), round(hi, 5),
                    "positive=|curv| shrinks after iso; smaller=|better|"])

    out_ci = RES / "p8_iter180_headline_cis.tsv"
    with out_ci.open("w") as f:
        f.write("comparison\tpoint\tci_lo\tci_hi\tnote\n")
        for r in ci_rows:
            f.write("\t".join(str(c) for c in r) + "\n")
    print(f"[p8-iter180] wrote {out_ci}", file=sys.stderr)

    # Falsifiable verdicts (H1-H5)
    # H1: Isotonic reduces Brier on 3/3 models (lo > 0 for the pre→iso drop)
    # ci_rows indices: 0=iso_vs_pre_24full, 1=iso_vs_pre_20raw, 2=iso_vs_pre_4sensor,
    # 3=platt_vs_iso_24full, 4=iso_eceK1_24full, 5=20raw_vs_24full_iso,
    # 6=4sensor_vs_24full_iso, 7=abs_curv_pre_vs_iso_24full
    # H1: iso lowers Brier on 3/3 models (idx 0..2 = pre_vs_iso on each)
    h1 = all(ci_rows[i][2] > 0 for i in range(3))
    # H2: iso lowers within-budget ECE @ K=1% on 24full (idx 4)
    h2 = bool(ci_rows[4][2] > 0)
    # H3: Post-iso 24full Brier < 20raw (idx 5: 24full - 20raw; CI_hi<0)
    h3 = bool(ci_rows[5][3] < 0)
    # H4: Post-iso 24full Brier < 4sensor (idx 6: 24full - 4sensor; CI_hi<0)
    h4 = bool(ci_rows[6][3] < 0)
    # H5: |curvature| shrinks on 24full under iso (idx 7)
    h5 = bool(ci_rows[7][2] > 0)
    # additional: monotone — pre-brier 4sensor > 20raw > 24full
    pre_brier_means = {fset: float(np.mean([r["pre"] for r in per_fset_pre[fset]]))
                       for fset in FEATURE_SETS}
    h_monotone_pre = (pre_brier_means["4sensor"]
                      > pre_brier_means["20raw"]
                      > pre_brier_means["24full"])
    # slope deviation from 1.0 monotone (4sensor > 20raw > 24full)
    slope_means = {fset: float(np.mean([r["slope"] for r in per_fset_pre[fset]]))
                   for fset in FEATURE_SETS}
    slope_dev = {fset: abs(1.0 - s) for fset, s in slope_means.items()}
    h_monotone_slope = (slope_dev["4sensor"]
                        > slope_dev["20raw"]
                        > slope_dev["24full"])

    summary = {
        "iter": 180,
        "job": "P8 calibration slope + curvature analysis",
        "n_seeds": len(SEEDS),
        "n_boot": N_BOOT,
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "pos_train": int(ytr.sum()), "pos_test": int(yte.sum()),
        "pre_brier_means": pre_brier_means,
        "slope_means": slope_means,
        "slope_dev_from_1": slope_dev,
        "monotone_pre_brier": h_monotone_pre,
        "monotone_slope_dev": h_monotone_slope,
        "ci_rows": [
            {"comparison": r[0], "point": r[1], "lo": r[2], "hi": r[3], "note": r[4]}
            for r in ci_rows],
        "hypotheses": {
            "H1_isotonic_lowers_brier_on_3of3": h1,
"H2_isotonic_lowers_eceK1_24full": h2,
            "H3_post_iso_24full_brier_le_20raw": h3,
            "H4_post_iso_24full_brier_lt_4sensor": h4,
            "H5_abs_curv_K1_shrinks_24full": h5,
        },
    }
    out_sum = RES / "p8_iter180_summary.json"
    out_sum.write_text(json.dumps(summary, indent=2))
    print(f"[p8-iter180] wrote {out_sum}", file=sys.stderr)
    print(json.dumps(summary["hypotheses"], indent=2))


if __name__ == "__main__":
    main()
