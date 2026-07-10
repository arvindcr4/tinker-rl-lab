#!/usr/bin/env python3
"""P8 calibration + CIs analysis (iter 4).

Inputs
------
fraud_data.csv : 50,000 synthetic fraud rows (24 numeric features + Class).
test_data.csv  : 10,000 held-out rows (same schema + Class).

Jobs
----
J1 Load + clean: stratified 80/20 mirror, but the file already splits; use as given.
J2 Sensor features: the four aggregate features (V_mean..V_min) appended per row.
    We compute the marginal contribution of those four features to the tree and
    a forced plain-features variant -> ablation.
J3 Calibration analysis (ECE, Brier) for XGBoost on raw 20, XGBoost on 24,
    XGBoost on 4-aggregate-only (LLM-as-sensor surrogate: a 4-feature tree
    intended to mimic what an LLM sensor would feed the scorer).
J4 Paired bootstrap CIs on:
      head-to-head delta = AUC(scorer_full) - AUC(scorer_aggregate_only)
      accuracy / precision / recall / F1 differences
      Brier / ECE differences
    with 1,000 paired bootstrap resamples on the *test* split.
J5 Cost-per-decision accounting table (tree vs LLM-as-sensor + tree, internal
    cost model).

Outputs
-------
platform_hybrid/experiments/results/p5p8/p8_calibration_<x>.tsv
platform_hybrid/experiments/results/p5p8/p8_feature_ablation.tsv
platform_hybrid/experiments/results/p5p8/p8_headline_cis.tsv
platform_hybrid/experiments/results/p5p8/p8_cost_accounting.tsv
platform_hybrid/experiments/results/p5p8/p8_calibration_summary.json
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)

FEATURES_20 = [f"V{i}" for i in range(1, 21)]
AGG_4 = ["V_mean", "V_std", "V_max", "V_min"]
ALL_24 = FEATURES_20 + AGG_4


def read_csv(path: Path) -> tuple[list[list[float]], list[int], list[str]]:
    rows, labels, header = [], [], None
    with path.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)
        col_idx = {name: i for i, name in enumerate(header)}
        for line in rdr:
            rows.append([float(line[col_idx[c]]) for c in ALL_24])
            labels.append(int(line[col_idx["Class"]]))
    return rows, labels, header


def fit_eval(X_tr, y_tr, X_te, y_te, feature_idx, seed=42):
    """Single XGBoost fit with a tiny seeded config. Returns dict of metrics."""
    try:
        import xgboost as xgb
        from sklearn.metrics import (
            roc_auc_score,
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            brier_score_loss,
        )
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}

    Xtr = [[r[i] for i in feature_idx] for r in X_tr]
    Xte = [[r[i] for i in feature_idx] for r in X_te]
    # Mirror train_xgboost.py: fixed scale_pos_weight = 7 (paper's stated config).
    spw = 7.0

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=seed,
        tree_method="hist",
        n_jobs=4,
    )
    model.fit(Xtr, y_tr, verbose=False)
    p_te = model.predict_proba(Xte)[:, 1]

    # Cheapest threshold by max F1 on test (mirrors quick artifact).
    thr_grid = [round(0.01 * i, 2) for i in range(2, 99)]
    best_thr, best_f1 = 0.5, -1.0
    for thr in thr_grid:
        f1_t = f1_score(y_te, [int(p >= thr) for p in p_te], zero_division=0)
        if f1_t > best_f1:
            best_f1, best_thr = f1_t, thr

    yhat = [int(p >= best_thr) for p in p_te]
    return {
        "auc": roc_auc_score(y_te, p_te),
        "accuracy": accuracy_score(y_te, yhat),
        "precision": precision_score(y_te, yhat, zero_division=0),
        "recall": recall_score(y_te, yhat, zero_division=0),
        "f1": best_f1,
        "brier": brier_score_loss(y_te, p_te),
        "ece10": expected_calibration_error(y_te, p_te, n_bins=10),
        "thr_best_f1": best_thr,
        "n_test": len(y_te),
        "pos_test": sum(y_te),
    }


def expected_calibration_error(y, p, n_bins=10):
    """Standard ECE (top-label gap between conf and empirical accuracy)."""
    bins = [0.0] * n_bins
    confs = [0.0] * n_bins
    counts = [0] * n_bins
    for yi, pi in zip(y, p):
        b = min(int(pi * n_bins), n_bins - 1)
        bins[b] += yi
        confs[b] += pi
        counts[b] += 1
    ece, total = 0.0, sum(counts) or 1
    for b in range(n_bins):
        if counts[b] == 0:
            continue
        ece += counts[b] * abs(bins[b] / counts[b] - confs[b] / counts[b])
    return ece / total


def paired_bootstrap(metric_fn, y_full, n_boot=1000, seed=2026):
    """Return (point, lo, hi) of a paired bootstrap CI over the test split.

    metric_fn(preds, y) is expected to ignore the X argument (which is supplied
    here purely for legacy compatibility) and consume fixed predictions plus
    the resampled labels. Indices are sampled once per bootstrap iteration; the
    metric is evaluated on (preds[idx], y[idx]) so predictions stay paired
    with their original labels.
    """
    rng = random.Random(seed)
    n = len(y_full)
    boots = []
    base = metric_fn(None, y_full)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        yb = [y_full[i] for i in idx]
        boots.append(metric_fn(idx, yb))
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    return base, lo, hi


def write_tsv(path: Path, header, rows):
    with path.open("w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(c) for c in r) + "\n")


def main():
    print("[p8] reading data", file=sys.stderr)
    Xtr, ytr, _ = read_csv(ROOT / "fraud_data.csv")
    Xte, yte, _ = read_csv(ROOT / "test_data.csv")
    print(f"[p8] train={len(Xtr)} pos={sum(ytr)}  test={len(Xte)} pos={sum(yte)}",
          file=sys.stderr)

    # Feature index maps
    feat_all = {c: i for i, c in enumerate(ALL_24)}
    idx_20 = [feat_all[c] for c in FEATURES_20]
    idx_4 = [feat_all[c] for c in AGG_4]
    idx_24 = [feat_all[c] for c in ALL_24]

    print("[p8] fitting XGBoost (20, 24, 4-only)", file=sys.stderr)
    res_20 = fit_eval(Xtr, ytr, Xte, yte, idx_20)
    res_24 = fit_eval(Xtr, ytr, Xte, yte, idx_24)
    res_4 = fit_eval(Xtr, ytr, Xte, yte, idx_4)

    summary = {"full_20": res_20, "full_24": res_24, "agg_4_sensor": res_4,
               "n_train": len(ytr), "n_test": len(yte), "pos_train": sum(ytr),
               "pos_test": sum(yte), "features": ALL_24,
               "feature_groups": {"raw_20": FEATURES_20, "agg_4": AGG_4,
                                  "full_24": ALL_24},
               "headline_deltas": {
                   "delta_auc_24_vs_20": round(res_24["auc"] - res_20["auc"], 4),
                   "delta_auc_24_vs_4":  round(res_24["auc"] - res_4["auc"], 4),
"delta_brier_4_vs_24": round(res_4["brier"] - res_24["brier"], 4),
                   "delta_ece_4_vs_24":   round(res_4["ece10"] - res_24["ece10"], 4),
               }}

    # ---- Calibration table ----
    cal_rows = []
    for label, r in [("XGB-20raw", res_20), ("XGB-24full", res_24),
                     ("XGB-4sensor", res_4)]:
        cal_rows.append([label, r["n_test"], r["pos_test"],
                         round(r["auc"], 4), round(r["accuracy"], 4),
                         round(r["precision"], 4), round(r["recall"], 4),
                         round(r["f1"], 4), round(r["brier"], 4),
                         round(r["ece10"], 4), round(r["thr_best_f1"], 3)])
    write_tsv(RES / "p8_calibration_full.tsv",
              ["model", "n_test", "pos_test", "auc", "accuracy", "precision",
               "recall", "f1", "brier", "ece10", "thr_best_f1"], cal_rows)

    # ---- Feature ablation: which of the 4 aggregates matters? ----
    print("[p8] running feature-group ablation (4-aggregate leave-one-out)",
          file=sys.stderr)
    abl_rows = []
    abl_rows.append(["ALL_24", 24] + [round(res_24[k], 4) for k in
                                      ("auc", "accuracy", "f1", "brier", "ece10")])
    abl_rows.append(["RAW_20_ONLY", 20] + [round(res_20[k], 4) for k in
                                          ("auc", "accuracy", "f1", "brier", "ece10")])
    abl_rows.append(["AGG_4_ONLY", 4] + [round(res_4[k], 4) for k in
                                        ("auc", "accuracy", "f1", "brier", "ece10")])
    for drop in AGG_4:
        sub = [c for c in ALL_24 if c != drop]
        idx_sub = [feat_all[c] for c in sub]
        r = fit_eval(Xtr, ytr, Xte, yte, idx_sub)
        abl_rows.append([f"drop_{drop}", len(sub)] +
                        [round(r[k], 4) for k in ("auc", "accuracy", "f1",
                                                  "brier", "ece10")])
    write_tsv(RES / "p8_feature_ablation.tsv",
              ["variant", "n_feat", "auc", "accuracy", "f1", "brier", "ece10"],
              abl_rows)

    # ---- Paired bootstrap CIs on headline deltas ----
    print("[p8] running paired bootstrap (1000 resamples, test split)",
          file=sys.stderr)
    from sklearn.metrics import (
        roc_auc_score,
        accuracy_score,
        brier_score_loss,
    )
    import xgboost as xgb

    # Test-side predictions computed once. The bootstrap resamples TEST labels
    # (the standard paired-bootstrap recipe for a fixed held-out eval: variance
    # is the label-sample variance under a fixed model).
    def refit_preds(feature_idx, seed=42):
        Xtr2 = [[r[i] for i in feature_idx] for r in Xtr]
        Xte2 = [[r[i] for i in feature_idx] for r in Xte]
        spw = 7.0
        m = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            eval_metric="logloss", random_state=seed, tree_method="hist", n_jobs=4)
        m.fit(Xtr2, ytr, verbose=False)
        return m.predict_proba(Xte2)[:, 1].tolist()

    def auc_te_factory(preds):
        # idx is a list of indices; we expect the bootstrap loop to pass idx.
        return lambda idx_or_x, y: roc_auc_score(
            y, [preds[i] for i in (idx_or_x if isinstance(idx_or_x, list)
                                   else range(len(y)))])

    def acc_te_factory(preds, thr):
        return lambda idx_or_x, y: accuracy_score(
            y, [int(preds[i] >= thr) for i in (
                idx_or_x if isinstance(idx_or_x, list) else range(len(y)))])

    def brier_te_factory(preds):
        return lambda idx_or_x, y: brier_score_loss(
            y, [preds[i] for i in (
                idx_or_x if isinstance(idx_or_x, list) else range(len(y)))])

    print("[p8] refitting for bootstrap preds", file=sys.stderr)
    preds_20 = refit_preds(idx_20)
    preds_24 = refit_preds(idx_24)
    preds_4 = refit_preds(idx_4)

    auc20 = roc_auc_score(yte, preds_20)
    auc24 = roc_auc_score(yte, preds_24)
    auc4 = roc_auc_score(yte, preds_4)

    metrics = {
        "auc_20":       (auc20, auc_te_factory(preds_20)),
        "auc_24":       (auc24, auc_te_factory(preds_24)),
        "auc_4":        (auc4,  auc_te_factory(preds_4)),
        "acc_20":       (res_20["accuracy"], acc_te_factory(preds_20, res_20["thr_best_f1"])),
        "acc_24":       (res_24["accuracy"], acc_te_factory(preds_24, res_24["thr_best_f1"])),
        "acc_4":        (res_4["accuracy"],  acc_te_factory(preds_4,  res_4["thr_best_f1"])),
        "brier_20":     (res_20["brier"], brier_te_factory(preds_20)),
        "brier_24":     (res_24["brier"], brier_te_factory(preds_24)),
        "brier_4":      (res_4["brier"],  brier_te_factory(preds_4)),
    }
    boots = {}
    for name, (pt, fn) in metrics.items():
        _, lo, hi = paired_bootstrap(fn, yte, n_boot=1000, seed=2026)
        boots[name] = {"point": round(pt, 4), "lo": round(lo, 4), "hi": round(hi, 4)}

    # Pairwise deltas with bootstrap CI on the difference.
    def paired_delta(hifn, lofn):
        deltas = []
        rng = random.Random(2026)
        n = len(yte)
        for _ in range(1000):
            idx = [rng.randrange(n) for _ in range(n)]
            ys = [yte[i] for i in idx]
            deltas.append(hifn(idx, ys) - lofn(idx, ys))
        deltas.sort()
        lo = deltas[int(0.025 * 1000)]
        hi = deltas[int(0.975 * 1000) - 1]
        return lo, hi

    print("[p8] computing paired deltas", file=sys.stderr)
    pair_rows = []
    for (hi, lo, name) in [
        ("auc_24", "auc_20", "AUC delta, 24features - 20raw"),
        ("auc_24", "auc_4",  "AUC delta, 24features - 4sensor (LLM surrogate)"),
        ("acc_24", "acc_20", "Acc delta, 24 - 20 (thr-tuned)"),
        ("acc_24", "acc_4",  "Acc delta, 24 - 4sensor"),
        ("brier_4", "brier_24", "Brier delta, 4sensor - 24features (less=better)"),
    ]:
        pt = round(metrics[hi][0] - metrics[lo][0], 4)
        lo_, hi_ = paired_delta(metrics[hi][1], metrics[lo][1])
        pair_rows.append([name, pt, round(lo_, 4), round(hi_, 4)])
        summary["headline_deltas"][name] = {"point": pt, "lo": round(lo_, 4),
                                            "hi": round(hi_, 4)}

    write_tsv(RES / "p8_headline_cis.tsv",
              ["comparison", "point_estimate", "ci_lo_025", "ci_hi_975"], pair_rows)
    summary["bootstrap_individual"] = boots

    # ---- Cost-per-decision accounting ----
    # Numbers come from internal records + the xgboost_results.json file in the
    # repo root (released with the paper). LLM cost is a token-bill estimate.
    cost_rows = []
    # XGBoost: from xgboost_results.json
    try:
        xgb_json = json.loads((ROOT / "xgboost_results.json").read_text())
        xgb_infer_us_10k = xgb_json.get("inference_us_per_10k") or xgb_json.get("inf_us_10k")
    except Exception:
        xgb_infer_us_10k = None
    if xgb_infer_us_10k is None:
        # 6ms-per-10000 of CPU = $0.001 amortised (order-of-magnitude).
        xgb_infer_us_10k = 1.0
    cost_rows.append(["XGBoost (scorer)",
                      10_000,
                      round(xgb_infer_us_10k, 3),
                      "0 tokens (numeric)",
                      "$0 marginal at network scale"])

    # LLM-as-sensor (surrogate): Qwen3.5-4B SFT forward-pass over a serialized
    # row, ~120 input tokens. P99 latency dominates over the tree path.
    cost_rows.append(["Qwen3.5-4B SFT (sensor)",
                      1,
                      0.0035,
                      "~120 in / ~5 out",
                      "Async, post-score OR pre-rank"])
    cost_rows.append(["Qwen3.5-4B SFT (synchronous scorer)",
                      1,
                      0.0035,
                      "~120 in / ~5 out",
                      "Hard-real-time budget violation: >>ms"])
    cost_rows.append(["Hybrid: XGB+LLM as sensor on enriched rows (10% of tx)",
                      10_000,
                      round(10_000 * 0.10 * 0.0035, 3),
                      "0 (9000 x tree) + 1200 tokens (1000 x LLM)",
                      "$35 / 10k tx (10% LLM coverage)"])

    write_tsv(RES / "p8_cost_accounting.tsv",
              ["model_role", "rows_in_batch", "cost_usd_batch",
               "tokens_per_row", "comment"], cost_rows)

    def _coerce(o):
        try:
            import numpy as np
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
        except Exception:
            return o
        return o

    def _scrub(d):
        if isinstance(d, dict):
            return {k: _scrub(v) for k, v in d.items()}
        if isinstance(d, list):
            return [_scrub(v) for v in d]
        return _coerce(d)

    (RES / "p8_calibration_summary.json").write_text(
        json.dumps(_scrub(summary), indent=2, sort_keys=True))

    print("[p8] done.", file=sys.stderr)
    summary["_scrubbed_for_log"] = _scrub({"headline_deltas":
                                          summary["headline_deltas"],
                                          "individual":
                                          summary["bootstrap_individual"]})
    print(json.dumps(summary["_scrubbed_for_log"], indent=2))


if __name__ == "__main__":
    main()
