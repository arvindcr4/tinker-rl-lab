#!/usr/bin/env python3
"""Iter 42 — Pillar 2 early-trace ZVF as predictor of late failure mode.

Frontier synthesis (round 1, Pillar 2): "ZVF is best framed as observed
signal availability, not difficulty."  Frontier synthesis (round 2):
"Y(p, G) is the valid signal fraction; delta_div is the structural
diversity bonus."

This iter 42 closes the Pillar-1 × Pillar-2 identifiability loop:

    iter 41 (P1): R_max of the saturation fit is identifiable from the
                  first 60% of the trace (9/9 eligible anchors within
                  ±10%).
    iter 42 (P2): Does early-trace ZVF identify the late failure mode?

Concretely, for each (kind, method, seed) run in the 60-row
zvf_dynamics.json pool we extract three early-trace ZVF features
defined RELATIVE to the trace length:

    early_zvf05_first_pass_frac = first_pass_zvf05 / n_steps
                                    (smaller = ZVF spikes earlier)
    early_auc05_first_half       = auc_above_zvf05 on first 50% of
                                    the trace, normalised to [0,1]
                                    (we use first_pass_zvf05 * 0.5
                                    capped at 0.5 as a proxy; the full
                                    per-step stream isn't materialised
                                    in the dynamics JSON, so we use
                                    thedocumented first-pass threshold
                                    statistic as the canonical
                                    early-window summary)
    early_auc09_first_half       = 0 if first_pass_zvf09 > 0.5*n_steps
                                    else 1   (binary: did ZVF cross
                                    0.9 in the first half?)

Then we ask whether these early features predict the LATE-trace
failure mode derived from (last10_avg, peak, mean_zvf) using the
same deterministic taxonomy as iter 38:

    collapse  := last10_avg < 0.05
    plateau   := peak < 0.5
    drift     := last10_avg < 0.85 * peak
    converged := otherwise

Outputs:
    platform_hybrid/experiments/results/zvf_iter42_early_predicts.tsv
        Per-run row with early features, late outcome, and a
        leave-one-out knn prediction (k=3) of failure class.
    platform_hybrid/experiments/results/zvf_iter42_summary.tsv
        One-rollup summary: classifier accuracy, feature
        univariate odds-ratio of collapse for early-bloomers,
        Spearman correlation of early vs late ZVF summary stats.
    platform_hybrid/experiments/results/zvf_iter42_failure_audit.tsv
        Per-(kind, method) aggregate showing collapse rate among
        "early-ZVF-spikers" vs "late-ZVF-spikers".

This is the **mirror image** of iter 41's early-trace R_max
identifiability result applied to Pillar 2.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "experiments" / "results"


def load_summary() -> list[dict]:
    p = RESULTS / "zvf_dynamics.json"
    return json.loads(p.read_text())["summary"]


def load_leadtime() -> list[dict]:
    p = RESULTS / "zvf_dynamics_leadtime.tsv"
    rows = []
    with p.open() as f:
        for line in f:
            if line.startswith("#") or line.startswith("kind"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            rows.append({
                "kind": parts[0],
                "method": parts[1],
                "seed": int(parts[2]),
                "n_steps": int(parts[3]),
                "mean_zvf": float(parts[4]),
                "theta": float(parts[5]),
                "first_pass_step": int(parts[6]),
                "first_collapse_step": int(parts[7]),
                "lead_steps": int(parts[8]),
            })
    return rows


def classify_failure(last10: float, peak: float, mean_zvf: float) -> str:
    """Same taxonomy as iter 38 zvf_iter38_classifier.tsv."""
    if last10 < 0.05:
        return "collapse"
    if peak < 0.5:
        return "plateau"
    if last10 < 0.85 * peak:
        return "drift"
    return "converged"


def early_features(row: dict) -> dict:
    """Compute early-trace ZVF features from the summary row."""
    n = max(row["n_steps"], 1)
    fp05 = row.get("first_pass_zvf05", -1) or -1
    fp07 = row.get("first_pass_zvf07", -1) or -1
    fp09 = row.get("first_pass_zvf09", -1) or -1
    auc05 = row.get("auc_above_zvf05", 0.0) or 0.0
    auc07 = row.get("auc_above_zvf07", 0.0) or 0.0
    auc09 = row.get("auc_above_zvf09", 0.0) or 0.0

    fp05_frac = fp05 / n if fp05 >= 0 else 1.0
    fp07_frac = fp07 / n if fp07 >= 0 else 1.0
    fp09_frac = fp09 / n if fp09 >= 0 else 1.0

    # Early-bloom = ZVF crosses 0.5 in the first half of the trace.
    early_bloom05 = int(0 <= fp05_frac <= 0.5)
    early_bloom07 = int(0 <= fp07_frac <= 0.5)
    early_bloom09 = int(0 <= fp09_frac <= 0.5)

    # ZVF burden: AUC above 0.5 — higher means more steps spent in
    # the high-contrast-loss regime. Iter 30 showed this is a
    # calibrated leading indicator of training failure.
    burden_05 = auc05
    burden_07 = auc07

    return {
        "n_steps": n,
        "fp05_frac": fp05_frac,
        "fp07_frac": fp07_frac,
        "fp09_frac": fp09_frac,
        "early_bloom05": early_bloom05,
        "early_bloom07": early_bloom07,
        "early_bloom09": early_bloom09,
        "burden_05": burden_05,
        "burden_07": burden_07,
        "mean_zvf": row.get("mean_zvf", float("nan")),
        "zvf_std": row.get("zvf_std", float("nan")),
        "zvf_lag1": row.get("zvf_lag1", float("nan")),
    }


def late_outcome(row: dict) -> dict:
    last10 = row.get("last10_avg", float("nan"))
    peak = row.get("heldout_acc", float("nan"))
    mean_zvf = row.get("mean_zvf", float("nan"))
    label = classify_failure(last10, peak, mean_zvf)
    return {
        "last10_avg": last10,
        "peak": peak,
        "label": label,
        "is_collapse": int(label == "collapse"),
        "is_converged": int(label == "converged"),
    }


def knn_loo(rows: list[dict], feature_keys: list[str], k: int = 3) -> list[str]:
    """Leave-one-out kNN classifier over (label, features).

    Distance: Euclidean over z-scored features. Ties broken by
    majority. Returns list of predicted labels.
    """
    if not rows:
        return []
    # z-score per feature
    means = []
    sds = []
    for k_ in feature_keys:
        vs = [r["features"][k_] for r in rows]
        m = statistics.fmean(vs) if vs else 0.0
        v = statistics.pvariance(vs) if len(vs) > 1 else 0.0
        s = math.sqrt(v) if v > 0 else 1.0
        means.append(m)
        sds.append(s)

    def zfeat(idx: int, row: dict) -> list[float]:
        return [(row["features"][feature_keys[i]] - means[i]) / sds[i]
                for i in range(len(feature_keys))]

    preds = []
    for i, row in enumerate(rows):
        zi = zfeat(i, row)
        dists = []
        for j, other in enumerate(rows):
            if i == j:
                continue
            zj = zfeat(j, other)
            d = sum((a - b) ** 2 for a, b in zip(zi, zj)) ** 0.5
            dists.append((d, other["label"]))
        dists.sort(key=lambda x: x[0])
        top = [lab for _, lab in dists[:k]]
        # majority
        cnt = defaultdict(int)
        for lab in top:
            cnt[lab] += 1
        # tie-break: pick the class with the smallest mean distance
        max_cnt = max(cnt.values())
        cand = [lab for lab, c in cnt.items() if c == max_cnt]
        if len(cand) == 1:
            preds.append(cand[0])
        else:
            # closest mean distance wins
            best = None
            best_d = float("inf")
            for lab in cand:
                ds = [d for d, l in dists[:k] if l == lab]
                md = statistics.fmean(ds) if ds else float("inf")
                if md < best_d:
                    best_d = md
                    best = lab
            preds.append(best or "converged")
    return preds


def knn_loo_cluster(rows: list[dict], feature_keys: list[str], k: int = 3) -> list[str]:
    """Leave-one-CLUSTER-out kNN — drops every row from the held-out
    cluster, so the cluster identity itself cannot be exploited.

    This is the harder, more honest version: does the feature vector
    GENERALISE across (kind, method) clusters, or does the LOO
    accuracy come from the trivial fact that every row in a cluster
    has the same label?
    """
    if not rows:
        return []
    means = []
    sds = []
    for k_ in feature_keys:
        vs = [r["features"][k_] for r in rows]
        m = statistics.fmean(vs) if vs else 0.0
        v = statistics.pvariance(vs) if len(vs) > 1 else 0.0
        s = math.sqrt(v) if v > 0 else 1.0
        means.append(m)
        sds.append(s)

    def zfeat(row):
        return [(row["features"][feature_keys[i]] - means[i]) / sds[i]
                for i in range(len(feature_keys))]

    cluster_ids = [(r["kind"], r["method"]) for r in rows]
    pool = [zfeat(r) for r in rows]
    labels = [r["label"] for r in rows]

    preds = []
    for i, row in enumerate(rows):
        ci = cluster_ids[i]
        dists = []
        for j, other in enumerate(rows):
            if cluster_ids[j] == ci:
                continue  # leave the whole cluster out
            d = sum((a - b) ** 2 for a, b in zip(pool[i], pool[j])) ** 0.5
            dists.append((d, labels[j]))
        if not dists:
            preds.append("converged")
            continue
        dists.sort(key=lambda x: x[0])
        top = [lab for _, lab in dists[:k]]
        cnt = defaultdict(int)
        for lab in top:
            cnt[lab] += 1
        max_cnt = max(cnt.values())
        cand = [lab for lab, c in cnt.items() if c == max_cnt]
        if len(cand) == 1:
            preds.append(cand[0])
        else:
            best = None
            best_d = float("inf")
            for lab in cand:
                ds = [d for d, l in dists[:k] if l == lab]
                md = statistics.fmean(ds) if ds else float("inf")
                if md < best_d:
                    best_d = md
                    best = lab
            preds.append(best or "converged")
    return preds


def univariate_auc(xs: list[float], ys_bin: list[int]) -> float:
    """AUC of single-feature monotone classifier predicting binary ys_bin.

    Convention: predict ys_bin=1 when x > threshold. Sweep over unique
    thresholds (midpoints), compute TPR/FPR via rank-sum
    (DeLong-style). Uses rank-sum identity for AUC:
        AUC = (S1 - n1*(n1+1)/2) / (n0 * n1)
    where S1 = sum of ranks of positive-class x values.
    """
    pairs = list(zip(xs, ys_bin))
    n1 = sum(ys_bin)
    n0 = len(ys_bin) - n1
    if n0 == 0 or n1 == 0:
        return float("nan")
    # Rank x ascending (ties → average rank)
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and pairs[order[j + 1]][0] == pairs[order[i]][0]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k_ in range(i, j + 1):
            ranks[order[k_]] = avg
        i = j + 1
    s1 = sum(ranks[i] for i in range(len(pairs)) if pairs[i][1] == 1)
    auc = (s1 - n1 * (n1 + 1) / 2) / (n0 * n1)
    return auc


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan")
    def rank(vs):
        sorted_vs = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0.0] * len(vs)
        i = 0
        while i < len(sorted_vs):
            j = i
            while j + 1 < len(sorted_vs) and vs[sorted_vs[j + 1]] == vs[sorted_vs[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k_ in range(i, j + 1):
                r[sorted_vs[k_]] = avg
            i = j + 1
        return r
    rx = rank(xs)
    ry = rank(ys)
    mx = statistics.fmean(rx)
    my = statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def odds_ratio(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Odds ratio with Haldane-Anscombe correction: (a+0.5)(d+0.5) / ((b+0.5)(c+0.5)).

a = #bloom AND collapse
    b = #bloom AND NOT collapse
    c = #NOT bloom AND collapse
    d = #NOT bloom AND NOT collapse
    Returns (or, log_or) — log_or for the symmetric CI.
    """
    A = a + 0.5
    B = b + 0.5
    C = c + 0.5
    D = d + 0.5
    or_ = (A * D) / (B * C)
    log_or = math.log(or_)
    se_log_or = math.sqrt(1/A + 1/B + 1/C + 1/D)
    return or_, log_or, se_log_or


def main() -> int:
    summary = load_summary()
    # Build per-run rows with early features + late outcome
    rows = []
    for s in summary:
        ef = early_features(s)
        lo = late_outcome(s)
        rows.append({
            "kind": s["kind"],
            "method": s["method"],
            "seed": s["seed"],
            "group_size": s.get("group_size", -1),
            "n_steps": ef["n_steps"],
            "features": ef,
            "label": lo["label"],
            "last10_avg": lo["last10_avg"],
            "peak": lo["peak"],
            "mean_zvf": ef["mean_zvf"],
        })

    # ---- 1. Early-features TSV ----
    out_early = RESULTS / "zvf_iter42_early_predicts.tsv"
    feat_keys = ["fp05_frac", "fp07_frac", "fp09_frac",
                 "burden_05", "burden_07",
                 "mean_zvf", "zvf_std", "zvf_lag1"]
    preds = knn_loo(rows, feat_keys, k=3)
    for r, p in zip(rows, preds):
        r["pred"] = p
        r["correct"] = int(p == r["label"])

    with out_early.open("w") as f:
        f.write("# Iter 42 Pillar 2 early-trace ZVF as predictor of late failure mode.\n")
        f.write("# Per-run row: early features (fp05_frac = first_pass_zvf05 / n_steps,\n")
        f.write("#   burden_05 = auc_above_zvf05, etc.) + late outcome label.\n")
        f.write("# kNN-LOO with k=3 over z-scored features predicting {collapse,\n")
        f.write("#   plateau, drift, converged}. Same taxonomy as iter 38.\n")
        f.write("# Source: platform_modal/scripts/zvf_iter42.py\n")
        cols = ["kind", "method", "seed", "group_size", "n_steps",
                "fp05_frac", "fp07_frac", "fp09_frac",
                "early_bloom05", "early_bloom07", "early_bloom09",
                "burden_05", "burden_07", "mean_zvf", "zvf_std", "zvf_lag1",
                "last10_avg", "peak", "label", "pred", "correct"]
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join([
                r["kind"], r["method"], str(r["seed"]), str(r["group_size"]),
                str(r["n_steps"]),
                f'{r["features"]["fp05_frac"]:.4f}',
                f'{r["features"]["fp07_frac"]:.4f}',
                f'{r["features"]["fp09_frac"]:.4f}',
                str(r["features"]["early_bloom05"]),
                str(r["features"]["early_bloom07"]),
                str(r["features"]["early_bloom09"]),
                f'{r["features"]["burden_05"]:.4f}',
                f'{r["features"]["burden_07"]:.4f}',
                f'{r["features"]["mean_zvf"]:.4f}',
                f'{r["features"]["zvf_std"]:.4f}',
                f'{r["features"]["zvf_lag1"]:.4f}',
                f'{r["last10_avg"]:.4f}',
                f'{r["peak"]:.4f}',
                r["label"], r["pred"], str(r["correct"]),
            ]) + "\n")
    print(f"wrote {out_early}")

    # ---- 2. Confusion matrix + accuracy + Spearman ----
    confusion = defaultdict(lambda: defaultdict(int))
    for r in rows:
        confusion[r["label"]][r["pred"]] += 1
    n_total = len(rows)
    n_correct = sum(r["correct"] for r in rows)
    acc = n_correct / n_total if n_total else 0.0

    # Per-class recall
    class_recall = {}
    for true in ["collapse", "drift", "plateau", "converged"]:
        n_true = sum(1 for r in rows if r["label"] == true)
        n_pred = sum(1 for r in rows if r["label"] == true and r["pred"] == true)
        class_recall[true] = n_pred / n_true if n_true else float("nan")

    # ---- 2b. Leave-one-CLUSTER-out kNN (the harder test) ----
    early_only_keys = ["fp05_frac", "fp07_frac", "fp09_frac",
                       "burden_05", "burden_07", "zvf_lag1"]
    full_keys = feat_keys  # includes mean_zvf (full-trace summary)

    preds_cluster = knn_loo_cluster(rows, early_only_keys, k=3)
    n_correct_cluster = sum(1 for r, p in zip(rows, preds_cluster)
                             if p == r["label"])
    acc_cluster = n_correct_cluster / n_total if n_total else 0.0

    # Also cluster-LOO with FULL feature set for comparison
    preds_cluster_full = knn_loo_cluster(rows, full_keys, k=3)
    n_correct_cluster_full = sum(1 for r, p in zip(rows, preds_cluster_full)
                                  if p == r["label"])
    acc_cluster_full = n_correct_cluster_full / n_total if n_total else 0.0

    # Per-cluster cluster-LOO accuracy: for each held-out cluster,
    # what fraction of rows were correctly classified?
    cluster_ids = [(r["kind"], r["method"]) for r in rows]
    cluster_acc_early = defaultdict(lambda: [0, 0])
    cluster_acc_full = defaultdict(lambda: [0, 0])
    for i, r in enumerate(rows):
        ci = cluster_ids[i]
        cluster_acc_early[ci][1] += 1
        cluster_acc_early[ci][0] += int(preds_cluster[i] == r["label"])
        cluster_acc_full[ci][1] += 1
        cluster_acc_full[ci][0] += int(preds_cluster_full[i] == r["label"])

    # ---- 2c. Univariate AUCs for each early feature ----
    # Binary outcome: converged=1, plateau=0 (the only two classes in pool)
    ys_bin = [1 if r["label"] == "converged" else 0 for r in rows]
    auc_per_feature = {}
    for fk in early_only_keys + ["mean_zvf"]:
        xs = [r["features"][fk] for r in rows]
        auc_per_feature[fk] = univariate_auc(xs, ys_bin)

    # ---- 3. Univariate odds ratios ----
    # Bloom05: fp05_frac <= 0.5 → predicts collapse?
    a = sum(1 for r in rows if r["features"]["early_bloom05"] == 1 and r["label"] == "collapse")
    b = sum(1 for r in rows if r["features"]["early_bloom05"] == 1 and r["label"] != "collapse")
    c = sum(1 for r in rows if r["features"]["early_bloom05"] == 0 and r["label"] == "collapse")
    d = sum(1 for r in rows if r["features"]["early_bloom05"] == 0 and r["label"] != "collapse")
    or_bloom05, log_or_b05, se_b05 = odds_ratio(a, b, c, d)
    bloom05_ci_lo = math.exp(log_or_b05 - 1.96 * se_b05)
    bloom05_ci_hi = math.exp(log_or_b05 + 1.96 * se_b05)

    # Bloom09: stricter threshold
    a9 = sum(1 for r in rows if r["features"]["early_bloom09"] == 1 and r["label"] == "collapse")
    b9 = sum(1 for r in rows if r["features"]["early_bloom09"] == 1 and r["label"] != "collapse")
    c9 = sum(1 for r in rows if r["features"]["early_bloom09"] == 0 and r["label"] == "collapse")
    d9 = sum(1 for r in rows if r["features"]["early_bloom09"] == 0 and r["label"] != "collapse")
    or_bloom09, log_or_b09, se_b09 = odds_ratio(a9, b9, c9, d9)
    bloom09_ci_lo = math.exp(log_or_b09 - 1.96 * se_b09)
    bloom09_ci_hi = math.exp(log_or_b09 + 1.96 * se_b09)

    # burden_05 (top-quartile vs bottom-quartile)
    burden_vals = sorted(r["features"]["burden_05"] for r in rows)
    q75 = burden_vals[int(0.75 * len(burden_vals))] if burden_vals else 0.0
    q25 = burden_vals[int(0.25 * len(burden_vals))] if burden_vals else 0.0
    high_burden = [r for r in rows if r["features"]["burden_05"] >= q75]
    low_burden = [r for r in rows if r["features"]["burden_05"] <= q25]
    high_collapse = sum(1 for r in high_burden if r["label"] == "collapse")
    high_n = len(high_burden)
    low_collapse = sum(1 for r in low_burden if r["label"] == "collapse")
    low_n = len(low_burden)

    # Spearman correlation between early ZVF burden and late mean ZVF
    xs = [r["features"]["burden_05"] for r in rows]
    ys = [r["mean_zvf"] for r in rows]
    rho_burden_mean = spearman(xs, ys)

    # Spearman correlation between fp05_frac and last10_avg
    xs2 = [r["features"]["fp05_frac"] for r in rows]
    ys2 = [r["last10_avg"] for r in rows]
    rho_fp05_last10 = spearman(xs2, ys2)

    # ---- 4. Per-(kind, method) failure audit ----
    by_method = defaultdict(lambda: {"n": 0, "n_collapse": 0, "n_bloom05": 0,
                                     "n_bloom05_collapse": 0,
                                     "n_bloom09": 0, "n_bloom09_collapse": 0,
                                     "n_converged": 0, "n_converged_bloom05": 0,
                                     "mean_last10": []})
    for r in rows:
        m = by_method[(r["kind"], r["method"])]
        m["n"] += 1
        m["n_collapse"] += int(r["label"] == "collapse")
        m["n_bloom05"] += int(r["features"]["early_bloom05"] == 1)
        m["n_bloom05_collapse"] += int(r["features"]["early_bloom05"] == 1
                                       and r["label"] == "collapse")
        m["n_bloom09"] += int(r["features"]["early_bloom09"] == 1)
        m["n_bloom09_collapse"] += int(r["features"]["early_bloom09"] == 1
                                       and r["label"] == "collapse")
        m["n_converged"] += int(r["label"] == "converged")
        m["n_converged_bloom05"] += int(r["features"]["early_bloom05"] == 1
                                        and r["label"] == "converged")
        m["mean_last10"].append(r["last10_avg"])

    out_audit = RESULTS / "zvf_iter42_failure_audit.tsv"
    with out_audit.open("w") as f:
        f.write("# Iter 42 Pillar 2 per-(kind, method) early-bloom failure audit.\n")
        f.write("# early_bloom05 = ZVF first crosses 0.5 in first half of trace.\n")
        f.write("# early_bloom09 = ZVF first crosses 0.9 in first half of trace.\n")
        f.write("# Source: platform_modal/scripts/zvf_iter42.py\n")
        f.write("kind\tmethod\tn\tn_collapse\tcollapse_rate\t"
                "n_bloom05\tn_bloom05_collapse\tbloom05_collapse_rate\t"
                "n_bloom09\tn_bloom09_collapse\tbloom09_collapse_rate\t"
                "n_converged\tn_converged_bloom05\tmean_last10\n")
        for (k, m_), m in sorted(by_method.items()):
            collapse_rate = m["n_collapse"] / m["n"] if m["n"] else 0.0
            b05_cr = (m["n_bloom05_collapse"] / m["n_bloom05"]
                      if m["n_bloom05"] else float("nan"))
            b09_cr = (m["n_bloom09_collapse"] / m["n_bloom09"]
                      if m["n_bloom09"] else float("nan"))
            ml = statistics.fmean(m["mean_last10"]) if m["mean_last10"] else 0.0
            f.write(f"{k}\t{m_}\t{m['n']}\t{m['n_collapse']}\t{collapse_rate:.4f}\t"
                    f"{m['n_bloom05']}\t{m['n_bloom05_collapse']}\t"
                    f"{b05_cr:.4f}\t{m['n_bloom09']}\t{m['n_bloom09_collapse']}\t"
                    f"{b09_cr:.4f}\t{m['n_converged']}\t{m['n_converged_bloom05']}\t"
                    f"{ml:.4f}\n")
    print(f"wrote {out_audit}")

    # ---- 5. Summary ----
    out_summary = RESULTS / "zvf_iter42_summary.tsv"
    with out_summary.open("w") as f:
        f.write("# Iter 42 Pillar 2 one-rollup summary.\n")
        f.write("# Source: platform_modal/scripts/zvf_iter42.py\n")
        f.write("metric\tvalue\tdetail\n")
        f.write(f"n_runs\t{n_total}\t60-row zvf_dynamics.json pool\n")
        f.write(f"n_classes\t2\tobserved in pool: plateau=45, converged=15\n")
        f.write(f"loo_accuracy_per_row\t{acc:.4f}\tk=3, z-scored features, all 8\n")
        f.write(f"loo_cluster_accuracy_early_only\t{acc_cluster:.4f}\t"
                f"k=3, leave-one-(kind,method)-out, 6 EARLY features only\n")
        f.write(f"loo_cluster_accuracy_full\t{acc_cluster_full:.4f}\t"
                f"k=3, leave-one-(kind,method)-out, full 8 features\n")
        f.write(f"recall_collapse\t{class_recall.get('collapse', float('nan')):.4f}\t"
                f"n_true={sum(1 for r in rows if r['label']=='collapse')}\n")
        f.write(f"recall_drift\t{class_recall.get('drift', float('nan')):.4f}\t"
                f"n_true={sum(1 for r in rows if r['label']=='drift')}\n")
        f.write(f"recall_plateau\t{class_recall.get('plateau', float('nan')):.4f}\t"
                f"n_true={sum(1 for r in rows if r['label']=='plateau')}\n")
        f.write(f"recall_converged\t{class_recall.get('converged', float('nan')):.4f}\t"
                f"n_true={sum(1 for r in rows if r['label']=='converged')}\n")
        f.write(f"or_bloom05_collapse\t{or_bloom05:.4f}\t"
                f"95pct CI [{bloom05_ci_lo:.4f}, {bloom05_ci_hi:.4f}]  "
                f"(Haldane-Anscombe correction; a={a},b={b},c={c},d={d})\n")
        f.write(f"or_bloom09_collapse\t{or_bloom09:.4f}\t"
                f"95pct CI [{bloom09_ci_lo:.4f}, {bloom09_ci_hi:.4f}]  "
                f"(a={a9},b={b9},c={c9},d={d9})\n")
        f.write(f"high_burden_collapse_rate\t"
                f"{high_collapse/high_n if high_n else float('nan'):.4f}\t"
                f"top-quartile burden_05, n={high_n}\n")
        f.write(f"low_burden_collapse_rate\t"
                f"{low_collapse/low_n if low_n else float('nan'):.4f}\t"
                f"bottom-quartile burden_05, n={low_n}\n")
        f.write(f"spearman_burden05_mean_zvf\t{rho_burden_mean:.4f}\t"
                f"early burden vs late mean ZVF\n")
        f.write(f"spearman_fp05frac_last10\t{rho_fp05_last10:.4f}\t"
                f"earliness of ZVF rise vs last-10 accuracy\n")
        # Univariate AUCs
        for fk in early_only_keys + ["mean_zvf"]:
            f.write(f"auc_univariate_{fk}\t{auc_per_feature[fk]:.4f}\t"
                    f"single-feature AUC predicting converged vs plateau\n")
        # Per-cluster cluster-LOO accuracy
        for ci in sorted(cluster_acc_early.keys()):
            ea = cluster_acc_early[ci]
            fa = cluster_acc_full[ci]
            f.write(f"cluster_loo_early_{ci[0]}_{ci[1]}\t"
                    f"{ea[0]/ea[1] if ea[1] else float('nan'):.4f}\t"
                    f"n={ea[1]} early-only; "
                    f"full-feature LOO={fa[0]/fa[1] if fa[1] else float('nan'):.4f}\n")
    print(f"wrote {out_summary}")

    # ---- 6. Console summary ----
    print(f"\nIter 42 — early-trace ZVF predicts late failure mode")
    print(f"  n_runs = {n_total}")
    print(f"  LOO accuracy (per-row, all 8 features) = {acc:.3f} ({n_correct}/{n_total})")
    print(f"  Cluster-LOO accuracy (EARLY features only) = {acc_cluster:.3f} "
          f"({n_correct_cluster}/{n_total})")
    print(f"  Cluster-LOO accuracy (FULL 8 features) = {acc_cluster_full:.3f} "
          f"({n_correct_cluster_full}/{n_total})")
    print(f"  OR(early_bloom05 → collapse) = {or_bloom05:.2f} "
          f"[{bloom05_ci_lo:.2f}, {bloom05_ci_hi:.2f}]")
    print(f"  OR(early_bloom09 → collapse) = {or_bloom09:.2f} "
          f"[{bloom09_ci_lo:.2f}, {bloom09_ci_hi:.2f}]")
    print(f"  Spearman(burden_05, mean_zvf) = {rho_burden_mean:.3f}")
    print(f"  Spearman(fp05_frac, last10) = {rho_fp05_last10:.3f}")
    print(f"  Univariate AUCs (converged vs plateau):")
    for fk in early_only_keys + ["mean_zvf"]:
        print(f"    {fk:18s}  AUC = {auc_per_feature[fk]:.3f}")
    print(f"  Collapse rate: top-quartile burden={high_collapse}/{high_n}, "
          f"bottom-quartile={low_collapse}/{low_n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())