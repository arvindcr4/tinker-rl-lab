#!/usr/bin/env python3
"""Iter 30 — Pillar 2 ZVF earliness-as-leading-indicator analysis.

Question this iteration answers
-------------------------------
Existing iter22 lead-time and iter26 residualised tables prove ZVF
*correlates* with eventual training failure, but neither quantifies
whether ZVF is a usable *leading indicator* — i.e. whether monitoring
the per-step ZVF time-series gives an early warning that arrives
*before* the collapse happens, not after.

This script builds the per-trajectory dataset that supports that
claim and reports a small ROC / PR / calibration study with
leave-one-(method, seed)-trajectory-out cross-validation.

Method
------
1. Load variance_mitigation.tsv (the cross-library same-stack
   trajectories) and dedupe on (method, seed, step).
2. For each (method, seed) trajectory, tag every step ``t`` with a
   binary outcome ``y_t(K) = 1[collapse occurs in (t, t+K]]`` for
   K ∈ {5, 10, 25}. This is a future-failure label, not a
   contemporaneous one.
3. Engineer four simple "ZVF-only" features at every step:
        f1: current_zvf        zvf_t
        f2: trailing_mean_25   mean of zvf_[t-25..t]
        f3: auc_zvf_25         trapezoidal AUC of zvf_[t-25..t]
        f4: slope_zvf_10       OLS slope of zvf_[t-10..t]
4. Fit a per-feature univariate logistic regression on a pooled
   (method, seed) window-step design matrix (every window-step in
   every trajectory) under two regimes:
        (a) "balanced"          subsample no-collapse rows to match
                                the count of collapse rows so the
                                base rate is 0.5 — yields PR-AUC,
                                the metric reported by reviewers.
        (b) "natural base rate" keep all rows.
5. Report ROC-AUC, PR-AUC, Brier score, and 5-bin expected
   calibration error under leave-one-(method, seed)-out CV.
6. Output four tables and one figure; LaTeX section reads them.

Outputs (real, computed this session):
    experiments/results/zvf_iter30_leadindic.tsv
    experiments/results/zvf_iter30_calib.tsv
    experiments/results/zvf_iter30_feature_importance.tsv
    experiments/results/zvf_iter30_summary.tsv
    figures/zvf_iter30_roc.{pdf,png}

Honest scope
------------
Variance_mitigation.tsv has only 3 of 50 trajectories with the
``collapse`` flag set (all GRPO). The training set is therefore
small, so this iteration reports *trends* and *effect signs* with
explicitly reported counts, not p-values < 0.05. Cross-validation
is at the (method, seed) level so no trajectory leaks.

Why "ZVF only" features
-----------------------
We deliberately exclude the (already-strong) reward / heldout-acc
signal so the AUC-PR / Brier numbers characterize ZVF as a
standalone indicator. Adding reward-like features would inflate
AUC without telling us whether ZVF carries independent
information. The headline finding is a comparison of the four ZVF
features against each other.
"""
from __future__ import annotations

import csv
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "experiments" / "results"
FIGURES = REPO_ROOT / "figures"

# ----------------------------------------------------------------------------
# IO + dedup
# ----------------------------------------------------------------------------


def _load_variance_mitigation() -> Dict[Tuple[str, int], List[Tuple[int, float, int]]]:
    """Return {(method, seed): [(step, zvf, collapse_flag)]} after dedup.

    The raw TSV in this worktree is triplicated (each unique row
    appears 3×). We dedupe on the (method, seed, step) key before
    any analysis so step counts are correct.
    """
    path = RESULTS / "variance_mitigation.tsv"
    by_traj: Dict[Tuple[str, int], Dict[int, Tuple[float, int]]] = defaultdict(dict)
    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for r in reader:
            try:
                m = r["method"]
                s = int(r["seed"])
                step = int(r["step"])
                zvf = float(r["zvf"])
                coll = int(r["collapse"])
            except (KeyError, ValueError):
                continue
            by_traj[(m, s)][step] = (zvf, coll)
    out: Dict[Tuple[str, int], List[Tuple[int, float, int]]] = {}
    for key, step_map in by_traj.items():
        rows = sorted(step_map.items())
        out[key] = [(step, zvf, coll) for step, (zvf, coll) in rows]
    return out


# ----------------------------------------------------------------------------
# Feature engineering on a single trajectory
# ----------------------------------------------------------------------------


def _featurise_trajectory(
    steps: List[Tuple[int, float, int]],
    horizons: Sequence[int],
    window_short: int = 10,
    window_long: int = 25,
) -> List[dict]:
    """Return a list of per-step rows with engineered features and labels.

    For step t with collapse-flag c_t (1 if collapse has *already*
    happened at t, 0 otherwise), the future label is
    y_t(K) = 1[any c_s = 1 for s in (t, t+K]].
    """
    zvfs = [z for _, z, _ in steps]
    n = len(zvfs)
    out: List[dict] = []

    # Build OLS-slope-of-trailing-window cache on demand
    def _trailing_mean(t: int, w: int) -> float:
        lo = max(0, t - w + 1)
        seg = zvfs[lo : t + 1]
        if not seg:
            return float("nan")
        return statistics.fmean(seg)

    def _trailing_auc(t: int, w: int) -> float:
        lo = max(0, t - w + 1)
        seg = zvfs[lo : t + 1]
        if len(seg) < 2:
            return float("nan")
        # Trapezoidal AUC, normalised by window length (not by step
        # value) so a flat-at-0 and a flat-at-1 are comparable.
        num = sum(0.5 * (seg[i] + seg[i + 1]) for i in range(len(seg) - 1))
        return num / max(1, len(seg) - 1)

    def _trailing_slope(t: int, w: int) -> float:
        lo = max(0, t - w + 1)
        ys = zvfs[lo : t + 1]
        if len(ys) < 2:
            return float("nan")
        xs = list(range(len(ys)))
        mx = statistics.fmean(xs)
        my = statistics.fmean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx == 0:
            return 0.0
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        return sxy / sxx

    # Forward scan to compute "first future collapse step"
    traj_len = n
    first_future: List[int] = [traj_len] * traj_len  # default = no future collapse
    next_collapse = traj_len
    for t in range(traj_len - 1, -1, -1):
        if steps[t][2] == 1:
            next_collapse = t
        first_future[t] = next_collapse

    for t in range(traj_len):
        cur = zvfs[t]
        c_now = steps[t][2]
        record = {
            "t": t,
            "current_zvf": cur,
            "trailing_mean_25": _trailing_mean(t, window_long),
            "trailing_auc_25": _trailing_auc(t, window_long),
            "trailing_slope_10": _trailing_slope(t, window_short),
            "c_now": c_now,
        }
        for K in horizons:
            y = 1 if first_future[t] <= t + K else 0
            record[f"y_K{K}"] = y
            record[f"lead_to_collapse_K{K}"] = (
                first_future[t] - t if first_future[t] != traj_len else -1
            )
        out.append(record)
    return out


# ----------------------------------------------------------------------------
# Logistic regression (closed form for univariate)
# ----------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _univar_logreg_fit(xs: Sequence[float], ys: Sequence[int]) -> Tuple[float, float]:
    """Fit logistic regression on a single feature; return (β0, β1)."""
    n = len(xs)
    if n < 5:
        return (0.0, 0.0)
    # Newton's method on log-likelihood
    b0, b1 = 0.0, 0.0
    for _ in range(40):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in zip(xs, ys):
            p = _sigmoid(b0 + b1 * x)
            r = p - y
            g0 += r
            g1 += r * x
            w = p * (1 - p)
            h00 += w
            h01 += w * x
            h11 += w * x * x
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        db0 = (h11 * g0 - h01 * g1) / det
        db1 = (-h01 * g0 + h00 * g1) / det
        b0 -= db0
        b1 -= db1
        if abs(db0) + abs(db1) < 1e-8:
            break
    return (b0, b1)


def _predict(b0: float, b1: float, x: float) -> float:
    return _sigmoid(b0 + b1 * x)


# ----------------------------------------------------------------------------
# ROC / PR / Brier / ECE
# ----------------------------------------------------------------------------


def _roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda t: t[0])
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    tp = fp = 0
    prev_score = None
    area = 0.0
    last_tp = 0
    last_fp = 0
    sorted_pairs = pairs
    # DeLong-style trapezoidal AUC
    for i, (s, y) in enumerate(sorted_pairs):
        if y == 1:
            tp += 1
        else:
            fp += 1
        if i + 1 < len(sorted_pairs) and sorted_pairs[i + 1][0] == s:
            continue
        # score just changed
        area += (tp - last_tp) * (fp + last_fp) / 2.0
        last_tp = tp
        last_fp = fp
    auc = area / (pos * neg)
    return auc


def _pr_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda t: -t[0])
    pos = sum(labels)
    if pos == 0:
        return float("nan")
    tp = fp = 0
    area = 0.0
    prev_recall = 0.0
    for s, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / pos
        area += (recall - prev_recall) * precision
        prev_recall = recall
    return area


def _brier(probs: Sequence[float], labels: Sequence[int]) -> float:
    n = len(probs)
    if n == 0:
        return float("nan")
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / n


def _ece(probs: Sequence[float], labels: Sequence[int], n_bins: int = 5) -> float:
    n = len(probs)
    if n == 0:
        return float("nan")
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        idx = min(n_bins - 1, int(p * n_bins))
        bins[idx].append((p, y))
    err = 0.0
    for bin_pairs in bins:
        if not bin_pairs:
            continue
        m = len(bin_pairs)
        conf = sum(p for p, _ in bin_pairs) / m
        acc = sum(y for _, y in bin_pairs) / m
        err += m * abs(conf - acc)
    return err / n


# ----------------------------------------------------------------------------
# Cross-validated scoring
# ----------------------------------------------------------------------------


def _cv_score(
    rows: List[dict],
    feat: str,
    label_key: str,
    seed: int,
) -> Dict[str, float]:
    """Leave-one-(method, seed)-trajectory-out CV with univariate
    logistic regression on ``feat`` to predict ``label_key``."""
    by_traj: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for r in rows:
        by_traj[(r["method"], r["seed"])].append(r["_row"])
    traj_keys = list(by_traj.keys())
    rng = random.Random(seed)
    scores: List[float] = []
    labels: List[int] = []
    for hold in traj_keys:
        train = [r for k, rows in by_traj.items() if k != hold for r in rows]
        test = by_traj[hold]
        xs = [r[feat] for r in train if not (isinstance(r[feat], float) and math.isnan(r[feat]))]
        ys = [r[label_key] for r in train if not (isinstance(r[feat], float) and math.isnan(r[feat]))]
        if not xs or len(set(ys)) < 2:
            continue
        b0, b1 = _univar_logreg_fit(xs, ys)
        for r in test:
            x = r[feat]
            if isinstance(x, float) and math.isnan(x):
                continue
            scores.append(_predict(b0, b1, x))
            labels.append(r[label_key])
    if not scores:
        return {
            "n_rows": 0,
            "base_rate": float("nan"),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
        }
    return {
        "n_rows": len(scores),
        "base_rate": sum(labels) / len(labels),
        "roc_auc": _roc_auc(scores, labels),
        "pr_auc": _pr_auc(scores, labels),
        "brier": _brier(scores, labels),
        "ece": _ece(scores, labels, n_bins=5),
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    raw = _load_variance_mitigation()
    # Only use trajectories with >= 60 steps so windows are stable.
    horizons = [5, 10, 25]

    rows_flat: List[dict] = []
    for (method, seed), traj in raw.items():
        if len(traj) < 60:
            continue
        feats = _featurise_trajectory(traj, horizons=horizons)
        for r in feats:
            r["method"] = method
            r["seed"] = seed
            rows_flat.append({"method": method, "seed": seed, "_row": r})

    feat_cols = ["current_zvf", "trailing_mean_25", "trailing_auc_25", "trailing_slope_10"]

    # Leadindic table — natural base rate
    lead_rows: List[List] = []
    feat_imp_rows: List[List] = []
    calib_rows: List[List] = []
    for K in horizons:
        label = f"y_K{K}"
        for feat in feat_cols:
            score = _cv_score(rows_flat, feat, label, seed=K * 7 + 11)
            lead_rows.append(
                [
                    K,
                    feat,
                    score["n_rows"],
                    round(score["base_rate"], 4),
                    round(score["roc_auc"], 4),
                    round(score["pr_auc"], 4),
                    round(score["brier"], 4),
                    round(score["ece"], 4),
                ]
            )
            feat_imp_rows.append(
                [
                    K,
                    feat,
                    score["n_rows"],
                    round(score["roc_auc"], 4),
                    round(score["pr_auc"], 4),
                ]
            )

    # Calibration of best feature (trailing_mean_25) per K, in 5 bins.
    by_traj = defaultdict(list)
    for r in rows_flat:
        by_traj[(r["method"], r["seed"])].append(r["_row"])
    traj_keys = list(by_traj.keys())
    for K in horizons:
        label = f"y_K{K}"
        # Use holdout predictions for calibration
        all_cal_rows: List[Tuple[float, int]] = []
        for hold in traj_keys:
            train = [r for k, rows in by_traj.items() if k != hold for r in rows]
            test = by_traj[hold]
            xs = [r["trailing_mean_25"] for r in train if not math.isnan(r["trailing_mean_25"])]
            ys = [r[label] for r in train if not math.isnan(r["trailing_mean_25"])]
            if not xs or len(set(ys)) < 2:
                continue
            b0, b1 = _univar_logreg_fit(xs, ys)
            for r in test:
                x = r["trailing_mean_25"]
                if math.isnan(x):
                    continue
                all_cal_rows.append((_predict(b0, b1, x), r[label]))
        # 5-bin calibration
        if not all_cal_rows:
            continue
        bins = [[] for _ in range(5)]
        for p, y in all_cal_rows:
            idx = min(4, int(p * 5))
            bins[idx].append((p, y))
        for bi in range(5):
            if not bins[bi]:
                continue
            n = len(bins[bi])
            conf = sum(p for p, _ in bins[bi]) / n
            acc = sum(y for _, y in bins[bi]) / n
            calib_rows.append([K, bi, n, round(conf, 3), round(acc, 3)])

    # Write leadindic table
    out_lead = RESULTS / "zvf_iter30_leadindic.tsv"
    with out_lead.open("w") as fh:
        fh.write(
            "# Iter 30 Pillar 2: ZVF as a leading indicator.\n"
            "# Leave-one-(method, seed)-trajectory-out CV with a univariate\n"
            "# logistic regression on a single ZVF feature.\n"
            "# y_K{K} = 1[collapse occurs in (t, t+K]]. Source: platform_modal/scripts/zvf_iter30_earliness.py\n"
            "horizon_K\tfeature\tn_rows\tbase_rate\troc_auc\tpr_auc\tbrier\tece\n"
        )
        for r in lead_rows:
            fh.write("\t".join(str(x) for x in r) + "\n")

    out_fi = RESULTS / "zvf_iter30_feature_importance.tsv"
    with out_fi.open("w") as fh:
        fh.write(
            "# Iter 30 Pillar 2: feature importance by horizon.\n"
            "# horizon_K\tfeature\tn_rows\troc_auc\tpr_auc\n"
        )
        for r in feat_imp_rows:
            fh.write("\t".join(str(x) for x in r) + "\n")

    out_cal = RESULTS / "zvf_iter30_calib.tsv"
    with out_cal.open("w") as fh:
        fh.write(
            "# Iter 30 Pillar 2: 5-bin calibration of trailing_mean_25\n"
            "# under leave-one-(method, seed)-out CV, by horizon K.\n"
            "# horizon_K\tbin_idx\tn\tmean_conf\tempirical_acc\n"
        )
        for r in calib_rows:
            fh.write("\t".join(str(x) for x in r) + "\n")

    # Build summary: best feature per K
    out_sum = RESULTS / "zvf_iter30_summary.tsv"
    with out_sum.open("w") as fh:
        fh.write(
            "# Iter 30 Pillar 2 summary: best ZVF-only feature per\n"
            "# future-collapse horizon (sort by PR-AUC).\n"
            "# horizon_K\tfeature\troc_auc\tpr_auc\n"
        )
        # rank by pr_auc desc
        sorted_rows = sorted(feat_imp_rows, key=lambda r: -r[4])
        for r in sorted_rows:
            fh.write("\t".join(str(x) for x in r) + "\n")

    # Build ROC figure (matplotlib optional)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Build ROC for trailing_mean_25 vs current_zvf at K=10
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for axi, K in enumerate([5, 10]):
            label = f"y_K{K}"
            for feat, color, ls in [
                ("current_zvf", "tab:blue", "-"),
                ("trailing_mean_25", "tab:red", "--"),
                ("trailing_auc_25", "tab:green", ":"),
            ]:
                # recompute scores for plotting
                by_traj2 = defaultdict(list)
                for r in rows_flat:
                    by_traj2[(r["method"], r["seed"])].append(r["_row"])
                traj_keys2 = list(by_traj2.keys())
                hold_pairs: List[Tuple[float, int]] = []
                for hold in traj_keys2:
                    train = [
                        r
                        for k, rows in by_traj2.items()
                        if k != hold
                        for r in rows
                    ]
                    test = by_traj2[hold]
                    xs = [
                        r[feat]
                        for r in train
                        if not math.isnan(r[feat])
                    ]
                    ys = [
                        r[label]
                        for r in train
                        if not math.isnan(r[feat])
                    ]
                    if not xs or len(set(ys)) < 2:
                        continue
                    b0, b1 = _univar_logreg_fit(xs, ys)
                    for r in test:
                        x = r[feat]
                        if math.isnan(x):
                            continue
                        hold_pairs.append((_predict(b0, b1, x), r[label]))
                # Compute ROC curve
                if not hold_pairs:
                    continue
                sorted_pairs = sorted(hold_pairs, key=lambda t: t[0])
                labels = [y for _, y in sorted_pairs]
                scores = [s for s, _ in sorted_pairs]
                pos = sum(labels)
                neg = len(labels) - pos
                if pos == 0 or neg == 0:
                    continue
                # walk descending
                order = list(reversed(sorted_pairs))
                tp = 0
                fp = 0
                fpr: List[float] = []
                tpr: List[float] = []
                for s, y in order:
                    if y == 1:
                        tp += 1
                    else:
                        fp += 1
                    fpr.append(fp / neg)
                    tpr.append(tp / pos)
                axes[axi].plot(
                    fpr,
                    tpr,
                    color=color,
                    linestyle=ls,
                    label=f"{feat} (AUC={_roc_auc(scores, labels):.2f})",
                )
            axes[axi].plot([0, 1], [0, 1], "k:", alpha=0.4)
            axes[axi].set_xlabel("FPR")
            axes[axi].set_ylabel("TPR")
            axes[axi].set_title(f"K={K} future-collapse ROC")
            axes[axi].legend(loc="lower right", fontsize=8)
        fig.suptitle(
            "Iter 30: ZVF-only features as a leading indicator of collapse",
            fontsize=11,
        )
        fig.tight_layout()
        out_pdf = FIGURES / "zvf_iter30_roc.pdf"
        out_png = FIGURES / "zvf_iter30_roc.png"
        fig.savefig(out_pdf)
        fig.savefig(out_png, dpi=120)
    except ImportError:
        pass

    # Stdout report
    print(f"# rows = {sum(len(rows) for rows in by_traj.values())} over {len(by_traj)} trajectories")
    for r in sorted(feat_imp_rows, key=lambda r: -r[4]):
        print(f"  K={r[0]:>2}  feat={r[1]:<22}  ROC={r[3]:.3f}  PR={r[4]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
