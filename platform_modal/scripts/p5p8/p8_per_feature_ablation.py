#!/usr/bin/env python3
"""P8 per-feature ablation of the 4 LLM-as-sensor aggregates (iter 24).

The iter-4 paper has a leave-one-OUT ablation of the four aggregates
(V_mean, V_std, V_max, V_min), and the iter-12 PR-AUC story and the
iter-20 threshold-stratified operating-point story both confirm the
aggregate-block adds a measurable signal at the moderate-precision regime.
This iter closes three reviewer-visible gaps:

  G1. **Bootstrap CIs on the per-feature ablation deltas.**
      The existing leave-one-OUT table reports single-point AUC/Brier/F1
      numbers. A reviewer who asks "is the V_max drop statistically
      distinguishable from noise?" cannot answer from that table. We
      add paired bootstrap CIs on every (variant vs ALL_24) contrast.

  G2. **Reverse leaves-one-IN ablation.**
      The existing ablation only asks "does removing V_mean hurt?". The
      reviewer's mirror question is "does adding V_mean alone, to the
      20 raw V-features, help?" -- which is exactly the per-feature
      attribution the brief calls for. We sweep 1-of-4, 2-of-6 pairs,
      and 3-of-4 subsets.

  G3. **Score-decile-stratified reliability diagram with bootstrap CIs.**
      The iter-4 calibration narrative reports global ECE-10, Brier, AUC.
      A reliability diagram (predicted decile mean vs observed positive
      rate) shows WHERE calibration drifts. We add 95% bootstrap CIs
      on every decile for XGB-20raw and XGB-24full and report the
      max absolute calibration drift per model.

Inputs
------
fraud_data.csv : 50k synthetic fraud rows (24 numeric features + Class).
test_data.csv  : 10k held-out rows (same schema + Class).

Outputs
-------
platform_hybrid/experiments/results/p5p8/p8_perfeat_loo.tsv        (LOO with bootstrap CI)
platform_hybrid/experiments/results/p5p8/p8_perfeat_loi.tsv        (LOI 1-of-4 with CI)
platform_hybrid/experiments/results/p5p8/p8_perfeat_loi_pairs.tsv  (LOI 2-of-6 pairs with CI)
platform_hybrid/experiments/results/p5p8/p8_perfeat_reliability.tsv (10 deciles x 3 models)
platform_hybrid/experiments/results/p5p8/p8_perfeat_summary.json
platform_hybrid/experiments/results/p5p8/figures/p8_reliability.{png,pdf}
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
FIG = RES / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TRAIN = ROOT / "fraud_data.csv"
TEST = ROOT / "test_data.csv"

V20 = [f"V{i}" for i in range(1, 21)]
AGG4 = ["V_mean", "V_std", "V_max", "V_min"]

SEED = 42
BOOT_SEED = 2026
N_BOOT = 1000
DECILES = 10

# Tree config (matches iter-4 release scripts)
TREE_KWARGS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=7.0,
    eval_metric="logloss",
    random_state=SEED,
    tree_method="hist",
    n_jobs=4,
)


def load(path: Path, feats: list[str]) -> tuple[list[list[float]], list[int]]:
    X, y = [], []
    with path.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)
        idx = [header.index(c) for c in feats]
        yi = header.index("Class")
        for row in rdr:
            X.append([float(row[i]) for i in idx])
            y.append(int(row[yi]))
    return X, y


def fit_predict(X_tr, y_tr, X_te) -> list[float]:
    import xgboost as xgb

    m = xgb.XGBClassifier(**TREE_KWARGS)
    m.fit(X_tr, y_tr, verbose=False)
    return m.predict_proba(X_te)[:, 1].tolist()


def metrics(y: list[int], p: list[float]) -> dict:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        brier_score_loss,
        log_loss,
    )

    auc = float(roc_auc_score(y, p))
    ap = float(average_precision_score(y, p))
    brier = float(brier_score_loss(y, p))
    n = len(y)
    n_pos = sum(y)
    pred_pos = sum(1 for pi in p if pi >= 0.5)
    tp = sum(1 for yi, pi in zip(y, p) if pi >= 0.5 and yi == 1)
    fp = sum(1 for yi, pi in zip(y, p) if pi >= 0.5 and yi == 0)
    fn = sum(1 for yi, pi in zip(y, p) if pi < 0.5 and yi == 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    # ECE-10
    bins = [[] for _ in range(DECILES)]
    for yi, pi in zip(y, p):
        b = min(int(pi * DECILES), DECILES - 1)
        bins[b].append((yi, pi))
    ece = 0.0
    for b in bins:
        if not b:
            continue
        conf = sum(pi for _, pi in b) / len(b)
        acc = sum(yi for yi, _ in b) / len(b)
        ece += (len(b) / n) * abs(acc - conf)
    return dict(
        auc=auc,
        ap=ap,
        brier=brier,
        ece10=float(ece),
        f1=float(f1),
        precision=float(precision),
        recall=float(recall),
        n=n,
        n_pos=n_pos,
        n_pred_pos=pred_pos,
    )


def paired_bootstrap_ci(
    y: list[int],
    p_a: list[float],
    p_b: list[float],
    metric: str = "auc",
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> tuple[float, float, float]:
    """Paired bootstrap on (a - b) for a single metric. Returns (delta, lo, hi).

    metric in {auc, brier, f1, ece10}.
    """
    from sklearn.metrics import roc_auc_score

    rng = random.Random(seed)
    n = len(y)

    def m(yv, pa, pb):
        if metric == "auc":
            if sum(yv) == 0 or sum(yv) == len(yv):
                return None
            return roc_auc_score(yv, pa) - roc_auc_score(yv, pb)
        if metric == "brier":
            return (
                sum((pi - yi) ** 2 for yi, pi in zip(yv, pa)) / len(yv)
                - sum((pi - yi) ** 2 for yi, pi in zip(yv, pb)) / len(yv)
            )
        if metric == "f1":
            ta = sum(1 for yi, pi in zip(yv, pa) if pi >= 0.5 and yi == 1)
            fa = sum(1 for yi, pi in zip(yv, pa) if pi >= 0.5 and yi == 0)
            tb = sum(1 for yi, pi in zip(yv, pb) if pi >= 0.5 and yi == 1)
            fb = sum(1 for yi, pi in zip(yv, pb) if pi >= 0.5 and yi == 0)
            pa_p = ta / max(ta + fa, 1)
            ra = ta / max(sum(yv), 1)
            pb_p = tb / max(tb + fb, 1)
            rb = tb / max(sum(yv), 1)
            fa_f1 = 2 * pa_p * ra / max(pa_p + ra, 1e-9)
            fb_f1 = 2 * pb_p * rb / max(pb_p + rb, 1e-9)
            return fa_f1 - fb_f1
        if metric == "ece10":
            def ece(pv):
                bins = [[] for _ in range(DECILES)]
                for yi, pi in zip(yv, pv):
                    b = min(int(pi * DECILES), DECILES - 1)
                    bins[b].append((yi, pi))
                e = 0.0
                for bb in bins:
                    if not bb:
                        continue
                    conf = sum(pi for _, pi in bb) / len(bb)
                    acc = sum(yi for yi, _ in bb) / len(bb)
                    e += (len(bb) / len(yv)) * abs(acc - conf)
                return e
            return ece(pa) - ece(pb)
        raise ValueError(metric)

    a_full = metrics(y, p_a)
    b_full = metrics(y, p_b)
    delta_full = a_full[metric] - b_full[metric]
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        yv = [y[i] for i in idx]
        pa = [p_a[i] for i in idx]
        pb = [p_b[i] for i in idx]
        d = m(yv, pa, pb)
        if d is None:
            continue
        deltas.append(d)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas))]
    return delta_full, lo, hi


def main() -> None:
    print("[perfeat] loading data ...")
    Xtr_24, y_tr = load(TRAIN, V20 + AGG4)
    Xtr_20, _ = load(TRAIN, V20)
    Xte_24, y_te = load(TEST, V20 + AGG4)
    Xte_20, _ = load(TEST, V20)

    print("[perfeat] fitting trees ...")
    p_24 = fit_predict(Xtr_24, y_tr, Xte_24)
    p_20 = fit_predict(Xtr_20, y_tr, Xte_20)
    m_24 = metrics(y_te, p_24)
    m_20 = metrics(y_te, p_20)
    print(f"  XGB-24full : AUC={m_24['auc']:.4f}  Brier={m_24['brier']:.4f}  F1={m_24['f1']:.3f}")
    print(f"  XGB-20raw  : AUC={m_20['auc']:.4f}  Brier={m_20['brier']:.4f}  F1={m_20['f1']:.3f}")

    # ---- G1: leaves-one-OUT with bootstrap CIs vs ALL_24 ----
    print("[perfeat] LOO + CIs ...")
    loo_rows = []
    loo_full = metrics(y_te, p_24)
    for drop in [None] + AGG4:
        feats = [c for c in V20 + AGG4 if c != drop]
        if drop is None:
            p_var = p_24
            label = "ALL_24"
        else:
            Xtr, _ = load(TRAIN, feats)
            Xte, _ = load(TEST, feats)
            p_var = fit_predict(Xtr, y_tr, Xte)
            label = f"drop_{drop}"
        m_var = metrics(y_te, p_var)
        for metric in ("auc", "brier", "f1", "ece10"):
            d, lo, hi = paired_bootstrap_ci(y_te, p_var, p_24, metric=metric)
            loo_rows.append(
                dict(
                    variant=label,
                    metric=metric,
                    value=m_var[metric],
                    delta_vs_all24=d,
                    ci_lo=lo,
                    ci_hi=hi,
                    excludes_zero=(lo > 0) or (hi < 0),
                    direction=("neg" if d < 0 else "pos"),
                )
            )

    with (RES / "p8_perfeat_loo.tsv").open("w") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(loo_rows[0].keys()),
            delimiter="\t",
        )
        w.writeheader()
        for r in loo_rows:
            w.writerow(r)

    # ---- G2: leaves-one-IN (1-of-4 reverse) ----
    print("[perfeat] LOI 1-of-4 ...")
    loi_rows = []
    for keep in AGG4:
        feats = V20 + [keep]
        Xtr, _ = load(TRAIN, feats)
        Xte, _ = load(TEST, feats)
        p_var = fit_predict(Xtr, y_tr, Xte)
        m_var = metrics(y_te, p_var)
        for metric in ("auc", "brier", "f1", "ece10"):
            d, lo, hi = paired_bootstrap_ci(y_te, p_var, p_20, metric=metric)
            loi_rows.append(
                dict(
                    variant=f"add_{keep}_only",
                    metric=metric,
                    value=m_var[metric],
                    delta_vs_20raw=d,
                    ci_lo=lo,
                    ci_hi=hi,
                    excludes_zero=(lo > 0) or (hi < 0),
                    direction=("neg" if d < 0 else "pos"),
                )
            )

    with (RES / "p8_perfeat_loi.tsv").open("w") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(loi_rows[0].keys()),
            delimiter="\t",
        )
        w.writeheader()
        for r in loi_rows:
            w.writerow(r)

    # ---- G2b: 2-of-6 pairs ----
    print("[perfeat] LOI 2-of-6 ...")
    pair_rows = []
    for combo in combinations(AGG4, 2):
        feats = V20 + list(combo)
        Xtr, _ = load(TRAIN, feats)
        Xte, _ = load(TEST, feats)
        p_var = fit_predict(Xtr, y_tr, Xte)
        m_var = metrics(y_te, p_var)
        for metric in ("auc", "f1"):
            d, lo, hi = paired_bootstrap_ci(y_te, p_var, p_20, metric=metric)
            pair_rows.append(
                dict(
                    variant=f"add_{'+'.join(combo)}",
                    metric=metric,
                    value=m_var[metric],
                    delta_vs_20raw=d,
                    ci_lo=lo,
                    ci_hi=hi,
                    excludes_zero=(lo > 0) or (hi < 0),
                    direction=("neg" if d < 0 else "pos"),
                )
            )

    with (RES / "p8_perfeat_loi_pairs.tsv").open("w") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(pair_rows[0].keys()),
            delimiter="\t",
        )
        w.writeheader()
        for r in pair_rows:
            w.writerow(r)

    # ---- G3: reliability diagram with bootstrap CIs ----
    print("[perfeat] reliability diagram ...")
    rel_rows = []
    rng = random.Random(BOOT_SEED)

    def reliability(y, p, decile):
        bins = [[] for _ in range(decile)]
        for yi, pi in zip(y, p):
            b = min(int(pi * decile), decile - 1)
            bins[b].append((yi, pi))
        out = []
        for b, members in enumerate(bins):
            if not members:
                out.append((None, None, 0))
                continue
            conf = sum(pi for _, pi in members) / len(members)
            acc = sum(yi for yi, _ in members) / len(members)
            n = len(members)
            # bootstrap CI on acc within bin (bin membership fixed)
            accs = []
            for _ in range(N_BOOT):
                boot = [members[rng.randrange(len(members))] for _ in range(len(members))]
                accs.append(sum(yi for yi, _ in boot) / len(boot))
            accs.sort()
            lo = accs[int(0.025 * len(accs))]
            hi = accs[int(0.975 * len(accs))]
            out.append((conf, acc, n, lo, hi))
        return out

    for label, p_pred in (("XGB-24full", p_24), ("XGB-20raw", p_20)):
        rel = reliability(y_te, p_pred, DECILES)
        for b, (conf, acc, n, lo, hi) in enumerate(rel):
            rel_rows.append(
                dict(
                    model=label,
                    decile=b,
                    conf=conf,
                    acc=acc,
                    n=n,
                    acc_lo=lo,
                    acc_hi=hi,
                )
            )

    with (RES / "p8_perfeat_reliability.tsv").open("w") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rel_rows[0].keys()),
            delimiter="\t",
        )
        w.writeheader()
        for r in rel_rows:
            w.writerow(r)

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for label, marker, color in (
        ("XGB-24full", "o", "tab:red"),
        ("XGB-20raw", "s", "tab:blue"),
    ):
        rows = [r for r in rel_rows if r["model"] == label and r["conf"] is not None]
        xs = [r["conf"] for r in rows]
        ys = [r["acc"] for r in rows]
        lo = [r["acc"] - r["acc_lo"] for r in rows]
        hi = [r["acc_hi"] - r["acc"] for r in rows]
        ax.errorbar(
            xs,
            ys,
            yerr=[lo, hi],
            marker=marker,
            color=color,
            label=label,
            capsize=2,
            linewidth=1.2,
            markersize=5,
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="perfect")
    ax.set_xlabel("predicted probability (decile mean)")
    ax.set_ylabel("observed positive rate (decile mean)")
    ax.set_title("Reliability diagram — held-out test split (n=10000, 144 positives)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "p8_reliability.png", dpi=140)
    fig.savefig(FIG / "p8_reliability.pdf")
    plt.close(fig)

    # ---- summary JSON ----
    def n_exclude(rows, metric, sign):
        return sum(
            1
            for r in rows
            if r["metric"] == metric
            and r["excludes_zero"]
            and ((sign == "pos" and r["direction"] == "pos") or (sign == "neg" and r["direction"] == "neg"))
        )

    summary = {
        "n_test": len(y_te),
        "n_pos_test": sum(y_te),
        "models": {"XGB-24full": m_24, "XGB-20raw": m_20},
        "loo": {
            "rows": len(loo_rows),
            "auc_exclude_zero_drops": n_exclude(loo_rows, "auc", "neg"),
            "f1_exclude_zero_drops": n_exclude(loo_rows, "f1", "neg"),
        },
        "loi_1of4": {
            "rows": len(loi_rows),
            "auc_exclude_zero_adds": n_exclude(loi_rows, "auc", "pos"),
            "f1_exclude_zero_adds": n_exclude(loi_rows, "f1", "pos"),
        },
        "loi_pairs": {
            "rows": len(pair_rows),
            "auc_exclude_zero_adds": n_exclude(pair_rows, "auc", "pos"),
            "f1_exclude_zero_adds": n_exclude(pair_rows, "f1", "pos"),
        },
        "reliability": {
            "deciles": DECILES,
            "max_drift_24full": max(
                abs(r["acc"] - r["conf"])
                for r in rel_rows
                if r["model"] == "XGB-24full" and r["conf"] is not None
            ),
            "max_drift_20raw": max(
                abs(r["acc"] - r["conf"])
                for r in rel_rows
                if r["model"] == "XGB-20raw" and r["conf"] is not None
            ),
        },
    }
    with (RES / "p8_perfeat_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()