#!/usr/bin/env python3
"""P8 JOB A (iter 68): single-sensor-feature ablations + cost-per-decision at
varying K budgets.

Fresh vein (not in the prior 77 P8 rows). Iter-24 (#31) did leave-one-OUT and
leaves-one-IN ablations of the four aggregates inside the 24-feature tree.
Iter-64 (#75) measured subgroup alert-distribution fairness.

This iter closes two distinct reviewer-visible gaps:

  G1. **Single-aggregate tree** -- what does each sensor feature
      (V_mean / V_std / V_max / V_min) carry ALONE when grafted onto the
      20 raw features? 4 single-aggregate trees (20 raw + 1 sensor) vs the
      24-full baseline. Sharpens iter-31 from "remove 1 from 24" to
      "does each aggregate carry ANY signal by itself".

  G2. **Pair-aggregate trees** (4 choose 2 = 6 pairs) -- the smallest sensor
      block that still captures the cross-axis information (mean AND
      dispersion, or mean AND extremes). Measures whether pairs add
      independent information over single-aggregate trees.

  G3. **Cost-per-decision at 5 K budgets** -- K=0.5%, 1%, 2%, 3%, 5%.
      Iter-28 (#35) and iter-56 (#66) measured K=2% dominance-switch; the
      whole cost story rests on one operating point. Here we plot
      $/fraud_caught vs K for every variant and report the K at which the
      cost-per-decision curve crosses the $5/alert gate (operational SLO).
      Cost model: XGB-20raw / XGB-24full / XGB-4sensor at $0.0001/decision;
      the LLM-as-scribe surrogate (which extracts the 4 sensor features)
      at $0.001/decision -- an order of magnitude more expensive, but only
      once per row (so it dominates per-decision cost only at very low K).

Inputs
------
fraud_data.csv : 50k synthetic fraud rows (24 numeric features + Class).
test_data.csv  : 10k held-out rows (same schema + Class).

Outputs
-------
platform_hybrid/experiments/results/p5p8/p8_single_sensor.tsv          (4 rows: 4 single)
platform_hybrid/experiments/results/p5p8/p8_pair_sensor.tsv           (6 rows: 6 pairs)
platform_hybrid/experiments/results/p5p8/p8_single_pair_boot.tsv      (paired bootstrap CIs)
platform_hybrid/experiments/results/p5p8/p8_cost_per_decision.tsv     (4 models x 5 K)
platform_hybrid/experiments/results/p5p8/p8_cost_per_decision_boot.tsv
platform_hybrid/experiments/results/p5p8/p8_single_sensor_summary.json
platform_hybrid/experiments/results/p5p8/figures/p8_single_sensor.{png,pdf}
platform_hybrid/experiments/results/p5p8/figures/p8_cost_per_decision.{png,pdf}

Stdlib + numpy + pandas + xgboost + matplotlib. <=290 lines.
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

SEED = 20260705
N_BOOT = 600
K_BUDGETS = [0.5, 1.0, 2.0, 3.0, 5.0]  # percent of test alerted
RAW20 = [f"V{i}" for i in range(1, 21)]
AGG4 = ["V_mean", "V_std", "V_max", "V_min"]
ALL24 = RAW20 + AGG4
COST_XGB = 0.0001   # dollars per decision for tree
COST_LLM = 0.0010   # dollars per decision for LLM sensor (once per row)


def load(path: Path):
    rows, labels, header = [], [], None
    with path.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)
        idx = {n: i for i, n in enumerate(header)}
        for line in rdr:
            rows.append([float(line[idx[c]]) for c in ALL24])
            labels.append(int(float(line[idx["Class"]])))
    return rows, labels, header


def auc_roc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    n_pos, n_neg = len(pos), len(neg)
    # Mann-Whitney U / (n_pos * n_neg)
    pos_sorted = sorted(pos)
    neg_sorted = sorted(neg)
    # Count pairs (n_p, n_g) where n_p > n_g
    inv = 0
    j = 0
    for p in pos_sorted:
        while j < n_neg and neg_sorted[j] < p:
            j += 1
        inv += j
    return inv / (n_pos * n_neg)


def brier(scores, labels):
    return sum((s - y) ** 2 for s, y in zip(scores, labels)) / len(labels)


def precision_at_k(scores, labels, k_pct):
    n = len(scores)
    k = max(1, int(round(n * k_pct / 100.0)))
    order = sorted(range(n), key=lambda i: -scores[i])
    top = order[:k]
    tp = sum(1 for i in top if labels[i] == 1)
    return tp / k, tp


def recall_at_k(scores, labels, k_pct):
    n = len(scores)
    k = max(1, int(round(n * k_pct / 100.0)))
    order = sorted(range(n), key=lambda i: -scores[i])
    top = order[:k]
    pos = sum(1 for y in labels if y == 1)
    tp = sum(1 for i in top if labels[i] == 1)
    return tp / pos if pos else 0.0, tp


def f1_at_k(scores, labels, k_pct):
    p, _ = precision_at_k(scores, labels, k_pct)
    r, _ = recall_at_k(scores, labels, k_pct)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def fit_xgb(X_tr, y_tr, X_te, n_est=300, depth=5, lr=0.1, seed=SEED):
    import xgboost as xgb

    m = xgb.XGBClassifier(
        n_estimators=n_est,
        max_depth=depth,
        learning_rate=lr,
        random_state=seed,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=2,
    )
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    return p.tolist()


def paired_bootstrap_ci(metric_a, metric_b, B=N_BOOT, seed=SEED):
    """Paired percentile CI on per-row differences, but here metrics are
    scalar (AUC / Brier / cost). We resample the test labels to estimate
    the metric variance, then compute paired-diff CIs."""
    rng = random.Random(seed)
    n = len(metric_a["scores"])
    a0 = metric_a["fn"](metric_a["scores"], metric_a["labels"])
    b0 = metric_b["fn"](metric_b["scores"], metric_b["labels"])
    diffs = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        sa = [metric_a["scores"][i] for i in idx]
        la = [metric_a["labels"][i] for i in idx]
        sb = [metric_b["scores"][i] for i in idx]
        lb = [metric_b["labels"][i] for i in idx]
        diffs.append(metric_a["fn"](sa, la) - metric_b["fn"](sb, lb))
    diffs.sort()
    lo = diffs[int(0.025 * B)]
    hi = diffs[int(0.975 * B) - 1]
    return a0, b0, a0 - b0, lo, hi


def cost_per_caught(scores, labels, k_pct, cost_per_decision):
    n = len(scores)
    k = max(1, int(round(n * k_pct / 100.0)))
    order = sorted(range(n), key=lambda i: -scores[i])
    top = order[:k]
    tp = sum(1 for i in top if labels[i] == 1)
    total_cost = k * cost_per_decision
    return total_cost / tp if tp else float("inf"), tp, k


def main():
    print(f"[p8_single_sensor] loading fraud_data.csv + test_data.csv")
    tr_rows, tr_labels, _ = load(ROOT / "fraud_data.csv")
    te_rows, te_labels, _ = load(ROOT / "test_data.csv")
    print(f"  train n={len(tr_rows)} pos={sum(tr_labels)}")
    print(f"  test  n={len(te_rows)} pos={sum(te_labels)}")

    # ---- Build the 4 single-sensor trees + 6 pair-sensor trees ----------
    variants = []
    # Baseline 20-raw
    variants.append(("XGB-20raw", RAW20, COST_XGB))
    # Baseline 24-full (4 sensor aggregates)
    variants.append(("XGB-24full", ALL24, COST_XGB))
    # 4 single-aggregate trees
    for s in AGG4:
        variants.append((f"XGB-20raw+{s}", RAW20 + [s], COST_XGB))
    # 6 pair-aggregate trees
    for s1, s2 in combinations(AGG4, 2):
        variants.append((f"XGB-20raw+{s1}+{s2}", RAW20 + [s1, s2], COST_XGB))

    # Fit
    print(f"[p8_single_sensor] fitting {len(variants)} trees")
    X_tr_all = [dict(zip(ALL24, r)) for r in tr_rows]
    X_te_all= [dict(zip(ALL24, r)) for r in te_rows]
    scores = {}
    for name, feats, cost_d in variants:
        X_tr = [[row[c] for c in feats] for row in X_tr_all]
        X_te = [[row[c] for c in feats] for row in X_te_all]
        scores[name] = (fit_xgb(X_tr, tr_labels, X_te), cost_d)

    # ---- AUC / Brier summary -------------------------------------------
    abl_rows = []
    for name, _, _ in variants:
        s, _ = scores[name]
        a = auc_roc(s, te_labels)
        b = brier(s, te_labels)
        abl_rows.append({"variant": name, "auc": round(a, 4), "brier": round(b, 4),
                         "n_features": len([f for f in (name.split("+")[1:] or [])
                                            if f in AGG4]) + 20})

    with (RES / "p8_single_sensor.tsv").open("w") as f:
        f.write("variant\tn_features\tauc\tbrier\n")
        for r in abl_rows:
            f.write(f"{r['variant']}\t{r['n_features']}\t{r['auc']}\t{r['brier']}\n")
    with (RES / "p8_pair_sensor.tsv").open("w") as f:
        f.write("variant\tn_features\tauc\tbrier\n")
        for r in abl_rows:
            if "+V_" in r["variant"] or "+V_m" in r["variant"]:
                f.write(f"{r['variant']}\t{r['n_features']}\t{r['auc']}\t{r['brier']}\n")

    # ---- Paired bootstrap CIs vs XGB-24full -----------------------------
    base_s, _ = scores["XGB-24full"]
    boot_rows = []
    for name, _, _ in variants:
        if name == "XGB-24full":
            continue
        s, _ = scores[name]
        a0, b0, d, lo, hi = paired_bootstrap_ci(
            {"scores": s, "labels": te_labels, "fn": auc_roc},
            {"scores": base_s, "labels": te_labels, "fn": auc_roc},
        )
        boot_rows.append({"variant": name, "auc": round(a0, 4),
                          "auc_24full": round(b0, 4), "delta_auc": round(d, 4),
                          "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                          "excludes_zero": "yes" if (lo > 0 or hi < 0) else "no"})
    with (RES / "p8_single_pair_boot.tsv").open("w") as f:
        f.write("variant\tauc\tauc_24full\tdelta_auc\tci_lo\tci_hi\texcludes_zero\n")
        for r in boot_rows:
            f.write(f"{r['variant']}\t{r['auc']}\t{r['auc_24full']}\t{r['delta_auc']}\t{r['ci_lo']}\t{r['ci_hi']}\t{r['excludes_zero']}\n")

    # ---- Cost-per-decision at 5 K budgets -------------------------------
    cost_rows = []
    cost_boot = []
    base_models = ["XGB-20raw", "XGB-24full"]
    # Add the best single-sensor tree (chosen by AUC)
    best_single = max(
        (r for r in abl_rows if "20raw+V_" in r["variant"] or "20raw+V_m" in r["variant"]),
        key=lambda r: r["auc"],
    )["variant"]
    base_models.append(best_single)
    # Add the best pair tree
    best_pair = max(
        (r for r in abl_rows if r["variant"].count("+") == 2 and "20raw" in r["variant"]),
        key=lambda r: r["auc"],
    )["variant"]
    base_models.append(best_pair)
    # LLM-as-scribe surrogate: same as 24-full but cost_per_decision = COST_LLM
    base_models.append("LLM-scribe-surrogate")

    for k_pct in K_BUDGETS:
        for model in base_models:
            if model == "LLM-scribe-surrogate":
                s, _ = scores["XGB-24full"]
                cost_d = COST_LLM
            else:
                s, cost_d = scores[model]
            cpc, tp, k = cost_per_caught(s, te_labels, k_pct, cost_d)
            p, _ = precision_at_k(s, te_labels, k_pct)
            r, _ = recall_at_k(s, te_labels, k_pct)
            cost_rows.append({"model": model, "k_pct": k_pct, "k_alerts": k,
                              "true_pos": tp, "precision": round(p, 4),
                              "recall": round(r, 4),
                              "cost_per_caught_dollars": round(cpc, 4)})
    with (RES / "p8_cost_per_decision.tsv").open("w") as f:
        f.write("model\tk_pct\tk_alerts\ttrue_pos\tprecision\trecall\tcost_per_caught_dollars\n")
        for r in cost_rows:
            f.write(f"{r['model']}\t{r['k_pct']}\t{r['k_alerts']}\t{r['true_pos']}\t{r['precision']}\t{r['recall']}\t{r['cost_per_caught_dollars']}\n")

    # ---- Cost-per-decision bootstrap CIs (XGB-24full vs others at each K) -
    for k_pct in K_BUDGETS:
        for model in base_models:
            if model == "XGB-24full":
                continue
            if model == "LLM-scribe-surrogate":
                s_a, _ = scores["XGB-24full"]
                cost_d_a = COST_XGB
                s_b, _ = scores["XGB-24full"]
                cost_d_b = COST_LLM
            else:
                s_a, cost_d_a = scores["XGB-24full"]
                s_b, cost_d_b = scores[model]
            # Bootstrap cost-per-caught
            rng = random.Random(SEED + int(k_pct * 10) + hash(model) % 1000)
            n = len(te_labels)
            diffs = []
            for _ in range(N_BOOT):
                idx = [rng.randrange(n) for _ in range(n)]
                sa = [s_a[i] for i in idx]
                la = [te_labels[i] for i in idx]
                sb = [s_b[i] for i in idx]
                lb = [te_labels[i] for i in idx]
                ca, _, _ = cost_per_caught(sa, la, k_pct, cost_d_a)
                cb, _, _ = cost_per_caught(sb, lb, k_pct, cost_d_b)
                if cb == 0 and ca == 0:
                    diffs.append(0.0)
                elif cb == float("inf") or ca == float("inf"):
                    diffs.append(0.0)
                else:
                    diffs.append(ca - cb)
            diffs.sort()
            lo = diffs[int(0.025 * N_BOOT)]
            hi = diffs[int(0.975 * N_BOOT) - 1]
            cost_boot.append({"model": model, "k_pct": k_pct,
                              "delta_cost_per_caught": round(diffs[len(diffs) // 2], 4),
                              "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                              "excludes_zero": "yes" if (lo > 0 or hi < 0) else "no"})
    with (RES / "p8_cost_per_decision_boot.tsv").open("w") as f:
        f.write("model\tk_pct\tdelta_cost_per_caught\tci_lo\tci_hi\texcludes_zero\n")
        for r in cost_boot:
            f.write(f"{r['model']}\t{r['k_pct']}\t{r['delta_cost_per_caught']}\t{r['ci_lo']}\t{r['ci_hi']}\t{r['excludes_zero']}\n")

    # ---- Figures --------------------------------------------------------
    # Figure 1: AUC bar chart with bootstrap CIs
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    labels_x = [r["variant"] for r in abl_rows]
    vals = [r["auc"] for r in abl_rows]
    boot_by_var = {r["variant"]: r for r in boot_rows}
    auc_24 = next(r["auc"] for r in abl_rows if r["variant"] == "XGB-24full")
    errs_lo = []
    errs_hi = []
    for r in abl_rows:
        if r["variant"] == "XGB-24full":
            errs_lo.append(0.0)
            errs_hi.append(0.0)
            continue
        b = boot_by_var.get(r["variant"])
        if b:
            # CI on the difference a - b; translate to bounds on a as (b + ci_lo, b + ci_hi)
            a_lo = auc_24 + b["ci_lo"]
            a_hi = auc_24 + b["ci_hi"]
            errs_lo.append(max(0.0, r["auc"] - a_lo))
            errs_hi.append(max(0.0, a_hi - r["auc"]))
        else:
            errs_lo.append(0.0)
            errs_hi.append(0.0)
    ax.bar(range(len(labels_x)), vals, yerr=[errs_lo, errs_hi], capsize=3)
    ax.set_xticks(range(len(labels_x)))
    ax.set_xticklabels(labels_x, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("AUC (test split)")
    ax.set_title("P8 single-sensor ablation — AUC with 95% paired bootstrap CI vs XGB-24full")
    ax.axhline(vals[labels_x.index("XGB-24full")], color="red", linestyle="--",
               label="XGB-24full baseline")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "p8_single_sensor.png", dpi=150)
    fig.savefig(FIG / "p8_single_sensor.pdf")
    plt.close(fig)

    # Figure 2: cost-per-decision vs K budget for each model
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    colors = {"XGB-20raw": "tab:blue", "XGB-24full": "tab:red",
best_single: "tab:green", best_pair: "tab:purple",
              "LLM-scribe-surrogate": "tab:orange"}
    for model in base_models:
        ks = [r["k_pct"] for r in cost_rows if r["model"] == model]
        cs = [r["cost_per_caught_dollars"] for r in cost_rows if r["model"] == model]
        ax.plot(ks, cs, marker="o", label=model, color=colors.get(model, "gray"))
    ax.set_xlabel("Global alert budget K (%)")
    ax.set_ylabel("$ / fraud caught (test)")
    ax.set_yscale("log")
    ax.set_title("P8 cost-per-fraud-caught across alert-budget regimes")
    ax.axhline(5.0, color="red", linestyle=":", label="$5/alert SLO")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "p8_cost_per_decision.png", dpi=150)
    fig.savefig(FIG / "p8_cost_per_decision.pdf")
    plt.close(fig)

    # ---- Summary JSON ---------------------------------------------------
    summary = {
        "n_train": len(tr_rows), "n_test": len(te_rows),
        "n_pos_test": sum(te_labels), "seed": SEED, "n_boot": N_BOOT,
        "k_budgets": K_BUDGETS,
        "abl_rows": len(abl_rows),
        "boot_rows": len(boot_rows),
        "n_excludes_zero": sum(1 for r in boot_rows if r["excludes_zero"] == "yes"),
        "best_single_aggregate": best_single,
        "best_pair": best_pair,
        "cost_rows": len(cost_rows),
        "cost_boot_rows": len(cost_boot),
        "cost_boot_excludes_zero": sum(1 for r in cost_boot if r["excludes_zero"] == "yes"),
    }
    with (RES / "p8_single_sensor_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()