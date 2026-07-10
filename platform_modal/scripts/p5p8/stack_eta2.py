#!/usr/bin/env python3
"""P5-03 — Stack-conditioning quantification via eta^2 decomposition.

Uses the N2 four-method same-stack tensors (grpo, aero, gift, areal; each
n=40 steps on the same stack: same model, G=8, seed=0, task slice) and the
G-sweep data (4 G values × 3 seeds each) to estimate variance-explained
(eta^2 = SS_between / SS_total) along two axes:

  axis A = algorithm axis   (4 methods, fixed stack)     — should be small
  axis B = group-size axis  (4 G values, 3 seeds each)  — should be large

If "stack conditions everything", eta^2_A << eta^2_B for telemetry that is
genuinely stack-driven (zvf, pcd, reward). This is a quantitative version of
the Pillar-1 "estimator equivalence / stack-conditioning" claim.

Outputs:
  platform_hybrid/experiments/results/p5p8/stack_eta2.tsv
  platform_hybrid/experiments/results/p5p8/stack_eta2.json
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N2 = ROOT / "experiments" / "results" / "n2_reward_tensor_resume" / "n2_metrics.tsv"
SWEEP = ROOT / "experiments" / "results" / "groupsize_zvf_sweep.tsv"
OUT_DIR = ROOT / "experiments" / "results" / "p5p8"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_n2():
    """Return dict[metric][method] -> list of values."""
    out: dict[str, dict[str, list[float]]] = {}
    with N2.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if not parts or parts[0] == "":
                continue
            d = dict(zip(header, parts))
            for k in header:
                if k in ("method", "seed"):
                    continue
                out.setdefault(k, {}).setdefault(d["method"], []).append(
                    float(d[k]))
    return out


def load_sweep():
    """Return dict[metric][G] -> list of (mean, se) pairs."""
    out: dict[str, dict[int, list[float]]] = {}
    with SWEEP.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if not parts or parts[0] == "":
                continue
            d = dict(zip(header, parts))
            g = int(d["G"])
            for k in ("mean_zvf", "mean_reward_train", "heldout_acc_mean"):
                out.setdefault(k, {}).setdefault(g, []).append(float(d[k]))
    return out


def eta2_from_groups(groups: list[list[float]]):
    """eta^2 = SS_between / SS_total for unequal group sizes (simple ANOVA)."""
    pooled = [v for g in groups for v in g]
    n_total = len(pooled)
    if n_total < 2:
        return float("nan"), float("nan"), 0
    grand = sum(pooled) / n_total
    ss_total = sum((v - grand) ** 2 for v in pooled)
    ss_within = 0.0
    ss_between = 0.0
    for g in groups:
        if not g:
            continue
        gm = sum(g) / len(g)
        ss_between += len(g) * (gm - grand) ** 2
        ss_within += sum((v - gm) ** 2 for v in g)
    if ss_total <= 0:
        return 0.0, 0.0, len(groups)
    return ss_between / ss_total, ss_between / max(1e-12, ss_within), len(groups)


def eta2_from_means_within(groups: list[list[float]]):
    """eta^2 if each group is summarised by its mean and SS_within is the
    sum of squared SEs (sigma² within = SE² * n). Used for the G-sweep axis
    where we only have means + SEs, not per-seed raw values."""
    pooled = [v for g in groups for v in g]
    n_total = len(pooled)
    grand = sum(pooled) / n_total
    ss_total = sum((v - grand) ** 2 for v in pooled)
    ss_between = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups)
    return ss_between / max(1e-12, ss_total)


def main() -> int:
    n2 = load_n2()
    sweep = load_sweep()
    rows = []

    # Axis A: algorithm (4 methods, 40 steps each, same stack)
    for metric in ("zvf", "pcd", "reward_mean", "mean_len", "cv_len", "loss"):
        groups = [n2.get(metric, {}).get(m, []) for m in
                  ("grpo", "aero", "gift", "areal")]
        eta2, f_ratio, n_groups = eta2_from_groups(groups)
        rows.append({
            "axis": "A_algorithm",
            "metric": metric,
            "n_groups": n_groups,
            "per_group_n": ",".join(str(len(g)) for g in groups),
            "eta2": round(eta2, 4) if not math.isnan(eta2) else "n/a",
            "f_ratio": round(f_ratio, 4) if not math.isnan(f_ratio) else "n/a",
            "groups": "|".join(("grpo,aero,gift,areal")),
        })

    # Axis B: G-sweep (4 G values, 3 seeds each)
    for metric in ("mean_zvf", "mean_reward_train", "heldout_acc_mean"):
        groups = [sweep.get(metric, {}).get(g, []) for g in (2, 4, 8, 16)]
        eta2 = eta2_from_means_within(groups)
        rows.append({
            "axis": "B_G",
            "metric": metric,
            "n_groups": len(groups),
            "per_group_n": ",".join(str(len(g)) for g in groups),
            "eta2": round(eta2, 4) if not math.isnan(eta2) else "n/a",
            "f_ratio": "n/a (means only)",
            "groups": "G=2|G=4|G=8|G=16",
        })

    out_tsv = OUT_DIR / "stack_eta2.tsv"
    with out_tsv.open("w", newline="") as f:
        cols = ["axis", "metric", "n_groups", "per_group_n",
                "eta2", "f_ratio", "groups"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Headline summary: compare eta^2 along the two axes for matching metrics
    headline = []
    metric_pairs = [("zvf", "mean_zvf"), ("reward_mean", "mean_reward_train")]
    for n2_metric, sweep_metric in metric_pairs:
        a = next((r for r in rows
                  if r["axis"] == "A_algorithm" and r["metric"] == n2_metric),
                 None)
        b = next((r for r in rows
                  if r["axis"] == "B_G" and r["metric"] == sweep_metric),
                 None)
        if a and b and a["eta2"] != "n/a" and b["eta2"] != "n/a":
            ratio = b["eta2"] / max(1e-6, a["eta2"])
            headline.append({
                "metric": n2_metric,
                "eta2_algorithm": a["eta2"],
                "eta2_G": b["eta2"],
                "ratio_G_over_algorithm": round(ratio, 2),
            })

    summary = {
        "rows": rows,
        "headline": headline,
        "interpretation": [
            "If the Pillar-1 'stack conditions everything' claim holds, then",
            "eta^2 across the G axis should be >> eta^2 across the algorithm",
            "axis for the same telemetry (zvf, reward). The headline table",
            "tests this directly on the four-method same-stack N2 data.",
        ],
    }
    with (OUT_DIR / "stack_eta2.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("eta^2 table:")
    for r in rows:
        print(f"  {r['axis']:>11} | {r['metric']:>17} | "
              f"eta^2 = {r['eta2']}")
    print("headline comparison:")
    for h in headline:
        print(f"  {h['metric']:>17} | "
              f"eta^2(alg)={h['eta2_algorithm']:.4f}  "
              f"eta^2(G)={h['eta2_G']:.4f}  "
              f"ratio={h['ratio_G_over_algorithm']:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())