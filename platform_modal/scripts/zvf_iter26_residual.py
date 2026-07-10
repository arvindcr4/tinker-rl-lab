#!/usr/bin/env python3
"""Iter 26 — ZVF residualised finding.

Question: does mean ZVF carry *residual* predictive signal for the
last-10 heldout accuracy after we control for mean reward? The naive
r(ZVF, last10) is confounded because both are dominated by task
difficulty. We regress last10 ~ reward on each library slice, then
correlate the residuals against ZVF.

Outputs:
    experiments/results/zvf_iter26_residual.tsv
        per-library slice: n, naive r, r(reward,last10), residual r.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "experiments" / "results"


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def _bootstrap(xs, ys, metric, B=2000, seed=0):
    import random
    rng = random.Random(seed)
    n = len(xs)
    if n < 3:
        return (float("nan"), (float("nan"), float("nan")))
    samples = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        v = metric([xs[i] for i in idx], [ys[i] for i in idx])
        if not (isinstance(v, float) and math.isnan(v)):
            samples.append(v)
    if not samples:
        return (float("nan"), (float("nan"), float("nan")))
    samples.sort()
    return (
        metric(xs, ys),
        (samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples)) - 1]),
    )


def load_summary(path: Path):
    rows = []
    header = None
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if header is None:
                header = parts
                continue
            if len(parts) < len(header):
                continue
            r = dict(zip(header, parts))
            # Parse numerics; skip NA
            try:
                r["mean_zvf"] = float(r["mean_zvf"])
                r["last10_avg"] = float(r["last10_avg"])
                r["mean_reward"] = float(r["mean_reward"])
            except (ValueError, KeyError):
                continue
            if any(math.isnan(v) for v in (r["mean_zvf"], r["last10_avg"], r["mean_reward"])):
                continue
            rows.append(r)
    return rows


def pool_per_experiment_model(rows):
    pools = defaultdict(list)
    for r in rows:
        key = (r["experiment"], r["model"], r["group_size"])
        pools[key].append(r)
    pooled = []
    for key, recs in pools.items():
        pooled.append(
            {
                "experiment": key[0],
                "model": key[1],
                "group_size": key[2],
                "mean_zvf": statistics.fmean(r["mean_zvf"] for r in recs),
                "last10": statistics.fmean(r["last10_avg"] for r in recs),
                "mean_reward": statistics.fmean(r["mean_reward"] for r in recs),
            }
        )
    return pooled


def analyse_slice(slice_rows, label, B=2000, seed=0):
    """Run naive + residual + reward correlations on one slice."""
    n = len(slice_rows)
    out = {"label": label, "n_pooled": n}
    if n < 5:
        out.update({"naive_r": "NA", "reward_r": "NA", "residual_r": "NA",
                    "naive_lo": "NA", "naive_hi": "NA",
                    "residual_lo": "NA", "residual_hi": "NA"})
        return out
    zvfs = [r["mean_zvf"] for r in slice_rows]
    lasts = [r["last10"] for r in slice_rows]
    rewards = [r["mean_reward"] for r in slice_rows]

    naive = _bootstrap(zvfs, lasts, _pearson, B=B, seed=seed)
    reward = _bootstrap(rewards, lasts, _pearson, B=B, seed=seed + 1)
    # Residual last10 ~ reward
    mx = statistics.fmean(rewards)
    my = statistics.fmean(lasts)
    num = sum((x - mx) * (y - my) for x, y in zip(rewards, lasts))
    den = sum((x - mx) ** 2 for x in rewards)
    b1 = num / den if den != 0 else 0.0
    b0 = my - b1 * mx
    resid = [y - (b0 + b1 * x) for x, y in zip(rewards, lasts)]
    residual = _bootstrap(zvfs, resid, _pearson, B=B, seed=seed + 2)

    out.update(
        {
            "naive_r": f"{naive[0]:+.3f}",
            "naive_lo": f"{naive[1][0]:+.3f}",
            "naive_hi": f"{naive[1][1]:+.3f}",
            "reward_r": f"{reward[0]:+.3f}",
            "residual_r": f"{residual[0]:+.3f}",
            "residual_lo": f"{residual[1][0]:+.3f}",
            "residual_hi": f"{residual[1][1]:+.3f}",
        }
    )
    return out


def _library_level_pool(rows):
    """Per-step rows in zvf_summary.tsv are autocorrelated; we instead
    pool at the (experiment, model, group_size) level for variance_mitigation
    and report one row per (method) with method-level means across all of
    its per-step rows. This keeps the matched-stack cross-library signal
    honest (each method has 500-1500 rollout steps)."""
    # Read the underlying variance_mitigation.tsv directly.
    path = RESULTS / "variance_mitigation.tsv"
    method_groups: dict[str, list] = {}
    if path.exists():
        with path.open() as fh:
            rdr = csv.DictReader(fh, delimiter="\t")
            for r in rdr:
                method_groups.setdefault(r["method"], []).append(
                    (float(r["zvf"]), float(r["heldout_acc"]), float(r["reward_mean"]))
                )
    method_pool = []
    for m, vals in sorted(method_groups.items()):
        zs = [v[0] for v in vals]
        ls = [v[1] for v in vals]
        rs = [v[2] for v in vals]
        method_pool.append(
            {
                "method": m,
                "mean_zvf": statistics.fmean(zs),
                "last10": statistics.fmean(ls),
                "mean_reward": statistics.fmean(rs),
                "n_steps": len(vals),
            }
        )
    return method_pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=RESULTS / "zvf_iter26_residual.tsv"
    )
    parser.add_argument("--B", type=int, default=2000)
    args = parser.parse_args()

    rows = load_summary(RESULTS / "zvf_summary.tsv")
    pooled = pool_per_experiment_model(rows)

    by_lib: dict[str, list] = defaultdict(list)
    for p in pooled:
        by_lib[p["experiment"]].append(p)

    method_pool = _library_level_pool(rows)
    # Non-drift slice excludes MCGRPO/GIFT/AREAL/ES (each has 1/5 drift rows).
    no_drift_methods = [m for m in method_pool if m["method"] not in ("mcgrpo", "gift", "areal", "es")]

    # Define analysis slices.
    slices = [
        ("pooled_all_per_experiment", pooled),
        ("pooled_variance_mitigation_per_methodseed", by_lib["variance_mitigation"]),
        ("method_level_variance_mitigation", method_pool),
        ("method_level_variance_mitigation_no_drift", no_drift_methods),
        ("groupsize_sweep", by_lib["groupsize_zvf_sweep"]),
        ("drgrpo_vs_grpo", by_lib["drgrpo_vs_grpo"]),
        ("samestack_ppo_grpo", by_lib["samestack_ppo_grpo"]),
        ("tinker_gsm8k_zvf", by_lib["tinker_gsm8k_zvf"]),
    ]

    # De-dup and require at least 3 rows (n=9 is fine for the method-level slice).
    out_rows = []
    for label, slice_rows in slices:
        if len(slice_rows) < 3:
            continue
        out_rows.append(analyse_slice(slice_rows, label, B=args.B))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.write("# Iter 26 Pillar 2: residualised ZVF finding\n")
        fh.write(
            "# Per-slice table: n_pooled, naive r(ZVF, last10),\n"
            "# r(reward, last10) (the confounder), and\n"
            "# r(residual(last10 ~ reward), ZVF) -- the residualised\n"
            "# partial correlation that isolates the ZVF contribution.\n"
            "# Bootstrap CIs use B=2000 percentile resamples.\n"
            "# Source: platform_modal/scripts/zvf_iter26_residual.py\n"
        )
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "slice",
                "n_pooled",
                "naive_r",
                "naive_lo",
                "naive_hi",
                "reward_r",
                "residual_r",
                "residual_lo",
                "residual_hi",
            )
        )
        for r in out_rows:
            writer.writerow(
                [
                    r["label"],
                    r["n_pooled"],
                    r["naive_r"],
                    r["naive_lo"],
                    r["naive_hi"],
                    r["reward_r"],
                    r["residual_r"],
                    r["residual_lo"],
                    r["residual_hi"],
                ]
            )

    # Headline print.
    for r in out_rows:
        print(
            f"[zvf-residual] {r['label']:>40s} n={r['n_pooled']:>3d} "
            f"naive={r['naive_r']:>6s} reward={r['reward_r']:>6s} "
            f"residual={r['residual_r']:>6s}"
        )
    print(f"[zvf-residual] wrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())