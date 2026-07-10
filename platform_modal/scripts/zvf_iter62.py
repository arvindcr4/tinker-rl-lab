#!/usr/bin/env python3
"""Pillar 2 -- Iter 62: difficulty-stratified ZVF cross-library diagnostic vs AERO.

Elevates ZVF from a single scalar to a *difficulty-stratified* function
ZVF(q) where q is a reward-quintile bin of training steps (proxy for the
prompt-difficulty distribution the policy currently sees). We then show
that AERO and the other variance-mitigation libraries attack the
*starvation* ZVF in the *lowest* reward quintile (hard prompts) more
strongly than vanilla GRPO, and that the gap reverses at the high end.

Inputs (real):
    experiments/results/variance_mitigation.tsv
        9 methods x 5 seeds x 100 steps (5540 rows total). Per-step
        (zvf, heldout_acc) pairs.

Outputs:
    experiments/results/zvf_iter62_difficulty_strata.tsv
        (method, quintile) -> mean_zvf, mean_acc, n_steps
    experiments/results/zvf_iter62_aero_minus_grpo.tsv
        per-quintile AERO-GRPO delta with bootstrap CI
    experiments/results/zvf_iter62_quintile_separability.tsv
        Mann-Whitney AUC per (method, quintile) for collapse vs healthy
    experiments/results/zvf_iter62_summary.tsv
        one-row-per-method summary of stratification
    figures/zvf_iter62_difficulty_strata.{pdf,png}

Methodology:
    1. Pool all (method, seed) steps; heldout_acc defines the "current
       difficulty the policy sees" (low acc ~ hard prompts dominate).
    2. Within each method, bin steps by heldout_acc into 5 equal-mass
       quintiles (so quintiles are defined per-method, isolating the
       cross-method contrast from any global accuracy shift).
    3. Per (method, quintile) emit mean ZVF, mean acc, n.
    4. Bootstrap the AERO - GRPO ZVF delta per quintile (B=2000).
    5. Per (method, quintile), run Mann-Whitney separability of
       ZVF|collapse vs ZVF|healthy (collapse=last10=0).

The whole point: a *single* ZVF scalar averages the hard-prompt tail
with the easy-prompt tail. A library that "reduces ZVF" by trimming
the easy tail (saturation) is not the same as one that reduces ZVF by
un-starving the hard tail. Difficulty stratification is what reveals
the difference.
"""

from __future__ import annotations
import csv
import json
import math
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path

random.seed(20260620)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

VM = RES / "variance_mitigation.tsv"


def load_variance_mitigation() -> list[dict]:
    rows = []
    with VM.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for r in reader:
            rows.append({
                "method": r["method"],
                "seed": int(r["seed"]),
                "step": int(r["step"]),
                "zvf": float(r["zvf"]),
                "acc": float(r["heldout_acc"]),
                "collapse": int(r["collapse"]),
            })
    return rows


def quintile_bin(values: list[float], n_bins: int = 5) -> list[int]:
    """Assign each value to an equal-mass bin 0..n_bins-1 using rank."""
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: values[i])
    bins = [0] * n
    for rank, idx in enumerate(order):
        b = min(int(rank * n_bins / n), n_bins - 1)
        bins[idx] = b
    return bins


def stratified_zvf(rows: list[dict]) -> dict:
    """For each method, bin steps by acc into 5 equal-mass quintiles."""
    out: dict = {}
    for method in sorted({r["method"] for r in rows}):
        sub = [r for r in rows if r["method"] == method]
        accs = [r["acc"] for r in sub]
        bins = quintile_bin(accs, 5)
        per_bin: dict = defaultdict(list)
        for r, b in zip(sub, bins):
            per_bin[b].append(r)
        method_out = []
        for b in range(5):
            bucket = per_bin[b]
            if not bucket:
                method_out.append({
                    "quintile": b, "n": 0, "mean_zvf": float("nan"),
                    "mean_acc": float("nan"), "zvf_se": float("nan"),
                    "n_collapse": 0,
                })
                continue
            zs = [r["zvf"] for r in bucket]
            acs = [r["acc"] for r in bucket]
            method_out.append({
                "quintile": b,
                "n": len(bucket),
                "mean_zvf": statistics.fmean(zs),
                "mean_acc": statistics.fmean(acs),
                "zvf_se": statistics.pstdev(zs) / math.sqrt(len(bucket)) if len(bucket) > 1 else 0.0,
                "n_collapse": sum(r["collapse"] for r in bucket),
            })
        out[method] = method_out
    return out


def bootstrap_diff(a: list[float], b: list[float], B: int = 2000) -> tuple[float, float, float]:
    """(mean diff, 2.5%ile, 97.5%ile) of a - b."""
    if not a or not b:
        return float("nan"), float("nan"), float("nan")
    diffs = []
    for _ in range(B):
        sa = [random.choice(a) for _ in range(len(a))]
        sb = [random.choice(b) for _ in range(len(b))]
        diffs.append(statistics.fmean(sa) - statistics.fmean(sb))
    diffs.sort()
    return statistics.fmean(diffs), diffs[int(0.025 * B)], diffs[int(0.975 * B)]


def per_method_per_quintile_zvf(rows: list[dict]) -> dict:
    """{method: {quintile: [zvf list]}}"""
    out: dict = {}
    for method in sorted({r["method"] for r in rows}):
        sub = [r for r in rows if r["method"] == method]
        accs = [r["acc"] for r in sub]
        bins = quintile_bin(accs, 5)
        d: dict = defaultdict(list)
        for r, b in zip(sub, bins):
            d[b].append(r["zvf"])
        out[method] = d
    return out


def mann_whitney_auc(xs: list[float], ys: list[float]) -> float:
    """AUC = P(x > y); on ties 0.5. 1.0 = xs always larger."""
    if not xs or not ys:
        return float("nan")
    wins = 0.0
    for x in xs:
        for y in ys:
            if x > y:
                wins += 1
            elif x == y:
                wins += 0.5
    return wins / (len(xs) * len(ys))


def main() -> None:
    rows = load_variance_mitigation()
    print(f"loaded {len(rows)} variance_mitigation rows; "
          f"{len({r['method'] for r in rows})} methods")

    strat = stratified_zvf(rows)
    per_mq = per_method_per_quintile_zvf(rows)

    # 1. write difficulty strata
    out1 = RES / "zvf_iter62_difficulty_strata.tsv"
    with out1.open("w") as fh:
        fh.write("# Pillar 2 Iter 62 difficulty-stratified ZVF\n")
        fh.write("# Per (method, heldout_acc quintile), mean ZVF.\n")
        fh.write("# Quintile 0 = lowest acc (hard-prompt-dominated training window);\n")
        fh.write("# Quintile 4 = highest acc (easy-prompt-dominated window).\n")
        fh.write("# Equal-mass bins PER METHOD, so cross-method contrast is not\n")
        fh.write("# confounded by global accuracy shifts.\n")
        fh.write("# Source: platform_modal/scripts/zvf_iter62.py\n")
        fh.write("method\tquintile\tacc_lo\tacc_hi\tn_steps\tmean_zvf\tzvf_se\tmean_acc\tn_collapse\n")
        for m in sorted(strat):
            quintiles = sorted(strat[m], key=lambda r: r["quintile"])
            for r in quintiles:
                # determine acc range for this quintile
                sub = [x for x in rows if x["method"] == m]
                accs = sorted(x["acc"] for x in sub)
                n = len(accs)
                lo_rank = int(r["quintile"] * n / 5)
                hi_rank = int((r["quintile"] + 1) * n / 5) - 1
                fh.write(f"{m}\t{r['quintile']}\t{accs[lo_rank]:.4f}\t{accs[hi_rank]:.4f}\t"
                         f"{r['n']}\t{r['mean_zvf']:.4f}\t{r['zvf_se']:.4f}\t"
                         f"{r['mean_acc']:.4f}\t{r['n_collapse']}\n")
    print(f"wrote {out1}")

    # 2. AERO - GRPO per-quintile ZVF delta
    out2 = RES / "zvf_iter62_aero_minus_grpo.tsv"
    with out2.open("w") as fh:
        fh.write("# AERO - GRPO ZVF delta per heldout_acc quintile\n")
        fh.write("# Negative = AERO has LOWER ZVF (better signal availability)\n")
        fh.write("# 95% bootstrap CI from B=2000 step-level resamples\n")
        fh.write("# Source: platform_modal/scripts/zvf_iter62.py\n")
        fh.write("quintile\taero_mean\tgrpo_mean\tdelta\tdelta_lo\tdelta_hi\tn_aero\tn_grpo\tinterpretation\n")
        for q in range(5):
            aero = per_mq.get("aero", {}).get(q, [])
            grpo = per_mq.get("grpo", {}).get(q, [])
            d, lo, hi = bootstrap_diff(aero, grpo, B=2000)
            if d != d:  # nan
                interp = "nan"
            elif hi < 0:
                interp = "AERO better (CI excl 0)"
            elif lo > 0:
                interp = "GRPO better (CI excl 0)"
            else:
                interp = "no significant diff"
            fh.write(f"{q}\t{statistics.fmean(aero) if aero else float('nan'):.4f}\t"
                     f"{statistics.fmean(grpo) if grpo else float('nan'):.4f}\t"
                     f"{d:.4f}\t{lo:.4f}\t{hi:.4f}\t{len(aero)}\t{len(grpo)}\t{interp}\n")
    print(f"wrote {out2}")

    # 3. Mann-Whitney separability per (method, quintile) of ZVF for collapse vs healthy
    out3 = RES / "zvf_iter62_quintile_separability.tsv"
    # collapse = any step with collapse flag in trailing-10 windows
    # here we use the per-step "collapse" flag directly
    collapse_zvfs = defaultdict(lambda: defaultdict(list))
    healthy_zvfs = defaultdict(lambda: defaultdict(list))
    for r in rows:
        b = quintile_bin([r["acc"]], 1)  # not used; re-bin per method
    for method in sorted({r["method"] for r in rows}):
        sub = [r for r in rows if r["method"] == method]
        accs = [r["acc"] for r in sub]
        bins = quintile_bin(accs, 5)
        for r, b in zip(sub, bins):
            if r["collapse"] == 1:
                collapse_zvfs[method][b].append(r["zvf"])
            else:
                healthy_zvfs[method][b].append(r["zvf"])
    with out3.open("w") as fh:
        fh.write("# Mann-Whitney AUC: P(ZVF_collapse > ZVF_healthy) per (method, quintile)\n")
        fh.write("# AUC > 0.5: collapsed runs have HIGHER ZVF; AUC < 0.5: lower\n")
        fh.write("# Note: many methods have 0 collapse steps; AUC = NaN there\n")
        fh.write("# Source: platform_modal/scripts/zvf_iter62.py\n")
        fh.write("method\tquintile\tauc\tn_collapse\tn_healthy\tmean_zvf_collapse\tmean_zvf_healthy\n")
        for m in sorted(collapse_zvfs):
            for q in range(5):
                cs = collapse_zvfs[m][q]
                hs = healthy_zvfs[m][q]
                if not cs or not hs:
                    auc = float("nan")
                else:
                    auc = mann_whitney_auc(cs, hs)
                fh.write(f"{m}\t{q}\t{auc:.4f}\t{len(cs)}\t{len(hs)}\t"
                         f"{statistics.fmean(cs) if cs else float('nan'):.4f}\t"
                         f"{statistics.fmean(hs) if hs else float('nan'):.4f}\n")
    print(f"wrote {out3}")

    # 4. one-row-per-method summary
    out4 = RES / "zvf_iter62_summary.tsv"
    with out4.open("w") as fh:
        fh.write("# Pillar 2 Iter 62: per-method ZVF stratification summary\n")
        fh.write("# mean_zvf_loq = mean ZVF in lowest acc quintile (hard prompts)\n")
        fh.write("# mean_zvf_hiq = mean ZVF in highest acc quintile (easy prompts)\n")
        fh.write("# delta_hiq_loq = high - low (positive => ZVF grows with accuracy)\n")
        fh.write("# Source: platform_modal/scripts/zvf_iter62.py\n")
        fh.write("method\tn_steps\tn_seeds\tmean_zvf_overall\tmean_zvf_loq\tmean_zvf_q2\t"
                 "mean_zvf_q3\tmean_zvf_q4\tmean_zvf_hiq\tdelta_hiq_loq\tmean_acc_overall\tn_collapse_steps\n")
        for m in sorted(strat):
            qs = {r["quintile"]: r for r in strat[m]}
            allsub = [r for r in rows if r["method"] == m]
            n_seeds = len({r["seed"] for r in allsub})
            overall_zvf = statistics.fmean(r["zvf"] for r in allsub)
            overall_acc = statistics.fmean(r["acc"] for r in allsub)
            lo = qs.get(0, {}).get("mean_zvf", float("nan"))
            hi = qs.get(4, {}).get("mean_zvf", float("nan"))
            d_hilo = (hi - lo) if (hi == hi and lo == lo) else float("nan")
            q2 = qs.get(1, {}).get("mean_zvf", float("nan"))
            q3 = qs.get(2, {}).get("mean_zvf", float("nan"))
            q4 = qs.get(3, {}).get("mean_zvf", float("nan"))
            n_coll = sum(r["collapse"] for r in allsub)
            fh.write(f"{m}\t{len(allsub)}\t{n_seeds}\t{overall_zvf:.4f}\t"
                     f"{lo:.4f}\t{q2:.4f}\t{q3:.4f}\t{q4:.4f}\t{hi:.4f}\t"
                     f"{d_hilo:.4f}\t{overall_acc:.4f}\t{n_coll}\n")
    print(f"wrote {out4}")

    # 5. figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        methods_ordered = ["grpo", "aero", "cppo", "ngrpo", "scafgrpo",
                           "mcgrpo", "gift", "areal", "es"]
        colors = {
            "grpo": "#888888", "aero": "#d62728", "cppo": "#1f77b4",
            "ngrpo": "#9467bd", "scafgrpo": "#2ca02c", "mcgrpo": "#ff7f0e",
            "gift": "#17becf", "areal": "#bcbd22", "es": "#8c564b",
        }
        ax = axes[0]
        for m in methods_ordered:
            if m not in strat:
                continue
            xs = [r["quintile"] for r in strat[m]]
            ys = [r["mean_zvf"] for r in strat[m]]
            ax.plot(xs, ys, "o-", color=colors.get(m, "k"), label=m, linewidth=2, markersize=6)
        ax.set_xlabel("Heldout-acc quintile (0=hard, 4=easy)")
        ax.set_ylabel("Mean ZVF (per-step)")
        ax.set_title("ZVF vs difficulty, per library")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best", ncol=2)
        # delta panel
        ax2 = axes[1]
        deltas = []
        labels = []
        with (RES / "zvf_iter62_aero_minus_grpo.tsv").open() as fh:
            rdr = csv.DictReader((l for l in fh if not l.startswith("#")), delimiter="\t")
            for i, r in enumerate(rdr):
                deltas.append(float(r["delta"]))
                labels.append(f"q{r['quintile']}")
        ax2.bar(labels, deltas,
                color=["#d62728" if d < 0 else "#1ca02c" for d in deltas],
                alpha=0.7, edgecolor="black")
        ax2.axhline(0, color="black", linewidth=1)
        ax2.set_xlabel("Heldout-acc quintile")
        ax2.set_ylabel("AERO - GRPO ZVF (per-step mean)")
        ax2.set_title("AERO-GRPO ZVF delta by quintile")
        ax2.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        for ext in ("pdf", "png"):
            p = FIG / f"zvf_iter62_difficulty_strata.{ext}"
            plt.savefig(p, bbox_inches="tight")
            print(f"wrote {p}")
        plt.close()
    except ImportError:
        print("matplotlib not available; skipping figure")

    # 6. JSON findings blob
    findings = {
        "iter": 62,
        "n_rows": len(rows),
        "n_methods": len({r["method"] for r in rows}),
        "aero_grpo_q0_delta": None,
        "aero_grpo_q4_delta": None,
    }
    with (RES / "zvf_iter62_aero_minus_grpo.tsv").open() as fh:
        rdr = csv.DictReader((l for l in fh if not l.startswith("#")), delimiter="\t")
        for r in rdr:
            q = int(r["quintile"])
            d = float(r["delta"])
            if q == 0:
                findings["aero_grpo_q0_delta"] = d
            if q == 4:
                findings["aero_grpo_q4_delta"] = d
    with (RES / "zvf_iter62_iter_meta.json").open("w") as fh:
        json.dump(findings, fh, indent=2)
    print(f"wrote {RES / 'zvf_iter62_iter_meta.json'}")


if __name__ == "__main__":
    main()
