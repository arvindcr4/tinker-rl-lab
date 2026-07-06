#!/usr/bin/env python3
"""
length_zvf_bootstrap.py — paired bootstrap CIs on length×ZVF coupling.

Tests:
  T1: per-(task, algo) ρ(len, ZVF) point estimate and 95% bootstrap CI
  T2: paired Δρ(len, ZVF) between Dr.GRPO and GRPO with paired bootstrap CI
  T3: length-bin conditional ZVF paired delta (L1 vs L3) per (task, algo)
"""
import json
import csv
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "experiments" / "results"


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.array(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return float("nan"), float("nan"), float("nan")
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(vals), size=len(vals))
        boots.append(vals[idx].mean())
    return float(vals.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_bootstrap_ci(diff_values, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.array(diff_values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return float("nan"), float("nan"), float("nan")
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(vals), size=len(vals))
        boots.append(vals[idx].mean())
    return float(vals.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def main():
    rows = list(csv.DictReader(open(RESULTS / "length_zvf_coupling.tsv"), delimiter="\t"))
    bin_rows = list(csv.DictReader(open(RESULTS / "length_zvf_bincond.tsv"), delimiter="\t"))

    # Group by (task, algo)
    grp = {}
    for r in rows:
        key = (r["task"], r["algo"])
        grp.setdefault(key, []).append(r)

    print("=" * 70)
    print("T1: per-(task, algo) ρ(len, ZVF) — mean ± 95% bootstrap CI")
    print("=" * 70)
    summary = {}
    for key in sorted(grp.keys()):
        rhos = [float(r["rho_len_zvf"]) for r in grp[key]]
        m, lo, hi = bootstrap_ci(rhos, seed=42)
        summary[key] = {"mean": m, "lo": lo, "hi": hi, "n": len(rhos)}
        sig = "**" if (lo > 0 or hi < 0) else " "
        print(f"  {key}: n={len(rhos)} mean={m:+.3f} CI=[{lo:+.3f},{hi:+.3f}] {sig}")

    print()
    print("=" * 70)
    print("T2: paired Δ(Dr.GRPO − GRPO) on ρ(len, ZVF) with paired bootstrap CI")
    print("=" * 70)
    tasks = sorted({r["task"] for r in rows})
    deltas_t2 = {}
    for task in tasks:
        grpo_seeds = {r["seed"] for r in grp.get((task, "grpo"), [])}
        drgrpo_seeds = {r["seed"] for r in grp.get((task, "dr_grpo"), [])}
        common = sorted(grpo_seeds & drgrpo_seeds)
        if not common:
            print(f"  {task}: no common seeds")
            continue
        diffs = []
        for seed in common:
            grpo_v = next((r["rho_len_zvf"] for r in grp[(task, "grpo")] if r["seed"] == seed), None)
            drgrpo_v = next((r["rho_len_zvf"] for r in grp[(task, "dr_grpo")] if r["seed"] == seed), None)
            if grpo_v is None or drgrpo_v is None:
                continue
            diffs.append(float(drgrpo_v) - float(grpo_v))
        m, lo, hi = paired_bootstrap_ci(diffs, seed=123)
        deltas_t2[task] = {"mean": m, "lo": lo, "hi": hi, "n": len(diffs)}
        sig = "**" if (lo > 0 or hi < 0) else " "
        print(f"  {task}: n={len(diffs)} Δ={m:+.3f} CI=[{lo:+.3f},{hi:+.3f}] {sig}")

    print()
    print("=" * 70)
    print("T3: per-bin ZVF across (task, algo) — does ZVF grow with length bin?")
    print("=" * 70)
    bin_grp = {}
    for b in bin_rows:
        key = (b["task"], b["algo"], b["bin"])
        bin_grp.setdefault(key, []).append(float(b["mean_zvf"]))
    bin_means = {}
    for key in sorted(bin_grp.keys()):
        arr = np.array(bin_grp[key])
        m, lo, hi = bootstrap_ci(arr, seed=42)
        bin_means[key] = (m, lo, hi, len(arr))
        print(f"  {key}: n={len(arr)} mean={m:.3f} CI=[{lo:.3f},{hi:.3f}]")
    print()
    print("Paired Δ(ZVF at high-length bin L3 − ZVF at low-length bin L1):")
    tasks = sorted({b["task"] for b in bin_rows})
    for task in tasks:
        for algo in ("grpo", "dr_grpo"):
            l1 = bin_grp.get((task, algo, "L1"), [])
            l3 = bin_grp.get((task, algo, "L3"), [])
            common = min(len(l1), len(l3))
            if common == 0:
                continue
            diffs = [a - b for a, b in zip(l3[:common], l1[:common])]
            m, lo, hi = paired_bootstrap_ci(diffs, seed=7)
            sig = "**" if (lo > 0 or hi < 0) else " "
            print(f"  {(task, algo)}: n={common} Δ(ZVF_L3−ZVF_L1)={m:+.3f} CI=[{lo:+.3f},{hi:+.3f}] {sig}")

    # Save summary
    out = RESULTS / "length_zvf_bootstrap_summary.tsv"
    with open(out, "w") as f:
        f.write("test\tkey\tmetric\tn\tmean\tci_lo\tci_hi\tsig\n")
        for key, s in summary.items():
            sig = "yes" if (s["lo"] > 0 or s["hi"] < 0) else "no"
            f.write(f"T1\t{key}\trho_len_zvf\t{s['n']}\t{s['mean']:.4f}\t{s['lo']:.4f}\t{s['hi']:.4f}\t{sig}\n")
        for task, s in deltas_t2.items():
            sig = "yes" if (s["lo"] > 0 or s["hi"] < 0) else "no"
            f.write(f"T2\t{task}\tdelta_rho\t{s['n']}\t{s['mean']:.4f}\t{s['lo']:.4f}\t{s['hi']:.4f}\t{sig}\n")
        for key, (m, lo, hi, n) in bin_means.items():
            f.write(f"T3\t{key}\tzvf\t{n}\t{m:.4f}\t{lo:.4f}\t{hi:.4f}\t-\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()