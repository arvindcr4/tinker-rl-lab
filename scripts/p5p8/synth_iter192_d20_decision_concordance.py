#!/usr/bin/env python3
"""P5P8-SYNTH D20 (iter 192) — Cross-Pillar Decision-Concordance Audit.

Fresh vein: 20th density domain (D20), NOT in any prior D1..D19 SYNTH row.

D19 measured information-weighted controller efficiency per (method, step).
D20 lifts the lens to cross-pillar decision-concordance: across P5, P6, P7,
P8, does the same method rank highest on the same headline metric?

For each of P5, P6, P7, P8, define a "headline metric" and rank the 4 N2
methods (grpo, aero, areal, gift) on that metric. Cross-pillar concordance
= Spearman correlation of method-ranks across the 4 pillars.

Pipeline:
  1. Load N2 four-method panel (160 rows = 4 methods x 40 steps).
  2. For each method, compute 4 pillar-specific headline metrics:
     - P5: mean reward across 40 steps (the headline RL outcome).
     - P6: mean zvf risk reduction (lower zvf is better for registry).
     - P7: controller-efficiency proxy (mean reward per std of reward;
       higher = more efficient use of rollouts).
     - P8: held-out transfer score (max reward / max-completion-len,
       normalized; higher = more transferable).
  3. Rank methods per pillar.
  4. Cross-pillar concordance: 6 Spearman rho pairs (P5-P6, P5-P7, P5-P8,
     P6-P7, P6-P8, P7-P8) + mean pairwise Spearman.
  5. Bootstrap CIs (B=2000, paired-step bootstrap of the 40 steps) on
     per-pillar headline gaps (best - worst method).
  6. 5 falsifiable hypotheses.

Outputs (experiments/results/p5p8/):
  synth_iter192_d20_per_method.tsv       4 methods x 4 pillars = 16 rows
  synth_iter192_d20_method_ranks.tsv     4 methods x 4 pillars = 16 rows
  synth_iter192_d20_concordance.tsv      6 pairs + mean = 7 rows
  synth_iter192_d20_per_method_step.tsv  4 methods x 40 steps x 4 pillars
  synth_iter192_d20_summary.json
"""
from __future__ import annotations
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)
N2 = ROOT / "experiments" / "results" / "n2_reward_tensor_resume" / "n2_metrics.tsv"
METHODS = ["grpo", "aero", "areal", "gift"]
PILLARS = ["P5", "P6", "P7", "P8"]
N_BOOT = 2000


def load_n2():
    by_m = {m: [] for m in METHODS}
    with N2.open() as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            m = r["method"]
            if m in by_m:
                by_m[m].append({
                    "step": int(r["step"]),
                    "reward_mean": float(r["reward_mean"]),
                    "zvf": float(r["zvf"]),
                    "mean_len": float(r["mean_len"]),
                    "loss": float(r["loss"]),
                })
    for m in METHODS:
        by_m[m].sort(key=lambda x: x["step"])
    return {m: np.array([[r["reward_mean"], r["zvf"], r["mean_len"], r["loss"]]
                          for r in rows]) for m, rows in by_m.items()}


def pillar_metric(data, m, pillar):
    """Per-pillar headline metric (higher = better, except P6 where lower is better).

    data[m] is (40, 4) array: [reward_mean, zvf, mean_len, loss].
    """
    arr = data[m]  # (40, 4)
    rew = arr[:, 0]
    zvf = arr[:, 1]
    ln = arr[:, 2]
    if pillar == "P5":
        # Mean reward across 40 steps (RL outcome).
        return float(rew.mean())
    elif pillar == "P6":
        # Negative mean ZVF (lower zvf is better for registry).
        return float(-zvf.mean())
    elif pillar == "P7":
        # Controller efficiency proxy: mean reward / std of reward.
        # Higher = more efficient use of rollouts (stable controller).
        return float(rew.mean() / max(rew.std(ddof=1), 1e-9))
    elif pillar == "P8":
        # Transfer score: mean reward / mean length (compact, useful response).
        # Higher = more transferable to deployment.
        return float((rew / np.maximum(ln, 1.0)).mean())
    raise ValueError(pillar)


def paired_bootstrap_ci(per_method_per_step, n_boot, seed):
    """Per-method, per-step metric arrays; compute CI of (best - worst).

    per_method_per_step: dict method -> (40,) array of per-step metric values.
    """
    rng = np.random.default_rng(seed)
    n = len(next(iter(per_method_per_step.values())))
    diffs = np.empty(n_boot)
    means = {m: per_method_per_step[m].mean() for m in METHODS}
    base_diff = max(means.values()) - min(means.values())
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = {m: per_method_per_step[m][idx].mean() for m in METHODS}
        diffs[i] = max(s.values()) - min(s.values())
    return {
        "mean": float(diffs.mean()),
        "lo": float(np.quantile(diffs, 0.025)),
        "hi": float(np.quantile(diffs, 0.975)),
        "point_diff": float(base_diff),
    }


def spearman(r1, r2):
    """Spearman rank correlation. ranks: 1-indexed."""
    return float(np.corrcoef(r1, r2)[0, 1])


def main():
    print("Loading N2 data...", flush=True)
    data = load_n2()
    for m in METHODS:
        print(f"  {m}: {data[m].shape}", flush=True)

    # Per-(method, pillar) headline metric
    print("Computing per-(method, pillar) metrics...", flush=True)
    headline = {}  # (method, pillar) -> value
    per_method_pillar_per_step = {p: {m: np.empty(40) for m in METHODS}
                                   for p in PILLARS}
    for m in METHODS:
        arr = data[m]
        for p in PILLARS:
            v = pillar_metric(data, m, p)
            headline[(m, p)] = v
            # Per-step contributions for bootstrap
            rew = arr[:, 0]
            zvf = arr[:, 1]
            ln = arr[:, 2]
            if p == "P5":
                per_method_pillar_per_step[p][m] = rew
            elif p == "P6":
                per_method_pillar_per_step[p][m] = -zvf
            elif p == "P7":
                std_r = rew.std(ddof=1)
                per_method_pillar_per_step[p][m] = rew / max(std_r, 1e-9)
            elif p == "P8":
                per_method_pillar_per_step[p][m] = rew / np.maximum(ln, 1.0)

    # Per-method per-pillar TSV
    out_pm = RES / "synth_iter192_d20_per_method.tsv"
    with out_pm.open("w") as f:
        f.write("method\tpillar\theadline_metric\n")
        for m in METHODS:
            for p in PILLARS:
                f.write(f"{m}\t{p}\t{headline[(m, p)]:.6f}\n")

    # Per-method rank per pillar (rank 1 = best)
    ranks = {}  # (m, p) -> rank
    out_rk = RES / "synth_iter192_d20_method_ranks.tsv"
    with out_rk.open("w") as f:
        f.write("method\tpillar\trank\theadline_metric\n")
        for p in PILLARS:
            # Higher = better; rank 1 = highest value
            sorted_methods = sorted(METHODS, key=lambda m: -headline[(m, p)])
            for r, m in enumerate(sorted_methods, 1):
                ranks[(m, p)] = r
                f.write(f"{m}\t{p}\t{r}\t{headline[(m, p)]:.6f}\n")

    # Cross-pillar Spearman concordance
    print("Computing cross-pillar concordance...", flush=True)
    out_c = RES / "synth_iter192_d20_concordance.tsv"
    pairs = list(combinations(PILLARS, 2))
    pair_spearman = {}
    with out_c.open("w") as f:
        f.write("pillar_a\tpillar_b\tspearman\n")
        for pa, pb in pairs:
            ra = [ranks[(m, pa)] for m in METHODS]
            rb = [ranks[(m, pb)] for m in METHODS]
            sp = spearman(ra, rb)
            pair_spearman[(pa, pb)] = sp
            f.write(f"{pa}\t{pb}\t{sp:.6f}\n")
        mean_sp = float(np.mean(list(pair_spearman.values())))
        f.write(f"MEAN\t-\t{mean_sp:.6f}\n")

    # Per-pillar best-worst gap CIs
    print("Computing per-pillar gap CIs...", flush=True)
    per_pillar_gaps = {}
    for p in PILLARS:
        per_m_step = {m: per_method_pillar_per_step[p][m] for m in METHODS}
        ci = paired_bootstrap_ci(per_m_step, N_BOOT, seed=20260706 + hash(p) % 1000)
        per_pillar_gaps[p] = ci

    # Per-method per-step per-pillar TSV
    out_step = RES / "synth_iter192_d20_per_method_step.tsv"
    with out_step.open("w") as f:
        f.write("method\tpillar\tstep\tmetric\n")
        for p in PILLARS:
            for m in METHODS:
                for s in range(40):
                    f.write(f"{m}\t{p}\t{s}\t"
                            f"{per_method_pillar_per_step[p][m][s]:.6f}\n")

    # ----- hypotheses -----
    print("Evaluating hypotheses...", flush=True)

    # H1: At least one cross-pillar Spearman rho > 0.5 (positive concordance)
    h1 = max(pair_spearman.values()) > 0.5

    # H2: The 6 cross-pillar Spearman rho values have mean > 0 (concordance
    #     direction: when one pillar ranks a method high, others tend to
    #     too — at least on average).
    h2 = mean_sp > 0.0

    # H3: The best-method-on-P5 is also the best-method-on-P6 (P5-P6
    #     specific concordance — the registry headline should match the
    #     RL outcome headline).
    p5_winner = min(METHODS, key=lambda m: ranks[(m, "P5")])
    p6_winner = min(METHODS, key=lambda m: ranks[(m, "P6")])
    h3 = p5_winner == p6_winner

    # H4: Per-pillar best-worst gap CI excludes zero in 4/4 pillars
    #     (each pillar has a discriminating signal).
    h4 = sum(1 for p in PILLARS if per_pillar_gaps[p]["lo"] > 0.0) >= 4

    # H5: P5 (mean reward) and P8 (transfer score) have concordance
    #     rho > 0.0 (the RL outcome aligns with the deployment transfer
    #     score — the most operationally important cross-pillar check).
    h5 = pair_spearman[("P5", "P8")] > 0.0

    summary = {
        "n_methods": 4,
        "n_pillars": 4,
        "n_steps_per_method": 40,
        "headline_per_method_per_pillar": {
            f"{m}|{p}": headline[(m, p)] for m in METHODS for p in PILLARS
        },
        "ranks_per_method_per_pillar": {
            f"{m}|{p}": ranks[(m, p)] for m in METHODS for p in PILLARS
        },
        "spearman_per_pair": {f"{a}-{b}": pair_spearman[(a, b)]
                               for a, b in pairs},
        "mean_spearman": mean_sp,
        "per_pillar_gap_ci": per_pillar_gaps,
        "h1_max_spearman_gt_0_5": bool(h1),
        "h1_max_spearman_value": float(max(pair_spearman.values())),
        "h2_mean_spearman_gt_0": bool(h2),
        "h2_mean_spearman_value": float(mean_sp),
        "h3_p5_p6_concordance": bool(h3),
        "h3_p5_winner": p5_winner,
        "h3_p6_winner": p6_winner,
        "h4_all_pillars_discriminating": bool(h4),
        "h4_pillars_with_ci_excludes_zero": int(
            sum(1 for p in PILLARS if per_pillar_gaps[p]["lo"] > 0.0)),
        "h5_p5_p8_concordance_gt_0": bool(h5),
        "h5_p5_p8_spearman": float(pair_spearman[("P5", "P8")]),
    }

    out_sum = RES / "synth_iter192_d20_summary.json"
    with out_sum.open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()