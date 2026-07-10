#!/usr/bin/env python3
"""fill_t3_prediction.py — compute the T3 closed-form G* prediction and
compare it with the empirical optimum (gameplan queue item 3).

T3's objective (theory/zvf_theory.tex eq:efficiency):
    J(G) = (1/G) * E_{p~phi}[ (1 - h_G(p)) * p(1-p) ],  h_G(p)=p^G+(1-p)^G
with phi taken as the empirical p_hat distribution of the stratum. This is
the VARIANCE-WEIGHTED signal-per-rollout, distinct from the naive
P_mixed(G)/G reported by analyze_t3_gstar.py. This script computes, per
difficulty stratum and pooled:

  - analytic J(G) argmax over the grid  -> t3_gstar_prediction
  - empirical J(G) argmax (subsampled groups; realized informative indicator
    times p_hat(1-p_hat), per rollout)
  - the earlier naive P_mixed/G argmax, for contrast

Divergence between the J-argmax and the naive argmax localizes how much the
Bernoulli-variance weighting in S(p,G) matters; agreement between analytic
and empirical J-argmax is the E-T3a validation the theory needs.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import RESULTS_DIR, load_pool, utc_now, write_result

GRID = [2, 3, 4, 6, 8, 12, 16, 24, 32]


def h(p: float, G: int) -> float:
    return p ** G + (1 - p) ** G


def analytic_J(stratum: list[dict], G: int) -> float:
    tot = 0.0
    for pr in stratum:
        p = pr["p_hat"]
        tot += (1 - h(p, G)) * p * (1 - p)
    return tot / (len(stratum) * G)


def empirical_J(rng: random.Random, stratum: list[dict], G: int,
                draws: int = 4000) -> float:
    tot = 0.0
    for _ in range(draws):
        pr = rng.choice(stratum)
        group = rng.sample(pr["rewards"], G)
        informative = len(set(group)) > 1
        if informative:
            p = pr["p_hat"]
            tot += p * (1 - p)
    return tot / (draws * G)


def naive_argmax(rng: random.Random, stratum: list[dict], grid: list[int],
                 draws: int = 3000) -> int:
    best, best_v = grid[0], -1.0
    for G in grid:
        hits = sum(1 for _ in range(draws)
                   if len(set(rng.sample(rng.choice(stratum)["rewards"], G))) > 1)
        v = hits / (draws * G)
        if v > best_v:
            best, best_v = G, v
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--n-strata", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = load_pool(args.pool)
    prompts = pool["prompts"]
    R = pool["rollouts_per_prompt"]
    grid = [G for G in GRID if G <= R]
    rng = random.Random(args.seed)

    ordered = sorted(prompts, key=lambda p: p["p_hat"])
    k = max(1, len(ordered) // args.n_strata)
    strata = [ordered[i:i + k] for i in range(0, len(ordered), k)][:args.n_strata]
    strata.append(prompts)  # pooled, as the final "stratum"

    rows = []
    for si, stratum in enumerate(strata):
        label = "pooled" if si == len(strata) - 1 else f"stratum{si}"
        aJ = {G: analytic_J(stratum, G) for G in grid}
        eJ = {G: empirical_J(rng, stratum, G) for G in grid}
        row = {
            "stratum": label,
            "n_prompts": len(stratum),
            "mean_p_hat": round(sum(p["p_hat"] for p in stratum) / len(stratum), 4),
            "t3_gstar_prediction": max(aJ, key=aJ.get),
            "empirical_J_argmax": max(eJ, key=eJ.get),
            "naive_pmixed_argmax": naive_argmax(rng, stratum, grid),
            "analytic_J": {str(G): round(v, 6) for G, v in aJ.items()},
            "empirical_J": {str(G): round(v, 6) for G, v in eJ.items()},
        }
        rows.append(row)
        print(f"{label}: p_hat~{row['mean_p_hat']:.2f} "
              f"T3-predicted G*={row['t3_gstar_prediction']} "
              f"empirical-J G*={row['empirical_J_argmax']} "
              f"naive G*={row['naive_pmixed_argmax']}", flush=True)

    agree = sum(1 for r in rows
                if r["t3_gstar_prediction"] == r["empirical_J_argmax"])
    out = RESULTS_DIR / f"t3_gstar_v2_{pool['tag']}.json"
    write_result(out, {
        "kind": "t3_gstar_v2_prediction_fill",
        "status": "complete",
        "pool": str(args.pool),
        "pool_tag": pool["tag"],
        "model": pool["model"],
        "grid": grid,
        "seed": args.seed,
        "objective": "J(G) = (1/G) E_phi[(1-h_G(p)) p(1-p)] "
                     "(zvf_theory.tex eq:efficiency)",
        "rows": rows,
        "analytic_empirical_agreement": f"{agree}/{len(rows)}",
        "generated_at": utc_now(),
    })
    print(f"agreement (analytic vs empirical J): {agree}/{len(rows)} -> {out}")


if __name__ == "__main__":
    main()
