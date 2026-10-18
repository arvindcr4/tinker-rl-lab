#!/usr/bin/env python3
"""analyze_t3_gstar.py — E-T3a: signal-per-rollout vs group size G.

Descriptive objective: signal_per_rollout(G) = P(mixed)/G.
This differs from the variance-weighted J(G) proxy. Neither empirical argmax
establishes a unique learning-optimal group size. Compare only identical objectives.

Pure offline analysis of a build_pool.py pool. G values above the pool's
rollouts_per_prompt are skipped (subsampling is without replacement).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from analysis_contract import validate_design, strata_all

from common import RESULTS_DIR, load_pool, utc_now, write_result

DEFAULT_GRID = [2, 3, 4, 6, 8, 12, 16, 24, 32]


def empirical_p_mixed(rng: random.Random, stratum: list[dict], G: int,
                      draws: int = 4000) -> float:
    hits = 0
    for _ in range(draws):
        p = rng.choice(stratum)
        group = rng.sample(p["rewards"], G)
        hits += int(len(set(group)) > 1)
    return hits / draws


def analytic_p_mixed(stratum: list[dict], G: int) -> float:
    """Mean over the stratum of 1 - p^G - (1-p)^G at p = p_hat. This treats
    p_hat as exact; the finite-R plug-in bias is visible as the gap between
    this curve and the empirical (without-replacement) one."""
    total = 0.0
    for p in stratum:
        ph = p["p_hat"]
        total += 1.0 - ph ** G - (1.0 - ph) ** G
    return total / len(stratum)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--grid", type=int, nargs="+", default=DEFAULT_GRID)
    ap.add_argument("--n-strata", type=int, default=4)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = load_pool(args.pool)
    prompts = pool["prompts"]
    R = pool["rollouts_per_prompt"]
    validate_design(groups=args.grid, counts=[args.n_strata,args.draws], available=R, n_prompts=len(prompts))
    rng = random.Random(args.seed)
    grid = [G for G in args.grid if G <= R]

    ordered = sorted(prompts, key=lambda p: p["p_hat"])
    strata = strata_all(ordered, args.n_strata)

    rows = []
    for si, stratum in enumerate(strata):
        curve = []
        for G in grid:
            emp = empirical_p_mixed(rng, stratum, G, args.draws)
            ana = analytic_p_mixed(stratum, G)
            curve.append({
                "G": G,
                "p_mixed_empirical": round(emp, 4),
                "p_mixed_analytic": round(ana, 4),
                "signal_per_rollout_empirical": round(emp / G, 5),
                "signal_per_rollout_analytic": round(ana / G, 5),
            })
        best_emp = max(curve, key=lambda c: c["signal_per_rollout_empirical"])
        best_ana = max(curve, key=lambda c: c["signal_per_rollout_analytic"])
        rows.append({
            "stratum": si,
            "n_prompts": len(stratum),
            "mean_p_hat": round(sum(p["p_hat"] for p in stratum) / len(stratum), 4),
            "curve": curve,
            "empirical_argmax_G": best_emp["G"],
            "analytic_argmax_G": best_ana["G"],
            "t3_gstar_prediction": None,  # fill from zvf_theory.tex closed form
        })
        print(f"stratum {si}: p_hat~{rows[-1]['mean_p_hat']:.2f} "
              f"empirical G*={best_emp['G']} analytic G*={best_ana['G']}",
              flush=True)

    out = RESULTS_DIR / f"t3_gstar_{pool['tag']}.json"
    write_result(out, {
        "claim_scope": "proxy_objective_only",
        "unique_learning_optimum_established": False,
        "objective_id": "unweighted_mixed_per_rollout_v1",
        "kind": "t3_gstar_curve",
        "status": "complete",
        "pool": str(args.pool),
        "pool_tag": pool["tag"],
        "model": pool["model"],
        "grid": grid,
        "draws_per_cell": args.draws,
        "seed": args.seed,
        "generated_at": utc_now(),
        "rows": rows,
        "next_step": "Independent matched-compute learning outcomes required; proxy argmax is descriptive",
    })
    print(f"-> {out}")


if __name__ == "__main__":
    main()
