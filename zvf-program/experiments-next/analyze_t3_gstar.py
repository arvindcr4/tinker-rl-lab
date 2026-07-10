#!/usr/bin/env python3
"""analyze_t3_gstar.py — E-T3a: signal-per-rollout vs group size G.

T3 (theory/zvf_theory.tex) derives a closed-form CANDIDATE optimal group size
G* maximizing expected learning signal per rollout. This script produces the
empirical curve it must match:

  For each difficulty stratum (by p_hat) and each G in the grid:
    P_mixed(G)      = fraction of size-G groups that are NOT zero-variance
                      (signal per group)
    signal_per_rollout(G) = P_mixed(G) / G

  Under Bernoulli(p) rewards the analytic curve is
    P_mixed(G) = 1 - p^G - (1-p)^G
  and the script overlays that prediction (computed from each stratum's p_hat
  distribution) against the empirical subsampled curve. The empirical argmax
  of signal_per_rollout is reported per stratum; T3's G* should land on or
  near it. Divergence localizes exactly which modeling assumption in S(p,G)
  fails.

NOTE: signal_per_rollout = P_mixed/G is the simplest per-rollout objective;
if T3's S(p,G) differs, add it in analytic_curves() and compare both. This
script is evidence, not proof (T3 uniqueness is an open gap).

Pure offline analysis of a build_pool.py pool. G values above the pool's
rollouts_per_prompt are skipped (subsampling is without replacement).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

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
    rng = random.Random(args.seed)
    grid = [G for G in args.grid if G <= R]

    ordered = sorted(prompts, key=lambda p: p["p_hat"])
    k = max(1, len(ordered) // args.n_strata)
    strata = [ordered[i:i + k] for i in range(0, len(ordered), k)][:args.n_strata]

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
        "next_step": "fill t3_gstar_prediction from the closed-form in "
                     "theory/zvf_theory.tex sec:T3 and compare to "
                     "empirical_argmax_G per stratum; then run E-T3b "
                     "(short training at G* +/- delta, matched rollouts)",
    })
    print(f"-> {out}")


if __name__ == "__main__":
    main()
