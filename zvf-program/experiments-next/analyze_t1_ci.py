#!/usr/bin/env python3
"""analyze_t1_ci.py — E-T1: empirical coverage of the T1 confidence interval.

T1 (theory/zvf_theory.tex) states ZVF_t is a sample mean of i.i.d. group
indicators with a closed-form binomial-proportion 95% CI. This script tests
that claim offline against a pool built by build_pool.py:

  1. Ground truth: for each group size G, subsample groups of size G from each
     prompt's R rollouts; population ZVF(G) = mean over ALL prompts of
     P(group is zero-variance), estimated from the full pool.
  2. Coverage: draw B batches of M prompt-groups, compute batch ZVF + CI
     (Wald and Wilson), measure the fraction of batches whose CI covers truth.
  3. Stress: repeat with CORRELATED batches (prompts sorted by p_hat, batch =
     contiguous block) to probe the across-group-i.i.d. assumption the theory
     paper flags as its biggest threat to validity.

Decision rule (pre-registered in sweep/README.md amendments): Wald coverage in
[0.93, 0.97] at M >= 32 under i.i.d. batching validates T1 for controller use;
systematic under-coverage under correlated batching means the theory needs a
clustered-variance correction before the controller ships.

Pure offline analysis — never contacts an API.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from common import RESULTS_DIR, load_pool, utc_now, write_result

Z95 = 1.959963984540054


def group_is_zero_variance(rewards: list[float]) -> bool:
    return len(set(rewards)) == 1


def subsample_indicator(rng: random.Random, rewards: list[float], G: int) -> int:
    """1 if a random size-G group drawn (without replacement) from this
    prompt's rollouts is zero-variance."""
    group = rng.sample(rewards, G)
    return 1 if group_is_zero_variance(group) else 0


def wald_ci(zvf: float, m: int) -> tuple[float, float]:
    half = Z95 * math.sqrt(max(zvf * (1.0 - zvf), 0.0) / m)
    return (zvf - half, zvf + half)


def wilson_ci(zvf: float, m: int) -> tuple[float, float]:
    z2 = Z95 * Z95
    denom = 1.0 + z2 / m
    center = (zvf + z2 / (2 * m)) / denom
    half = (Z95 / denom) * math.sqrt(zvf * (1.0 - zvf) / m + z2 / (4 * m * m))
    return (center - half, center + half)


def population_zvf(prompts: list[dict], G: int, rng: random.Random,
                   draws_per_prompt: int = 64) -> float:
    """Population ZVF(G). For binary rewards this is computed EXACTLY via
    hypergeometric terms (drawing G of the m recorded rewards without
    replacement): P(uniform) = [C(k,G) + C(m-k,G)] / C(m,G) with k ones.
    The earlier 64-draw Monte-Carlo estimate was treated as exact truth in
    the coverage analysis (fixed 2026-07-11). Non-binary rewards fall back
    to resampling."""
    from math import comb
    total = 0.0
    n = 0
    for p in prompts:
        rewards = p["rewards"]
        m = len(rewards)
        if m < G:
            continue
        if set(rewards) <= {0, 0.0, 1, 1.0}:
            k = sum(1 for r in rewards if r in (1, 1.0))
            total += (comb(k, G) + comb(m - k, G)) / comb(m, G)
        else:
            s = 0
            for _ in range(draws_per_prompt):
                s += subsample_indicator(rng, rewards, G)
            total += s / draws_per_prompt
        n += 1
    return total / n


def run_coverage(prompts: list[dict], G: int, M: int, B: int, truth: float,
                 rng: random.Random, correlated: bool) -> dict:
    ordered = sorted(prompts, key=lambda p: p["p_hat"]) if correlated else None
    wald_hits = wilson_hits = 0
    widths: list[float] = []
    for _ in range(B):
        if correlated:
            start = rng.randrange(0, len(ordered) - M + 1)
            batch = ordered[start:start + M]
        else:
            batch = rng.sample(prompts, M)
        zvf = sum(subsample_indicator(rng, p["rewards"], G) for p in batch) / M
        lo, hi = wald_ci(zvf, M)
        wald_hits += int(lo <= truth <= hi)
        lo_w, hi_w = wilson_ci(zvf, M)
        wilson_hits += int(lo_w <= truth <= hi_w)
        widths.append(hi - lo)
    return {
        "G": G, "M": M, "B": B, "correlated": correlated,
        "population_zvf": round(truth, 4),
        "wald_coverage": round(wald_hits / B, 4),
        "wilson_coverage": round(wilson_hits / B, 4),
        "mean_wald_width": round(sum(widths) / len(widths), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--group-sizes", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--batch-sizes", type=int, nargs="+",
                    default=[8, 16, 32, 64, 128])
    ap.add_argument("--batches", type=int, default=2000,
                    help="B resampled batches per cell")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = load_pool(args.pool)
    prompts = pool["prompts"]
    R = pool["rollouts_per_prompt"]
    rng = random.Random(args.seed)

    rows = []
    for G in args.group_sizes:
        if G > R:
            print(f"skip G={G} > rollouts_per_prompt={R}")
            continue
        truth = population_zvf(prompts, G, rng)
        for M in args.batch_sizes:
            if M > len(prompts):
                continue
            for correlated in (False, True):
                row = run_coverage(prompts, G, M, args.batches, truth, rng,
                                   correlated)
                rows.append(row)
                mode = "corr" if correlated else "iid "
                print(f"G={G:>2} M={M:>3} [{mode}] truth={truth:.3f} "
                      f"wald={row['wald_coverage']:.3f} "
                      f"wilson={row['wilson_coverage']:.3f} "
                      f"width={row['mean_wald_width']:.3f}", flush=True)

    out = RESULTS_DIR / f"t1_ci_coverage_{pool['tag']}.json"
    write_result(out, {
        "kind": "t1_ci_coverage",
        "status": "complete",
        "pool": str(args.pool),
        "pool_tag": pool["tag"],
        "model": pool["model"],
        "batches_per_cell": args.batches,
        "seed": args.seed,
        "generated_at": utc_now(),
        "rows": rows,
        "decision_rule": "T1 usable if Wald coverage in [0.93,0.97] at M>=32 (iid); "
                         "correlated under-coverage => clustered-variance correction needed",
    })
    print(f"-> {out}")


if __name__ == "__main__":
    main()
