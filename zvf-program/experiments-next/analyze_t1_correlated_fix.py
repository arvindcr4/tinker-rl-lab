#!/usr/bin/env python3
"""analyze_t1_correlated_fix.py — diagnose and fix the correlated-batch
coverage collapse found by analyze_t1_ci.py.

The E-T1 result: under difficulty-sorted (curriculum-style) batching, the
binomial CI's coverage of the GLOBAL population ZVF collapses to 0.08-0.40
with *narrower* intervals. Two candidate explanations with different fixes:

  (H-bias)  The batch ZVF is a nearly-unbiased, well-calibrated estimate of
            the LOCAL (window) ZVF, which simply differs from the global
            one. Fix = redefine the estimand: under curriculum ordering the
            CI is valid for the curriculum-stage ZVF (which is what a
            stage-local controller acts on anyway); global inference needs
            stratified batches.
  (H-var)   Within-window correlation inflates variance beyond binomial.
            Fix = clustered/robust variance.

This script adjudicates by measuring, on a real pool:
  1. correlated batches vs LOCAL truth  -> if coverage ~0.95, H-bias wins
  2. correlated batches vs GLOBAL truth -> the known failure (reproduced)
  3. STRATIFIED batches (one prompt per difficulty stratum slot) vs GLOBAL
     truth -> tests the proposed design fix for global inference

Pure offline analysis of a build_pool.py pool. Output feeds the T1 section
rewrite in theory/zvf_theory.tex.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from common import RESULTS_DIR, load_pool, utc_now, write_result

Z95 = 1.959963984540054


def zv_indicator(rng: random.Random, rewards: list[float], G: int) -> int:
    return 1 if len(set(rng.sample(rewards, G))) == 1 else 0


def wilson_ci(zvf: float, m: int) -> tuple[float, float]:
    z2 = Z95 * Z95
    denom = 1.0 + z2 / m
    center = (zvf + z2 / (2 * m)) / denom
    half = (Z95 / denom) * math.sqrt(zvf * (1 - zvf) / m + z2 / (4 * m * m))
    return center - half, center + half


def prompt_zv_prob(rng: random.Random, rewards: list[float], G: int,
                   draws: int = 48) -> float:
    return sum(zv_indicator(rng, rewards, G) for _ in range(draws)) / draws


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--batches", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = load_pool(args.pool)
    prompts = pool["prompts"]
    G, M, B = args.group_size, args.batch_size, args.batches
    rng = random.Random(args.seed)

    # per-prompt zero-variance probability at G (smooth local truth)
    pz = [prompt_zv_prob(rng, p["rewards"], G) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: prompts[i]["p_hat"])
    global_truth = sum(pz) / len(pz)

    # --- 1+2: correlated (contiguous sorted) batches ---
    cov_local = cov_global = 0
    for _ in range(B):
        start = rng.randrange(0, len(order) - M + 1)
        idx = order[start:start + M]
        zvf = sum(zv_indicator(rng, prompts[i]["rewards"], G)
                  for i in idx) / M
        lo, hi = wilson_ci(zvf, M)
        local_truth = sum(pz[i] for i in idx) / M
        cov_local += int(lo <= local_truth <= hi)
        cov_global += int(lo <= global_truth <= hi)

    # --- 3: stratified batches (one draw per difficulty slot) ---
    k = max(1, len(order) // M)
    strata = [order[i * k:(i + 1) * k] for i in range(M)]
    strata = [s for s in strata if s]
    cov_strat = 0
    for _ in range(B):
        idx = [rng.choice(s) for s in strata]
        zvf = sum(zv_indicator(rng, prompts[i]["rewards"], G)
                  for i in idx) / len(idx)
        lo, hi = wilson_ci(zvf, len(idx))
        cov_strat += int(lo <= global_truth <= hi)

    result = {
        "kind": "t1_correlated_fix",
        "status": "complete",
        "pool": str(args.pool),
        "pool_tag": pool["tag"],
        "model": pool["model"],
        "G": G, "M": M, "B": B, "seed": args.seed,
        "global_truth_zvf": round(global_truth, 4),
        "coverage_correlated_vs_local_truth": round(cov_local / B, 4),
        "coverage_correlated_vs_global_truth": round(cov_global / B, 4),
        "coverage_stratified_vs_global_truth": round(cov_strat / B, 4),
        "interpretation": (
            "H-bias confirmed if local~0.95 while global collapses; "
            "stratified~0.95 validates the design fix for global inference"
        ),
        "generated_at": utc_now(),
    }
    out = RESULTS_DIR / f"t1_correlated_fix_{pool['tag']}_G{G}_M{M}.json"
    write_result(out, result)
    print(f"global truth ZVF(G={G}) = {global_truth:.3f}")
    print(f"correlated vs LOCAL truth:  {cov_local / B:.3f}")
    print(f"correlated vs GLOBAL truth: {cov_global / B:.3f}")
    print(f"stratified vs GLOBAL truth: {cov_strat / B:.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
