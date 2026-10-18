#!/usr/bin/env python3
"""Exploratory finite-pool coverage under ordered and stratified sampling.

Local-window and global prompt-population estimands are distinct. Binary
finite-pool probabilities use exact hypergeometric terms. One independent draw
per stratum uses population weights and a conservative weighted Hoeffding
interval, not an ordinary Wilson interval or an unidentified sample variance.
Monte Carlo coverage uncertainty is reported separately. This simulation does
not validate general curriculum inference or establish controller readiness.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from common import RESULTS_DIR, load_pool, utc_now, write_result

from analyze_t1_ci import population_zvf
from analysis_contract import validate_design, strata_all, stratified_interval, T1_DECISION

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
    return population_zvf([{"rewards": rewards}], G, rng)


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
    validate_design(groups=[G], counts=[M,B], available=pool["rollouts_per_prompt"], n_prompts=len(prompts))
    if M > len(prompts):
        raise ValueError("batch size exceeds prompt population")
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
    strata = strata_all(order, M)
    weights = [len(st)/len(order) for st in strata]
    cov_strat = 0
    for _ in range(B):
        idx = [rng.choice(s) for s in strata]
        values = [zv_indicator(rng, prompts[i]["rewards"], G) for i in idx]
        lo, hi = stratified_interval(values, weights)
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
        "decision_contract": T1_DECISION,
        "truth_estimator": "exact hypergeometric conditional on finite binary pool",
        "stratified_interval": "weighted Hoeffding bound; independent one-draw strata",
        "coverage_mc_ci95": {"local": wilson_ci(cov_local/B,B),
                             "global": wilson_ci(cov_global/B,B),
                             "stratified": wilson_ci(cov_strat/B,B)},
        "interpretation": "Exploratory finite-pool coverage; no general controller validation",
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
