#!/usr/bin/env python3
"""sim_bandit_reallocation.py — E-B pilot (gameplan §1.4), offline edition.

Question: treat the per-group choice of difficulty stratum as a bandit whose
reward is the realized learning signal — does adaptive allocation recover
the oracle static policy and beat uniform / naive-hardest allocation?

Offline simulation against a real build_pool.py pool (no training, no API):
each "arm" is a difficulty stratum; pulling an arm draws a real prompt from
that stratum and subsamples a size-G group from its recorded rollouts.

Policies:
  uniform         random stratum per group
  static-oracle   always the stratum with highest true J(G) (upper bound
                  for any static policy; needs oracle knowledge)
  static-hardest  always the hardest stratum (DARS caricature: difficulty
                  != signal — expected to underperform)
  thompson-gu     Beta-Bernoulli Thompson sampling, reward = group is
                  informative (non-zero-variance)

Metrics per policy (mean over sims): informative groups per 1k rollouts
(GU/rollout) and realized J-signal per 1k rollouts.

Success bar (experiment design): thompson >= static policies on GU/rollout
and within noise of static-oracle. This is the go/no-go for building the
training-time bandit controller pillar.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import RESULTS_DIR, load_pool, utc_now, write_result


def group_draw(rng: random.Random, stratum: list[dict], G: int):
    pr = rng.choice(stratum)
    group = rng.sample(pr["rewards"], G)
    informative = len(set(group)) > 1
    p = pr["p_hat"]
    jsig = p * (1 - p) if informative else 0.0
    return informative, jsig


def true_J(stratum: list[dict], G: int) -> float:
    tot = 0.0
    for pr in stratum:
        p = pr["p_hat"]
        h = p ** G + (1 - p) ** G
        tot += (1 - h) * p * (1 - p)
    return tot / len(stratum)


def simulate(rng, strata, G, groups_budget, policy, oracle_idx, hardest_idx):
    n = len(strata)
    a = [1.0] * n  # Beta successes
    b = [1.0] * n  # Beta failures
    informative_total = 0
    jsig_total = 0.0
    for _ in range(groups_budget):
        if policy == "uniform":
            arm = rng.randrange(n)
        elif policy == "static-oracle":
            arm = oracle_idx
        elif policy == "static-hardest":
            arm = hardest_idx
        else:  # thompson-gu
            samples = [rng.betavariate(a[i], b[i]) for i in range(n)]
            arm = max(range(n), key=lambda i: samples[i])
        informative, jsig = group_draw(rng, strata[arm], G)
        informative_total += int(informative)
        jsig_total += jsig
        if policy == "thompson-gu":
            if informative:
                a[arm] += 1
            else:
                b[arm] += 1
    return informative_total, jsig_total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--n-strata", type=int, default=6)
    ap.add_argument("--groups-budget", type=int, default=250,
                    help="groups per simulation (x G = rollout budget)")
    ap.add_argument("--sims", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = load_pool(args.pool)
    prompts = pool["prompts"]
    G = args.group_size
    rng = random.Random(args.seed)

    ordered = sorted(prompts, key=lambda p: p["p_hat"])
    k = max(1, len(ordered) // args.n_strata)
    strata = [ordered[i:i + k] for i in range(0, len(ordered), k)][:args.n_strata]
    Js = [true_J(s, G) for s in strata]
    oracle_idx = max(range(len(strata)), key=lambda i: Js[i])
    hardest_idx = 0  # lowest p_hat stratum
    rollouts = args.groups_budget * G

    results = {}
    for policy in ("uniform", "static-oracle", "static-hardest", "thompson-gu"):
        inf_tot = 0
        j_tot = 0.0
        for s in range(args.sims):
            srng = random.Random(args.seed * 1_000_003 + s)
            i, j = simulate(srng, strata, G, args.groups_budget, policy,
                            oracle_idx, hardest_idx)
            inf_tot += i
            j_tot += j
        results[policy] = {
            "informative_groups_per_1k_rollouts":
                round(1000 * inf_tot / (args.sims * rollouts), 3),
            "J_signal_per_1k_rollouts":
                round(1000 * j_tot / (args.sims * rollouts), 4),
        }
        r = results[policy]
        print(f"{policy:>15}: GU/1k={r['informative_groups_per_1k_rollouts']:7.3f} "
              f"J/1k={r['J_signal_per_1k_rollouts']:7.4f}", flush=True)

    th = results["thompson-gu"]["informative_groups_per_1k_rollouts"]
    orc = results["static-oracle"]["informative_groups_per_1k_rollouts"]
    uni = results["uniform"]["informative_groups_per_1k_rollouts"]
    verdict = ("GO: thompson within 5% of oracle and beats uniform"
               if th >= 0.95 * orc and th > uni else
               "NO-GO or investigate: thompson did not meet the bar")
    print(verdict)

    out = RESULTS_DIR / f"bandit_realloc_{pool['tag']}_G{G}.json"
    write_result(out, {
        "kind": "bandit_reallocation_sim",
        "status": "complete",
        "pool": str(args.pool),
        "pool_tag": pool["tag"],
        "model": pool["model"],
        "G": G, "n_strata": args.n_strata,
        "groups_budget": args.groups_budget, "sims": args.sims,
        "seed": args.seed,
        "stratum_true_J": [round(j, 5) for j in Js],
        "oracle_stratum": oracle_idx,
        "results": results,
        "verdict": verdict,
        "caveat": "offline simulation on a frozen pool; a training-time "
                  "pilot must confirm under distribution shift (p_hat "
                  "drifts as the policy learns)",
        "generated_at": utc_now(),
    })
    print(f"-> {out}")


if __name__ == "__main__":
    main()
