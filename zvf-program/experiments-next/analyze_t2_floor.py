#!/usr/bin/env python3
"""analyze_t2_floor.py — E-T2: empirical test of the wasted-compute floor.

T2 (theory/zvf_theory.tex) lower-bounds the number of additional rollouts
required before the next NON-DEGENERATE advantage event (a mixed-reward
group) when the observed ZVF is high. The bound instantiated here:

    P(no mixed group in n consecutive groups) = ZVF^n <= delta
    =>  n(delta, ZVF) = ceil( ln(delta) / ln(ZVF) )   groups
    =>  rollout floor = G * n(delta, ZVF)

NOTE: this is the geometric instantiation of T2; cross-check the constant
against the theorem statement in zvf_theory.tex before citing (the theorem
carries proof-gap markers — this script is evidence, not proof).

Method (pure offline, resampled from a build_pool.py pool):
  1. Stratify prompts by p_hat into strata whose group-level ZVF(G) spans
     ~[0.5, 0.99].
  2. Within each stratum, repeatedly draw groups (size G, prompts sampled with
     replacement, rollouts subsampled without replacement) until the first
     mixed group; record rollouts consumed.
  3. Compare the empirical distribution of rollouts-to-first-mixed against the
     floor at delta in {0.5, 0.1, 0.05}: the bound HOLDS if the empirical
     (1-delta)-quantile >= floor(delta); it is TIGHT if the ratio is near 1.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from common import RESULTS_DIR, load_pool, utc_now, write_result


def group_zvf_of_stratum(rng: random.Random, stratum: list[dict], G: int,
                         draws: int = 4000) -> float:
    hits = 0
    for _ in range(draws):
        p = rng.choice(stratum)
        group = rng.sample(p["rewards"], G)
        hits += int(len(set(group)) == 1)
    return hits / draws


def rollouts_until_mixed(rng: random.Random, stratum: list[dict], G: int,
                         cap_groups: int = 100000) -> int:
    n = 0
    while n < cap_groups:
        n += 1
        p = rng.choice(stratum)
        group = rng.sample(p["rewards"], G)
        if len(set(group)) > 1:
            return n * G
    return cap_groups * G  # censored


def floor_rollouts(zvf: float, delta: float, G: int) -> float:
    if zvf <= 0.0 or zvf >= 1.0:
        return float("nan")
    return G * math.ceil(math.log(delta) / math.log(zvf))


def quantile(xs: list[int], q: float) -> float:
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, int(q * len(ys))))
    return float(ys[idx])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--trials", type=int, default=2000,
                    help="sequential-draw trials per stratum")
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.5, 0.1, 0.05])
    ap.add_argument("--n-strata", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = load_pool(args.pool)
    prompts = pool["prompts"]
    G = args.group_size
    if G > pool["rollouts_per_prompt"]:
        raise SystemExit(f"G={G} exceeds pool rollouts_per_prompt")
    rng = random.Random(args.seed)

    # Stratify by p_hat quantiles; extreme p_hat strata have high ZVF(G).
    ordered = sorted(prompts, key=lambda p: p["p_hat"])
    k = max(1, len(ordered) // args.n_strata)
    strata = [ordered[i:i + k] for i in range(0, len(ordered), k)][:args.n_strata]

    rows = []
    for si, stratum in enumerate(strata):
        zvf = group_zvf_of_stratum(rng, stratum, G)
        if zvf >= 0.999 or zvf <= 0.001:
            note = "degenerate stratum (ZVF ~ 0 or 1); floor undefined/infinite"
        else:
            note = ""
        obs = [rollouts_until_mixed(rng, stratum, G) for _ in range(args.trials)]
        row = {
            "stratum": si,
            "n_prompts": len(stratum),
            "mean_p_hat": round(sum(p["p_hat"] for p in stratum) / len(stratum), 4),
            "zvf_G": round(zvf, 4),
            "G": G,
            "trials": args.trials,
            "observed_mean_rollouts": round(sum(obs) / len(obs), 1),
            "note": note,
            "bounds": [],
        }
        for delta in args.deltas:
            fl = floor_rollouts(zvf, delta, G)
            q = quantile(obs, 1.0 - delta)
            holds = (math.isnan(fl)) or (q >= fl)
            row["bounds"].append({
                "delta": delta,
                "floor_rollouts": None if math.isnan(fl) else fl,
                "observed_q_at_1_minus_delta": q,
                "bound_holds": bool(holds),
                "tightness_ratio": None if (math.isnan(fl) or fl == 0)
                                   else round(q / fl, 3),
            })
        rows.append(row)
        b = row["bounds"][1] if len(row["bounds"]) > 1 else row["bounds"][0]
        print(f"stratum {si}: p_hat~{row['mean_p_hat']:.2f} ZVF={zvf:.3f} "
              f"mean_obs={row['observed_mean_rollouts']:.0f} "
              f"floor(d={b['delta']})={b['floor_rollouts']} "
              f"holds={b['bound_holds']} tight={b['tightness_ratio']}",
              flush=True)

    out = RESULTS_DIR / f"t2_floor_{pool['tag']}_G{G}.json"
    write_result(out, {
        "kind": "t2_wasted_compute_floor",
        "status": "complete",
        "pool": str(args.pool),
        "pool_tag": pool["tag"],
        "model": pool["model"],
        "seed": args.seed,
        "generated_at": utc_now(),
        "bound_form": "rollouts >= G * ceil(ln(delta)/ln(ZVF)) -- geometric "
                      "instantiation; cross-check constant vs zvf_theory.tex T2",
        "rows": rows,
    })
    print(f"-> {out}")


if __name__ == "__main__":
    main()
