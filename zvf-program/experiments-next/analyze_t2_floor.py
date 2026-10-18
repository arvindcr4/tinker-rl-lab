#!/usr/bin/env python3
"""analyze_t2_floor.py — E-T2: empirical test of the waiting-time quantile.

T2 (theory/zvf_theory.tex) gives a RELIABILITY BUDGET: the (1-delta)-quantile
of rollouts until the next NON-DEGENERATE advantage event (a mixed-reward
group) when the observed ZVF is high. Instantiated here:

    P(no mixed group in n consecutive groups) = ZVF^n <= delta
    =>  n(delta, ZVF) = ceil( ln(delta) / ln(ZVF) )   groups
    =>  reliability budget = G * n(delta, ZVF)        rollouts

This is a quantile, NOT a minimum: an informative group arrives within the
first G rollouts with probability 1-ZVF. The check below descriptively compares the
empirical (1-delta)-quantile matches (and, given the ceiling, does not
exceed) the geometric model's quantile.

NOTE: cross-check the constant against the theorem statement in
zvf_theory.tex before citing (the theorem carries proof-gap markers — this
script is exploratory, not validation or proof).

Method (pure offline, resampled from a build_pool.py pool):
  1. Stratify prompts by p_hat into strata whose group-level ZVF(G) spans
     ~[0.5, 0.99].
  2. Within each stratum, repeatedly draw groups (size G, prompts sampled with
     replacement, rollouts subsampled without replacement) until the first
     mixed group; record rollouts consumed.
  3. Compare the empirical distribution of rollouts-to-first-mixed against the
     model quantile at delta in {0.5, 0.1, 0.05}: the model FITS if the
     empirical (1-delta)-quantile is <= the (ceiled) budget, and is EXACT if
     the ratio is near 1.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from analysis_contract import validate_design, strata_all
from analyze_t1_ci import population_zvf, wilson_ci

from common import RESULTS_DIR, load_pool, utc_now, write_result


def group_zvf_of_stratum(rng: random.Random, stratum: list[dict], G: int,
                         draws: int = 4000) -> float:
    return population_zvf(stratum, G, rng)


def rollouts_until_mixed(rng: random.Random, stratum: list[dict], G: int,
                         cap_groups: int = 100000) -> dict:
    validate_design(groups=[G], counts=[cap_groups], n_prompts=len(stratum))
    n = 0
    while n < cap_groups:
        n += 1
        p = rng.choice(stratum)
        group = rng.sample(p["rewards"], G)
        if len(set(group)) > 1:
            return {"rollouts": n * G, "censored": False}
    return {"rollouts": cap_groups * G, "censored": True}


def floor_rollouts(zvf: float, delta: float, G: int) -> float:
    if not 0 < delta < 1 or G < 2:
        raise ValueError("require 0 < delta < 1 and G >= 2")
    if zvf <= 0.0 or zvf >= 1.0:
        return float("nan")
    return G * math.ceil(math.log(delta) / math.log(zvf))


def quantile(xs: list[int], q: float) -> float:
    if not xs or not 0 <= q <= 1:
        raise ValueError("nonempty sample and quantile in [0,1] required")
    ys = sorted(xs)
    idx = max(0, math.ceil(q * len(ys))-1)
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
    validate_design(groups=[G], counts=[args.trials,args.n_strata], available=pool["rollouts_per_prompt"], n_prompts=len(prompts))
    if any(not 0 < d < 1 for d in args.deltas):
        raise ValueError("all deltas must lie strictly between zero and one")
    if G > pool["rollouts_per_prompt"]:
        raise SystemExit(f"G={G} exceeds pool rollouts_per_prompt")
    rng = random.Random(args.seed)

    # Stratify by p_hat quantiles; extreme p_hat strata have high ZVF(G).
    ordered = sorted(prompts, key=lambda p: p["p_hat"])
    strata = strata_all(ordered, args.n_strata)

    rows = []
    for si, stratum in enumerate(strata):
        zvf = group_zvf_of_stratum(rng, stratum, G)
        if zvf >= 0.999 or zvf <= 0.001:
            note = "degenerate stratum (ZVF ~ 0 or 1); floor undefined/infinite"
        else:
            note = ""
        obs = [rollouts_until_mixed(rng, stratum, G) for _ in range(args.trials)]
        times = [o["rollouts"] for o in obs]
        censoring_fraction = sum(o["censored"] for o in obs)/len(obs)
        epsilon = math.sqrt(math.log(40)/(2*len(obs)))
        def identifiable_quantile(q):
            value = quantile(times,q)
            if any(o["censored"] and o["rollouts"] <= value for o in obs):
                return None
            return value
        row = {
            "stratum": si,
            "n_prompts": len(stratum),
            "mean_p_hat": round(sum(p["p_hat"] for p in stratum) / len(stratum), 4),
            "zvf_G": round(zvf, 4),
            "G": G,
            "trials": args.trials,
            "observed_mean_rollouts": round(sum(times)/len(times),1),
            "mean_is_lower_bound": censoring_fraction > 0,
            "censoring_fraction": censoring_fraction,
            "quantile_convention": "nearest_rank",
            "note": note,
            "bounds": [],
        }
        for delta in args.deltas:
            fl = floor_rollouts(zvf, delta, G)
            q = identifiable_quantile(1.0-delta)
            q_ci = [identifiable_quantile(max(0,1-delta-epsilon)),
                    identifiable_quantile(min(1,1-delta+epsilon))]
            # fl is the geometric-model (1-delta)-quantile of rollouts to the
            # first informative group (a reliability budget, NOT a minimum).
            # The ceiling makes fl an upper estimate of the exact quantile, so
            # the model-consistency check is q <= fl; the earlier `q >= fl`
            # tested the inverted statement.
            quantile_ok = None if q is None or math.isnan(fl) else q <= fl
            # The event T <= budget is observed only if censoring occurs after
            # the budget. Otherwise its coverage remains unidentified.
            coverage_identified = not math.isnan(fl) and all(
                not o["censored"] or o["rollouts"] >= fl for o in obs)
            hits = sum(not o["censored"] and o["rollouts"] <= fl for o in obs) if coverage_identified else None
            row["bounds"].append({
                "delta": delta,
                "reliability_budget_rollouts": None if math.isnan(fl) else fl,
                "observed_q_at_1_minus_delta": q,
                "quantile_within_budget": quantile_ok,
                "quantile_ci95_dkw": q_ci,
                "ratio_ci95_dkw": [v/fl if v is not None and not math.isnan(fl) and fl else None for v in q_ci],
                "model_validation_established": False,
                "budget_event_coverage": hits/len(obs) if hits is not None else None,
                "budget_event_mc_ci95": wilson_ci(hits/len(obs),len(obs)) if hits is not None else None,
                "coverage_target": 1-delta,
                "fit_ratio_obs_over_model": None if (q is None or math.isnan(fl) or fl == 0)
                                            else round(q / fl, 3),
            })
        rows.append(row)
        b = row["bounds"][1] if len(row["bounds"]) > 1 else row["bounds"][0]
        print(f"stratum {si}: p_hat~{row['mean_p_hat']:.2f} ZVF={zvf:.3f} "
              f"mean_obs={row['observed_mean_rollouts']:.0f} "
              f"budget(d={b['delta']})={b['reliability_budget_rollouts']} "
              f"fits={b['quantile_within_budget']} "
              f"ratio={b['fit_ratio_obs_over_model']}",
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
        "bound_form": "reliability budget: (1-delta)-quantile of rollouts to "
                      "first mixed group = G * ceil(ln(delta)/ln(ZVF)); a "
                      "quantile, not a minimum (2026-07-11 direction fix)",
        "rows": rows,
    })
    print(f"-> {out}")


if __name__ == "__main__":
    main()
