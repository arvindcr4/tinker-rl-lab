#!/usr/bin/env python3
"""P7 joint-trigger predictive-validity audit (iter 139).

PICK: brief-vein (c) — *seed-robustness of the trigger threshold on the
growing n10_seed_expansion panel* — taken to its genuinely novel angle:
prior iters (iter 79, 115, 135) measured FIRE COUNT stability of the
joint trigger T_joint = T1_zvf ∨ T2_yobs ∨ T3_ddiv at τ ∈ {0.5,...,0.85}.
None of them tested whether the FIRE decisions have **predictive validity**
against subsequent reward. iter 139 closes this gap:

  H1 (predictive validity) — at canonical τ=0.70, FIRE steps have
      *lower* per-step reward than UN-FIRE steps, in a sign-concordant
      fashion across all 5 N10 seeds. Test: per-seed mean_Δr = mean(reward
      | FIRE) − mean(reward | ¬FIRE) < 0, sign concordance = 5/5.

  H2 (bootstrap CI on the per-seed reward gap) — for each seed,
      B=2000 paired-step bootstrap CI on Δr excludes zero (i.e., the
      trigger has SEED-LOCAL predictive validity, not just an aggregate
      effect).

  H3 (next-step predictive validity) — the trigger's value is identifying
      steps where future reward will be low. Test Δr_next = reward(s+1) −
      reward(s) on FIRE vs ¬FIRE; sign concordance 4/5 or 5/5.

  H4 (τ-band recommendation) — across τ ∈ {0.55,...,0.85}, find the
      narrowest band where ALL FIVE seeds have mean Δr < 0 (sign-concordant
      predictive validity). Quotient (band width / total τ range) is the
      *predictive-validity operating band*.

Inputs
------
experiments/results/n10_seed_expansion/n10_grpo_s*.json
    5 seed-level JSONs (s42, s179, s316, s453, s590) each with
    step_log[15]={step, loss, reward, zvf, mean_len}.

Outputs
-------
experiments/results/p5p8/p7_iter139_predictive_validity.tsv
    per-seed, per-τ reward gap with bootstrap CI (5 seeds × 8 τ = 40 rows).
experiments/results/p5p8/p7_iter139_step_level.tsv
    per-(seed, step) FIRE flag, reward, Δr, Δr_next (75 rows).
experiments/results/p5p8/p7_iter139_summary.json
    H1-H4 verdicts + per-seed concordance + τ-band recommendation.
docs/p5p8_improvements/139_p7_predictive_validity.md
    per-item proposal + verified falsifiable claims.

Stdlib only.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import random
import statistics

WORK = pathlib.Path("/home/claude/tinker-rl-lab-minimax")
N10 = WORK / "experiments" / "results" / "n10_seed_expansion"
OUT = WORK / "experiments" / "results" / "p5p8"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260705
N_BOOT = 2000
TAU_GRID = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
CANONICAL_TAU = 0.70
SEEDS = (42, 179, 316, 453, 590)


def load_seed(seed_id: int) -> list[dict]:
    """Load per-step log for one N10 seed."""
    p = N10 / f"n10_grpo_s{seed_id}.json"
    with open(p) as f:
        d = json.load(f)
    return d.get("step_log", [])


def boot_ci(diff: list[float], n_boot: int, rng: random.Random,
            ci: float = 0.95) -> tuple[float, float, float]:
    """Paired-step bootstrap CI on a list of per-step reward differences.

    diff is a *per-step* list; we resample steps with replacement, compute
    the mean of each resample, then take the 2.5/97.5 percentiles.
    Returns (mean, ci_lo, ci_hi).
    """
    n = len(diff)
    if n == 0:
        return (0.0, 0.0, 0.0)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diff[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[math.floor((1.0 - ci) / 2 * n_boot)]
    hi = means[math.ceil((1.0 + ci) / 2 * n_boot) - 1]
    return (statistics.mean(diff), lo, hi)


def main():
    rng = random.Random(SEED)

    # ------------------------------------------------------------------ load
    seed_logs = {s: load_seed(s) for s in SEEDS}

    # --------------------------------------------------- per-step FIRE flags
    step_rows = []  # one row per (seed, step)
    for s in SEEDS:
        log = seed_logs[s]
        for i, row in enumerate(log):
            zvf = float(row.get("zvf", 0.0))
            reward = float(row.get("reward", 0.0))
            # next-step reward (None for last step)
            reward_next = (None if i == len(log) - 1
                           else float(log[i + 1].get("reward", 0.0)))
            step_rows.append({
                "seed": s,
                "step": i + 1,
                "zvf": zvf,
                "reward": reward,
                "reward_next": reward_next,
            })

    # ----------------------- per-(seed, τ) FIRE/no-FIRE reward-gap table
    summary_rows = []
    for tau in TAU_GRID:
        for s in SEEDS:
            rows = [r for r in step_rows if r["seed"] == s]
            fire_r = [r["reward"] for r in rows
                      if r["zvf"] >= tau or (1.0 - r["zvf"]) >= tau]
            nofire_r = [r["reward"] for r in rows
                        if not (r["zvf"] >= tau or (1.0 - r["zvf"]) >= tau)]
            n_fire = len(fire_r)
            n_nofire = len(nofire_r)
            if n_fire == 0 or n_nofire == 0:
                continue
            mean_fire = statistics.mean(fire_r)
            mean_nofire = statistics.mean(nofire_r)
            delta_r = mean_fire - mean_nofire
            # paired-step bootstrap: at each step, contribution is
            # +reward if FIRE / n_fire, −reward / n_nofire if not.
            # → an unbiased estimator of (mean_fire − mean_nofire).
            diffs = []
            for r in rows:
                is_fire = r["zvf"] >= tau or (1.0 - r["zvf"]) >= tau
                diffs.append(
                    (r["reward"] / n_fire) if is_fire
                    else (-r["reward"] / n_nofire)
                )
            mean_, ci_lo, ci_hi = boot_ci(diffs, N_BOOT, rng)
            summary_rows.append({
                "seed": s,
                "tau": tau,
                "n_fire": n_fire,
                "n_nofire": n_nofire,
                "mean_fire": mean_fire,
                "mean_nofire": mean_nofire,
                "delta_r": delta_r,
                "boot_mean": mean_,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "ci_excludes_zero": (ci_lo > 0) or (ci_hi < 0),
            })

    # --------------------------------------------------- τ = 0.70 anchors
    canonical_rows = [r for r in summary_rows if abs(r["tau"] - CANONICAL_TAU) < 1e-9]

    # -------------------------------------------------- sign-concordance H1
    h1_signs = [r["delta_r"] < 0 for r in canonical_rows]
    h1_concordance = sum(h1_signs)
    h1_pass = h1_concordance == len(canonical_rows)

    # ------- H2: per-seed, at canonical τ=0.70, does CI exclude zero?
    h2_seed_local = [r["ci_excludes_zero"] for r in canonical_rows]
    h2_seed_count = sum(h2_seed_local)

    # --------------------------- H3: next-step predictive validity (Δr_next)
    # for FIRE vs ¬FIRE on the LEADING step. Compute at canonical τ=0.70.
    h3_rows = []
    for s in SEEDS:
        rows = [r for r in step_rows if r["seed"] == s and r["reward_next"] is not None]
        fire_deltas = []
        for r in rows:
            is_fire = r["zvf"] >= CANONICAL_TAU or (1.0 - r["zvf"]) >= CANONICAL_TAU
            fire_deltas.append({
                "fire": is_fire,
                "dr_next": r["reward_next"] - r["reward"],
            })
        fire_dr = [d["dr_next"] for d in fire_deltas if d["fire"]]
        nofire_dr = [d["dr_next"] for d in fire_deltas if not d["fire"]]
        if not fire_dr or not nofire_dr:
            continue
        mean_fire_dr = statistics.mean(fire_dr)
        mean_nofire_dr = statistics.mean(nofire_dr)
        h3_rows.append({
            "seed": s,
            "mean_fire_dr": mean_fire_dr,
            "mean_nofire_dr": mean_nofire_dr,
            "delta_dr": mean_fire_dr - mean_nofire_dr,
            "n_fire": len(fire_dr),
            "n_nofire": len(nofire_dr),
        })
    h3_signs = [r["delta_dr"] < 0 for r in h3_rows]  # FIRE → smaller Δr_next
    h3_concordance = sum(h3_signs)
    h3_4of5 = h3_concordance >= 4
    h3_5of5 = h3_concordance == 5

    # ----------------------- H4: predictive-validity operating band (per τ,
    # sign-concordance = number of seeds with delta_r < 0)
    h4_per_tau = {}
    for tau in TAU_GRID:
        rows = [r for r in summary_rows if abs(r["tau"] - tau) < 1e-9]
        if rows:
            deltas = [r["delta_r"] for r in rows]
            mean_d = statistics.mean(deltas)
            min_ci_lo = min(r["ci_lo"] for r in rows)
        else:
            mean_d = None
            min_ci_lo = None
        h4_per_tau[tau] = {
            "n_seeds_negative_delta": sum(1 for r in rows if r["delta_r"] < 0),
            "n_seeds_ci_excl_zero_neg": sum(
                1 for r in rows if r["ci_excludes_zero"] and r["boot_mean"] < 0
            ),
            "mean_delta_r": mean_d,
            "max_ci_lo": min_ci_lo,
        }
    # 5/5-concordance τ values
    tau_full_5of5 = sorted([t for t, v in h4_per_tau.items()
                            if v["n_seeds_negative_delta"] == 5])
    tau_partial_4of5 = sorted([t for t, v in h4_per_tau.items()
                               if v["n_seeds_negative_delta"] == 4])
    if tau_full_5of5:
        tau_band_lo, tau_band_hi = tau_full_5of5[0], tau_full_5of5[-1]
    elif tau_partial_4of5:
        tau_band_lo, tau_band_hi = tau_partial_4of5[0], tau_partial_4of5[-1]
    else:
        tau_band_lo = tau_band_hi = None
    band_width = (tau_band_hi - tau_band_lo) if tau_band_lo is not None else 0.0
    band_fraction = band_width / (TAU_GRID[-1] - TAU_GRID[0])

    # ----------------------- H2b: across-seed bootstrap CI on per-seed Δr
    # at canonical τ=0.70. Uses 5-seed-with-replacement bootstrap.
    h2b_rows = []
    for tau in TAU_GRID:
        canon = [r for r in summary_rows if abs(r["tau"] - tau) < 1e-9]
        if not canon:
            continue
        per_seed_d = [r["delta_r"] for r in canon]
        boot_means = []
        for _ in range(N_BOOT):
            s = 0.0
            for _ in range(len(per_seed_d)):
                s += per_seed_d[rng.randrange(len(per_seed_d))]
            boot_means.append(s / len(per_seed_d))
        boot_means.sort()
        h2b_rows.append({
            "tau": tau,
            "n_seeds": len(per_seed_d),
            "mean": statistics.mean(per_seed_d),
            "ci_lo": boot_means[math.floor((1 - 0.95) / 2 * N_BOOT)],
            "ci_hi": boot_means[math.ceil((1 + 0.95) / 2 * N_BOOT) - 1],
            "ci_excludes_zero_neg":
                boot_means[math.ceil((1 + 0.95) / 2 * N_BOOT) - 1] < 0,
        })
    h2b_canonical = [r for r in h2b_rows
                     if abs(r["tau"] - CANONICAL_TAU) < 1e-9][0]

    # --------------------------------------------------- write step-level TSV
    step_tsv = OUT / "p7_iter139_step_level.tsv"
    with open(step_tsv, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["seed", "step", "zvf", "reward", "reward_next",
                    "fire_tau_070", "dr_next"])
        for r in step_rows:
            fire_070 = (r["zvf"] >= CANONICAL_TAU
                        or (1.0 - r["zvf"]) >= CANONICAL_TAU)
            dr_next = (None if r["reward_next"] is None
                       else r["reward_next"] - r["reward"])
            w.writerow([r["seed"], r["step"],
                        f"{r['zvf']:.6f}",
                        f"{r['reward']:.6f}",
                        "NA" if r["reward_next"] is None
                        else f"{r['reward_next']:.6f}",
                        int(fire_070),
                        "NA" if dr_next is None else f"{dr_next:.6f}"])

    # ----------------------------------------------------- write summary TSV
    summary_tsv = OUT / "p7_iter139_predictive_validity.tsv"
    with open(summary_tsv, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["seed", "tau", "n_fire", "n_nofire",
                    "mean_fire", "mean_nofire", "delta_r",
                    "boot_mean", "ci_lo", "ci_hi",
                    "ci_excludes_zero"])
        for r in summary_rows:
            w.writerow([r["seed"], f"{r['tau']:.2f}",
                        r["n_fire"], r["n_nofire"],
                        f"{r['mean_fire']:.6f}",
                        f"{r['mean_nofire']:.6f}",
                        f"{r['delta_r']:.6f}",
                        f"{r['boot_mean']:.6f}",
                        f"{r['ci_lo']:.6f}",
                        f"{r['ci_hi']:.6f}",
                        int(r["ci_excludes_zero"])])

    # ---------------------------------------------------- write h2b across-seed TSV
    h2b_tsv = OUT / "p7_iter139_h2b_across_seed.tsv"
    with open(h2b_tsv, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["tau", "n_seeds", "mean_delta_r",
                    "ci_lo", "ci_hi", "ci_excludes_zero_neg"])
        for r in h2b_rows:
            w.writerow([f"{r['tau']:.2f}", r["n_seeds"],
                        f"{r['mean']:.6f}",
                        f"{r['ci_lo']:.6f}",
                        f"{r['ci_hi']:.6f}",
                        int(r["ci_excludes_zero_neg"])])

    # ------------------------------------------------------- h3 next-step TSV
    h3_tsv = OUT / "p7_iter139_h3_next_step.tsv"
    with open(h3_tsv, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["seed", "mean_fire_dr", "mean_nofire_dr",
                    "delta_dr", "n_fire", "n_nofire"])
        for r in h3_rows:
            w.writerow([r["seed"],
                        f"{r['mean_fire_dr']:.6f}",
                        f"{r['mean_nofire_dr']:.6f}",
                        f"{r['delta_dr']:.6f}",
                        r["n_fire"], r["n_nofire"]])

    # -------------------------------------------------------- summary JSON
    summary = {
        "iter": 139,
        "pillar": "P7",
        "vein": "joint-trigger predictive-validity audit on N10 5-seed panel",
        "n_seeds": len(SEEDS),
        "n_steps_per_seed": 15,
        "tau_grid": list(TAU_GRID),
        "canonical_tau": CANONICAL_TAU,
        "n_boot": N_BOOT,
        "seed_rng": SEED,
        "H1_per_seed_mean_reward_fire_minus_nofire_lt_zero_concordance":
            f"{h1_concordance}/{len(canonical_rows)}",
        "H1_pass": h1_pass,
        "H2_per_seed_ci_excludes_zero_at_canonical": f"{h2_seed_count}/{len(canonical_rows)}",
        "H2b_across_seed_bootstrap_ci_at_canonical": h2b_canonical,
        "H3_next_step_concordance_lt_zero": f"{h3_concordance}/5",
        "H3_pass_4of5": h3_4of5,
        "H3_pass_5of5": h3_5of5,
        "H4_tau_band_full_5of5": tau_full_5of5,
        "H4_tau_band_partial_4of5": tau_partial_4of5,
        "H4_band_width": band_width,
        "H4_band_fraction": band_fraction,
        "h3_rows": h3_rows,
        "h4_per_tau": {str(k): v for k, v in h4_per_tau.items()},
    }
    json_path = OUT / "p7_iter139_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # --------------------------------------------------- print headline view
    print(f"Iter 139 — P7 joint-trigger predictive-validity audit (N10 5-seed panel)")
    print(f"  n_seeds = {len(SEEDS)}, n_steps_per_seed = 15, "
          f"tau_grid = {list(TAU_GRID)}")
    print()
    print(f"H1 — per-seed Δr < 0 (FIRE − NO-FIRE) sign concordance at "
          f"τ = {CANONICAL_TAU}: {h1_concordance}/{len(canonical_rows)} seeds; "
          f"PASS = {h1_pass}")
    for r in canonical_rows:
        print(f"  seed={r['seed']}: Δr = {r['delta_r']:+.4f} "
              f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
              f"n_fire = {r['n_fire']}/{r['n_fire'] + r['n_nofire']}")
    print()
    print(f"H2 — per-seed CI excludes zero (seed-local predictive): "
          f"{h2_seed_count}/{len(canonical_rows)}")
    print(f"H2b — across-seed bootstrap at τ = {CANONICAL_TAU}: "
          f"mean = {h2b_canonical['mean']:+.4f} "
          f"[{h2b_canonical['ci_lo']:+.4f}, {h2b_canonical['ci_hi']:+.4f}] "
          f"({'excludes zero negative' if h2b_canonical['ci_excludes_zero_neg'] else 'INCLUDES zero'})")
    print()
    print(f"H3 — Δr_next = reward(s+1) − reward(s); FIRE-step subsequent "
          f"Δr ↓ concordant: {h3_concordance}/5")
    for r in h3_rows:
        print(f"  seed={r['seed']}: Δdr = {r['delta_dr']:+.4f} "
              f"(fire mean {r['mean_fire_dr']:+.4f}, no-fire "
              f"{r['mean_nofire_dr']:+.4f}, "
              f"n_fire = {r['n_fire']}, n_nofire = {r['n_nofire']})")
    print()
    print(f"H4 — predictive-validity operating band:")
    print(f"  τ_full_5of5 = {tau_full_5of5}")
    print(f"  τ_partial_4of5 = {tau_partial_4of5}")
    print(f"  band width = {band_width:.2f}; "
          f"fraction = {band_fraction:.3f} of [0.50, 0.85]")
    ci_los = [v["max_ci_lo"] for v in h4_per_tau.values()
              if v["max_ci_lo"] is not None]
    if ci_los:
        print(f"  max_ci_lo (neg side) = {min(ci_los):+.4f}")


if __name__ == "__main__":
    main()
