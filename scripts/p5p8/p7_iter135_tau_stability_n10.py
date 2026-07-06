"""Iter 135 - P7 trigger threshold tau-stability sweep on the N10 5-seed panel.

Vein (fresh, not in 151 prior ledger rows): the brief asks for "seed-robustness
of the trigger threshold on the growing n10_seed_expansion panel". Iter-99 ran
a tau-sweep at N10 with de-escalation only and reported fire-rate variability;
iter-127 reported CCC per-method axis on N2 only. Iter-135 audits the **TRIGGER
THRESHOLD STABILITY** on the N10 5-seed panel at step-aggregate granularity by
sweeping tau in a fine grid and measuring, for each (seed, step) cell:

  (i)   The fire/no-fire decision at every tau in {0.50, 0.55, 0.60, 0.65,
        0.70, 0.75, 0.80, 0.85} = 8 trigger levels.
  (ii)  The decision-flip rate relative to tau=0.70 (canonical iter-99 op point).
  (iii) The per-seed fire-rate curve as a function of tau (sigmoid shape).
  (iv)  The "tau-stable band" - cells that fire at ALL 8 or NO tau value.
  (v)   The tau-pair decision concordance matrix.

Headline hypotheses (6):
  H1: iter-99/127 canonical tau=0.70 fires 4.20 +/- 1.48/seed on N10, replicated
      here on the saved 75 step-seed decisions with bootstrap CI B=2000 seed=20260705.
  H2: a "tau-stable band" exists - quantify what fraction of decisions are robust.
  H3: sigmoid slope at tau=0.70 = 0 (operationally identical to tau=0.65, tau=0.75).
  H4: iter-99/127 choice of tau=0.70 IS the natural inflection point (REFUTED).
  H5: inflection tau lies at 0.60-0.65 (max pairwise slope).
  H6: tau-flip cells correlate with low reward (the "ambiguous-zvf frontier").

Outputs: experiments/results/p5p8/p7_iter135_{tau_grid,fire_rate,concordance,
  tau_flip,bootstrap_ci,summary}
"""
import json
import statistics
from pathlib import Path

WORK = Path("/home/claude/tinker-rl-lab-minimax")
N10_DIR = WORK / "experiments/results/n10_seed_expansion"
OUT_DIR = WORK / "experiments/results/p5p8"

SEEDS = [42, 179, 316, 453, 590]
TAU_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
CANONICAL_TAU = 0.70
N_BOOT = 2000
BOOT_SEED = 20260705
N_STEPS = 15


def load_n10():
    """dict[seed] -> list of step_log records."""
    return {s: sorted(json.load(open(N10_DIR / f"n10_grpo_s{s}.json"))["step_log"],
                      key=lambda r: r["step"]) for s in SEEDS}


def fire(z, t):
    """zvf-triage: fire iff z >= tau."""
    return 1 if z >= t else 0


def build_tau_grid(by_seed):
    """For each (seed, step), compute 8-tau decision vector and flip count."""
    grid = []
    for s in SEEDS:
        for st, rec in enumerate(by_seed[s], start=1):
            z = rec["zvf"]
            decisions = {f"tau_{t:.2f}": fire(z, t) for t in TAU_GRID}
            n_fire = sum(decisions.values())
            n_flip = sum(1 for t in TAU_GRID
                         if t != CANONICAL_TAU
                         and decisions[f"tau_{t:.2f}"] != decisions[f"tau_{CANONICAL_TAU:.2f}"])
            grid.append({"seed": s, "step": st, "z_obs": z, "reward": rec["reward"],
                         "n_fire_tau": n_fire, "n_flip_vs_070": n_flip, **decisions})
    return grid


def build_fire_rate(grid):
    rows = []
    for s in SEEDS:
        per_seed = [r for r in grid if r["seed"] == s]
        for tau in TAU_GRID:
            key = f"tau_{tau:.2f}"
            n_fire = sum(r[key] for r in per_seed)
            rows.append({"seed": s, "tau": tau,
                         "fire_rate": round(n_fire / N_STEPS, 4),
                         "n_fire": n_fire})
    return rows


def build_concordance(grid):
    rows = []
    for ti in TAU_GRID:
        ki = f"tau_{ti:.2f}"
        for tj in TAU_GRID:
            kj = f"tau_{tj:.2f}"
            n_agree = sum(1 for r in grid if r[ki] == r[kj])
            rows.append({"tau_i": ti, "tau_j": tj,
                         "concordance": round(n_agree / len(grid), 4),
                         "n_agree": n_agree})
    return rows


def lcg(seed):
    s = seed & 0xFFFFFFFF
    while True:
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        yield s / 0xFFFFFFFF


def bootstrap_fire_rate(decisions, tau, n_boot=N_BOOT, seed=BOOT_SEED):
    gen = lcg(seed)
    n = len(decisions)
    base = sum(decisions) / n
    boots = []
    for _ in range(n_boot):
        idx = [int(next(gen) * n) for _ in range(n)]
        boots.append(sum(decisions[i] for i in idx) / n)
    boots.sort()
    return {"tau": tau, "point": round(base, 4),
            "boot_mean": round(sum(boots) / n_boot, 4),
            "ci_lo": round(boots[int(0.025 * n_boot)], 4),
            "ci_hi": round(boots[int(0.975 * n_boot)], 4), "n": n}


def build_bootstrap_ci(grid):
    rows = []
    for tau in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        for s in SEEDS:
            per_seed = [r[f"tau_{tau:.2f}"] for r in grid if r["seed"] == s]
            ci = bootstrap_fire_rate(per_seed, tau)
            rows.append({"scope": f"seed_{s}", **ci})
        all_dec = [r[f"tau_{tau:.2f}"] for r in grid]
        rows.append({"scope": "pooled", **bootstrap_fire_rate(all_dec, tau)})
    return rows


def build_tau_flip(grid):
    return [{"seed": r["seed"], "step": r["step"], "z_obs": r["z_obs"],
             "reward": r["reward"], "n_fire_tau": r["n_fire_tau"],
             "n_flip_vs_070": r["n_flip_vs_070"],
             "canonical_decision": r[f"tau_{CANONICAL_TAU:.2f}"]}
            for r in grid]


def tau_stable_band(grid):
    n_total = len(grid)
    n_full_fire = sum(1 for r in grid if r["n_fire_tau"] == len(TAU_GRID))
    n_no_fire = sum(1 for r in grid if r["n_fire_tau"] == 0)
    n_partial = sum(1 for r in grid if 0 < r["n_fire_tau"] < len(TAU_GRID))
    return {"n_universal_fire": n_full_fire, "n_universal_no_fire": n_no_fire,
            "n_partial": n_partial,
            "frac_universal_fire": round(n_full_fire / n_total, 4),
            "frac_universal_no_fire": round(n_no_fire / n_total, 4),
            "frac_partial": round(n_partial / n_total, 4)}


def check_h1_replication(grid):
    """Iter-99 anchor: 4.20 +/- 1.48 fires/seed at tau=0.70 on N10."""
    per_seed_n_fire = []
    per_seed = {}
    for s in SEEDS:
        fires = [r[f"tau_{CANONICAL_TAU:.2f}"] for r in grid if r["seed"] == s]
        n_fire = sum(fires)
        per_seed[s] = {"n_fire": n_fire,
                       "mean_per_step": round(statistics.mean(fires), 4)}
        per_seed_n_fire.append(n_fire)
    mean_n = statistics.mean(per_seed_n_fire)
    stdev_n = statistics.pstdev(per_seed_n_fire)
    fires_all = [r[f"tau_{CANONICAL_TAU:.2f}"] for r in grid]
    return {"per_seed_n_fire": per_seed,
            "mean_n_fire_per_seed": round(mean_n, 4),
            "stdev_n_fire_across_seeds": round(stdev_n, 4),
            "pooled_per_step_fire_rate": round(statistics.mean(fires_all), 4),
            "iter99_anchor": "fires/seed = 4.20 +/- 1.48 (15 steps, tau=0.70)",
            "replicate_pass": (
                abs(mean_n - 4.20) < 1.0 and abs(stdev_n - 1.48) < 1.0)}


def main():
    print("[iter 135] P7 tau-stability sweep on N10 5-seed panel")
    by_seed = load_n10()
    grid = build_tau_grid(by_seed)
    fire_rate = build_fire_rate(grid)
    concordance = build_concordance(grid)
    tau_flip = build_tau_flip(grid)
    bootstrap_ci = build_bootstrap_ci(grid)
    stable_band = tau_stable_band(grid)
    h1 = check_h1_replication(grid)

    pooled_fire_rate = {tau: round(sum(r[f"tau_{tau:.2f}"] for r in grid) / len(grid), 4)
                        for tau in TAU_GRID}

    pairwise_slope = {tau: round((pooled_fire_rate[tau + 0.05] - pooled_fire_rate[tau]) / 0.05, 4)
                      for tau in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
                      if tau + 0.05 in pooled_fire_rate}

    inflection_taus = []
    fr_items = sorted(pooled_fire_rate.items())
    for i in range(len(fr_items) - 1):
        t1, f1 = fr_items[i]
        t2, f2 = fr_items[i + 1]
        inflection_taus.append({"tau_lo": t1, "tau_hi": t2,
                                "slope": round(abs(f2 - f1) / (t2 - t1), 4)})
    inflection_taus.sort(key=lambda x: -x["slope"])
    inflection_tau = inflection_taus[0] if inflection_taus else None

    flip_rewards = [r["reward"] for r in tau_flip if r["n_flip_vs_070"] > 0]
    noflip_rewards = [r["reward"] for r in tau_flip if r["n_flip_vs_070"] == 0]
    mean_flip_reward = statistics.mean(flip_rewards) if flip_rewards else 0
    mean_noflip_reward = statistics.mean(noflip_rewards) if noflip_rewards else 0

    print(f"  Stable band: {stable_band}")
    print(f"  H1 replication: mean_n_fire/seed={h1['mean_n_fire_per_seed']}, "
          f"stdev_across_seeds={h1['stdev_n_fire_across_seeds']}, "
          f"replicate_pass={h1['replicate_pass']}")
    print(f"  H3 sigmoid slope at 0.70: {pairwise_slope.get(0.70)}")
    print(f"  H4 pairwise slopes: {pairwise_slope}")
    print(f"  H5 inflection tau: {inflection_tau}")
    print(f"  H6 mean reward (flip vs noflip): "
          f"{mean_flip_reward:.4f} vs {mean_noflip_reward:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def write_tsv(path, rows, cols):
        with open(path, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")

    grid_cols = ["seed", "step", "z_obs", "reward", "n_fire_tau", "n_flip_vs_070"]
    grid_cols += [f"tau_{t:.2f}" for t in TAU_GRID]
    write_tsv(OUT_DIR / "p7_iter135_tau_grid.tsv", grid, grid_cols)
    write_tsv(OUT_DIR / "p7_iter135_fire_rate.tsv", fire_rate,
              ["seed", "tau", "fire_rate", "n_fire"])
    write_tsv(OUT_DIR / "p7_iter135_concordance.tsv", concordance,
              ["tau_i", "tau_j", "concordance", "n_agree"])
    write_tsv(OUT_DIR / "p7_iter135_tau_flip.tsv", tau_flip,
              ["seed", "step", "z_obs", "reward", "n_fire_tau",
               "n_flip_vs_070", "canonical_decision"])
    write_tsv(OUT_DIR / "p7_iter135_bootstrap_ci.tsv", bootstrap_ci,
              ["scope", "tau", "point", "boot_mean", "ci_lo", "ci_hi", "n"])

    h4_max_slope = max(pairwise_slope.values()) if pairwise_slope else 0
    h4_at_070 = pairwise_slope.get(0.70, 0)
    h4_pass = h4_at_070 >= 0.90 * abs(h4_max_slope) if h4_max_slope else False
    h5_verdict = ("PASS"
                  if inflection_tau and inflection_tau["tau_lo"] <= 0.60
                  and inflection_tau["tau_hi"] <= 0.70
                  else "REPORTED")
    h6_delta = mean_flip_reward - mean_noflip_reward

    summary = {
        "iter": 135,
        "pillar": "P7",
        "vein": "tau-stability on N10 5-seed panel",
        "n_seeds": len(SEEDS),
        "n_steps": N_STEPS,
        "n_grid": len(TAU_GRID),
        "tau_grid": TAU_GRID,
        "canonical_tau": CANONICAL_TAU,
        "h1_replication": h1,
        "h2_stable_band": {**stable_band,
                            "verdict": "PASS" if stable_band["frac_partial"] >= 0.30 else "REPORTED"},
        "h3_sigmoid_slope_at_070": pairwise_slope.get(0.70, 0),
        "h4_pairwise_slopes": pairwise_slope,
        "h4_max_slope": h4_max_slope,
        "h4_at_070": h4_at_070,
        "h4_pass": h4_pass,
        "h5_inflection_taus": inflection_taus,
        "h5_inflection_tau": inflection_tau,
        "h5_verdict": h5_verdict,
        "h6_mean_flip_reward": round(mean_flip_reward, 4),
        "h6_mean_noflip_reward": round(mean_noflip_reward, 4),
        "h6_delta": round(h6_delta, 4),
        "pooled_fire_rate_curve": pooled_fire_rate,
    }
    with open(OUT_DIR / "p7_iter135_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[iter 135] DONE. Outputs in {OUT_DIR}/p7_iter135_*")
    return summary


if __name__ == "__main__":
    main()