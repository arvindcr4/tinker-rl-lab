"""
P7 iter-155 — tau-trigger stability audit on the GROWING N10 5-seed panel.

Closes brief vein (c): "seed-robustness of the trigger threshold on the
growing n10_seed_expansion panel" + brief vein (d) "bootstrap CIs on every
P7 headline".

Pipeline
--------
1. Load 5 GRPO seeds (s42, s179, s316, s453, s590) from the n10_seed_expansion
   panel. Each carries 15 (step, reward, zvf) tuples.
2. For each threshold tau in a 11-point sweep
   {0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80}:
   - Fire rate per seed = fraction of 15 steps with zvf < tau
   - Mean fire rate across 5 seeds
   - Cross-seed SD on fire rate
   - Cross-seed CV (SD/mean)
   - Bootstrap 2000 resamples (seed=20260705) for CI95 on mean fire rate
3. Test falsifiable hypotheses:
   - H1 (stability): CV at tau=0.70 < 30%
   - H2 (narrow plateau): [0.60, 0.70] tau plateaus — fire rate CV < 30%
       and  max-min fire rate < 0.20 across this band
   - H3 (anchor reproducibility): mean fire rate at tau=0.70 in
       [0.20, 0.40] (the iter-99/135 N10 anchor range)
   - H4 (steady-state predicts accuracy): heldout_acc correlates with
       mean(last-5 zvf) > 0 across seeds
4. Outputs: per-tau table + per-seed table + summary JSON.

Stdlib only. ~250 LoC.
"""

import csv
import json
import math
import os
import random
from pathlib import Path

REPO = Path("/home/claude/tinker-rl-lab-minimax")
PANEL = REPO / "platform_hybrid/experiments/results/n10_seed_expansion"
OUT = REPO / "platform_hybrid/experiments/results/p5p8"
SEEDS = [42, 179, 316, 453, 590]
TAUS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
N_BOOT = 2000
BOOT_SEED = 20260705
LAST_K = 5  # for steady-state ZVF (last K steps)


def load_seed(seed):
    """Load N10 seed's 15-step trajectory."""
    fp = PANEL / f"n10_grpo_s{seed}.json"
    d = json.loads(fp.read_text())
    steps = d["step_log"]
    return {
        "seed": seed,
        "n_steps": len(steps),
        "zvf": [s["zvf"] for s in steps],
        "reward": [s["reward"] for s in steps],
        "mean_zvf": d["mean_zvf"],
        "last10_avg_reward": d["last10_avg_reward"],
        "first5_avg_reward": d["first5_avg_reward"],
        "heldout_acc": d["heldout_acc"],
        "mean_len_last5": d["mean_len_last5"],
    }


def bootstrap_ci(values, n_boot=N_BOOT, seed=BOOT_SEED, alpha=0.05):
    """Bootstrap CI on mean of `values`."""
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (None, None, None)
    means = []
    for _ in range(n_boot):
        s = sum(values[rng.randint(0, n - 1)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return (sum(values) / n, lo, hi)


def fire_rate(zvf_list, tau):
    if not zvf_list:
        return 0.0
    return sum(1 for z in zvf_list if z < tau) / len(zvf_list)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = [load_seed(s) for s in SEEDS]

    # --- per-seed table ---
    per_seed_rows = []
    for s in data:
        for tau in TAUS:
            fr = fire_rate(s["zvf"], tau)
            per_seed_rows.append({
                "seed": s["seed"],
                "tau": tau,
                "fire_count": sum(1 for z in s["zvf"] if z < tau),
                "n_steps": s["n_steps"],
                "fire_rate": round(fr, 4),
                "heldout_acc": s["heldout_acc"],
                "mean_zvf": s["mean_zvf"],
                "last5_zvf_mean": round(sum(s["zvf"][-LAST_K:]) / LAST_K, 4),
                "last10_reward": s["last10_avg_reward"],
            })

    with open(OUT / "p7_iter155_per_seed.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_seed_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(per_seed_rows)

    # --- per-tau aggregate ---
    per_tau_rows = []
    for tau in TAUS:
        per_seed_fr = [fire_rate(s["zvf"], tau) for s in data]
        m, lo, hi = bootstrap_ci(per_seed_fr)
        sd = (sum((x - m) ** 2 for x in per_seed_fr) / max(1, len(per_seed_fr) - 1)) ** 0.5
        cv = sd / m if m > 0 else float("nan")
        per_tau_rows.append({
            "tau": tau,
            "n_seeds": len(per_seed_fr),
            "mean_fire_rate": round(m, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "sd_across_seeds": round(sd, 4),
            "cv_across_seeds": round(cv, 4),
            "min_fire_rate": round(min(per_seed_fr), 4),
            "max_fire_rate": round(max(per_seed_fr), 4),
            "max_min_spread": round(max(per_seed_fr) - min(per_seed_fr), 4),
        })

    with open(OUT / "p7_iter155_per_tau.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_tau_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(per_tau_rows)

    # --- H1..H4 ---
    # find tau=0.70 row
    row070 = next(r for r in per_tau_rows if abs(r["tau"] - 0.70) < 1e-9)
    band = [r for r in per_tau_rows if 0.60 <= r["tau"] <= 0.70]

    H1 = row070["cv_across_seeds"] < 0.30
    H2 = (
        all(r["cv_across_seeds"] < 0.30 for r in band)
        and max(r["max_min_spread"] for r in band) < 0.20
    )
    H3 = 0.20 <= row070["mean_fire_rate"] <= 0.40

    # H4: heldout_acc vs mean(last-K zvf) Pearson r
    heldout = [s["heldout_acc"] for s in data]
    last5_zvf = [sum(s["zvf"][-LAST_K:]) / LAST_K for s in data]
    last10_zvf = [sum(s["zvf"][-10:]) / 10 for s in data]
    mean_zvf = [s["mean_zvf"] for s in data]

    def pearson(x, y):
        n = len(x)
        if n < 2:
            return 0.0
        mx = sum(x) / n
        my = sum(y) / n
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        den_x = sum((xi - mx) ** 2 for xi in x) ** 0.5
        den_y = sum((yi - my) ** 2 for yi in y) ** 0.5
        if den_x * den_y == 0:
            return 0.0
        return num / (den_x * den_y)

    # bootstrap CI on Pearson r for last5_zvf vs heldout (small n=5 so wide CI)
    rng = random.Random(BOOT_SEED)
    boot_rs = []
    for _ in range(N_BOOT):
        idx = [rng.randint(0, n - 1) for n in [len(heldout)] * len(heldout)]
        bx = [last5_zvf[i] for i in idx]
        by = [heldout[i] for i in idx]
        boot_rs.append(pearson(bx, by))
    boot_rs.sort()
    r_lo = boot_rs[int(0.025 * N_BOOT)]
    r_hi = boot_rs[int(0.975 * N_BOOT)]

    r_heldout_last5 = pearson(last5_zvf, heldout)
    r_heldout_last10 = pearson(last10_zvf, heldout)
    r_heldout_meanzvf = pearson(mean_zvf, heldout)
    r_heldout_last10rwd = pearson(
        [s["last10_avg_reward"] for s in data], heldout
    )

    # Iter-99 anchor: steady-state predicts accuracy > 0 (one-sided test).
    H4 = r_heldout_last5 > 0 and r_lo > 0

    # Bootstrap CI on tau=0.70 fire rate
    per_seed_fr_070 = [fire_rate(s["zvf"], 0.70) for s in data]
    m070, lo070, hi070 = bootstrap_ci(per_seed_fr_070)

    summary = {
        "iter": 155,
        "n_seeds": len(SEEDS),
        "n_steps_per_seed": data[0]["n_steps"],
        "panel_total_step_observations": sum(s["n_steps"] for s in data),
        "taus_swept": TAUS,
        "tau_070_anchor": {
            "mean_fire_rate": round(m070, 4),
            "ci_lo": round(lo070, 4),
            "ci_hi": round(hi070, 4),
            "cv_across_seeds": round(row070["cv_across_seeds"], 4),
            "min_fr": round(row070["min_fire_rate"], 4),
            "max_fr": round(row070["max_fire_rate"], 4),
        },
        "per_seed_heldout": {s["seed"]: s["heldout_acc"] for s in data},
        "per_seed_mean_zvf": {s["seed"]: s["mean_zvf"] for s in data},
        "per_seed_last5_zvf": {s["seed"]: round(sum(s["zvf"][-LAST_K:]) / LAST_K, 4) for s in data},
        "correlations": {
            "r_heldout_vs_last5_zvf": round(r_heldout_last5, 4),
            "r_heldout_vs_last5_zvf_ci": [round(r_lo, 4), round(r_hi, 4)],
            "r_heldout_vs_last10_zvf": round(r_heldout_last10, 4),
            "r_heldout_vs_mean_zvf": round(r_heldout_meanzvf, 4),
            "r_heldout_vs_last10_reward": round(r_heldout_last10rwd, 4),
        },
        "hypotheses": {
            "H1_tau070_stable_cv_lt_030": {"verdict": "PASS" if H1 else "FAIL",
                                            "cv": round(row070["cv_across_seeds"], 4),
                                            "bar": 0.30},
            "H2_plateau_060_070_narrow": {
                "verdict": "PASS" if H2 else "FAIL",
                "band_max_min_spread": round(max(r["max_min_spread"] for r in band), 4),
                "bar_spread": 0.20,
            },
            "H3_tau070_anchor_in_020_040": {
                "verdict": "PASS" if H3 else "FAIL",
                "mean_fire_rate": round(m070, 4),
                "ci": [round(lo070, 4), round(hi070, 4)],
                "band": [0.20, 0.40],
            },
            "H4_steady_state_predicts_heldout": {
                "verdict": "PASS" if H4 else "FAIL",
                "r": round(r_heldout_last5, 4),
                "ci_lo": round(r_lo, 4),
                "bar_one_sided": 0.0,
            },
        },
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
    }

    with open(OUT / "p7_iter155_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- print to stdout for quick read ---
    print(f"Iter 155 P7 τ-stability on N10 5-seed panel")
    print(f"  Panel: {len(data)} seeds × {data[0]['n_steps']} steps = "
          f"{sum(s['n_steps'] for s in data)} step-observations")
    print(f"  Heldout acc range: [{min(heldout):.4f}, {max(heldout):.4f}]")
    print(f"  Mean ZVF range: [{min(mean_zvf):.4f}, {max(mean_zvf):.4f}]")
    print()
    print(f"Per-tau table:")
    print(f"  {'tau':>5} {'mean_fr':>8} {'CI95':>16} {'SD':>6} {'CV':>6} {'spread':>7}")
    for r in per_tau_rows:
        print(f"  {r['tau']:5.2f} {r['mean_fire_rate']:8.4f} "
              f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] "
              f"{r['sd_across_seeds']:6.3f} {r['cv_across_seeds']:6.3f} "
              f"{r['max_min_spread']:7.3f}")
    print()
    print(f"τ=0.70 anchor: mean fire rate = {m070:.4f} "
          f"[{lo070:.4f}, {hi070:.4f}]")
    print(f"H1 (CV<0.30 @ τ=0.70): {summary['hypotheses']['H1_tau070_stable_cv_lt_030']['verdict']}")
    print(f"H2 (plateau [0.60,0.70] narrow): {summary['hypotheses']['H2_plateau_060_070_narrow']['verdict']}")
    print(f"H3 (anchor in [0.20,0.40]): {summary['hypotheses']['H3_tau070_anchor_in_020_040']['verdict']}")
    print(f"H4 (last5_zvf predicts heldout): {summary['hypotheses']['H4_steady_state_predicts_heldout']['verdict']}")
    print()
    print(f"Correlations vs heldout_acc:")
    print(f"  r(heldout, last5_zvf)  = {r_heldout_last5:+.4f}  CI [{r_lo:.4f}, {r_hi:.4f}]")
    print(f"  r(heldout, last10_zvf) = {r_heldout_last10:+.4f}")
    print(f"  r(heldout, mean_zvf)   = {r_heldout_meanzvf:+.4f}")
    print(f"  r(heldout, last10_rwd) = {r_heldout_last10rwd:+.4f}")


if __name__ == "__main__":
    main()