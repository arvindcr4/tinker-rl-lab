#!/usr/bin/env python3
"""Iter 183 — P7 Trigger-Threshold Seed-Robustness on N10 5-seed Panel.

Vein: brief vein (c) — "seed-robustness of the trigger threshold on the growing
n10_seed_expansion panel". Prior iters have calibrated the trigger threshold
on the N2 reward tensors (4 methods × 40 steps, 4 prompt batches):
- iter-79 multitrigger N2 — basic tau sweep
- iter-87/88 hysteresis + n10 single-seed
- iter-92 asymmetric hysteresis
- iter-99 seed-threshold robustness (P7 row 102)
- iter-135 tau stability n10 (P7 row 146)
- iter-155 tau stability 5-seed (P7 row 169)
- iter-175 calibrated-hybrid
- iter-179 contrast-restored (P7 row 187)

But none of those have computed **the trigger firing-rate bootstrap CI and
cross-seed TOST at the GROWING n10_seed_expansion panel** specifically.
This iter is the missing piece — the panel grew live after iter-155 and the
seed-robustness audit at the trigger-bit level has never been re-run.

Definitions:
  fire_{s,t} = 1[ zvf_{s,step} >= tau_t ]    for each (seed, step, tau)
  rate_{s,t} = mean over steps of fire_{s,t}
  mean_rate_t = mean over seeds of rate_{s,t}
  cross_sd_t = std over seeds of rate_{s,t}
  rank_correlation_sp_{s,s'} = Spearman rho of fire-vectors (step, step, ...)

Operational hypotheses (5/6 expected PASS, 1 honest negative):
  H1 — at tau=0.55, mean_rate > 0.50 (signal-rich regime; CI lo > 0.40)
  H2 — at tau=0.80, mean_rate < 0.10 (sparse regime; CI hi < 0.20)
  H3 — firing rate monotone non-increasing in tau on 5/5 seeds
  H4 — at tau=0.70, cross_seed_sd < 0.20 (seed-robustness at natural break)
  H5 — TOST-equivalence across all 10 seed pairs at tau=0.65 (margin 0.15)
  H6 — mean Spearman rho across seed pairs > 0.50

Outputs:
  experiments/results/p5p8/p7_iter183_per_obs.tsv
      75 rows × (seed, step, zvf, reward, fire_0.50..fire_0.85)
  experiments/results/p5p8/p7_iter183_per_seed_rate.tsv
      5 × 8 rows × (seed, tau, rate, ci_lo, ci_hi)
  experiments/results/p5p8/p7_iter183_cross_seed_ci.tsv
      8 rows × (tau, mean_rate, cross_sd, cross_ci_lo, cross_ci_hi)
  experiments/results/p5p8/p7_iter183_tost.tsv
      10 pairs × 8 taus = 80 rows × (pair, tau, lo, hi, tost_pass)
  experiments/results/p5p8/p7_iter183_spearman.tsv
      10 pairs × 1 row × (pair, spearman_rho)
  experiments/results/p5p8/p7_iter183_summary.json
      H1-H6 verdicts + structured stats

Stdlib only; deterministic LCG bootstrap B=2000 seed=20260705.
"""
from __future__ import annotations
import csv
import json
import math
import os
import random
import statistics
from itertools import combinations

WORKTREE = "/home/claude/tinker-rl-lab-minimax"
N10_DIR = os.path.join(WORKTREE, "experiments/results/n10_seed_expansion")
OUT_DIR = os.path.join(WORKTREE, "experiments/results/p5p8")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 179, 316, 453, 590]
TAU_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
B = 2000
SEED = 20260705
ALPHA = 0.05
TOST_MARGIN = 0.15  # equivalence margin for firing-rate pairs
SPEARMAN_BAR = 0.50
CROSS_SD_BAR = 0.20
RATE_HIGH_BAR = 0.50
RATE_LOW_BAR = 0.10


def load_n10():
    """Load N10 5-seed panel; return list of dicts (one per seed)."""
    out = []
    for s in SEEDS:
        path = os.path.join(N10_DIR, f"n10_grpo_s{s}.json")
        with open(path) as fh:
            d = json.load(fh)
        out.append({"seed": s, "step_log": d["step_log"]})
    return out


def bootstrap_ci(v, stat_fn, rng, alpha=ALPHA):
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    pt = stat_fn(v)
    boots = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(stat_fn([v[i] for i in idx]))
    boots.sort()
    lo = boots[int(alpha / 2 * B)]
    hi = boots[int((1 - alpha / 2) * B)]
    return pt, lo, hi, B


def spearman(x, y):
    """Spearman rank correlation (handles ties via midrank)."""
    n = len(x)
    if n < 2:
        return float("nan")

    def rank_with_ties(v):
        sorted_idx = sorted(range(n), key=lambda i: v[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and v[sorted_idx[j]] == v[sorted_idx[i]]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[sorted_idx[k]] = avg_rank
            i = j
        return ranks

    rx, ry = rank_with_ties(x), rank_with_ties(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def main():
    rng = random.Random(SEED)
    panel = load_n10()
    n_steps = len(panel[0]["step_log"])
    print(f"loaded {len(panel)} seeds × {n_steps} steps = {len(panel)*n_steps} obs")

    # Build per-obs fire bit vector
    per_obs = []
    fire_matrix = {s: {t: [] for t in TAU_GRID} for s in SEEDS}  # seed -> tau -> [0/1]
    zvf_by_seed = {}
    reward_by_seed = {}
    for s_obj in panel:
        s = s_obj["seed"]
        sl = s_obj["step_log"]
        zvfs = [x["zvf"] for x in sl]
        rewards = [x["reward"] for x in sl]
        zvf_by_seed[s] = zvfs
        reward_by_seed[s] = rewards
        for i in range(n_steps):
            row = {"seed": s, "step": sl[i]["step"], "zvf": zvfs[i], "reward": rewards[i]}
            for t in TAU_GRID:
                bit = 1 if zvfs[i] >= t else 0
                row[f"fire_{t:.2f}"] = bit
                fire_matrix[s][t].append(bit)
            per_obs.append(row)

    # Per-obs TSV
    with open(os.path.join(OUT_DIR, "p7_iter183_per_obs.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        cols = ["seed", "step", "zvf", "reward"] + [f"fire_{t:.2f}" for t in TAU_GRID]
        w.writerow(cols)
        for r in per_obs:
            w.writerow([r[c] for c in cols])
    print(f"wrote per_obs.tsv ({len(per_obs)} rows)")

    # Per-seed rate + bootstrap CI per tau
    per_seed_rate = []
    for s in SEEDS:
        for t in TAU_GRID:
            v = fire_matrix[s][t]
            pt, lo, hi, b = bootstrap_ci(v, statistics.mean, rng)
            per_seed_rate.append({
                "seed": s, "tau": t, "n_steps": n_steps, "n_fires": sum(v),
                "rate": pt, "ci_lo": lo, "ci_hi": hi
            })
    with open(os.path.join(OUT_DIR, "p7_iter183_per_seed_rate.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["seed", "tau", "n_steps", "n_fires", "rate", "ci_lo", "ci_hi"])
        for r in per_seed_rate:
            w.writerow([r["seed"], r["tau"], r["n_steps"], r["n_fires"],
                        f"{r['rate']:.4f}", f"{r['ci_lo']:.4f}", f"{r['ci_hi']:.4f}"])
    print(f"wrote per_seed_rate.tsv ({len(per_seed_rate)} rows)")

    # Cross-seed aggregate + bootstrap CI of cross-seed mean
    cross_seed = []
    for t in TAU_GRID:
        rates_t = [fire_matrix[s][t] for s in SEEDS]  # list of lists (length 5)
        # per-seed point rates
        seed_rates = [statistics.mean(r) for r in rates_t]
        mean_r = statistics.mean(seed_rates)
        # bootstrap resampling over seeds (with replacement) at the per-seed level
        boots = []
        n_seed = len(SEEDS)
        for _ in range(B):
            idx = [rng.randrange(n_seed) for _ in range(n_seed)]
            boots.append(statistics.mean([seed_rates[i] for i in idx]))
        boots.sort()
        lo = boots[int(ALPHA / 2 * B)]
        hi = boots[int((1 - ALPHA / 2) * B)]
        cross_sd = statistics.stdev(seed_rates) if len(seed_rates) > 1 else 0.0
        cross_seed.append({
            "tau": t, "mean_rate": mean_r, "cross_sd": cross_sd,
            "cross_ci_lo": lo, "cross_ci_hi": hi,
            "min_seed_rate": min(seed_rates), "max_seed_rate": max(seed_rates)
        })
    with open(os.path.join(OUT_DIR, "p7_iter183_cross_seed_ci.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["tau", "mean_rate", "cross_sd", "cross_ci_lo", "cross_ci_hi",
                    "min_seed_rate", "max_seed_rate"])
        for r in cross_seed:
            w.writerow([r["tau"], f"{r['mean_rate']:.4f}", f"{r['cross_sd']:.4f}",
                        f"{r['cross_ci_lo']:.4f}", f"{r['cross_ci_hi']:.4f}",
                        f"{r['min_seed_rate']:.4f}", f"{r['max_seed_rate']:.4f}"])
    print(f"wrote cross_seed_ci.tsv ({len(cross_seed)} rows)")

    # TOST across seed pairs × tau grid
    pairs = list(combinations(SEEDS, 2))
    tost_rows = []
    for s1, s2 in pairs:
        for t in TAU_GRID:
            r1 = statistics.mean(fire_matrix[s1][t])
            r2 = statistics.mean(fire_matrix[s2][t])
            diff = r1 - r2
            # Welch-style TOST via paired block-bootstrap (paired by step)
            n = n_steps
            v1 = fire_matrix[s1][t]
            v2 = fire_matrix[s2][t]
            boots = []
            for _ in range(B):
                idx = [rng.randrange(n) for _ in range(n)]
                bs1 = [v1[i] for i in idx]
                bs2 = [v2[i] for i in idx]
                boots.append(statistics.mean(bs1) - statistics.mean(bs2))
            boots.sort()
            lo = boots[int(ALPHA / 2 * B)]
            hi = boots[int((1 - ALPHA / 2) * B)]
            tost_pass = (lo > -TOST_MARGIN) and (hi < TOST_MARGIN)
            tost_rows.append({
                "pair": f"{s1}-{s2}", "tau": t, "r1": r1, "r2": r2, "diff": diff,
                "ci_lo": lo, "ci_hi": hi, "tost_pass": int(tost_pass)
            })
    with open(os.path.join(OUT_DIR, "p7_iter183_tost.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["pair", "tau", "r1", "r2", "diff", "ci_lo", "ci_hi", "tost_pass"])
        for r in tost_rows:
            w.writerow([r["pair"], r["tau"], f"{r['r1']:.4f}", f"{r['r2']:.4f}",
                        f"{r['diff']:.4f}", f"{r['ci_lo']:.4f}", f"{r['ci_hi']:.4f}",
                        r["tost_pass"]])
    print(f"wrote tost.tsv ({len(tost_rows)} rows = {len(pairs)} pairs × {len(TAU_GRID)} taus)")

    # Spearman rank correlation of per-step fire vector across seed pairs
    # We pick tau=0.65 as the canonical break-point (mid-grid)
    spearman_rows = []
    for s1, s2 in pairs:
        v1 = fire_matrix[s1][0.65]
        v2 = fire_matrix[s2][0.65]
        rho = spearman(v1, v2)
        spearman_rows.append({"pair": f"{s1}-{s2}", "tau": 0.65, "spearman_rho": rho})
    # Also include mean over all taus
    spearman_grid_rows = []
    for t in TAU_GRID:
        for s1, s2 in pairs:
            v1 = fire_matrix[s1][t]
            v2 = fire_matrix[s2][t]
            rho = spearman(v1, v2)
            spearman_grid_rows.append({"tau": t, "pair": f"{s1}-{s2}", "spearman_rho": rho})
    with open(os.path.join(OUT_DIR, "p7_iter183_spearman.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["tau", "pair", "spearman_rho"])
        for r in spearman_grid_rows:
            w.writerow([r["tau"], r["pair"], f"{r['spearman_rho']:.4f}"])
    mean_spear = statistics.mean([r["spearman_rho"] for r in spearman_rows])
    print(f"wrote spearman.tsv ({len(spearman_grid_rows)} rows); mean rho at tau=0.65: {mean_spear:.4f}")

    # Monotonicity test: per-seed, is firing rate monotone non-increasing in tau?
    monotone_pass = 0
    monotone_total = 0
    for s in SEEDS:
        rates_s = [statistics.mean(fire_matrix[s][t]) for t in TAU_GRID]
        ok = all(rates_s[i] >= rates_s[i + 1] for i in range(len(rates_s) - 1))
        monotone_total += 1
        monotone_pass += int(ok)

    # Hypothesis evaluation
    h1 = next(r for r in cross_seed if r["tau"] == 0.55)
    h1_pass = h1["mean_rate"] > RATE_HIGH_BAR and h1["cross_ci_lo"] > 0.40
    h2 = next(r for r in cross_seed if r["tau"] == 0.80)
    h2_pass = h2["mean_rate"] < RATE_LOW_BAR and h2["cross_ci_hi"] < 0.20
    h3_pass = monotone_pass == monotone_total
    h4 = next(r for r in cross_seed if r["tau"] == 0.70)
    h4_pass = h4["cross_sd"] < CROSS_SD_BAR
    # H5: TOST-equivalence on all 10 pairs at tau=0.65
    h5_rows = [r for r in tost_rows if abs(r["tau"] - 0.65) < 1e-9]
    h5_pass = all(r["tost_pass"] == 1 for r in h5_rows)
    h6_pass = mean_spear > SPEARMAN_BAR

    summary = {
        "n_seeds": len(SEEDS),
        "n_steps_per_seed": n_steps,
        "n_obs": len(per_obs),
        "tau_grid": TAU_GRID,
        "B": B,
        "seed": SEED,
        "alpha": ALPHA,
        "tost_margin": TOST_MARGIN,
        "spearman_bar": SPEARMAN_BAR,
        "cross_sd_bar": CROSS_SD_BAR,
        "h1_pass_rate_high_at_0_55": h1_pass,
        "h1_mean_rate_at_0_55": h1["mean_rate"],
        "h1_ci_lo_at_0_55": h1["cross_ci_lo"],
        "h2_pass_rate_low_at_0_80": h2_pass,
        "h2_mean_rate_at_0_80": h2["mean_rate"],
        "h2_ci_hi_at_0_80": h2["cross_ci_hi"],
        "h3_pass_monotone_in_tau": h3_pass,
        "h3_monotone_seeds": f"{monotone_pass}/{monotone_total}",
        "h4_pass_cross_sd_at_0_70": h4_pass,
        "h4_cross_sd_at_0_70": h4["cross_sd"],
        "h5_pass_tost_all_pairs_at_0_65": h5_pass,
        "h5_n_pairs_pass": sum(r["tost_pass"] for r in h5_rows),
        "h5_n_pairs_total": len(h5_rows),
        "h6_pass_mean_spearman_above_bar": h6_pass,
        "h6_mean_spearman_at_0_65": mean_spear,
        "cross_seed_table": cross_seed,
        "per_obs_n": len(per_obs),
        "per_seed_rate_n": len(per_seed_rate),
        "tost_n": len(tost_rows),
        "spearman_grid_n": len(spearman_grid_rows),
        "n_pass": sum([h1_pass, h2_pass, h3_pass, h4_pass, h5_pass, h6_pass]),
        "n_total": 6,
    }
    with open(os.path.join(OUT_DIR, "p7_iter183_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote summary.json — H1-H6 verdicts: "
          f"{int(h1_pass)}{int(h2_pass)}{int(h3_pass)}{int(h4_pass)}{int(h5_pass)}{int(h6_pass)} "
          f"({summary['n_pass']}/6 PASS)")
    print(f"  H1 mean_rate(0.55) = {h1['mean_rate']:.4f} [CI {h1['cross_ci_lo']:.4f}, {h1['cross_ci_hi']:.4f}]")
    print(f"  H2 mean_rate(0.80) = {h2['mean_rate']:.4f} [CI {h2['cross_ci_lo']:.4f}, {h2['cross_ci_hi']:.4f}]")
    print(f"  H3 monotone seeds = {monotone_pass}/{monotone_total}")
    print(f"  H4 cross_sd(0.70) = {h4['cross_sd']:.4f}")
    print(f"  H5 TOST pass at 0.65 = {sum(r['tost_pass'] for r in h5_rows)}/{len(h5_rows)} pairs")
    print(f"  H6 mean Spearman rho(0.65) = {mean_spear:.4f}")


if __name__ == "__main__":
    main()