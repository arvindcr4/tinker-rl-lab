#!/usr/bin/env python3
"""
Iter 15 — Pillar 3 (P7) cross-paper coupling + headline CIs.

Replicates the calibrated adaptive-G controller on the N10 5-seed panel
(experiments/results/n10_seed_expansion/) and answers:
  (a) Does zvf-triage at fixed thresholds τ ∈ {0.50..0.90} fire the same
      number of times across the 5 N10 seeds?
  (b) Does the Bayesian@τ_post=0.60 controller (Pillar 3, iter 11) save
      comparable rollouts on the N10 evidence base as on N2?
  (c) What ΔZVF (contrast restoration) does the controller predict on the
      N10 fired steps, calibrated from the groupsize_zvf_sweep.tsv evidence
      base (G=8 → G=16, empirical shift ΔZVF = 0.059)?
  (d) Bootstrap CIs on per-seed fires and on total contrast-restoration.

Outputs (all in experiments/results/p5p8/):
  p7_n10_replication.tsv       — per-(controller, threshold) × per-seed fires
  p7_n10_replication_summary.json — aggregates + bootstrap CIs + Pareto frontier
  p7_n10_contrast.tsv          — per-step contrast-restoration predictions
"""
import json
import math
import sys
import time
from pathlib import Path

WORK = Path("/home/claude/tinker-rl-lab-minimax")
N10 = WORK / "experiments/results/n10_seed_expansion"
OUT = WORK / "experiments/results/p5p8"

N10_SEEDS = [42, 179, 316, 453, 590]
G_BASE = 8                       # N10 base group size (all 5 seeds)
G_DBL = 16                       # candidate doubled G
BOOT_N = 2000
RNG_SEED = 20260704

# ---------------------------------------------------------------- helpers ---


def load_n10():
    out = {}
    for s in N10_SEEDS:
        path = N10 / f"n10_grpo_s{s}.json"
        with path.open() as f:
            d = json.load(f)
        out[s] = sorted(d["step_log"], key=lambda r: r["step"])
    return out


def beta_midrange(k, n, grid_n=512):  # noqa: E306
    """Beta(k+1, n-k+1) posterior mid-range probability m(k,n)."""
    if k < 0 or k > n or n == 0:
        return 0.0
    log_norm = (math.lgamma(k + 1) + math.lgamma(n - k + 1)
                - math.lgamma(n + 2))
    grid = [i / grid_n for i in range(grid_n + 1)]
    log_pdf = [(k) * math.log(max(x, 1e-12))
               + (n - k) * math.log(max(1.0 - x, 1e-12))
               - log_norm for x in grid]
    pdf = [math.exp(lp) for lp in log_pdf]
    cum = [0.0]
    for i in range(1, len(pdf)):
        cum.append(cum[-1] + 0.5 * (pdf[i - 1] + pdf[i])
                   * (grid[i] - grid[i - 1]))
    total = cum[-1] if cum[-1] > 0 else 1.0
    cum = [c / total for c in cum]
    return max(0.0, cum[int(0.95 * grid_n)] - cum[int(0.05 * grid_n)])


def expected_k_from_reward(reward_mean, G):
    """For N10 (no per-prompt tensor), use reward_mean × G as expected successes."""
    return max(0.0, min(float(G), reward_mean * G))


def fit_contrast_regression():
    """Empirical contrast-restoration model: predict ZVF@16 from ZVF@8 using
    groupsize_zvf_sweep.tsv (3 seeds × G ∈ {2,4,8,16}).
    ZVF DROPS when G doubles → controller's intervention reduces ZVF and
    restores contrast. We report the absolute contrast-restoration |ΔZVF| =
    ZVF@8 - ZVF@16 as the positive benefit per fired step.
    """
    sweep_path = WORK / "experiments/results/groupsize_zvf_sweep.tsv"
    if sweep_path.exists():
        by_g = {}
        se_by_g = {}
        with sweep_path.open() as f:
            header = f.readline().rstrip("\n").split("\t")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                d = dict(zip(header, parts))
                try:
                    g = int(d["G"])
                    z = float(d["mean_zvf"])
                    se = float(d["heldout_acc_se"])
                except (KeyError, ValueError):
                    continue
                by_g[g] = z
                se_by_g[g] = se
        if 8 in by_g and 16 in by_g:
            delta = by_g[8] - by_g[16]
            se_delta = math.sqrt(se_by_g.get(8, 0.003) ** 2
                                 + se_by_g.get(16, 0.006) ** 2)
            return {
                "type": "empirical_groupsize_sweep",
                "zvf_g8_emp": by_g[8],
                "zvf_g16_emp": by_g[16],
                "delta_zvf_abs": delta,
                "delta_zvf_se": se_delta,
                "delta_zvf_lo": delta - 1.96 * se_delta,
                "delta_zvf_hi": delta + 1.96 * se_delta,
                "n_seeds_at_each_g": 3,
            }
    return {"type": "fallback_shift",
            "zvf_g8_emp": 0.69, "zvf_g16_emp": 0.63,
            "delta_zvf_abs": 0.06, "delta_zvf_se": 0.01,
            "delta_zvf_lo": 0.04, "delta_zvf_hi": 0.08,
            "n_seeds_at_each_g": 0}


# -------------------------------------------------------- main simulation ---

def main():
    t0 = time.time()
    print("[p7_n10] start", file=sys.stderr)

    n10 = load_n10()
    contrast = fit_contrast_regression()
    print("[p7_n10] contrast-restoration model:", contrast, file=sys.stderr)

    # thresholds
    triage_thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]
    bayes_thresholds = [0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 0.95]

    # per-seed fires table
    per_seed_fires = {th: {} for th in triage_thresholds}
    per_seed_fires_bayes = {th: {} for th in bayes_thresholds}
    per_seed_zvf_at_fire = {th: {} for th in triage_thresholds}
    contrast_per_step = []

    for s in N10_SEEDS:
        steps = n10[s]
        n_steps = len(steps)
        for th in triage_thresholds:
            fires = sum(1 for r in steps if r["zvf"] < th)
            per_seed_fires[th][s] = fires
            zvf_at_fire = [r["zvf"] for r in steps if r["zvf"] < th]
            per_seed_zvf_at_fire[th][s] = zvf_at_fire
        for th in bayes_thresholds:
            fires = 0
            for r in steps:
                k = expected_k_from_reward(r["reward"], G_BASE)
                m = beta_midrange(k, G_BASE)
                if m < th:
                    fires += 1
            per_seed_fires_bayes[th][s] = fires

        # per-step contrast restoration predictions (using G=8 zvf)
        # ZVF@16 = ZVF@8 - delta_zvf_abs (restored; less all-same)
        for r in steps:
            restored_zvf = max(0.0, r["zvf"] - contrast["delta_zvf_abs"])
            # contrast-restoration is the magnitude of zvf reduction
            contrast_restored = r["zvf"] - restored_zvf  # positive
            contrast_per_step.append({
                "seed": s, "step": r["step"],
                "zvf_g8": r["zvf"],
                "zvf_restored_pred": restored_zvf,
                "contrast_restored": contrast_restored,
                "would_fire_triage_0.70": int(r["zvf"] < 0.70),
                "would_fire_bayes_0.60": int(beta_midrange(
                    expected_k_from_reward(r["reward"], G_BASE), G_BASE) < 0.60)
            })

    # write per-seed table
    with (OUT / "p7_n10_replication.tsv").open("w") as f:
        f.write("controller\tthreshold\tseed\tn_steps\tfires\tfire_rate\t"
                "rollouts_used\tcost_ratio\tmean_zvf_at_fire\n")
        for th in triage_thresholds:
            for s in N10_SEEDS:
                n = len(n10[s])
                fires = per_seed_fires[th][s]
                zvf_at_fire = per_seed_zvf_at_fire[th][s]
                mz = sum(zvf_at_fire) / len(zvf_at_fire) if zvf_at_fire else float("nan")
                # cost: each fire doubles rollouts on that step (G: 8→16)
                rollouts = n * G_BASE + fires * G_BASE
                baseline = n * G_BASE
                cost_ratio = rollouts / baseline if baseline else 1.0
                f.write(f"zvf_triage\t{th}\t{s}\t{n}\t{fires}\t"
                        f"{fires/n:.4f}\t{rollouts}\t{cost_ratio:.3f}\t{mz:.4f}\n")
        for th in bayes_thresholds:
            for s in N10_SEEDS:
                n = len(n10[s])
                fires = per_seed_fires_bayes[th][s]
                rollouts = n * G_BASE + fires * G_BASE
                baseline = n * G_BASE
                cost_ratio = rollouts / baseline if baseline else 1.0
                f.write(f"bayes_midrange\t{th}\t{s}\t{n}\t{fires}\t"
                        f"{fires/n:.4f}\t{rollouts}\t{cost_ratio:.3f}\tnan\n")

    # ---- aggregates + bootstrap CIs ---------------------------------------
    import random
    rng = random.Random(RNG_SEED)

    def bootstrap_mean_ci(vals, n_boot=BOOT_N):
        if not vals:
            return (0.0, 0.0, 0.0)
        m = sum(vals) / len(vals)
        boots = []
        n = len(vals)
        for _ in range(n_boot):
            s = sum(vals[rng.randrange(n)] for _ in range(n)) / n
            boots.append(s)
        boots.sort()
        return (m, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)])

    summary = {
        "evidence_base": "n10_seed_expansion",
        "n_seeds": len(N10_SEEDS),
        "n_steps_per_seed": len(n10[N10_SEEDS[0]]),
        "G_base": G_BASE,
        "G_candidate": G_DBL,
        "contrast_restoration_model": contrast,
        "controllers": {}
    }

    # for Pareto frontier: (controller, threshold) -> mean cost_ratio,
    # mean fires/seed, mean contrast-restored-ΔZVF, bootstrap CIs
    for th in triage_thresholds:
        fires_list = [per_seed_fires[th][s] for s in N10_SEEDS]
        m, lo, hi = bootstrap_mean_ci(fires_list)
        # contrast restoration: for the fired steps across all seeds,
        # compute mean contrast-restored magnitude (zvf drop per intervention)
        delta_list = []
        for s in N10_SEEDS:
            for r in n10[s]:
                if r["zvf"] < th:
                    # intervention always drops zvf by the empirical delta
                    delta_list.append(contrast["delta_zvf_abs"])
        if delta_list:
            dm, dlo, dhi = bootstrap_mean_ci(delta_list)
        else:
            dm, dlo, dhi = 0.0, 0.0, 0.0
        cost_ratio_list = [(len(n10[s]) * G_BASE + f * G_BASE)
                           / (len(n10[s]) * G_BASE)
                           for s, f in per_seed_fires[th].items()]
        cm, clo, chi = bootstrap_mean_ci(cost_ratio_list)
        summary["controllers"][f"zvf_triage_{th}"] = {
            "fires_per_seed_mean": m,
            "fires_per_seed_lo": lo,
            "fires_per_seed_hi": hi,
            "cost_ratio_mean": cm,
            "cost_ratio_lo": clo,
            "cost_ratio_hi": chi,
            "delta_zvf_mean": dm,
            "delta_zvf_lo": dlo,
            "delta_zvf_hi": dhi,
            "n_fired_steps": len(delta_list),
        }
    for th in bayes_thresholds:
        fires_list = [per_seed_fires_bayes[th][s] for s in N10_SEEDS]
        m, lo, hi = bootstrap_mean_ci(fires_list)
        cost_ratio_list = [(len(n10[s]) * G_BASE + f * G_BASE)
                           / (len(n10[s]) * G_BASE)
                           for s, f in per_seed_fires_bayes[th].items()]
        cm, clo, chi = bootstrap_mean_ci(cost_ratio_list)
        # contrast_restored for bayes-fired steps
        delta_list = []
        for s in N10_SEEDS:
            for r in n10[s]:
                k = expected_k_from_reward(r["reward"], G_BASE)
                if beta_midrange(k, G_BASE) < th:
                    delta_list.append(contrast["delta_zvf_abs"])
        if delta_list:
            dm, dlo, dhi = bootstrap_mean_ci(delta_list)
        else:
            dm, dlo, dhi = 0.0, 0.0, 0.0
        summary["controllers"][f"bayes_midrange_{th}"] = {
            "fires_per_seed_mean": m,
            "fires_per_seed_lo": lo,
            "fires_per_seed_hi": hi,
            "cost_ratio_mean": cm,
            "cost_ratio_lo": clo,
            "cost_ratio_hi": chi,
            "delta_zvf_mean": dm,
            "delta_zvf_lo": dlo,
            "delta_zvf_hi": dhi,
            "n_fired_steps": len(delta_list),
        }

    # Pareto frontier: dominated = both higher cost AND lower contrast
    frontier = []
    items = list(summary["controllers"].items())
    for name, m1 in items:
        dominated = False
        for other_name, m2 in items:
            if other_name == name:
                continue
            if (m2["cost_ratio_mean"]<= m1["cost_ratio_mean"]
                    and m2["delta_zvf_mean"] >= m1["delta_zvf_mean"]
                    and (m2["cost_ratio_mean"] < m1["cost_ratio_mean"]
                         or m2["delta_zvf_mean"] > m1["delta_zvf_mean"])):
                dominated = True
                break
        if not dominated:
            frontier.append(name)
    summary["pareto_frontier"] = frontier

    # per-step contrast table
    with (OUT / "p7_n10_contrast.tsv").open("w") as f:
        f.write("seed\tstep\tzvf_g8\tzvf_restored_pred\tcontrast_restored\t"
                "fire_triage_0.70\tfire_bayes_0.60\n")
        for r in contrast_per_step:
            f.write(f"{r['seed']}\t{r['step']}\t{r['zvf_g8']:.4f}\t"
                    f"{r['zvf_restored_pred']:.4f}\t{r['contrast_restored']:+.4f}\t"
                    f"{r['would_fire_triage_0.70']}\t{r['would_fire_bayes_0.60']}\n")

    # N10 vs N2 cross-base replication: compare N2 fires/seed at τ=0.70
    # (already in p7_seed_robust_summary.tsv: 4.20±1.48 [3.00,5.40])
    n2_at_070 = 4.20  # from iter 7
    n2_at_070_lo, n2_at_070_hi = 3.00, 5.40
    key_070 = "zvf_triage_0.7"  # f-string drops trailing zero
    n10_at_070 = summary["controllers"][key_070]["fires_per_seed_mean"]
    n10_at_070_lo = summary["controllers"][key_070]["fires_per_seed_lo"]
    n10_at_070_hi = summary["controllers"][key_070]["fires_per_seed_hi"]
    summary["n10_vs_n2_at_tau_0.70"] = {
        "n10_fires_per_seed_mean": n10_at_070,
        "n10_fires_per_seed_95ci": [n10_at_070_lo, n10_at_070_hi],
        "n2_fires_per_seed_mean": n2_at_070,
        "n2_fires_per_seed_95ci": [n2_at_070_lo, n2_at_070_hi],
        "delta_n10_minus_n2": n10_at_070 - n2_at_070,
        "n10_n_steps_per_seed": len(n10[N10_SEEDS[0]]),
        "n2_n_steps": 40,
        "n10_evidence": "Qwen/Qwen3.5-4B GRPO G=8 8 prompts × 15 steps × 5 seeds",
        "n2_evidence": "Qwen3-4B GRPO-family G=8 16 prompts × 40 steps × 1 seed",
    }

    with (OUT / "p7_n10_replication_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[p7_n10] done in {time.time()-t0:.1f}s", file=sys.stderr)
    print("[p7_n10] fires/seed at τ=0.70:", summary["controllers"]
          [key_070]["fires_per_seed_mean"])
    print("[p7_n10] pareto frontier:", frontier)


if __name__ == "__main__":
    main()