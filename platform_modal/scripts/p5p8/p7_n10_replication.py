#!/usr/bin/env python3
"""
Iter 15 — Pillar 3 (P7) cross-paper coupling + headline CIs.

Replicates the calibrated adaptive-G controller on the N10 5-seed panel
(platform_hybrid/experiments/results/n10_seed_expansion/) and answers:
  (a) Does zvf-triage at fixed thresholds τ ∈ {0.50..0.90} fire the same
      number of times across the 5 N10 seeds?
  (b) Does the Bayesian@τ_post=0.60 controller (Pillar 3, iter 11) save
      comparable rollouts on the N10 evidence base as on N2?
  (c) What ΔZVF (contrast restoration) does the controller predict on the
      N10 fired steps, calibrated from the groupsize_zvf_sweep.tsv evidence
      base (paired per-seed G=8 → G=16 ZVF changes)?
  (d) Bootstrap CIs on per-seed fires and on total contrast-restoration.

Outputs (all in platform_hybrid/experiments/results/p5p8/):
  p7_n10_replication.tsv       — per-(controller, threshold) × per-seed fires
  p7_n10_replication_summary.json — aggregates + bootstrap CIs + Pareto frontier
  p7_n10_contrast.tsv          — per-step contrast-restoration predictions
"""
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # platform_modal/scripts, for _paths
from _paths import REPO_ROOT
WORK = REPO_ROOT
N10 = WORK / "platform_hybrid/experiments/results/n10_seed_expansion"
OUT = WORK / "platform_hybrid/experiments/results/p5p8/corrected-review20-v2"

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


def beta_midrange(k, n, grid_n=512):
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
    """Require paired per-seed ZVF contrasts; never substitute accuracy SE."""
    import csv
    path = WORK / "platform_hybrid/experiments/results/groupsize_zvf_sweep.tsv"
    if not path.exists():
        raise ValueError("UNAVAILABLE: missing per-seed ZVF calibration")
    groups = {}
    with path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if "seed" not in row:
                raise ValueError("UNAVAILABLE: calibration lacks seed identity")
            g = int(row["G"])
            if g not in (8,16): continue
            key = (row["seed"], g)
            if key in groups: raise ValueError("duplicate seed/group calibration")
            groups[key] = float(row["mean_zvf"])
            if not math.isfinite(groups[key]) or not 0 <= groups[key] <= 1:
                raise ValueError("calibration ZVF must be finite and in [0, 1]")
    seeds = sorted({seed for seed,g in groups if (seed,8) in groups and (seed,16) in groups})
    if len(seeds) < 2:
        raise ValueError("UNAVAILABLE: at least two paired ZVF calibration seeds required")
    import statistics
    differences = [groups[(seed,8)]-groups[(seed,16)] for seed in seeds]
    delta = statistics.mean(differences)
    se = statistics.stdev(differences) / math.sqrt(len(seeds))
    return dict(type="paired_seed_zvf_calibration", zvf_g8_emp=statistics.mean(groups[(seed,8)] for seed in seeds),
                zvf_g16_emp=statistics.mean(groups[(seed,16)] for seed in seeds),
                delta_zvf_abs=delta, delta_zvf_se=se, delta_zvf_lo=delta-1.96*se,
                delta_zvf_hi=delta+1.96*se, n_seeds_at_each_g=len(seeds),
                uncertainty_status="paired_seed_calibration_available", per_seed_differences=differences,
                calibration_seeds=seeds)



TRIAGE_RULE_ID = "zvf_triage_high_tail_ge_v2"


def triage_fires(zvf, threshold):
    return zvf >= threshold


def predicted_zvf_after(zvf, delta):
    """Clamp the predicted outcome, retaining adverse (negative) changes."""
    if not math.isfinite(zvf) or not 0 <= zvf <= 1 or not math.isfinite(delta):
        raise ValueError("invalid ZVF or calibration delta")
    return min(1.0, max(0.0, zvf - delta))


def bootstrap_restoration(n10, calibration, fires, n_boot=None, seed=None):
    """Propagate independent calibration-seed and trajectory-seed uncertainty.

    A single sampled calibration mean applies to every intervention in one
    replicate. Trajectories are resampled as whole seed clusters, preserving
    within-seed dependence. Per-intervention ratios are undefined if no step
    fires; totals per seed and per-step averages remain defined at zero.
    This is a transported model prediction, not a measured treatment effect.
    """
    n_boot = BOOT_N if n_boot is None else n_boot
    seed = RNG_SEED if seed is None else seed
    differences = calibration["per_seed_differences"]
    if len(differences) < 2 or not all(math.isfinite(d) for d in differences):
        raise ValueError("at least two finite paired calibration differences required")
    trajectories = list(n10.values())
    if not trajectories or any(not rows for rows in trajectories) or n_boot < 1:
        raise ValueError("nonempty trajectory seeds and positive bootstrap count required")

    # Cache action decisions, especially the numerical Bayesian integration.
    trajectories = [(len(rows), [row["zvf"] for row in rows if fires(row)])
                    for rows in trajectories]

    def evaluate(sample, delta):
        total, count = 0.0, 0
        n_steps = sum(size for size, _ in sample)
        for _, selected_zvf in sample:
            for zvf in selected_zvf:
                total += zvf - predicted_zvf_after(zvf, delta)
                count += 1
        return {"delta_zvf": total / count if count else None,
                "restoration_total_per_seed": total / len(sample),
                "restoration_per_step": total / n_steps, "n_fired_steps": count}

    point = evaluate(trajectories, sum(differences) / len(differences))
    draws = {key: [] for key in ("delta_zvf", "restoration_total_per_seed", "restoration_per_step")}
    rng = random.Random(seed)
    for _ in range(n_boot):
        delta = sum(differences[rng.randrange(len(differences))]
                    for _ in differences) / len(differences)
        sample = [trajectories[rng.randrange(len(trajectories))] for _ in trajectories]
        result = evaluate(sample, delta)
        for key in draws:
            if result[key] is not None:
                draws[key].append(result[key])
    out = {"n_fired_steps": point["n_fired_steps"],
           "effect_status": "transported_calibration_prediction",
           "uncertainty_status": "independent_calibration_and_trajectory_seed_bootstrap",
           "n_boot": n_boot, "n_valid_effect_boot": len(draws["delta_zvf"]),
           "effect_interval_status": ("unavailable_no_fires" if not draws["delta_zvf"] else
                                      "conditional_on_nonzero_fires" if len(draws["delta_zvf"]) < n_boot else "defined")}
    for key, values in draws.items():
        values.sort()
        out[f"{key}_mean"] = point[key]
        out[f"{key}_lo"] = values[int(0.025 * len(values))] if values else None
        out[f"{key}_hi"] = values[min(len(values)-1, int(0.975 * len(values)))] if values else None
    return out


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
            fires = sum(1 for r in steps if triage_fires(r["zvf"], th))
            per_seed_fires[th][s] = fires
            zvf_at_fire = [r["zvf"] for r in steps if triage_fires(r["zvf"], th)]
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
            restored_zvf = predicted_zvf_after(r["zvf"], contrast["delta_zvf_abs"])
            # contrast-restoration is the magnitude of zvf reduction
            contrast_restored = r["zvf"] - restored_zvf  # positive
            contrast_per_step.append({
                "seed": s, "step": r["step"],
                "zvf_g8": r["zvf"],
                "zvf_restored_pred": restored_zvf,
                "contrast_restored": contrast_restored,
                "would_fire_triage_0.70": int(triage_fires(r["zvf"], 0.70)),
                "would_fire_bayes_0.60": int(beta_midrange(
                    expected_k_from_reward(r["reward"], G_BASE), G_BASE) < 0.60)
            })

    OUT.mkdir(parents=True, exist_ok=True)
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
        "triage_rule_id": TRIAGE_RULE_ID,
        "effect_status": "transported_calibration_prediction_not_observed",
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
        effect = bootstrap_restoration(n10, contrast,
                                       lambda row, threshold=th: triage_fires(row["zvf"], threshold),
                                       n_boot=BOOT_N)
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
            **effect,
        }
    for th in bayes_thresholds:
        fires_list = [per_seed_fires_bayes[th][s] for s in N10_SEEDS]
        m, lo, hi = bootstrap_mean_ci(fires_list)
        cost_ratio_list = [(len(n10[s]) * G_BASE + f * G_BASE)
                           / (len(n10[s]) * G_BASE)
                           for s, f in per_seed_fires_bayes[th].items()]
        cm, clo, chi = bootstrap_mean_ci(cost_ratio_list)
        effect = bootstrap_restoration(
            n10, contrast,
            lambda row, threshold=th: beta_midrange(expected_k_from_reward(row["reward"], G_BASE), G_BASE) < threshold,
            n_boot=BOOT_N)
        summary["controllers"][f"bayes_midrange_{th}"] = {
            "fires_per_seed_mean": m,
            "fires_per_seed_lo": lo,
            "fires_per_seed_hi": hi,
            "cost_ratio_mean": cm,
            "cost_ratio_lo": clo,
            "cost_ratio_hi": chi,
            **effect,
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
                    and m2["restoration_per_step_mean"] >= m1["restoration_per_step_mean"]
                    and (m2["cost_ratio_mean"] < m1["cost_ratio_mean"]
                         or m2["restoration_per_step_mean"] > m1["restoration_per_step_mean"])):
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

    # The legacy scalar N2 reference has neither a replayable rule nor the
    # same horizon. Do not subtract it from the corrected high-tail N10 result.
    key_070 = "zvf_triage_0.7"
    summary["n10_vs_n2_at_tau_0.70"] = {
        "status": "unavailable_requires_rule_and_horizon_matched_n2_replay",
        "n10_fires_per_seed_mean": summary["controllers"][key_070]["fires_per_seed_mean"],
        "n2_fires_per_seed_mean": None, "delta_n10_minus_n2": None,
    }

    with (OUT / "p7_n10_replication_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[p7_n10] done in {time.time()-t0:.1f}s", file=sys.stderr)
    print("[p7_n10] fires/seed at τ=0.70:", summary["controllers"]
          [key_070]["fires_per_seed_mean"])
    print("[p7_n10] pareto frontier:", frontier)


if __name__ == "__main__":
    main()
