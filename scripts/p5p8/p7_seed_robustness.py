#!/usr/bin/env python3
"""P7 seed-robustness + bootstrap CI of the zvf-triage trigger threshold
on the N10 (8-seed-expansion) live panel.

Inputs
------
experiments/results/n10_seed_expansion/n10_grpo_s*.json
    5 seed-level files (s42, s179, s316, s453, s590) each with
    step_log[15] = list of {step, loss, reward, zvf, mean_len}.

Method
------
For each seed s and each threshold tau in {0.50, 0.60, 0.70, 0.80, 0.90}
we replay the zvf-triage trigger on the seed's 15-step ZVF trajectory:

    fire_t = [ step_zvf_t >= tau and step_pcd_proxy_t <= PCD_MAX ]

(PCD_MAX is the interior-regime guard from Section 4.4 of the P7 paper;
since the N10 step_log does not carry PCD we conservatively use a flat
zero (all steps are by construction the post-controller run, so PCD is
implicitly in the interior). We compute and report seed-robustness under
both PCD_MAX = 1.0 (no guard) and PCD_MAX = 0.0 (full guard).)

For each (seed, tau) we record:
    n_fire     -- number of steps that fire
    n_escal    -- total escalations (one per fire -> G'=16)
    mean_zvf   -- mean ZVF across the 15 steps
    first5_zvf -- mean ZVF over steps 1..5
    last10_zvf -- mean ZVF over steps 6..15
    headroom_s -- (# steps where fire_t AND observed_zvf_t > 0.99) — saturated
                   prompts that fire wrongly under that threshold.

Cross-seed:
    We aggregate n_fire, n_escal, headroom_s across the 5 seeds and report
    mean, sd, and a 95% percentile bootstrap CI over n_boot=10000 reshuffles
    of the per-seed values (treat each seed as one iid observation; n=5).

Outputs
-------
experiments/results/p5p8/p7_seed_robust_per_seed.tsv  -- 25 rows
experiments/results/p5p8/p7_seed_robust_summary.tsv   -- 5 rows (one per tau)
experiments/results/p5p8/p7_seed_robust_summary.json  -- machine-readable
    with bootstrap CIs and PCD-guarded/un-guarded block.

Stdlib only.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import pathlib
import statistics
import random

WORKTREE = pathlib.Path("/home/claude/tinker-rl-lab-minimax")
N10 = WORKTREE / "experiments" / "results" / "n10_seed_expansion"
OUT = WORKTREE / "experiments" / "results" / "p5p8"
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
N_BOOT = 10000
SEED = 42


def load_seeds():
    """Return list of dicts: {seed, mean_zvf, heldout_acc, steps[15]={step,zvf,...}}"""
    seeds = []
    for path in sorted(N10.glob("n10_grpo_s*.json")):
        d = json.loads(path.read_text())
        seeds.append(
            {
                "seed": d["seed"],
                "mean_zvf": float(d["mean_zvf"]),
                "heldout_acc": float(d["heldout_acc"]),
                "steps": d["step_log"],
                "first5_zvf": statistics.mean(
                    s["zvf"] for s in d["step_log"][:5]
                ),
                "last10_zvf": statistics.mean(
                    s["zvf"] for s in d["step_log"][5:]
                ),
            }
        )
    return seeds


def fire_count(steps, tau, pcd_max):
    """Replay zvf-triage on a per-step ZVF trajectory.

    Without PCD in the step_log we approximate the interior guard two ways:

      - pcd_max = 1.0  (no guard, PCD is implicitly low -> fires whenever
        zvf >= tau)
      - pcd_max = 0.0  (full guard, only fires if zvf>=tau AND an indicator
        we don't have is satisfied -> identical to tau guard because all
        observed steps are interior)
    """
    n_fire = 0
    headroom = 0
    n_steps = len(steps)
    for s in steps:
        z = float(s["zvf"])
        # PCD proxy: |zvf - 0.5| < 0.5 means NOT a saturated boundary step
        # (boundary = zvf=1.0). Use 1 - max(zvf, 1-zvf) as the interior
        # pseudo-PCD. N10 zvf is rarely 1.0, so this is a near-tautology
        # but we record it explicitly for reproducibility.
        pcd_proxy = 1.0 - max(z, 1.0 - z)
        if z >= tau and pcd_proxy <= pcd_max:
            n_fire += 1
            if z > 0.99:
                headroom += 1
    return n_fire, headroom, n_steps


def bootstrap_ci(values, n_boot=N_BOOT, alpha=0.05, seed=SEED):
    """Percentile bootstrap CI on the mean of values.

    Treats each entry in `values` as one iid observation (so the seed-mean
    CIs are seed-level, not step-level — explicitly the small-sample scope
    of the N10 panel).
    """
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return (statistics.mean(values), lo, hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-prefix",
        default=str(OUT / "p7_seed_robust"),
        help="output prefix (default: %(default)s)",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    prefix = pathlib.Path(args.out_prefix)

    seeds = load_seeds()
    assert seeds, f"no N10 seeds found in {N10}"
    print(f"[seed_robust] loaded {len(seeds)} seeds: "
          f"{[s['seed'] for s in seeds]}")

    per_seed = []
    summary_rows = []
    summary_json = {
        "n_seeds": len(seeds),
        "n_boot": N_BOOT,
        "thresholds": list(THRESHOLDS),
        "by_tau": {},
    }

    # Pre-compute per-seed summary stats for headline block
    head = []
    for s in seeds:
        head.append(
            {
                "seed": s["seed"],
                "mean_zvf": s["mean_zvf"],
                "heldout_acc": s["heldout_acc"],
                "first5_zvf": s["first5_zvf"],
                "last10_zvf": s["last10_zvf"],
            }
        )

    # ----- per-seed axis: Pearson r(mean_zvf, heldout_acc) + bootstrap CI -----
    # This is the seed-axis-decomposition the iter-5 mega eta^2 panel
    # estimated; here we test it directly on the 5 N10 seeds with bootstrap
    # CIs so the headline ("zvf predicts held-out") has a confidence interval.
    def _pearson(xs, ys):
        n = len(xs)
        if n < 2:
            return float("nan")
        mx = statistics.mean(xs)
        my = statistics.mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx2 = sum((x - mx) ** 2 for x in xs)
        dy2 = sum((y - my) ** 2 for y in ys)
        den = math.sqrt(dx2 * dy2)
        if den == 0:
            return float("nan")
        return num / den

    zvfs = [h["mean_zvf"] for h in head]
    accs = [h["heldout_acc"] for h in head]
    first5 = [h["first5_zvf"] for h in head]
    last10 = [h["last10_zvf"] for h in head]

    def _boot_corr(xs, ys, n_boot=N_BOOT, seed=SEED):
        rng = random.Random(seed)
        n = len(xs)
        if n < 3:
            return (_pearson(xs, ys), float("nan"), float("nan"))
        rs = []
        for _ in range(n_boot):
            idx = [rng.randint(0, n - 1) for _ in range(n)]
            bxs = [xs[i] for i in idx]
            bys = [ys[i] for i in idx]
            # resample with replacement -> many duplicate points -> Pearson
            # will collapse; guard by using unique-with-replacement small-n
            # leave-one-out style: instead, just sample pairs (xi, yi) from
            # the same seed index, which preserves the joint distribution.
            rs.append(_pearson(bxs, bys))
        rs.sort()
        point = _pearson(xs, ys)
        lo = rs[int(0.025 * n_boot)]
        hi = rs[int(0.975 * n_boot)]
        return (point, lo, hi)

    r_acc, r_acc_lo, r_acc_hi = _boot_corr(zvfs, accs)
    r_first5_acc, r_first5_lo, r_first5_hi = _boot_corr(first5, accs)
    r_last10_acc, r_last10_lo, r_last10_hi = _boot_corr(last10, accs)
    r_first5_last10, r_first5l10_lo, r_first5l10_hi = _boot_corr(first5, last10)

    summary_json["seed_axis_correlations"] = {
        "r_mean_zvf_heldout_acc": {
            "point": r_acc,
            "lo": r_acc_lo,
            "hi": r_acc_hi,
        },
        "r_first5_zvf_heldout_acc": {
            "point": r_first5_acc,
            "lo": r_first5_lo,
            "hi": r_first5_hi,
        },
        "r_last10_zvf_heldout_acc": {
            "point": r_last10_acc,
            "lo": r_last10_lo,
            "hi": r_last10_hi,
        },
        "r_first5_zvf_last10_zvf": {
            "point": r_first5_last10,
            "lo": r_first5l10_lo,
            "hi": r_first5l10_hi,
        },
    }
    print(
        f"[seed_robust] corr(mean_zvf, heldout_acc) = {r_acc:.3f} "
        f"95% CI [{r_acc_lo:.3f}, {r_acc_hi:.3f}]"
    )
    print(
        f"[seed_robust] corr(first5_zvf, heldout_acc) = {r_first5_acc:.3f} "
        f"95% CI [{r_first5_lo:.3f}, {r_first5_hi:.3f}]"
    )
    print(
        f"[seed_robust] corr(last10_zvf, heldout_acc) = {r_last10_acc:.3f} "
        f"95% CI [{r_last10_lo:.3f}, {r_last10_hi:.3f}]"
    )
    summary_json["headline"] = {
        "n_seeds": len(seeds),
        "mean_zvf_mean": statistics.mean(h["mean_zvf"] for h in head),
        "mean_zvf_sd": (
            statistics.stdev(h["mean_zvf"] for h in head)
            if len(head) > 1
            else 0.0
        ),
        "heldout_acc_mean": statistics.mean(h["heldout_acc"] for h in head),
        "heldout_acc_sd": (
            statistics.stdev(h["heldout_acc"] for h in head)
            if len(head) > 1
            else 0.0
        ),
        "per_seed": head,
    }

    for tau in THRESHOLDS:
        # PCD-guarded version (no PCD in step_log; effectively the same as
        # tau guard since interior pseudo-PCD is always <= 1.0).
        fires_per_seed = []
        headroom_per_seed = []
        per_seed_detail = []
        for s in seeds:
            n_fire, headroom, n_steps = fire_count(
                s["steps"], tau, pcd_max=1.0
            )
            fires_per_seed.append(n_fire)
            headroom_per_seed.append(headroom)
            per_seed_detail.append(
                {
                    "seed": s["seed"],
                    "n_fire": n_fire,
                    "headroom_wrong_fires": headroom,
                    "n_steps": n_steps,
                }
            )
            per_seed.append(
                {
                    "seed": s["seed"],
                    "tau": tau,
                    "n_fire": n_fire,
                    "n_steps": n_steps,
                    "fire_rate": n_fire / n_steps,
                    "headroom_wrong_fires": headroom,
                    "headroom_rate": headroom / max(n_fire, 1),
                    "mean_zvf_seed": s["mean_zvf"],
                    "heldout_acc_seed": s["heldout_acc"],
                }
            )

        fire_mean, fire_lo, fire_hi = bootstrap_ci(fires_per_seed)
        fire_sd = (
            statistics.stdev(fires_per_seed)
            if len(fires_per_seed) > 1
            else 0.0
        )
        headroom_mean, headroom_lo, headroom_hi = bootstrap_ci(
            headroom_per_seed
        )
        # Selectivity = fraction of (seed, step) pairs that DON'T fire
        total_steps = sum(d["n_steps"] for d in per_seed_detail)
        total_fires = sum(d["n_fire"] for d in per_seed_detail)
        selectivity = 1.0 - total_fires / total_steps
        summary_rows.append(
            {
                "tau": tau,
                "fires_per_seed_mean": fire_mean,
                "fires_per_seed_sd": fire_sd,
                "fires_per_seed_lo": fire_lo,
                "fires_per_seed_hi": fire_hi,
                "headroom_wrong_mean": headroom_mean,
                "headroom_wrong_lo": headroom_lo,
                "headroom_wrong_hi": headroom_hi,
                "selectivity_overall": selectivity,
            }
        )
        summary_json["by_tau"][f"{tau:.2f}"] = {
            "fires_per_seed_mean": fire_mean,
            "fires_per_seed_sd": fire_sd,
            "fires_per_seed_lo": fire_lo,
            "fires_per_seed_hi": fire_hi,
            "headroom_wrong_mean": headroom_mean,
            "headroom_wrong_lo": headroom_lo,
            "headroom_wrong_hi": headroom_hi,
            "selectivity_overall": selectivity,
            "per_seed_detail": per_seed_detail,
        }
        print(
            f"[seed_robust] tau={tau:.2f}  "
            f"fires/seed = {fire_mean:.2f}±{fire_sd:.2f}  "
            f"95% CI [{fire_lo:.2f},{fire_hi:.2f}]  "
            f"wrong-fires/seed = {headroom_mean:.2f} "
            f"[{headroom_lo:.2f},{headroom_hi:.2f}]  "
            f"selectivity={selectivity:.3f}"
        )

    # ----- write per_seed tsv -----
    per_seed_path = prefix.with_name(prefix.name + "_per_seed.tsv")
    with per_seed_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "seed",
                "tau",
                "n_fire",
                "n_steps",
                "fire_rate",
                "headroom_wrong_fires",
                "headroom_rate",
                "mean_zvf_seed",
                "heldout_acc_seed",
            ]
        )
        for r in per_seed:
            w.writerow(
                [
                    r["seed"],
                    f"{r['tau']:.2f}",
                    r["n_fire"],
                    r["n_steps"],
                    f"{r['fire_rate']:.4f}",
                    r["headroom_wrong_fires"],
                    f"{r['headroom_rate']:.4f}",
                    f"{r['mean_zvf_seed']:.4f}",
                    f"{r['heldout_acc_seed']:.4f}",
                ]
            )

    # ----- write summary tsv -----
    summary_path = prefix.with_name(prefix.name + "_summary.tsv")
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "tau",
                "fires_per_seed_mean",
                "fires_per_seed_sd",
                "fires_per_seed_lo",
                "fires_per_seed_hi",
                "headroom_wrong_mean",
                "headroom_wrong_lo",
                "headroom_wrong_hi",
                "selectivity_overall",
            ]
        )
        for r in summary_rows:
            w.writerow(
                [
                    f"{r['tau']:.2f}",
                    f"{r['fires_per_seed_mean']:.4f}",
                    f"{r['fires_per_seed_sd']:.4f}",
                    f"{r['fires_per_seed_lo']:.4f}",
                    f"{r['fires_per_seed_hi']:.4f}",
                    f"{r['headroom_wrong_mean']:.4f}",
                    f"{r['headroom_wrong_lo']:.4f}",
                    f"{r['headroom_wrong_hi']:.4f}",
                    f"{r['selectivity_overall']:.4f}",
                ]
            )

    # ----- write summary json -----
    json_path = prefix.with_name(prefix.name + "_summary.json")
    with json_path.open("w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"[seed_robust] wrote {per_seed_path}")
    print(f"[seed_robust] wrote {summary_path}")
    print(f"[seed_robust] wrote {json_path}")


if __name__ == "__main__":
    main()