#!/usr/bin/env python3
"""P7 iter-83: bootstrap CIs on Iso-G@0.90 vs zvf-triage yield-per-1k-extra.

Reads:
  experiments/results/p5p8/p7_iter83_iso_g_per_prompt.tsv
Writes:
  experiments/results/p5p8/p7_iter83_iso_g_boot.tsv
  experiments/results/p5p8/p7_iter83_iso_g_boot.json

Step-level resample with replacement (n_boot=2000, seed=20260705).
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

WORK = Path("/home/claude/tinker-rl-lab-minimax")
OUT = WORK / "experiments/results/p5p8"
G_BASE = 8
N_PROMPTS_PER_STEP = 16
N_BOOT = 2000
RNG_SEED = 20260705

METHODS = ["grpo", "aero", "gift", "areal"]


def main():
    # Load per-prompt rows
    pp_path = OUT / "p7_iter83_iso_g_per_prompt.tsv"
    pp_by_mcs = defaultdict(list)  # (method, controller, step) -> list of (G', delta_y)
    steps_per_method = defaultdict(set)
    with open(pp_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            key = (r["method"], r["controller"], int(r["step"]))
            pp_by_mcs[key].append((int(r["G_prime"]), float(r["delta_yield"])))
            steps_per_method[r["method"]].add(int(r["step"]))

    rng = random.Random(RNG_SEED)

    boot_results = {}
    for m in METHODS:
        steps = sorted(steps_per_method[m])
        n_steps = len(steps)
        baseline = n_steps * N_PROMPTS_PER_STEP * G_BASE
        iso_yields, zvf_yields, ratios = [], [], []
        for _ in range(N_BOOT):
            sample = [steps[rng.randrange(n_steps)] for _ in range(n_steps)]
            iso_dy, iso_g, zvf_dy, zvf_g = 0.0, 0, 0.0, 0
            for ss in sample:
                for gp, dy in pp_by_mcs[(m, "C5_iso_g_0.90", ss)]:
                    iso_dy += dy
                    iso_g += gp
                for gp, dy in pp_by_mcs[(m, "C1_zvf_triage", ss)]:
                    zvf_dy += dy
                    zvf_g += gp
            iso_extra = iso_g - baseline
            zvf_extra = zvf_g - baseline
            iy = iso_dy * 1000.0 / iso_extra if iso_extra > 0 else 0.0
            zy = zvf_dy * 1000.0 / zvf_extra if zvf_extra > 0 else 0.0
            iso_yields.append(iy)
            zvf_yields.append(zy)
            ratios.append(iy / zy if zy > 0 else float("inf"))

        def pct(arr, q):
            s = sorted(arr)
            n = len(s)
            return s[int(q * n)], s[min(n - 1, int((1 - q) * n))]

        boot_results[m] = {
            "n_boot": N_BOOT,
            "n_steps_resampled": n_steps,
            "baseline_rollouts": baseline,
            "iso_g_0.90": {
                "yield_per_1k_mean": sum(iso_yields) / N_BOOT,
                "ci95_lo": pct(iso_yields, 0.025)[0],
                "ci95_hi": pct(iso_yields, 0.025)[1],
            },
            "zvf_triage_0.70": {
                "yield_per_1k_mean": sum(zvf_yields) / N_BOOT,
                "ci95_lo": pct(zvf_yields, 0.025)[0],
                "ci95_hi": pct(zvf_yields, 0.025)[1],
            },
            "ratio_iso_over_zvf": {
                "mean": sum(ratios) / N_BOOT,
                "ci95_lo": pct(ratios, 0.025)[0],
                "ci95_hi": pct(ratios, 0.025)[1],
                "excludes_1.0": pct(ratios, 0.025)[0] > 1.0,
            },
        }
        b = boot_results[m]
        print(f"  {m}: iso={b['iso_g_0.90']['yield_per_1k_mean']:.2f}/1k "
              f"[{b['iso_g_0.90']['ci95_lo']:.2f},{b['iso_g_0.90']['ci95_hi']:.2f}] ; "
              f"zvf={b['zvf_triage_0.70']['yield_per_1k_mean']:.2f}/1k "
              f"[{b['zvf_triage_0.70']['ci95_lo']:.2f},{b['zvf_triage_0.70']['ci95_hi']:.2f}] ; "
              f"ratio={b['ratio_iso_over_zvf']['mean']:.2f}x "
              f"[{b['ratio_iso_over_zvf']['ci95_lo']:.2f}x,{b['ratio_iso_over_zvf']['ci95_hi']:.2f}x] "
              f"excl1.0={b['ratio_iso_over_zvf']['excludes_1.0']}")

    # TSV summary
    tsv_path = OUT / "p7_iter83_iso_g_boot.tsv"
    cols = ["method", "iso_yield_mean", "iso_ci_lo", "iso_ci_hi",
            "zvf_yield_mean", "zvf_ci_lo", "zvf_ci_hi",
            "ratio_mean", "ratio_ci_lo", "ratio_ci_hi", "excludes_1.0"]
    with open(tsv_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for m in METHODS:
            b = boot_results[m]
            f.write("\t".join([
                m,
                f"{b['iso_g_0.90']['yield_per_1k_mean']:.4f}",
                f"{b['iso_g_0.90']['ci95_lo']:.4f}",
                f"{b['iso_g_0.90']['ci95_hi']:.4f}",
                f"{b['zvf_triage_0.70']['yield_per_1k_mean']:.4f}",
                f"{b['zvf_triage_0.70']['ci95_lo']:.4f}",
                f"{b['zvf_triage_0.70']['ci95_hi']:.4f}",
                f"{b['ratio_iso_over_zvf']['mean']:.4f}",
                f"{b['ratio_iso_over_zvf']['ci95_lo']:.4f}",
                f"{b['ratio_iso_over_zvf']['ci95_hi']:.4f}",
                "yes" if b["ratio_iso_over_zvf"]["excludes_1.0"] else "no",
            ]) + "\n")
    print(f"[write] {tsv_path}")

    json_path = OUT / "p7_iter83_iso_g_boot.json"
    with open(json_path, "w") as f:
        json.dump({
            "iter": 83,
            "pillar": "P7",
            "n_boot": N_BOOT,
            "rng_seed": RNG_SEED,
            "method": "step-level resample with replacement",
            "results": boot_results,
        }, f, indent=2)
    print(f"[write] {json_path}")


if __name__ == "__main__":
    main()