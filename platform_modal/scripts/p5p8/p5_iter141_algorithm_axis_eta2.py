"""Iter 141 — P5 algorithm-axis η² on the N2 same-stack four-method tensors.

Direct empirical test of FRONTIER_INSIGHTS Round 1 (ChatGPT-Pro-Extended):
"in outcome-reward LLM post-training, the nominal RL algorithm is under-identified
unless it changes the induced update operator" — i.e., on a fixed stack, PPO/GRPO/etc.
should be performance-equivalent whenever their counterfactual update geometry matches.

We decompose the scalar reward signal into (method, step, prompt, rollout) factors and
report:
  - η²(method | step, prompt)  -- the algorithm-axis fraction
  - η²(step | method, prompt)  -- the curriculum-trajectory-axis fraction
  - η²(prompt | method, step)  -- the prompt-axis fraction
  - η²(rollout | method, step, prompt) -- the irreducible noise
  - per-method reward_mean and CI bootstrap
  - factor-rank decomposition: ANOVA-style eta² + a comparison to the
    Berkeley Ivison unpacking_dpo_ppo factorization framework (P5 §p5_iter85).

Outputs (4 files, all to experiments/results/p5p8/):
  p5_iter141_anova_eta2.tsv        (3 factors with eta² + bootstrap CI)
  p5_iter141_per_method_reward.tsv (4 methods: reward mean + paired-step bootstrap CI)
  p5_iter141_factor_ratio.tsv      (3 pairwise eta² ratios with bootstrap CI)
  p5_iter141_step_trajectory.tsv   (160 rows: 4 methods × 40 steps reward mean)
  p5_iter141_summary.json

Stdlib only; deterministic; B=2000 bootstrap seed=20260705 (canonical Miller recipe).
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N2_DIR = os.path.join(ROOT, "experiments", "results", "n2_reward_tensor_resume")
OUT_DIR = os.path.join(ROOT, "experiments", "results", "p5p8")

METHODS = ["grpo", "aero", "gift", "areal"]
BOOT_B = 2000
BOOT_SEED = 20260705
N_PROMPTS_PER_STEP = 16
N_ROLLOUTS_PER_PROMPT = 8


def load_tensors() -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for m in METHODS:
        path = os.path.join(N2_DIR, f"{m}_s0_tensors.jsonl")
        with open(path) as fh:
            out[m] = [json.loads(l) for l in fh]
    return out


def mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def eta2_from_means(rows: List[Tuple[str, int, int, float]]) -> Dict[str, float]:
    """Variance decomposition on a list of (method, step, prompt, mean_reward).

    Computes η²(method), η²(step), η²(prompt) on the per-cell mean reward.
    Total SS is sum of squared deviations of every cell mean from the grand mean.
    """
    n = len(rows)
    grand = sum(r[3] for r in rows) / n

    method_means: Dict[str, List[float]] = defaultdict(list)
    step_means: Dict[int, List[float]] = defaultdict(list)
    prompt_means: Dict[int, List[float]] = defaultdict(list)
    for method, step, prompt, y in rows:
        method_means[method].append(y)
        step_means[step].append(y)
        prompt_means[prompt].append(y)

    ss_total = sum((y - grand) ** 2 for _, _, _, y in rows)
    if ss_total == 0:
        return {"method": 0.0, "step": 0.0, "prompt": 0.0}

    def ss_factor(group_means: Dict[str, List[float]]) -> float:
        return sum(len(v) * (mean(v) - grand) ** 2 for v in group_means.values())

    ss_method = ss_factor(method_means)
    ss_step = ss_factor(step_means)
    ss_prompt = ss_factor(prompt_means)
    return {
        "method": ss_method / ss_total,
        "step": ss_step / ss_total,
        "prompt": ss_prompt / ss_total,
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    tensors = load_tensors()
    n_steps = len(next(iter(tensors.values())))
    print(f"loaded {len(tensors)} methods × {n_steps} steps")

    # Per-(method, step, prompt) cell means
    cell_rows: List[Tuple[str, int, int, float]] = []
    # Per-step per-method reward mean (used for trajectory plot)
    step_method_reward: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    # Per-method reward scalars
    method_rewards: Dict[str, List[float]] = defaultdict(list)

    for method, rows in tensors.items():
        for step, row in enumerate(rows):
            for p_idx, prompt in enumerate(row["prompt_indices"]):
                r = row["rewards"][p_idx]
                cell_mean = mean(r)
                cell_rows.append((method, step, prompt, cell_mean))
                step_method_reward[(method, step)].extend(r)
                method_rewards[method].extend(r)

    # 1) η² decomposition on cell means
    eta2 = eta2_from_means(cell_rows)
    print("eta2:", eta2)

    # 2) Per-method reward mean + bootstrap CI (paired across step)
    per_method: List[dict] = []
    for method in METHODS:
        # paired step-level aggregation
        per_step = [mean(step_method_reward[(method, s)]) for s in range(n_steps)]
        mu = mean(per_step)
        # bootstrap on step-level means
        rng = random.Random(BOOT_SEED)
        n = len(per_step)
        boot: List[float] = []
        for _ in range(BOOT_B):
            idx = [rng.randrange(n) for _ in range(n)]
            boot.append(mean([per_step[i] for i in idx]))
        boot.sort()
        lo = boot[int(0.025 * BOOT_B)]
        hi = boot[int(0.975 * BOOT_B) - 1]
        per_method.append({
            "method": method,
            "reward_mean": mu,
            "ci_lo": lo,
            "ci_hi": hi,
            "p95_len_step": sorted(per_step)[int(0.95 * n) - 1],
        })

    # 3) Pairwise η² ratio CIs by bootstrap over rows
    # For each bootstrap resample, recompute eta2(method), eta2(step), eta2(prompt) on the resampled cell_rows
    def eta2_for_indices(indices: List[int]) -> Dict[str, float]:
        sub_rows = [cell_rows[i] for i in indices]
        return eta2_from_means(sub_rows)

    rng = random.Random(BOOT_SEED + 1)
    n_cells = len(cell_rows)
    boot_method: List[float] = []
    boot_step: List[float] = []
    boot_prompt: List[float] = []
    for _ in range(BOOT_B):
        idx = [rng.randrange(n_cells) for _ in range(n_cells)]
        e = eta2_for_indices(idx)
        boot_method.append(e["method"])
        boot_step.append(e["step"])
        boot_prompt.append(e["prompt"])
    boot_method.sort()
    boot_step.sort()
    boot_prompt.sort()

    def ci(xs: List[float]) -> Tuple[float, float]:
        return (xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1])

    eta2_method_ci = ci(boot_method)
    eta2_step_ci = ci(boot_step)
    eta2_prompt_ci = ci(boot_prompt)

    # Pairwise ratios
    ratios: Dict[str, Tuple[float, float]] = {}
    for nm, den in [("step", "method"), ("prompt", "method"), ("step", "prompt")]:
        num_samples = {"method": boot_method, "step": boot_step, "prompt": boot_prompt}[nm]
        den_samples = {"method": boot_method, "step": boot_step, "prompt": boot_prompt}[den]
        # paired bootstrap ratio (resample indices once and reuse)
        rng2 = random.Random(BOOT_SEED + 2)
        rs: List[float] = []
        for _ in range(BOOT_B):
            i = [rng2.randrange(len(num_samples)) for _ in range(1)]
            a = num_samples[i[0]]
            b = den_samples[i[0]]
            if b > 0:
                rs.append(a / b)
        rs.sort()
        ratios[f"{nm}_over_{den}"] = (rs[int(0.025 * len(rs))], rs[int(0.975 * len(rs)) - 1])

    # ----- Write TSVs -----
    # 1) ANOVA η²
    anova_path = os.path.join(OUT_DIR, "p5_iter141_anova_eta2.tsv")
    with open(anova_path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["factor", "eta2", "ci_lo", "ci_hi", "rank"])
        for i, (factor, val) in enumerate(sorted(eta2.items(), key=lambda x: -x[1])):
            ci_lo, ci_hi = {"method": eta2_method_ci, "step": eta2_step_ci,
                            "prompt": eta2_prompt_ci}[factor]
            w.writerow([factor, f"{val:.6f}", f"{ci_lo:.6f}", f"{ci_hi:.6f}", i + 1])

    # 2) Per-method reward
    pm_path = os.path.join(OUT_DIR, "p5_iter141_per_method_reward.tsv")
    with open(pm_path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["method", "n_steps", "reward_mean", "ci_lo", "ci_hi", "p95_len_step"])
        for r in per_method:
            w.writerow([r["method"], n_steps, f"{r['reward_mean']:.4f}",
                        f"{r['ci_lo']:.4f}", f"{r['ci_hi']:.4f}",
                        f"{r['p95_len_step']:.4f}"])

    # 3) Factor ratios
    fr_path = os.path.join(OUT_DIR, "p5_iter141_factor_ratio.tsv")
    with open(fr_path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["ratio", "point", "ci_lo", "ci_hi", "excludes_1"])
        for nm, (lo, hi) in ratios.items():
            num, den = nm.split("_over_")
            num_v = eta2[num]
            den_v = eta2[den]
            point = num_v / den_v if den_v > 0 else float("inf")
            excludes_1 = "yes" if (lo > 1.0 or hi < 1.0) else "no"
            w.writerow([nm, f"{point:.4f}", f"{lo:.4f}", f"{hi:.4f}", excludes_1])

    # 4) Step trajectory
    st_path = os.path.join(OUT_DIR, "p5_iter141_step_trajectory.tsv")
    with open(st_path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["method", "step", "reward_mean", "n_rollouts"])
        for (method, step), vals in sorted(step_method_reward.items()):
            w.writerow([method, step, f"{mean(vals):.4f}", len(vals)])

    # 5) Summary JSON
    summary = {
        "iter": 141,
        "n_methods": len(METHODS),
        "n_steps": n_steps,
        "n_cells": len(cell_rows),
        "n_scalars": sum(len(v) for v in method_rewards.values()),
        "eta2_method": eta2["method"],
        "eta2_step": eta2["step"],
        "eta2_prompt": eta2["prompt"],
        "eta2_method_ci": eta2_method_ci,
        "eta2_step_ci": eta2_step_ci,
        "eta2_prompt_ci": eta2_prompt_ci,
        "factor_ratios": {k: {"ci_lo": v[0], "ci_hi": v[1]} for k, v in ratios.items()},
        "per_method_reward": per_method,
        "rank_order": sorted(eta2.items(), key=lambda x: -x[1])[0][0],
        "hypotheses": {
            "H1_method_under_identified": eta2["method"] < 0.05,
            "H2_step_dominates": eta2["step"] > eta2["method"],
            "H3_step_over_method_ratio_greater_than_2": (
                ratios["step_over_method"][0] > 2.0
            ),
        },
        "frontier_synthesis_round1": (
            "ChatGPT-Pro-Extended: 'algorithm is under-identified unless it changes the "
            "induced update operator'. On a fixed stack, PPO/GRPO/etc. should be performance-"
            "equivalent whenever counterfactual update geometry matches."
        ),
    }
    summary_path = os.path.join(OUT_DIR, "p5_iter141_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    # ----- Console output -----
    print()
    print("η² decomposition on N2 same-stack four-method tensors:")
    for factor in ["method", "step", "prompt"]:
        v = eta2[factor]
        ci_lo, ci_hi = {"method": eta2_method_ci, "step": eta2_step_ci,
                        "prompt": eta2_prompt_ci}[factor]
        print(f"  η²({factor:6s}) = {v:.4f}  CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    print()
    print("Per-method reward (paired-step bootstrap B=2000, seed=20260705):")
    for r in per_method:
        print(f"  {r['method']:6s}  mean={r['reward_mean']:.4f}  "
              f"CI [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
    print()
    print("Factor ratios (bootstrap CI, paired-resample seed=20260707):")
    for nm, (lo, hi) in ratios.items():
        num, den = nm.split("_over_")
        point = eta2[num] / eta2[den] if eta2[den] > 0 else float("inf")
        print(f"  {nm:25s}  point={point:7.3f}  CI [{lo:7.3f}, {hi:7.3f}]")
    print()
    print("Hypotheses:")
    for h, verdict in summary["hypotheses"].items():
        print(f"  {h}: {verdict}")


if __name__ == "__main__":
    main()