#!/usr/bin/env python3
"""
Pillar-7 (P7) Counterfactual Granularity Replay on Real N2 Reward Tensors.

Iter 59: brief vein (a). Replays the adaptive-G controller at three
granularities on the REAL 4-method N2 reward tensors (40 steps x 4 methods
x 16 prompts = 2560 prompt-step decisions, G_actual=8 rollouts each):

  Policy 1 (actual)      : always G=8 (the empirical rollout count).
  Policy 2 (per_step)     : one G per (method, step) from step-level ZVF
                            scalar (the iter-51 unified controller).
  Policy 3 (per_prompt)   : one G per prompt using per-prompt boundary
                            indicator (the iter-51 per-prompt controller).
  Policy 4 (oracle)       : perfect information: G=2 for contrast prompts
                            (zvf_p=0), G=8 for boundary prompts (zvf_p=2).
                            Computes the lower bound on rollouts.

Novel headroom measurement (this iter): for each boundary prompt in
actual G=8, bootstrap-simulate G=16 by drawing 8 extra rollouts with
replacement from the observed 8. Estimate "what contrast would the
controller have RESTORED if it had fired G_ESC on a contrast-starved
prompt". This is the counterfactual headroom the iter-51 controller
left on the table.

Outputs (worktree-relative):
  platform_hybrid/experiments/results/p5p8/p7_cf_granularity_per_step.tsv
  platform_hybrid/experiments/results/p5p8/p7_cf_granularity_per_prompt.tsv
  platform_hybrid/experiments/results/p5p8/p7_cf_granularity_summary.tsv
  platform_hybrid/experiments/results/p5p8/p7_cf_granularity_boot.tsv
  platform_hybrid/experiments/results/p5p8/p7_cf_granularity_summary.json

Stdlib only.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import random
import statistics
from typing import Iterable

ROOT = pathlib.Path("/home/claude/tinker-rl-lab-minimax")
N2_DIR = ROOT / "experiments" / "results" / "n2_reward_tensor_resume"
OUT = ROOT / "experiments" / "results" / "p5p8"
OUT.mkdir(parents=True, exist_ok=True)

# Stack-conditional constants
G_ACTUAL = 8             # what the actual N2 rollouts used
G_ESC = 16               # escalation (would have restored contrast?)
G_DES = 4                # de-escalation (compute-saving)
G_ORACLE_CONTRAST = 2    # oracle minimum for contrast prompts
N_STEPS = 40
N_PROMPTS_PER_STEP = 16
N_BOOT = 2000
RNG_SEED = 59001

# Per-prompt controller rule (matches iter-51): G_p = G_ESC if zvf_p < 0.5
# (contrast), G_p = G_DES if zvf_p >= 0.99 (boundary), G_BASE = G_ACTUAL
# otherwise. Per-step controller uses step-level zvf_step.

# Trigger thresholds calibrated in iter-51
TAU_ESC_STEP = 0.70
TAU_DES_STEP = 0.95


# ============================================================================
# Data loading
# ----------------------------------------------------------------------------

def load_n2_tensors(methods=("grpo", "aero", "gift", "areal")) -> dict:
    """Return dict[method] -> list[step_dict].

    Each step_dict has:
      step, zvf_step (scalar), pcd_step, frac_all_zero, frac_all_one,
      prompts -> list of prompt_dict, each with
        k_actual (sum of rewards), rewards (list of 8 floats), zvf_actual.
    """
    out = {}
    for m in methods:
        path = N2_DIR / f"{m}_s0_tensors.jsonl"
        if not path.exists():
            continue
        with open(path) as fh:
            rows = [json.loads(l) for l in fh]
        steps = []
        for r in rows:
            prompts = []
            for g in r["rewards"]:
                k = int(sum(g))
                # Binary per-prompt boundary indicator: 1 if k=0 or k=G, else 0
                zvf_p = 1.0 if (k == 0 or k == len(g)) else 0.0
                prompts.append({
                    "k_actual": k,
                    "rewards": [float(x) for x in g],
                    "zvf_actual": zvf_p,
                })
            steps.append({
                "step": int(r["step"]),
                "zvf_step": float(r["zvf"]),
                "pcd_step": float(r["pcd"]),
                "frac_all_zero": float(r["frac_all_zero"]),
                "frac_all_one": float(r["frac_all_one"]),
                "prompts": prompts,
            })
        out[m] = steps
    return out


# ============================================================================
# Policy computations
# ----------------------------------------------------------------------------

def per_step_policy(zvf_step: float) -> int:
    """One G per step using step-level ZVF scalar (iter-51 controller)."""
    if zvf_step >= TAU_DES_STEP:
        return G_DES
    elif zvf_step >= TAU_ESC_STEP:
        return G_ESC
    else:
        return G_ACTUAL


def per_prompt_policy(zvf_p: float) -> int:
    """One G per prompt using per-prompt boundary indicator.

    zvf_p is binary in {0, 1}: 0 = contrast (rollouts span 0 and 1),
    1 = boundary (all 0 or all 1).

    Policy: contrast (zvf_p=0) -> escalate (G_ESC=16, hope to amplify
    contrast signal). Boundary (zvf_p=1) -> de-escalate (G_DES=4, no
    benefit from more rollouts since all same; save compute).
    """
    if zvf_p < 0.5:
        return G_ESC
    elif zvf_p >= 0.99:
        return G_DES
    else:
        return G_ACTUAL


def oracle_policy(zvf_p: float) -> int:
    """Oracle: perfect-information minimum rollouts.

    contrast (zvf_p=0) -> G=2 (min that still captures within-group
    contrast). boundary (zvf_p=1) -> G=8 (the actual; cannot restore
    contrast, no point wasting rollouts).
    """
    return G_ORACLE_CONTRAST if zvf_p < 0.5 else G_ACTUAL


# ============================================================================
# Counterfactual contrast-restoration simulation
# ----------------------------------------------------------------------------

def bootstrap_contrast_at_g(rewards: list[float], G_target: int,
                            rng: random.Random) -> float:
    """Estimate contrast probability at G_target by bootstrapping the
    observed G_actual=8 rollouts to size G_target.

    Returns Pr(0 < k < G_target) estimated from n=128 bootstrap samples.

    For boundary prompts (k_actual=0 or k_actual=G): the prompt is
    structurally degenerate — every observed rollout is the same. No
    amount of extra sampling can restore contrast. Return 0 immediately.

    For contrast prompts (0 < k_actual < G): the empirical p_estimate is
    in (0,1), so bootstrap samples can either preserve contrast (likely)
    or hit a boundary (rare). This estimates "how much contrast G_ESC=16
    would have RESTORED for a contrast-starved prompt." For most
    contrast prompts, contrast is preserved at G=16 (with probability
    ~1 - p^16 - (1-p)^16).
    """
    n_obs = len(rewards)
    n_succ = int(sum(rewards))
    # Boundary prompts: structurally degenerate; no G can restore contrast
    if n_succ == 0 or n_succ == n_obs:
        return 0.0
    n_boot_local = 128
    contrast_count = 0
    for _ in range(n_boot_local):
        # Bootstrap the 8 observed 0/1 rewards to size G_target
        sample = [rewards[rng.randrange(n_obs)] for _ in range(G_target)]
        k_new = int(sum(sample))
        if 0 < k_new < G_target:
            contrast_count += 1
    return contrast_count / n_boot_local


# ============================================================================
# Per-step aggregation
# ----------------------------------------------------------------------------

def compute_step_metrics(method: str, step: dict) -> dict:
    """Compute per-step metrics for all four policies + headroom estimate."""
    zvf_step = step["zvf_step"]
    prompts = step["prompts"]

    # Policy 1: actual (always G_ACTUAL=8)
    g_actual_total = G_ACTUAL * len(prompts)
    contrast_actual = sum(1 for p in prompts if p["zvf_actual"] < 0.5)

    # Policy 2: per-step controller
    g_per_step = per_step_policy(zvf_step)
    g_per_step_total = g_per_step * len(prompts)

    # Policy 3: per-prompt controller
    g_per_prompt_list = [per_prompt_policy(p["zvf_actual"]) for p in prompts]
    g_per_prompt_total = sum(g_per_prompt_list)
    contrast_per_prompt = sum(
        1 for p, gp in zip(prompts, g_per_prompt_list)
        if p["zvf_actual"] < 0.5 and gp >= G_ACTUAL  # escalated
    )

    # Policy 4: oracle
    g_oracle_total = sum(oracle_policy(p["zvf_actual"]) for p in prompts)
    contrast_oracle = sum(1 for p in prompts if p["zvf_actual"] < 0.5)

    # Headroom: for boundary prompts in actual, what fraction would
    # G_ESC=16 have RESTORED (bootstrap-simulated)?
    rng = random.Random(RNG_SEED + step["step"] * 31 + hash(method) % 9973)
    boundary_prompts = [p for p in prompts if p["zvf_actual"] >= 0.5]
    contrast_prompts = [p for p in prompts if p["zvf_actual"] < 0.5]

    if boundary_prompts:
        restore_rates_boundary = [
            bootstrap_contrast_at_g(p["rewards"], G_ESC, rng)
            for p in boundary_prompts
        ]
        mean_restore_rate_boundary = sum(restore_rates_boundary) / len(restore_rates_boundary)
        # How many boundary prompts would likely become contrast at G_ESC?
        n_restored_at_esc = sum(1 for r in restore_rates_boundary if r > 0.5)
        restore_rate_at_esc = n_restored_at_esc / len(boundary_prompts)
    else:
        mean_restore_rate_boundary = float("nan")
        restore_rate_at_esc = 0.0

    if contrast_prompts:
        preserve_rates_contrast = [
            bootstrap_contrast_at_g(p["rewards"], G_DES, rng)
            for p in contrast_prompts
        ]
        mean_preserve_rate_des = sum(preserve_rates_contrast) / len(preserve_rates_contrast)
    else:
        mean_preserve_rate_des = float("nan")

    return {
        "method": method,
        "step": step["step"],
        "zvf_step": zvf_step,
        "pcd_step": step["pcd_step"],
        "n_boundary_prompts": len(boundary_prompts),
        "n_contrast_prompts": len(contrast_prompts),
        "g_actual_total": g_actual_total,
        "g_per_step": g_per_step,
        "g_per_step_total": g_per_step_total,
        "g_per_prompt_total": g_per_prompt_total,
        "g_oracle_total": g_oracle_total,
        "savings_per_step": (g_actual_total - g_per_step_total) / g_actual_total,
        "savings_per_prompt": (g_actual_total - g_per_prompt_total) / g_actual_total,
        "regret_per_prompt": (g_per_prompt_total - g_oracle_total) / g_actual_total,
        "contrast_actual": contrast_actual,
        "contrast_per_prompt": contrast_per_prompt,
        "contrast_oracle": contrast_oracle,
        "mean_restore_rate_boundary_esc": mean_restore_rate_boundary,
        "restore_rate_at_esc": restore_rate_at_esc,
        "mean_preserve_rate_contrast_des": mean_preserve_rate_des,
    }


# ============================================================================
# Bootstrap CI
# ----------------------------------------------------------------------------

def bootstrap_ci(values: list[float], n_boot: int = N_BOOT, alpha: float = 0.05,
                 rng: random.Random | None = None) -> tuple[float, float, float]:
    if rng is None:
        rng = random.Random(RNG_SEED)
    n = len(values)
    if n < 2:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_boot):
        s = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return sum(values) / n, lo, hi


# ============================================================================
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    rng = random.Random(RNG_SEED)
    n2 = load_n2_tensors()
    methods = sorted(n2.keys())

    # 1) Per-step metrics for all (method, step)
    per_step_rows = []
    for m in methods:
        for step in n2[m]:
            row = compute_step_metrics(m, step)
            per_step_rows.append(row)

    # 2) Aggregate per-method summary with bootstrap CI on savings_per_prompt,
    #    regret_per_prompt, restore_rate_at_esc
    summary_rows = []
    for m in methods:
        rows = [r for r in per_step_rows if r["method"] == m]
        savings_pp = [r["savings_per_prompt"] for r in rows]
        regret_pp = [r["regret_per_prompt"] for r in rows]
        restore_rates = [r["restore_rate_at_esc"] for r in rows]
        mean_restore_rate = sum(restore_rates) / len(restore_rates)
        total_actual = sum(r["g_actual_total"] for r in rows)
        total_per_step = sum(r["g_per_step_total"] for r in rows)
        total_per_prompt = sum(r["g_per_prompt_total"] for r in rows)
        total_oracle = sum(r["g_oracle_total"] for r in rows)
        # Aggregate contrast counts
        n_boundary = sum(r["n_boundary_prompts"] for r in rows)
        n_contrast = sum(r["n_contrast_prompts"] for r in rows)
        n_restored_at_esc = sum(
            int(r["restore_rate_at_esc"] * r["n_boundary_prompts"])
            for r in rows
        )
        m_savings_pp, m_savings_pp_lo, m_savings_pp_hi = bootstrap_ci(
            savings_pp, n_boot=N_BOOT, rng=rng)
        m_regret_pp, m_regret_pp_lo, m_regret_pp_hi = bootstrap_ci(
            regret_pp, n_boot=N_BOOT, rng=rng)
        m_restore, m_restore_lo, m_restore_hi = bootstrap_ci(
            restore_rates, n_boot=N_BOOT, rng=rng)
        summary_rows.append({
            "method": m,
            "n_steps": len(rows),
            "total_rollouts_actual": total_actual,
            "total_rollouts_per_step": total_per_step,
            "total_rollouts_per_prompt": total_per_prompt,
            "total_rollouts_oracle": total_oracle,
            "savings_per_step_pct": round(
                (total_actual - total_per_step) / total_actual * 100, 2),
            "savings_per_prompt_pct": round(
                (total_actual - total_per_prompt) / total_actual * 100, 2),
            "regret_per_prompt_pct": round(
                (total_per_prompt - total_oracle) / total_actual * 100, 2),
            "savings_per_prompt_mean": round(m_savings_pp, 4),
            "savings_per_prompt_ci_lo": round(m_savings_pp_lo, 4),
            "savings_per_prompt_ci_hi": round(m_savings_pp_hi, 4),
            "regret_per_prompt_mean": round(m_regret_pp, 4),
            "regret_per_prompt_ci_lo": round(m_regret_pp_lo, 4),
            "regret_per_prompt_ci_hi": round(m_regret_pp_hi, 4),
            "n_boundary_prompts_total": n_boundary,
            "n_contrast_prompts_total": n_contrast,
            "n_restored_at_esc": n_restored_at_esc,
            "restore_rate_at_esc_mean": round(m_restore, 4),
            "restore_rate_at_esc_ci_lo": round(m_restore_lo, 4),
            "restore_rate_at_esc_ci_hi": round(m_restore_hi, 4),
            "headroom_dollar_per_step": round(mean_restore_rate, 4),
        })

    # 3) Overall (pooled across methods) headline
    pooled_savings = [r["savings_per_prompt"] for r in per_step_rows]
    pooled_regret = [r["regret_per_prompt"] for r in per_step_rows]
    pooled_restore = [r["restore_rate_at_esc"] for r in per_step_rows]
    pooled_mean_sav, pooled_sav_lo, pooled_sav_hi = bootstrap_ci(
        pooled_savings, n_boot=N_BOOT, rng=rng)
    pooled_mean_reg, pooled_reg_lo, pooled_reg_hi = bootstrap_ci(
        pooled_regret, n_boot=N_BOOT, rng=rng)
    pooled_mean_rest, pooled_rest_lo, pooled_rest_hi = bootstrap_ci(
        pooled_restore, n_boot=N_BOOT, rng=rng)
    total_actual_all = sum(r["total_rollouts_actual"] for r in summary_rows)
    total_pp_all = sum(r["total_rollouts_per_prompt"] for r in summary_rows)
    total_oracle_all = sum(r["total_rollouts_oracle"] for r in summary_rows)
    total_boundary_all = sum(r["n_boundary_prompts_total"] for r in summary_rows)
    total_restored_all = sum(r["n_restored_at_esc"] for r in summary_rows)

    pooled_row = {
        "method": "pooled",
        "n_steps": len(per_step_rows),
        "total_rollouts_actual": total_actual_all,
        "total_rollouts_per_step": sum(r["total_rollouts_per_step"] for r in summary_rows),
        "total_rollouts_per_prompt": total_pp_all,
        "total_rollouts_oracle": total_oracle_all,
        "savings_per_step_pct": round(
            (total_actual_all - sum(r["total_rollouts_per_step"] for r in summary_rows))
            / total_actual_all * 100, 2),
        "savings_per_prompt_pct": round(
            (total_actual_all - total_pp_all) / total_actual_all * 100, 2),
        "regret_per_prompt_pct": round(
            (total_pp_all - total_oracle_all) / total_actual_all * 100, 2),
        "savings_per_prompt_mean": round(pooled_mean_sav, 4),
        "savings_per_prompt_ci_lo": round(pooled_sav_lo, 4),
        "savings_per_prompt_ci_hi": round(pooled_sav_hi, 4),
        "regret_per_prompt_mean": round(pooled_mean_reg, 4),
        "regret_per_prompt_ci_lo": round(pooled_reg_lo, 4),
        "regret_per_prompt_ci_hi": round(pooled_reg_hi, 4),
        "n_boundary_prompts_total": total_boundary_all,
        "n_contrast_prompts_total": sum(r["n_contrast_prompts_total"] for r in summary_rows),
        "n_restored_at_esc": total_restored_all,
        "restore_rate_at_esc_mean": round(pooled_mean_rest, 4),
        "restore_rate_at_esc_ci_lo": round(pooled_rest_lo, 4),
        "restore_rate_at_esc_ci_hi": round(pooled_rest_hi, 4),
        "headroom_dollar_per_step": round(pooled_mean_rest, 4),
    }
    summary_rows.append(pooled_row)

    # 4) Bootstrap comparison: per-prompt vs oracle (paired test) at
    #    step level — are they statistically distinguishable?
    boot_rows = []
    for m in methods:
        rows = [r for r in per_step_rows if r["method"] == m]
        obs_diff = [r["g_per_prompt_total"] - r["g_oracle_total"] for r in rows]
        obs_mean = sum(obs_diff) / len(obs_diff)
        # Paired bootstrap on the difference
        n = len(obs_diff)
        diffs_boot = []
        for _ in range(N_BOOT):
            sample = [obs_diff[rng.randrange(n)] for _ in range(n)]
            diffs_boot.append(sum(sample) / n)
        diffs_boot.sort()
        diff_lo = diffs_boot[int(0.025 * N_BOOT)]
        diff_hi = diffs_boot[int(0.975 * N_BOOT) - 1]
        boot_rows.append({
            "method": m,
            "metric": "per_prompt_minus_oracle_rollouts",
            "obs_mean": round(obs_mean, 2),
            "boot_ci_lo": round(diff_lo, 2),
            "boot_ci_hi": round(diff_hi, 2),
            "ci_excludes_zero": "yes" if (diff_lo > 0 or diff_hi < 0) else "no",
        })

    # 5) Write outputs
    def write_tsv(path, rows, fields):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})

    per_step_fields = [
        "method", "step", "zvf_step", "pcd_step",
        "n_boundary_prompts", "n_contrast_prompts",
        "g_actual_total", "g_per_step", "g_per_step_total",
        "g_per_prompt_total", "g_oracle_total",
        "savings_per_step", "savings_per_prompt", "regret_per_prompt",
        "contrast_actual", "contrast_per_prompt", "contrast_oracle",
        "mean_restore_rate_boundary_esc", "restore_rate_at_esc",
        "mean_preserve_rate_contrast_des",
    ]
    write_tsv(OUT / "p7_cf_granularity_per_step.tsv", per_step_rows, per_step_fields)

    summary_fields = [
        "method", "n_steps",
        "total_rollouts_actual", "total_rollouts_per_step",
        "total_rollouts_per_prompt", "total_rollouts_oracle",
        "savings_per_step_pct", "savings_per_prompt_pct", "regret_per_prompt_pct",
        "savings_per_prompt_mean", "savings_per_prompt_ci_lo", "savings_per_prompt_ci_hi",
        "regret_per_prompt_mean", "regret_per_prompt_ci_lo", "regret_per_prompt_ci_hi",
        "n_boundary_prompts_total", "n_contrast_prompts_total",
        "n_restored_at_esc",
        "restore_rate_at_esc_mean", "restore_rate_at_esc_ci_lo", "restore_rate_at_esc_ci_hi",
        "headroom_dollar_per_step",
    ]
    write_tsv(OUT / "p7_cf_granularity_summary.tsv", summary_rows, summary_fields)

    boot_fields = ["method", "metric", "obs_mean", "boot_ci_lo", "boot_ci_hi", "ci_excludes_zero"]
    write_tsv(OUT / "p7_cf_granularity_boot.tsv", boot_rows, boot_fields)

    # Per-prompt detail: include the per-prompt G decisions + headroom estimate
    per_prompt_rows = []
    rng2 = random.Random(RNG_SEED + 7)
    for m in methods:
        for step in n2[m]:
            for i, p in enumerate(step["prompts"]):
                g_pp = per_prompt_policy(p["zvf_actual"])
                g_or = oracle_policy(p["zvf_actual"])
                restore_esc = bootstrap_contrast_at_g(p["rewards"], G_ESC, rng2)
                restore_des = bootstrap_contrast_at_g(p["rewards"], G_DES, rng2)
                per_prompt_rows.append({
                    "method": m,
                    "step": step["step"],
                    "prompt_idx": i,
                    "k_actual": p["k_actual"],
                    "zvf_actual": p["zvf_actual"],
                    "g_actual": G_ACTUAL,
                    "g_per_prompt": g_pp,
                    "g_oracle": g_or,
                    "restore_at_g_esc": round(restore_esc, 3),
                    "restore_at_g_des": round(restore_des, 3),
                    "is_boundary": 1 if p["zvf_actual"] >= 0.5 else 0,
                    "is_contrast": 1 if p["zvf_actual"] < 0.5 else 0,
                })
    per_prompt_fields = [
        "method", "step", "prompt_idx",
        "k_actual", "zvf_actual", "is_boundary", "is_contrast",
        "g_actual", "g_per_prompt", "g_oracle",
        "restore_at_g_esc", "restore_at_g_des",
    ]
    write_tsv(OUT / "p7_cf_granularity_per_prompt.tsv", per_prompt_rows, per_prompt_fields)

    # JSON summary
    out_json = {
        "iter": 59,
        "panel_n2": {
            "methods": methods,
            "n_steps_per_method": N_STEPS,
            "n_prompts_per_step": N_PROMPTS_PER_STEP,
            "G_actual": G_ACTUAL,
        },
        "policies": {
            "actual": "always G=8 (the empirical N2 rollout count)",
            "per_step": "one G per (method, step) from zvf_step scalar; "
                        f"tau_esc={TAU_ESC_STEP}, tau_des={TAU_DES_STEP}",
            "per_prompt": "one G per prompt using per-prompt boundary "
                          "indicator; contrast->G_ESC, boundary->G_DES",
            "oracle": "G=2 for contrast prompts, G=8 for boundary prompts "
                      "(perfect-information lower bound)",
        },
        "trigger_thresholds": {
            "tau_esc_step": TAU_ESC_STEP,
            "tau_des_step": TAU_DES_STEP,
            "G_ESC": G_ESC,
            "G_DES": G_DES,
        },
        "n_decisions": len(per_prompt_rows),
        "headline": {
            "pooled_savings_per_prompt_pct": pooled_row["savings_per_prompt_pct"],
            "pooled_savings_ci": [
                pooled_row["savings_per_prompt_ci_lo"],
                pooled_row["savings_per_prompt_ci_hi"],
            ],
            "pooled_regret_per_prompt_pct": pooled_row["regret_per_prompt_pct"],
            "pooled_regret_ci": [
                pooled_row["regret_per_prompt_ci_lo"],
                pooled_row["regret_per_prompt_ci_hi"],
            ],
            "pooled_restore_rate_at_esc": pooled_row["restore_rate_at_esc_mean"],
            "pooled_restore_rate_ci": [
                pooled_row["restore_rate_at_esc_ci_lo"],
                pooled_row["restore_rate_at_esc_ci_hi"],
            ],
            "total_boundary_prompts": total_boundary_all,
            "total_restored_at_esc": total_restored_all,
            "restore_rate_aggregate": round(
                total_restored_all / max(1, total_boundary_all), 4),
        },
        "per_method_summary": [
            {k: v for k, v in r.items()} for r in summary_rows
        ],
        "bootstrap_paired_tests": boot_rows,
        "headline_text": (
            f"Counterfactual replay on N2 tensors ({len(per_prompt_rows)} "
            f"prompt-step decisions, 4 methods x 40 steps x 16 prompts). "
            f"Pooled savings per-prompt controller vs always-G=8: "
            f"{pooled_row['savings_per_prompt_pct']:+.2f}% "
            f"CI [{pooled_row['savings_per_prompt_ci_lo']:+.4f}, "
            f"{pooled_row['savings_per_prompt_ci_hi']:+.4f}]. "
            f"Regret vs oracle: "
            f"{pooled_row['regret_per_prompt_pct']:+.2f}% "
            f"CI [{pooled_row['regret_per_prompt_ci_lo']:+.4f}, "
            f"{pooled_row['regret_per_prompt_ci_hi']:+.4f}]. "
            f"Bootstrap-simulated restore rate at G_ESC=16 for boundary "
            f"prompts: {pooled_row['restore_rate_at_esc_mean']:.4f} "
            f"[{pooled_row['restore_rate_at_esc_ci_lo']:.4f}, "
            f"{pooled_row['restore_rate_at_esc_ci_hi']:.4f}] = "
            f"{total_restored_all}/{total_boundary_all} boundary prompts "
            f"would have been RESTORED by escalating to G=16."
        ),
    }
    (OUT / "p7_cf_granularity_summary.json").write_text(
        json.dumps(out_json, indent=2))

    print(out_json["headline_text"])
    print(f"\nWrote:")
    print(f"  per_step      -> {OUT/'p7_cf_granularity_per_step.tsv'} ({len(per_step_rows)} rows)")
    print(f"  per_prompt    -> {OUT/'p7_cf_granularity_per_prompt.tsv'} ({len(per_prompt_rows)} rows)")
    print(f"  summary       -> {OUT/'p7_cf_granularity_summary.tsv'} ({len(summary_rows)} rows)")
    print(f"  boot          -> {OUT/'p7_cf_granularity_boot.tsv'} ({len(boot_rows)} rows)")
    print(f"  json          -> {OUT/'p7_cf_granularity_summary.json'}")


if __name__ == "__main__":
    main()