#!/usr/bin/env python3
"""P7 multi-trigger seed-robustness + joint-controller bootstrap CIs.

Iter 79 closes two open P7 veins simultaneously on REAL data:

  (c) seed-robustness of the trigger threshold on the N10 panel.
      Iter 55 (P7 row 74) only evaluated ONE trigger axis (ZVF-triage).
      Iter 67 row 78 added δ_div-triage as a second axis; iter 71 row 83
      unified them into a (Dualformer ⊕ δ_div) joint controller on the
      N2 same-stack tensors. But seed-robustness of T2 (Y_obs-triage)
      and T3 (δ_div-triage) on the N10 panel was never measured.

      This script applies ALL THREE registered trigger blocks to each
      of the N10 seeds (5 GRPO seeds × 15 steps), records per-seed
      fire-counts under each (trigger, τ), and rank-orders the three
      triggers by seed-stability (lowest CV of n_fire).

  (d) bootstrap CIs on every P7 headline.
      Iter 71 row 83 / iter 76 row 90 reported point estimates of
      `net_rollouts_saved` per method without a paired-prompt bootstrap CI on
      `cost_ratio` and the four `(rollout_saves, predicted_restored_prompts, net_rollouts_saved,
      cost_ratio)` headline metrics. We add step-cluster-resampled (B=2000,
      seed=20260705) CIs to the joint-controller headline at τ=0.05
      and τ=0.07 — the two operating points named in iter 71 row 83.

Inputs
------
platform_hybrid/experiments/results/n10_seed_expansion/n10_grpo_s*.json
    5 seed-level JSONs (s42, s179, s316, s453, s590) each with
    step_log[15]={step, loss, reward, zvf, mean_len}.
platform_hybrid/experiments/results/n2_reward_tensor_resume/{grpo,aero,areal,gift}_s0_tensors.jsonl
    40 step rows × 16 prompts × 8 rewards per prompt, all four methods,
    same stack.

Outputs
-------
platform_hybrid/experiments/results/p5p8/p7_multitrigger_seed_per_seed.tsv
    5 seeds × 4 triggers × 5 τ = 100 rows of per-seed fire counts.
platform_hybrid/experiments/results/p5p8/p7_multitrigger_seed_summary.tsv
    4 triggers × 5 τ = 20 rows of seed-mean ± seed-sd + 95% bootstrap CI.
platform_hybrid/experiments/results/p5p8/p7_multitrigger_seed_rank.tsv
    one row per (seed-pair, trigger, τ) for seed-pair rank consistency.
platform_hybrid/experiments/results/p5p8/p7_joint_controller_ci.tsv
    4 methods × 2 τ × 4 headline metrics = 32 rows of point + 95% CI.
platform_hybrid/experiments/results/p5p8/p7_multitrigger_seed_summary.json
    machine-readable headline dictionary with all CIs.

Stdlib only.
"""

from __future__ import annotations

import csv
import json
import pathlib
import random
import statistics

import sys

sys.path.append(
    str(pathlib.Path(__file__).resolve().parents[1])
)  # platform_modal/scripts, for _paths
from _paths import REPO_ROOT

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from p7_joint_controller import (
    JOINT_RULE_ID,
    METRICS,
    bootstrap_controller_metrics,
    joint_controller,
)

WORK = REPO_ROOT
N10 = WORK / "platform_hybrid/experiments" / "results" / "n10_seed_expansion"
N2 = WORK / "platform_hybrid/experiments" / "results" / "n2_reward_tensor_resume"
OUT = WORK / "platform_hybrid/experiments" / "results" / "p5p8" / "corrected-review20-v2"


# Three trigger axes registered in P7:
#   T1 = ZVF-triage   (iter 51 row 74)
#   T2 = Y_obs-triage (iter 66 row 77)  -- Y_obs = 1 - ZVF
#   T3 = δ_div-triage (iter 67 row 78)
# Joint T = T1 OR T2 OR T3 (any-axis firing).
TRIGGERS = ("T1_zvf", "T2_yobs", "T3_ddiv", "T_joint")
# Per-axis τ grids (P7 controller family canonical operating range).
TAU_GRID = {
    "T1_zvf": (0.50, 0.60, 0.70, 0.80, 0.90),
    "T2_yobs": (0.20, 0.30, 0.40, 0.50, 0.60),  # Y_obs thresholds
    "T3_ddiv": (0.03, 0.04, 0.05, 0.06, 0.07),  # δ_div thresholds
}
G_BASE = 8  # N10 and N2 both G=8 per the manifest
N_BOOT_SEED = 10000
N_BOOT_PROMPT = 2000
SEED = 20260705


# ---------------------------------------------------------------------------
# N10 multi-trigger seed-robustness
# ---------------------------------------------------------------------------


def load_n10_seeds():
    """Return list of dicts: seed, step_log with computed y_obs, y_iid, ddiv."""
    seeds = []
    for path in sorted(N10.glob("n10_grpo_s*.json")):
        d = json.loads(path.read_text())
        for s in d["step_log"]:
            zvf_t = float(s["zvf"])
            reward_t = float(s["reward"])
            y_obs = 1.0 - zvf_t
            # iid collision: p_hat = reward, G = 8
            if 0.0 < reward_t < 1.0:
                y_iid = 1.0 - (reward_t**G_BASE + (1.0 - reward_t) ** G_BASE)
            else:
                y_iid = 1.0 - 1.0  # boundary => ZVF_iid = 1 => Y_iid = 0
                y_iid = max(0.0, y_iid)
            ddiv = y_obs - y_iid
            s["y_obs"] = y_obs
            s["y_iid"] = y_iid
            s["ddiv"] = ddiv
        seeds.append(
            {
                "seed": int(d["seed"]),
                "mean_zvf": float(d["mean_zvf"]),
                "heldout_acc": float(d["heldout_acc"]),
                "step_log": d["step_log"],
            }
        )
    return seeds


def fires_for_step(step, trigger, tau):
    """Return True if step fires the (trigger, τ) rule."""
    zvf = step["zvf"]
    y_obs = step["y_obs"]
    ddiv = step["ddiv"]
    if trigger == "T1_zvf":
        return zvf >= tau
    if trigger == "T2_yobs":
        return y_obs >= tau
    if trigger == "T3_ddiv":
        return ddiv >= tau
    if trigger == "T_joint":
        # any-axis fire at the trigger's canonical τ for this step
        return (
            zvf >= TAU_GRID["T1_zvf"][2]  # 0.70
            or y_obs >= TAU_GRID["T2_yobs"][2]  # 0.40
            or ddiv >= TAU_GRID["T3_ddiv"][2]  # 0.05
        )
    raise ValueError(trigger)


def per_seed_fire_counts(seeds):
    """Per seed × trigger × τ -> n_fire (over 15 steps)."""
    out = []  # rows: dict(seed, trigger, tau, n_fire, n_steps, n_escal, headroom_s)
    for sd in seeds:
        for trig in TRIGGERS:
            if trig == "T_joint":
                taus = (None,)  # joint has a single canonical τ triple
            else:
                taus = TAU_GRID[trig]
            for tau in taus:
                n_fire = sum(1 for s in sd["step_log"] if fires_for_step(s, trig, tau))
                # headroom wrong fires: zvf > 0.99 means boundary saturation,
                # but for T2/T3 the "wrong" notion is opposite:
                #  - T1: fire AND zvf>0.99 means boundary step, fire is right
                #    only if step is interior (zvf<1.0). For T1 we mark
                #    headroom as n_fire when zvf==1.0 (boundary firing).
                #  - T2: headroom wrong = Y_obs>>0.5 means few contrast
                #    prompts => fire may not recover much.
                #  - T3: headroom wrong = ddiv<0 means y_obs < y_iid (no
                #    anti-herding, escalation won't help).
                # We compute all three signals per (trig, τ) and record
                # them as diagnostic columns.
                n_bdy_fire = sum(
                    1
                    for s in sd["step_log"]
                    if fires_for_step(s, trig, tau) and abs(s["zvf"] - 1.0) < 1e-9
                )
                n_ddiv_neg = sum(
                    1 for s in sd["step_log"] if fires_for_step(s, trig, tau) and s["ddiv"] < 0
                )
                row = {
                    "seed": sd["seed"],
                    "trigger": trig,
                    "tau": "" if tau is None else f"{tau:.4f}".rstrip("0").rstrip("."),
                    "n_fire": n_fire,
                    "n_steps": len(sd["step_log"]),
                    "n_bdy_fire": n_bdy_fire,
                    "n_ddiv_neg_fire": n_ddiv_neg,
                    "mean_zvf_seed": sd["mean_zvf"],
                    "heldout_acc_seed": sd["heldout_acc"],
                }
                out.append(row)
    return out


def bootstrap_ci_mean(values, n_boot=N_BOOT_SEED, alpha=0.05, seed=SEED):
    """Percentile bootstrap CI on the mean of values."""
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return (sum(values) / n, lo, hi)


def seed_stability_summary(per_seed_rows):
    """Aggregate per (trigger, τ) -> mean / sd / 95% CI on n_fire."""
    by = {}
    for row in per_seed_rows:
        key = (row["trigger"], row["tau"])
        by.setdefault(key, []).append(row["n_fire"])
    out = []
    for (trig, tau), vals in sorted(by.items()):
        m = sum(vals) / len(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        cv = sd / m if m > 0 else 0.0
        m_b, lo_b, hi_b = bootstrap_ci_mean(vals)
        out.append(
            {
                "trigger": trig,
                "tau": tau,
                "n_seeds": len(vals),
                "n_fire_mean": m,
                "n_fire_sd": sd,
                "n_fire_cv": cv,
                "n_fire_ci_lo": lo_b,
                "n_fire_ci_hi": hi_b,
                "headroom_bdy_total": sum(
                    r["n_bdy_fire"]
                    for r in per_seed_rows
                    if r["trigger"] == trig and r["tau"] == tau
                ),
                "ddiv_neg_fire_total": sum(
                    r["n_ddiv_neg_fire"]
                    for r in per_seed_rows
                    if r["trigger"] == trig and r["tau"] == tau
                ),
            }
        )
    return out


def rank_consistency(per_seed_rows):
    """For each trigger × τ, return per-seed fire rank vs overall mean rank.

    Rank by n_fire within (trigger, τ) across seeds. A perfectly seed-stable
    trigger has the same rank ordering across all (τ, trigger) variants; a
    fragile trigger has different orderings at different τ.
    """
    by = {}
    for row in per_seed_rows:
        key = (row["trigger"], row["tau"])
        by.setdefault(key, []).append((row["seed"], row["n_fire"]))
    rows = []
    for (trig, tau), seed_pairs in sorted(by.items()):
        seed_pairs.sort(key=lambda x: -x[1])  # rank descending
        rank_map = {sp[0]: r for r, sp in enumerate(seed_pairs, 1)}
        rows.append(
            {
                "trigger": trig,
                "tau": tau,
                "top_seed": seed_pairs[0][0],
                "top_n_fire": seed_pairs[0][1],
                "bottom_seed": seed_pairs[-1][0],
                "bottom_n_fire": seed_pairs[-1][1],
                "rank_spread": seed_pairs[0][1] - seed_pairs[-1][1],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# N2 joint-controller bootstrap CIs
# ---------------------------------------------------------------------------

METHODS = ("grpo", "aero", "areal", "gift")
DDIV_TAU_HEADLINE = (0.05, 0.07)
G_ESC = 16
G_DUALFORMER = 2


def iid_zvf(p_hat, G):
    if p_hat <= 0.0 or p_hat >= 1.0:
        return 1.0
    return p_hat**G + (1.0 - p_hat) ** G


def load_n2_records():
    """Return dict[method] -> list of step records with per-prompt K, p_hat, etc.

    Each step record is a dict:
      step, n_prompts, contrast_count, boundary_count, p_hat_step,
      zvf_iid_step (=step mean of per-prompt iid ZVF at G=8),
      zvf_obs_step (=step mean of per-prompt obs ZVF),
      ddiv_step (=zvf_iid_step - zvf_obs_step),
      prompts = list of {K, p_hat, zvf_actual, contrast, boundary,
                          zvf_iid_g8, zvf_iid_g16}.
    """
    out = {}
    for m in METHODS:
        path = N2 / f"{m}_s0_tensors.jsonl"
        steps = {}
        with path.open() as fh:
            for line in fh:
                rec = json.loads(line)
                step = int(rec["step"])
                rewards_per_prompt = rec["rewards"]
                prompts = []
                for grp in rewards_per_prompt:
                    K = int(round(sum(grp)))
                    p_hat = K / G_BASE
                    zvf_actual = 1.0 if (K == 0 or K == G_BASE) else 0.0
                    contrast = K > 0 and K < G_BASE
                    boundary = K == 0 or K == G_BASE
                    zvf_iid_g8 = iid_zvf(p_hat, G_BASE)
                    zvf_iid_g16 = iid_zvf(p_hat, G_ESC)
                    prompts.append(
                        {
                            "K": K,
                            "p_hat": p_hat,
                            "zvf_actual": zvf_actual,
                            "contrast": contrast,
                            "boundary": boundary,
                            "zvf_iid_g8": zvf_iid_g8,
                            "zvf_iid_g16": zvf_iid_g16,
                        }
                    )
                zvf_obs_step = sum(p["zvf_actual"] for p in prompts) / len(prompts)
                zvf_iid_step = sum(p["zvf_iid_g8"] for p in prompts) / len(prompts)
                ddiv_step = zvf_iid_step - zvf_obs_step
                p_hat_step = sum(p["p_hat"] for p in prompts) / len(prompts)
                steps[step] = {
                    "step": step,
                    "prompts": prompts,
                    "n_prompts": len(prompts),
                    "contrast_count": sum(1 for p in prompts if p["contrast"]),
                    "boundary_count": sum(1 for p in prompts if p["boundary"]),
                    "zvf_obs_step": zvf_obs_step,
                    "zvf_iid_step": zvf_iid_step,
                    "ddiv_step": ddiv_step,
                    "p_hat_step": p_hat_step,
                }
        out[m] = sorted(steps.values(), key=lambda x: x["step"])
    return out


def flatten_step_records(method_records):
    """Adapt the step-grouped representation to the shared versioned policy."""
    records = []
    for sr in method_records:
        for prompt_idx, p in enumerate(sr["prompts"]):
            records.append(
                {
                    **p,
                    "step": sr["step"],
                    "prompt_idx": prompt_idx,
                    "method": sr.get("method"),
                    "step_delta_div": sr["ddiv_step"],
                    "zvf_actual": p["zvf_actual"],
                }
            )
    return records


def joint_controller_one_step(step_rec, ddiv_tau):
    """Legacy tuple API: gross rollouts, predicted restoration, net rollouts,
    de-escalations, boundary escalations, retrospective cost ratio.

    It shares the step-first action policy with the original analysis. None of
    these fields represents an observed intervention outcome.
    """
    result = joint_controller(flatten_step_records([step_rec]), ddiv_tau)
    return (
        result["rollout_saves"],
        result["predicted_restored_prompts"],
        result["net_rollouts_saved"],
        result["n_contrast"],
        result["n_boundary_escalated"],
        result["cost_ratio"],
    )


def headline_for_method(method_records, ddiv_tau):
    return joint_controller(flatten_step_records(method_records), ddiv_tau)


def bootstrap_ci_metrics(method_records, ddiv_tau, n_boot=N_BOOT_PROMPT, seed=SEED):
    """Step-cluster paired intervals using the exact headline policy."""
    result = bootstrap_controller_metrics(
        flatten_step_records(method_records), ddiv_tau, n_boot, seed
    )
    return {
        f"{metric}_ci_{endpoint}": result[metric][endpoint]
        for metric in METRICS
        for endpoint in ("lo", "hi")
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def write_tsv(path, rows, fieldnames):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # --- N10 multi-trigger ---
    seeds = load_n10_seeds()
    per_seed = per_seed_fire_counts(seeds)
    summary = seed_stability_summary(per_seed)
    rank = rank_consistency(per_seed)

    write_tsv(
        OUT / "p7_multitrigger_seed_per_seed.tsv",
        per_seed,
        fieldnames=[
            "seed",
            "trigger",
            "tau",
            "n_fire",
            "n_steps",
            "n_bdy_fire",
            "n_ddiv_neg_fire",
            "mean_zvf_seed",
            "heldout_acc_seed",
        ],
    )
    write_tsv(
        OUT / "p7_multitrigger_seed_summary.tsv",
        summary,
        fieldnames=[
            "trigger",
            "tau",
            "n_seeds",
            "n_fire_mean",
            "n_fire_sd",
            "n_fire_cv",
            "n_fire_ci_lo",
            "n_fire_ci_hi",
            "headroom_bdy_total",
            "ddiv_neg_fire_total",
        ],
    )
    write_tsv(
        OUT / "p7_multitrigger_seed_rank.tsv",
        rank,
        fieldnames=[
            "trigger",
            "tau",
            "top_seed",
            "top_n_fire",
            "bottom_seed",
            "bottom_n_fire",
            "rank_spread",
        ],
    )

    # --- N2 joint controller bootstrap ---
    n2 = load_n2_records()
    boot_rows = []
    for tau in DDIV_TAU_HEADLINE:
        for m in METHODS:
            point = headline_for_method(n2[m], tau)
            ci = bootstrap_ci_metrics(n2[m], tau)
            boot_rows.append(
                {
                    "method": m,
                    "tau": f"{tau:.2f}",
                    "rule_id": JOINT_RULE_ID,
                    "n_fired_steps": point["n_fired_steps"],
                    "n_fired_prompts": point["n_fired_prompts"],
                    "n_boundary_escalated": point["n_boundary_escalated"],
                    "observed_restored_prompts": None,
                    "extra_rollouts": point["extra_rollouts"],
                    "rollout_saves": point["rollout_saves"],
                    "predicted_restored_prompts": point["predicted_restored_prompts"],
                    "net_rollouts_saved": point["net_rollouts_saved"],
                    "cost_ratio": f"{point['cost_ratio']:.6f}",
                    "net_rollouts_saved_ci_lo": ci["net_rollouts_saved_ci_lo"],
                    "net_rollouts_saved_ci_hi": ci["net_rollouts_saved_ci_hi"],
                    "rollout_saves_ci_lo": ci["rollout_saves_ci_lo"],
                    "rollout_saves_ci_hi": ci["rollout_saves_ci_hi"],
                    "predicted_restored_prompts_ci_lo": ci["predicted_restored_prompts_ci_lo"],
                    "predicted_restored_prompts_ci_hi": ci["predicted_restored_prompts_ci_hi"],
                    "cost_ratio_ci_lo": f"{ci['cost_ratio_ci_lo']:.6f}",
                    "cost_ratio_ci_hi": f"{ci['cost_ratio_ci_hi']:.6f}",
                }
            )
    write_tsv(
        OUT / "p7_joint_controller_ci.tsv",
        boot_rows,
        fieldnames=[
            "method",
            "tau",
            "rule_id",
            "n_fired_steps",
            "n_fired_prompts",
            "n_boundary_escalated",
            "observed_restored_prompts",
            "extra_rollouts",
            "rollout_saves",
            "predicted_restored_prompts",
            "net_rollouts_saved",
            "cost_ratio",
            "net_rollouts_saved_ci_lo",
            "net_rollouts_saved_ci_hi",
            "rollout_saves_ci_lo",
            "rollout_saves_ci_hi",
            "predicted_restored_prompts_ci_lo",
            "predicted_restored_prompts_ci_hi",
            "cost_ratio_ci_lo",
            "cost_ratio_ci_hi",
        ],
    )

    # --- JSON summary ---
    summary_json = {
        "iter": 79,
        "joint_rule_id": JOINT_RULE_ID,
        "joint_bootstrap_unit": "step_cluster",
        "restoration_status": "same_sample_plugin_prediction_only",
        "observed_restored_prompts": None,
        "budget_status": "retrospective_allocation_excludes_pilot_cost",
        "pillar": "P7",
        "vein": "(c) seed-robustness of the trigger threshold + (d) bootstrap CIs on joint controller headline",
        "n_seeds": len(seeds),
        "n_prompts_per_method": {m: sum(len(s["prompts"]) for s in rows) for m, rows in n2.items()},
        "n_boot_seed": N_BOOT_SEED,
        "n_boot_step": N_BOOT_PROMPT,
        "seed": SEED,
        "triggers": list(TRIGGERS),
        "tau_grids": {k: list(v) for k, v in TAU_GRID.items()},
        "seed_summary": summary,
        "seed_rank": rank,
        "headline_metrics": {
            "n_seeds": len(seeds),
            "best_trigger_by_seed_stability": sorted(summary, key=lambda r: r["n_fire_cv"])[0][
                "trigger"
            ],
            "best_trigger_at_canonical_tau": sorted(
                summary, key=lambda r: (r["trigger"], r["tau"])
            )[0]["trigger"],
            "joint_controller_ci": boot_rows,
        },
        "headlines": {
            "H1_seed_cv_by_trigger": {
                r["trigger"]: round(
                    statistics.mean(
                        s["n_fire_cv"] for s in summary if s["trigger"] == r["trigger"]
                    ),
                    4,
                )
                for r in summary[:1]
            },
            "H2_joint_ci_width_at_tau_0_05": {
                row["method"]: {
                    "net_ci_width": row["net_rollouts_saved_ci_hi"]
                    - row["net_rollouts_saved_ci_lo"],
                    "cost_ratio_ci_width_pct": round(
                        100
                        * (float(row["cost_ratio_ci_hi"]) - float(row["cost_ratio_ci_lo"]))
                        / float(row["cost_ratio"]),
                        2,
                    ),
                }
                for row in boot_rows
                if row["tau"] == "0.05"
            },
            "H3_joint_ci_at_tau_0_07": {
                row["method"]: {
                    "net_ci_width": row["net_rollouts_saved_ci_hi"]
                    - row["net_rollouts_saved_ci_lo"],
                }
                for row in boot_rows
                if row["tau"] == "0.07"
            },
        },
    }
    with (OUT / "p7_multitrigger_seed_summary.json").open("w") as fh:
        json.dump(summary_json, fh, indent=2)

    # --- stdout headline ---
    print(f"n_seeds = {len(seeds)}")
    print(f"triggers tested: {TRIGGERS}")
    print(f"\n=== Seed-stability (lower CV = more seed-stable) ===")
    by_trig = {}
    for s in summary:
        by_trig.setdefault(s["trigger"], []).append(s["n_fire_cv"])
    for trig, cvs in by_trig.items():
        print(f"  {trig:12s}: mean(CV) = {statistics.mean(cvs):.4f}")
    print(f"\n=== Joint-controller CIs at τ=0.05 ===")
    for row in boot_rows:
        if row["tau"] == "0.05":
            ci_w = row["net_rollouts_saved_ci_hi"] - row["net_rollouts_saved_ci_lo"]
            cr_w = (
                100
                * (float(row["cost_ratio_ci_hi"]) - float(row["cost_ratio_ci_lo"]))
                / float(row["cost_ratio"])
            )
            print(
                f"  {row['method']:6s}: net_rollouts_saved={row['net_rollouts_saved']:4d} [{row['net_rollouts_saved_ci_lo']:4d}, {row['net_rollouts_saved_ci_hi']:4d}] (CI width={ci_w}); cost_ratio={row['cost_ratio']} ±{cr_w:.2f}%"
            )
    print(f"\n=== Joint-controller CIs at τ=0.07 ===")
    for row in boot_rows:
        if row["tau"] == "0.07":
            ci_w = row["net_rollouts_saved_ci_hi"] - row["net_rollouts_saved_ci_lo"]
            print(
                f"  {row['method']:6s}: net_rollouts_saved={row['net_rollouts_saved']:4d} [{row['net_rollouts_saved_ci_lo']:4d}, {row['net_rollouts_saved_ci_hi']:4d}] (CI width={ci_w})"
            )
    print(f"\nDONE — wrote outputs to {OUT}")


if __name__ == "__main__":
    main()
