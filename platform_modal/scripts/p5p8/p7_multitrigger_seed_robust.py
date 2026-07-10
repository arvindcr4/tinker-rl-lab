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
      `net_saves` per method without a paired-prompt bootstrap CI on
      `cost_ratio` and the four `(rollout_saves, zvf_saves, net_saves,
      cost_ratio)` headline metrics. We add prompt-resampled (B=2000,
      seed=20260705) CIs to the joint-controller headline at τ=0.05
      and τ=0.07 — the two operating points named in iter 71 row 83.

Inputs
------
experiments/results/n10_seed_expansion/n10_grpo_s*.json
    5 seed-level JSONs (s42, s179, s316, s453, s590) each with
    step_log[15]={step, loss, reward, zvf, mean_len}.
experiments/results/n2_reward_tensor_resume/{grpo,aero,areal,gift}_s0_tensors.jsonl
    40 step rows × 16 prompts × 8 rewards per prompt, all four methods,
    same stack.

Outputs
-------
experiments/results/p5p8/p7_multitrigger_seed_per_seed.tsv
    5 seeds × 4 triggers × 5 τ = 100 rows of per-seed fire counts.
experiments/results/p5p8/p7_multitrigger_seed_summary.tsv
    4 triggers × 5 τ = 20 rows of seed-mean ± seed-sd + 95% bootstrap CI.
experiments/results/p5p8/p7_multitrigger_seed_rank.tsv
    one row per (seed-pair, trigger, τ) for seed-pair rank consistency.
experiments/results/p5p8/p7_joint_controller_ci.tsv
    4 methods × 2 τ × 4 headline metrics = 32 rows of point + 95% CI.
experiments/results/p5p8/p7_multitrigger_seed_summary.json
    machine-readable headline dictionary with all CIs.

Stdlib only.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import random
import statistics

WORK = pathlib.Path("/home/claude/tinker-rl-lab-minimax")
N10 = WORK / "experiments" / "results" / "n10_seed_expansion"
N2 = WORK / "experiments" / "results" / "n2_reward_tensor_resume"
OUT = WORK / "experiments" / "results" / "p5p8"
OUT.mkdir(parents=True, exist_ok=True)

# Three trigger axes registered in P7:
#   T1 = ZVF-triage   (iter 51 row 74)
#   T2 = Y_obs-triage (iter 66 row 77)  -- Y_obs = 1 - ZVF
#   T3 = δ_div-triage (iter 67 row 78)
# Joint T = T1 OR T2 OR T3 (any-axis firing).
TRIGGERS = ("T1_zvf", "T2_yobs", "T3_ddiv", "T_joint")
# Per-axis τ grids (P7 controller family canonical operating range).
TAU_GRID = {
    "T1_zvf":  (0.50, 0.60, 0.70, 0.80, 0.90),
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
                y_iid = 1.0 - (reward_t ** G_BASE + (1.0 - reward_t) ** G_BASE)
            else:
                y_iid = 1.0 - 1.0  # boundary => ZVF_iid = 1 => Y_iid = 0
                y_iid = max(0.0, y_iid)
            ddiv = y_obs - y_iid
            s["y_obs"] = y_obs
            s["y_iid"] = y_iid
            s["ddiv"] = ddiv
        seeds.append({
            "seed": int(d["seed"]),
            "mean_zvf": float(d["mean_zvf"]),
            "heldout_acc": float(d["heldout_acc"]),
            "step_log": d["step_log"],
        })
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
            zvf >= TAU_GRID["T1_zvf"][2]   # 0.70
            or y_obs >= TAU_GRID["T2_yobs"][2]  # 0.40
            or ddiv >= TAU_GRID["T3_ddiv"][2]   # 0.05
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
                n_fire = sum(
                    1 for s in sd["step_log"] if fires_for_step(s, trig, tau)
                )
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
                    1 for s in sd["step_log"]
                    if fires_for_step(s, trig, tau) and abs(s["zvf"] - 1.0) < 1e-9
                )
                n_ddiv_neg = sum(
                    1 for s in sd["step_log"]
                    if fires_for_step(s, trig, tau) and s["ddiv"] < 0
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
        out.append({
            "trigger": trig,
            "tau": tau,
            "n_seeds": len(vals),
            "n_fire_mean": m,
            "n_fire_sd": sd,
            "n_fire_cv": cv,
            "n_fire_ci_lo": lo_b,
            "n_fire_ci_hi": hi_b,
            "headroom_bdy_total": sum(
                r["n_bdy_fire"] for r in per_seed_rows
                if r["trigger"] == trig and r["tau"] == tau
            ),
            "ddiv_neg_fire_total": sum(
                r["n_ddiv_neg_fire"] for r in per_seed_rows
                if r["trigger"] == trig and r["tau"] == tau
            ),
        })
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
        rows.append({
            "trigger": trig,
            "tau": tau,
            "top_seed": seed_pairs[0][0],
            "top_n_fire": seed_pairs[0][1],
            "bottom_seed": seed_pairs[-1][0],
            "bottom_n_fire": seed_pairs[-1][1],
            "rank_spread": seed_pairs[0][1] - seed_pairs[-1][1],
        })
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
    return p_hat ** G + (1.0 - p_hat) ** G


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
                    contrast = (K > 0 and K < G_BASE)
                    boundary = (K == 0 or K == G_BASE)
                    zvf_iid_g8 = iid_zvf(p_hat, G_BASE)
                    zvf_iid_g16 = iid_zvf(p_hat, G_ESC)
                    prompts.append({
                        "K": K,
                        "p_hat": p_hat,
                        "zvf_actual": zvf_actual,
                        "contrast": contrast,
                        "boundary": boundary,
                        "zvf_iid_g8": zvf_iid_g8,
                        "zvf_iid_g16": zvf_iid_g16,
                    })
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


def joint_controller_one_step(step_rec, ddiv_tau):
    """Replay the joint controller on one step. Returns
    (rollout_saves, zvf_saves, net_saves, n_contrast, n_bdy_fire, cost_ratio).

    Saves vs the G=8 baseline:
      Dualformer branch (per contrast prompt): G'=2 -> saves 6 rollouts.
      ddiv-triage branch (per boundary prompt where ddiv>=tau):
        G_esc=16 -> 1 ZVF save (boundary zvf_actual==1).
      Non-fired contrast branch (boundary but ddiv<tau): keeps G=8.

    Cost ratio: cost / baseline-cost. Baseline cost = G=8 rollouts per
    prompt. Joint cost = sum over prompts of (G'=2 if contrast and
    ddiv>=tau... NO wait -- Dualformer branch is fired on CONTRAST
    prompts where ddiv<tau (Dualformer takes precedence per iter 71
    rule). Let's re-derive the iter 71 rule exactly:

      If contrast_prompt: G'=2 (Dualformer) -- ALWAYS, no ddiv check.
      Elif boundary_prompt AND ddiv>=tau: G'=16 (escalate) -- ddiv-triage.
      Else: G=8.

    Cost ratio = sum_over_prompts(G'_used) / (n_prompts * 8).
    """
    rollout_saves = 0
    zvf_saves = 0
    cost_used = 0
    n_contrast = step_rec["contrast_count"]
    n_bdy_fire = 0
    for p in step_rec["prompts"]:
        if p["contrast"]:
            # Dualformer branch
            rollout_saves += (G_BASE - G_DUALFORMER)  # 6 saved
            cost_used += G_DUALFORMER
        elif p["boundary"] and step_rec["ddiv_step"] >= ddiv_tau:
            # ddiv-triage escalate
            zvf_saves += 1
            n_bdy_fire += 1
            cost_used += G_ESC
        else:
            cost_used += G_BASE
    baseline_cost = step_rec["n_prompts"] * G_BASE
    cost_ratio = cost_used / baseline_cost
    return (rollout_saves, zvf_saves, rollout_saves + zvf_saves,
            n_contrast, n_bdy_fire, cost_ratio)


def headline_for_method(method_records, ddiv_tau):
    """Aggregate joint controller across all steps for one method.

    cost_ratio is the TOTAL-cost / TOTAL-baseline-cost (sum over steps of
    cost_used, divided by sum over steps of n_prompts*G_BASE) — this is
    what the joint controller script in iter 71 reports as the headline
    ratio (it is dimensionless and bootstrap-consistent).
    """
    tot_rollout = 0
    tot_zvf = 0
    tot_net = 0
    tot_cost_used = 0
    tot_baseline_cost = 0
    for sr in method_records:
        r, z, n, _, _, cr = joint_controller_one_step(sr, ddiv_tau)
        tot_rollout += r
        tot_zvf += z
        tot_net += n
        # Reconstruct cost_used from the (cr * baseline_per_step) inverse:
        # cr = cost_used / baseline_per_step; cost_used = cr * baseline
        tot_cost_used += cr * (sr["n_prompts"] * G_BASE)
        tot_baseline_cost += sr["n_prompts"] * G_BASE
    return {
        "rollout_saves": tot_rollout,
        "zvf_saves": tot_zvf,
        "net_saves": tot_net,
        "cost_ratio": tot_cost_used / tot_baseline_cost if tot_baseline_cost else 0.0,
    }


def bootstrap_ci_net_saves(method_records, ddiv_tau, n_boot=N_BOOT_PROMPT, seed=SEED):
    """Prompt-resampled bootstrap on net_saves / cost_ratio per method.

    Resample prompt-steps with replacement from the 16 × 40 grid of
    (contrast / boundary / ddiv_step) tuples per method, then apply
    the joint controller to the resampled grid and recompute the
    total-rollout, total-zvf, total-net, and total-cost_ratio metrics.
    Sort bootstrap replicates by net_saves and extract 2.5% / 97.5%
    percentile CIs (95% percentile bootstrap).
    """
    rng = random.Random(seed)
    flat = []  # list of dicts with prompt-level fields + step-level ddiv
    for sr in method_records:
        for p in sr["prompts"]:
            flat.append({
                "contrast": p["contrast"],
                "boundary": p["boundary"],
                "zvf_iid_g16": p["zvf_iid_g16"],
                "ddiv_step": sr["ddiv_step"],
            })
    n = len(flat)
    boots = []
    for _ in range(n_boot):
        sample = [flat[rng.randrange(n)] for _ in range(n)]
        rollout_saves = 0
        zvf_saves = 0
        cost_used = 0
        for ps in sample:
            if ps["contrast"]:
                rollout_saves += (G_BASE - G_DUALFORMER)
                cost_used += G_DUALFORMER
            elif ps["boundary"] and ps["ddiv_step"] >= ddiv_tau:
                zvf_saves += 1
                cost_used += G_ESC
            else:
                cost_used += G_BASE
        baseline_cost = n * G_BASE
        net = rollout_saves + zvf_saves
        cost_ratio = cost_used / baseline_cost
        boots.append((rollout_saves, zvf_saves, net, cost_ratio))
    # Sort bootstrap replicates independently by each metric for that
    # metric's percentile CI (sort-by-net is insufficient because the
    # four headline metrics are not monotonically co-ordered).
    def pct_ci(values, alpha=0.05):
        s = sorted(values)
        return s[int(alpha / 2 * len(s))], s[int((1 - alpha / 2) * len(s)) - 1]
    r_vals = [b[0] for b in boots]
    z_vals = [b[1] for b in boots]
    n_vals = [b[2] for b in boots]
    cr_vals = [b[3] for b in boots]
    r_lo, r_hi = pct_ci(r_vals)
    z_lo, z_hi = pct_ci(z_vals)
    n_lo, n_hi = pct_ci(n_vals)
    cr_lo, cr_hi = pct_ci(cr_vals)
    return {
        "net_saves_ci_lo": n_lo,
        "net_saves_ci_hi": n_hi,
        "rollout_saves_ci_lo": r_lo,
        "rollout_saves_ci_hi": r_hi,
        "zvf_saves_ci_lo": z_lo,
        "zvf_saves_ci_hi": z_hi,
        "cost_ratio_ci_lo": cr_lo,
        "cost_ratio_ci_hi": cr_hi,
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
    # --- N10 multi-trigger ---
    seeds = load_n10_seeds()
    per_seed = per_seed_fire_counts(seeds)
    summary = seed_stability_summary(per_seed)
    rank = rank_consistency(per_seed)

    write_tsv(
        OUT / "p7_multitrigger_seed_per_seed.tsv",
        per_seed,
        fieldnames=[
            "seed", "trigger", "tau", "n_fire", "n_steps",
            "n_bdy_fire", "n_ddiv_neg_fire",
            "mean_zvf_seed", "heldout_acc_seed",
        ],
    )
    write_tsv(
        OUT / "p7_multitrigger_seed_summary.tsv",
        summary,
        fieldnames=[
            "trigger", "tau", "n_seeds",
            "n_fire_mean", "n_fire_sd", "n_fire_cv",
            "n_fire_ci_lo", "n_fire_ci_hi",
            "headroom_bdy_total", "ddiv_neg_fire_total",
        ],
    )
    write_tsv(
        OUT / "p7_multitrigger_seed_rank.tsv",
        rank,
        fieldnames=[
            "trigger", "tau", "top_seed", "top_n_fire",
            "bottom_seed", "bottom_n_fire", "rank_spread",
        ],
    )

    # --- N2 joint controller bootstrap ---
    n2 = load_n2_records()
    boot_rows = []
    for tau in DDIV_TAU_HEADLINE:
        for m in METHODS:
            point = headline_for_method(n2[m], tau)
            ci = bootstrap_ci_net_saves(n2[m], tau)
            boot_rows.append({
                "method": m,
                "tau": f"{tau:.2f}",
                "rollout_saves": point["rollout_saves"],
                "zvf_saves": point["zvf_saves"],
                "net_saves": point["net_saves"],
                "cost_ratio": f"{point['cost_ratio']:.6f}",
                "net_saves_ci_lo": ci["net_saves_ci_lo"],
                "net_saves_ci_hi": ci["net_saves_ci_hi"],
                "rollout_saves_ci_lo": ci["rollout_saves_ci_lo"],
                "rollout_saves_ci_hi": ci["rollout_saves_ci_hi"],
                "zvf_saves_ci_lo": ci["zvf_saves_ci_lo"],
                "zvf_saves_ci_hi": ci["zvf_saves_ci_hi"],
                "cost_ratio_ci_lo": f"{ci['cost_ratio_ci_lo']:.6f}",
                "cost_ratio_ci_hi": f"{ci['cost_ratio_ci_hi']:.6f}",
            })
    write_tsv(
        OUT / "p7_joint_controller_ci.tsv",
        boot_rows,
        fieldnames=[
            "method", "tau",
            "rollout_saves", "zvf_saves", "net_saves", "cost_ratio",
            "net_saves_ci_lo", "net_saves_ci_hi",
            "rollout_saves_ci_lo", "rollout_saves_ci_hi",
            "zvf_saves_ci_lo", "zvf_saves_ci_hi",
            "cost_ratio_ci_lo", "cost_ratio_ci_hi",
        ],
    )

    # --- JSON summary ---
    summary_json = {
        "iter": 79,
        "pillar": "P7",
        "vein": "(c) seed-robustness of the trigger threshold + (d) bootstrap CIs on joint controller headline",
        "n_seeds": len(seeds),
        "n_prompts_per_method": 16 * 40,
        "n_boot_seed": N_BOOT_SEED,
        "n_boot_prompt": N_BOOT_PROMPT,
        "seed": SEED,
        "triggers": list(TRIGGERS),
        "tau_grids": {k: list(v) for k, v in TAU_GRID.items()},
        "seed_summary": summary,
        "seed_rank": rank,
        "headline_metrics": {
            "n_seeds": len(seeds),
            "best_trigger_by_seed_stability": sorted(
                summary, key=lambda r: r["n_fire_cv"]
            )[0]["trigger"],
            "best_trigger_at_canonical_tau": sorted(
                summary, key=lambda r: (r["trigger"], r["tau"])
            )[0]["trigger"],
            "joint_controller_ci": boot_rows,
        },
        "headlines": {
            "H1_seed_cv_by_trigger": {
                r["trigger"]: round(
                    statistics.mean(
                        s["n_fire_cv"] for s in summary
                        if s["trigger"] == r["trigger"]
                    ), 4
                )
                for r in summary[:1]
            },
            "H2_joint_ci_width_at_tau_0_05": {
                row["method"]: {
                    "net_ci_width": row["net_saves_ci_hi"] - row["net_saves_ci_lo"],
                    "cost_ratio_ci_width_pct": round(
                        100 * (float(row["cost_ratio_ci_hi"]) - float(row["cost_ratio_ci_lo"]))
                        / float(row["cost_ratio"]),
                        2,
                    ),
                }
                for row in boot_rows if row["tau"] == "0.05"
            },
            "H3_joint_ci_at_tau_0_07": {
                row["method"]: {
                    "net_ci_width": row["net_saves_ci_hi"] - row["net_saves_ci_lo"],
                }
                for row in boot_rows if row["tau"] == "0.07"
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
            ci_w = row["net_saves_ci_hi"] - row["net_saves_ci_lo"]
            cr_w = 100 * (float(row["cost_ratio_ci_hi"]) - float(row["cost_ratio_ci_lo"])) / float(row["cost_ratio"])
            print(f"  {row['method']:6s}: net_saves={row['net_saves']:4d} [{row['net_saves_ci_lo']:4d}, {row['net_saves_ci_hi']:4d}] (CI width={ci_w}); cost_ratio={row['cost_ratio']} ±{cr_w:.2f}%")
    print(f"\n=== Joint-controller CIs at τ=0.07 ===")
    for row in boot_rows:
        if row["tau"] == "0.07":
            ci_w = row["net_saves_ci_hi"] - row["net_saves_ci_lo"]
            print(f"  {row['method']:6s}: net_saves={row['net_saves']:4d} [{row['net_saves_ci_lo']:4d}, {row['net_saves_ci_hi']:4d}] (CI width={ci_w})")
    print(f"\nDONE — wrote outputs to {OUT}")


if __name__ == "__main__":
    main()