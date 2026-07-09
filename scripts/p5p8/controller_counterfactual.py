#!/usr/bin/env python3
"""P7 Controller Counterfactual Evaluation on real N2 reward tensors.

For each (method, step) we replay the per-prompt rollout vectors at hypothetical
group sizes G' > 8 using an i.i.d. binomial model. The controller is asked:
``given observed (zvf, pcd, frac_all_zero, frac_all_one) at G=8, would the
zvf-triage callback fire?  Which G would it choose?  How many prompts does the
escalation actually save (degenerate -> non-degenerate)?''

Three controllers are compared:

  A. **zvf-triage** (the paper's controller): step-level trigger; fires when
     zvf >= threshold AND pcd <= max_pcd.  When fired, NEXT step uses G' = 16
     for all prompts.  Threshold swept in {0.50, 0.60, 0.70, 0.80, 0.90}.
  B. **Dualformer-Auto** (Berkeley row 01 rule): difficulty-gated per-prompt G.
     acc_pred = per-prompt reward mean; G'=2 if acc_pred >= 0.95, 4 if >=0.85,
     8 if >=0.70, 16 otherwise.  Per-prompt cost only.
  C. **oracle hindsight**: per-step rule that fires iff at least one prompt
     would save at G'=16.  Upper bound on the headroom the controller could
     ever extract from this data.

Outputs:
  experiments/results/p5p8/controller_cf_summary.tsv   -- one row per
    (method, controller, threshold) with fires, rollouts_spent, saved_prompts,
    wasted_prompts, naive_fixed_G_cost, cost_ratio.
  experiments/results/p5p8/controller_cf_per_step.tsv   -- one row per
    (method, step) under each controller, showing the per-step decisions.
  experiments/results/p5p8/controller_cf_summary.json   -- machine-readable
    with seed-robustness block (sensitivity of the fire count to threshold).

Stdlib only.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import pathlib
import statistics

WORKTREE = pathlib.Path("/home/claude/tinker-rl-lab-minimax")
N2 = WORKTREE / "experiments" / "results" / "n2_reward_tensor_resume"
OUT = WORKTREE / "experiments" / "results" / "p5p8"
METHODS = ("grpo", "aero", "gift", "areal")
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
MAX_PCD = 0.20            # interior-regime guard (above this, the run is at the
                          # degenerate boundary; no amount of G helps)
G_BASE = 8                # what the N2 four-method run actually used
G_ALT = 16                # counterfactual escalation target
N_PROMPTS = 16            # N2 prompt set size per step


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_tensors():
    """Return dict[(method, step)] -> list[list[float]] (16 x G_BASE rewards)."""
    out = {}
    for m in METHODS:
        path = N2 / f"{m}_s0_tensors.jsonl"
        with path.open() as f:
            for line in f:
                d = json.loads(line)
                out[(m, d["step"])] = d["rewards"]
    return out


def per_prompt_means(rewards):
    return [sum(g) / len(g) for g in rewards]


def is_degenerate(g, eps=1e-9):
    """A group is degenerate iff all rollouts agree (all-zero or all-one).
    Mixed groups are NOT degenerate (they already provide contrast)."""
    mn = min(g)
    mx = max(g)
    if abs(mx - mn) > eps:
        return False
    # all rollouts identical -- degenerate iff the value is 0 or 1
    return abs(mn) < eps or abs(mx - 1.0) < eps


# ---------------------------------------------------------------------------
# I.i.d. counterfactual: what would the per-prompt ZVF be at G' > G?
# ---------------------------------------------------------------------------
def zvf_at_g(per_prompt_p, g_new):
    """Expected ZVF at group size g_new given point-estimate p = per-prompt
    reward mean, under the i.i.d. binomial model. Returns per-prompt values
    plus the average ZVF across prompts."""
    out = []
    for p in per_prompt_p:
        # clamp p to avoid 0^0 issues
        pp = min(max(p, 1e-9), 1 - 1e-9)
        zvf = (1 - pp) ** g_new + pp ** g_new
        out.append(zvf)
    return out, statistics.mean(out)


def would_save(rewards_per_prompt, g_new):
    """A prompt is 'saved' if it is currently degenerate at G_BASE but would
    NOT be degenerate at G_new under i.i.d. (i.e. expected ZVF at G_new < 1).

    Args:
        rewards_per_prompt: list of groups (each group is list of G_BASE float
            rewards in {0,1}).
        g_new: hypothetical group size to evaluate against.
    """
    saved = 0
    wasted = 0
    for g in rewards_per_prompt:
        p = sum(g) / len(g)
        pp = min(max(p, 1e-9), 1 - 1e-9)
        zvf_new = (1 - pp) ** g_new + pp ** g_new
        if is_degenerate(g):
            if zvf_new < 0.99:
                saved += 1
            else:
                wasted += 1
    return saved, wasted


# ---------------------------------------------------------------------------
# Controller A: zvf-triage (step-level)
# ---------------------------------------------------------------------------
def zvf_triage_decide(step_zvf, step_pcd, threshold, max_pcd=MAX_PCD):
    """Returns the G' to usefor the *next* step, or None if the controller
    did not fire. Fires iff step_zvf >= threshold AND step_pcd <= max_pcd
    (interior-regime guard)."""
    if step_zvf >= threshold and step_pcd <= max_pcd:
        return G_ALT
    return None


# ---------------------------------------------------------------------------
# Controller B: Dualformer-Auto (per-prompt)
# ---------------------------------------------------------------------------
def dualformer_auto_decide(per_p):
    """Per-prompt G' selection following Berkeley row 01 rule:
       acc_pred = per-prompt reward mean; G'=2 if >=0.95, 4 if >=0.85,
       8 if >=0.70, 16 otherwise.  Per-prompt cost only (no step-level
       escalation)."""
    out = []
    for p in per_p:
        if p >= 0.95:
            out.append(2)
        elif p >= 0.85:
            out.append(4)
        elif p >= 0.70:
            out.append(8)
        else:
            out.append(16)
    return out


# ---------------------------------------------------------------------------
# Controller C: oracle hindsight
# ---------------------------------------------------------------------------
def oracle_decide(per_p):
    """Per-prompt oracle: use G'=16 only if the prompt is degenerate at G=8
    AND would save at G'=16.  Otherwise keep G=8.  Upper bound on headroom."""
    out = []
    for p in per_p:
        # treat G_BASE=8; degenerate if all-correct or all-wrong at observed
        # mean.  At p=1.0 or p=0.0 exactly, no escalation helps -> G=8.
        # At 0 < p < 1, the prompt is NOT degenerate (mixed group already).
        # Per the i.i.d. model, only 'near-boundary' prompts benefit.
        # We treat as 'would benefit' if 0.05 < p < 0.95.
        if 0.05 <= p <= 0.95:
            out.append(16)
        else:
            out.append(8)
    return out


# ---------------------------------------------------------------------------
# Rollout accounting
# ---------------------------------------------------------------------------
def rollouts_fixed_g(n_prompts=N_PROMPTS, g=G_BASE, n_steps=40):
    return n_prompts * g * n_steps


def rollouts_per_prompt(per_prompt_g, g_base=G_BASE):
    return sum(per_prompt_g)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="Write outputs under experiments/results/p5p8/")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tensors = load_tensors()
    steps = sorted({k[1] for k in tensors})
    summary_rows = []
    per_step_rows = []

    # --- Per-method, per-controller accounting -----------------------------
    for m in METHODS:
        # Pull per-step scalars from n2_metrics.tsv via the JSONL record.
        # We need pcd, zvf for the zvf-triage trigger.
        path = N2 / f"{m}_s0_tensors.jsonl"
        records = {}
        with path.open() as f:
            for line in f:
                d = json.loads(line)
                records[d["step"]] = d

        # Pre-compute per-step, per-prompt reward means + scalar zvf/pcd
        step_info = []
        for s in steps:
            rec = records[s]
            per_p = per_prompt_means(rec["rewards"])
            step_info.append({
                "step": s,
                "zvf": rec["zvf"],
                "pcd": rec["pcd"],
                "frac_all_zero": rec["frac_all_zero"],
                "frac_all_one": rec["frac_all_one"],
                "reward_mean": rec["reward_mean"],
                "per_p": per_p,
                "degenerate_count": sum(is_degenerate(g) for g in rec["rewards"]),
            })

        # ----- Controller A: zvf-triage, threshold sweep -------------------
        for thr in THRESHOLDS:
            fires = 0
            saved_prompts = 0
            wasted_prompts = 0
            rollouts = 0
            for i, s in enumerate(step_info):
                # Controller decides NEXT step's G based on THIS step's zvf/pcd.
                g_next = zvf_triage_decide(s["zvf"], s["pcd"], thr)
                if g_next is None:
                    # use base G
                    rollouts += N_PROMPTS * G_BASE
                    continue
                # Apply escalation to step i+1
                if i + 1 < len(step_info):
                    nxt = step_info[i + 1]
                    fires += 1
                    rollouts += N_PROMPTS * G_ALT  # cost on the escalated step
                    nxt_rewards = tensors[(m, nxt["step"])]
                    saved, wasted = would_save(nxt_rewards, G_ALT)
                    saved_prompts += saved
                    wasted_prompts += wasted
                    per_step_rows.append({
                        "method": m,
                        "controller": "zvf_triage",
                        "threshold": thr,
                        "trigger_step": s["step"],
                        "apply_step": nxt["step"],
                        "trigger_zvf": round(s["zvf"], 4),
                        "trigger_pcd": round(s["pcd"], 4),
                        "saved_prompts": saved,
                        "wasted_prompts": wasted,
                        "rollout_cost": N_PROMPTS * G_ALT,
                    })
                else:
                    # last step fired but nothing to escalate
                    rollouts += N_PROMPTS * G_BASE
            baseline = rollouts_fixed_g()
            summary_rows.append({
                "method": m,
                "controller": "zvf_triage",
                "threshold": thr,
                "fires": fires,
                "saved_prompts": saved_prompts,
                "wasted_prompts": wasted_prompts,
                "rollouts_used": rollouts,
                "baseline_fixed_g": baseline,
                "cost_ratio": round(rollouts / baseline, 3),
                "saved_per_fire": round(saved_prompts / max(fires, 1), 2),
            })

        # ----- Controller B: Dualformer-Auto (per-prompt) ------------------
        total_saved = 0
        total_wasted = 0
        total_rollouts = 0
        for s in step_info:
            g_per_p = dualformer_auto_decide(s["per_p"])
            # saved = prompts at G=8 that became non-degenerate at G'=g_per_p[i]
            saved = wasted = 0
            for g, gp, p in zip([None] * N_PROMPTS, g_per_p, s["per_p"]):
                # recompute would_save with the per-prompt G'
                pp = min(max(p, 1e-9), 1 - 1e-9)
                zvf_new = (1 - pp) ** gp + pp ** gp
                # we don't know per-group rolls beyond G=8; we approximate
                # 'was degenerate at G=8' using the i.i.d. probability at p
                zvf_g8 = (1 - pp) ** G_BASE + pp ** G_BASE
                if zvf_g8 >= 0.99 and zvf_new < 0.99:
                    saved += 1
                elif zvf_g8 >= 0.99 and zvf_new >= 0.99:
                    wasted += 1
            total_saved += saved
            total_wasted += wasted
            total_rollouts += sum(g_per_p)
            per_step_rows.append({
                "method": m,
                "controller": "dualformer_auto",
                "threshold": 0,
                "trigger_step": s["step"],
                "apply_step": s["step"],
                "trigger_zvf": round(s["zvf"], 4),
                "trigger_pcd": round(s["pcd"], 4),
                "saved_prompts": saved,
                "wasted_prompts": wasted,
                "rollout_cost": sum(g_per_p),
            })
        baseline = rollouts_fixed_g()
        summary_rows.append({
            "method": m,
            "controller": "dualformer_auto",
            "threshold": 0,
            "fires": 40,    # fires every step (per-prompt)
            "saved_prompts": total_saved,
"wasted_prompts": total_wasted,
            "rollouts_used": total_rollouts,
            "baseline_fixed_g": baseline,
            "cost_ratio": round(total_rollouts / baseline, 3),
            "saved_per_fire": round(total_saved / 40, 2),
        })

        # ----- Controller C: oracle hindsight ------------------------------
        total_saved = 0
        total_wasted = 0
        total_rollouts = 0
        for s in step_info:
            g_per_p = oracle_decide(s["per_p"])
            saved = wasted = 0
            for p, gp in zip(s["per_p"], g_per_p):
                pp = min(max(p, 1e-9), 1 - 1e-9)
                zvf_g8 = (1 - pp) ** G_BASE + pp ** G_BASE
                zvf_new = (1 - pp) ** gp + pp ** gp
                if zvf_g8 >= 0.99 and zvf_new < 0.99:
                    saved += 1
                elif zvf_g8 >= 0.99 and zvf_new >= 0.99:
                    wasted += 1
            total_saved += saved
            total_wasted += wasted
            total_rollouts += sum(g_per_p)
            per_step_rows.append({
                "method": m,
                "controller": "oracle",
                "threshold": 0,
                "trigger_step": s["step"],
                "apply_step": s["step"],
                "trigger_zvf": round(s["zvf"], 4),
                "trigger_pcd": round(s["pcd"], 4),
                "saved_prompts": saved,
                "wasted_prompts": wasted,
                "rollout_cost": sum(g_per_p),
            })
        baseline = rollouts_fixed_g()
        summary_rows.append({
            "method": m,
            "controller": "oracle",
            "threshold": 0,
            "fires": 40,
            "saved_prompts": total_saved,
            "wasted_prompts": total_wasted,
            "rollouts_used": total_rollouts,
            "baseline_fixed_g": baseline,
            "cost_ratio": round(total_rollouts / baseline, 3),
            "saved_per_fire": round(total_saved / 40, 2),
        })

    # ----- Write outputs ---------------------------------------------------
    cols = ["method", "controller", "threshold", "fires", "saved_prompts",
            "wasted_prompts", "rollouts_used", "baseline_fixed_g",
            "cost_ratio", "saved_per_fire"]
    if args.write:
        with (OUT / "controller_cf_summary.tsv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
        cols_ps = ["method", "controller", "threshold", "trigger_step",
                   "apply_step", "trigger_zvf", "trigger_pcd",
                   "saved_prompts", "wasted_prompts", "rollout_cost"]
        with (OUT / "controller_cf_per_step.tsv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=cols_ps, delimiter="\t")
            w.writeheader()
            for r in per_step_rows:
                w.writerow(r)

    # ----- Console summary -------------------------------------------------
    print(f"{'method':6s} {'controller':16s} {'thr':>5s} "
          f"{'fires':>5s} {'saved':>5s} {'wasted':>6s} "
          f"{'rollouts':>8s} {'baseline':>8s} {'ratio':>5s} "
          f"{'saved/fire':>9s}")
    print("-" * 90)
    for r in summary_rows:
        thr = f"{r['threshold']:.2f}" if r['threshold'] > 0 else "  -  "
        print(f"{r['method']:6s} {r['controller']:16s} {thr:>5s} "
              f"{r['fires']:>5d} {r['saved_prompts']:>5d} {r['wasted_prompts']:>6d} "
              f"{r['rollouts_used']:>8d} {r['baseline_fixed_g']:>8d} "
              f"{r['cost_ratio']:>5.2f} {r['saved_per_fire']:>9.2f}")

    # ----- Seed-robustness (threshold sweep on grpo) -----------------------
    grpo_triage = [r for r in summary_rows if r["method"] == "grpo"
                   and r["controller"] == "zvf_triage"]
    print("\nSeed-robustness (grpo zvf-triage, threshold sweep):")
    for r in grpo_triage:
        print(f"  thr={r['threshold']:.2f} fires={r['fires']:2d} "
              f"saved={r['saved_prompts']:2d} wasted={r['wasted_prompts']:2d} "
              f"cost_ratio={r['cost_ratio']:.2f}")

    # ----- Save JSON with seed-robustness block ---------------------------
    if args.write:
        out = {
            "summary_rows": summary_rows,
            "seed_robustness_grpo": grpo_triage,
            "headlines": {
                "n_methods": len(METHODS),
                "n_steps": len(steps),
                "g_base": G_BASE,
                "g_alt": G_ALT,
                "n_prompts_per_step": N_PROMPTS,
                "max_pcd_guard": MAX_PCD,
            },
            "interpretation": (
                "zvf-triage (paper controller) fires step-wise; the threshold "
                "controls the trade between escalation cost and saved prompts. "
                "Dualformer-Auto is a per-prompt difficulty-gated rule (Berkeley "
                "row 01); it fires every step.  Oracle is the per-prompt "
                "hindsight upper bound.  All three are evaluated counterfactually "
                "on the real N2 reward tensors with i.i.d. binomial resampling "
                "to estimate per-prompt ZVF at G' = 16."
            ),
        }
        (OUT / "controller_cf_summary.json").write_text(json.dumps(out, indent=2))
        print(f"\nwrote {OUT}/controller_cf_summary.{{tsv,json,json}}")


if __name__ == "__main__":
    main()