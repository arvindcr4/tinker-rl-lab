#!/usr/bin/env python3
"""P7 controller protocol v2: observed-state schedule cost replay.

Saved/wasted prompt outcomes are unmeasured without independent rollouts.
The interior heuristic is not an oracle. This replay does not estimate the
state trajectory that training under an alternative schedule would produce.
Writes versioned v2 outputs only with --write; historical evidence is untouched.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import pathlib
import statistics

import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # platform_modal/scripts, for _paths
from _paths import REPO_ROOT  # noqa: E402
from _p5p7_common import load_tensors_c  # noqa: E402
WORKTREE = REPO_ROOT
N2 = WORKTREE / "platform_hybrid/experiments" / "results" / "n2_reward_tensor_resume"
OUT = WORKTREE / "platform_hybrid/experiments" / "results" / "p5p8"
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
    """Recovery is unidentifiable from the same observed G=8 groups.

    Protocol v2 deliberately reports missing realized outcomes. Independent
    rollouts (or a separately specified posterior model) are required.
    """
    return None, None


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
def interior_heuristic_decide(per_p):
    """Descriptive interior-probability heuristic; not an oracle or bound."""
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
def triage_schedule(step_info, threshold):
    """Charge the applied group size once; terminal decisions incur no cost."""
    pending_g = G_BASE
    rows = []
    for step in step_info:
        rows.append({"step": step["step"], "g_applied": pending_g,
                     "rollout_cost": len(step["per_p"]) * pending_g})
        pending_g = zvf_triage_decide(step["zvf"], step["pcd"], threshold) or G_BASE
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    summary, details = [], []
    for method in METHODS:
        with (N2 / f"{method}_s0_tensors.jsonl").open() as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        records.sort(key=lambda r: r["step"])
        if not records or len({r["step"] for r in records}) != len(records):
            raise ValueError("Expected nonempty unique step identities")
        steps = [dict(r, per_p=per_prompt_means(r["rewards"])) for r in records]
        baseline = sum(len(s["per_p"]) * G_BASE for s in steps)
        for controller, threshold in [("zvf_triage", t) for t in THRESHOLDS] + [
                ("dualformer_auto", None), ("interior_heuristic", None)]:
            if controller == "zvf_triage":
                schedule = triage_schedule(steps, threshold)
                fires = sum(r["g_applied"] == G_ALT for r in schedule)
            else:
                decide = dualformer_auto_decide if controller == "dualformer_auto" else interior_heuristic_decide
                schedule = [{"step": s["step"], "g_applied": decide(s["per_p"]),
                             "rollout_cost": sum(decide(s["per_p"]))} for s in steps]
                fires = len(steps)
            used = sum(r["rollout_cost"] for r in schedule)
            summary.append(dict(protocol_version=2, method=method, controller=controller,
                threshold=threshold, fires=fires, rollouts_used=used,
                baseline_fixed_g=baseline, cost_ratio=used / baseline,
                saved_prompts=None, wasted_prompts=None,
                recovery_status="UNMEASURED_requires_independent_rollouts"))
            details.extend(dict(r, method=method, controller=controller, threshold=threshold)
                           for r in schedule)
    output = {"protocol_version": 2, "summary_rows": summary, "per_step_rows": details,
              "interpretation": "Observed-state schedule cost replay only. No realized recovery, oracle bound, or seed robustness is estimated."}
    print(json.dumps(output, indent=2))
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "controller_cf_v2_summary.json").write_text(json.dumps(output, indent=2) + "\n")
        for name, rows in (("summary", summary), ("per_step", details)):
            with (OUT / f"controller_cf_v2_{name}.tsv").open("w") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main()
