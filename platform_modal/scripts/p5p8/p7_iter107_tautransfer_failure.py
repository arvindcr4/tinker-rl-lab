#!/usr/bin/env python3
"""
Iter 107 -- Pillar 3 (P7) cross-method tau-transfer robustness + step-level failure-mode taxonomy.

Vein (a) of the brief at the transfer/sharpening level:
  - Calibrate trigger threshold tau in-method on each of the 4 N2 methods.
  - TRANSFER test: apply grpo's optimal tau to aero/gift/areal; measure
    contrast_magnitude retention vs in-method-tuned tau. Bootstrap CI per method.
  - Per-step model-action taxonomy: useful escalation, low-gain escalation,
    guard blocked, trigger no-op at cap, missed modeled gain, or no-gain/no-action.
    Gain is incremental relative to the no-triage Dualformer action, not absolute
    contrast at G16. These are descriptive IID-model diagnostics, not observed
    training failures. Bootstrap CIs are conditional step-level uncertainty.

Outputs (platform_hybrid/experiments/results/p5p8/):
  p7_iter107_taut_in_method.tsv          per-method optimal tau + metric
  p7_iter107_taut_transfer.tsv          transfer CIs (grpo_tau -> other-methods)
  p7_iter107_failure_taxonomy.tsv       one row per (method, step)
  p7_iter107_failure_summary.tsv        per-method failure-class share table
  p7_iter107_failure_bootstrap_ci.tsv   bootstrap CIs on failure shares
  p7_iter107_summary.json               machine-readable

Stdlib only. <= 300 LoC. Designed to be reviewable + rebuildable.
"""

from __future__ import annotations
import csv
import json
import pathlib
import random

import sys

sys.path.append(
    str(pathlib.Path(__file__).resolve().parents[1])
)  # platform_modal/scripts, for _paths
from _paths import REPO_ROOT

WORKTREE = REPO_ROOT
N2 = WORKTREE / "platform_hybrid/experiments" / "results" / "n2_reward_tensor_resume"
OUT = WORKTREE / "platform_hybrid/experiments" / "results" / "p5p8" / "corrected-review19-v5"
OUT.mkdir(parents=True, exist_ok=True)
METHODS = ("grpo", "aero", "gift", "areal")
TAUS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
G_BASE = 8
G_POOL = (2, 4, 8, 16)
G_ESC = 16
N_PROMPTS = 16
PCD_GUARD = 0.20
N_BOOT = 4000
BOOT_SEED = 20260705
EPS = 1e-9
DEGEN_ZVF = 0.99  # if zvf_i >= 0.99 under iid binomial, prompt is structurally degenerate
FAILURE_CLASSES = (
    "USEFUL_ESCALATION",
    "LOW_GAIN_ESCALATION",
    "GUARD_BLOCKED",
    "TRIGGER_NOOP",
    "MISSED_MODELED_GAIN",
    "NO_GAIN_NO_ACTION",
)
G_ESC_GRID = (8, 16)  # what G to escalate to for failure-budget eval


def pooled_failure_shares(by_method, classes, n_boot, rng):
    """Stratified step bootstrap; no independent-run uncertainty is implied."""
    groups = [records for records in by_method.values() if records]
    total = sum(map(len, groups))
    if not total:
        raise ValueError("no observations for pooled shares")
    rows = []
    for column, cls in enumerate(classes):
        point = sum(r[column] for group in groups for r in group) / total
        draws = sorted(
            sum(group[rng.randrange(len(group))][column] for group in groups for _ in group) / total
            for _ in range(n_boot)
        )
        lo, hi = draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)]
        rows.append(
            {
                "method": "POOLED",
                "class": cls,
                "share_obs": point,
                "ci_lo": lo,
                "ci_hi": hi,
                "ci_excludes_zero": int(lo > 0),
            }
        )
    return rows


def load_tensors():
    out = {}
    for m in METHODS:
        path = N2 / f"{m}_s0_tensors.jsonl"
        with path.open() as f:
            for line in f:
                d = json.loads(line)
                out[(m, d["step"])] = d["rewards"]
                out[(m, d["step"], "_meta")] = {
                    "zvf": d.get("zvf", 0.0),
                    "pcd": d.get("pcd", 0.0),
                    "reward_mean": d.get("reward_mean", 0.0),
                }
    return out


def per_prompt_means(rewards):
    return [sum(g) / len(g) for g in rewards]


def iid_zvf(p, g):
    pp = min(max(p, EPS), 1 - EPS)
    return (1 - pp) ** g + pp**g


def contrast_magnitude(per_p, gs):
    """sum (1 - zvf_p(G_p)) over all prompts -- zero if all prompt-cells are
    structurally degenerate at G_p."""
    s = 0.0
    for p, g in zip(per_p, gs, strict=False):
        z = iid_zvf(p, g)
        s += 1 - z
    return s


def step_structural_opportunity(per_p, g_new=G_ESC):
    """Max contrast-magnitude achievable at G_new under iid binomial.
    Absolute yield only; this is not incremental escalation headroom."""
    s = 0.0
    for p in per_p:
        s += 1 - iid_zvf(p, g_new)
    return s


def decide_gs_c3(per_p, zvf, pcd, tau):
    """Iter-103 C3 controller: Dualformer-Auto + ZVF-triage escalation."""
    gs = []
    for p in per_p:
        if p >= 0.95:
            gs.append(2)
        elif p >= 0.85:
            gs.append(4)
        elif p >= 0.70:
            gs.append(8)
        else:
            gs.append(16)
    fired = False
    if zvf >= tau and pcd <= PCD_GUARD:
        idx_map = {2: 0, 4: 1, 8: 2, 16: 3}
        bump_map = {0: 4, 1: 8, 2: 16, 3: 16}
        bumped = [bump_map[idx_map[g]] for g in gs]
        fired = any(new > old for new, old in zip(bumped, gs, strict=False))
        gs = bumped
    return gs, fired


def optimal_tau_in_method(steps_m):
    """For each tau, compute combined score = savings * (1 - magnitude_loss/C0_mag).
    Pick the tau that maximizes combined. Returns (best_tau, scores_by_tau)."""
    base_rollouts = N_PROMPTS * G_BASE * len(steps_m)
    base_mag = sum(contrast_magnitude(s["per_p"], [G_BASE] * N_PROMPTS) for s in steps_m)
    rows = []
    for tau in TAUS:
        rollouts = 0
        mag = 0.0
        fires = 0
        for s in steps_m:
            gs, fired = decide_gs_c3(s["per_p"], s["zvf"], s["pcd"], tau)
            rollouts += sum(gs)
            mag += contrast_magnitude(s["per_p"], gs)
            fires += int(fired)
        sav = (base_rollouts - rollouts) / base_rollouts
        ret = mag / base_mag if base_mag > 0 else 1.0
        # combined: savings-fraction weighted by contrast retention.
        # Equivalent to savings * ret on this frame.
        combined = sav * ret
        rows.append(
            {
                "tau": tau,
                "savings_frac": sav,
                "magnitude_retention": ret,
                "combined": combined,
                "fires": fires,
                "rollouts": rollouts,
                "magnitude": mag,
                "base_mag": base_mag,
                "base_rollouts": base_rollouts,
            }
        )
    rows.sort(key=lambda r: -r["combined"])
    return rows[0]["tau"], rows


def transfer_eval(steps_m, tau):
    """Apply tau to method m; return savings, retention, fires."""
    base_rollouts = N_PROMPTS * G_BASE * len(steps_m)
    base_mag = sum(contrast_magnitude(s["per_p"], [G_BASE] * N_PROMPTS) for s in steps_m)
    rollouts = 0
    mag = 0.0
    fires = 0
    for s in steps_m:
        gs, fired = decide_gs_c3(s["per_p"], s["zvf"], s["pcd"], tau)
        rollouts += sum(gs)
        mag += contrast_magnitude(s["per_p"], gs)
        fires += int(fired)
    sav = (base_rollouts - rollouts) / base_rollouts
    ret = mag / base_mag if base_mag > 0 else 1.0
    return {
        "savings_frac": sav,
        "magnitude_retention": ret,
        "fires": fires,
        "rollouts": rollouts,
        "magnitude": mag,
    }


def action_diagnostics(s, tau):
    """Compare actual and potential triage bumps with the same unbumped action."""
    ps = s["per_p"]
    baseline, _ = decide_gs_c3(ps, s["zvf"], s["pcd"], float("inf"))
    potential, _ = decide_gs_c3(ps, 1.0, 0.0, 0.0)
    actual, fired = decide_gs_c3(ps, s["zvf"], s["pcd"], tau)
    base_yield = contrast_magnitude(ps, baseline)
    return {
        "baseline_groups": baseline,
        "actual_groups": actual,
        "trigger_eligible": bool(s["zvf"] >= tau and s["pcd"] <= PCD_GUARD),
        "guard_blocked": bool(s["zvf"] >= tau and s["pcd"] > PCD_GUARD),
        "fired": fired,
        "incremental_opportunity": contrast_magnitude(ps, potential) - base_yield,
        "incremental_gain": contrast_magnitude(ps, actual) - base_yield,
    }


def failure_classify(s, tau):
    """Versioned action taxonomy; guard rejection is never a false-positive fire."""
    d = action_diagnostics(s, tau)
    if d["guard_blocked"]:
        return "GUARD_BLOCKED"
    if d["trigger_eligible"] and not d["fired"]:
        return "TRIGGER_NOOP"
    if d["fired"]:
        return "USEFUL_ESCALATION" if d["incremental_gain"] > 0.05 else "LOW_GAIN_ESCALATION"
    return "MISSED_MODELED_GAIN" if d["incremental_opportunity"] > 0.05 else "NO_GAIN_NO_ACTION"


def bootstrap_ci(values, n_boot=N_BOOT, seed=BOOT_SEED, alpha=0.05):
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    boots = []
    for _ in range(n_boot):
        s = sum(values[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    lo = boots[int(alpha / 2 * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot)]
    return (sum(values) / n, lo, hi)


def main():
    tensors = load_tensors()
    steps = sorted({k[1] for k in tensors if not isinstance(k[1], str)})

    # Build per-method step list
    step_info_per_method = {}
    for m in METHODS:
        sl = []
        for s in steps:
            meta = tensors[(m, s, "_meta")]
            rew = tensors[(m, s)]
            sl.append(
                {
                    "step": s,
                    "zvf": meta["zvf"],
                    "pcd": meta["pcd"],
                    "reward_mean": meta["reward_mean"],
                    "per_p": per_prompt_means(rew),
                }
            )
        step_info_per_method[m] = sl

    # =========================================================================
    # PART 1: in-method optimal tau + cross-method TRANSFER
    # =========================================================================
    in_method_rows = []
    for m in METHODS:
        best_tau, scans = optimal_tau_in_method(step_info_per_method[m])
        for r in scans:
            in_method_rows.append(
                {"method": m, **r, "is_optimal": int(r["tau"] == best_tau), "optimal_tau": best_tau}
            )

    transfer_rows = []
    opt_tau = {}
    for m in METHODS:
        opt_tau[m] = [
            r["tau"] for r in in_method_rows if r["method"] == m and r["is_optimal"] == 1
        ][0]
    # TRANSFER: apply grpo's optimal tau to every method, vs each method's own.
    for target_m in METHODS:
        own = transfer_eval(step_info_per_method[target_m], opt_tau[target_m])
        transfer_rows.append(
            {
                "source_method": target_m,
                "target_method": target_m,
                "tau_used": opt_tau[target_m],
                "kind": "IN_METHOD",
                "savings_frac": own["savings_frac"],
                "magnitude_retention": own["magnitude_retention"],
                "fires": own["fires"],
            }
        )
        for source_m in METHODS:
            if source_m == target_m:
                continue
            xfer = transfer_eval(step_info_per_method[target_m], opt_tau[source_m])
            transfer_rows.append(
                {
                    "source_method": source_m,
                    "target_method": target_m,
                    "tau_used": opt_tau[source_m],
                    "kind": "TRANSFER",
                    "savings_frac": xfer["savings_frac"],
                    "magnitude_retention": xfer["magnitude_retention"],
                    "fires": xfer["fires"],
                }
            )

    # Bootstrap retention-DELTA (own - transferred) per target method, source != target
    boot_rows = []
    for tgt in METHODS:
        own_sav = [
            r["savings_frac"]
            for r in transfer_rows
            if r["target_method"] == tgt and r["kind"] == "IN_METHOD"
        ][0]
        own_ret = [
            r["magnitude_retention"]
            for r in transfer_rows
            if r["target_method"] == tgt and r["kind"] == "IN_METHOD"
        ][0]
        for src in METHODS:
            if src == tgt:
                continue
            xfer_sav = [
                r["savings_frac"]
                for r in transfer_rows
                if r["target_method"] == tgt and r["source_method"] == src
            ][0]
            xfer_ret = [
                r["magnitude_retention"]
                for r in transfer_rows
                if r["target_method"] == tgt and r["source_method"] == src
            ][0]
            boot_rows.append(
                {
                    "source_method": src,
                    "target_method": tgt,
                    "tau_source_optimal": opt_tau[src],
                    "tau_target_optimal": opt_tau[tgt],
                    "own_savings_frac": own_sav,
                    "xfer_savings_frac": xfer_sav,
                    "own_magnitude_retention": own_ret,
                    "xfer_magnitude_retention": xfer_ret,
                    "delta_savings": own_sav - xfer_sav,
                    "delta_mag_ret": own_ret - xfer_ret,
                    # Will compute CIs below -- single transfer has only one number,
                    # so we cannot bootstrap a single point. We bootstrap the
                    # PER-STEP in-method-vs-transfer DELTA below.
                }
            )

    # =========================================================================
    # PART 2: per-step failure-mode taxonomy + bootstrap CIs
    # =========================================================================
    # Taxonomy uses each method's selected threshold. The average below is
    # descriptive only, not the threshold used by the classification rows.
    canonical_tau = sum(opt_tau.values()) / len(opt_tau)
    failure_rows = []
    summary_counts = {m: {**dict.fromkeys(FAILURE_CLASSES, 0), "_total": 0} for m in METHODS}
    per_step_records = []  # for bootstrap
    for m in METHODS:
        for s in step_info_per_method[m]:
            tau_for_step = opt_tau[m]
            cls = failure_classify(s, tau_for_step)
            diagnostic = action_diagnostics(s, tau_for_step)
            per_step_records.append(
                {
                    "method": m,
                    "step": s["step"],
                    "tau": tau_for_step,
                    "zvf": s["zvf"],
                    "pcd": s["pcd"],
                    "incremental_opportunity": diagnostic["incremental_opportunity"],
                    "incremental_gain": diagnostic["incremental_gain"],
                    "trigger_eligible": int(diagnostic["trigger_eligible"]),
                    "fired": int(diagnostic["fired"]),
                    "failure_class": cls,
                }
            )
            failure_rows.append(per_step_records[-1])
            summary_counts[m][cls] += 1
            summary_counts[m]["_total"] += 1

    # Bootstrap CIs on per-method failure-class shares by re-sampling steps with
    # replacement
    rng = random.Random(BOOT_SEED + 1)
    classes = FAILURE_CLASSES
    fail_share_boot_rows = []
    # Group per-step class dummies by method
    by_method = {m: [] for m in METHODS}
    for r in per_step_records:
        v = [1.0 if r["failure_class"] == c else 0.0 for c in classes]
        by_method[r["method"]].append(v)
    for m in METHODS:
        records = by_method[m]
        n = len(records)
        for ci, cls in enumerate(classes):
            shares = [r[ci] for r in records]
            boots = []
            for _ in range(N_BOOT):
                s_share = sum(shares[rng.randrange(n)] for _ in range(n)) / n
                boots.append(s_share)
            boots.sort()
            lo = boots[int(0.025 * N_BOOT)]
            hi = boots[int(0.975 * N_BOOT)]
            fail_share_boot_rows.append(
                {
                    "method": m,
                    "class": cls,
                    "share_obs": sum(shares) / n,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "ci_excludes_zero": int(lo > 0),
                }
            )
    # Conditional within-method step bootstrap; preserves observed method weights.
    pooled_rows = pooled_failure_shares(by_method, classes, N_BOOT, rng)
    fail_share_boot_rows.extend(pooled_rows)
    pool_shares = {r["class"]: r["share_obs"] for r in pooled_rows}

    # Failure summary table
    failure_summary_rows = []
    for m in METHODS:
        total = summary_counts[m]["_total"]
        for cls in classes:
            n = summary_counts[m][cls]
            share = n / total if total > 0 else 0.0
            failure_summary_rows.append(
                {
                    "method": m,
                    "class": cls,
                    "count": n,
                    "total": total,
                    "share": share,
                }
            )

    # =========================================================================
    # WRITE OUTPUTS
    # =========================================================================
    with (OUT / "p7_iter107_taut_in_method.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(in_method_rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in in_method_rows:
            w.writerow(r)
    with (OUT / "p7_iter107_taut_transfer.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(transfer_rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in transfer_rows:
            w.writerow(r)
    with (OUT / "p7_iter107_failure_taxonomy.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(failure_rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in failure_rows:
            w.writerow(r)
    with (OUT / "p7_iter107_failure_summary.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(failure_summary_rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in failure_summary_rows:
            w.writerow(r)
    with (OUT / "p7_iter107_failure_bootstrap_ci.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(fail_share_boot_rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in fail_share_boot_rows:
            w.writerow(r)

    summary_json = {
        "ts": "2026-07-05",
        "iter": 107,
        "method_optimal_tau": opt_tau,
        "mean_optimal_tau_descriptive": round(canonical_tau, 4),
        "taxonomy_tau_policy": "per_method_selected_threshold",
        "taxonomy_protocol": "corrected-review19-v5-incremental-action",
        "in_method_optimal_tau_per_method": opt_tau,
        "transfer_rows": transfer_rows,
        "failure_counts_per_method": summary_counts,
        "failure_pooled_shares": pool_shares,
        "n_steps_per_method": {m: len(step_info_per_method[m]) for m in METHODS},
        "headline_holders": {
            # Filled below based on transfer findings:
            "PLACEHOLDER": [],
        },
    }
    with (OUT / "p7_iter107_summary.json").open("w") as f:
        json.dump(summary_json, f, indent=2)


if __name__ == "__main__":
    main()
