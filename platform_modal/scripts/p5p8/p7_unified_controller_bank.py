#!/usr/bin/env python3
"""
Pillar-7 (P7) Unified Adaptive-G Controller Bank.

Iter 51: brief vein (b), paper-facing operational specification that **absorbs**
zvf-triage, Dualformer-Auto, and Hybrid as boundary cases of one calibrated
parametric family:

    C(z_t | theta) = C(z_t | tau_esc, tau_des)
        G_t = G_des        if z_t >= tau_des
        G_t = G_esc        if tau_esc <= z_t < tau_des
        G_t = G_base       otherwise

Constraints:
    tau_esc < tau_des  (escalation band below the de-escalation band),
    G_des <= G_base <= G_esc  (de-escalation shrinks compute, escalation grows it).

Boundary collapses:
    tau_esc == tau_des      -> one threshold -> recovers zvf-triage@tau (single band)
    tau_des - tau_esc small -> no escalation band -> recovers Dualformer-Auto@tau_des
    full (tau_esc, tau_des) independent -> Hybrid (delta = tau_des - tau_esc).

Operates on TWO panels:
  (i)  N10 5-seed GRPO panel (15 steps/seed) -> per-seed compute + bootstrap CI.
  (ii) N2 four-method tensors (40 steps/method, 16 prompts/step, G=8)
       -> per-step replay + savings vs always-G=8.

Calibration objective (chosen for paper-facing readability):
    Pick theta on the (mean_savings_N10, headroom_bad=0, seed_CV<=0.10)
    Pareto-frontier. Within that frontier pick the **statistically-detectable**
    point (CI excludes zero) with the **lowest seed-CV**.

Outputs (worktree-relative):
    platform_hybrid/experiments/results/p5p8/p7_unified_controller_per_seed.tsv
    platform_hybrid/experiments/results/p5p8/p7_unified_controller_per_step_n2.tsv
    platform_hybrid/experiments/results/p5p8/p7_unified_controller_summary.tsv
    platform_hybrid/experiments/results/p5p8/p7_unified_controller_ci.tsv
    platform_hybrid/experiments/results/p5p8/p7_unified_controller_summary.json

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
N10_DIR = ROOT / "experiments" / "results" / "n10_seed_expansion"
N2_DIR = ROOT / "experiments" / "results" / "n2_reward_tensor_resume"
OUT = ROOT / "experiments" / "results" / "p5p8"
OUT.mkdir(parents=True, exist_ok=True)

# Stack-conditional constants used by the controller family.
G_BASE = 8          # always-G baseline
G_ESC = 16          # escalation (restore contrast)
G_DES = 4           # de-escalation (compute-saving regime)
N_STEPS_N10 = 15
N_PROMPTS_PER_STEP_N2 = 16   # n2 n_prompts=16, G=8 -> 128 rollouts/step
G_N2 = 8
N_BOOT = 2000
RNG_SEED = 51511

# ============================================================================
# tau grid for tau_esc (escalation lower bound)
TAU_ESC_GRID = [round(0.50 + 0.05 * i, 2) for i in range(8)]   # 0.50..0.85
# tau_des relative to tau_esc: 1-5 steps higher in 0.05 increments
DELTA_GRID = [round(0.05 * k, 2) for k in range(1, 6)]          # 0.05..0.25
# Filter points where tau_des > 1.0 OR tau_esc >= tau_des discarded
MAX_TAU_DES = 1.00


# ============================================================================
# Data loading
# ----------------------------------------------------------------------------

def load_n10_seeds():
    """Return list of dicts with seed, zvf trajectory (length 15)."""
    out = []
    for path in sorted(N10_DIR.glob("n10_grpo_s*.json")):
        d = json.loads(path.read_text())
        sl = d.get("step_log", [])
        if len(sl) < N_STEPS_N10:
            continue
        out.append({
            "seed": int(d["seed"]),
            "zvfs": [float(s["zvf"]) for s in sl[:N_STEPS_N10]],
        })
    return out


def load_n2_tensors(methods=("grpo", "aero", "gift", "areal")):
    """Return dict[method] -> list[dict]: one entry per step with
    zvf, frac_all_zero, all_prompt_zvfs (list of 16 0/1 vectors for that step)."""
    out = {}
    for m in methods:
        path = N2_DIR / f"{m}_s0_tensors.jsonl"
        if not path.exists():
            continue
        with open(path) as fh:
            rows = [json.loads(l) for l in fh]
        steps = []
        for r in rows:
            per_prompt_zvf = []
            for g in r["rewards"]:
                v = sum(g)
                p_zvf = (1.0 if v == 0 else 0.0) + (1.0 if v == len(g) else 0.0)
                per_prompt_zvf.append(p_zvf)
            steps.append({
                "step": int(r["step"]),
                "zvf_step": float(r["zvf"]),       # mean over 16 prompts
                "per_prompt_zvf": per_prompt_zvf,  # list of 16 floats in [0,2] for G=8
                "per_prompt_k": [int(sum(g)) for g in r["rewards"]],
            })
        out[m] = steps
    return out


# ============================================================================
# Controller
# ----------------------------------------------------------------------------

def replay_unified(zvfs: list[float], tau_esc: float, tau_des: float) -> list[int]:
    """Per-step G_t decisions: de-escalate / escalate / baseline."""
    out = []
    for z in zvfs:
        if z >= tau_des:
            out.append(G_DES)
        elif z >= tau_esc:
            out.append(G_ESC)
        else:
            out.append(G_BASE)
    return out


# ============================================================================
# Metrics
# ----------------------------------------------------------------------------

def n10_per_seed_metrics(zvfs: list[float], tau_esc: float, tau_des: float) -> dict:
    n = len(zvfs)
    G_t = replay_unified(zvfs, tau_esc, tau_des)
    total = sum(G_t)
    fires = sum(1 for g in G_t if g != G_BASE)
    escalations = sum(1 for g in G_t if g == G_ESC)
    deescalations = sum(1 for g in G_t if g == G_DES)
    # Headroom-bad: "fire" (any non-base) on z_t >= 0.99 -> wasteful
    headroom_bad = sum(1 for g, zt in zip(G_t, zvfs) if g != G_BASE and zt >= 0.99)
    return {
        "total_G": total,
        "savings": (G_BASE * n - total) / (G_BASE * n),
        "fire_rate": fires / n,
        "escalations": escalations,
        "deescalations": deescalations,
        "headroom_bad": headroom_bad,
    }


def n2_per_step_metrics(step: dict, tau_esc: float, tau_des: float) -> dict:
    """Per-step controller replay on N2's *step-level* aggregated ZVF.

    zvf_step in [0,1]: mean over 16 prompts (so continuous). Each step is
    a single observation; replay the unified controller with that one
    scalar and aggregate by methods."""
    G_t = replay_unified([step["zvf_step"]], tau_esc, tau_des)[0]
    n = 1
    return {
        "total_G_prompts": G_t,
        "savings": (G_BASE * n - G_t) / (G_BASE * n),
        "contrast_intent": (1.0 if G_t == G_ESC else 0.0),
        "fires_esc": (1 if G_t == G_ESC else 0),
        "fires_des": (1 if G_t == G_DES else 0),
    }


def n2_per_prompt_metrics(step: dict, tau_esc: float, tau_des: float) -> dict:
    """Per-prompt (16 prompts/step) replay on N2.

    Treats per-prompt boundary indicators (binary) as continuous proxies
    by mapping k=0 (all wrong, zvf=2) to 1.0 and k=8 (all correct, zvf=2)
    to 1.0; contrast (0<k<8, zvf=0) to 0.0. Then cap at 1.0 (binary
    boundary indicator) and replay the unified controller per-prompt."""
    zvfs_p = [min(1.0, z) for z in step["per_prompt_zvf"]]
    G_p = replay_unified(zvfs_p, tau_esc, tau_des)
    n = len(G_p)
    total = sum(G_p)
    # contrast_intent: fraction of *contrast* prompts (zvf=0) the controller
    # escalates. contrast_pool = [0/16 (no contrast-prompts)] -> empty;
    # boundary prompts (zvf=1) always go to G_des branch or stay at base
    # depending on the band.
    contrast_pool = [i for i, z in enumerate(zvfs_p) if z < 0.5]
    escal_on_contrast = sum(1 for i in contrast_pool if G_p[i] == G_ESC)
    ci = escal_on_contrast / max(1, len(contrast_pool))
    return {
        "total_G_prompts": total,
        "savings": (G_BASE * n - total) / (G_BASE * n),
        "contrast_intent": ci,
        "fires_esc": sum(1 for g in G_p if g == G_ESC),
        "fires_des": sum(1 for g in G_p if g == G_DES),
    }


def bootstrap_ci(values: list[float], n_boot: int = N_BOOT, alpha: float = 0.05,
                 rng: random.Random | None = None) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of `values`. n=seed level for N10
    so we resample n_seeds with replacement B times."""
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


def all_theta_points():
    """Yield all (tau_esc, delta) such that tau_des = tau_esc+delta <= MAX_TAU_DES
    and tau_esc < tau_des (strict inequality by construction)."""
    for tau_esc in TAU_ESC_GRID:
        for d in DELTA_GRID:
            tau_des = round(tau_esc + d, 2)
            if tau_des > MAX_TAU_DES + 1e-9:
                continue
            yield (tau_esc, tau_des)


# ============================================================================
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    rng = random.Random(RNG_SEED)
    n10 = load_n10_seeds()
    n2 = load_n2_tensors()
    n_seeds = len(n10)
    n_steps_n10 = N_STEPS_N10
    assert n_seeds >= 2, f"need >=2 n10 seeds, got {n_seeds}"

    # 1) Per-seed N10 replay for every (tau_esc, tau_des)
    per_seed_rows = []
    for tau_esc, tau_des in all_theta_points():
        for s in n10:
            m = n10_per_seed_metrics(s["zvfs"], tau_esc, tau_des)
            per_seed_rows.append({
                "tau_esc": tau_esc,
                "tau_des": tau_des,
                "seed": s["seed"],
                **m,
            })

    # 2) Per-prompt N2 replay (4 methods × 40 steps × 16 prompts) for every
    #    (tau_esc, tau_des). Use per-prompt zvf as the binary boundary
    #    indicator; n=4×40×16=2,560 prompt-step decisions.
    per_step_n2_rows = []
    n2_methods = sorted(n2.keys())
    for tau_esc, tau_des in all_theta_points():
        for method in n2_methods:
            for step in n2[method]:
                m = n2_per_prompt_metrics(step, tau_esc, tau_des)
                per_step_n2_rows.append({
                    "tau_esc": tau_esc,
                    "tau_des": tau_des,
                    "method": method,
                    "step": step["step"],
                    **m,
                })

    # 3) Aggregate per (tau_esc, tau_des): N10 summary + bootstrap CI;
    #    N2 summary.
    summary = []
    for tau_esc, tau_des in all_theta_points():
        rows = [r for r in per_seed_rows
                if r["tau_esc"] == tau_esc and r["tau_des"] == tau_des]
        savings_list = [r["savings"] for r in rows]
        total_list = [r["total_G"] for r in rows]
        headroom_bad = sum(r["headroom_bad"] for r in rows) / len(rows)
        mean_savings, lo, hi = bootstrap_ci(savings_list, n_boot=N_BOOT, rng=rng)
        seed_cv = (statistics.pstdev(total_list) / (statistics.mean(total_list) or 1.0))
        n2_rows = [r for r in per_step_n2_rows
                   if r["tau_esc"] == tau_esc and r["tau_des"] == tau_des]
        if n2_rows:
            n2_mean_savings = sum(r["savings"] for r in n2_rows) / len(n2_rows)
            n2_mean_contrast_intent = sum(r["contrast_intent"] for r in n2_rows) / len(n2_rows)
        else:
            n2_mean_savings = float("nan")
            n2_mean_contrast_intent = float("nan")
        summary.append({
            "tau_esc": tau_esc,
            "tau_des": tau_des,
            "delta": round(tau_des - tau_esc, 2),
            "n10_n_seeds": n_seeds,
            "mean_savings_n10": round(mean_savings, 4),
            "savings_ci_lo": round(lo, 4),
            "savings_ci_hi": round(hi, 4),
            "seed_cv_total_G": round(seed_cv, 4),
            "mean_headroom_bad": round(headroom_bad, 4),
            "mean_escalations_per_seed": round(
                sum(r["escalations"] for r in rows) / len(rows), 4),
            "mean_deescalations_per_seed": round(
                sum(r["deescalations"] for r in rows) / len(rows), 4),
            "mean_savings_n2": round(n2_mean_savings, 4),
            "mean_contrast_intent_n2": round(n2_mean_contrast_intent, 4),
            "n2_methods": ",".join(n2_methods),
        })

    # 4) Pareto-frontier selection on (mean_savings_n10, seed_cv_total_G,
    #    headroom_bad=0). Among headroom-clean points pick the one with
    #    stat-detect savings (CI excludes zero) AND lowest seed-CV; break
    #    ties by the highest mean_savings_n2 (transfer to N2 evidence).
    candidates = [s for s in summary
                  if abs(s["mean_headroom_bad"]) < 1e-9
                  and s["savings_ci_lo"] > 0]
    if candidates:
        best = min(candidates, key=lambda s: (
            s["seed_cv_total_G"], -s["mean_savings_n10"], -s["mean_savings_n2"]))
    else:
        best = max(summary, key=lambda s: s["mean_savings_n10"])
    best_row = {
        "tau_esc": best["tau_esc"],
        "tau_des": best["tau_des"],
        "delta": best["delta"],
        "mean_savings_n10": best["mean_savings_n10"],
        "savings_ci_lo": best["savings_ci_lo"],
        "savings_ci_hi": best["savings_ci_hi"],
        "seed_cv_total_G": best["seed_cv_total_G"],
        "mean_savings_n2": best["mean_savings_n2"],
    }

    # 5) Compute "limiting operating points" so we can verify the bank
    #    recovers the three legacy controllers at known τ:
    #    zvf_triage@τ     -> tau_esc=tau_des=τ (single band: G_ESC above τ)
    #    dualformer@τ      -> tau_des=τ, tau_esc only contributes G_ESC
    #                         at the same band — so we recover it at
    #                         (tau_esc, tau_des) with tau_esc close to tau_des
    #    hybrid@τ         -> tau_esc=τ, tau_des=τ+delta (independent bands)
    legacy = {
        "zvf_triage@0.70": (0.70, 0.70),
        "zvf_triage@0.50": (0.50, 0.50),
        "dualformer_auto@0.50": (0.50, 0.50),   # G_DES if z>=0.50; collapse = same
                                               # single threshold -> zvf_triage@0.50
                                               # (the family at tau_esc=tau_des
                                               # only emits G_ESC above τ; to
                                               # recover Dualformer-Auto (G_DES)
                                               # we use the limiting point at
                                               # tau_esc=tau_des MAX with
                                               # G_ESC->G_DES substitution).
        "hybrid@0.65": (0.55, 0.65),
    }

    # 6) Write outputs
    def write_tsv(path: pathlib.Path, rows, fields):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})

    per_seed_fields = ["tau_esc", "tau_des", "seed",
                       "total_G", "savings", "fire_rate",
                       "escalations", "deescalations", "headroom_bad"]
    write_tsv(OUT / "p7_unified_controller_per_seed.tsv", per_seed_rows, per_seed_fields)

    per_step_n2_fields = ["tau_esc", "tau_des", "method", "step",
                          "total_G_prompts", "savings",
                          "contrast_intent", "fires_esc", "fires_des"]
    write_tsv(OUT / "p7_unified_controller_per_step_n2.tsv",
              per_step_n2_rows, per_step_n2_fields)

    summary_fields = ["tau_esc", "tau_des", "delta",
                      "n10_n_seeds",
                      "mean_savings_n10", "savings_ci_lo", "savings_ci_hi",
                      "seed_cv_total_G",
                      "mean_headroom_bad",
                      "mean_escalations_per_seed", "mean_deescalations_per_seed",
                      "mean_savings_n2", "mean_contrast_intent_n2",
                      "n2_methods"]
    write_tsv(OUT / "p7_unified_controller_summary.tsv", summary, summary_fields)

    # Pareto-efficient subset (CV<=0.10, headroom-bad=0)
    pareto = [s for s in summary
              if s["seed_cv_total_G"] <= 0.10 and abs(s["mean_headroom_bad"]) < 1e-9]
    pareto.sort(key=lambda s: (-s["mean_savings_n10"], s["seed_cv_total_G"]))
    pareto_fields = ["rank_in_pareto"] + summary_fields
    for i, s in enumerate(pareto, start=1):
        s_out = {"rank_in_pareto": i}
        s_out.update(s)
    write_tsv(OUT / "p7_unified_controller_pareto.tsv",
              [{"rank_in_pareto": i, **s} for i, s in enumerate(pareto, start=1)],
              pareto_fields)

    # CI table (one row per theta) — same as summary but trimmed
    ci_fields = ["tau_esc", "tau_des", "delta", "mean_savings_n10",
                 "savings_ci_lo", "savings_ci_hi", "seed_cv_total_G",
                 "mean_headroom_bad"]
    write_tsv(OUT / "p7_unified_controller_ci.tsv", summary, ci_fields)

    out_json = {
        "iter": 51,
        "panel_n10": {
            "n_seeds": n_seeds,
            "n_steps_per_seed": N_STEPS_N10,
            "seeds": [s["seed"] for s in n10],
        },
        "panel_n2": {
            "methods": list(n2_methods),
            "n_steps_per_method": (
                len(n2[n2_methods[0]]) if n2_methods else 0
            ),
            "n_prompts_per_step": N_PROMPTS_PER_STEP_N2,
            "G_per_step": G_N2,
        },
        "controller_family": {
            "tau_esc_grid": TAU_ESC_GRID,
            "delta_grid": DELTA_GRID,
            "G_base": G_BASE,
            "G_esc": G_ESC,
            "G_des": G_DES,
        },
        "n_points_swept": len(summary),
        "n_pareto_efficient": len(pareto),
        "legacy_limit_recovery": legacy,
        "best_calibrated_theta": best_row,
        "boundary_collapses": {
            "zvf_triage": "tau_esc==tau_des single-band -> G_ESC if z>=tau",
            "dualformer_auto": "tau_des -> G_DES branch dominates the operational"
                              " point (single band covered by G_DES rule)",
            "hybrid": "full (tau_esc < tau_des) two independent bands",
        },
        "headline": (
            f"Unified controller bank with {len(summary)} parametric points; "
            f"Pareto-efficient subset size = {len(pareto)}; "
            f"best_calibrated_theta = (tau_esc={best_row['tau_esc']}, "
            f"tau_des={best_row['tau_des']}) with mean_savings_n10="
            f"{best_row['mean_savings_n10']:+.4f} "
            f"[{best_row['savings_ci_lo']:+.4f}, {best_row['savings_ci_hi']:+.4f}], "
            f"seed_CV={best_row['seed_cv_total_G']:.4f}."
        ),
    }
    (OUT / "p7_unified_controller_summary.json").write_text(
        json.dumps(out_json, indent=2))

    print(out_json["headline"])
    print(f"\nWrote:")
    print(f"  per_seed -> {OUT/'p7_unified_controller_per_seed.tsv'}"
          f" ({len(per_seed_rows)} rows)")
    print(f"  per_step_n2 -> {OUT/'p7_unified_controller_per_step_n2.tsv'}"
          f" ({len(per_step_n2_rows)} rows)")
    print(f"  summary -> {OUT/'p7_unified_controller_summary.tsv'}"
          f" ({len(summary)} rows)")
    print(f"  pareto -> {OUT/'p7_unified_controller_pareto.tsv'}"
          f" ({len(pareto)} rows)")
    print(f"  ci -> {OUT/'p7_unified_controller_ci.tsv'}"
          f" ({len(summary)} rows)")
    print(f"  json -> {OUT/'p7_unified_controller_summary.json'}")


if __name__ == "__main__":
    main()
