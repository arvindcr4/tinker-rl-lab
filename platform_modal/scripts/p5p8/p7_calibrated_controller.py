#!/usr/bin/env python3
"""P7 Calibrated Controller — bootstrap CIs on every N2 headline +
hybrid (Dualformer-Auto + zvf-triage + fixed-G) Pareto evaluation.

Inputs:
  experiments/results/n2_reward_tensor_resume/{grpo,aero,gift,areal}_s0_tensors.jsonl

Outputs:
  experiments/results/p5p8/p7_headline_cis.tsv    -- per-method bootstrap
    CI for mean zvf / reward_mean / pcd / lag1_autocorr / mean_len / loss.
  experiments/results/p5p8/p7_calibrated_controller.tsv
    -- per-(method, controller) cost_ratio with bootstrap CI, plus Pareto.
  experiments/results/p5p8/p7_calibrated_controller.json
    -- machine-readable (hybrid rule, headline table, Pareto).

Stdlib only. <= 300 LoC.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import pathlib
import random
import statistics

WORKTREE = pathlib.Path("/home/claude/tinker-rl-lab-minimax")
N2 = WORKTREE / "experiments" / "results" / "n2_reward_tensor_resume"
OUT = WORKTREE / "experiments" / "results" / "p5p8"
METHODS = ("grpo", "aero", "gift", "areal")
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
G_BASE = 8
G_ALT = 16
N_PROMPTS = 16
MAX_PCD = 0.20            # interior-regime guard
N_BOOT = 10_000
RNG = random.Random(20260704)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_tensors():
    """Return dict[method] -> list[step_dict] in step order."""
    out = {m: [] for m in METHODS}
    for m in METHODS:
        with (N2 / f"{m}_s0_tensors.jsonl").open() as f:
            for line in f:
                d = json.loads(line)
                out[m].append(d)
    for m in METHODS:
        out[m].sort(key=lambda d: d["step"])
    return out


def per_prompt_p(step):
    return [sum(g) / len(g) for g in step["rewards"]]


def zvf_per_p_at_g(per_p, g):
    """Per-prompt expected ZVF under i.i.d. binomial at group size g."""
    return [(1 - pp) ** g + pp ** g for pp in
            (min(max(p, 1e-9), 1 - 1e-9) for p in per_p)]


def is_degenerate_at_g(per_p, g, eps=1e-9):
    """True iff per-prompt expected ZVF at g is >= 0.99 (boundary)."""
    return [zvf >= 0.99 for zvf in zvf_per_p_at_g(per_p, g)]


# ---------------------------------------------------------------------------
# Bootstrap CI (percentile)
# ---------------------------------------------------------------------------
def boot_ci(values, stat=statistics.mean, n_boot=N_BOOT, alpha=0.05, rng=RNG):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(stat([values[i] for i in idx]))
    boots.sort()
    lo = boots[int(alpha / 2 * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot)]
    point = stat(values)
    return point, lo, hi, n


# ---------------------------------------------------------------------------
# Controllers — operate on a step dict
# ---------------------------------------------------------------------------
def cost_fixed_g():
    """Baseline rollouts: 40 steps x 16 prompts x 8."""
    return N_PROMPTS * G_BASE


def ctrl_fixed_g(step):
    return [G_BASE] * N_PROMPTS


def ctrl_zvf_triage(step, threshold):
    """Per-step: if zvf >= threshold AND pcd <= MAX_PCD, escalate to G_ALT."""
    fires = step["zvf"] >= threshold and step["pcd"] <= MAX_PCD
    g = G_ALT if fires else G_BASE
    return [g] * N_PROMPTS


def ctrl_dualformer_auto(step):
    """Per-prompt difficulty-gated G (Berkeley row 01)."""
    out = []
    for p in per_prompt_p(step):
        if p >= 0.95:
            out.append(2)
        elif p >= 0.85:
            out.append(4)
        elif p >= 0.70:
            out.append(8)
        else:
            out.append(16)
    return out


def ctrl_hybrid(step, threshold=0.70):
    """Hybrid: at boundary (PCD > 0.20) shrink per-prompt (Dualformer-Auto);
    at interior (PCD <= 0.20) AND zvf >= threshold escalate (zvf-triage);
    else fixed-G."""
    pcd = step["pcd"]
    zvf = step["zvf"]
    if pcd > MAX_PCD:
        return ctrl_dualformer_auto(step)
    if zvf >= threshold:
        return [G_ALT] * N_PROMPTS
    return [G_BASE] * N_PROMPTS


def rollout_cost(g_per_p):
    return sum(g_per_p)


def headroom_saved(per_p, g_per_p):
    """saved = currently degenerate at G_BASE but recovered at g_per_p."""
    base_deg = is_degenerate_at_g(per_p, G_BASE)
    new_zvf = zvf_per_p_at_g(per_p, max(g_per_p))
    saved = 0
    wasted = 0
    for i in range(N_PROMPTS):
        if base_deg[i]:
            if new_zvf[i] < 0.99:
                saved += 1
            else:
                wasted += 1
    return saved, wasted


# ---------------------------------------------------------------------------
# Headline CIs: per-method bootstrap over the 40-step trajectory
# ---------------------------------------------------------------------------
def headline_cis(tensors):
    rows = []
    for m in METHODS:
        steps = tensors[m]
        n = len(steps)
        for metric in ("zvf", "reward_mean", "pcd", "lag1_autocorr",
                       "mean_len", "loss"):
            vals = [s[metric] for s in steps if s[metric] == s[metric]]  # drop NaN
            if not vals:
                rows.append({"method": m, "metric": metric, "n": 0,
                             "point": "NaN", "lo": "NaN", "hi": "NaN"})
                continue
            pt, lo, hi, nboot = boot_ci(vals)
            rows.append({
                "method": m, "metric": metric, "n": nboot,
                "point": round(pt, 4),
                "lo": round(lo, 4),
                "hi": round(hi, 4),
            })
    return rows


# ---------------------------------------------------------------------------
# Controller cost evaluation with bootstrap CI
# ---------------------------------------------------------------------------
def evaluate_controllers(tensors, controllers):
    """returns dict[(method, ctrl_name)] -> {cost_ratio, saved, wasted, fires,
    per_step_cost}."""
    out = {}
    for m in METHODS:
        steps = tensors[m]
        for name, fn in controllers.items():
            per_step_costs = []
            total_saved = 0
            total_wasted = 0
            total_fires = 0
            for s in steps:
                g_per_p = fn(s)
                per_step_costs.append(rollout_cost(g_per_p))
                saved, wasted = headroom_saved(per_prompt_p(s), g_per_p)
                total_saved += saved
                total_wasted += wasted
                # fires = count of non-baseline-G prompts
                fires = sum(1 for g in g_per_p if g != G_BASE)
                total_fires += fires
            baseline = N_PROMPTS * G_BASE * len(steps)
            cost = sum(per_step_costs)
            ratio = cost / baseline
            out[(m, name)] = {
                "cost_ratio": ratio,
                "saved": total_saved,
                "wasted": total_wasted,
                "fires": total_fires,
                "per_step_cost": per_step_costs,
            }
    return out


def bootstrap_cost_ci(per_step_costs, baseline, n_boot=N_BOOT, rng=RNG):
    """Bootstrap CI on the cost ratio by resampling per-step costs."""
    n = len(per_step_costs)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        ratio = sum(per_step_costs[i] for i in idx) / baseline
        boots.append(ratio)
    boots.sort()
    point = sum(per_step_costs) / baseline
    return point, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def ctrl_hybrid_no_triage(step):
    """Hybrid without zvf-triage: only Dualformer at boundary, fixed-G elsewhere.
    Tests whether the escalation component of the full hybrid is even needed."""
    if step["pcd"] > MAX_PCD:
        return ctrl_dualformer_auto(step)
    return [G_BASE] * N_PROMPTS


def build_controllers():
    out = {
        "fixed_g": ctrl_fixed_g,
        "zvf_triage_0.50": lambda s: ctrl_zvf_triage(s, 0.50),
        "zvf_triage_0.70": lambda s: ctrl_zvf_triage(s, 0.70),
        "zvf_triage_0.80": lambda s: ctrl_zvf_triage(s, 0.80),
        "zvf_triage_0.90": lambda s: ctrl_zvf_triage(s, 0.90),
        "dualformer_auto": ctrl_dualformer_auto,
        "hybrid_0.70": ctrl_hybrid,
        "hybrid_no_triage": ctrl_hybrid_no_triage,
    }
    return out


def regime_split(tensors):
    """For each method, count steps falling into each regime quadrant:
       interior_low (PCD<=0.20, zvf<0.70), interior_high (PCD<=0.20, zvf>=0.70),
       boundary_low (PCD>0.20, zvf<0.70), boundary_high (PCD>0.20, zvf>=0.70)."""
    out = {}
    for m in METHODS:
        c = {"interior_low": 0, "interior_high": 0,
             "boundary_low": 0, "boundary_high": 0}
        for s in tensors[m]:
            pcd = s["pcd"]
            zvf = s["zvf"]
            if pcd > MAX_PCD:
                c["boundary_high" if zvf >= 0.70 else "boundary_low"] += 1
            else:
                c["interior_high" if zvf >= 0.70 else "interior_low"] += 1
        out[m] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    tensors = load_tensors()
    n_steps = len(tensors[METHODS[0]])
    baseline = N_PROMPTS * G_BASE * n_steps

    # ----- Headline CIs --------------------------------------------------
    headlines = headline_cis(tensors)

    # ----- Controller eval ----------------------------------------------
    controllers = build_controllers()
    evals = evaluate_controllers(tensors, controllers)

    # ----- Bootstrap CI on cost ratio per (method, controller) -----------
    cost_rows = []
    for (m, name), v in evals.items():
        pt, lo, hi = bootstrap_cost_ci(v["per_step_cost"], baseline)
        cost_rows.append({
            "method": m,
            "controller": name,
            "fires": v["fires"],
            "saved": v["saved"],
            "wasted": v["wasted"],
            "cost_ratio_pt": round(pt, 3),
            "cost_ratio_lo": round(lo, 3),
            "cost_ratio_hi": round(hi, 3),
        })

    # ----- Pareto summary (mean across methods) -------------------------
    pareto = {}
    for name in controllers:
        rs = [r for r in cost_rows if r["controller"] == name]
        ratios = [r["cost_ratio_pt"] for r in rs]
        saves = [r["saved"] for r in rs]
        pareto[name] = {
            "mean_cost_ratio": round(statistics.mean(ratios), 3),
            "mean_saved": round(statistics.mean(saves), 2),
            "max_saved": max(saves),
        }

    # ----- Regime split (where would each controller activate?) -----------
    regimes = regime_split(tensors)

    # ----- Write ---------------------------------------------------------
    if args.write:
        # Headline CIs TSV
        cols_h = ["method", "metric", "n", "point", "lo", "hi"]
        with (OUT / "p7_headline_cis.tsv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=cols_h, delimiter="\t")
            w.writeheader()
            for r in headlines:
                w.writerow(r)

        # Cost-ratio CIs TSV
        cols_c = ["method", "controller", "fires", "saved", "wasted",
                  "cost_ratio_pt", "cost_ratio_lo", "cost_ratio_hi"]
        with (OUT / "p7_calibrated_controller.tsv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=cols_c, delimiter="\t")
            w.writeheader()
            for r in cost_rows:
                w.writerow(r)

        # JSON with Pareto
        out = {
            "headlines": headlines,
            "controllers": cost_rows,
            "pareto": pareto,
            "regime_split": regimes,
            "baseline": baseline,
            "n_steps": n_steps,
            "n_prompts_per_step": N_PROMPTS,
            "g_base": G_BASE,
            "g_alt": G_ALT,
            "max_pcd_guard": MAX_PCD,
            "n_boot": N_BOOT,
            "hybrid_rule": (
                "Hybrid(threshold): if step.pcd > MAX_PCD (boundary regime) "
                "shrink per-prompt G via Dualformer-Auto rule; else if "
                "step.zvf >= threshold escalate to G_ALT=16 (zvf-triage); "
                "else fixed-G=8. MAX_PCD=0.20, threshold=0.70 default."
            ),
            "interpretation": (
                "Three calibrated headline signals: (1) mean ZVF per method "
                "(saturated-prompt regime where zvf-triage has 0 headroom); "
                "(2) controller cost_ratio with bootstrap CI on the per-step "
                "rollout cost; (3) Pareto table comparing 8 controllers. "
                "The hybrid rule combines the Dualformer-Auto shrinkage at "
                "the boundary and the zvf-triage escalation at the interior. "
                "On the N2 data the regime is almost entirely interior_high "
                "(zvf >= 0.70 AND pcd <= 0.20), so the hybrid degenerates to "
                "zvf-triage and its cost_ratio matches zvf-triage_0.70 exactly."
            ),
        }
        (OUT / "p7_calibrated_controller.json").write_text(
            json.dumps(out, indent=2))
        print(f"wrote {OUT}/p7_headline_cis.tsv")
        print(f"wrote {OUT}/p7_calibrated_controller.{{tsv,json}}")

    # ----- Console ---------------------------------------------------------
    print("\n=== P7 headline bootstrap CIs (per-method, n=40 steps) ===")
    print(f"{'method':6s} {'metric':16s} {'n':>4s} "
          f"{'point':>8s} {'lo':>8s} {'hi':>8s}")
    print("-" * 60)
    for r in headlines:
        print(f"{r['method']:6s} {r['metric']:16s} {r['n']:>4d} "
              f"{r['point']:>8} {r['lo']:>8} {r['hi']:>8}")

    print("\n=== Controller cost ratio with bootstrap CI (95%, n_boot=10000) ===")
    print(f"{'method':6s} {'controller':18s} "
          f"{'fires':>5s} {'saved':>5s} {'wasted':>6s} "
          f"{'ratio_pt':>9s} {'CI_lo':>6s} {'CI_hi':>6s}")
    print("-" * 75)
    for r in cost_rows:
        print(f"{r['method']:6s} {r['controller']:18s} "
              f"{r['fires']:>5d} {r['saved']:>5d} {r['wasted']:>6d} "
              f"{r['cost_ratio_pt']:>9.3f} "
              f"{r['cost_ratio_lo']:>6.3f} {r['cost_ratio_hi']:>6.3f}")

    print("\n=== Pareto (mean across methods) ===")
    print(f"{'controller':20s} {'mean_ratio':>10s} {'mean_saved':>10s} "
          f"{'max_saved':>9s}")
    print("-" * 55)
    for name, v in sorted(pareto.items(), key=lambda kv: kv[1]["mean_cost_ratio"]):
        print(f"{name:20s} {v['mean_cost_ratio']:>10.3f} "
              f"{v['mean_saved']:>10.2f} {v['max_saved']:>9d}")

    print("\n=== Regime split (40 steps per method) ===")
    print(f"{'method':6s} {'interior_low':>12s} {'interior_high':>13s} "
          f"{'boundary_low':>12s} {'boundary_high':>13s}")
    print("-" * 70)
    for m, c in regimes.items():
        print(f"{m:6s} {c['interior_low']:>12d} {c['interior_high']:>13d} "
              f"{c['boundary_low']:>12d} {c['boundary_high']:>13d}")


if __name__ == "__main__":
    main()