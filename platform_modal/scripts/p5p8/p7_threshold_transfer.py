#!/usr/bin/env python3
"""P7 Cross-Method Trigger-Threshold Transfer.

Question:  when τ is *tuned* on method-A (using method-A's own per-step
ZVF trajectory), how does it perform on method-B? Does τ*_A generalise to
B/C/D, or does each method need its own τ?

This is a clean cross-stack generalisation test for the calibrated
controller and an honest probe of whether the seed-robust τ ∈ [0.70,
0.80] operating range (N10 five-seed, Section~\\ref{sec:p7-controller-seedrobust})
holds when the "seed" axis is replaced by the "method" axis.

Inputs:
  platform_hybrid/experiments/results/n2_reward_tensor_resume/{grpo,aero,gift,areal}_s0_tensors.jsonl

Outputs:
  platform_hybrid/experiments/results/p5p8/p7_threshold_transfer_summary.tsv
    -- per-(source, target, τ) Pareto cell: fires, cost_ratio, saved,
       wasted, plus 95% bootstrap CI on cost_ratio from per-step
       resampling of the (target, τ) trajectory.
  platform_hybrid/experiments/results/p5p8/p7_threshold_transfer_per_step.tsv
    -- one row per (source, target, τ, step): zvf_A (tuning source's
       ZVF on step s), zvf_B (test target's ZVF on step s), fire,
       rollouts.
  platform_hybrid/experiments/results/p5p8/p7_threshold_transfer_summary.json
    -- machine-readable headline.

Stdlib only. <= 300 LoC. seed=20260704, n_boot=2000.
"""
from __future__ import annotations
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
THRESHOLDS = (0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
G_BASE = 8
G_ESC = 16
N_PROMPTS = 16
N_STEPS = 40
BASELINE_ROLLOUTS = N_STEPS * N_PROMPTS * G_BASE  # 5,120
N_BOOT = 2000
RNG = random.Random(20260704)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_tensors():
    """Return dict[method] -> list[step_records] sorted by step idx."""
    out = {m: [] for m in METHODS}
    for m in METHODS:
        path = N2 / f"{m}_s0_tensors.jsonl"
        with path.open() as f:
            for line in f:
                d = json.loads(line)
                out[m].append(d)
    for m in METHODS:
        out[m].sort(key=lambda d: d["step"])
    return out


def zvf_per_p(step):
    """Per-prompt p̂ estimate from G=8 binary rewards."""
    return [sum(g) / len(g) for g in step["rewards"]]


def zvf_pred_at_g(p, g):
    """i.i.d. binomial expected ZVF at group size g given empirical p̂."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    return (1 - p) ** g + p ** g


# ---------------------------------------------------------------------------
# Trigger semantics (step-level zvf-triage with interior-regime guard)
# ---------------------------------------------------------------------------
# Per the iter-31 / iter-43 / iter-47 design: the step fires iff (a) the
# aggregate step-zvf >= τ AND (b) the step is in the interior regime
# (PCD <= MAX_PCD). Burn-in (step < MIN_STEP) silences the controller
# to avoid early-trajectory noise.
MAX_PCD = 0.20
MIN_STEP = 2


def fires(step, tau):
    """Return True iff step's aggregate ZVF exceeds τ (with PCD guard)."""
    if step["step"] < MIN_STEP:
        return False
    if step.get("pcd", 0.0) > MAX_PCD:
        return False
    return step.get("zvf", 0.0) >= tau


def cost_ratio_for(fires_seq):
    """Total rollouts used under the controller over a 40-step trajectory."""
    if not fires_seq:
        return 1.0
    n_fires = sum(fires_seq)
    rollouts = (N_STEPS - n_fires) * N_PROMPTS * G_BASE + n_fires * N_PROMPTS * G_ESC
    return rollouts / BASELINE_ROLLOUTS


# ---------------------------------------------------------------------------
# Per-prompt-headroom accounting
# ---------------------------------------------------------------------------
def headroom(step, g_esc=G_ESC):
    """Number of currently-degenerate prompts with predicted ZVF<0.99 at g_esc."""
    pp = zvf_per_p(step)
    base_degen = sum(1 for p in pp if zvf_pred_at_g(p, G_BASE) >= 0.99)
    esc_degen = sum(1 for p in pp if zvf_pred_at_g(p, g_esc) >= 0.99)
    return base_degen - esc_degen  # saved prompts at g_esc (>= 0)


def wasted(step, g_esc=G_ESC):
    """Prompts already degenerate at G=8 AND still degenerate at g_esc."""
    pp = zvf_per_p(step)
    base_degen = sum(1 for p in pp if zvf_pred_at_g(p, G_BASE) >= 0.99)
    esc_still_degen = sum(
        1 for p in pp
        if zvf_pred_at_g(p, G_BASE) >= 0.99 and zvf_pred_at_g(p, g_esc) >= 0.99
    )
    return esc_still_degen


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
def boot_ci(values, stat, n_boot=N_BOOT, alpha=0.05, rng=RNG):
    """Percentile bootstrap CI."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = []
    for _ in range(n_boot):
        s = stat([values[rng.randrange(n)] for _ in range(n)])
        boots.append(s)
    boots.sort()
    lo = boots[int(alpha / 2 * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    mean = sum(boots) / len(boots)
    return mean, lo, hi


# ---------------------------------------------------------------------------
# Core: per-method Pareto over τ
# ---------------------------------------------------------------------------
def per_method_curve(method_steps):
    """For each τ, return (fires_per_step, cost_ratio, headroom_total)."""
    out = {}
    for tau in THRESHOLDS:
        fires_seq = [fires(s, tau) for s in method_steps]
        ratio = cost_ratio_for(fires_seq)
        saved = sum(headroom(s) for s in method_steps if fires(s, tau))
        wasted_prompts = sum(wasted(s) for s in method_steps if fires(s, tau))
        out[tau] = {
            "fires_total": sum(fires_seq),
            "cost_ratio": ratio,
            "saved_total": saved,
            "wasted_total": wasted_prompts,
        }
    return out


# ---------------------------------------------------------------------------
# Cross-method transfer: source → target
# ---------------------------------------------------------------------------
def transfer_table(steps):
    """Build per-(source, target, τ) cells.

    *τ_selected* uses the source method's Pareto-best τ by minimum cost.
    Then we apply that same τ to the target's trajectory and compare.
    """
    summary_rows = []
    per_step_rows = []
    curves = {m: per_method_curve(steps[m]) for m in METHODS}

    for source in METHODS:
        # pick the τ that minimises cost_ratio while saving >= 1 prompt
        # (i.e. the "true" Pareto operating point on the source).
        best_tau = None
        best_cost = math.inf
        best_saved = 0
        for tau in THRESHOLDS:
            c = curves[source][tau]
            if c["saved_total"] >= 1 and c["cost_ratio"] < best_cost:
                best_cost = c["cost_ratio"]
                best_tau = tau
                best_saved = c["saved_total"]
        # worst-case (highest-firing) τ on source for comparison
        worst_tau = max(THRESHOLDS, key=lambda t: curves[source][t]["fires_total"])
        # marker for the τ that the source itself selected (could be None)
        best_tau_eff = best_tau if best_tau is not None else -1.0

        for target in METHODS:
            for tau in THRESHOLDS:
                fires_seq = [fires(s, tau) for s in steps[target]]
                ratio = cost_ratio_for(fires_seq)
                ratio_mean, ratio_lo, ratio_hi = boot_ci(
                    [1.0 if f else 0.0 for f in fires_seq],
                    stat=lambda xs: cost_ratio_for(xs),
                    n_boot=N_BOOT,
                )
                saved = sum(headroom(s) for s in steps[target] if fires(s, tau))
                wasted_prompts = sum(wasted(s) for s in steps[target] if fires(s, tau))
                summary_rows.append({
                    "source": source,
                    "target": target,
                    "tau": f"{tau:.2f}",
                    "is_source_self": "1" if source == target else "0",
                    "is_source_selected": "1" if abs(tau - best_tau_eff) < 1e-9 else "0",
                    "fires": sum(fires_seq),
                    "cost_ratio": f"{ratio:.4f}",
                    "cost_ratio_ci_lo": f"{ratio_lo:.4f}",
                    "cost_ratio_ci_hi": f"{ratio_hi:.4f}",
                    "saved": saved,
                    "wasted": wasted_prompts,
                })
            # additionally record the headline transfer cell (best_tau)
            for s in steps[target]:
                per_step_rows.append({
                    "source": source,
                    "source_best_tau": f"{best_tau:.2f}" if best_tau is not None else "n/a",
                    "source_worst_tau": f"{worst_tau:.2f}",
                    "target": target,
                    "step": s["step"],
                    "zvf_target": f"{s.get('zvf', 0.0):.6f}",
                    "zvf_source": f"{steps[source][s['step']].get('zvf', 0.0):.6f}",
                    "fires_at_source_best": (
                        "1" if (best_tau is not None and fires(s, best_tau)) else "0"
                    ),
                    "fires_at_source_worst": (
                        "1" if (worst_tau is not None and fires(s, worst_tau)) else "0"
                    ),
                })
    return summary_rows, per_step_rows, curves


# ---------------------------------------------------------------------------
# Per-method optimal τ detection and transfer-penalty accounting
# ---------------------------------------------------------------------------
def transfer_penalty_table(steps, curves):
    """For each (source, target, τ*), report Δcost_ratio vs target-optimal τ*."""
    def pick_best(method):
        cands = [t for t in THRESHOLDS if curves[method][t]["saved_total"] >= 1]
        if not cands:
            return None
        return min(cands, key=lambda t: curves[method][t]["cost_ratio"])

    rows = []
    for source in METHODS:
        source_best = pick_best(source)
        for target in METHODS:
            target_best = pick_best(target)
            if source_best is None and target_best is None:
                # both methods have no savings at any threshold; report
                # trivial zero-penalty cell.
                rows.append({
                    "source": source, "target": target,
                    "tau_used": "n/a", "tau_target_opt": "n/a",
                    "cost_used": "1.0000", "cost_target_opt": "1.0000",
                    "transfer_penalty": "+0.0000",
                })
                continue
            # If source has no candidate, fall back to target's best τ
            tau_use = source_best if source_best is not None else target_best
            ratio_use = curves[target][tau_use]["cost_ratio"]
            ratio_target_opt = curves[target][target_best]["cost_ratio"] \
                if target_best is not None else 1.0
            penalty = ratio_use - ratio_target_opt
            rows.append({
                "source": source,
                "target": target,
                "tau_used": f"{tau_use:.2f}",
                "tau_target_opt": f"{target_best:.2f}" if target_best else "n/a",
                "cost_used": f"{ratio_use:.4f}",
                "cost_target_opt": f"{ratio_target_opt:.4f}",
                "transfer_penalty": f"{penalty:+.4f}",
            })
    return rows


# ---------------------------------------------------------------------------
# Cross-method fire-decision agreement (Jaccard)
# ---------------------------------------------------------------------------
def fire_decisions(steps, method, tau):
    """Boolean vector fires(s_i, tau) over the 40 steps for method."""
    return [fires(s, tau) for s in steps[method]]


def jaccard(a, b):
    inter = sum(1 for x, y in zip(a, b) if x and y)
    union = sum(1 for x, y in zip(a, b) if x or y)
    if union == 0:
        return float("nan")
    return inter / union


def cross_method_agreement(steps, taus=(0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)):
    """Per-τ pairwise Jaccard agreement of fire decisions across methods."""
    out = []
    for tau in taus:
        decs = {m: fire_decisions(steps, m, tau) for m in METHODS}
        for s in METHODS:
            for t in METHODS:
                if s >= t:
                    continue
                j = jaccard(decs[s], decs[t])
                out.append({
                    "tau": f"{tau:.2f}",
                    "method_a": s,
                    "method_b": t,
                    "fires_a": sum(decs[s]),
                    "fires_b": sum(decs[t]),
                    "jaccard": f"{j:.4f}" if not math.isnan(j) else "n/a",
                })
    return out


def fixed_tau_transfer(steps, taus=(0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)):
    """For each τ, report per-method cost_ratio and the cross-method range/SD."""
    out = []
    for tau in taus:
        per_method = {m: cost_ratio_for(fire_decisions(steps, m, tau)) for m in METHODS}
        vals = [per_method[m] for m in METHODS]
        mean = sum(vals) / len(vals)
        sd = statistics.pstdev(vals)
        rng = max(vals) - min(vals)
        out.append({
            "tau": f"{tau:.2f}",
            "grpo": f"{per_method['grpo']:.4f}",
            "aero": f"{per_method['aero']:.4f}",
            "gift": f"{per_method['gift']:.4f}",
            "areal": f"{per_method['areal']:.4f}",
            "mean": f"{mean:.4f}",
            "sd": f"{sd:.4f}",
            "range": f"{rng:.4f}",
            "max_minus_min_pct": f"{(rng / mean * 100):.2f}%",
        })
    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_tsv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    steps = load_tensors()
    summary_rows, per_step_rows, curves = transfer_table(steps)
    penalty_rows = transfer_penalty_table(steps, curves)
    agreement_rows = cross_method_agreement(steps)
    fixed_tau_rows = fixed_tau_transfer(steps)

    OUT.mkdir(parents=True, exist_ok=True)
    write_tsv(OUT / "p7_threshold_transfer_summary.tsv", summary_rows)
    write_tsv(OUT / "p7_threshold_transfer_per_step.tsv", per_step_rows)
    write_tsv(OUT / "p7_threshold_transfer_penalty.tsv", penalty_rows)
    write_tsv(OUT / "p7_threshold_transfer_agreement.tsv", agreement_rows)
    write_tsv(OUT / "p7_threshold_transfer_fixed_tau.tsv", fixed_tau_rows)

    # also pretty-print the source/target cost matrix at τ=0.70
    matrix = {t: {} for t in THRESHOLDS}
    for r in summary_rows:
        if r["source"] == r["target"]:
            continue
        matrix[float(r["tau"])][(r["source"], r["target"])] = (
            r["cost_ratio"], r["fires"], r["saved"]
        )

    json_out = {
        "n_steps": N_STEPS,
        "n_prompts_per_step": N_PROMPTS,
        "thresholds": list(THRESHOLDS),
        "methods": list(METHODS),
        "baseline_rollouts": BASELINE_ROLLOUTS,
        "transfer_matrix_per_tau": {
            f"{t:.2f}": {
                f"{s}->{t2}": {"cost_ratio": matrix[t][(s, t2)][0],
                                "fires": matrix[t][(s, t2)][1],
                                "saved": matrix[t][(s, t2)][2]}
                for s in METHODS for t2 in METHODS if s != t2
                  and (s, t2) in matrix[t]
            }
            for t in THRESHOLDS
        },
        "per_source_best_tau": {
            m: (min(
                (t for t in THRESHOLDS if curves[m][t]["saved_total"] >= 1),
                key=lambda t: curves[m][t]["cost_ratio"],
            ) if any(curves[m][t]["saved_total"] >= 1 for t in THRESHOLDS)
                else None)
            for m in METHODS
        },
        "transfer_penalty_rows": penalty_rows,
        "cross_method_agreement_rows": agreement_rows,
        "fixed_tau_transfer_rows": fixed_tau_rows,
    }
    (OUT / "p7_threshold_transfer_summary.json").write_text(
        json.dumps(json_out, indent=2)
    )

    print(f"wrote {OUT/'p7_threshold_transfer_summary.tsv'}")
    print(f"wrote {OUT/'p7_threshold_transfer_per_step.tsv'}")
    print(f"wrote {OUT/'p7_threshold_transfer_penalty.tsv'}")
    print(f"wrote {OUT/'p7_threshold_transfer_agreement.tsv'}")
    print(f"wrote {OUT/'p7_threshold_transfer_fixed_tau.tsv'}")
    print(f"wrote {OUT/'p7_threshold_transfer_summary.json'}")
    print()
    print("Per-method best τ and resulting cost_ratio (source target/trained):")
    for m in METHODS:
        best = json_out["per_source_best_tau"][m]
        if best is None:
            print(f"  {m}: no τ saves any prompt at any threshold")
        else:
            print(
                f"  {m}: best τ = {best:.2f}  → cost_ratio = {curves[m][best]['cost_ratio']:.4f}, "
                f"saved={curves[m][best]['saved_total']}, fires={curves[m][best]['fires_total']}"
            )
    print()
    print("Fixed-τ cross-method transfer (cost_ratio per method, mean, range):")
    for r in fixed_tau_rows:
        print(
            f"  τ={r['tau']}: {r['grpo']}/{r['aero']}/{r['gift']}/{r['areal']} "
            f" mean={r['mean']} sd={r['sd']} range={r['range']} ({r['max_minus_min_pct']})"
        )
    print()
    print("Pairwise Jaccard agreement of fire decisions across methods at τ=0.70:")
    for r in agreement_rows:
        if r["tau"] != "0.70":
            continue
        print(
            f"  {r['method_a']} ↔ {r['method_b']}: "
            f"fires_a={r['fires_a']} fires_b={r['fires_b']} jaccard={r['jaccard']}"
        )


if __name__ == "__main__":
    main()
