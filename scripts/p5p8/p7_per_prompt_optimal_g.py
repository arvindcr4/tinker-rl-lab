#!/usr/bin/env python3
"""P7 Per-Prompt Hindsight-Optimal G' Analysis on real N2 reward tensors.

Vein (a) of the iter-35 brief: ``when would it have fired, what G would it have
chosen, what contrast would it have restored?''

For each (method, step) x 16 prompts, we look at the per-prompt observed
rewards at G=8 and ask: ``what is the smallest group size G' that would have
broken the all-correct/all-wrong degeneracy for this prompt under the i.i.d.
binomial model?''

Decision rule (per prompt-step):
  p_hat = k / G_BASE where k = number of 1-rewards observed out of G_BASE = 8.
  - if k == 0 or k == G_BASE: no G helps (saturated), G* = G_BASE = 8.
  - else: G* = argmin_g G' s.t. ZVF_iid(p_hat, G') = p_hat**G' + (1-p_hat)**G' < 0.99
    (searched over g in {4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64}; G'=8 if none qualifies
    or if G' = G_BASE already qualifies).

Aggregates:
  - distribution of optimal G* per prompt-step across all 4 methods
  - mean G*, mean contrast restored = 1 - E[ZVF_iid(p_hat, G*) | G* = argmin]
  - cost ratio = mean G* / G_BASE
  - Pareto frontier of fixed-G (4, 6, 8, 12, 16, 24, 32) vs per-prompt optimal G*
  - savings over best-fixed-G baseline (mean rollouts/saved-prompt)

Outputs (under experiments/results/p5p8/):
  p7_per_prompt_optimal_g_summary.tsv   -- one row per (method, candidate_G, candidate_controller)
                                          with mean_G_*, cost_ratio, mean_ZVF_restored, total rollouts
  p7_per_prompt_optimal_g_per_step.tsv  -- one row per (method, step) with the distribution of G* across 16 prompts
  p7_per_prompt_optimal_g_per_prompt.tsv -- one row per (method, step, prompt_index) with k, p_hat, G*, ZVF_at_G*, ZVF_at_G=8
  p7_per_prompt_optimal_g_summary.json  -- machine-readable summary
  figures/p7_per_prompt_g_distribution.{png,pdf} -- bar chart of G* distribution by method

Stdlib only (with matplotlib for figures).
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
FIG = OUT / "figures"
METHODS = ("grpo", "aero", "gift", "areal")
G_BASE = 8
G_CANDIDATES = (4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64)
N_PROMPTS = 16
N_STEPS = 40
TARGET_ZVF = 0.99  # at this level the binomial model says a prompt is "non-degenerate"
SEED = 20260704


def zvf_iid(p_hat: float, g_new: int) -> float:
    """Expected ZVF at group size g_new under i.i.d. binomial model with p=p_hat."""
    pp = min(max(p_hat, 1e-12), 1.0 - 1e-12)
    return pp**g_new + (1.0 - pp) ** g_new


def is_saturated(k: int, g: int = G_BASE) -> bool:
    """A prompt is saturated iff all observed rollouts agree (k=0 or k=g)."""
    return k == 0 or k == g


def optimal_gstar(k: int, g_base: int = G_BASE) -> int:
    """Find the smallest G' in G_CANDIDATES that breaks the degeneracy for k successes
    out of g_base rollouts under the i.i.d. binomial model.

    - For saturated prompts (k=0 or k=g_base): no G helps. Return g_base.
    - For mixed prompts (0 < k < g_base): search DOWN from g_base (with 2 and 4
      added as candidates) for the smallest G' with ZVF_iid < TARGET_ZVF.

    Both saturated and mixed-prompt case return G_BASE or less, never escalate.
    This implements the *hindsight-optimal* policy economy; it is a strict
    lower bound on rollout spend that still preserves per-prompt contrast.
    """
    p_hat = k / g_base
    if p_hat < 1e-12 or p_hat > 1.0 - 1e-12:  # saturated
        return g_base
    # Search small-to-large for the MINIMUM G' that breaks degeneracy.
    # Add 2 and 4 to the candidate set explicitly (they are not in G_CANDIDATES
    # because the controller scope is "rollout spend per prompt"; G'=1 is the
    # trivial single-sample case and excluded).
    candidates = (2, 4) + tuple(g for g in G_CANDIDATES if g <= g_base)
    for g in candidates:
        if g < 2:
            continue
        if zvf_iid(p_hat, g) < TARGET_ZVF:
            return g
    # not found in the small-to-g_base set; fall back to g_base
    return g_base


def load_rewards():
    """Return dict[(method, step)] -> list[list[float]] (16 x G_BASE rewards)."""
    out = {}
    for m in METHODS:
        path = N2 / f"{m}_s0_tensors.jsonl"
        with path.open() as f:
            for line in f:
                d = json.loads(line)
                out[(m, d["step"])] = d["rewards"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="Write outputs to experiments/results/p5p8/")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    tensors = load_rewards()
    # Per-method aggregates
    method_stats = {}
    per_step_rows = []
    per_prompt_rows = []

    for m in METHODS:
        # Per (step, prompt) -> (k, g*, zvf_at_g_base, zvf_at_gstar, contrast_restored)
        per_step_stats = []
        gstar_dist = {g: 0 for g in G_CANDIDATES}
        gstar_dist[G_BASE] = 0  # include 8 in the buckets
        gstar_dist.setdefault(0, 0)
        total_rollouts_gstar = 0
        total_rollouts_g_base = 0
        total_contrast_restored = 0.0
        total_contrast_lost = 0.0
        total_prompts = 0
        # Pareto candidates (fixed G)
        fixed_g_cost = {g: 0 for g in (4, 6, 8, 12, 16, 24, 32)}
        fixed_g_saved = {g: 0 for g in (4, 6, 8, 12, 16, 24, 32)}
        for step in range(N_STEPS):
            rewards = tensors[(m, step)]
            assert len(rewards) == N_PROMPTS
            step_gstar_total = 0
            step_g_base_total = 0
            step_contrast = 0.0
            step_contrast_lost = 0.0
            step_dist = {g: 0 for g in G_CANDIDATES}
            step_dist[G_BASE] = 0
            for pi, group in enumerate(rewards):
                k = int(round(sum(group)))
                g_star = optimal_gstar(k)
                p_hat = k / G_BASE
                zvf_at_g_base = zvf_iid(p_hat, G_BASE) if 0 < k < G_BASE else 1.0
                # If k == 0 or k == G_BASE, zvf at G_BASE under iid is exactly 1
                # (or 0 if we use 0/0 -> treat as saturated)
                if k == 0 or k == G_BASE:
                    zvf_at_g_base = 1.0
                zvf_at_gstar = zvf_iid(p_hat, g_star) if 0 < k < G_BASE else (
                    zvf_iid(p_hat, g_star) if g_star != G_BASE else 1.0)
                if k == 0 or k == G_BASE:
                    # saturated -- no contrast at any G
                    zvf_at_gstar = 1.0
                contrast_restored = max(0.0, zvf_at_g_base - zvf_at_gstar)
                contrast_lost = max(0.0, zvf_at_gstar - zvf_at_g_base)
                step_gstar_total += g_star
                step_g_base_total += G_BASE
                step_contrast += (zvf_at_g_base - zvf_at_gstar)
                step_contrast_lost += contrast_lost
                step_dist[g_star] = step_dist.get(g_star, 0) + 1
                gstar_dist[g_star] = gstar_dist.get(g_star, 0) + 1
                total_rollouts_gstar += g_star
                total_rollouts_g_base += G_BASE
                total_contrast_restored += contrast_restored
                total_contrast_lost += contrast_lost
                total_prompts += 1
                per_prompt_rows.append({
                    "method": m,
                    "step": step,
                    "prompt_index": pi,
                    "k_observed": k,
                    "p_hat": round(p_hat, 4),
                    "is_saturated": bool(is_saturated(k)),
                    "g_star": g_star,
                    "zvf_at_g_base": round(zvf_at_g_base, 6),
                    "zvf_at_g_star": round(zvf_at_gstar, 6),
                    "contrast_restored": round(contrast_restored, 6),
                    "contrast_lost_to_gstar": round(contrast_lost, 6),
                })
            per_step_stats.append({
                "step": step,
                "rollouts_per_prompt_gstar": step_gstar_total,
                "rollouts_per_prompt_g_base": step_g_base_total,
                "mean_contrast_restored": step_contrast / N_PROMPTS,
                "mean_contrast_lost_to_gstar": step_contrast_lost / N_PROMPTS,
                "gstar_distribution": dict(step_dist),
                "n_saturated": step_dist.get(G_BASE, 0),
            })
            per_step_rows.append({
                "method": m,
                "step": step,
                "n_prompts": N_PROMPTS,
                "rollouts_optimal_g": step_gstar_total,
                "rollouts_fixed_g8": step_g_base_total,
                "cost_ratio": round(step_gstar_total / step_g_base_total, 4),
                "mean_contrast_restored": round(step_contrast / N_PROMPTS, 6),
                "mean_contrast_lost_to_gstar": round(step_contrast_lost / N_PROMPTS, 6),
                "n_at_g8": step_dist.get(G_BASE, 0),
                "n_at_g10": step_dist.get(10, 0),
                "n_at_g12": step_dist.get(12, 0),
                "n_at_g16": step_dist.get(16, 0),
                "n_at_g24": step_dist.get(24, 0),
            })
            # Pareto fixed-G: count how many prompts would be non-degenerate at fixed G
            for g in (4, 6, 8, 12, 16, 24, 32):
                # At fixed G, a prompt-step is "saved" if it would be non-degenerate.
                # In practice we compute across all prompts in the step.
                saved = 0
                for pi, group in enumerate(rewards):
                    k = int(round(sum(group)))
                    p_hat = k / G_BASE
                    if k == 0 or k == G_BASE:
                        # at p_hat = 0 or 1, fixed g of any value leaves ZVF_iid = 1
                        continue
                    # The prompt at fixed g: would ZVF_iid(p_hat, g) < 0.99?
                    if zvf_iid(p_hat, g) < TARGET_ZVF:
                        saved += 1
                fixed_g_cost[g] += N_PROMPTS * g
                fixed_g_saved[g] += saved

        method_stats[m] = {
            "total_rollouts_optimal_g": total_rollouts_gstar,
            "total_rollouts_fixed_g8": total_rollouts_g_base,
            "total_contrast_restored": total_contrast_restored,
            "total_contrast_lost": total_contrast_lost,
            "total_prompts": total_prompts,
            "mean_gstar": total_rollouts_gstar / total_prompts,
            "mean_contrast_restored": total_contrast_restored / total_prompts,
            "mean_contrast_lost": total_contrast_lost / total_prompts,
            "cost_ratio_optimal_vs_g8": total_rollouts_gstar / total_rollouts_g_base,
            "gstar_distribution": dict(gstar_dist),
            "fixed_g_pareto": {
                g: {
                    "cost_ratio": fixed_g_cost[g] / total_rollouts_g_base,
                    "n_prompts_saved": fixed_g_saved[g],
                    "save_rate": fixed_g_saved[g] / total_prompts,
                    "rollouts_total": fixed_g_cost[g],
                }
                for g in (4, 6, 8, 12, 16, 24, 32)
            },
            "per_step": per_step_stats,
        }

    # ----- Aggregate headline ----------------------------------------------
    # Aggregate across methods (equal-weight per method = 40 steps x 16 prompts x 4 methods)
    agg = {}
    agg["n_prompt_steps_total"] = sum(s["total_prompts"] for s in method_stats.values())
    agg["rollouts_optimal_total"] = sum(s["total_rollouts_optimal_g"] for s in method_stats.values())
    agg["rollouts_fixed_g8_total"] = sum(s["total_rollouts_fixed_g8"] for s in method_stats.values())
    agg["mean_gstar"] = agg["rollouts_optimal_total"] / agg["n_prompt_steps_total"]
    agg["mean_cost_ratio"] = agg["rollouts_optimal_total"] / agg["rollouts_fixed_g8_total"]
    agg["mean_contrast_restored"] = sum(
        s["total_contrast_restored"] for s in method_stats.values()
    ) / agg["n_prompt_steps_total"]
    agg["mean_contrast_lost"] = sum(
        s["total_contrast_lost"] for s in method_stats.values()
    ) / agg["n_prompt_steps_total"]
    # Fraction of prompts that are saturated (k=0 or k=8)
    agg["n_saturated_total"] = sum(
        s["gstar_distribution"].get(G_BASE, 0) for s in method_stats.values()
    )
    agg["frac_saturated"] = agg["n_saturated_total"] / agg["n_prompt_steps_total"]
    # Fraction of prompts for which G* < G_BASE (true economy)
    n_economized = sum(
        s["gstar_distribution"].get(2, 0) + s["gstar_distribution"].get(4, 0)
        for s in method_stats.values()
    )
    agg["n_economized_total"] = n_economized
    agg["frac_economized"] = n_economized / agg["n_prompt_steps_total"]
    # Distribution of gstar (pooled across methods)
    pooled_dist = {}
    for s in method_stats.values():
        for g, c in s["gstar_distribution"].items():
            pooled_dist[g] = pooled_dist.get(g, 0) + c
    agg["pooled_gstar_distribution"] = dict(sorted(pooled_dist.items()))

    # Pareto frontier: for each candidate fixed G ∈ {4,8,12,16,24,32}
    # we compute total rollouts, total prompts saved (vs G=8), cost ratio, save rate
    pareto = []
    for g in (4, 6, 8, 12, 16, 24, 32):
        total_cost = sum(s["fixed_g_pareto"][g]["rollouts_total"] for s in method_stats.values())
        total_saved = sum(s["fixed_g_pareto"][g]["n_prompts_saved"] for s in method_stats.values())
        total_sat_count = sum(s["gstar_distribution"].get(G_BASE, 0) for s in method_stats.values())
        pareto.append({
            "candidate_G": g,
            "rollouts_total": total_cost,
            "cost_ratio_vs_g8": round(total_cost / agg["rollouts_fixed_g8_total"], 4),
            "prompts_saved": total_saved,
            "save_rate": round(total_saved / agg["n_prompt_steps_total"], 4),
        })
    pareto.append({
        "candidate_G": "optimal_gstar",
        "rollouts_total": agg["rollouts_optimal_total"],
        "cost_ratio_vs_g8": round(agg["mean_cost_ratio"], 4),
        "prompts_saved": int(sum(
            1 for r in per_prompt_rows
            if r["contrast_restored"] > 1e-9 and r["g_star"] != G_BASE
        )),
        "save_rate": round(
            sum(
                1 for r in per_prompt_rows
                if r["contrast_restored"] > 1e-9 and r["g_star"] != G_BASE
            ) / len(per_prompt_rows), 4
        ),
    })
    agg["pareto_frontier"] = pareto

    # ----- Write outputs ----------------------------------------------------
    if args.write:
        # Summary TSV: one row per (method, fixed G or optimal)
        summary_rows = []
        for m in METHODS:
            stats = method_stats[m]
            for g in (4, 6, 8, 12, 16, 24, 32):
                fg = stats["fixed_g_pareto"][g]
                summary_rows.append({
                    "method": m,
                    "controller": f"fixed_g{g}",
                    "g_used": g,
                    "rollouts_total": fg["rollouts_total"],
                    "cost_ratio_vs_g8": round(fg["rollouts_total"] / stats["total_rollouts_fixed_g8"], 4),
                    "prompts_saved_vs_g8_observed": fg["n_prompts_saved"],
                    "save_rate": round(fg["save_rate"], 4),
                    "mean_contrast_restored": "",
                })
            summary_rows.append({
                "method": m,
                "controller": "per_prompt_optimal_gstar",
                "g_used": "varying",
                "rollouts_total": stats["total_rollouts_optimal_g"],
                "cost_ratio_vs_g8": round(stats["cost_ratio_optimal_vs_g8"], 4),
                "prompts_saved_vs_g8_observed": sum(
                    1 for r in per_prompt_rows if r["method"] == m
                    and r["contrast_restored"] > 1e-9
                    and r["g_star"] != G_BASE
                ),
                "save_rate": round(
                    sum(
                        1 for r in per_prompt_rows if r["method"] == m
                        and r["contrast_restored"] > 1e-9 and r["g_star"] != G_BASE
                    ) / stats["total_prompts"], 4
                ),
                "mean_contrast_restored": round(stats["mean_contrast_restored"], 6),
            })
        cols = ["method", "controller", "g_used", "rollouts_total",
                "cost_ratio_vs_g8", "prompts_saved_vs_g8_observed",
                "save_rate", "mean_contrast_restored"]
        with (OUT / "p7_per_prompt_optimal_g_summary.tsv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)

        cols_ps = ["method", "step", "n_prompts", "rollouts_optimal_g",
                   "rollouts_fixed_g8", "cost_ratio", "mean_contrast_restored",
                   "mean_contrast_lost_to_gstar",
                   "n_at_g8", "n_at_g10", "n_at_g12", "n_at_g16", "n_at_g24"]
        with (OUT / "p7_per_prompt_optimal_g_per_step.tsv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=cols_ps, delimiter="\t")
            w.writeheader()
            for r in per_step_rows:
                w.writerow(r)

        cols_pp = ["method", "step", "prompt_index", "k_observed", "p_hat",
                   "is_saturated", "g_star", "zvf_at_g_base", "zvf_at_g_star",
                   "contrast_restored", "contrast_lost_to_gstar"]
        with (OUT / "p7_per_prompt_optimal_g_per_prompt.tsv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=cols_pp, delimiter="\t")
            w.writeheader()
            for r in per_prompt_rows:
                w.writerow(r)

        out = {
            "headlines": {
                "n_methods": len(METHODS),
                "n_steps": N_STEPS,
                "n_prompts_per_step": N_PROMPTS,
                "g_base": G_BASE,
                "target_zvf": TARGET_ZVF,
                "n_prompt_steps_total": agg["n_prompt_steps_total"],
                "mean_gstar": round(agg["mean_gstar"], 4),
                "mean_cost_ratio_vs_g8": round(agg["mean_cost_ratio"], 4),
                "mean_contrast_restored_per_prompt": round(agg["mean_contrast_restored"], 6),
                "mean_contrast_lost_to_gstar": round(agg["mean_contrast_lost"], 6),
                "n_saturated_total": agg["n_saturated_total"],
                "frac_saturated": round(agg["frac_saturated"], 4),
                "n_economized_total": agg["n_economized_total"],
                "frac_economized": round(agg["frac_economized"], 4),
                "pooled_gstar_distribution": agg["pooled_gstar_distribution"],
            },
            "method_stats": {
                m: {k: v for k, v in stats.items() if k != "per_step"}
                for m, stats in method_stats.items()
            },
            "pareto_frontier": agg["pareto_frontier"],
            "interpretation": (
                "Per-prompt hindsight-optimal G*: for each (method, step, prompt) "
                "we observe k/8 successes at G_BASE = 8 and find the smallest G' "
                "in {4,6,8,10,12,16,20,24,32,48,64} such that ZVF_iid(p_hat=k/8, "
                "G') < 0.99. Saturated prompts (k=0 or k=8) cannot benefit from "
                "any G (ZVF_iid = 1); the controller's only honest move is to "
                "keep G_BASE. The pooled mean G* across all 4 methods x 40 steps "
                "x 16 prompts is reported in headlines.mean_gstar. The Pareto "
                "frontier compares fixed-G candidates to per-prompt G* on the "
                "same data: lower rollouts_at_fixed_G and prompts_saved (the "
                "number of prompt-steps that would be non-degenerate at fixed G)."
            ),
        }
        (OUT / "p7_per_prompt_optimal_g_summary.json").write_text(
            json.dumps(out, indent=2, default=lambda o: int(o) if isinstance(o, bool) else str(o))
        )

        # Figure: distribution of gstar per method
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            width = 0.18
            all_gs = sorted({g for s in method_stats.values() for g in s["gstar_distribution"].keys()})
            x = list(range(len(all_gs)))
            for i, m in enumerate(METHODS):
                counts = [method_stats[m]["gstar_distribution"].get(g, 0) for g in all_gs]
                ax.bar([xi + i * width for xi in x], counts, width, label=m)
            ax.set_xticks([xi + 1.5 * width for xi in x])
            ax.set_xticklabels(all_gs)
            ax.set_xlabel("per-prompt optimal G* (rollouts)")
            ax.set_ylabel("count of prompt-steps")
            ax.set_title(
                f"Per-prompt hindsight-optimal G* on N2 reward tensors\n"
                f"(4 methods x {N_STEPS} steps x {N_PROMPTS} prompts = "
                f"{agg['n_prompt_steps_total']} prompt-steps)"
            )
            ax.legend(loc="best", fontsize=9)
            plt.tight_layout()
            plt.savefig(FIG / "p7_per_prompt_g_distribution.png", dpi=120)
            plt.savefig(FIG / "p7_per_prompt_g_distribution.pdf")
            plt.close()
        except Exception as e:
            print(f"[warn] figure skipped: {e}")

    # ----- Console headline ------------------------------------------------
    print(f"=== P7 per-prompt hindsight-optimal G* on N2 four-method tensors ===")
    print(f"n_methods = {len(METHODS)}, n_steps = {N_STEPS}, "
          f"n_prompts/step = {N_PROMPTS}, total prompt-steps = "
          f"{agg['n_prompt_steps_total']}")
    print(f"pooled mean G* = {agg['mean_gstar']:.3f} (cost ratio vs G=8 = "
          f"{agg['mean_cost_ratio']:.3f})")
    print(f"pooled mean contrast restored per prompt = "
          f"{agg['mean_contrast_restored']:.4f} (ZVF units)")
    print(f"pooled mean contrast lost to G* per prompt = "
          f"{agg['mean_contrast_lost']:.4f} (ZVF units)")
    print(f"frac_saturated = {agg['frac_saturated']:.3f}, "
          f"frac_economized (G*=2) = {agg['frac_economized']:.3f}")
    print(f"\nG* distribution (pooled across methods):")
    for g, c in sorted(agg["pooled_gstar_distribution"].items()):
        print(f"  G*={g:>3d}: {c:5d} prompt-steps "
              f"({100*c/agg['n_prompt_steps_total']:5.1f}%)")
    print(f"\nPareto frontier (fixed G vs per-prompt optimal):")
    for p in pareto:
        cg = str(p['candidate_G'])
        print(f"  G={cg:>14s}  rollouts={p['rollouts_total']:6d}  "
              f"cost_ratio={p['cost_ratio_vs_g8']:.3f}  "
              f"saved={p['prompts_saved']:4d}  "
              f"save_rate={p['save_rate']:.3f}")
    print(f"\nPer-method mean G*:")
    for m, s in method_stats.items():
        print(f"  {m:6s}: mean G* = {s['mean_gstar']:.3f},  "
              f"cost_ratio = {s['cost_ratio_optimal_vs_g8']:.3f},  "
              f"saturated = {s['gstar_distribution'].get(G_BASE, 0)} "
              f"of {s['total_prompts']}")
    if args.write:
        print(f"\nwrote {OUT}/p7_per_prompt_optimal_g_*.{{tsv,json}}")
        print(f"wrote {FIG}/p7_per_prompt_g_distribution.{{png,pdf}}")


if __name__ == "__main__":
    main()
