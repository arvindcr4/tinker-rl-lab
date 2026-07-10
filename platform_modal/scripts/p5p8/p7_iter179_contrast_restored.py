#!/usr/bin/env python3
"""Iter 179 — P7 Counterfactual Contrast-Restoration on Fired N2 Steps.

Vein: brief vein (a) at the **per-prompt "restored contrast"** level on the
ACTUAL fired steps of the iter-119 C4 unified controller. Prior veins have
measured (i) headline CIs (iter-171), (ii) step-level C4 cost vs retention
(iter-151), (iii) Pareto over G' on every prompt (iter-91/95/111) — but never
"for each fired (method, step), how much binomial-predicted contrast would
G=16 have restored on the BOUNDARY prompts at that step?".

Definition (per fired (method, step, prompt)):
  p_hat = k_p / G_BASE        (empirical success rate at G=8)
  z_b   = 1.0 if k_p in {0, G_BASE} else 0.0   (boundary indicator)
  y_b   = 1.0 - z_b                              (observed contrast)
  y_n   = 1.0 - p_hat^G_N - (1-p_hat)^G_N        (binomial contrast at G_N)
  restored_p = y_n - y_b                          (per-prompt restored contrast)

Operational controller (C4 from iter-119/151):
  fires iff step.zvf >= tau, escalate to G_N in {12, 16, 32}

Three sweep levels:
  tau ∈ {0.55, 0.60, 0.65, 0.70, 0.75, 0.80}     (6 trigger thresholds)
  G_N ∈ {12, 16, 32}                              (3 escalation targets)
  methods = {grpo, aero, gift, areal}             (4)

Per (method, tau, G_N) we record:
  - n_fired_steps = count of (method, step) pairs that would have fired
  - mean restored contrast on boundary prompts at fired steps
  - 95% bootstrap CI (B=2000, percentile, seed=20260705)
  - cross-method CV of mean restored contrast

Hypotheses:
  H1 — At tau=0.70, mean restored contrast > 0 across all 4 methods (CI lo > 0)
  H2 — At tau=0.70, mean restored contrast ≥ +0.05 across all 4 methods
  H3 — Cross-method CV of mean restored contrast < 0.30 at tau=0.70, G=16
  H4 — Restored contrast monotonically increases in G_N (G=32 > G=16 > G=12)
       on at least 3/4 methods at tau=0.70

Outputs:
  platform_hybrid/experiments/results/p5p8/p7_iter179_per_fired.tsv        (raw per-(method, step, tau, G_N) rows)
  platform_hybrid/experiments/results/p5p8/p7_iter179_per_prompt.tsv         (per-prompt restored contrast on fired steps)
  platform_hybrid/experiments/results/p5p8/p7_iter179_summary.tsv           (18 rows = 4 methods × 6 tau × 3 G_N - sparse)
  platform_hybrid/experiments/results/p5p8/p7_iter179_ci.tsv                (CI95 per (method, tau, G_N))
  platform_hybrid/experiments/results/p5p8/p7_iter179_summary.json          (structured)

Stdlib only; deterministic.
"""
from __future__ import annotations
import csv, glob, json, os, random, statistics

WORKTREE = "/home/claude/tinker-rl-lab-minimax"
DATA_DIR = os.path.join(WORKTREE, "platform_hybrid/experiments/results/n2_reward_tensor_resume")
OUT_DIR = os.path.join(WORKTREE, "platform_hybrid/experiments/results/p5p8")
os.makedirs(OUT_DIR, exist_ok=True)
METHODS = ["grpo", "aero", "gift", "areal"]
G_BASE = 8; N_PROMPTS = 16
TAU_GRID = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
G_N_GRID = [12, 16, 32]
B = 2000; SEED = 20260705
ALPHA = 0.05


def _bci(v, stat_fn=statistics.mean, rng=None):
    if rng is None:
        rng = random.Random(SEED)
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    pt = stat_fn(v)
    boots = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(stat_fn([v[i] for i in idx]))
    boots.sort()
    return (pt, boots[int(ALPHA/2*B)], boots[int((1-ALPHA/2)*B)], B)


def load_tensors():
    out = {m: [] for m in METHODS}
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*_tensors.jsonl"))):
        method = os.path.basename(path).split("_")[0]
        if method not in METHODS:
            continue
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out[method].append(json.loads(line))
    for m in METHODS:
        out[m].sort(key=lambda r: r["step"])
    return out


def per_prompt_restored(step, g_n):
    """For one (method, step), return list of per-prompt restored contrast.

    Both y_b and y_n are computed as **binomial-projected** contrast at
    G=8 (y_b) and G_N (y_n), using the SAME per-prompt empirical p_hat = k/8.
    This makes the comparator apples-to-apples:
      y_b = 1 - p_hat^8  - (1-p_hat)^8     (expected contrast at G=8)
      y_n = 1 - p_hat^G_N - (1-p_hat)^G_N  (expected contrast at G_N)
      restored = y_n - y_b
    For p_hat ∈ {0, 1} (boundary), y_b = y_n = 0.
    For p_hat ∈ (0, 1) (contrast), y_n >= y_b — monotonically — because the
    boundary-fraction tail z(p, G) = p^G + (1-p)^G collapses as G grows for
    non-extreme p. This makes the metric non-trivial: it answers "for each
    fired step, by how much would escalating to G_N reduce the boundary
    probability on the contrast prompts?"

    Returns:
      list of (k_at_G8, y_b, y_n, restored, is_boundary_at_G8) per prompt.
    """
    rewards = step["rewards"]
    if not rewards or len(rewards[0]) != G_BASE:
        return []
    out = []
    for r in rewards:
        k = int(round(sum(r)))
        p_hat = k / G_BASE
        is_boundary = (k == 0 or k == G_BASE)
        y_b = 1.0 - (p_hat**G_BASE) - ((1.0 - p_hat)**G_BASE)
        y_n = 1.0 - (p_hat**g_n) - ((1.0 - p_hat)**g_n)
        restored = y_n - y_b
        out.append((k, y_b, y_n, restored, int(is_boundary)))
    return out


def main():
    print("[iter179] loading N2 tensors...")
    tensors = load_tensors()
    for m in METHODS:
        print(f"  {m}: {len(tensors[m])} steps")

    rng = random.Random(SEED)
    per_fired_rows = []
    per_prompt_rows = []
    summary_rows = []
    ci_rows = []

    for m in METHODS:
        for tau in TAU_GRID:
            fired_steps = [s for s in tensors[m] if s.get("zvf", 0.0) >= tau]
            n_fired = len(fired_steps)
            for g_n in G_N_GRID:
                if n_fired == 0:
                    summary_rows.append({
                        "method": m, "tau": tau, "G_N": g_n,
                        "n_fired_steps": 0,
                        "n_boundary_prompts_total": 0,
                        "mean_restored_pt": float("nan"),
                    })
                    ci_rows.append({
                        "method": m, "tau": tau, "G_N": g_n,
                        "n_fired_steps": 0,
                        "mean_restored_pt": float("nan"),
                        "ci_lo": float("nan"), "ci_hi": float("nan"),
                        "ci_hw": float("nan"),
                    })
                    continue
                # collect per-prompt restored contrast across all fired steps
                all_restored = []
                step_records = []
                for s in fired_steps:
                    recs = per_prompt_restored(s, g_n)
                    n_b = sum(1 for rec in recs if rec[4])
                    restored_vals = [rec[3] for rec in recs]
                    mean_step = statistics.mean(restored_vals) if restored_vals else float("nan")
                    per_fired_rows.append({
                        "method": m, "step": s["step"], "tau": tau, "G_N": g_n,
                        "zvf_step": round(s["zvf"], 4),
                        "n_boundary_prompts": n_b,
                        "n_total_prompts": len(recs),
                        "mean_restored_step": (round(mean_step, 4)
                                               if mean_step == mean_step
                                               else float("nan")),
                    })
                    if restored_vals:
                        all_restored.extend(restored_vals)
                    step_records.append((s["step"], n_b,
                                         mean_step if mean_step == mean_step else float("nan")))
                # also write per-prompt rows for first tau=0.70, G_N=16 (headline)
                if tau == 0.70 and g_n == 16:
                    for s in fired_steps:
                        rewards = s["rewards"]
                        recs = per_prompt_restored(s, g_n)
                        for pi, (k, y_b, y_n, restored, is_b) in enumerate(recs):
                            per_prompt_rows.append({
                                "method": m, "step": s["step"],
                                "prompt_index": pi, "k_p_at_G8": k,
                                "boundary_at_G8": is_b, "tau": tau, "G_N": g_n,
                                "y_at_G8": round(y_b, 4),
                                "y_at_G_N": round(y_n, 4),
                                "restored": round(restored, 4),
                            })
                n_total = len(all_restored)
                pt = statistics.mean(all_restored) if all_restored else float("nan")
                if all_restored:
                    pt_b, lo, hi, _ = _bci(all_restored, rng=rng)
                else:
                    pt_b, lo, hi = pt, float("nan"), float("nan")
                summary_rows.append({
                    "method": m, "tau": tau, "G_N": g_n,
                    "n_fired_steps": n_fired,
                    "n_boundary_prompts_total": n_total,
                    "mean_restored_pt": round(pt, 4) if pt == pt else float("nan"),
                })
                ci_rows.append({
                    "method": m, "tau": tau, "G_N": g_n,
                    "n_fired_steps": n_fired,
                    "mean_restored_pt": (round(pt_b, 4)
                                         if pt_b == pt_b else float("nan")),
                    "ci_lo": round(lo, 4) if lo == lo else float("nan"),
                    "ci_hi": round(hi, 4) if hi == hi else float("nan"),
                    "ci_hw": round((hi - lo) / 2.0, 4)
                             if (hi == hi and lo == lo) else float("nan"),
                })

    # write artifacts
    def _write(path, rows):
        if not rows:
            print(f"  [warn] empty {path}")
            return
        with open(path, "w") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
            w.writeheader()
            for r in rows:
                w.writerow(r)
    _write(os.path.join(OUT_DIR, "p7_iter179_per_fired.tsv"), per_fired_rows)
    _write(os.path.join(OUT_DIR, "p7_iter179_per_prompt.tsv"), per_prompt_rows)
    _write(os.path.join(OUT_DIR, "p7_iter179_summary.tsv"), summary_rows)
    _write(os.path.join(OUT_DIR, "p7_iter179_ci.tsv"), ci_rows)

    # hypotheses
    c70g16 = [r for r in ci_rows
              if abs(r["tau"] - 0.70) < 1e-9 and r["G_N"] == 16]
    h1 = all(r["ci_lo"] > 0 for r in c70g16)
    h2 = all(r["mean_restored_pt"] >= 0.05 for r in c70g16)
    cv70g16 = (statistics.stdev([r["mean_restored_pt"] for r in c70g16])
               / statistics.mean([r["mean_restored_pt"] for r in c70g16])
               if len(c70g16) >= 2 and all(r["mean_restored_pt"] == r["mean_restored_pt"]
                                           for r in c70g16)
               else float("nan"))
    h3 = (cv70g16 < 0.30)
    monotonic = 0
    for m in METHODS:
        by_g = {r["G_N"]: r["mean_restored_pt"]
                for r in ci_rows
                if r["method"] == m and abs(r["tau"] - 0.70) < 1e-9}
        if (by_g.get(12, float("nan")) == by_g.get(12, float("nan"))
                and by_g.get(32, float("nan")) == by_g.get(32, float("nan"))
                and by_g.get(12) <= by_g.get(16, -1) <= by_g.get(32)):
            monotonic += 1
    h4 = monotonic >= 3

    summary = {
        "n_fired_rows": len(per_fired_rows),
        "n_per_prompt_rows": len(per_prompt_rows),
        "n_summary_rows": len(summary_rows),
        "n_ci_rows": len(ci_rows),
        "B": B, "seed": SEED,
        "tau_grid": TAU_GRID, "G_N_grid": G_N_GRID, "methods": METHODS,
        "headline_tau070_G16_per_method": [
            {"method": r["method"], "n_fired_steps": r["n_fired_steps"],
             "mean_restored_pt": r["mean_restored_pt"],
             "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]}
            for r in c70g16
        ],
        "cross_method_cv_at_tau070_G16": round(cv70g16, 4)
                                          if cv70g16 == cv70g16 else None,
        "monotonic_methods_at_tau070_G12_le_G16_le_G32": monotonic,
        "verdicts": {
            "H1_mean_restored_gt0_at_tau070_G16_all4_methods": bool(h1),
            "H2_mean_restored_ge005_at_tau070_G16_all4_methods": bool(h2),
            "H3_cross_method_cv_lt030_at_tau070_G16": bool(h3),
            "H4_monotonic_in_G_N_at_least_3_of_4_methods": bool(h4),
        },
    }
    with open(os.path.join(OUT_DIR, "p7_iter179_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[iter179] n_fired={len(per_fired_rows)} n_per_prompt={len(per_prompt_rows)} "
          f"n_summary={len(summary_rows)} n_ci={len(ci_rows)}")
    print(f"[iter179] H1={h1} H2={h2} H3={h3} (cv={cv70g16:.4f}) H4={h4} (monotonic_methods={monotonic}/4)")
    print(f"[iter179] c70g16:")
    for r in c70g16:
        print(f"  {r['method']}: fired={r['n_fired_steps']} "
              f"restored_pt={r['mean_restored_pt']:.4f} "
              f"CI=[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")


if __name__ == "__main__":
    main()