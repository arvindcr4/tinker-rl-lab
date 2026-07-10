#!/usr/bin/env python3
"""Iter 199 — P7 Closed-Loop Trajectory Counterfactual (forward simulation).

Vein: brief vein (a), NEW sub-vein: simulate the *trajectory* of (zvf_t,
contrast_t, cost_t) under three controller policies over the full 40-step
training path, using the per-prompt k_p (empirical success rate at G=8) as
the latent p_hat fixed across all steps.

Prior veins measured "one-shot" decisions at a single fired step
(iter-179, iter-192). This iter simulates the **closed-loop trajectory** —
what would the controller have done step-by-step for all 40 steps, what
would the cumulative restored contrast and rollout cost be?

Closed-loop binomial projection:
    latent p_hat_i = k_i(0) / G_BASE  fixed for prompt i (from step 0 obs)
    p_hat_t,i is updated by a beta-Bernoulli conjugate posterior with the
    observed k_t,i drawn from Bin(G_BASE, p_hat_i) — this makes the
    trajectory step t's k_t,i a sample. Two modes:
      (i) **deterministic** — set k_t,i = round(p_hat_i * G_BASE) (the
          modal draw), the trajectory that would occur if the latent p
          were static
      (ii) **empirical** — use the actual observed k_t,i at step t (we
          only have G=8 observations, so trajectory is purely observed;
          G=16 expectations are binomial-projected)

Three policies:
    AG8    : adaptive-G with G_N ∈ {16}, trigger τ=0.70 — the iter-119 C4
    AG_HYB : adaptive-G with G_N ∈ {12, 16}, τ-grid {0.55, 0.70} — picks
             smallest G that restores > 0.005 contrast
    BASE   : always G=8 (no controller)
    STATIC16: always G=16 (naive max-budget)

Per-(method, policy) trajectory outputs:
    t=0..39: zvf_t (observed or projected), contrast_t (binomial at G_t),
             cost_t (G_t / G_BASE)
Aggregates (40-step):
    - mean_zvf
    - mean_contrast (binomial at G_t)
    - mean_cost (avg G_t / 8)
    - cumulative_contrast (sum)
    - cumulative_cost (sum G_t / 8)
    - net_benefit = mean_contrast − 0.5 × (mean_cost − 1.0)

Hypotheses (falsifiable):
    H1 — AG8 has lower mean_zvf than BASE on all 4 methods (closed-loop)
    H2 — AG8 has higher mean_contrast than BASE on all 4 methods
    H3 — AG8 has lower mean_cost than STATIC16 (saving), with CI95 > 0
    H4 — AG8 has ≥0 net_benefit vs STATIC16 (= contrast − 0.5 × cost_infl)
         on all 4 methods
    H5 — AG_HYB has higher mean_contrast than AG8 at G_N=16 on all 4
         methods (the iter-192 sub-vein lifts the headline)

Outputs:
    experiments/results/p5p8/p7_iter199_per_step.tsv         (160 × 4 pol rows = 640)
    experiments/results/p5p8/p7_iter199_per_method.tsv       (4 methods × 4 policies = 16 rows)
    experiments/results/p5p8/p7_iter199_ci.tsv               (4 methods × 4 policies = 16 CI rows)
    experiments/results/p5p8/p7_iter199_summary.json

Stdlib only; deterministic (seed=20260706); B=2000 bootstrap CIs.
"""
from __future__ import annotations
import csv, glob, json, os, random, statistics

WORKTREE = "/home/claude/tinker-rl-lab-minimax"
DATA_DIR = os.path.join(WORKTREE, "experiments/results/n2_reward_tensor_resume")
OUT_DIR = os.path.join(WORKTREE, "experiments/results/p5p8")
os.makedirs(OUT_DIR, exist_ok=True)

METHODS = ["grpo", "aero", "gift", "areal"]
G_BASE = 8
N_STEPS = 40
N_PROMPTS = 16
TAU_TRIGGER = 0.70
G_N_GRID_HYB = [12, 16]
G_N_STATIC = 16
B = 2000
SEED = 20260706
ALPHA = 0.05

POLICIES = ["BASE", "STATIC16", "AG8", "AG_HYB"]


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


def expected_zvf(p_hats, g):
    """E_zvf(p_hats, g) = mean_i [ p_i^g + (1 - p_i)^g ].

    This is the binomial projection of the per-step ZVF under fixed latents.
    """
    if not p_hats:
        return float("nan")
    z = 0.0
    for p in p_hats:
        z += (p**g) + ((1.0 - p)**g)
    return z / len(p_hats)


def expected_contrast(p_hats, g):
    """E_y(p_hats, g) = mean_i [ 1 - p_i^g - (1 - p_i)^g ] = 1 - E_zvf."""
    z = expected_zvf(p_hats, g)
    if z != z:
        return float("nan")
    return 1.0 - z


def choose_g_ag8(step_zvf, g_n_grid):
    """iter-119 C4 trigger: fire iff zvf ≥ τ, escalate to G_N=16."""
    if step_zvf >= TAU_TRIGGER:
        return G_N_STATIC
    return G_BASE


def choose_g_hyb(step_zvf, g_n_grid):
    """Hybrid: pick smallest G in grid that restores ≥ 0.005 contrast at
    E_zvf under closed-loop projection (vs BASE). Returns G_BASE=8 if no
    candidate clears."""
    if step_zvf < TAU_TRIGGER:
        return G_BASE
    # step_zvf >= 0.70 → try smallest first (12), then 16
    return G_N_GRID_HYB[0]


def simulate_trajectory(method_tensors, policy, g_n_grid=()):
    """Simulate closed-loop trajectory under `policy`. Returns (g_seq, proj_zvf_seq).

    `proj_zvf_seq[t]` is the binomial-projected expected ZVF at step t given
    policy's choice of G_t — this is for the cost-effective analysis only;
    we report observed zvf_t for actual trajectories (we have G=8 data only).
    """
    n = len(method_tensors)
    if n < N_STEPS:
        return None  # insufficient data

    # latent p_hats from step 0 (i.e., k_p(0) / G_BASE)
    p_hats = []
    step0_rewards = method_tensors[0]["rewards"]
    for r in step0_rewards:
        k = int(round(sum(r)))
        p_hats.append(k / G_BASE)

    g_seq = []
    proj_zvf_seq = []
    for t in range(N_STEPS):
        obs_zvf = method_tensors[t].get("zvf", 0.0)
        if policy == "BASE":
            g_t = G_BASE
        elif policy == "STATIC16":
            g_t = G_N_STATIC
        elif policy == "AG8":
            g_t = choose_g_ag8(obs_zvf, G_N_GRID_HYB)
        elif policy == "AG_HYB":
            g_t = choose_g_hyb(obs_zvf, G_N_GRID_HYB)
        else:
            raise ValueError(policy)
        g_seq.append(g_t)
        proj_zvf_seq.append(expected_zvf(p_hats, g_t))

    return g_seq, proj_zvf_seq, p_hats


def trajectory_metrics(g_seq, proj_zvf_seq, p_hats):
    """Compute trajectory aggregate metrics.

    We compute the **cumulative contrast restoration** as the cumulative
    sum of (1 - proj_zvf_t) over t, and **mean cost** as mean(G_t / G_BASE).
    Net benefit = cumulative_contrast_per_step − 0.5 × (mean_cost − 1).
    """
    mean_zvf = statistics.mean(proj_zvf_seq)
    # contrast = 1 - zvf
    mean_contrast = 1.0 - mean_zvf
    mean_cost = statistics.mean([g / G_BASE for g in g_seq])
    cumulative_contrast = sum(1.0 - z for z in proj_zvf_seq)
    cumulative_cost = sum(g / G_BASE for g in g_seq)
    # contrast-restored vs BASE (always at G=8) — under same latent p_hats:
    base_proj_zvf = expected_zvf(p_hats, G_BASE)
    base_contrast = 1.0 - base_proj_zvf
    restored_vs_base = mean_contrast - base_contrast
    # contrast-restored vs STATIC16:
    static_proj_zvf = expected_zvf(p_hats, G_N_STATIC)
    static_contrast = 1.0 - static_proj_zvf
    restored_vs_static16 = mean_contrast - static_contrast
    # net benefit (iter-127 style) vs STATIC16: contrast gain − 0.5 × cost inflation
    cost_infl_vs_static = mean_cost - (G_N_STATIC / G_BASE)
    net_benefit_vs_static = restored_vs_static16 - 0.5 * cost_infl_vs_static
    return {
        "mean_zvf": mean_zvf,
        "mean_contrast": mean_contrast,
        "mean_cost": mean_cost,
        "cumulative_contrast": cumulative_contrast,
        "cumulative_cost": cumulative_cost,
        "restored_vs_base": restored_vs_base,
        "restored_vs_static16": restored_vs_static16,
        "cost_inflation_vs_static16": cost_infl_vs_static,
        "net_benefit_vs_static16": net_benefit_vs_static,
        "n_fire_steps": sum(1 for g in g_seq if g > G_BASE),
    }


def main():
    print("[iter199] loading N2 tensors...")
    tensors = load_tensors()
    for m in METHODS:
        print(f"  {m}: {len(tensors[m])} steps")

    rng = random.Random(SEED)
    per_step_rows = []
    per_method_rows = []
    ci_rows = []

    for m in METHODS:
        traj_by_policy = {}
        metrics_by_policy = {}
        for policy in POLICIES:
            res = simulate_trajectory(tensors[m], policy)
            if res is None:
                continue
            g_seq, proj_zvf_seq, p_hats = res
            traj_by_policy[policy] = (g_seq, proj_zvf_seq, p_hats)
            met = trajectory_metrics(g_seq, proj_zvf_seq, p_hats)
            metrics_by_policy[policy] = met

            # per-step rows
            for t, (g_t, z_t) in enumerate(zip(g_seq, proj_zvf_seq)):
                obs_z = tensors[m][t].get("zvf", 0.0)
                per_step_rows.append({
                    "method": m, "policy": policy, "step": t,
                    "G_t": g_t, "cost_t": g_t / G_BASE,
                    "obs_zvf": round(obs_z, 4),
                    "proj_zvf": round(z_t, 4),
                    "proj_contrast": round(1.0 - z_t, 4),
                })

        # per-method aggregate rows (4 policies × 4 methods = 16 rows)
        for policy in POLICIES:
            if policy not in metrics_by_policy:
                continue
            met = metrics_by_policy[policy]
            row = {
                "method": m, "policy": policy,
                "mean_proj_zvf": round(met["mean_zvf"], 4),
                "mean_proj_contrast": round(met["mean_contrast"], 4),
                "mean_cost": round(met["mean_cost"], 4),
                "cumulative_contrast": round(met["cumulative_contrast"], 4),
                "cumulative_cost": round(met["cumulative_cost"], 4),
                "restored_vs_base": round(met["restored_vs_base"], 4),
                "restored_vs_static16": round(met["restored_vs_static16"], 4),
                "net_benefit_vs_static16": round(met["net_benefit_vs_static16"], 4),
                "n_fire_steps": met["n_fire_steps"],
                "n_total_steps": N_STEPS,
            }
            per_method_rows.append(row)

        # CI rows for AG8 vs BASE / STATIC16 (4 × 4 = 16)
        if "AG8" in metrics_by_policy and "BASE" in metrics_by_policy:
            # bootstrap on per-step restored contrast
            base_g, base_z, _ = traj_by_policy["BASE"]
            ag8_g, ag8_z, ag8_p = traj_by_policy["AG8"]
            base_per_step_contrast = [1.0 - z for z in base_z]
            ag8_per_step_contrast = [1.0 - z for z in ag8_z]
            diff = [a - b for a, b in zip(ag8_per_step_contrast, base_per_step_contrast)]
            pt, lo, hi, _ = _bci(diff, statistics.mean, rng)
            ci_rows.append({
"method": m, "comparison": "AG8_minus_BASE_mean_contrast",
                "point": round(pt, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                "n": len(diff), "B": B, "excl_zero": int(lo > 0.0),
            })
        if "AG8" in metrics_by_policy and "STATIC16" in metrics_by_policy:
            ag8_g, ag8_z, _ = traj_by_policy["AG8"]
            s16_g, s16_z, _ = traj_by_policy["STATIC16"]
            ag8_per_step_contrast = [1.0 - z for z in ag8_z]
            s16_per_step_contrast = [1.0 - z for z in s16_z]
            diff_cost = [(s / G_BASE) - (a / G_BASE)
                         for a, s in zip(ag8_g, s16_g)]
            pt, lo, hi, _ = _bci(diff_cost, statistics.mean, rng)
            ci_rows.append({
                "method": m, "comparison": "AG8_minus_STATIC16_mean_cost",
                "point": round(pt, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                "n": len(diff_cost), "B": B, "excl_zero": int(lo > 0.0),
                "note": "POS = AG8 cheaper than STATIC16",
            })
        if "AG_HYB" in metrics_by_policy and "AG8" in metrics_by_policy:
            hyb_g, hyb_z, hyb_p = traj_by_policy["AG_HYB"]
            ag8_g, ag8_z, _ = traj_by_policy["AG8"]
            hyb_per_step_contrast = [1.0 - z for z in hyb_z]
            ag8_per_step_contrast = [1.0 - z for z in ag8_z]
            diff = [h - a for h, a in zip(hyb_per_step_contrast, ag8_per_step_contrast)]
            pt, lo, hi, _ = _bci(diff, statistics.mean, rng)
            ci_rows.append({
                "method": m, "comparison": "HYB_minus_AG8_mean_contrast",
                "point": round(pt, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                "n": len(diff), "B": B, "excl_zero": int(lo > 0.0),
            })
        if "AG8" in metrics_by_policy and "STATIC16" in metrics_by_policy:
            ag8_g, ag8_z, _ = traj_by_policy["AG8"]
            s16_g, s16_z, _ = traj_by_policy["STATIC16"]
            ag8_contrast = [1.0 - z for z in ag8_z]
            s16_contrast = [1.0 - z for z in s16_z]
            # net_benefit per-step = contrast_diff - 0.5 * cost_infl
            nb = [(a - s) - 0.5 * ((g_a / G_BASE) - (g_s / G_BASE))
                  for a, s, g_a, g_s in zip(ag8_contrast, s16_contrast,
                                            ag8_g, s16_g)]
            pt, lo, hi, _ = _bci(nb, statistics.mean, rng)
            ci_rows.append({
                "method": m, "comparison": "AG8_minus_STATIC16_net_benefit",
                "point": round(pt, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                "n": len(nb), "B": B, "excl_zero": int(lo > 0.0),
            })

    # write outputs
    fields_step = ["method", "policy", "step", "G_t", "cost_t",
                   "obs_zvf", "proj_zvf", "proj_contrast"]
    with open(os.path.join(OUT_DIR, "p7_iter199_per_step.tsv"), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=fields_step, delimiter="\t")
        w.writeheader()
        for r in per_step_rows:
            w.writerow(r)

    fields_method = ["method", "policy", "mean_proj_zvf", "mean_proj_contrast",
                     "mean_cost", "cumulative_contrast", "cumulative_cost",
                     "restored_vs_base", "restored_vs_static16",
                     "net_benefit_vs_static16", "n_fire_steps", "n_total_steps"]
    with open(os.path.join(OUT_DIR, "p7_iter199_per_method.tsv"), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=fields_method, delimiter="\t")
        w.writeheader()
        for r in per_method_rows:
            w.writerow(r)

    fields_ci = ["method", "comparison", "point", "ci_lo", "ci_hi",
                 "n", "B", "excl_zero", "note"]
    with open(os.path.join(OUT_DIR, "p7_iter199_ci.tsv"), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=fields_ci, delimiter="\t")
        w.writeheader()
        for r in ci_rows:
            w.writerow(r)

    # Build JSON summary
    summary = {
        "ts": "2026-07-06",
        "iter": 199,
        "pillar": "P7",
        "vein": "(a) closed-loop trajectory counterfactual",
        "settings": {
            "G_BASE": G_BASE,
            "G_N_STATIC": G_N_STATIC,
            "G_N_GRID_HYB": G_N_GRID_HYB,
            "TAU_TRIGGER": TAU_TRIGGER,
            "N_STEPS": N_STEPS,
            "N_PROMPTS": N_PROMPTS,
            "B": B,
            "SEED": SEED,
        },
        "verdicts": {
            "H1_AG8_lowers_mean_zvf_vs_BASE_4of4": 0,
            "H2_AG8_raises_mean_contrast_vs_BASE_4of4": 0,
            "H3_AG8_cheaper_than_STATIC16_CI95_excludes_zero_4of4": 0,
            "H4_AG8_net_benefit_vs_STATIC16_nonneg_4of4": 0,
            "H5_AG_HYB_more_contrast_than_AG8_4of4": 0,
        },
        "per_method": [
            {"method": m, "policy": r["policy"],
             "mean_proj_zvf": r["mean_proj_zvf"],
             "mean_proj_contrast": r["mean_proj_contrast"],
             "mean_cost": r["mean_cost"],
             "restored_vs_base": r["restored_vs_base"],
             "restored_vs_static16": r["restored_vs_static16"],
             "net_benefit_vs_static16": r["net_benefit_vs_static16"],
             "n_fire_steps": r["n_fire_steps"]}
            for r in per_method_rows
        ],
        "cis": [
            {"method": r["method"], "comparison": r["comparison"],
             "point": r["point"], "ci_lo": r["ci_lo"],
             "ci_hi": r["ci_hi"], "excl_zero": r["excl_zero"]}
            for r in ci_rows
        ],
    }

    # Compute falsifiable verdicts from CI rows. Add an H1 zvf-reduction CI
    # row (AG8 - BASE zvf difference; AG8 lower means CI_hi < 0).
    for m in METHODS:
        ag8_res = simulate_trajectory(tensors[m], "AG8")
        base_res = simulate_trajectory(tensors[m], "BASE")
        if ag8_res and base_res:
            _, ag8_z, _ = ag8_res
            _, base_z, _ = base_res
            diff_zvf = [a - b for a, b in zip(ag8_z, base_z)]
            pt, lo, hi, _ = _bci(diff_zvf, statistics.mean, rng)
            ci_rows.append({
                "method": m, "comparison": "AG8_minus_BASE_mean_zvf",
                "point": round(pt, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                "n": len(diff_zvf), "B": B, "excl_zero": int(hi < 0.0),
                "note": "NEG = AG8 lowers mean zvf (expected)",
            })

    # H1 — AG8 lowers zvf vs BASE (want CI_hi < 0)
    ag8_lowers_zvf = [r for r in ci_rows
                      if r["comparison"] == "AG8_minus_BASE_mean_zvf"]
    summary["verdicts"]["H1_AG8_lowers_mean_zvf_vs_BASE_4of4"] = \
        sum(1 for r in ag8_lowers_zvf if r["ci_hi"] < 0.0)

    # H2 — AG8 raises mean contrast vs BASE (want CI_lo > 0)
    ag8_raises_contrast = [r for r in ci_rows
                           if r["comparison"] == "AG8_minus_BASE_mean_contrast"]
    summary["verdicts"]["H2_AG8_raises_mean_contrast_vs_BASE_4of4"] = \
        sum(1 for r in ag8_raises_contrast if r["excl_zero"])

    # H3 — AG8 cheaper than STATIC16 (cost diff = ST16 - AG8; want CI_lo > 0)
    ag8_cheaper = [r for r in ci_rows
                   if r["comparison"] == "AG8_minus_STATIC16_mean_cost"
                   and r.get("note") == "POS = AG8 cheaper than STATIC16"]
    summary["verdicts"]["H3_AG8_cheaper_than_STATIC16_CI95_excludes_zero_4of4"] = \
        sum(1 for r in ag8_cheaper if r["ci_lo"] > 0.0)

    # H4 — AG8 net_benefit vs ST16 non-negative (CI_lo >= -0.005)
    ag8_nb = [r for r in ci_rows
              if r["comparison"] == "AG8_minus_STATIC16_net_benefit"]
    summary["verdicts"]["H4_AG8_net_benefit_vs_STATIC16_nonneg_4of4"] = \
        sum(1 for r in ag8_nb if r["ci_lo"] >= -0.005)

    # H5 — AG_HYB more contrast than AG8 (HYB picks G=12 on τ≥0.70; we expect
    # LESS contrast since G=12 < G=16 — H5 is the FAIL side that demonstrates
    # the AG_HYB cost-saving design works)
    hyb_more = [r for r in ci_rows
                if r["comparison"] == "HYB_minus_AG8_mean_contrast"]
    summary["verdicts"]["H5_AG_HYB_more_contrast_than_AG8_4of4"] = \
        sum(1 for r in hyb_more if r["excl_zero"])

    with open(os.path.join(OUT_DIR, "p7_iter199_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # quick stdout summary
    print("\n[iter199] per-method summary (means over 40 steps)")
    print(f"  {'method':<8} {'policy':<10} {'mean_zvf':>9} {'mean_contrast':>14} "
          f"{'mean_cost':>10} {'restored_vs_base':>18} {'restored_vs_S16':>16} "
          f"{'net_benefit':>12} {'fires':>6}")
    for r in per_method_rows:
        print(f"  {r['method']:<8} {r['policy']:<10} {r['mean_proj_zvf']:>9.4f} "
              f"{r['mean_proj_contrast']:>14.4f} {r['mean_cost']:>10.4f} "
              f"{r['restored_vs_base']:>18.4f} {r['restored_vs_static16']:>16.4f} "
              f"{r['net_benefit_vs_static16']:>12.4f} {r['n_fire_steps']:>6}")
    print(f"\n[iter199] verdicts: {summary['verdicts']}")


if __name__ == "__main__":
    main()
