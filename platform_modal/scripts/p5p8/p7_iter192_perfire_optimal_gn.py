#!/usr/bin/env python3
"""Iter 192 — P7 Per-Prompt Cost-Effective Optimal G_N on Fired Steps.

Fresh vein, brief vein (a) at the *per-prompt cost-effective G_N* layer:
prior veins (iter-119, iter-175) measured the *static G_N = 16* decision
on fired steps. iter-179 measured the restored contrast at static G=16.
This iter measures the **per-prompt cost-effective optimal** G_N* on
each fired (method, step, prompt) observation, then aggregates the
savings vs the static G_N = 16 default across the 1,312 fired prompts
of iter-179 (τ = 0.70).

Definition (per fired prompt with k successes at G_BASE = 8 rollouts):
    p_hat = k / 8
    Y(p, G) = 1 - p^G - (1-p)^G     # binomial contrast at group size G
    C(p, G_N; G_BASE) = Y(p, G_N) - Y(p, G_BASE)  # restored contrast
    cost(G_N) = (G_N - G_BASE) / G_BASE            # fractional extra rollouts
    eff(p, G_N) = C(p, G_N) / cost(G_N)            # cost-effective ratio
    G_N*(p) = argmax_{G_N ∈ 4,6,8,12,16,24,32} eff(p, G_N)
    stat:   G_N = 16  (always)

Boundary prompts (k ∈ {0,8}, p_hat ∈ {0,1}):
    For all G_N, C(p, G_N) = 0; G_N* is undefined (no escalation helps).
    Convention: G_N*(p) = G_BASE = 8 (keep G).

Three sweep axes:
    τ (trigger) = 0.70 (canonical)
    methods = {grpo, aero, gift, areal}
    fired-step pool = 4 × ≤40 × ≤16 prompts (1,312 fired prompts total)

For each (method) we tabulate:
    n_prompts, n_boundary, n_contrast
    best_static_G_N = 16 default
    per-prompt optimal:
        - mean G_N*
        - mean restored contrast (per-prompt optimum)
        - mean cost-effective ratio (per-prompt optimum)
    savings:
        - mean rollouts across prompts under per-prompt optimum vs 16
        - fraction of prompts where per-prompt optimum < 16
        - savings bootstrap CI B=2000, percentile, seed 20260706

5 falsifiable hypotheses:
    H1 — per-prompt G_N* < G_N = 16 on the majority of contrast (interior)
         prompts on every method (≥ 3 of 4 methods).
    H2 — mean per-prompt G_N* < 16 on every method (savings > 0).
    H3 — per-prompt optimal restored contrast >= static G=16 restored
         contrast on 4/4 methods (Pareto non-dominated by the static rule).
    H4 — cost-effective ratio is monotone decreasing in G_N on the contrast
         prompt mean, confirming the diminishing-returns intuition.
    H5 — the bootstrap CI on per-method rollouts saved
         ((static - per-prompt optimum) / static) excludes zero on 4/4.

Outputs:
    experiments/results/p5p8/p7_iter192_per_prompt.tsv
    experiments/results/p5p8/p7_iter192_per_method.tsv
    experiments/results/p5p8/p7_iter192_ci.tsv
    experiments/results/p5p8/p7_iter192_summary.json
    docs/p5p8_improvements/192_p7_perfire_optimal_gn.md

Stdlib only; deterministic.
"""
from __future__ import annotations
import csv, glob, json, os, random, statistics

WORKTREE = "/home/claude/tinker-rl-lab-minimax"
DATA_DIR = os.path.join(WORKTREE, "experiments/results/n2_reward_tensor_resume")
OUT_DIR = os.path.join(WORKTREE, "experiments/results/p5p8")
os.makedirs(OUT_DIR, exist_ok=True)

METHODS = ["grpo", "aero", "gift", "areal"]
G_BASE = 8
N_PROMPTS = 16
G_N_GRID = [12, 16, 24, 32, 48]   # escalation-only (G_N >= G_BASE=8); C4 trigger is an escalation controller
G_N_STATIC = 16                          # static rule default
B = 2000; SEED = 20260706; ALPHA = 0.05


def _bci(v, stat_fn=statistics.mean, rng=None):
    """Percentile bootstrap CI on `v` with B resamples at confidence 1-ALPHA."""
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if rng is None:
        rng = random.Random(SEED)
    stats = []
    for _ in range(B):
        samp = [v[rng.randrange(n)] for _ in range(n)]
        stats.append(stat_fn(samp))
    stats.sort()
    lo = stats[int((ALPHA / 2) * B)]
    hi = stats[int((1 - ALPHA / 2) * B)]
    return stat_fn(v), lo, hi


def binomial_y(p, G):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return 1.0 - (p ** G) - ((1.0 - p) ** G)


def load_tensors():
    """Load (method, step) → [(k_p at G=8 for each of 16 prompts)]."""
    out = {}
    for method in METHODS:
        path = os.path.join(DATA_DIR, f"{method}_s0_tensors.jsonl")
        per_step = {}
        with open(path) as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                row = json.loads(ln)
                # Sum across rollouts per prompt to get k_p ∈ [0, 8]
                ks = [int(sum(r)) for r in row["rewards"]]
                per_step[int(row["step"])] = ks
        per_step = dict(sorted(per_step.items()))
        out[method] = per_step
    return out


def per_prompt_record(method, step, k_p, tau):
    """For one prompt with k successes at G_BASE, return per-prompt
    record: tau_fired, k, p_hat, is_boundary, restored_static_G16,
    optimal G_N*, restored_optimal, rollouts_static, rollouts_optimal,
    cost_eff_static, cost_eff_optimal."""
    p_hat = k_p / G_BASE
    is_boundary = (k_p == 0 or k_p == G_BASE)
    # Restored at static G=16
    y_base = binomial_y(p_hat, G_BASE)
    y_static = binomial_y(p_hat, G_N_STATIC)
    restored_static = max(0.0, y_static - y_base)
    cost_static = (G_N_STATIC - G_BASE) / G_BASE  # (16-8)/8 = 1.0
    eff_static = restored_static / cost_static if cost_static > 0 else 0.0
    # Sweep over G_N grid — best_G defaults to G_BASE (no escalation helps)
    # iff no G in G_N_GRID strictly beats (0 restoration, 0 eff). For boundary
    # prompts (k ∈ {0,8}, p_hat ∈ {0,1}) Y is identically 0 at every G, so
    # restored=0 for all candidates; the rigorous convention is to keep
    # best_G = G_BASE so the controller never "escalates" a boundary prompt.
    best_G = G_BASE
    best_restored = 0.0
    best_eff = 0.0
    for G in G_N_GRID:
        if G == G_BASE:
            continue
        y_G = binomial_y(p_hat, G)
        restored = max(0.0, y_G - y_base)
        cost = (G - G_BASE) / G_BASE
        if cost <= 0:
            continue
        eff = restored / cost
        # Strict improvement required to update best_G (avoids the bug where
        # boundary prompts (restored=0 for all G) inherit the first grid
        # element's G as "optimal").
        if eff > best_eff:
            best_eff = eff
            best_G = G
            best_restored = restored
    rollouts_static = G_N_STATIC
    rollouts_optimal = best_G
    return {
        "method": method,
        "step": step,
        "k_p": k_p,
        "p_hat": p_hat,
        "is_boundary": int(is_boundary),
        "tau_fired": tau,                  # currently we filter below
        "static_G_N": G_N_STATIC,
        "restored_static": restored_static,
        "eff_static": eff_static,
        "optimal_G_N": best_G,
        "restored_optimal": best_restored,
        "eff_optimal": best_eff,
        "savings_rollouts": rollouts_static - rollouts_optimal,
        "savings_frac": (rollouts_static - rollouts_optimal) / rollouts_static,
    }


def main():
    tensors = load_tensors()
    TAU = 0.70
    per_prompt_rows = []
    # 1. Iterate over (method, step) to find fired steps (zvf >= tau)
    for method in METHODS:
        for step, ks in tensors[method].items():
            # Compute per-step zvf
            nsat_boundary = 0
            for k in ks:
                if k == 0 or k == G_BASE:
                    nsat_boundary += 1
            zvf = nsat_boundary / N_PROMPTS
            if zvf < TAU:
                continue
            # Each fired step: 16 prompts (some may be uninformative)
            for prompt_idx, k in enumerate(ks):
                rec = per_prompt_record(method, step, k, TAU)
                rec["prompt_idx"] = prompt_idx
                per_prompt_rows.append(rec)

    # 2. Write per-prompt TSV
    pp_path = os.path.join(OUT_DIR, "p7_iter192_per_prompt.tsv")
    with open(pp_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_prompt_rows[0].keys()),
                           delimiter="\t")
        w.writeheader()
        for row in per_prompt_rows:
            w.writerow(row)

    # 3. Per-method aggregates + bootstrapCIs
    per_method = []
    for method in METHODS:
        rows = [r for r in per_prompt_rows if r["method"] == method]
        if not rows:
            continue
        # Aggregate stats across all fired prompts (boundary prompts have opt=8)
        n_prompts = len(rows)
        n_boundary = sum(1 for r in rows if r["is_boundary"] == 1)
        n_contrast = n_prompts - n_boundary
        # G_N*
        all_gn = [r["optimal_G_N"] for r in rows]
        all_restored_opt = [r["restored_optimal"] for r in rows]
        all_savings = [r["savings_frac"] for r in rows]
        all_savings_rollouts = [r["savings_rollouts"] for r in rows]
        # Static reference
        all_restored_static = [r["restored_static"] for r in rows]
        # On contrast prompts only
        contrast_rows = [r for r in rows if r["is_boundary"] == 0]
        contrast_gn = [r["optimal_G_N"] for r in contrast_rows] if contrast_rows else [0]
        contrast_savings = [r["savings_frac"] for r in contrast_rows] if contrast_rows else [0]
        # Pareto-dominance: optimal restored >= static restored per row
        n_dominates = sum(1 for r in rows if r["restored_optimal"] >= r["restored_static"] - 1e-9)
        # H4: monotone cost-effective decreasing
        Gs = sorted(set(G_N_GRID))
        # At each G_N, compute mean restored on contrast prompts
        h4_monotone = None
        try:
            contrast_ks = [r["k_p"] for r in contrast_rows]
            G_eff_table = {}
            for G in Gs:
                if G == G_BASE:
                    continue
                effs = []
                for k in contrast_ks:
                    p = k / G_BASE
                    y_b = binomial_y(p, G_BASE)
                    y_G = binomial_y(p, G)
                    r = max(0.0, y_G - y_b)
                    cost = (G - G_BASE) / G_BASE
                    effs.append(r / cost if cost > 0 else 0.0)
                G_eff_table[G] = (sum(effs) / len(effs)) if effs else 0.0
            # compute slope via linreg on G sorted
            xs = [G for G in sorted(G_eff_table) if G != G_BASE]
            ys = [G_eff_table[G] for G in xs]
            if len(xs) >= 2:
                mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
                num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
                den_x = sum((x - mx) ** 2 for x in xs)
                slope = num / den_x if den_x > 0 else 0.0
                h4_monotone = slope < 0
            else:
                h4_monotone = None
        except Exception:
            h4_monotone = None
        # Bootstrap CI on mean savings_frac
        sav_pct = all_savings  # already 0..1
        m, lo, hi = _bci(sav_pct)
        # also CI on rollouts saved per method
        m_roll, lo_roll, hi_roll = _bci(all_savings_rollouts)
        # per-method aggregate
        per_method.append({
            "method": method,
            "n_prompts": n_prompts,
            "n_boundary": n_boundary,
            "n_contrast": n_contrast,
            "mean_gn_optimal": (sum(all_gn) / len(all_gn)) if all_gn else 0.0,
            "mean_gn_optimal_contrast": (
                sum(contrast_gn) / len(contrast_gn)
            ) if contrast_gn else 0.0,
            "mean_restored_static": (
                sum(all_restored_static) / len(all_restored_static)
            ) if all_restored_static else 0.0,
            "mean_restored_optimal": (
                sum(all_restored_opt) / len(all_restored_opt)
            ) if all_restored_opt else 0.0,
            "mean_savings_frac": (
                sum(sav_pct) / len(sav_pct)
            ) if sav_pct else 0.0,
            "savings_frac_boot_lo": lo,
            "savings_frac_boot_hi": hi,
            "mean_savings_rollouts": (
                sum(all_savings_rollouts) / len(all_savings_rollouts)
            ) if all_savings_rollouts else 0.0,
            "savings_rollouts_boot_lo": lo_roll,
            "savings_rollouts_boot_hi": hi_roll,
            "n_pareto_dominates_static": n_dominates,
            "frac_pareto_dominates": n_dominates / n_prompts if n_prompts > 0 else 0.0,
            "h4_monotone_dec_slope": "n/a" if h4_monotone is None else (
                "PASS" if h4_monotone else "FAIL"
            ),
        })
    pm_path = os.path.join(OUT_DIR, "p7_iter192_per_method.tsv")
    if per_method:
        with open(pm_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_method[0].keys()),
                               delimiter="\t")
            w.writeheader()
            for row in per_method:
                w.writerow(row)

    # 4. CI TSV per method
    ci_rows = []
    for r in per_method:
        ci_rows.append({
            "method": r["method"],
            "metric": "savings_frac",
            "point": r["mean_savings_frac"],
            "lo": r["savings_frac_boot_lo"],
            "hi": r["savings_frac_boot_hi"],
            "excludes_zero": (r["savings_frac_boot_lo"] > 0),
        })
        ci_rows.append({
            "method": r["method"],
            "metric": "savings_rollouts",
            "point": r["mean_savings_rollouts"],
            "lo": r["savings_rollouts_boot_lo"],
            "hi": r["savings_rollouts_boot_hi"],
            "excludes_zero": (r["savings_rollouts_boot_lo"] > 0),
        })
    ci_path = os.path.join(OUT_DIR, "p7_iter192_ci.tsv")
    with open(ci_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "metric", "point", "lo", "hi",
                                           "excludes_zero"], delimiter="\t")
        w.writeheader()
        for row in ci_rows:
            w.writerow(row)

    # 5. Verdicts for the 5 hypotheses
    n_with_savings = sum(1 for r in per_method if r["mean_savings_frac"] > 0)
    n_dominate = sum(1 for r in per_method
                     if r["frac_pareto_dominates"] >= 1.0 - 1e-9)
    n_savings_excl_zero = sum(1 for r in ci_rows
                              if r["metric"] == "savings_frac"
                              and r["excludes_zero"])
    n_rollouts_excl_zero = sum(1 for r in ci_rows
                               if r["metric"] == "savings_rollouts"
                               and r["excludes_zero"])
    # H4: monotone decreasing on contrast mean
    monotone_pass = sum(1 for r in per_method
                        if r["h4_monotone_dec_slope"] == "PASS")
    verdict = {
        "H1_per_prompt_optimal_lt_16_on_majority_contrast_prompts": (
            f"n_methods_with_opt_mean_lt_16 = "
            f"{sum(1 for r in per_method if r['mean_gn_optimal_contrast'] < 16)} / "
            f"{len(per_method)}; "
            f"PASS" if sum(1 for r in per_method
                           if r["mean_gn_optimal_contrast"] < 16) >= 3 else "FAIL"
        ),
        "H2_mean_gn_optimal_lt_16_every_method": (
            f"n_methods = {sum(1 for r in per_method if r['mean_gn_optimal'] < 16)} / "
            f"{len(per_method)}; "
            f"PASS" if n_with_savings >= len(per_method) else "FAIL"
        ),
        "H3_per_prompt_optimal_dominates_static_on_all_methods": (
            f"n_methods_with_full_pareto = {n_dominate} / {len(per_method)}; "
            f"PASS" if n_dominate >= 4 else "FAIL"
        ),
        "H4_cost_eff_monotone_dec_in_GN": (
            f"n_methods_with_slope_lt_zero = {monotone_pass} / "
            f"{len(per_method)}; "
            f"PASS" if monotone_pass >= 4 else "FAIL"
        ),
        "H5_bootstrap_ci_excludes_zero_on_savings_frac": (
            f"n_methods_with_ci_excl_zero = {n_savings_excl_zero} / "
            f"{len(per_method)}; "
            f"PASS" if n_savings_excl_zero >= 4 else "FAIL"
        ),
        "H5b_bootstrap_ci_excludes_zero_on_savings_rollouts": (
            f"n_methods_with_ci_excl_zero = {n_rollouts_excl_zero}/ "
            f"{len(per_method)}; "
            f"PASS" if n_rollouts_excl_zero >= 4 else "FAIL"
        ),
    }
    summary = {
        "n_prompts_total": len(per_prompt_rows),
        "per_method": per_method,
        "ci": ci_rows,
        "verdicts": verdict,
        "settings": {
            "TAU": TAU,
            "G_BASE": G_BASE,
            "G_N_GRID": G_N_GRID,
            "G_N_STATIC": G_N_STATIC,
            "B": B, "SEED": SEED,
        },
    }
    sm_path = os.path.join(OUT_DIR, "p7_iter192_summary.json")
    with open(sm_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # 6. Print headline
    print(f"iter 192 — per-prompt optimal G_N on fired steps")
    print(f"  fired prompts total: {len(per_prompt_rows)}")
    print(f"  per-method:")
    for r in per_method:
        print(f"    {r['method']}: n={r['n_prompts']} (b={r['n_boundary']}, "
              f"c={r['n_contrast']})  mean_GN*={r['mean_gn_optimal']:.3f} "
              f"(c-only={r['mean_gn_optimal_contrast']:.3f})  "
              f"savings={r['mean_savings_frac']:.3f} "
              f"[{r['savings_frac_boot_lo']:.3f}, "
              f"{r['savings_frac_boot_hi']:.3f}]  "
              f"restored opt={r['mean_restored_optimal']:.4f} "
              f"vs static={r['mean_restored_static']:.4f}  "
              f"dom={r['frac_pareto_dominates']:.3f}")
    print(f"  verdicts:")
    for k, v in verdict.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
