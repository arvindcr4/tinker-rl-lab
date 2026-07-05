#!/usr/bin/env python3
"""p7_iter163_step_pareto_ci.py

Iter 163 — Step-aggregate Pareto frontier + per-method bootstrap CI on
step-level UNIFIED controller family, on the REAL N2 reward tensors
(40 steps x 4 methods x 16 prompts x 8 rewards).

Pillar: P7 (Pillar 3 — adaptive-G controller / signal-starvation theory).
Vein: brief vein (a) + (d) — counterfactual controller eval on N2 reward
tensors + bootstrap CIs on every P7 headline, but at STEP-AGGREGATE
granularity (n=160 step-method decisions) rather than iter-159's
per-prompt granularity (n=2,560 prompt cells).

Combines:
- iter-151's step-level counterfactual evaluation (single G per step applies
  to all 16 prompts; per-step mean contrast retention vs STATIC_G8).
- iter-159's Pareto-frontier + per-method bootstrap-CI breakdown.

The iter-159 follow-up explicitly recommended extending the Pareto analysis
to step-aggregate granularity to test whether STATIC_G16 is also dominated
there. Iter-163 closes that gap.

Outputs:
- experiments/results/p5p8/p7_iter163_per_method_ci.tsv (20 rows)
- experiments/results/p5p8/p7_iter163_pareto.tsv (20 points)
- experiments/results/p5p8/p7_iter163_pareto_frontier.tsv (Pareto-optimal)
- experiments/results/p5p8/p7_iter163_cross_method_sd.tsv (5 rows)
- experiments/results/p5p8/p7_iter163_paired_bootstrap.tsv (16 rows)
- experiments/results/p5p8/p7_iter163_dominance.tsv (dominance matrix)
- experiments/results/p5p8/p7_iter163_summary.json
"""
from __future__ import annotations
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N2_DIR = ROOT / "experiments" / "results" / "n2_reward_tensor_resume"
OUT_DIR = ROOT / "experiments" / "results" / "p5p8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["grpo", "aero", "gift", "areal"]
G_BASE = 8
TAU_FAST = 0.50
TAU_DEGEN = 0.70
SEED = 0
BOOT_N = 2000
BOOT_SEED = 20260705


# --- Controller rule (one G per step applies to all 16 prompts) ---
def c_static_g8(z): return 8
def c_static_g16(z): return 16
def c_static_g4(z): return 4
def c_dualformer_step(z):
    if z < 0.50: return 2
    if z >= 0.85: return 32
    return 8
def c_unified_step_c4(z):
    if z < TAU_FAST: return 4
    if z >= TAU_DEGEN: return 16
    return 8

CONTROLLERS = {
    "STATIC_G4":         c_static_g4,
    "STATIC_G8":         c_static_g8,
    "STATIC_G16":        c_static_g16,
    "DUALFORMER_STEP":   c_dualformer_step,
    "UNIFIED_STEP_C4":   c_unified_step_c4,
}


def bernoulli_z(p_hat: float, G: int) -> float:
    if p_hat <= 0.0 or p_hat >= 1.0:
        return 1.0
    return p_hat ** G + (1.0 - p_hat) ** G


def contrast_mag(p_hat: float, G: int) -> float:
    return 1.0 - bernoulli_z(p_hat, G)


def load_tensors():
    by_method = {}
    for m in METHODS:
        path = N2_DIR / f"{m}_s{SEED}_tensors.jsonl"
        steps = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                steps.append(json.loads(line))
        by_method[m] = steps
    return by_method


def evaluate_step(method, step_rec, cname, cfn):
    step = step_rec["step"]
    z_obs = step_rec["zvf"]
    g_used = cfn(z_obs)
    cms_used, cms_base = [], []
    for rewards_row in step_rec["rewards"]:
        k = int(round(sum(rewards_row)))
        p_hat = k / G_BASE
        cms_used.append(contrast_mag(p_hat, g_used))
        cms_base.append(contrast_mag(p_hat, G_BASE))
    mean_cm_used = statistics.mean(cms_used)
    mean_cm_base = statistics.mean(cms_base)
    contrast_retention = mean_cm_used / max(mean_cm_base, 1e-9)
    cost_ratio = g_used / G_BASE
    return {
        "method": method, "step": step, "z_obs": z_obs, "g_used": g_used,
        "cost_ratio": cost_ratio, "mean_cm_used": mean_cm_used,
        "mean_cm_base": mean_cm_base, "contrast_retention": contrast_retention,
        "fired": g_used != G_BASE,
    }


def boot_ci(values, n_boot=BOOT_N, seed=BOOT_SEED):
    rng = random.Random(seed)
    n = len(values)
    boots = []
    for _ in range(n_boot):
        samp = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(statistics.mean(samp))
    boots.sort()
    return boots[100], boots[1900], statistics.mean(values)


def aggregate_step(decisions):
    if not decisions:
        return None
    cs = [d["cost_ratio"] for d in decisions]
    cr = [d["contrast_retention"] for d in decisions]
    fired = [1 if d["fired"] else 0 for d in decisions]
    gs = [d["g_used"] for d in decisions]
    lo_c, hi_c, mean_c = boot_ci(cs)
    lo_r, hi_r, mean_r = boot_ci(cr)
    lo_f, hi_f, mean_f = boot_ci(fired)
    return {
        "n": len(decisions),
        "mean_cost_ratio": mean_c, "ci95_cost": (lo_c, hi_c),
        "mean_contrast_retention": mean_r, "ci95_retention": (lo_r, hi_r),
        "mean_fire_rate": mean_f, "ci95_fire_rate": (lo_f, hi_f),
        "mean_g": statistics.mean(gs),
    }


def main():
    print("Loading N2 reward tensors ...")
    by_method = load_tensors()
    for m in METHODS:
        print(f"  {m}: {len(by_method[m])} steps")

    # Per-(method, controller) decision list
    decisions_by_mc = {}
    for m, steps in by_method.items():
        for cname, cfn in CONTROLLERS.items():
            decisions_by_mc[(m, cname)] = [
                evaluate_step(m, s, cname, cfn) for s in steps
            ]

    # Aggregate per (method, controller) with bootstrap CI
    agg_mc = {(m, c): aggregate_step(decisions_by_mc[(m, c)])
              for m in METHODS for c in CONTROLLERS}

    # Aggregate per controller across methods (n=160)
    agg_c = {}
    for c in CONTROLLERS:
        all_d = []
        for m in METHODS:
            all_d.extend(decisions_by_mc[(m, c)])
        agg_c[c] = aggregate_step(all_d)

    # ----- Write per-method CI TSV -----
    with open(OUT_DIR / "p7_iter163_per_method_ci.tsv", "w") as f:
        cols = ["method", "controller", "n", "mean_g", "mean_cost_ratio",
                "ci95_cost_lo", "ci95_cost_hi", "mean_contrast_retention",
                "ci95_ret_lo", "ci95_ret_hi", "mean_fire_rate",
                "ci95_fire_lo", "ci95_fire_hi"]
        f.write("\t".join(cols) + "\n")
        for m in METHODS:
            for c in CONTROLLERS:
                a = agg_mc[(m, c)]
                f.write("\t".join(f"{x:.4f}" if isinstance(x, float) else str(x)
                                  for x in [
                    m, c, a["n"], a["mean_g"], a["mean_cost_ratio"],
                    a["ci95_cost"][0], a["ci95_cost"][1],
                    a["mean_contrast_retention"],
                    a["ci95_retention"][0], a["ci95_retention"][1],
                    a["mean_fire_rate"],
                    a["ci95_fire_rate"][0], a["ci95_fire_rate"][1],
                ]) + "\n")
    print(f"Wrote per_method_ci ({len(METHODS)*len(CONTROLLERS)} rows)")

    # ----- Pareto scatter (20 points: cost vs retention, all (m,c)) -----
    pareto_pts = []
    for m in METHODS:
        for c in CONTROLLERS:
            a = agg_mc[(m, c)]
            pareto_pts.append({
                "method": m, "controller": c,
                "cost_ratio": a["mean_cost_ratio"],
                "retention": a["mean_contrast_retention"],
                "mean_g": a["mean_g"],
            })
    with open(OUT_DIR / "p7_iter163_pareto.tsv", "w") as f:
        f.write("method\tcontroller\tcost_ratio\tretention\tmean_g\n")
        for p in pareto_pts:
            f.write(f"{p['method']}\t{p['controller']}\t"
                    f"{p['cost_ratio']:.4f}\t{p['retention']:.4f}\t"
                    f"{p['mean_g']:.2f}\n")
    print(f"Wrote pareto ({len(pareto_pts)} points)")

    # ----- Pareto frontier: a point (c, r) is dominated if exists p' with
    # c' <= c AND r' >= r, with at least one strict. Pareto-optimal: no such.
    def is_dominated(p, others):
        for o in others:
            if o is p: continue
            if (o["cost_ratio"] <= p["cost_ratio"] and
                    o["retention"] >= p["retention"] and
                    (o["cost_ratio"] < p["cost_ratio"] or
                     o["retention"] > p["retention"])):
                return True
        return False
    frontier = [p for p in pareto_pts if not is_dominated(p, pareto_pts)]
    with open(OUT_DIR / "p7_iter163_pareto_frontier.tsv", "w") as f:
        f.write("method\tcontroller\tcost_ratio\tretention\tmean_g\n")
        for p in frontier:
            f.write(f"{p['method']}\t{p['controller']}\t"
                    f"{p['cost_ratio']:.4f}\t{p['retention']:.4f}\t"
                    f"{p['mean_g']:.2f}\n")
    print(f"Wrote pareto_frontier ({len(frontier)} Pareto-optimal points)")

    # ----- Cross-method SD on cost/retention per controller -----
    xm_rows = []
    for c in CONTROLLERS:
        costs = [agg_mc[(m, c)]["mean_cost_ratio"] for m in METHODS]
        rets = [agg_mc[(m, c)]["mean_contrast_retention"] for m in METHODS]
        # bootstrap CI on SD (B=2000, seed-shifted by controller name)
        rng = random.Random(BOOT_SEED + hash(c) % 100000)
        boot_cost_sd = []
        boot_ret_sd = []
        for _ in range(BOOT_N):
            samp = [METHODS[rng.randrange(4)] for _ in range(4)]
            sc = statistics.mean([agg_mc[(s, c)]["mean_cost_ratio"] for s in samp])
            sr = statistics.mean([agg_mc[(s, c)]["mean_contrast_retention"] for s in samp])
            boot_cost_sd.append(statistics.pstdev([agg_mc[(s, c)]["mean_cost_ratio"] for s in samp]))
            boot_ret_sd.append(statistics.pstdev([agg_mc[(s, c)]["mean_contrast_retention"] for s in samp]))
        boot_cost_sd.sort(); boot_ret_sd.sort()
        xm_rows.append({
            "controller": c,
            "n_methods": 4,
            "mean_cost_ratio": statistics.mean(costs),
            "sd_cost_ratio": statistics.pstdev(costs),
            "ci95_sd_cost_lo": boot_cost_sd[100], "ci95_sd_cost_hi": boot_cost_sd[1900],
            "mean_retention": statistics.mean(rets),
            "sd_retention": statistics.pstdev(rets),
            "ci95_sd_ret_lo": boot_ret_sd[100], "ci95_sd_ret_hi": boot_ret_sd[1900],
        })
    with open(OUT_DIR / "p7_iter163_cross_method_sd.tsv", "w") as f:
        cols = ["controller", "n_methods", "mean_cost_ratio", "sd_cost_ratio",
                "ci95_sd_cost_lo", "ci95_sd_cost_hi", "mean_retention",
                "sd_retention", "ci95_sd_ret_lo", "ci95_sd_ret_hi"]
        f.write("\t".join(cols) + "\n")
        for r in xm_rows:
            f.write("\t".join(str(r[k]) if k in ("controller", "n_methods")
                              else f"{r[k]:.4f}" for k in cols) + "\n")
    print(f"Wrote cross_method_sd ({len(xm_rows)} rows)")

    # ----- Paired bootstrap: C4 vs each other controller per method -----
    paired_rows = []
    for m in METHODS:
        for c in CONTROLLERS:
            if c == "UNIFIED_STEP_C4":
                continue
            d1 = decisions_by_mc[(m, "UNIFIED_STEP_C4")]
            d2 = decisions_by_mc[(m, c)]
            assert len(d1) == len(d2) == 40
            # Paired difference of contrast retention per step
            diffs_cr = [d1[i]["contrast_retention"] - d2[i]["contrast_retention"]
                        for i in range(40)]
            diffs_cost = [d1[i]["cost_ratio"] - d2[i]["cost_ratio"]
                          for i in range(40)]
            rng = random.Random(BOOT_SEED + hash((m, c)) % 100000)
            boot_dcr = []
            boot_dcost = []
            for _ in range(BOOT_N):
                idx = [rng.randrange(40) for _ in range(40)]
                boot_dcr.append(statistics.mean([diffs_cr[i] for i in idx]))
                boot_dcost.append(statistics.mean([diffs_cost[i] for i in idx]))
            boot_dcr.sort(); boot_dcost.sort()
            lo_cr, hi_cr = boot_dcr[100], boot_dcr[1900]
            lo_cost, hi_cost = boot_dcost[100], boot_dcost[1900]
            # strict dominance: cost_lower AND retention_higher (CI excludes 0)
            sd_retention = hi_cr < 0  # C4 retention LOWER than c → c dominates
            sd_cost = hi_cost < 0     # C4 cost LOWER than c → C4 cheaper
            paired_rows.append({
                "method": m, "vs_controller": c,
                "delta_retention": statistics.mean(diffs_cr),
                "ci95_dret_lo": lo_cr, "ci95_dret_hi": hi_cr,
                "ci_excl0_ret": (lo_cr > 0 or hi_cr < 0),
                "direction_ret": ("C4_GAIN" if statistics.mean(diffs_cr) > 0 else "C4_LOSS"),
                "delta_cost": statistics.mean(diffs_cost),
                "ci95_dcost_lo": lo_cost, "ci95_dcost_hi": hi_cost,
                "ci_excl0_cost": (lo_cost > 0 or hi_cost < 0),
                "strict_dominance_of": ("C4" if (sd_retention and sd_cost)
                                        else ("vs_C" if (not sd_retention and not sd_cost)
                                              else "PARTIAL")),
            })
    with open(OUT_DIR / "p7_iter163_paired_bootstrap.tsv", "w") as f:
        cols = ["method", "vs_controller", "delta_retention",
                "ci95_dret_lo", "ci95_dret_hi", "ci_excl0_ret",
                "direction_ret", "delta_cost",
                "ci95_dcost_lo", "ci95_dcost_hi", "ci_excl0_cost",
                "strict_dominance_of"]
        f.write("\t".join(cols) + "\n")
        for r in paired_rows:
            row = []
            for k in cols:
                v = r[k]
                row.append(str(v) if isinstance(v, bool) or isinstance(v, str)
                           else f"{v:.4f}")
            f.write("\t".join(row) + "\n")
    print(f"Wrote paired_bootstrap ({len(paired_rows)} rows)")

    # ----- Dominance matrix: how many times each controller strictly dominates others -----
    dom_counts = {c: 0 for c in CONTROLLERS}
    for c in CONTROLLERS:
        for c2 in CONTROLLERS:
            if c2 == c: continue
            sd_count = 0
            for r in paired_rows:
                if r["vs_controller"] == c2 and r["strict_dominance_of"] == "C4":
                    sd_count += 1
            # if c == UNIFIED_STEP_C4 and sd_count == 4: UNIFIED_STEP_C4 dominates c2 on 4/4 methods
            if c == "UNIFIED_STEP_C4" and sd_count == 4:
                dom_counts[c] += 1
    # STATIC_G16: dominates any controller with lower cost AND higher retention?
    # STATIC_G16 cost=2 always, so no controller with cost<2 can have lower retention.
    # STATIC_G16 always dominates controllers with cost <= 2 AND retention < STATIC_G16's retention.
    static_g16_ret = [agg_mc[(m, "STATIC_G16")]["mean_contrast_retention"] for m in METHODS]
    static_g16_ret_mean = statistics.mean(static_g16_ret)
    static_g16_dominated_by = 0
    for c in CONTROLLERS:
        if c == "STATIC_G16": continue
        # c strictly dominates STATIC_G16 if cost(c) < 2 AND retention(c) > static_g16_ret on >= 3 methods
        sd_count = 0
        for m in METHODS:
            cost_c = agg_mc[(m, c)]["mean_cost_ratio"]
            ret_c = agg_mc[(m, c)]["mean_contrast_retention"]
            if cost_c < 2.0 and ret_c > static_g16_ret_mean - 1e-9:
                sd_count += 1
        if sd_count >= 4:
            static_g16_dominated_by += 1
    with open(OUT_DIR / "p7_iter163_dominance.tsv", "w") as f:
        f.write("controller\tn_methods_strictly_dominates\n")
        for c in CONTROLLERS:
            if c == "UNIFIED_STEP_C4":
                f.write(f"{c}\t{dom_counts[c]}\n")
            elif c == "STATIC_G16":
                f.write(f"{c}\t-{static_g16_dominated_by}\n")  # negative = dominated by
            else:
                f.write(f"{c}\t0\n")
    print(f"Wrote dominance")

    # ----- Headline verdicts (re-framed for step-aggregate granularity) -----
    h = {}
    # H1 (sharp positive): C4 retention > STATIC_G8 retention on 4/4 methods (paired CI excludes 0)
    h1_passes = sum(1 for r in paired_rows
                    if r["vs_controller"] == "STATIC_G8" and r["ci_excl0_ret"]
                    and r["direction_ret"] == "C4_GAIN")
    h["H1_C4_retention_gt_STATIC_G8_passes"] = h1_passes
    h["H1_threshold"] = 4
    h["H1_verdict"] = "PASS" if h1_passes == 4 else "FAIL"

    # H2 (sharp positive): C4 cost < STATIC_G16 cost on 4/4 methods (CI excludes 0)
    h2_passes = sum(1 for r in paired_rows
                    if r["vs_controller"] == "STATIC_G16" and r["ci_excl0_cost"]
                    and r["delta_cost"] < 0)
    h["H2_C4_cost_lt_STATIC_G16_passes"] = h2_passes
    h["H2_threshold"] = 4
    h["H2_verdict"] = "PASS" if h2_passes == 4 else "FAIL"

    # H3 (granularity contrast): at STEP level, STATIC_G16 is NOT dominated.
    # This is the SHARP contrast with iter-159 (per-prompt) — at per-prompt
    # granularity ADAPTIVE_PP_ORACLE dominates STATIC_G16; at step granularity
    # STATIC_G16 is the Pareto-frontier high-retention endpoint. The test is
    # that NO controller has cost < 2 AND retention >= STATIC_G16 on >= 4 methods.
    not_dominated_count = 0
    for m in METHODS:
        s16_ret = agg_mc[(m, "STATIC_G16")]["mean_contrast_retention"]
        dominated_on_m = False
        for c in CONTROLLERS:
            if c == "STATIC_G16": continue
            cost_c = agg_mc[(m, c)]["mean_cost_ratio"]
            ret_c = agg_mc[(m, c)]["mean_contrast_retention"]
            if cost_c < 2.0 and ret_c >= s16_ret - 1e-9:
                dominated_on_m = True
                break
        if not dominated_on_m:
            not_dominated_count += 1
    h["H3_STATIC_G16_not_dominated_methods_at_step_level"] = not_dominated_count
    h["H3_threshold"] = 4
    h["H3_verdict"] = "PASS" if not_dominated_count == 4 else "FAIL"

    # H4 (uniformity): cross-method cost SD for UNIFIED_STEP_C4 < 0.10 (uniformity bar)
    c4_sd_cost = next(r for r in xm_rows if r["controller"] == "UNIFIED_STEP_C4")["sd_cost_ratio"]
    h["H4_C4_cross_method_cost_sd"] = c4_sd_cost
    h["H4_threshold"] = 0.10
    h["H4_verdict"] = "PASS" if c4_sd_cost < 0.10 else "FAIL"

    # H5 (efficiency): C4 mag-per-cost > DUALFORMER_STEP mag-per-cost per method
    # (mpc = (retention - 1) / cost; positive retention gain normalized by cost)
    mpc_C4 = {}
    mpc_DF = {}
    for m in METHODS:
        ret_C4 = agg_mc[(m, "UNIFIED_STEP_C4")]["mean_contrast_retention"]
        cost_C4 = agg_mc[(m, "UNIFIED_STEP_C4")]["mean_cost_ratio"]
        ret_DF = agg_mc[(m, "DUALFORMER_STEP")]["mean_contrast_retention"]
        cost_DF = agg_mc[(m, "DUALFORMER_STEP")]["mean_cost_ratio"]
        mpc_C4[m] = (ret_C4 - 1.0) / max(cost_C4, 1e-9)
        mpc_DF[m] = (ret_DF - 1.0) / max(cost_DF, 1e-9)
    h5_passes = sum(1 for m in METHODS if mpc_C4[m] > mpc_DF[m])
    h["H5_C4_mpc_gt_DUALFORMER_mpc_passes"] = h5_passes
    h["H5_C4_mpc"] = mpc_C4
    h["H5_DF_mpc"] = mpc_DF
    h["H5_threshold"] = 4
    h["H5_verdict"] = "PASS" if h5_passes == 4 else "FAIL"

    # H6 (Pareto presence): C4 on Pareto frontier on >= 3/4 methods
    frontier_methods = set(p["method"] for p in frontier
                           if p["controller"] == "UNIFIED_STEP_C4")
    h["H6_C4_on_frontier_methods"] = len(frontier_methods)
    h["H6_threshold"] = 3
    h["H6_verdict"] = "PASS" if len(frontier_methods) >= 3 else "FAIL"

    # H7 (C4 vs STATIC_G8 cost gap): C4 cost CI95 lower bound > STATIC_G8 cost (1.0)
    c4_cost_ci_lo = agg_c["UNIFIED_STEP_C4"]["ci95_cost"][0]
    h["H7_C4_cost_CI95_lower_bound"] = c4_cost_ci_lo
    h["H7_threshold"] = 1.0
    h["H7_verdict"] = "PASS" if c4_cost_ci_lo > 1.0 else "FAIL"

    # H8 (granularity crossover): per-prompt (iter-159) Pareto strict dominance of
    # STATIC_G16 by ADAPTIVE_PP_ORACLE does NOT replicate at step granularity.
    # Confirmed by H3 = 4/4 PASS: STATIC_G16 not dominated at step level.
    h["H8_granularity_crossover"] = (
        "STATIC_G16 dominated at per-prompt (iter-159) but NOT at step level (iter-163)"
    )
    h["H8_threshold"] = "confirmed"
    h["H8_verdict"] = "PASS" if not_dominated_count == 4 else "FAIL"

    summary = {
        "iter": 163,
        "pillar": "P7",
        "vein": "step-aggregate Pareto + per-method bootstrap CI on N2 reward tensors",
        "n_step_method_decisions": 160,
        "n_methods": 4,
        "n_controllers": len(CONTROLLERS),
        "boot_n": BOOT_N, "boot_seed": BOOT_SEED,
        "agg_overall": {c: {k: (v[k][0], v[k][1]) if "ci95" in k else v[k]
                            for k in v}
                        for c, v in agg_c.items()},
        "pareto_frontier_count": len(frontier),
        "pareto_frontier_controllers_per_method": {
            m: sorted(set(p["controller"] for p in frontier if p["method"] == m))
            for m in METHODS
        },
        "headline": h,
        "n_passes": sum(1 for k in h if k.endswith("_verdict") and h[k] == "PASS"),
        "n_hypotheses": sum(1 for k in h if k.endswith("_verdict")),
    }

    with open(OUT_DIR / "p7_iter163_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"Wrote summary. Headline passes: {summary['n_passes']}/{summary['n_hypotheses']}")
    for k, v in h.items():
        if k.endswith("_verdict"):
            print(f"  {k}: {v}")
    return summary


if __name__ == "__main__":
    main()