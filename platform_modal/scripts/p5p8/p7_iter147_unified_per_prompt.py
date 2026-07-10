#!/usr/bin/env python3
"""
p7_iter147_unified_per_prompt.py
=================================
Iter 147 — Counterfactual evaluation of the UNIFIED C4 controller
(regime-gated composition of Dualformer-auto-G + ZVF-triage + gamma*=0)
at PER-PROMPT granularity on the REAL N2 reward tensors.

Extends iter131 (per-prompt adaptive-G* family on N2) by replacing the
per-prompt-only family with the iter119 C4 unified controller evaluated
per-prompt, and adds bootstrap CIs on every headline metric.

Inputs:  platform_hybrid/experiments/results/n2_reward_tensor_resume/{grpo,aero,gift,areal}_s0_tensors.jsonl
Outputs: platform_hybrid/experiments/results/p5p8/p7_iter147_*.{tsv,json}

Pillar: P7 (Pillar 3 — adaptive-G controller / signal-starvation theory)
"""
from __future__ import annotations
import json, math, os, sys, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N2_DIR = ROOT / "experiments" / "results" / "n2_reward_tensor_resume"
OUT_DIR = ROOT / "experiments" / "results" / "p5p8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["grpo", "aero", "gift", "areal"]
G_BASE = 8
G_CANDIDATES = [4, 8, 16, 32]
TAU_DEGEN = 0.70  # DEGENERATE threshold (iter119)
GAMMA_STAR = 0.0   # AlphaProof gamma*=0 (iter119)
RHO_SAT = 0.85     # Dualformer saturation threshold (iter119)
SEED = 0


def bernoulli_z(p_hat: float, G: int) -> float:
    """Closed-form Bernoulli zero-variance fraction: p^G + (1-p)^G."""
    if p_hat <= 0.0:
        return 1.0
    if p_hat >= 1.0:
        return 1.0
    return p_hat ** G + (1.0 - p_hat) ** G


def contrast_mag(p_hat: float, G: int) -> float:
    """1 - z(p, G) — within-group contrast magnitude."""
    return 1.0 - bernoulli_z(p_hat, G)


def is_boundary(p_hat: float) -> bool:
    """Boundary prompt (k=0 or k=G): no contrast possible at any G."""
    return p_hat <= 0.0 or p_hat >= 1.0


# ---------------------------------------------------------------------------
# Five controllers (per-prompt granularity on real N2 tensors)
# ---------------------------------------------------------------------------
def c_static_g8(p_hat, z_obs):
    return G_BASE

def c_static_g16(p_hat, z_obs):
    return 16

def c_dualformer_pp(p_hat, z_obs):
    """Berkeley row 01 Dualformer auto-G rule: difficulty-gated G.
       Escalate when contrast is low, drop when saturated, base otherwise."""
    if z_obs < 0.50:
        # fast regime — task learned, drop G
        return 2 if is_boundary(p_hat) else 4
    if z_obs >= TAU_DEGEN:
        return min(G_BASE, 8)  # DEGENERATE: drop to base
    return G_BASE  # baseline regime

def c_adaptive_pp_oracle(p_hat, z_obs):
    """Per-prompt oracle: pick G that minimises z(p_hat, G)."""
    if is_boundary(p_hat):
        return G_BASE
    best_g, best_z = G_BASE, bernoulli_z(p_hat, G_BASE)
    for g in G_CANDIDATES + [16, 32]:
        z = bernoulli_z(p_hat, g)
        if z < best_z - 1e-9:
            best_z, best_g = z, g
    return best_g

def c_unified_c4(p_hat, z_obs):
    """Iter119 C4 unified controller (regime-gated composition).
       Per-prompt application of:
         - Dualformer auto-G (fast regime)
         - Adaptive-G*-Bernoulli (degenerate regime, capped at G=32)
         - gamma*=0 baseline tightening (no G change, but tracked)
    """
    if z_obs < 0.50:
        # FAST regime: drop G (Dualformer)
        return 2 if is_boundary(p_hat) else 4
    if z_obs >= TAU_DEGEN:
        # DEGENERATE regime: escalate via Bernoulli inversion, cap G=32
        if is_boundary(p_hat):
            return G_BASE
        target_z = max(0.5, 0.5 * z_obs)
        best_g = G_BASE
        for g in [16, 32]:
            if bernoulli_z(p_hat, g) < target_z:
                best_g = g
                break
        return best_g
    # BASELINE regime
    return G_BASE

CONTROLLERS = {
    "STATIC_G8": c_static_g8,
    "STATIC_G16": c_static_g16,
    "DUALFORMER_PP": c_dualformer_pp,
    "ADAPTIVE_PP_ORACLE": c_adaptive_pp_oracle,
    "UNIFIED_C4": c_unified_c4,
}


# ---------------------------------------------------------------------------
# Load real N2 tensors
# ---------------------------------------------------------------------------
def load_tensors():
    """Returns list of step-records: dict[method] -> list of {step, prompt_rewards[...]}.
       Each step has 16 prompts; each prompt has 8 rewards."""
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


def per_prompt_k(rewards_row):
    """k_p = number of correct (1.0) rewards in a single prompt's group."""
    return int(round(sum(rewards_row)))


def evaluate_cell(method, step, prompt_idx, rewards_row, z_obs):
    """One (method, step, prompt) cell. Returns dict[controller] -> metrics."""
    p_hat = per_prompt_k(rewards_row) / G_BASE
    # baseline per-prompt contrast magnitude (at G=8)
    cm_base = contrast_mag(p_hat, G_BASE)
    out = {
        "method": method, "step": step, "prompt_idx": prompt_idx,
        "k_p": per_prompt_k(rewards_row), "p_hat": p_hat,
        "z_obs": z_obs, "cm_base": cm_base,
        "is_boundary": is_boundary(p_hat),
    }
    for cname, cfn in CONTROLLERS.items():
        g_used = cfn(p_hat, z_obs)
        cm_used = contrast_mag(p_hat, g_used)
        cost_ratio = g_used / G_BASE
        out[f"g_{cname}"] = g_used
        out[f"cm_{cname}"] = cm_used
        out[f"cost_{cname}"] = cost_ratio
    return out


def aggregate(cells, n_boot=1000):
    """Aggregate metrics across cells with bootstrap CIs (resample cells with replacement)."""
    import random
    random.seed(42)
    n = len(cells)
    summary = {}
    for cname in CONTROLLERS:
        costs = [c[f"cost_{cname}"] for c in cells]
        cms = [c[f"cm_{cname}"] for c in cells]
        # mean metrics
        mean_cost = statistics.mean(costs)
        mean_cm = statistics.mean(cms)
        # contrast-retention = mean(cm_used) / mean(cm_base)
        contrast_retention = mean_cm / max(statistics.mean([c["cm_base"] for c in cells]), 1e-9)
        # bootstrap CIs
        boot_costs = []
        boot_cms = []
        boot_ret = []
        for _ in range(n_boot):
            sample = [cells[random.randrange(n)] for _ in range(n)]
            sc = statistics.mean(x[f"cost_{cname}"] for x in sample)
            sm = statistics.mean(x[f"cm_{cname}"] for x in sample)
            sr = sm / max(statistics.mean(x["cm_base"] for x in sample), 1e-9)
            boot_costs.append(sc)
            boot_cms.append(sm)
            boot_ret.append(sr)
        boot_costs.sort()
        boot_cms.sort()
        boot_ret.sort()
        ci_cost = (boot_costs[25], boot_costs[975])
        ci_cm = (boot_cms[25], boot_cms[975])
        ci_ret = (boot_ret[25], boot_ret[975])
        summary[cname] = {
            "n_cells": n,
            "mean_cost_ratio": mean_cost,
            "mean_contrast_mag": mean_cm,
            "contrast_retention": contrast_retention,
            "ci95_cost": ci_cost,
            "ci95_contrast_mag": ci_cm,
            "ci95_retention": ci_ret,
            # magnitude / cost ratio (efficiency)
            "mag_per_cost": mean_cm / mean_cost,
        }
    return summary


def main():
    by_method = load_tensors()
    all_cells = []
    for m, steps in by_method.items():
        for srec in steps:
            step =srec["step"]
            z_obs = srec["zvf"]
            for p_idx, rewards_row in enumerate(srec["rewards"]):
                cell = evaluate_cell(m, step, p_idx, rewards_row, z_obs)
                all_cells.append(cell)
    print(f"Loaded {len(all_cells)} (method, step, prompt) cells")

    # Aggregate overall
    overall = aggregate(all_cells)
    # Per-method
    by_m = {}
    for m in METHODS:
        m_cells = [c for c in all_cells if c["method"] == m]
        by_m[m] = aggregate(m_cells)

    # ---- Write TSV: per-controller headline row ----
    headline_path = OUT_DIR / "p7_iter147_headline.tsv"
    with open(headline_path, "w") as f:
        f.write("controller\tn_cells\tmean_cost\tci95_cost_lo\tci95_cost_hi\t"
                "contrast_retention\tci95_ret_lo\tci95_ret_hi\tmag_per_cost\n")
        for cname, s in overall.items():
            f.write(f"{cname}\t{s['n_cells']}\t"
                    f"{s['mean_cost_ratio']:.4f}\t{s['ci95_cost'][0]:.4f}\t{s['ci95_cost'][1]:.4f}\t"
                    f"{s['contrast_retention']:.4f}\t{s['ci95_retention'][0]:.4f}\t{s['ci95_retention'][1]:.4f}\t"
                    f"{s['mag_per_cost']:.4f}\n")
    print(f"Wrote {headline_path}")

    # ---- Write TSV: per-method × per-controller ----
    permethod_path = OUT_DIR / "p7_iter147_per_method.tsv"
    with open(permethod_path, "w") as f:
        f.write("method\tcontroller\tmean_cost\tci95_cost_lo\tci95_cost_hi\t"
                "contrast_retention\tci95_ret_lo\tci95_ret_hi\tmean_cm\tmag_per_cost\n")
        for m in METHODS:
            for cname, s in by_m[m].items():
                f.write(f"{m}\t{cname}\t"
                        f"{s['mean_cost_ratio']:.4f}\t{s['ci95_cost'][0]:.4f}\t{s['ci95_cost'][1]:.4f}\t"
                        f"{s['contrast_retention']:.4f}\t{s['ci95_retention'][0]:.4f}\t{s['ci95_retention'][1]:.4f}\t"
                        f"{s['mean_contrast_mag']:.4f}\t{s['mag_per_cost']:.4f}\n")
    print(f"Wrote {permethod_path}")

    # ---- Write TSV: per-cell (2560 rows) ----
    percell_path = OUT_DIR / "p7_iter147_per_cell.tsv"
    with open(percell_path, "w") as f:
        header = ["method", "step", "prompt_idx", "k_p", "p_hat", "z_obs",
                  "is_boundary", "cm_base"]
        for cname in CONTROLLERS:
            header += [f"g_{cname}", f"cm_{cname}", f"cost_{cname}"]
        f.write("\t".join(header) + "\n")
        for c in all_cells:
            row = [str(c[h]) for h in header if h not in ("is_boundary",)]
            row.append("1" if c["is_boundary"] else "0")
            f.write("\t".join(row) + "\n")
    print(f"Wrote {percell_path}")

    # ---- Write JSON summary ----
    summary = {
        "iter": 147,
        "pillar": "P7",
        "vein": "(b) Counterfactual C4 unified controller at per-prompt granularity on N2 (extends iter131 family + iter119 C4 unification)",
        "n_cells": len(all_cells),
        "methods": METHODS,
        "controllers": list(CONTROLLERS.keys()),
        "G_base": G_BASE,
        "G_candidates": G_CANDIDATES,
        "tau_degen": TAU_DEGEN,
        "gamma_star": GAMMA_STAR,
        "rho_sat": RHO_SAT,
        "headline_overall": overall,
        "headline_by_method": by_m,
        "headline_falsifiable_claims": [],
    }

    # ---- Falsifiable claims ----
    claims = []
    # H1: UNIFIED_C4 dominates STATIC_G8 on contrast retention
    s8 = overall["STATIC_G8"]
    c4 = overall["UNIFIED_C4"]
    h1 = {
        "id": "H1",
        "claim": "UNIFIED_C4 dominates STATIC_G8 on contrast retention (C4 >= G8)",
        "c4_retention": c4["contrast_retention"],
        "g8_retention": s8["contrast_retention"],
        "c4_ci95": c4["ci95_retention"],
        "g8_ci95": s8["ci95_retention"],
        "verdict": "PASS" if c4["contrast_retention"] >= s8["contrast_retention"] - 0.001 else "FAIL",
    }
    claims.append(h1)
    # H2: UNIFIED_C4 mean cost < STATIC_G16 cost (=2.0)
    s16 = overall["STATIC_G16"]
    h2 = {
        "id": "H2",
        "claim": "UNIFIED_C4 mean cost < STATIC_G16 (=2.0)",
        "c4_cost": c4["mean_cost_ratio"],
        "s16_cost": s16["mean_cost_ratio"],
        "c4_ci95": c4["ci95_cost"],
        "verdict": "PASS" if c4["mean_cost_ratio"] < 1.99 else "FAIL",
    }
    claims.append(h2)
    # H3: UNIFIED_C4 mag-per-cost > STATIC_G8 mag-per-cost (efficiency gain)
    h3 = {
        "id": "H3",
        "claim": "UNIFIED_C4 mag-per-cost > STATIC_G8 mag-per-cost (efficiency gain)",
        "c4_mpc": c4["mag_per_cost"],
        "g8_mpc": s8["mag_per_cost"],
        "verdict": "PASS" if c4["mag_per_cost"] > s8["mag_per_cost"] else "FAIL",
    }
    claims.append(h3)
    # H4: UNIFIED_C4 cost CI excludes 1.0 (statistically distinguishable from G8 baseline)
    c4_ci = c4["ci95_cost"]
    h4 = {
        "id": "H4",
        "claim": "UNIFIED_C4 cost CI95 excludes 1.0 (statistically differs from STATIC_G8)",
        "c4_ci95": list(c4_ci),
        "excludes_unity": c4_ci[0] > 1.0 or c4_ci[1] < 1.0,
        "verdict": "PASS" if (c4_ci[0] > 1.0 or c4_ci[1] < 1.0) else "FAIL",
    }
    claims.append(h4)
    # H5: UNIFIED_C4 Pareto-better than ADAPTIVE_PP_ORACLE on at least one cell (oracle strict-dominance fail expected)
    n_oracle_strict = 0
    n_c4_strict = 0
    n_tie = 0
    for c in all_cells:
        co, cc = c["cost_ADAPTIVE_PP_ORACLE"], c["cost_UNIFIED_C4"]
        mo, mc = c["cm_ADAPTIVE_PP_ORACLE"], c["cm_UNIFIED_C4"]
        if co < cc and mo >= mc - 1e-9:
            n_oracle_strict += 1
        elif cc < co and mc >= mo - 1e-9:
            n_c4_strict += 1
        else:
            n_tie += 1
    h5 = {
        "id": "H5",
        "claim": "UNIFIED_C4 never strictly dominates and is never strictly dominated by ADAPTIVE_PP_ORACLE on any cell (defensive composition)",
        "n_oracle_strictly_better": n_oracle_strict,
        "n_c4_strictly_better": n_c4_strict,
        "n_tie": n_tie,
        "n_total": len(all_cells),
        "verdict": "PASS" if (n_oracle_strict == 0 and n_c4_strict == 0) else "FAIL",
    }
    claims.append(h5)
    summary["headline_falsifiable_claims"] = claims
    json_path = OUT_DIR / "p7_iter147_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {json_path}")
    print()
    print("=== HEADLINE (overall, n=2560 cells) ===")
    for cname, s in overall.items():
        print(f"  {cname:24s} cost={s['mean_cost_ratio']:.4f} ret={s['contrast_retention']:.4f} "
              f"mag_per_cost={s['mag_per_cost']:.4f}")
    print()
    print("=== FALSIFIABLE CLAIMS ===")
    for c in claims:
        print(f"  {c['id']}: {c['verdict']} — {c['claim']}")


if __name__ == "__main__":
    main()
