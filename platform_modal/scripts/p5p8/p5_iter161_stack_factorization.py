"""
P5 stack-conditioning factorization (iter 161, vein brief-(b))

Two-tier decomposition:

(A) SAME-STACK (N2 four-method tensor panel): 4 methods (grpo, gift, aero, areal)
    × 40 steps × G=8 × seed 0. With stack pinned, measure eta^2(method) on
    per-step reward_mean — Ivison "algorithm axis" decomposition.

(B) STACK-AXIS (mega_20260704 cells.tsv): 98 cells spanning
    {model × task_slice × G × temperature × seed}. With algorithm fixed at
    "training-free" sampling, measure eta^2 of each stack axis on mean_reward.

Then the central claim is: the algorithm axis alone (N2) explains at most a
small fraction of total variance, while the stack axes (mega) collectively
explain the lion's share. That is the empirical case for "report the stack,
not the label".

Reuses: platform_modal/scripts/berkeley/unpacking_dpo_ppo_factorization.py helpers
        (axis_variance_fraction, eta^2 = SS_axis / SS_total).

Hypotheses (validated against the data):
  H1 N2 method-axis (same-stack)        : eta^2(method) <= 0.05
  H2 mega stack axes (model,task,G,T)   : eta^2_union(stack) >= 0.50
  H3 mega G axis                        : eta^2(G)        >= eta^2(method N2)
  H4 mega temperature axis              : eta^2(T)        >= eta^2(method N2)
  H5 mega model axis                    : eta^2(model)    >= eta^2(method N2)
  H6 mega seed axis                     : eta^2(seed)     <= 0.10  (i.e., stable)
  H7 N2 step axis (within-run drift)    : eta^2(step)     <= 0.10  (i.e., flat)

Outputs:
  platform_hybrid/experiments/results/p5p8/p5_iter161_stack_factorization.tsv
  platform_hybrid/experiments/results/p5p8/p5_iter161_stack_factorization.json
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from statistics import fmean, pstdev

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES  = os.path.join(ROOT, "experiments", "results")
OUT  = os.path.join(RES, "p5p8")
os.makedirs(OUT, exist_ok=True)

# ---- eta^2 helper (same logic as unpacking_dpo_ppo_factorization.py) ----

def axis_variance_fraction(rows, axis_key, value_key):
    """SS_axis / SS_total nested ANOVA-style decomposition.

    Returns (eta2, ss_axis, ss_within, n_groups, grand_mean).
    """
    grand = []
    by_axis = defaultdict(list)
    for r in rows:
        v = r.get(value_key)
        if v is None:
            continue
        grand.append(v)
        by_axis[r[axis_key]].append(v)
    if not grand or len(by_axis) < 2:
        return float("nan"), 0.0, 0.0, len(by_axis), float("nan")
    grand_mean = fmean(grand)
    ss_total = sum((x - grand_mean) ** 2 for x in grand)
    ss_axis = sum(len(vs) * (fmean(vs) - grand_mean) ** 2 for vs in by_axis.values())
    ss_within = ss_total - ss_axis
    eta2 = ss_axis / ss_total if ss_total > 1e-12 else float("nan")
    return eta2, ss_axis, ss_within, len(by_axis), grand_mean


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = fmean(a), fmean(b)
    sa, sb = pstdev(a), pstdev(b)
    sp = math.sqrt(((len(a) - 1) * sa * sa + (len(b) - 1) * sb * sb)
                   / (len(a) + len(b) - 2))
    return (ma - mb) / sp if sp > 1e-12 else float("nan")


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---- loaders ----

def load_n2():
    """Return list[dict] of per-(method, step) terminal reward stats from N2 panel.

    4 methods × 40 steps = 160 rows. Fields: method, step, reward_mean, zvf.
    """
    rows = []
    p = os.path.join(RES, "n2_reward_tensor_resume", "n2_metrics.tsv")
    with open(p) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            rows.append({
                "method":      r["method"],
                "step":        int(r["step"]),
                "reward_mean": float(r["reward_mean"]),
                "zvf":         float(r["zvf"]),
                "pcd":         float(r["pcd"]),
                "larq":        float(r["larq"]),
                "mean_len":    float(r["mean_len"]),
            })
    return rows


def load_mega():
    """Return list[dict] of cells with mean_reward as the response.

    98 cells spanning model × task_slice × G × temperature × seed.
    """
    rows = []
    p = os.path.join(RES, "mega_20260704", "cells.tsv")
    with open(p) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            try:
                mr = float(r["mean_reward"])
            except (ValueError, KeyError):
                continue
            rows.append({
                "cell_id":     r["cell_id"],
                "model":       r["model"],
                "task_slice":  r["task_slice"],
                "G":           int(r["G"]),
                "temperature": float(r["temperature"]),
                "seed":        int(r["seed"]),
                "mean_reward": mr,
                "zvf":         float(r["zvf"]) if r["zvf"] not in ("", "nan") else float("nan"),
                "pcd":         float(r["pcd"]) if r["pcd"] not in ("", "nan") else float("nan"),
                "n_groups":    int(r["n_groups"]) if r["n_groups"] not in ("", "nan") else 0,
            })
    return rows


def load_n2_method_terminal():
    """Per-(method, seed) terminal-5 mean reward_mean from N2 panel.

    For each method, average reward_mean over the last 5 steps (35..39).
    Returns dict[(method, seed)] -> float. Only seed 0 in N2.
    """
    rows = load_n2()
    terminal = defaultdict(list)
    for r in rows:
        if r["step"] >= 35:
            terminal[(r["method"], 0)].append(r["reward_mean"])
    return {k: fmean(v) for k, v in terminal.items()}


# ---- computations ----

def main():
    findings = []

    # ---- A: N2 same-stack (algorithm axis) ----
    n2_rows = load_n2()
    print(f"\n=== A: N2 same-stack algorithm-axis decomposition ===")
    print(f"  loaded {len(n2_rows)} (method, step) rows; "
          f"methods={sorted(set(r['method'] for r in n2_rows))}")
    eta2_method, ss_m_axis, ss_m_within, ng_m, gm_m = axis_variance_fraction(
        n2_rows, "method", "reward_mean"
    )
    eta2_step,  ss_s_axis, ss_s_within, ng_s, gm_s = axis_variance_fraction(
        n2_rows, "step",   "reward_mean"
    )
    eta2_zvf,  _, _, _, _ = axis_variance_fraction(
        n2_rows, "method", "zvf"
    )
    per_method = defaultdict(list)
    for r in n2_rows:
        per_method[r["method"]].append(r["reward_mean"])
    per_method_mean = {m: fmean(v) for m, vs in per_method.items() for _ in [None]
                       if False}
    # rebuild cleanly
    per_method_mean = {m: fmean(vs) for m, vs in per_method.items()}
    method_spread = (max(per_method_mean.values())
                     - min(per_method_mean.values()))
    print(f"  eta^2(method -> reward_mean) = {eta2_method:.4f}  "
          f"(SS_axis={ss_m_axis:.4f}, SS_within={ss_m_within:.4f}, n_groups={ng_m})")
    print(f"  eta^2(step   -> reward_mean) = {eta2_step:.4f}  "
          f"(SS_axis={ss_s_axis:.4f}, SS_within={ss_s_within:.4f}, n_groups={ng_s})")
    print(f"  per-method reward_mean:       { {m: round(v, 4) for m, v in per_method_mean.items()} }")
    print(f"  spread (max - min method mean) = {method_spread:.4f}")

    # Cohen's d across methods for sensitivity
    pair_d = {}
    methods = sorted(per_method.keys())
    for i, mi in enumerate(methods):
        for mj in methods[i + 1:]:
            d = cohens_d(per_method[mi], per_method[mj])
            pair_d[f"{mi}_vs_{mj}"] = d
    # pick the most-distant pair
    max_pair = max(pair_d.items(), key=lambda kv: abs(kv[1]))
    print(f"  max Cohen's d: {max_pair[0]} = {max_pair[1]:+.3f}")

    h1_decisive = (eta2_method <= 0.05)
    h1_verdict = "DECISIVE" if h1_decisive else "SUGGESTIVE" if eta2_method <= 0.10 else "NULL"
    h7_decisive = (eta2_step <= 0.10)
    h7_verdict = "DECISIVE" if h7_decisive else "SUGGESTIVE" if eta2_step <= 0.20 else "NULL"

    findings.append({
        "hypothesis": "H1: eta^2(method|stack) <= 0.05 on N2 same-stack panel",
        "n_rows": len(n2_rows),
        "n_groups": ng_m,
        "eta2": round(eta2_method, 4),
        "ss_axis": round(ss_m_axis, 4),
        "ss_within": round(ss_m_within, 4),
        "method_spread": round(method_spread, 4),
        "max_cohens_d_pair": max_pair[0],
        "max_cohens_d": round(max_pair[1], 3),
        "per_method_mean": {m: round(v, 4) for m, v in per_method_mean.items()},
        "verdict": h1_verdict,
    })

    findings.append({
        "hypothesis": "H7: eta^2(step -> reward_mean) <= 0.10 (within-run drift is flat)",
        "n_steps": ng_s,
        "eta2": round(eta2_step, 4),
        "verdict": h7_verdict,
    })

    # ---- B: mega stack-axis decomposition ----
    mega_rows = load_mega()
    print(f"\n=== B: mega stack-axis decomposition ===")
    print(f"  loaded {len(mega_rows)} cells")
    axes = ["model", "task_slice", "G", "temperature", "seed"]
    eta2_axes = {}
    for axis in axes:
        e2, ss_a, ss_w, ng, gm = axis_variance_fraction(mega_rows, axis, "mean_reward")
        eta2_axes[axis] = e2
        print(f"  eta^2({axis:<12} -> mean_reward) = {e2:.4f}  "
              f"(n_groups={ng}, grand_mean={gm:.4f})")

    # Union-stack eta^2: variance of per-(model, task, G, T) cell-mean over grand mean
    stack_key = lambda r: (r["model"], r["task_slice"], r["G"], r["temperature"])
    stack_cells = defaultdict(list)
    for r in mega_rows:
        stack_cells[stack_key(r)].append(r["mean_reward"])
    stack_means = [fmean(vs) for vs in stack_cells.values()]
    gm_all = fmean([r["mean_reward"] for r in mega_rows])
    ss_total_all = sum((r["mean_reward"] - gm_all) ** 2 for r in mega_rows)
    ss_stack = sum(len(vs) * (fmean(vs) - gm_all) ** 2 for vs in stack_cells.values())
    eta2_union = ss_stack / ss_total_all if ss_total_all > 1e-12 else float("nan")
    print(f"  eta^2((model,task,G,T) -> mean_reward) = {eta2_union:.4f}  "
          f"(n_stack_cells={len(stack_cells)})")

    # mega cell coverage: number of (model, task, G, T) stacks with BOTH seed 0 and seed 1
    seed_groups = defaultdict(set)
    for r in mega_rows:
        seed_groups[stack_key(r)].add(r["seed"])
    paired_stacks = sum(1 for s in seed_groups.values() if len(s) == 2)
    n_stacks = len(seed_groups)
    pct_paired = paired_stacks / n_stacks if n_stacks else 0.0
    pct_lo, pct_hi = wilson_ci(pct_paired, n_stacks)
    print(f"  stack coverage: {paired_stacks}/{n_stacks} stacks have BOTH seeds "
          f"({pct_paired:.4f}, Wilson [{pct_lo:.4f}, {pct_hi:.4f}])")

    # H2 union-stack decisive
    h2_decisive = (eta2_union >= 0.50)
    h2_verdict = "DECISIVE" if h2_decisive else "SUGGESTIVE" if eta2_union >= 0.30 else "NULL"
    findings.append({
        "hypothesis": "H2: eta^2_union(stack) >= 0.50 on mega cells",
        "n_stack_cells": len(stack_cells),
        "eta2_union": round(eta2_union, 4),
        "verdict": h2_verdict,
    })

    # H3..H6: per-axis comparisons against the N2 algorithm axis baseline
    for axis in axes:
        e2 = eta2_axes[axis]
        if axis == "seed":
            # H6: seed axis is SMALL (stable)
            h_decisive = (e2 <= 0.10)
            h_verdict = "DECISIVE" if h_decisive else "SUGGESTIVE" if e2 <= 0.20 else "NULL"
            tag = "H6"
        else:
            # H3..H5: stack axis dominates algorithm axis
            h_decisive = (e2 >= eta2_method)
            h_verdict = "DECISIVE" if h_decisive else "SUGGESTIVE" if e2 >= 0.5 * eta2_method else "NULL"
            tag = {"G": "H3", "temperature": "H4", "model": "H5", "task_slice": "H5b"}[axis]
        print(f"  [{tag}] eta^2({axis}) = {e2:.4f}  vs eta^2(N2 method) = {eta2_method:.4f}  → {h_verdict}")
        findings.append({
            "hypothesis": f"{tag}: eta^2({axis}) vs eta^2(method) on mega stack",
            "axis": axis,
            "eta2_axis": round(e2, 4),
            "eta2_method_baseline": round(eta2_method, 4),
            "ratio": round(e2 / eta2_method, 3) if eta2_method > 1e-12 else float("nan"),
            "verdict": h_verdict,
        })

    # ---- C: head-to-head summary ----
    # How many axes dominate algorithm?
    n_axes_dom = sum(1 for a in axes if a != "seed"
                     and eta2_axes[a] >= eta2_method)
    print(f"\n=== C: head-to-head summary ===")
    print(f"  eta^2(N2 method)              = {eta2_method:.4f}")
    print(f"  eta^2(mega G)                 = {eta2_axes['G']:.4f}")
    print(f"  eta^2(mega temperature)       = {eta2_axes['temperature']:.4f}")
    print(f"  eta^2(mega model)             = {eta2_axes['model']:.4f}")
    print(f"  eta^2(mega task_slice)        = {eta2_axes['task_slice']:.4f}")
    print(f"  eta^2(mega seed)              = {eta2_axes['seed']:.4f}")
    print(f"  eta^2_union(mega stack)       = {eta2_union:.4f}")
    print(f"  axes dominating algorithm     = {n_axes_dom}/{len(axes) - 1}")

    n_decisive = sum(1 for h in findings if h.get("verdict") == "DECISIVE")
    n_suggestive = sum(1 for h in findings if h.get("verdict") == "SUGGESTIVE")
    n_total = len(findings)

    # ---- save outputs ----
    summary = {
        "ts": "2026-07-05",
        "iteration": 161,
        "pillar": "P5",
        "vein": "(b) stack-conditioning factorization: N2 same-stack + mega cells",
        "framework": ("Ivison et al. 2024 (Unpacking DPO and PPO, NeurIPS 2024, "
                      "arXiv:2406.09279) 4-axis decomposition; for verifiable "
                      "RL stacks (data, prompts pinned), testableaxes = "
                      "algorithm + reward-intervention; here we extend to "
                      "stack axes (model, task, G, temperature, seed) at scale."),
        "data_sources": {
            "n2_panel": "platform_hybrid/experiments/results/n2_reward_tensor_resume/n2_metrics.tsv "
                        "(4 methods × 40 steps × 1 seed × G=8 = 160 rows)",
            "mega_cells": "platform_hybrid/experiments/results/mega_20260704/cells.tsv "
                          "(98 cells spanning 2 models × 3 tasks × 5 G × 2 T × 2 seeds)",
            "manifests": "platform_hybrid/experiments/results/mega_20260704/manifests/*.json (98)",
        },
        "hypotheses": findings,
        "headline": {
            "eta2_method_N2": round(eta2_method, 4),
            "eta2_step_N2": round(eta2_step, 4),
            "eta2_union_mega_stack": round(eta2_union, 4),
            "eta2_axes_mega": {a: round(e, 4) for a, e in eta2_axes.items()},
            "n_stack_axes_dominating_algorithm": n_axes_dom,
            "n_axes_minus_seed": len(axes) - 1,
            "paired_stack_coverage": {
                "n_paired": paired_stacks,
                "n_total_stacks": n_stacks,
                "fraction": round(pct_paired, 4),
                "wilson_lo": round(pct_lo, 4),
                "wilson_hi": round(pct_hi, 4),
            },
            "n_decisive": n_decisive,
            "n_suggestive": n_suggestive,
            "n_total": n_total,
        },
    }
    out_json = os.path.join(OUT, "p5_iter161_stack_factorization.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # TSV row per hypothesis
    out_tsv = os.path.join(OUT, "p5_iter161_stack_factorization.tsv")
    with open(out_tsv, "w") as f:
        f.write("hypothesis\taxis\tn_groups\tn_rows\teta2\teta2_method_baseline\t"
                "ratio\tverdict\n")
        for h in findings:
            f.write("\t".join([
                h.get("hypothesis", ""),
                h.get("axis", ""),
                str(h.get("n_groups", h.get("n_steps", h.get("n_stack_cells", "")))),
                str(h.get("n_rows", h.get("n_paired", ""))),
                str(h.get("eta2", h.get("eta2_axis", h.get("eta2_union", "")))),
                str(h.get("eta2_method_baseline", "")),
                str(h.get("ratio", "")),
                h.get("verdict", ""),
            ]) + "\n")

    print(f"\n=== HEADLINE: {n_decisive}/{n_total} hypotheses DECISIVE "
          f"(+{n_suggestive} suggestive) ===")
    print(f"  outputs:")
    print(f"    {out_json}")
    print(f"    {out_tsv}")


if __name__ == "__main__":
    main()