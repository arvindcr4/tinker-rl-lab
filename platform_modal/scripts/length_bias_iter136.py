#!/usr/bin/env python3
"""Iter 136 -- Pillar 4 (Length Bias / Dr.GRPO):
STEP-LEVEL TRAJECTORY COUPLING between reward and length.

Prior pillars (108, 112, 116, 120, 124, 128, 132) measured the GR-vs-DR
gap at progressively finer units:
  iter108: 10-step WINDOW level backward/forward CCF (|bwd|_GR vs DR)
  iter120: pooled Spearman(severship, |bwd_CCF|^GR) = +0.556, p<0.001
  iter128: RUN-level Pareto frontier: Dr.GR Pareto-dominates GR on
           (length-contraction, reward-gain) -- Cohen's d=+3.62 arith_easy
  iter132: causal chain: severship -> |bwd_CCF| shrinkage -> efficiency gain.

What is STILL missing is the STEP-LEVEL co-movement signature.  Iter 108
worked at windows of 10 steps; iter 132 worked at windows of n_w=4.  Neither
asked whether GR's per-step DELTAS (Δreward_t, Δlength_t) are coupled to
each other (a moving treadmill) while DR's are decoupled (independent axes).

This is the most granular test of the iter-128/132 orthogonality claim.

Falsifiable hypotheses (computed per run on step_log):
  H1 -- step-velocity Spearman coupling:
       rho(Δreward_t, Δlength_t) over t=1..T-1 deltas.
       Pre-registered sign: GR |rho| > DR |rho|  (i.e. delta-decoupled).

  H2 -- length-trajectory trendiness:
       lag-1 Spearman autocorrelation of mean_comp_len_t.
       Pre-registered sign: GR |rho| > DR |rho|  (GR trending, DR noisy).

  H3 -- late-training efficiency:
       eff = (mean_reward_last10 - mean_reward_first5) /
              (|mean_comp_len_last10 - mean_comp_len_first5| + 1)
       Pre-registered sign: DR eff > GR eff  (per-token gain).

  H4 -- ZVF-length co-movement:
       rho(Δzvf_t, Δlength_t) over deltas.
       Pre-registered sign: GR rho < 0 (zvf-fall couples with length-shrink),
                              DR rho ≈ 0 (decoupled).

For each (task, seed), the four statistics are paired across (GR, DR).  We
report paired Wilcoxon one-sided p, Cohen's d, permutation p (B=50000), and
binomial over the 2 tasks.

INPUTS :
  experiments/results/drgrpo_vs_grpo.json            (arithmetic_easy, 5 seeds)
  experiments/results/drgrpo_gsm8k_cot_full.json     (gsm8k_cot,        3 seeds)
OUTPUTS (5 TSV + meta):
  length_bias_iter136_step_coupling.tsv    -- per-run (algo,task,seed,h,N)
  length_bias_iter136_paired_tests.tsv     -- paired (task, hypothesis)
  length_bias_iter136_permutation_null.tsv -- permutation p per test
  length_bias_iter136_cross_task.tsv       -- binomial summary
  length_bias_iter136_summary.tsv          -- headline numbers
  length_bias_iter136_meta.json

USAGE : python3 platform_modal/scripts/length_bias_iter136.py [--B_perm 50000 --seed_base 20260704]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results")

DRGR_VS_GRPO = os.path.join(RES, "drgrpo_vs_grpo.json")
DRGR_GSM8K = os.path.join(RES, "drgrpo_gsm8k_cot_full.json")


def load_runs() -> list[dict[str, Any]]:
    """Return a flat list of runs with (task, algo, seed, step_log).

    Each step_log entry has fields {step, mean_reward, zvf, mean_comp_len}.
    """
    out: list[dict[str, Any]] = []
    d_arith = json.load(open(DRGR_VS_GRPO))
    for r in d_arith["runs"]:
        out.append({
            "task": "arithmetic_easy",
            "algo": r["algo"],
            "seed": r["seed"],
            "step_log": r["step_log"],
            "model": r["model"],
        })
    d_gsm = json.load(open(DRGR_GSM8K))
    for r in d_gsm["runs"]:
        out.append({
            "task": "gsm8k_cot",
            "algo": r["algo"],
            "seed": r["seed"],
            "step_log": r["step_log"],
            "model": r["model"],
        })
    return out


# ---------------------------------------------------------------------------
# Per-run statistics
# ---------------------------------------------------------------------------

def step_coupling_stats(step_log: list[dict[str, Any]]) -> dict[str, float]:
    """Compute the four H1-H4 statistics from a single run's step_log.

    Returns dict with keys: rho_dR_dL, abs_rho_dR_dL, rho_len_lag1, abs_rho_len_lag1,
                            late_eff, rho_dZ_dL, n_steps.
    """
    r = np.array([s["mean_reward"] for s in step_log], dtype=float)
    L = np.array([s["mean_comp_len"] for s in step_log], dtype=float)
    Z = np.array([s["zvf"] for s in step_log], dtype=float)
    T = len(r)
    d_r = np.diff(r)
    d_L = np.diff(L)
    d_Z = np.diff(Z)

    # H1: rho(Δreward_t, Δlength_t) over deltas
    rho_dR_dL = float(stats.spearmanr(d_r, d_L).correlation)
    abs_rho_dR_dL = abs(rho_dR_dL)

    # H2: lag-1 Spearman autocorrelation of mean_comp_len
    rho_len_lag1 = float(stats.spearmanr(L[:-1], L[1:]).correlation)
    abs_rho_len_lag1 = abs(rho_len_lag1)

    # H3: late-training efficiency = Δreward_last10 / |Δlength_last10|
    n_last = min(10, T // 2)
    n_first = min(5, T // 4)
    if n_first < 2 or n_last < 2:
        late_eff = float("nan")
    else:
        mean_r_first = float(r[:n_first].mean())
        mean_r_last = float(r[-n_last:].mean())
        mean_L_first = float(L[:n_first].mean())
        mean_L_last = float(L[-n_last:].mean())
        late_eff = (mean_r_last - mean_r_first) / (abs(mean_L_last - mean_L_first) + 1.0)

    # H4: rho(Δzvf_t, Δlength_t)
    rho_dZ_dL = float(stats.spearmanr(d_Z, d_L).correlation)

    return {
        "rho_dR_dL": rho_dR_dL,
        "abs_rho_dR_dL": abs_rho_dR_dL,
        "rho_len_lag1": rho_len_lag1,
        "abs_rho_len_lag1": abs_rho_len_lag1,
        "late_eff": late_eff,
        "rho_dZ_dL": rho_dZ_dL,
        "n_steps": int(T),
    }


# ---------------------------------------------------------------------------
# Paired test machinery
# ---------------------------------------------------------------------------

def paired_one_sided(gr_vals: np.ndarray, dr_vals: np.ndarray, direction: str):
    """Return (mean_delta, W, p_one_sided_param, p_one_sided_perm_placeholder, n_pairs).

    direction = "GR>DR"  -> test that GR > DR (one-sided alternative)
    direction = "DR>GR"  -> test that DR > GR
    """
    n = len(gr_vals)
    assert len(dr_vals) == n
    delta = dr_vals - gr_vals  # DR - GR
    # Paired Wilcoxon: W is the rank-sum of positive deltas
    abs_d = np.abs(delta)
    ranks = stats.rankdata(abs_d)
    W_pos = float(ranks[delta > 0].sum())
    # One-sided p-value via scipy's wilcoxon with alternative
    try:
        if direction == "GR>DR":
            alt = "greater"
            # For GR>DR test, use -delta so that "greater" tests positive
            w_res = stats.wilcoxon(-delta, alternative="greater", zero_method="wilcox")
        else:  # DR>GR
            alt = "greater"
            w_res = stats.wilcoxon(delta, alternative="greater", zero_method="wilcox")
        p_param = float(w_res.pvalue)
    except ValueError:
        p_param = float("nan")
    mean_delta = float(delta.mean())
    return mean_delta, W_pos, p_param, n


def permutation_paired(gr_vals: np.ndarray, dr_vals: np.ndarray, direction: str,
                       rng: np.random.Generator, B: int) -> float:
    """Two-sided permutation p: fraction of |mean(Δ*)| >= |mean(Δ_obs)| under sign-flip null."""
    delta = dr_vals - gr_vals
    obs = abs(delta.mean())
    signs = rng.choice([-1.0, 1.0], size=(B, len(delta)))
    null = np.abs((signs * delta).mean(axis=1))
    return float((null >= obs).mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B_perm", type=int, default=50000)
    ap.add_argument("--seed_base", type=int, default=20260704)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed_base)

    runs = load_runs()
    # Per-run statistics
    per_run_rows: list[dict[str, Any]] = []
    by_run: dict[tuple[str, int], dict[str, dict[str, float]]] = {}
    for r in runs:
        st = step_coupling_stats(r["step_log"])
        row = {
            "task": r["task"],
            "algo": r["algo"],
            "seed": r["seed"],
            "model": r["model"],
            "n_steps": st["n_steps"],
            "rho_dR_dL": f"{st['rho_dR_dL']:.4f}",
            "abs_rho_dR_dL": f"{st['abs_rho_dR_dL']:.4f}",
            "rho_len_lag1": f"{st['rho_len_lag1']:.4f}",
            "abs_rho_len_lag1": f"{st['abs_rho_len_lag1']:.4f}",
            "late_eff": f"{st['late_eff']:.4f}",
            "rho_dZ_dL": f"{st['rho_dZ_dL']:.4f}",
        }
        per_run_rows.append(row)
        by_run.setdefault((r["task"], r["seed"]), {})[r["algo"]] = st

    # Paired tests: for each (task, hypothesis) compute paired diffs across seeds
    hypotheses = [
        # (h_name, gr_key, dr_key, direction, pre_reg_direction_text)
        ("H1_abs_rho_dR_dL_DR_smaller", "abs_rho_dR_dL", "abs_rho_dR_dL", "DR_smaller",
         "GR>DR: GR tightly coupled velocity, DR decoupled"),
        ("H2_abs_rho_len_lag1_DR_smaller", "abs_rho_len_lag1", "abs_rho_len_lag1", "DR_smaller",
         "GR>DR: GR trending length, DR no-trend"),
        ("H3_late_eff_DR_larger", "late_eff", "late_eff", "DR_larger",
         "DR>GR: DR more reward-gain per unit length change"),
        ("H4_rho_dZ_dL_DR_lessneg", "rho_dZ_dL", "rho_dZ_dL", "DR_lessneg",
         "GR<0<DR: GR negative ZVF-length coupling, DR decoupled toward zero"),
    ]

    paired_rows: list[dict[str, Any]] = []
    perm_null_rows: list[dict[str, Any]] = []

    for task in sorted({k[0] for k in by_run.keys()}):
        seeds = sorted({k[1] for k in by_run.keys() if k[0] == task})
        for h_name, gr_k, dr_k, direction, pre_reg in hypotheses:
            gr_vals = np.array([by_run[(task, s)]["grpo"][gr_k] for s in seeds], dtype=float)
            dr_vals = np.array([by_run[(task, s)]["dr_grpo"][dr_k] for s in seeds], dtype=float)
            n_pairs = len(seeds)
            if direction == "DR_smaller":
                mean_delta, W, p_param, n = paired_one_sided(gr_vals, dr_vals, "GR>DR")
                # For permutation: fraction of |mean(delta*)| >= |mean(delta_obs)|
                p_perm = permutation_paired(gr_vals, dr_vals, "GR>DR", rng, args.B_perm)
            elif direction == "DR_larger":
                mean_delta, W, p_param, n = paired_one_sided(gr_vals, dr_vals, "DR>GR")
                p_perm = permutation_paired(gr_vals, dr_vals, "DR>GR", rng, args.B_perm)
            elif direction == "DR_lessneg":
                # Test DR > GR (i.e. delta > 0 means DR less negative / more positive)
                mean_delta, W, p_param, n = paired_one_sided(gr_vals, dr_vals, "DR>GR")
                p_perm = permutation_paired(gr_vals, dr_vals, "DR>GR", rng, args.B_perm)
            else:
                raise ValueError(f"unknown direction {direction}")
            # Cohen's d paired (delta)
            d = mean_delta / (np.std(dr_vals - gr_vals, ddof=1) + 1e-9)
            paired_rows.append({
                "task": task,
                "hypothesis": h_name,
                "pre_reg": pre_reg,
                "direction": direction,
                "n_pairs": n_pairs,
                "mean_gr": f"{float(gr_vals.mean()):.4f}",
                "mean_dr": f"{float(dr_vals.mean()):.4f}",
                "mean_delta_dr_minus_gr": f"{mean_delta:.4f}",
                "W": int(W),
                "p_param_one_sided": f"{p_param:.4f}",
                "p_perm_two_sided": f"{p_perm:.4f}",
                "cohens_d_paired": f"{d:.3f}",
                "verdict": "FAVOURS" if p_param < 0.05 else ("marginal" if p_param < 0.10 else "null"),
            })
            perm_null_rows.append({
                "task": task,
                "hypothesis": h_name,
                "B_perm": args.B_perm,
                "p_perm_two_sided": f"{p_perm:.4f}",
                "p_param_one_sided": f"{p_param:.4f}",
                "n_pairs": n_pairs,
            })

    # Cross-task binomial: for each hypothesis, count FAVOURS across 2 tasks
    cross_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    h_names_unique = sorted({r["hypothesis"] for r in paired_rows})
    for h_name in h_names_unique:
        rows = [r for r in paired_rows if r["hypothesis"] == h_name]
        n_fav = sum(1 for r in rows if r["verdict"] == "FAVOURS")
        n_marg = sum(1 for r in rows if r["verdict"] == "marginal")
        n_null = sum(1 for r in rows if r["verdict"] == "null")
        # Binomial: at least 2/2 favours -> p = 0.5^2 = 0.25 (one-sided)
        binom_p_two = 0.5 ** 2 if n_fav == 2 else (2 * 0.5 ** 2 if n_fav == 1 else 1.0)
        binom_p_two_sided = binom_p_two  # symmetric
        # Direction-consistency: do all favoured tasks have the SAME delta sign?
        deltas = []
        for r in rows:
            if r["verdict"] in ("FAVOURS", "marginal"):
                deltas.append(float(r["mean_delta_dr_minus_gr"]))
        consistent = (len(deltas) > 0 and all(d > 0 for d in deltas)) or \
                     (len(deltas) > 0 and all(d < 0 for d in deltas))
        # Sign test on direction alone (regardless of significance)
        n_dir_pos = sum(1 for r in rows if float(r["mean_delta_dr_minus_gr"]) > 0)
        n_dir_neg = sum(1 for r in rows if float(r["mean_delta_dr_minus_gr"]) < 0)
        sign_test_p_one = 0.5 ** max(n_dir_pos, n_dir_neg)
        cross_rows.append({
            "hypothesis": h_name,
            "n_tasks": len(rows),
            "n_favours": n_fav,
            "n_marginal": n_marg,
            "n_null": n_null,
            "n_dir_positive": n_dir_pos,
            "n_dir_negative": n_dir_neg,
            "direction_consistent": "yes" if consistent else "no",
            "binom_p_two_sided_strict": f"{binom_p_two_sided:.4f}",
            "binom_p_one_sided_signtest": f"{sign_test_p_one:.4f}",
        })
        summary_rows.append({
            "hypothesis": h_name,
            "tasks_favoured": ",".join(r["task"] for r in rows if r["verdict"] == "FAVOURS"),
            "tasks_marginal": ",".join(r["task"] for r in rows if r["verdict"] == "marginal"),
            "n_favours": n_fav,
            "n_dir_pos": n_dir_pos,
            "binom_p_two_sided_strict": f"{binom_p_two_sided:.4f}",
            "binom_p_one_sided_signtest": f"{sign_test_p_one:.4f}",
            "headline": ("REJECT_NULL" if n_fav == 2 else
                         ("DIRECTIONAL" if (n_dir_pos == 2 or n_dir_neg == 2) else "INCONCLUSIVE")),
        })

    # Global sign test: aggregate over ALL 8 paired tests (4 hypotheses x 2 tasks)
    # For each test, encode "predicted direction" per the hypothesis definition:
    #   DR_smaller  -> predicted Δ = DR - GR < 0
    #   DR_larger   -> predicted Δ = DR - GR > 0
    #   DR_lessneg  -> predicted Δ = DR - GR > 0
    def predicted_direction_positive(direction: str) -> bool:
        return direction in ("DR_larger", "DR_lessneg")
    n_in_pred_dir = 0
    n_against = 0
    for r in paired_rows:
        delta = float(r["mean_delta_dr_minus_gr"])
        pred_pos = predicted_direction_positive(r["direction"])
        if (pred_pos and delta > 0) or (not pred_pos and delta < 0):
            n_in_pred_dir += 1
        else:
            n_against += 1
    n_global = len(paired_rows)
    global_sign_p = 0.5 ** n_in_pred_dir  # one-sided under uniform-null
    cross_rows.append({
        "hypothesis": "GLOBAL_sign_test_over_all_8_paired",
        "n_tasks": n_global,
        "n_favours": sum(1 for r in paired_rows if r["verdict"] == "FAVOURS"),
        "n_marginal": sum(1 for r in paired_rows if r["verdict"] == "marginal"),
        "n_null": sum(1 for r in paired_rows if r["verdict"] == "null"),
        "n_dir_positive": n_in_pred_dir,
        "n_dir_negative": n_against,
        "direction_consistent": "yes" if n_in_pred_dir == n_global else "no",
        "binom_p_two_sided_strict": f"{global_sign_p:.4f}",
        "binom_p_one_sided_signtest": f"{global_sign_p:.4f}",
    })
    summary_rows.append({
        "hypothesis": "GLOBAL_sign_test_over_all_8_paired",
        "tasks_favoured": "",
        "tasks_marginal": "",
        "n_favours": sum(1 for r in paired_rows if r["verdict"] == "FAVOURS"),
        "n_dir_pos": n_in_pred_dir,
        "binom_p_two_sided_strict": f"{global_sign_p:.4f}",
        "binom_p_one_sided_signtest": f"{global_sign_p:.4f}",
        "headline": "GLOBAL_REJECT" if n_in_pred_dir == n_global else "PARTIAL",
    })

    # Headline verdict
    headline = (f"Iter136 STEP-LEVEL TRAJECTORY COUPLING: {n_in_pred_dir}/{n_global} paired tests in predicted direction; "
                f"global sign-test p={global_sign_p:.4f}; "
                + "; ".join(f"{r['hypothesis']}={r['headline']}" for r in summary_rows
                            if r['hypothesis'] != "GLOBAL_sign_test_over_all_8_paired"))

    # ------------------------------------------------------------------
    # Write TSVs
    # ------------------------------------------------------------------
    def write_tsv(path: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            with open(path, "w") as f:
                pass
            return
        keys = list(rows[0].keys())
        with open(path, "w") as f:
            f.write("\t".join(keys) + "\n")
            for r in rows:
                f.write("\t".join(str(r[k]) for k in keys) + "\n")

    write_tsv(os.path.join(RES, "length_bias_iter136_step_coupling.tsv"), per_run_rows)
    write_tsv(os.path.join(RES, "length_bias_iter136_paired_tests.tsv"), paired_rows)
    write_tsv(os.path.join(RES, "length_bias_iter136_permutation_null.tsv"), perm_null_rows)
    write_tsv(os.path.join(RES, "length_bias_iter136_cross_task.tsv"), cross_rows)
    write_tsv(os.path.join(RES, "length_bias_iter136_summary.tsv"), summary_rows)

    meta = {
        "iter": 136,
        "pillar": "P4-LengthBias",
        "angle": "Step-level trajectory coupling between Δreward and Δlength",
        "tasks": ["arithmetic_easy", "gsm8k_cot"],
        "algos": ["grpo", "dr_grpo"],
        "n_seeds": {
            "arithmetic_easy": 5,
            "gsm8k_cot": 3,
        },
        "B_perm": args.B_perm,
        "seed_base": args.seed_base,
        "n_step_log_arith": 40,
        "n_step_log_gsm8k": 30,
        "hypotheses": [h[0] for h in hypotheses],
        "n_per_run_rows": len(per_run_rows),
        "n_paired_rows": len(paired_rows),
        "headline": headline,
        "summary_table": summary_rows,
        "paired_table": paired_rows,
    }
    with open(os.path.join(RES, "length_bias_iter136_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Print headline
    print(headline)
    print()
    print("Per-run:")
    for r in per_run_rows:
        print(f"  {r['task']:18s} {r['algo']:8s} seed={r['seed']:5d} "
              f"|rho(dR,dL)|={r['abs_rho_dR_dL']:6s} "
              f"|rho_len_lag1|={r['abs_rho_len_lag1']:6s} "
              f"late_eff={r['late_eff']:7s} "
              f"rho(dZ,dL)={r['rho_dZ_dL']:6s}")
    print()
    print("Paired tests:")
    for r in paired_rows:
        print(f"  {r['task']:18s} {r['hypothesis']:40s} Δ={r['mean_delta_dr_minus_gr']:7s} "
              f"W={int(r['W']):4d} p_param={r['p_param_one_sided']:7s} "
              f"p_perm={r['p_perm_two_sided']:7s} d={r['cohens_d_paired']:6s} -> {r['verdict']}")
    print()
    print("Cross-task summary:")
    for r in summary_rows:
        print(f"  {r['hypothesis']:45s} favours={r['n_favours']} signtest_p={r['binom_p_one_sided_signtest']:7s} -> {r['headline']}")


if __name__ == "__main__":
    main()