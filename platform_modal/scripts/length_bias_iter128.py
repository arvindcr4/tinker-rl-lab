#!/usr/bin/env python3
"""Iter 128 -- Pillar 4 (Length Bias / Dr.GRPO):
The Length-Efficiency Frontier.

Iter 108 measured per-window |CCF| magnitudes; iter 120 and iter 124
localised severship to the BACKWARD-CCF axis and the LENGTH-INFLATION
VELOCITY axis respectively.  All three confirmed that Dr.GR alters
the GRPO length-reward coupling but said nothing about EFFICIENCY.

Iter 128 asks the practitioner question:
  Does Dr.GR trade a measurable amount of accuracy for a meaningful
  amount of length parsimony, or does it sit strictly on (or inside)
  the GRPO length-efficiency frontier?

Sharp, falsifiable hypotheses:

  H1 -- Pareto dominance on the length-efficiency frontier.
        Per (algo, seed, task):
            dL  = mean_comp_len_first5 - mean_comp_len_last5  (within-run contraction)
            dR  = mean_reward_last5    - mean_reward_first5   (within-run gain)
            eff = dR / dL                                    (reward per token dropped)
        Dr.GR eff  >=  GR eff  in BOTH tasks.  Tested with paired
        Wilcoxon on per-seed ratios and Cohen's d on the paired delta.

  H2 -- Signed per-window CCF decoupling.
        From iter108_perrun_progress.tsv, per (algo, seed, task, window):
            |bwd_signed| (length -> reward, signed cross-correlation)
        Dr.GR's mean |bwd_signed|  <  GR's mean |bwd_signed|.
        Tested with paired Wilcoxon and permutation null on
        (algo, seed) shuffling within (task, window).

  H3 -- Cross-task envelope.
        Per-task mean eff(Dr.GR)  >=  mean eff(GR); both monotone
        across tasks in the same direction.

  H4 -- Heldout (GSM8K) efficiency frontier.
        GSM8K has per-prompt pre/post heldout accuracy.  Per seed:
            dL_heldout  = len_first5 - len_last5
            dacc_heldout = post_acc  - pre_acc
        Scatter and quantify the (dL, dacc) frontier; Dr.GR sits
        on a different trade-off curve, with smaller |dL| for
        similar |dacc|.

  H5 -- Sign-test on Dr.GR - GR eff > 0  over (task x seed).
        At least 7/9 paired (task, seed) cells favour Dr.GR; one-sided
        binomial p = P(X >= 7 | n=9, p=0.5) reported.

INPUTS :
  platform_hybrid/experiments/results/drgrpo_vs_grpo.json           (arithmetic_easy; 5 seeds)
  platform_hybrid/experiments/results/drgrpo_gsm8k_cot_full.json     (gsm8k_cot;       3 seeds)
  platform_hybrid/experiments/results/length_bias_iter108_perrun_progress.tsv (signed CCF)
OUTPUTS (5 TSV + meta) under platform_hybrid/experiments/results/length_bias_iter128_*
  length_bias_iter128_efficiency_frontier.tsvper (algo, seed, task)
  length_bias_iter128_signed_ccf.tsv              per (algo, seed, task, window)
  length_bias_iter128_pooled_h1_efficiency.tsv    pooled tests
  length_bias_iter128_permutation_null.tsv        H1, H2 permutation p-values
  length_bias_iter128_summary.tsv                 one-line headline per test
  length_bias_iter128_meta.json                   run metadata

USAGE  : python3 platform_modal/scripts/length_bias_iter128.py [--B_perm 50000]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results")

DRGR_VS_GRPO = os.path.join(RES, "drgrpo_vs_grpo.json")
DRGR_GSM8K = os.path.join(RES, "drgrpo_gsm8k_cot_full.json")
ITER108_PERRUN = os.path.join(RES, "length_bias_iter108_perrun_progress.tsv")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_drgrpo_arith() -> list[dict[str, Any]]:
    """arithmetic_easy: 5 seeds x 2 algos, no heldout pre/post, step_log present."""
    d = json.load(open(DRGR_VS_GRPO))
    out = []
    for r in d["runs"]:
        sl = r["step_log"]
        if len(sl) < 10:
            continue
        m_len_first5 = float(np.mean([s["mean_comp_len"] for s in sl[:5]]))
        m_len_last5 = float(np.mean([s["mean_comp_len"] for s in sl[-5:]]))
        m_rwd_first5 = float(np.mean([s["mean_reward"] for s in sl[:5]]))
        m_rwd_last5 = float(np.mean([s["mean_reward"] for s in sl[-5:]]))
        out.append({
            "task": "arithmetic_easy",
            "algo": r["algo"],
            "seed": int(r["seed"]),
            "n_steps": len(sl),
            "len_first5": m_len_first5,
            "len_last5": m_len_last5,
            "rwd_first5": m_rwd_first5,
            "rwd_last5": m_rwd_last5,
            "dL_within": m_len_first5 - m_len_last5,
            "dR_within": m_rwd_last5 - m_rwd_first5,
            # arithmetic_easy has no heldout pre/post in this file
            "pre_acc": float("nan"),
            "post_acc": float("nan"),
        })
    return out


def load_drgrpo_gsm8k() -> list[dict[str, Any]]:
    """gsm8k_cot: 3 seeds x 2 algos, heldout pre/post present."""
    d = json.load(open(DRGR_GSM8K))
    out = []
    for r in d["runs"]:
        out.append({
            "task": "gsm8k_cot",
            "algo": r["algo"],
            "seed": int(r["seed"]),
            "n_steps": len(r["step_log"]),
            "len_first5": r["mean_comp_len_first5"],
            "len_last5": r["mean_comp_len_last5"],
            "rwd_first5": float("nan"),   # not summarised
            "rwd_last5": float("nan"),
            "dL_within": r["mean_comp_len_first5"] - r["mean_comp_len_last5"],
            "dR_within": float("nan"),
            "pre_acc": r["heldout_pre_acc"],
            "post_acc": r["heldout_post_acc"],
            "dacc_heldout": r["heldout_post_acc"] - r["heldout_pre_acc"],
        })
    return out


def load_iter108_signed_ccf() -> list[dict[str, Any]]:
    rows = []
    with open(ITER108_PERRUN) as fh:
        hdr = fh.readline().rstrip().split("\t")
        for line in fh:
            f = line.rstrip().split("\t")
            row = dict(zip(hdr, f))
            row["window"] = int(row["window"])
            row["seed"] = int(row["seed"])
            row["n_total"] = int(row["n_total"])
            row["n_in_window"] = int(row["n_in_window"])
            for k in ("phi_L", "phi_R", "bwd", "fwd",
                      "bwd_signed", "fwd_signed"):
                row[k] = float(row[k])
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
def efficiency_frontier(runs: list[dict]) -> list[dict]:
    """For each run, return efficiency = dR / dL (where defined).

    For arithmetic_easy we have dR_within > 0 and dL_within > 0 (length
    contracts, reward grows).  For gsm8k_cot we have dL_within > 0 but
    no within-run reward summary; we fall back to dacc_heldout as the
    numerator (heldout accuracy gain over training).
    """
    out = []
    for r in runs:
        dL = r["dL_within"]
        # numerator choice:
        if not np.isnan(r["dR_within"]):
            num = r["dR_within"]
            metric = "dR_within"
        elif not np.isnan(r.get("dacc_heldout", float("nan"))):
            num = r["dacc_heldout"]
            metric = "dacc_heldout"
        else:
            num = float("nan")
            metric = "none"
        eff = num / dL if (dL > 0 and not np.isnan(num)) else float("nan")
        out.append({
            **r,
            "efficiency": eff,
            "efficiency_metric": metric,
        })
    return out


def paired_efficiency_test(
    runs_with_eff: list[dict],
) -> dict[str, Any]:
    """For each task, paired (Dr.GR, GR) over seeds -> Wilcoxon + Cohen's d."""
    out = {}
    tasks = sorted({r["task"] for r in runs_with_eff})
    for task in tasks:
        sub = [r for r in runs_with_eff if r["task"] == task]
        gr_e = [r["efficiency"] for r in sub if r["algo"] == "grpo"
                and not np.isnan(r["efficiency"])]
        dr_e = [r["efficiency"] for r in sub if r["algo"] == "dr_grpo"
                and not np.isnan(r["efficiency"])]
        # align by seed
        seeds = sorted({r["seed"] for r in sub})
        gr_by_seed = {r["seed"]: r["efficiency"] for r in sub if r["algo"] == "grpo"}
        dr_by_seed = {r["seed"]: r["efficiency"] for r in sub if r["algo"] == "dr_grpo"}
        deltas = []
        for s in seeds:
            if s in gr_by_seed and s in dr_by_seed:
                if not np.isnan(gr_by_seed[s]) and not np.isnan(dr_by_seed[s]):
                    deltas.append(dr_by_seed[s] - gr_by_seed[s])
        if len(deltas) >= 2:
            w = stats.wilcoxon(deltas, alternative="greater")
            mean_delta = float(np.mean(deltas))
            sd_delta = float(np.std(deltas, ddof=1))
            d = mean_delta / sd_delta if sd_delta > 0 else float("nan")
            out[task] = {
                "n_pairs": len(deltas),
                "mean_eff_gr": float(np.mean(gr_e)) if gr_e else float("nan"),
                "mean_eff_dr": float(np.mean(dr_e)) if dr_e else float("nan"),
                "mean_delta_dr_minus_gr": mean_delta,
                "wilcoxon_W": float(w.statistic),
                "wilcoxon_p_one_sided": float(w.pvalue),
                "cohens_d": d,
            }
        else:
            out[task] = {
                "n_pairs": len(deltas),
                "mean_eff_gr": float(np.mean(gr_e)) if gr_e else float("nan"),
                "mean_eff_dr": float(np.mean(dr_e)) if dr_e else float("nan"),
                "mean_delta_dr_minus_gr": float("nan"),
                "wilcoxon_W": float("nan"),
                "wilcoxon_p_one_sided": float("nan"),
                "cohens_d": float("nan"),
            }
    return out


def paired_signed_ccf_test(
    signed_rows: list[dict],
) -> dict[str, Any]:
    """For each task, paired Dr.GR vs GR |bwd_signed| over (seed, window)."""
    out = {}
    tasks = sorted({r["task"] for r in signed_rows})
    for task in tasks:
        sub = [r for r in signed_rows if r["task"] == task]
        # group by (seed, window), take |bwd_signed| per algo
        keys = sorted({(r["seed"], r["window"]) for r in sub})
        deltas = []
        for (s, w) in keys:
            gr = [r for r in sub if r["seed"] == s and r["window"] == w
                  and r["algo"] == "grpo"]
            dr = [r for r in sub if r["seed"] == s and r["window"] == w
                  and r["algo"] == "dr_grpo"]
            if gr and dr:
                delta = abs(dr[0]["bwd_signed"]) - abs(gr[0]["bwd_signed"])
                deltas.append(delta)
        if len(deltas) >= 4:
            w = stats.wilcoxon(deltas, alternative="less")
            mean_delta = float(np.mean(deltas))
            sd_delta = float(np.std(deltas, ddof=1))
            d = mean_delta / sd_delta if sd_delta > 0 else float("nan")
            out[task] = {
                "n_pairs": len(deltas),
                "mean_abs_bwd_gr": float(np.mean([abs(r["bwd_signed"]) for r in sub
                                                  if r["algo"] == "grpo"])),
                "mean_abs_bwd_dr": float(np.mean([abs(r["bwd_signed"]) for r in sub
                                                  if r["algo"] == "dr_grpo"])),
                "mean_delta_dr_minus_gr": mean_delta,
                "wilcoxon_W": float(w.statistic),
                "wilcoxon_p_one_sided": float(w.pvalue),
                "cohens_d": d,
            }
        else:
            out[task] = {
                "n_pairs": len(deltas),
                "mean_delta_dr_minus_gr": float("nan"),
                "wilcoxon_p_one_sided": float("nan"),
            }
    return out


def permutation_null_paired(
    paired_values: list[tuple[float, float]],
    alternative: str = "greater",
    B: int = 50000,
    seed: int = 20260703,
) -> float:
    """One-sample paired permutation null on the list of (dr, gr) pairs.

    Tests H0: mean(dr - gr) <= 0  vs  H1: mean(dr - gr) > 0 (or 'less' for reverse).
    Returns two-sided p-value approximated from the permutation distribution
    of the *mean* delta, with sign-flip (within-pair swap) as the permuter.
    """
    rng = np.random.default_rng(seed)
    if not paired_values:
        return float("nan")
    arr = np.asarray(paired_values, dtype=float)  # (n, 2)
    n = arr.shape[0]
    obs_deltas = arr[:, 0] - arr[:, 1]
    obs_mean = float(np.mean(obs_deltas))
    if alternative == "greater":
        obs_abs = obs_mean
    elif alternative == "less":
        obs_abs = -obs_mean
    else:
        obs_abs = abs(obs_mean)
    count = 0
    for _ in range(B):
        flip = rng.integers(0, 2, size=n) * 2 - 1   # +/-1
        perm_mean = float(np.mean(obs_deltas * flip))
        if alternative == "greater":
            if perm_mean >= obs_abs:
                count += 1
        elif alternative == "less":
            if -perm_mean >= obs_abs:
                count += 1
        else:
            if abs(perm_mean) >= obs_abs:
                count += 1
    return (count + 1) / (B + 1)


def cross_task_envelope(
    eff_results_by_task: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Per-task mean eff(Dr.GR) / mean eff(GR).  Both should be >= 1."""
    out = {}
    for task, vals in eff_results_by_task.items():
        out[task] = {
            "mean_eff_gr": vals["mean_eff_gr"],
            "mean_eff_dr": vals["mean_eff_dr"],
            "ratio_dr_over_gr": (vals["mean_eff_dr"] / vals["mean_eff_gr"]
                                 if vals["mean_eff_gr"] != 0
                                 else float("nan")),
            "dr_minus_gr": vals["mean_eff_dr"] - vals["mean_eff_gr"],
        }
    return out


def sign_test(
    runs_with_eff: list[dict],
) -> dict[str, Any]:
    """Sign test on Dr.GR - GR eff across (task, seed) pairs."""
    tasks = sorted({r["task"] for r in runs_with_eff})
    cells = []
    for task in tasks:
        sub = [r for r in runs_with_eff if r["task"] == task]
        seeds = sorted({r["seed"] for r in sub})
        for s in seeds:
            gr = [r for r in sub if r["seed"] == s and r["algo"] == "grpo"]
            dr = [r for r in sub if r["seed"] == s and r["algo"] == "dr_grpo"]
            if gr and dr and not np.isnan(gr[0]["efficiency"]) \
                    and not np.isnan(dr[0]["efficiency"]):
                cells.append({
                    "task": task,
                    "seed": s,
                    "eff_gr": gr[0]["efficiency"],
                    "eff_dr": dr[0]["efficiency"],
                    "delta_dr_minus_gr": dr[0]["efficiency"] - gr[0]["efficiency"],
                })
    n = len(cells)
    n_dr_wins = sum(1 for c in cells if c["delta_dr_minus_gr"] > 0)
    n_gr_wins = sum(1 for c in cells if c["delta_dr_minus_gr"] < 0)
    n_ties = n - n_dr_wins - n_gr_wins
    if n > 0 and (n_dr_wins + n_gr_wins) > 0:
        # one-sided binomial p: P(X >= n_dr_wins | n_eff, p=0.5)
        n_eff = n_dr_wins + n_gr_wins
        p_one_sided = float(stats.binomtest(n_dr_wins, n_eff, 0.5,
                                            alternative="greater").pvalue)
    else:
        p_one_sided = float("nan")
    return {
        "n_cells": n,
        "n_dr_wins": n_dr_wins,
        "n_gr_wins": n_gr_wins,
        "n_ties": n_ties,
        "binom_p_one_sided": p_one_sided,
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_tsv(path: str, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        for row in rows:
            w.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B_perm", type=int, default=50000)
    ap.add_argument("--seed_base", type=int, default=20260703)
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    arith = load_drgrpo_arith()
    gsm = load_drgrpo_gsm8k()
    signed = load_iter108_signed_ccf()

    # arithmetic_easy: efficiency uses dR/dL (within-run training reward)
    # gsm8k_cot:        efficiency uses dacc_heldout / dL
    runs = arith + gsm
    runs_with_eff = efficiency_frontier(runs)

    # ------------------------------------------------------------------
    # 2. H1 -- per-task paired Wilcoxon + Cohen's d on efficiency ratio
    # ------------------------------------------------------------------
    h1_by_task = paired_efficiency_test(runs_with_eff)

    # ------------------------------------------------------------------
    # 3. H2 -- paired signed CCF (|bwd_signed|) per task
    # ------------------------------------------------------------------
    h2_by_task = paired_signed_ccf_test(signed)

    # ------------------------------------------------------------------
    # 4. H3 -- cross-task envelope
    # ------------------------------------------------------------------
    h3_envelope = cross_task_envelope(h1_by_task)

    # ------------------------------------------------------------------
    # 5. H4 -- heldout (GSM8K) frontier: per-seed (dL, dacc)
    # ------------------------------------------------------------------
    gsm_rows = [r for r in runs_with_eff if r["task"] == "gsm8k_cot"
                and not np.isnan(r.get("dacc_heldout", float("nan")))]

    # ------------------------------------------------------------------
    # 6. H5 -- sign test across (task, seed)
    # ------------------------------------------------------------------
    h5 = sign_test(runs_with_eff)

    # ------------------------------------------------------------------
    # 7. Permutation nulls
    # ------------------------------------------------------------------
    # H1 permutation null: paired (dr_eff, gr_eff) per (task, seed)
    h1_pairs = [(c["eff_dr"], c["eff_gr"]) for c in h5["cells"]]
    h1_perm_p = permutation_null_paired(h1_pairs, alternative="greater",
                                        B=args.B_perm, seed=args.seed_base)

    # H2 permutation null: paired |bwd_signed| per (task, seed, window)
    h2_pairs = []
    for r in signed:
        # match by (task, seed, window)
        match = [s for s in signed
                 if s["task"] == r["task"] and s["seed"] == r["seed"]
                 and s["window"] == r["window"]
                 and s["algo"] != r["algo"]]
        if match:
            other = match[0]
            if r["algo"] == "dr_grpo":
                h2_pairs.append((abs(r["bwd_signed"]),
                                 abs(other["bwd_signed"])))
    h2_perm_p = permutation_null_paired(h2_pairs, alternative="less",
                                        B=args.B_perm, seed=args.seed_base + 1)

    # ------------------------------------------------------------------
    # 8. Write outputs
    # ------------------------------------------------------------------
    # (a) efficiency frontier table -- one row per (algo, seed, task)
    ef_header = ["task", "algo", "seed", "n_steps", "len_first5", "len_last5",
                 "dL_within", "rwd_first5", "rwd_last5", "dR_within",
                 "pre_acc", "post_acc", "dacc_heldout",
                 "efficiency", "efficiency_metric"]
    ef_rows = []
    for r in runs_with_eff:
        ef_rows.append([
            r["task"], r["algo"], r["seed"], r["n_steps"],
            f"{r['len_first5']:.6f}", f"{r['len_last5']:.6f}",
            f"{r['dL_within']:.6f}",
            f"{r.get('rwd_first5', float('nan')):.6f}",
            f"{r.get('rwd_last5', float('nan')):.6f}",
            f"{r.get('dR_within', float('nan')):.6f}",
            f"{r.get('pre_acc', float('nan')):.6f}",
            f"{r.get('post_acc', float('nan')):.6f}",
            f"{r.get('dacc_heldout', float('nan')):.6f}",
            f"{r.get('efficiency', float('nan')):.6f}",
            r.get("efficiency_metric", "none"),
        ])
    write_tsv(os.path.join(RES, "length_bias_iter128_efficiency_frontier.tsv"),
              ef_header, ef_rows)

    # (b) signed CCF table -- one row per (algo, seed, task, window)
    cc_header = ["task", "algo", "seed", "window", "n_in_window",
                 "phi_L", "phi_R", "bwd", "fwd", "bwd_signed", "fwd_signed",
                 "abs_bwd_signed"]
    cc_rows = []
    for r in sorted(signed,
                    key=lambda x: (x["task"], x["algo"], x["seed"], x["window"])):
        cc_rows.append([
            r["task"], r["algo"], r["seed"], r["window"], r["n_in_window"],
            f"{r['phi_L']:.4f}", f"{r['phi_R']:.4f}",
            f"{r['bwd']:.4f}", f"{r['fwd']:.4f}",
            f"{r['bwd_signed']:.4f}", f"{r['fwd_signed']:.4f}",
            f"{abs(r['bwd_signed']):.4f}",
        ])
    write_tsv(os.path.join(RES, "length_bias_iter128_signed_ccf.tsv"),
              cc_header, cc_rows)

    # (c) pooled H1 efficiency table -- per-task paired tests
    pool_header = ["task", "n_pairs", "mean_eff_gr", "mean_eff_dr",
                   "mean_delta_dr_minus_gr", "wilcoxon_W",
                   "wilcoxon_p_one_sided", "cohens_d"]
    pool_rows = []
    for task, vals in h1_by_task.items():
        pool_rows.append([
            task, vals["n_pairs"],
            f"{vals['mean_eff_gr']:.6f}",
            f"{vals['mean_eff_dr']:.6f}",
            f"{vals['mean_delta_dr_minus_gr']:.6f}",
            f"{vals['wilcoxon_W']:.4f}",
            f"{vals['wilcoxon_p_one_sided']:.6f}",
            f"{vals['cohens_d']:.4f}",
        ])
    write_tsv(os.path.join(RES, "length_bias_iter128_pooled_h1_efficiency.tsv"),
              pool_header, pool_rows)

    # (d) permutation null table -- one row per hypothesis
    perm_header = ["hypothesis", "alternative", "n_pairs",
                   "observed_stat", "B", "permutation_p"]
    perm_rows = []
    # compute observed stats
    obs_mean_delta_h1 = float(np.mean([c[0] - c[1] for c in h1_pairs])) \
        if h1_pairs else float("nan")
    obs_mean_delta_h2 = float(np.mean([c[0] - c[1] for c in h2_pairs])) \
        if h2_pairs else float("nan")
    perm_rows.append(["H1_efficiency_pairwise", "greater",
                      len(h1_pairs), f"{obs_mean_delta_h1:.6f}",
                      args.B_perm, f"{h1_perm_p:.6f}"])
    perm_rows.append(["H2_signed_ccf_pairwise", "less",
                      len(h2_pairs), f"{obs_mean_delta_h2:.6f}",
                      args.B_perm, f"{h2_perm_p:.6f}"])
    write_tsv(os.path.join(RES, "length_bias_iter128_permutation_null.tsv"),
              perm_header, perm_rows)

    # (e) summary table -- one headline per test
    sum_header = ["test", "description", "result", "p", "n"]
    sum_rows = []
    # H1: per-task
    for task, vals in h1_by_task.items():
        verdict = "FAVOURS Dr.GR" if vals["mean_delta_dr_minus_gr"] > 0 else "FAVOURS GR"
        sum_rows.append([
            f"H1_efficiency[{task}]",
            f"Wilcoxon paired Dr.GR eff vs GR eff (one-sided, alt=greater)",
            verdict,
            f"{vals['wilcoxon_p_one_sided']:.6f}",
            vals["n_pairs"],
        ])
    # H2: per-task
    for task, vals in h2_by_task.items():
        verdict = "FAVOURS Dr.GR (tighter)" if vals["mean_delta_dr_minus_gr"] < 0 \
            else "FAVOURS GR"
        sum_rows.append([
            f"H2_signed_ccf[{task}]",
            f"Wilcoxon paired |bwd_signed| Dr.GR vs GR (one-sided, alt=less)",
            verdict,
            f"{vals['wilcoxon_p_one_sided']:.6f}",
            vals["n_pairs"],
        ])
    # H3: envelope ratio
    for task, vals in h3_envelope.items():
        verdict = "Dr.GR >= GR" if vals["ratio_dr_over_gr"] >= 1.0 \
            else "Dr.GR < GR"
        sum_rows.append([
            f"H3_envelope_ratio[{task}]",
            "Cross-task envelope of eff(Dr.GR)/eff(GR)",
            verdict,
            "n/a",
            "all_seeds",
        ])
    # H4: GSM8K heldout frontier (qualitative)
    if gsm_rows:
        # mean within-task frontier
        sub_dr = [r for r in gsm_rows if r["algo"] == "dr_grpo"]
        sub_gr = [r for r in gsm_rows if r["algo"] == "grpo"]
        if sub_dr and sub_gr:
            m_dL_dr = float(np.mean([r["dL_within"] for r in sub_dr]))
            m_dL_gr = float(np.mean([r["dL_within"] for r in sub_gr]))
            m_da_dr = float(np.mean([r["dacc_heldout"] for r in sub_dr]))
            m_da_gr = float(np.mean([r["dacc_heldout"] for r in sub_gr]))
            verdict = "Dr.GR shorter at comparable acc" \
                if (m_dL_dr < m_dL_gr and abs(m_da_dr - m_da_gr) < 0.02) \
                else "trade-off visible"
            sum_rows.append([
                "H4_heldout_frontier[gsm8k_cot]",
                f"mean dL(DR)={m_dL_dr:.2f} vs dL(GR)={m_dL_gr:.2f}; "
                f"mean dacc(DR)={m_da_dr:+.4f} vs dacc(GR)={m_da_gr:+.4f}",
                verdict,
                "n/a",
                len(gsm_rows),
            ])
    # H5: sign test
    sum_rows.append([
        "H5_sign_test",
        f"Binomial P(Dr.GR wins | n={h5['n_dr_wins'] + h5['n_gr_wins']}, p=0.5)",
        f"Dr.GR wins {h5['n_dr_wins']}/{h5['n_dr_wins'] + h5['n_gr_wins']}",
        f"{h5['binom_p_one_sided']:.6f}",
        h5["n_cells"],
    ])
    write_tsv(os.path.join(RES, "length_bias_iter128_summary.tsv"),
              sum_header, sum_rows)

    # (f) meta JSON
    meta = {
        "iter": 128,
        "pillar": "P4-LengthBias",
        "B_perm": args.B_perm,
        "seed_base": args.seed_base,
        "tasks": ["arithmetic_easy", "gsm8k_cot"],
        "algos": ["grpo", "dr_grpo"],
        "n_seeds_arithmetic": 5,
        "n_seeds_gsm8k": 3,
        "n_efficiency_cells": len(h1_pairs),
        "n_signed_ccf_pairs": len(h2_pairs),
        "h1_paired_by_task": h1_by_task,
        "h2_paired_by_task": h2_by_task,
        "h3_envelope": h3_envelope,
        "h5_sign_test": {k: v for k, v in h5.items() if k != "cells"},
        "h1_perm_p": h1_perm_p,
        "h2_perm_p": h2_perm_p,
    }
    with open(os.path.join(RES, "length_bias_iter128_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # ------------------------------------------------------------------
    # 9. Print concise headline
    # ------------------------------------------------------------------
    print("=" * 72)
    print("Iter 128 -- Length-Efficiency Frontier")
    print("=" * 72)
    for task, vals in h1_by_task.items():
        print(f"  H1 efficiency[{task:14s}]: "
              f"GR eff={vals['mean_eff_gr']:+.4f}  "
              f"DR eff={vals['mean_eff_dr']:+.4f}  "
              f"delta={vals['mean_delta_dr_minus_gr']:+.4f}  "
              f"p={vals['wilcoxon_p_one_sided']:.4f}  "
              f"d={vals['cohens_d']:+.2f}")
    for task, vals in h2_by_task.items():
        print(f"  H2 |bwd_signed|[{task:14s}]: "
              f"GR={vals['mean_abs_bwd_gr']:.3f}  "
              f"DR={vals['mean_abs_bwd_dr']:.3f}  "
              f"delta={vals['mean_delta_dr_minus_gr']:+.3f}  "
              f"p={vals['wilcoxon_p_one_sided']:.4f}")
    print(f"  H3 envelope ratios:")
    for task, vals in h3_envelope.items():
        print(f"    [{task:14s}] ratio={vals['ratio_dr_over_gr']:+.3f}")
    print(f"  H5 sign test: Dr.GR wins {h5['n_dr_wins']}/{h5['n_dr_wins'] + h5['n_gr_wins']}, "
          f"binomial p={h5['binom_p_one_sided']:.4f}")
    print(f"  H1 permutation p={h1_perm_p:.4f}")
    print(f"  H2 permutation p={h2_perm_p:.4f}")


if __name__ == "__main__":
    main()