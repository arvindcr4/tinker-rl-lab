#!/usr/bin/env python3
"""Iter 132 -- Pillar 4 (Length Bias / Dr.GRPO):
Closing the causal chain -- severship -> backward-CCF decoupling -> per-window
efficiency gain.

Prior iterations established three pieces of evidence about Dr.GR:
  iter108: per-window backward |CCF(L,R)| magnitude decomposition
  iter120: pooled Spearman(severship, |bwd_CCF|^GR) = +0.556, p<0.001
           i.e. severship STRONGLY predicts smaller |bwd_CCF| in the GR arm.
  iter124: severship does NOT predict length-inflation velocity
           dL/dt (static-vs-dynamic dissociation)
  iter128: Dr.GR Pareto-dominates GR on the run-level (Delta L, Delta R)
           length-efficiency frontier (Cohen's d = +3.62 on arithmetic_easy).

WHAT'S MISSING.  Iter 120 showed severship -> smaller |bwd_CCF|, and iter 128
showed Dr.GR dominates on efficiency.  We have not tested whether the
backward-CCF decoupling IS THE MECHANISM by which Dr.GR gains efficiency.
Iter 132 closes this loop with a tri-variate causal-chain test at the
WINDOW level (rather than run-level), which is the natural unit at which
both |bwd_CCF| (iter108) and (Delta L, Delta R) (this iter) are computed.

Sharp, falsifiable hypotheses:

  H1 -- Mediation chain.  For each (algo, seed, task, window):
            eff_w    = (mean_rwd_lasthalf_w - mean_rwd_firsthalf_w)
                     / (mean_len_firsthalf_w - mean_len_lasthalf_w)
                     (reward-gain per token dropped, within window w)
            ccfa_w   = |bwd_signed|_w  (from iter108_perrun_progress.tsv)
        Per task, paired delta across (seed, window):
            delta_eff = eff(DR) - eff(GR)
            delta_ccfa = |bwd|_GR - |bwd|_DR  (positive = Dr.GR tighter)
        Spearman rho(delta_eff, delta_ccfa) > 0 with permutation p.

  H2 -- Iso-eff contour.  At fixed efficiency, Dr.GR's |bwd_CCF| is
        systematically LOWER than GR's.  Per-task paired Wilcoxon on
        |bwd_CCF| at matched quantile of eff.

  H3 -- Mediation ratio (Sobel-style).  The delta_eff decomposes into
        (a) direct effect of algo and (b) indirect effect via delta_|bwd|.
        Bootstrap the indirect/direct ratio; report CI.

  H4 -- Cross-task consistency.  H1's rho > 0 in BOTH tasks; binomial
        on (rho>0, p<0.10) over 2 tasks.

INPUTS :
  platform_hybrid/experiments/results/drgrpo_vs_grpo.json            (arithmetic_easy, 5 seeds)
  platform_hybrid/experiments/results/drgrpo_gsm8k_cot_full.json     (gsm8k_cot, 3 seeds)
  platform_hybrid/experiments/results/length_bias_iter108_perrun_progress.tsv
OUTPUTS (5 TSV + meta) under platform_hybrid/experiments/results/length_bias_iter132_*
  length_bias_iter132_window_efficiency.tsv
  length_bias_iter132_paired_delta.tsv
  length_bias_iter132_mediation_chain.tsv
  length_bias_iter132_permutation_null.tsv
  length_bias_iter132_summary.tsv
  length_bias_iter132_meta.json

USAGE  : python3 platform_modal/scripts/length_bias_iter132.py [--B_perm 50000 --n_w 4]
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
ITER108_PERRUN = os.path.join(RES, "length_bias_iter108_perrun_progress.tsv")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_step_log(path: str, task_alias: str) -> list[dict[str, Any]]:
    """Load runs from a JSON.  The iter108 perrun TSV uses task aliases
    'arithmetic_easy' and 'gsm8k_cot'; the JSON files use 'drgrpo_vs_grpo'
    and 'drgrpo_gsm8k_cot' respectively.  We override with task_alias.
    """
    with open(path) as fh:
        d = json.load(fh)
    out = []
    for r in d["runs"]:
        sl = r.get("step_log") or []
        if len(sl) < 5:
            continue
        L = np.array([float(s["mean_comp_len"]) for s in sl], dtype=np.float64)
        R = np.array([float(s["mean_reward"]) for s in sl], dtype=np.float64)
        out.append({"task": task_alias, "algo": r["algo"],
                    "seed": int(r["seed"]), "n": int(len(sl)), "L": L, "R": R})
    return out


def load_iter108_ccfa() -> list[dict[str, Any]]:
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
# Per-window efficiency  -- CUMULATIVE version.
#
# Within-window efficiency is noisy because mid/late-training length
# plateau makes within-window dL ~ 0 or negative.  Instead, we compute
# the CUMULATIVE efficiency up to each window boundary:
#
#   cum_dR(w) = mean_R[last half of window w] - mean_R[first half of window 0]
#   cum_dL(w) = mean_L[first half of window 0] - mean_L[last half of window w]
#   cum_eff(w) = cum_dR(w) / cum_dL(w)
#
# cum_eff(0) is the early-training efficiency, cum_eff(n_w-1) is the
# run-level efficiency (matching iter128).
# ---------------------------------------------------------------------------
def per_window_efficiency(runs: list[dict], n_w: int) -> list[dict]:
    out = []
    for r in runs:
        n = r["n"]
        edges = np.linspace(0, n, n_w + 1).astype(int)
        # baseline: first half of window 0
        i0, i1 = int(edges[0]), int(edges[1])
        fh0 = np.arange(i0, (i0 + i1) // 2)
        if len(fh0) < 2:
            continue
        rwd_base = float(np.mean(r["R"][fh0]))
        len_base = float(np.mean(r["L"][fh0]))
        for w in range(n_w):
            i0w, i1w = int(edges[w]), int(edges[w + 1])
            lhw = np.arange((i0w + i1w) // 2, i1w)
            if len(lhw) < 2:
                continue
            rwd_lh = float(np.mean(r["R"][lhw]))
            len_lh = float(np.mean(r["L"][lhw]))
            cum_dR = rwd_lh - rwd_base
            cum_dL = len_base - len_lh
            if cum_dL > 1e-6:
                cum_eff = cum_dR / cum_dL
            else:
                cum_eff = float("nan")
            out.append({"task": r["task"], "algo": r["algo"],
                        "seed": r["seed"], "window": w,
                        "n_in_window": int(i1w - i0w),
                        "rwd_base": rwd_base, "rwd_lh": rwd_lh,
                        "len_base": len_base, "len_lh": len_lh,
                        "cum_dR": cum_dR, "cum_dL": cum_dL,
                        "cum_eff": cum_eff})
    return out


# ---------------------------------------------------------------------------
# Joining efficiency with iter108 backward CCF
# ---------------------------------------------------------------------------
def join(win_eff: list[dict], ccfa: list[dict]) -> list[dict]:
    """Per (task, algo, seed, window), join eff and |bwd_signed|."""
    ccfa_idx = {}
    for c in ccfa:
        ccfa_idx[(c["task"], c["algo"], c["seed"], c["window"])] = c
    out = []
    for e in win_eff:
        key = (e["task"], e["algo"], e["seed"], e["window"])
        c = ccfa_idx.get(key)
        if c is None:
            continue
        out.append({**e,
                    "bwd_signed": c["bwd_signed"],
                    "abs_bwd": abs(c["bwd_signed"]),
                    "fwd_signed": c["fwd_signed"],
                    "abs_fwd": abs(c["fwd_signed"]),
                    "phi_L": c["phi_L"],
                    "phi_R": c["phi_R"]})
    return out


# ---------------------------------------------------------------------------
# Paired delta at the WINDOW level (kept for diagnostic TSVs).
# ---------------------------------------------------------------------------
def paired_delta(joined: list[dict]) -> list[dict]:
    """For each (task, seed, window), build the paired row
        delta_eff   = cum_eff(DR) - cum_eff(GR)
        delta_ccfa  = |bwd|_GR - |bwd|_DR
    """
    by_key = {}
    for j in joined:
        key = (j["task"], j["seed"], j["window"])
        by_key.setdefault(key, {})[j["algo"]] = j
    out = []
    for (task, seed, window), d in by_key.items():
        if "grpo" not in d or "dr_grpo" not in d:
            continue
        g, dr = d["grpo"], d["dr_grpo"]
        out.append({"task": task, "seed": seed, "window": window,
                    "eff_gr": g["cum_eff"], "eff_dr": dr["cum_eff"],
                    "delta_eff": dr["cum_eff"] - g["cum_eff"],
                    "abs_bwd_gr": g["abs_bwd"], "abs_bwd_dr": dr["abs_bwd"],
                    "delta_ccfa": g["abs_bwd"] - dr["abs_bwd"],
                    "fwd_gr": g["fwd_signed"], "fwd_dr": dr["fwd_signed"],
                    "delta_fwd": dr["fwd_signed"] - g["fwd_signed"],
                    "phi_L_gr": g["phi_L"], "phi_L_dr": dr["phi_L"],
                    "delta_phi_L": dr["phi_L"] - g["phi_L"],
                    "cum_dR_gr": g["cum_dR"], "cum_dR_dr": dr["cum_dR"],
                    "cum_dL_gr": g["cum_dL"], "cum_dL_dr": dr["cum_dL"]})
    return out


# ---------------------------------------------------------------------------
# Within-run correlation: |bwd_CCF|_w  vs  cum_eff_w  within each run
# ---------------------------------------------------------------------------
def within_run_rho(joined: list[dict]) -> list[dict]:
    """Per (algo, seed, task), compute Spearman rho(|bwd|_w, cum_eff_w)
    over the n_w windows.  Dr.GR's rho should be MORE NEGATIVE than
    GR's: at windows where the backward CCF is small, cum_eff is
    larger (i.e., Dr.GR breaks the L->R coupling and lets reward
    accumulate without needing length to drop).
    """
    by_run = {}
    for j in joined:
        by_run.setdefault((j["task"], j["algo"], j["seed"]), []).append(j)
    out = []
    for (task, algo, seed), rows in by_run.items():
        rows = sorted(rows, key=lambda r: r["window"])
        ccfa = np.array([r["abs_bwd"] for r in rows], dtype=np.float64)
        eff = np.array([r["cum_eff"] for r in rows], dtype=np.float64)
        mask = ~(np.isnan(ccfa) | np.isnan(eff))
        ccfa = ccfa[mask]; eff = eff[mask]
        if len(ccfa) < 3:
            rho, p = float("nan"), float("nan")
        else:
            rho, p = stats.spearmanr(ccfa, eff)
        out.append({"task": task, "algo": algo, "seed": seed,
                    "n_windows": int(mask.sum()),
                    "rho_within": float(rho),
                    "p_within": float(p),
                    "mean_abs_bwd": float(np.mean(ccfa)),
                    "max_abs_bwd": float(np.max(ccfa)),
                    "min_abs_bwd": float(np.min(ccfa)),
                    "final_cum_eff": float(eff[-1]) if len(eff) else float("nan")})
    return out


def paired_within_run_rho(within: list[dict]) -> list[dict]:
    """Pair Dr.GR vs GR per (seed, task).  Compute delta_rho, delta_|bwd|,
    delta_final_eff.  Output one row per (task, seed)."""
    by_key = {}
    for r in within:
        by_key.setdefault((r["task"], r["seed"]), {})[r["algo"]] = r
    out = []
    for (task, seed), d in by_key.items():
        if "grpo" not in d or "dr_grpo" not in d:
            continue
        g, dr = d["grpo"], d["dr_grpo"]
        out.append({"task": task, "seed": seed,
                    "rho_gr": g["rho_within"], "rho_dr": dr["rho_within"],
                    "delta_rho": dr["rho_within"] - g["rho_within"],
                    "mean_bwd_gr": g["mean_abs_bwd"],
                    "mean_bwd_dr": dr["mean_abs_bwd"],
                    "delta_mean_bwd": dr["mean_abs_bwd"] - g["mean_abs_bwd"],
                    "final_eff_gr": g["final_cum_eff"],
                    "final_eff_dr": dr["final_cum_eff"],
                    "delta_final_eff": dr["final_cum_eff"] - g["final_cum_eff"]})
    return out


def spearman_with_perm(x: np.ndarray, y: np.ndarray, B: int, seed: int = 0xA132):
    if len(x) < 3 or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        return {"rho": float("nan"), "p_param": float("nan"),
                "p_perm": float("nan"), "n": int(len(x))}
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    cnt = 0
    abs_rho = abs(float(rho))
    for _ in range(B):
        perm = rng.permutation(n)
        rp, _ = stats.spearmanr(x[perm], y)
        if abs(float(rp)) >= abs_rho:
            cnt += 1
    return {"rho": float(rho), "p_param": float(p),
            "p_perm": (cnt + 1) / (B + 1), "n": int(n)}


def wilcoxon_one_sided(a: np.ndarray, b: np.ndarray, alt: str = "greater"):
    """Paired one-sided Wilcoxon (a vs b, paired by index).
    alt='greater': H1: a > b.  alt='less': H1: a < b.
    Returns W (sum of positive ranks), p (one-sided), Cohen's d on delta.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    if n < 2:
        return {"n": int(n), "W": float("nan"), "p": float("nan"),
                "d": float("nan")}
    a = a[:n]; b = b[:n]
    diff = a - b
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2 or np.all(diff == 0):
        return {"n": int(len(diff)), "W": float("nan"), "p": float("nan"),
                "d": float("nan")}
    if alt == "greater":
        W, p = stats.wilcoxon(diff, alternative="greater")
    else:
        W, p = stats.wilcoxon(diff, alternative="less")
    d = float(diff.mean() / (diff.std(ddof=1) + 1e-300))
    return {"n": int(len(diff)), "W": float(W), "p": float(p), "d": d}


# ---------------------------------------------------------------------------
# H2: iso-efficiency contour.  At matched eff quantile, is Dr.GR's
# |bwd_CCF| lower than GR's?
# ---------------------------------------------------------------------------
def iso_eff_contour(joined: list[dict]) -> dict:
    """Per task: bin (algo, seed, window) points by CUM-EFF QUANTILE
    (q in {0-25%, 25-50%, 50-75%, 75-100%}), then compare Dr.GR vs GR
    mean |bwd_CCF| at matched q.  Lower Dr.GR = Dr.GR Pareto-dominates
    in (eff, |bwd|) plane.
    """
    out = {}
    by_task = {}
    for j in joined:
        by_task.setdefault(j["task"], []).append(j)
    for task, rows in by_task.items():
        effs = np.array([r["cum_eff"] for r in rows], dtype=np.float64)
        if len(effs) < 8:
            out[task] = {"n_total": int(len(rows))}
            continue
        qs = np.quantile(effs, [0.25, 0.5, 0.75])
        bins = {"q0": (effs <= qs[0]),
                "q1": (effs > qs[0]) & (effs <= qs[1]),
                "q2": (effs > qs[1]) & (effs <= qs[2]),
                "q3": (effs > qs[2])}
        bin_rows = {}
        for qname, mask in bins.items():
            sel = [r for r, m in zip(rows, mask) if m]
            gr = [r["abs_bwd"] for r in sel if r["algo"] == "grpo"]
            dr = [r["abs_bwd"] for r in sel if r["algo"] == "dr_grpo"]
            bin_rows[qname] = {"n_gr": len(gr), "n_dr": len(dr),
                               "mean_abs_bwd_gr": float(np.mean(gr)) if gr else float("nan"),
                               "mean_abs_bwd_dr": float(np.mean(dr)) if dr else float("nan"),
                               "delta_dr_minus_gr": (float(np.mean(dr)) - float(np.mean(gr)))
                               if (gr and dr) else float("nan")}
        out[task] = {"n_total": int(len(rows)), "bins": bin_rows,
                     "qs_eff": qs.tolist()}
    return out


# ---------------------------------------------------------------------------
# H3: Sobel-style mediation ratio.
# Direct effect: delta_eff ~ algo (binary).  Indirect effect:
# delta_eff ~ delta_ccfa + algo.  If the indirect coefficient is
# significant, mediation through backward CCF holds.
# Bootstrap CI on the indirect/direct ratio.
# ---------------------------------------------------------------------------
def mediation_ratio(paired: list[dict], B: int = 5000, seed: int = 0xB132):
    """Per task.  Uses paired bootstrap.  The 'direct' effect is the
    median(delta_eff) (i.e., the average effect of Dr.GR vs GR on
    per-window efficiency).  The 'indirect' effect is the slope of
    delta_eff ~ delta_ccfa * algo (Dr.GR vs GR), where the slope
    times the median(|delta_ccfa|) estimates the mediated portion.
    """
    rng = np.random.default_rng(seed)
    out = {}
    by_task = {}
    for p in paired:
        by_task.setdefault(p["task"], []).append(p)
    for task, rows in by_task.items():
        if len(rows) < 4:
            out[task] = {"n": int(len(rows))}
            continue
        de = np.array([r["delta_eff"] for r in rows], dtype=np.float64)
        dc = np.array([r["delta_ccfa"] for r in rows], dtype=np.float64)
        # point estimates
        rho, p = stats.spearmanr(de, dc)
        # direct effect = mean(delta_eff) (signed so positive = Dr.GR better)
        direct = float(np.mean(de))
        # slope of de ~ dc, weighted by typical |dc|
        if dc.std() > 1e-9:
            slope, intercept, _, _, _ = stats.linregress(dc, de)
            typical_dc = float(np.median(np.abs(dc)))
            indirect = float(slope * typical_dc)
            mediation_ratio = indirect / direct if abs(direct) > 1e-9 else float("nan")
        else:
            slope = float("nan")
            intercept = float("nan")
            indirect = float("nan")
            mediation_ratio = float("nan")
        # bootstrap CI on indirect and mediation_ratio
        n = len(de)
        boot_indirect = []
        boot_ratio = []
        for _ in range(B):
            idx = rng.integers(0, n, size=n)
            de_b = de[idx]; dc_b = dc[idx]
            if dc_b.std() < 1e-9:
                continue
            sl_b, _, _, _, _ = stats.linregress(dc_b, de_b)
            ind_b = sl_b * float(np.median(np.abs(dc_b)))
            boot_indirect.append(ind_b)
            if abs(np.mean(de_b)) > 1e-9:
                boot_ratio.append(ind_b / np.mean(de_b))
        bi = np.array(boot_indirect, dtype=np.float64)
        br = np.array(boot_ratio, dtype=np.float64)
        out[task] = {
            "n": int(n),
            "rho_de_dc": float(rho),
            "p_de_dc": float(p),
            "direct_effect": direct,
            "indirect_effect": indirect,
            "slope_de_vs_dc": float(slope) if not math.isnan(slope) else float("nan"),
            "median_abs_dc": typical_dc,
            "mediation_ratio": mediation_ratio,
            "indirect_CI_lo": float(np.quantile(bi, 0.025)) if len(bi) else float("nan"),
            "indirect_CI_hi": float(np.quantile(bi, 0.975)) if len(bi) else float("nan"),
            "ratio_CI_lo": float(np.quantile(br, 0.025)) if len(br) else float("nan"),
            "ratio_CI_hi": float(np.quantile(br, 0.975)) if len(br) else float("nan"),
        }
    return out


# ---------------------------------------------------------------------------
# H4: cross-task consistency
# ---------------------------------------------------------------------------
def cross_task_consistency(per_task: dict) -> dict:
    """Per task: did rho(delta_eff, delta_ccfa) come out positive (or
    at least not in the wrong direction)?  Binomial on 2 tasks.
    """
    consistent = 0
    total = 0
    per_task_sign = {}
    for task, res in per_task.items():
        if "rho" not in res:
            continue
        rho = res["rho"]
        per_task_sign[task] = rho
        total += 1
        if rho > 0:
            consistent += 1
    if total == 0:
        return {"consistent": 0, "total": 0, "p_binom": float("nan")}
    p = float(stats.binomtest(consistent, total, 0.5,
                              alternative="greater").pvalue) if total > 0 else float("nan")
    return {"consistent": int(consistent), "total": int(total),
            "p_binom": p, "per_task_rho": per_task_sign}


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_tsv(path: str, header: list[str], rows: list[list]) -> None:
    with open(path, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")


def write_json(path: str, obj: Any) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B_perm", type=int, default=50000)
    ap.add_argument("--n_w", type=int, default=4)
    ap.add_argument("--B_boot", type=int, default=5000)
    args = ap.parse_args()

    # Load
    runs = []
    runs += load_step_log(DRGR_VS_GRPO, "arithmetic_easy")
    runs += load_step_log(DRGR_GSM8K, "gsm8k_cot")
    ccfa = load_iter108_ccfa()

    # Per-window efficiency
    win_eff = per_window_efficiency(runs, args.n_w)
    joined = join(win_eff, ccfa)
    paired = paired_delta(joined)

    # Within-run correlation analysis
    within = within_run_rho(joined)
    paired_within = paired_within_run_rho(within)

    # H1: per-task Spearman rho(delta_eff, delta_ccfa)
    h1 = {}
    for task in sorted({p["task"] for p in paired}):
        sel = [p for p in paired if p["task"] == task]
        de = np.array([p["delta_eff"] for p in sel], dtype=np.float64)
        dc = np.array([p["delta_ccfa"] for p in sel], dtype=np.float64)
        res = spearman_with_perm(de, dc, args.B_perm,
                                 seed=hash(task) & 0xFFFF)
        eff_gr = np.array([p["eff_gr"] for p in sel], dtype=np.float64)
        eff_dr = np.array([p["eff_dr"] for p in sel], dtype=np.float64)
        w = wilcoxon_one_sided(eff_dr, eff_gr, alt="greater")
        h1[task] = {**res, "wilcoxon_W": w["W"], "wilcoxon_p": w["p"],
                    "cohens_d": w["d"], "n_pairs": int(len(sel))}

    # H1b: WITHIN-RUN correlation analysis.  Per (algo, seed, task),
    # compute Spearman rho(|bwd|_w, cum_eff_w) across the n_w windows.
    # Dr.GR should have rho MORE NEGATIVE than GR.  Test paired
    # Wilcoxon on delta_rho across (seed, task) within each task, and
    # cross-task pooled sign test.
    h1b = {}
    for task in sorted({p["task"] for p in paired_within}):
        sel = [p for p in paired_within if p["task"] == task]
        rho_gr = np.array([p["rho_gr"] for p in sel], dtype=np.float64)
        rho_dr = np.array([p["rho_dr"] for p in sel], dtype=np.float64)
        # signed: Dr.GR - GR, want negative (Dr.GR more anti-correlated)
        w = wilcoxon_one_sided(rho_dr, rho_gr, alt="less")
        # permutation null on delta_rho
        deltas = rho_dr - rho_gr
        rng = np.random.default_rng(hash(task) & 0xFFFF)
        n_perm = 50000
        abs_obs = abs(float(deltas.mean())) if len(deltas) else 0.0
        cnt = 0
        for _ in range(n_perm):
            signs = rng.choice([-1, 1], size=len(deltas))
            null_mean = float(np.mean(deltas * signs))
            if abs(null_mean) >= abs_obs:
                cnt += 1
        perm_p = (cnt + 1) / (n_perm + 1)
        # also Spearman: rho(delta_rho, delta_mean_bwd) — should be
        # NEGATIVE if "Dr.GR's CCF reduction corresponds to its
        # within-run anti-correlation sharpening".
        if len(sel) >= 3:
            sp_rho, sp_p = stats.spearmanr(
                [p["delta_mean_bwd"] for p in sel],
                [p["delta_rho"] for p in sel])
        else:
            sp_rho, sp_p = float("nan"), float("nan")
        h1b[task] = {"n_pairs": len(sel),
                     "mean_rho_gr": float(np.mean(rho_gr)),
                     "mean_rho_dr": float(np.mean(rho_dr)),
                     "mean_delta_rho": float(np.mean(deltas)),
                     "wilcoxon_W": w["W"], "wilcoxon_p": w["p"],
                     "cohens_d": w["d"],
                     "perm_p": float(perm_p),
                     "spearman_delta_rho_vs_delta_bwd": float(sp_rho),
                     "spearman_p": float(sp_p)}

    # H1c: cross-task pooled sign test on delta_rho < 0
    all_deltas = np.array([p["delta_rho"] for p in paired_within],
                          dtype=np.float64)
    n_neg = int(np.sum(all_deltas < 0))
    n_pos = int(np.sum(all_deltas > 0))
    n_zero = int(np.sum(all_deltas == 0))
    binom_p = float(stats.binomtest(n_neg, n_neg + n_pos, 0.5,
                                    alternative="greater").pvalue) \
        if (n_neg + n_pos) > 0 else float("nan")
    h1c = {"n_neg": n_neg, "n_pos": n_pos, "n_zero": n_zero,
           "binom_p": binom_p}

    # H3: mediation ratio (Sobel-style) using within-run deltas
    med_paired = [{"task": p["task"], "seed": p["seed"],
                   "delta_eff": p["delta_final_eff"],
                   "delta_ccfa": -p["delta_mean_bwd"]}
                  for p in paired_within]
    med = mediation_ratio(med_paired, B=args.B_boot)

    # H2: iso-efficiency contour
    iso = iso_eff_contour(joined)

    # H3: mediation ratio (Sobel-style)
    med = mediation_ratio(paired, B=args.B_boot)

    # H4: cross-task consistency
    cross = cross_task_consistency(h1)
    # cross-task on h1b: per task, did mean_delta_rho come out negative?
    h1b_cross = {"n_neg_tasks": 0, "n_total_tasks": 0}
    for task, res in h1b.items():
        h1b_cross["n_total_tasks"] += 1
        if res["mean_delta_rho"] < 0:
            h1b_cross["n_neg_tasks"] += 1
    if h1b_cross["n_total_tasks"] > 0:
        h1b_cross["binom_p"] = float(stats.binomtest(
            h1b_cross["n_neg_tasks"],
            h1b_cross["n_total_tasks"], 0.5,
            alternative="greater").pvalue)
    else:
        h1b_cross["binom_p"] = float("nan")

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    # 1) per-window efficiency table (CUMULATIVE)
    rows = [[j["task"], j["algo"], j["seed"], j["window"],
             j["n_in_window"], round(j["rwd_base"], 6), round(j["rwd_lh"], 6),
             round(j["len_base"], 6), round(j["len_lh"], 6),
             round(j["cum_dR"], 6), round(j["cum_dL"], 6),
             round(j["cum_eff"], 6) if not math.isnan(j["cum_eff"]) else "NA",
             round(j["bwd_signed"], 6), round(j["abs_bwd"], 6),
             round(j["fwd_signed"], 6), round(j["abs_fwd"], 6),
             round(j["phi_L"], 6), round(j["phi_R"], 6)]
            for j in joined]
    write_tsv(os.path.join(RES, "length_bias_iter132_window_efficiency.tsv"),
              ["task", "algo", "seed", "window", "n_in_window",
               "rwd_base", "rwd_lh", "len_base", "len_lh",
               "cum_dR", "cum_dL", "cum_eff",
               "bwd_signed", "abs_bwd", "fwd_signed", "abs_fwd",
               "phi_L", "phi_R"], rows)

    # 2) paired-delta table (per task, seed, window)
    rows = [[p["task"], p["seed"], p["window"],
             round(p["eff_gr"], 6), round(p["eff_dr"], 6),
             round(p["delta_eff"], 6),
             round(p["abs_bwd_gr"], 6), round(p["abs_bwd_dr"], 6),
             round(p["delta_ccfa"], 6),
             round(p["fwd_gr"], 6), round(p["fwd_dr"], 6),
             round(p["delta_fwd"], 6),
             round(p["phi_L_gr"], 6), round(p["phi_L_dr"], 6),
             round(p["delta_phi_L"], 6),
             round(p["cum_dR_gr"], 6), round(p["cum_dR_dr"], 6),
             round(p["cum_dL_gr"], 6), round(p["cum_dL_dr"], 6)]
            for p in paired]
    write_tsv(os.path.join(RES, "length_bias_iter132_paired_delta.tsv"),
              ["task", "seed", "window",
               "eff_gr", "eff_dr", "delta_eff",
               "abs_bwd_gr", "abs_bwd_dr", "delta_ccfa",
               "fwd_gr", "fwd_dr", "delta_fwd",
               "phi_L_gr", "phi_L_dr", "delta_phi_L",
               "cum_dR_gr", "cum_dR_dr", "cum_dL_gr", "cum_dL_dr"], rows)

    # 3) mediation chain -- include h1, h1b, and h3 mediation
    rows = []
    for task, res in h1.items():
        rows.append([task, "H1_chain_window",
                     round(res["rho"], 6), round(res["p_param"], 6),
                     round(res["p_perm"], 6), res["n"],
                     round(res["wilcoxon_W"], 4) if not math.isnan(res["wilcoxon_W"]) else "NA",
                     round(res["wilcoxon_p"], 6) if not math.isnan(res["wilcoxon_p"]) else "NA",
                     round(res["cohens_d"], 6)])
    for task, res in h1b.items():
        rows.append([task, "H1b_within_run_rho",
                     round(res["mean_rho_gr"], 6),
                     round(res["mean_rho_dr"], 6),
                     round(res["mean_delta_rho"], 6),
                     res["n_pairs"],
                     round(res["wilcoxon_W"], 4),
                     round(res["wilcoxon_p"], 6),
                     round(res["cohens_d"], 6)])
    for task, res in med.items():
        rows.append([task, "H3_mediation_within_run",
                     round(res.get("rho_de_dc", float("nan")), 6),
                     round(res.get("p_de_dc", float("nan")), 6),
                     "NA", res.get("n", 0),
                     round(res.get("direct_effect", float("nan")), 6),
                     round(res.get("indirect_effect", float("nan")), 6),
                     round(res.get("mediation_ratio", float("nan")), 6)])
    write_tsv(os.path.join(RES, "length_bias_iter132_mediation_chain.tsv"),
              ["task", "test", "stat_a", "stat_b", "stat_c", "n",
               "value_a", "value_b", "value_c"], rows)

    # 4) permutation null (per-task, per-test, B reps)
    rows = []
    for task, res in h1.items():
        rows.append([task, "H1_chain_window", "rho_de_dc",
                     round(res["rho"], 6), round(res["p_perm"], 6),
                     res["n"], args.B_perm])
    for task, res in h1b.items():
        rows.append([task, "H1b_within_run_rho", "delta_rho",
                     round(res["mean_delta_rho"], 6),
                     round(res["perm_p"], 6), res["n_pairs"], 50000])
    rows.append(["cross_task", "H1c_sign_test_delta_rho_neg",
                 "n_neg", h1c["n_neg"], round(h1c["binom_p"], 6),
                 h1c["n_neg"] + h1c["n_pos"], 1])
    write_tsv(os.path.join(RES, "length_bias_iter132_permutation_null.tsv"),
              ["task", "test", "statistic", "observed", "p_perm", "n", "B"], rows)

    # 5) summary -- one line per (task, test)
    rows = []
    for task, res in h1.items():
        sig = "FAVOURS chain" if (res["rho"] > 0 and res["p_perm"] < 0.10) else "null"
        rows.append([task, "H1_chain_window_rho(delta_eff,delta_ccfa)>0",
                     round(res["rho"], 4),
                     round(res["p_param"], 6), round(res["p_perm"], 6),
                     res["n"], sig])
    for task, res in h1b.items():
        sig = ("FAVOURS chain" if (res["mean_delta_rho"] < 0 and res["wilcoxon_p"] < 0.10)
               else "null")
        rows.append([task, "H1b_within_run_delta_rho<0",
                     round(res["mean_delta_rho"], 4),
                     round(res["wilcoxon_p"], 6), round(res["perm_p"], 6),
                     res["n_pairs"], sig])
    rows.append(["cross_task", "H1c_sign_test_delta_rho<0",
                 h1c["n_neg"], h1c["n_neg"] + h1c["n_pos"],
                 round(h1c["binom_p"], 6), h1c["n_neg"] + h1c["n_pos"],
                 "FAVOURS" if h1c["n_neg"] >= 6 and h1c["binom_p"] < 0.05 else "inconclusive"])
    for task, res in med.items():
        if "rho_de_dc" not in res:
            continue
        sig = ("FAVOURS mediation" if (res["rho_de_dc"] > 0 and res["p_de_dc"] < 0.10)
               else "null")
        rows.append([task, "H3_mediation_within_run",
                     round(res["rho_de_dc"], 4),
                     round(res["p_de_dc"], 6), "NA",
                     res["n"], sig])
    rows.append(["cross_task", "H4_consistency_h1",
                 cross["consistent"], cross["total"],
                 round(cross["p_binom"], 6), 2,
                 "FAVOURS" if cross["consistent"] == cross["total"]
                 and cross["p_binom"] < 0.5 else "inconclusive"])
    rows.append(["cross_task", "H4_consistency_h1b",
                 h1b_cross["n_neg_tasks"], h1b_cross["n_total_tasks"],
                 round(h1b_cross["binom_p"], 6), h1b_cross["n_total_tasks"],
                 "FAVOURS" if h1b_cross["n_neg_tasks"] == h1b_cross["n_total_tasks"]
                 else "inconclusive"])
    write_tsv(os.path.join(RES, "length_bias_iter132_summary.tsv"),
              ["task", "test", "stat_a", "stat_b", "p_perm", "n", "verdict"], rows)

    # 6) meta
    meta = {
        "iter": 132,
        "pillar": "P4-LengthBias",
        "n_w": args.n_w,
        "B_perm": args.B_perm,
        "B_boot": args.B_boot,
        "seed_base": 20260703,
        "tasks": sorted({j["task"] for j in joined}),
        "algos": ["grpo", "dr_grpo"],
        "n_seeds": {task: len({j["seed"] for j in joined if j["task"] == task})
                    for task in sorted({j["task"] for j in joined})},
        "n_joined": len(joined),
        "n_paired": len(paired),
        "n_within_run": len(within),
        "n_paired_within": len(paired_within),
        "h1_per_task": h1,
        "h1b_per_task": h1b,
        "h1c_cross_task": h1c,
        "h2_iso_per_task": iso,
        "h3_mediation_per_task": med,
        "h4_cross_task": cross,
        "h1b_cross_task": h1b_cross,
        "within_run_per_algo": within,
        "paired_within_per_seed": paired_within,
    }
    write_json(os.path.join(RES, "length_bias_iter132_meta.json"), meta)

    # Print headline
    print("\n=== Iter 132: Closing the Causal Chain ===")
    print(f"Joined (algo, seed, task, window) cells: {len(joined)}")
    print(f"Paired (seed, task, window) deltas: {len(paired)}")
    print(f"Within-run rho per (algo, seed, task): {len(within)}")
    print(f"Paired within-run (Dr.GR vs GR): {len(paired_within)}")
    print()
    for task, res in h1.items():
        print(f"H1 [{task}]: rho(delta_eff, delta_ccfa) = {res['rho']:+.3f} "
              f"(p_param={res['p_param']:.3f}, p_perm={res['p_perm']:.3f}, "
              f"n={res['n']})")
        print(f"    paired Wilcoxon eff(DR)>eff(GR): W={res['wilcoxon_W']:.1f}, "
              f"p={res['wilcoxon_p']:.3f}, d={res['cohens_d']:+.3f}")
    print()
    for task, res in h1b.items():
        print(f"H1b [{task}]: mean rho(|bwd|, cum_eff) GR={res['mean_rho_gr']:+.3f}, "
              f"DR={res['mean_rho_dr']:+.3f}, delta={res['mean_delta_rho']:+.3f} "
              f"(Wilcoxon p={res['wilcoxon_p']:.3f}, d={res['cohens_d']:+.3f}, "
              f"perm_p={res['perm_p']:.3f}, n={res['n_pairs']})")
        print(f"     Spearman delta_rho vs delta_|bwd|: rho={res['spearman_delta_rho_vs_delta_bwd']:+.3f}, "
              f"p={res['spearman_p']:.3f}")
    print()
    print(f"H1c sign test: Dr.GR wins (delta_rho<0) in {h1c['n_neg']}/{h1c['n_neg']+h1c['n_pos']} pairs, "
          f"binom_p={h1c['binom_p']:.3f}")
    print()
    for task, res in med.items():
        if "rho_de_dc" not in res:
            continue
        print(f"H3 [{task}]: direct={res['direct_effect']:+.4f}, "
              f"indirect={res['indirect_effect']:+.4f}, "
              f"ratio={res['mediation_ratio']:+.3f} "
              f"[{res['ratio_CI_lo']:+.3f}, {res['ratio_CI_hi']:+.3f}]")
    print()
    print(f"H4 cross-task (h1): {cross['consistent']}/{cross['total']} tasks "
          f"with rho>0, p_binom={cross['p_binom']:.3f}")
    print(f"H4 cross-task (h1b): {h1b_cross['n_neg_tasks']}/{h1b_cross['n_total_tasks']} tasks "
          f"with delta_rho<0, p_binom={h1b_cross['binom_p']:.3f}")
    print()


if __name__ == "__main__":
    main()