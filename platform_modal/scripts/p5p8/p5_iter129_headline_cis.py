#!/usr/bin/env python3
"""
P5 iter-129 — Headline-CI audit (fresh vein, not in 138 prior rows).

Closes brief vein (c): "bootstrap CIs on every P5 headline number (reuse
scripts/berkeley/adding_error_bars_to_evals.py)". Iter-123 did this for
P7 (19 headlines) and set the template; iter-129 does the same for P5.

P5 headlines (15 numerical point estimates across paper/sections/p5_*.tex):
  H01  iter-85: eta^2(algo, mean over 6 channels) = 0.0331 (N2 four-method)
  H02  iter-85: eta^2(algo, zvf) = 0.0454 on N2 four-method
  H03  iter-85: eta^2(algo, pcd) = 0.0357 on N2 four-method
  H04  iter-85: eta^2(algo, loss) = 0.9867 (positive control)
  H05  iter-85: Cohen's d on zvf = +1.899 (GIFT vs other 3, last-10 pooled)
  H06  iter-85: Cohen's d on pcd = -1.605 (GIFT vs other 3, last-10 pooled)
  H07  iter-89: eta^2 bootstrap UB on zvf = 0.113 (exceeds Ivison 0.05)
  H08  iter-101: eta^2(algo, zvf_risk) = 0.763 on 9-method zvf130 panel
  H09  iter-101: eta^2(seed, zvf_risk) = 0.0071 (seed-axis control)
  H10  iter-101: SCAFGRPO rel_drop = -0.1042 on zvf_risk (LOMO)
  H11  iter-125: chained R(zvf, task_slice) = 10.32 [mega vs N2 eta^2 ratio]
  H12  iter-125: chained R(pcd, task_slice) = 12.62
  H13  iter-125: chained R(zvf, G) = 9.77
  H14  iter-125: chained R(pcd, G) = 6.45
  H15  iter-121: M1+M2 blind-spot rate = 0.0% detection (196 mutations)

Bootstrap: B=2000, percentile method, paired where the underlying
design demands (paired-step for N2 = 160 paired diffs; paired-cell for
mega = 98 paired diffs; paired-seed for zvf130 = 5 paired seeds).

Verdicts:
  PASS — recomputed CI contains the published point estimate.
  TENSION — recomputed CI does NOT contain the published point estimate
             but does contain a plausible alternative within ±2 SE.
  REPORTED — point estimate is structural or single-seed (cannot bootstrap).
  INSUFFICIENT_N — n < 4 effective paired diffs.

Outputs:
  experiments/results/p5p8/p5_iter129_headline_cis.tsv     (15 rows)
  experiments/results/p5p8/p5_iter129_headline_cis.json    (per-class tally)
"""
from __future__ import annotations
import csv
import json
import math
import random
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
N2_TSV   = ROOT / "experiments/results/n2_reward_tensor_resume/n2_metrics.tsv"
ZV130    = ROOT / "experiments/results/zvf_iter130_risk_index.tsv"
MEGA     = ROOT / "experiments/results/mega_20260704/cells.tsv"
OUT_DIR  = ROOT / "experiments/results/p5p8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260705
B = 2000
ALPHA = 0.05

# ---- bootstrap primitives (from adding_error_bars_to_evals.py) ----

def bootstrap_ci_paired(diff_values, B=B, alpha=ALPHA, seed=SEED):
    n = len(diff_values)
    if n < 2:
        return float("nan"), float("nan"), n
    rng = random.Random(seed)
    boot = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        sub = [diff_values[i] for i in idx]
        boot.append(sum(sub) / n)
    boot.sort()
    return boot[int(B * alpha / 2)], boot[int(B * (1 - alpha / 2))], n


def bootstrap_ci_mean(values, B=B, alpha=ALPHA, seed=SEED):
    n = len(values)
    if n < 2:
        return float("nan"), float("nan"), n
    rng = random.Random(seed)
    boot = []
    for _ in range(B):
        s = sum(rng.choice(values) for _ in range(n)) / n
        boot.append(s)
    boot.sort()
    return boot[int(B * alpha / 2)], boot[int(B * (1 - alpha / 2))], n


def bootstrap_ci_eta2(groups, B=B, alpha=ALPHA, seed=SEED):
    """Bootstrap CI on eta^2 = SS_axis / SS_total for a categorical axis.
    groups: list of value arrays, one per category.

    Resampling strategy: WITHIN each group (preserves group structure) so
    that the eta^2 decomposition is recomputable from each resample.
    """
    all_vals = [v for g in groups for v in g]
    grand_mean = sum(all_vals) / len(all_vals)
    SS_total = sum((v - grand_mean) ** 2 for v in all_vals)
    if SS_total <= 0:
        return float("nan"), float("nan"), float("nan")
    K = len(groups)
    n_per = [len(g) for g in groups]
    SS_axis = sum(n * (sum(g) / n - grand_mean) ** 2
                  for g, n in zip(groups, n_per))
    eta2 = SS_axis / SS_total
    boot = []
    rng = random.Random(seed)
    for _ in range(B):
        # Resample WITHIN each group (preserves group identity & sizes)
        new_groups = []
        for g in groups:
            ng = len(g)
            new_groups.append([g[rng.randrange(ng)] for _ in range(ng)])
        ng_all = [v for g in new_groups for v in g]
        ng_mean = sum(ng_all) / len(ng_all)
        ng_total = sum((v - ng_mean) ** 2 for v in ng_all)
        if ng_total <= 0:
            continue
        ng_axis = sum(len(g) * (sum(g) / len(g) - ng_mean) ** 2
                      for g in new_groups)
        boot.append(ng_axis / ng_total)
    if len(boot) < 100:
        return eta2, float("nan"), float("nan")
    boot.sort()
    return eta2, boot[int(B * alpha / 2)], boot[int(B * (1 - alpha / 2))]


def cohens_d_paired(a, b):
    """Cohen's d for paired samples: mean(diff)/std(diff)."""
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return float("nan")
    diff = [a[i] - b[i] for i in range(len(a))]
    m = sum(diff) / len(diff)
    var = sum((d - m) ** 2 for d in diff) / (len(diff) - 1) if len(diff) > 1 else 0.0
    return m / math.sqrt(var) if var > 1e-12 else float("nan")


# ---- loaders ----

def load_n2():
    """Load N2 four-method per-step metrics.
    Returns dict: {(method, step): {zvf, pcd, ..., loss}}."""
    out = {}
    with open(N2_TSV) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            key = (r["method"], int(r["step"]))
            out[key] = {
                "zvf": float(r["zvf"]),
                "pcd": float(r["pcd"]),
                "larq": float(r["larq"]) if r["larq"] != "nan" else float("nan"),
                "reward_mean": float(r["reward_mean"]),
                "mean_len": float(r["mean_len"]),
                "cv_len": float(r["cv_len"]),
                "loss": float(r["loss"]),
            }
    return out


def load_zvf130_methods():
    """Load zvf130 method panel (5 seeds × 9 methods = 45 rows)."""
    out = defaultdict(dict)
    with open(ZV130) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            method = r["method"]
            seed = r["seed"]
            if seed == "agg":
                continue
            try:
                out[method][int(seed)] = {
                    "zvf_risk": float(r["zvf_risk"]),
                    "mean_zvf": float(r["mean_zvf"]),
                    "mag": float(r["risk_mag"]),
                    "csd": float(r["risk_csd"]),
                }
            except (ValueError, KeyError):
                continue
    return out


def load_mega_cells():
    """Load cells.tsv as a list of dicts."""
    out = []
    with open(MEGA) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            try:
                out.append({
                    "cell_id": r["cell_id"],
                    "model_family": r["model_family"],
                    "task_slice": r["task_slice"],
                    "G": int(r["G"]),
                    "temperature": float(r["temperature"]),
                    "seed": int(r["seed"]),
                    "zvf": float(r["zvf"]),
                    "pcd": float(r["pcd"]),
                    "mean_reward": float(r["mean_reward"]),
                })
            except (ValueError, KeyError):
                continue
    return out


# ---- headline computations ----

def compute_n2_eta2_per_channel(n2):
    """Returns {channel: eta2} for the N2 four-method algorithm axis."""
    methods = ["grpo", "aero", "gift", "areal"]
    channels = ["zvf", "pcd", "larq", "reward_mean", "mean_len", "cv_len", "loss"]
    out = {}
    steps = sorted({k[1] for k in n2.keys()})
    for ch in channels:
        groups = []
        for m in methods:
            vals = [n2[(m, s)][ch] for s in steps if not math.isnan(n2[(m, s)][ch])]
            groups.append(vals)
        eta2, lo, hi = bootstrap_ci_eta2(groups)
        out[ch] = (eta2, lo, hi, len(steps))
    return out


def compute_n2_cohens_d(n2):
    """Cohen's d for GIFT vs other 3 methods on zvf and pcd, last-10 pooled.
    Returns dict with point + bootstrap CI (resample which steps are 'last-10')."""
    methods_other = ["grpo", "aero", "areal"]
    steps = sorted({k[1] for k in n2.keys()})
    last10 = [s for s in steps if s >= max(steps) - 9]
    # GIFT samples
    gift_zvf = [n2[("gift", s)]["zvf"] for s in last10]
    gift_pcd = [n2[("gift", s)]["pcd"] for s in last10]
    other_zvf = [n2[(m, s)]["zvf"] for m in methods_other for s in last10]
    other_pcd = [n2[(m, s)]["pcd"] for m in methods_other for s in last10]
    # pooled variance
    def pooled_cd(g, o):
        m_g = sum(g) / len(g); m_o = sum(o) / len(o)
        var_g = sum((x - m_g) ** 2 for x in g) / (len(g) - 1)
        var_o = sum((x - m_o) ** 2 for x in o) / (len(o) - 1)
        sp = math.sqrt(((len(g) - 1) * var_g + (len(o) - 1) * var_o)
                        / (len(g) + len(o) - 2))
        return (m_g - m_o) / sp if sp > 1e-12 else float("nan")
    point_zvf = pooled_cd(gift_zvf, other_zvf)
    point_pcd = pooled_cd(gift_pcd, other_pcd)
    # Bootstrap CI: resample the step index for both groups, preserving
    # the per-method structure of 'other' (so resample 'other' as a whole
    # 36-vector)
    rng = random.Random(SEED)
    boot_zvf = []
    boot_pcd = []
    for _ in range(B):
        # resample gift steps with replacement
        gz = [gift_zvf[rng.randrange(len(gift_zvf))] for _ in range(len(gift_zvf))]
        gp = [gift_pcd[rng.randrange(len(gift_pcd))] for _ in range(len(gift_pcd))]
        # resample other rows
        oz = [other_zvf[rng.randrange(len(other_zvf))] for _ in range(len(other_zvf))]
        op = [other_pcd[rng.randrange(len(other_pcd))] for _ in range(len(other_pcd))]
        if len(set(gz)) < 2 or len(set(oz)) < 2:
            continue
        boot_zvf.append(pooled_cd(gz, oz))
        boot_pcd.append(pooled_cd(gp, op))
    boot_zvf.sort(); boot_pcd.sort()
    zvf_lo = boot_zvf[int(len(boot_zvf) * ALPHA / 2)] if len(boot_zvf) >= 100 else point_zvf
    zvf_hi = boot_zvf[int(len(boot_zvf) * (1 - ALPHA / 2))] if len(boot_zvf) >= 100 else point_zvf
    pcd_lo = boot_pcd[int(len(boot_pcd) * ALPHA / 2)] if len(boot_pcd) >= 100 else point_pcd
    pcd_hi = boot_pcd[int(len(boot_pcd) * (1 - ALPHA / 2))] if len(boot_pcd) >= 100 else point_pcd
    return {
        "zvf": point_zvf, "zvf_lo": zvf_lo, "zvf_hi": zvf_hi,
        "pcd": point_pcd, "pcd_lo": pcd_lo, "pcd_hi": pcd_hi,
        "n_gift": len(gift_zvf),
        "n_other": len(other_zvf),
    }


def compute_zvf130_eta2(zvf130):
    """eta^2(algo, zvf_risk) on the 9-method zvf130 panel."""
    methods = [m for m, seeds in zvf130.items() if len(seeds) >= 3]
    groups = []
    for m in methods:
        vals = [s["zvf_risk"] for s in zvf130[m].values()]
        groups.append(vals)
    eta2, lo, hi = bootstrap_ci_eta2(groups)
    return eta2, lo, hi, methods


def compute_zvf130_seed_eta2(zvf130):
    """eta^2(seed, zvf_risk) within GRPO (5 seeds, the seed-axis control)."""
    if "grpo" not in zvf130:
        return float("nan"), float("nan"), float("nan")
    # We only have one method (GRPO); eta^2 = 0 by construction.
