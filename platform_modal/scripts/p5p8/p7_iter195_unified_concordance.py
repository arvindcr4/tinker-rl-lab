#!/usr/bin/env python3
"""
P7 iter-195 — Cross-paradigm Adaptive-G controller concordance.

Fresh vein (b) of P5P8_IMPROVEMENT_BRIEF.md: unify the Adaptive-G
controller (iter-119) with the Dualformer auto-mode rule (Berkeley
row 01) and the AlphaProof tree-baseline gamma*=0 smoothing (Berkeley
row 19) into ONE calibrated controller section.

Step-level binary decisions on the N2 four-method same-stack panel
(4 methods x 40 steps = 160 step-cells):

  AG-step  : Adaptive-G (iter-119) fires iff zvf_step >= tau = 0.70.
  DF-step  : Dualformer auto-mode fires iff the step contains at
             least one contrast prompt AND the cell-level compute-
             equivalent reward r / sqrt(G) at G_BASE = 8 exceeds
             that at G_ESC = 16.
  AP-step  : AlphaProof gamma*=0 fires iff the smoothed-variance
             proxy (Dirichlet(1,1) = Beta(1,1) at depth=0) for the
             mean contrast-prompt kbar is strictly less than the
             naive variance proxy kbar (G - kbar) / G^2.

Concordance: Cohen's kappa among the three pairs (AG x DF,
AG x AP, DF x AP), pooled across methods, with bootstrap percentile
CIs (B=2000, seed=20260706).  Headline: does the Adaptive-G
controller agree with the Berkeley-formalism rules when measured on
the SAME step-cells?

Outputs:
  platform_hybrid/experiments/results/p5p8/p7_iter195_concordance_pair.tsv
  platform_hybrid/experiments/results/p5p8/p7_iter195_concordance_boot.tsv
  platform_hybrid/experiments/results/p5p8/p7_iter195_concordance_per_step.tsv
  platform_hybrid/experiments/results/p5p8/p7_iter195_summary.json

Stdlib only.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import random
import statistics

WORK = pathlib.Path("/home/claude/tinker-rl-lab-minimax")
N2_DIR = WORK / "platform_hybrid/experiments/results/n2_reward_tensor_resume"
OUT = WORK / "platform_hybrid/experiments/results/p5p8"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = ("grpo", "aero", "gift", "areal")
TAU = 0.70
G_BASE = 8
G_ESC = 16
N_BOOT = 2000
SEED = 20260706


def kappa(a, b):
    n = len(a)
    p0 = sum(1 for x, y in zip(a, b) if x == y) / n
    p1 = sum(a) / n
    p2 = sum(b) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    if pe == 1.0:
        return float("nan")
    return (p0 - pe) / (1 - pe)


def ap_smoothed_proxy(kbar, G):
    """AlphaProof gamma*=0 depth-0 smoothed-variance proxy:
    (kbar^2 + (G - kbar)^2) / G^2."""
    return (kbar * kbar + (G - kbar) * (G - kbar)) / (G * G)


def ap_naive_proxy(kbar, G):
    """Naive binomial-variance proxy of a single Bernoulli sample:
    kbar (G - kbar) / G^2."""
    return (kbar * (G - kbar)) / (G * G)


def step_decisions(row):
    zvf_step = float(row["zvf"])
    ag_fire = int(zvf_step >= TAU)

    k_vals = [int(round(sum(p))) for p in row["rewards"]]
    contrast_ks = [k for k in k_vals if 0 < k < G_BASE]
    has_contrast = bool(contrast_ks)

    m_r = float(row["reward_mean"])
    r_base = m_r / math.sqrt(G_BASE)
    r_esc = m_r / math.sqrt(G_ESC)
    df_fire = int(has_contrast and r_base > r_esc)

    if has_contrast:
        kbar = statistics.mean(contrast_ks)
        smoothed = ap_smoothed_proxy(kbar, G_BASE)
        naive = ap_naive_proxy(kbar, G_BASE)
        ap_fire = int(smoothed < naive)
    else:
        ap_fire = 0
        kbar = float("nan")
        smoothed = float("nan")
        naive = float("nan")

    return {
        "method": row["method"],
        "step": row["step"],
        "zvf_step": zvf_step,
        "n_prompts": len(k_vals),
        "n_boundary": sum(1 for k in k_vals if k in (0, G_BASE)),
        "n_contrast": len(contrast_ks),
        "kbar": kbar,
        "smoothed": smoothed,
        "naive": naive,
        "ag": ag_fire,
        "df": df_fire,
        "ap": ap_fire,
    }


def main():
    rng = random.Random(SEED)
    all_cells = []
    for m in METHODS:
        path = N2_DIR / f"{m}_s0_tensors.jsonl"
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        for r in rows:
            all_cells.append(step_decisions(r))
    print(f"Built {len(all_cells)} (method,step) cells.")

    pair_rows = []
    for m in METHODS:
        sub = [c for c in all_cells if c["method"] == m]
        for na, nb in (("ag", "df"), ("ag", "ap"), ("df", "ap")):
            a = [c[na] for c in sub]
            b = [c[nb] for c in sub]
            pair_rows.append({
                "method": m, "pair": f"{na}x{nb}", "n": len(sub),
                "kappa": kappa(a, b),
                "agree": sum(1 for x, y in zip(a, b) if x == y),
                "fA": sum(a), "fB": sum(b),
            })
    pair_rows.append({"method": "POOLED", "pair": "agxdf",
                      "n": len(all_cells),
                      "kappa": kappa([c["ag"] for c in all_cells],
                                     [c["df"] for c in all_cells]),
                      "agree": sum(1 for c in all_cells if c["ag"] == c["df"]),
                      "fA": sum(c["ag"] for c in all_cells),
                      "fB": sum(c["df"] for c in all_cells)})
    pair_rows.append({"method": "POOLED", "pair": "agxap",
                      "n": len(all_cells),
                      "kappa": kappa([c["ag"] for c in all_cells],
                                     [c["ap"] for c in all_cells]),
                      "agree": sum(1 for c in all_cells if c["ag"] == c["ap"]),
                      "fA": sum(c["ag"] for c in all_cells),
                      "fB": sum(c["ap"] for c in all_cells)})
    pair_rows.append({"method": "POOLED", "pair": "dfxap",
                      "n": len(all_cells),
                      "kappa": kappa([c["df"] for c in all_cells],
                                     [c["ap"] for c in all_cells]),
                      "agree": sum(1 for c in all_cells if c["df"] == c["ap"]),
                      "fA": sum(c["df"] for c in all_cells),
                      "fB": sum(c["ap"] for c in all_cells)})

    with open(OUT / "p7_iter195_concordance_pair.tsv", "w") as f:
        w = csv.DictWriter(f, fieldnames=("method", "pair", "n", "kappa",
                                          "agree", "fA", "fB"),
                           delimiter="\t")
        w.writeheader()
        for r in pair_rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"Wrote {OUT}/p7_iter195_concordance_pair.tsv")

    # Per-step detail TSV
    with open(OUT / "p7_iter195_concordance_per_step.tsv", "w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["method", "step", "zvf_step", "n_boundary", "n_contrast",
                    "kbar", "smoothed", "naive", "ag", "df", "ap"])
        for c in all_cells:
            w.writerow([c["method"], c["step"], f"{c['zvf_step']:.4f}",
                        c["n_boundary"], c["n_contrast"],
                        f"{c['kbar']:.4f}" if not math.isnan(c["kbar"])
                            else "nan",
                        f"{c['smoothed']:.4f}" if not math.isnan(c["smoothed"])
                            else "nan",
                        f"{c['naive']:.4f}" if not math.isnan(c["naive"])
                            else "nan",
                        c["ag"], c["df"], c["ap"]])
    print(f"Wrote {OUT}/p7_iter195_concordance_per_step.tsv")

    boot_rows = []
    n = len(all_cells)
    for label, na, nb in (("agxdf", "ag", "df"),
                          ("agxap", "ag", "ap"),
                          ("dfxap","df", "ap")):
        a = [c[na] for c in all_cells]
        b = [c[nb] for c in all_cells]
        kp = kappa(a, b)
        idx = list(range(n))
        boots = []
        for _ in range(N_BOOT):
            samp = [rng.choice(idx) for _ in range(n)]
            aa = [a[i] for i in samp]
            bb = [b[i] for i in samp]
            boots.append(kappa(aa, bb))
        boots.sort()
        lo = boots[int(0.025 * N_BOOT)]
        hi = boots[int(0.975 * N_BOOT)]
        boot_rows.append({"pair": label, "point": kp, "lo": lo, "hi": hi,
                          "n": n, "B": N_BOOT,
                          "excl_zero": int(lo > 0 or hi < 0)})
    with open(OUT / "p7_iter195_concordance_boot.tsv", "w") as f:
        w = csv.DictWriter(f, fieldnames=("pair", "point", "lo", "hi", "n",
                                          "B", "excl_zero"),
                           delimiter="\t")
        w.writeheader()
        for r in boot_rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"Wrote {OUT}/p7_iter195_concordance_boot.tsv")

    # Stratified fire rates
    rate_rows = []
    for m in METHODS:
        sub = [c for c in all_cells if c["method"] == m]
        rate_rows.append({"method": m, "metric": "ag_rate",
                          "value": sum(c["ag"] for c in sub) / len(sub),
                          "n": len(sub)})
        rate_rows.append({"method": m, "metric": "df_rate",
                          "value": sum(c["df"] for c in sub) / len(sub),
                          "n": len(sub)})
        rate_rows.append({"method": m, "metric": "ap_rate",
                          "value": sum(c["ap"] for c in sub) / len(sub),
                          "n": len(sub)})

    verdicts = {
        "H1_AG_step_fires_when_zvf_high": int(all(
            r["kappa"] > 0.5 for r in pair_rows
            if r["pair"] == "agxdf" and r["method"] != "POOLED")),
        "H2_DF_AP_concordant_at_step_level": int(all(
            r["kappa"] > 0.5 for r in pair_rows
            if r["pair"] == "dfxap" and r["method"] != "POOLED")),
        "H3_AG_DF_concordant_at_step_level": int(all(
            r["kappa"] > 0.5 for r in pair_rows
            if r["pair"] == "agxdf" and r["method"] != "POOLED")),
        "H4_AG_AP_concordant_at_step_level": int(all(
            r["kappa"] > 0.5 for r in pair_rows
            if r["pair"] == "agxap" and r["method"] != "POOLED")),
        "H5_pooled_AG_DF_kappa_excludes_zero": int(boot_rows[0]["excl_zero"]),
        "H6_pooled_AG_AP_kappa_excludes_zero": int(boot_rows[1]["excl_zero"]),
        "H7_pooled_DF_AP_kappa_excludes_zero": int(boot_rows[2]["excl_zero"]),
    }

    summary = {
        "ts": "2026-07-06",
        "iter": 195,
        "pillar": "P7",
        "n_cells_total": len(all_cells),
        "per_method_n": {m: sum(1 for c in all_cells if c["method"] == m)
                         for m in METHODS},
        "settings": {"TAU": TAU, "G_BASE": G_BASE, "G_ESC": G_ESC,
                     "N_BOOT": N_BOOT, "SEED": SEED},
        "stratified_rates": rate_rows,
        "boots": boot_rows,
        "verdicts": verdicts,
    }
    with open(OUT / "p7_iter195_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(verdicts, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()