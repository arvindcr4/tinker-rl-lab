#!/usr/bin/env python3
"""Iter 51 — Pillar 3: Reward-vs-G curves, broader-scale G=4~G=32 test, Wu 2025 fit.

Inputs (read-only):
  experiments/results/group_size_token_normalized.tsv   (4 budgets x 5 G)
  experiments/results/groupsize_zvf_sweep.tsv          (4 G rows from n=3 seeds)
  experiments/results/group_size_iter43_summary.tsv     (Wu retention at 4 budgets)

Outputs (TSVs, written to experiments/results/):
  group_size_iter51_reward_vs_G.tsv          reward/G curve per budget
  group_size_iter51_broader_tost.tsv         G=4 vs G=32 TOST at 4 budgets x 4 epsilons
  group_size_iter51_peak_shift.tsv            argmax G as fn of budget
  group_size_iter51_wu_loglinear.tsv          per-budget log-linear fit + Wu 2025 intersect
  group_size_iter51_lit_compare.tsv           Wu 2025 G=2~G=16 vs our G=4~G=32 retention
  group_size_iter51_summary.tsv               headline rollup
"""
from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
SEED = 20240702
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------


def load_token_norm():
    """Return list of dicts: {T, G, acc, ci_lo, ci_hi, gu}."""
    out = []
    with open(RES / "group_size_token_normalized.tsv") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            out.append(
                {
                    "T": int(row["budget_tokens"]),
                    "G": int(row["G"]),
                    "acc": float(row["heldout_acc_mean"]),
                    "ci_lo": float(row["heldout_acc_ci_low"]),
                    "ci_hi": float(row["heldout_acc_ci_high"]),
                    "gu": float(row["gu_estimate"]),
                }
            )
    return out


def load_zvf_sweep():
    out = []
    with open(RES / "groupsize_zvf_sweep.tsv") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            out.append(
                {
                    "G": int(row["G"]),
                    "n_seeds": int(row["n_seeds"]),
                    "acc": float(row["heldout_acc_mean"]),
                    "acc_se": float(row["heldout_acc_se"]),
                    "last10": float(row["last10_mean"]),
                    "mean_zvf": float(row["mean_zvf"]),
                    "zvf_th": float(row["zvf_theory_at_mean_p"]),
                }
            )
    return out


def write_tsv(path, header, rows):
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(h, "")) for h in header) + "\n")


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


def bootstrap_ci(samples, stat_fn, B=2000, alpha=0.05, seed=SEED):
    """Percentile bootstrap CI."""
    rng = np.random.default_rng(seed)
    n = len(samples)
    stats = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        stats.append(stat_fn([samples[i] for i in idx]))
    stats = sorted(stats)
    lo = stats[int(B * alpha / 2)]
    hi = stats[min(int(B * (1 - alpha / 2)), B - 1)]
    return lo, hi


def bootstrap_diff_ci(a, b, stat_fn=lambda x: x[0] - x[1], B=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = min(len(a), len(b))
    a = np.array(a)
    b = np.array(b)
    diffs = []
    for _ in range(B):
        ai = rng.integers(0, len(a), len(a))
        bi = rng.integers(0, len(b), len(b))
        diffs.append(stat_fn([a[ai].mean() - b[bi].mean()]))
    diffs = sorted(diffs)
    lo = diffs[int(B * 0.025)]
    hi = diffs[min(int(B * 0.975), B - 1)]
    return lo, hi


# ---------------------------------------------------------------------------
# A. Reward vs G curve
# ---------------------------------------------------------------------------


def reward_vs_G(rows):
    out = []
    budgets = sorted({r["T"] for r in rows})
    for T in budgets:
        cell = [r for r in rows if r["T"] == T]
        cell_sorted = sorted(cell, key=lambda r: r["G"])
        peak = max(cell_sorted, key=lambda r: r["acc"])
        for r in cell_sorted:
            out.append(
                {
                    "T_tokens": T,
                    "G": r["G"],
                    "acc": r["acc"],
                    "ci_lo": r["ci_lo"],
                    "ci_hi": r["ci_hi"],
                    "gu": r["gu"],
                    "is_peak": "yes" if r is peak else "no",
                    "peak_G": peak["G"],
                    "peak_acc": peak["acc"],
                }
            )
    return out


# ---------------------------------------------------------------------------
# B. Broader-scale TOST G=4 vs G=32
# ---------------------------------------------------------------------------


def broaden_tost(rows):
    out = []
    budgets = sorted({r["T"] for r in rows})
    eps_list = [0.02, 0.05, 0.10, 0.20]
    # The Wu 2025 claim is that G=4 is within 97.6% of G=32 ⇒ absolute diff < 0.024
    # on the [0,1] acc scale.  We test at multiple epsilons.
    for T in budgets:
        g4 = next(r for r in rows if r["T"] == T and r["G"] == 4)
        g32 = next(r for r in rows if r["T"] == T and r["G"] == 32)
        diff = g4["acc"] - g32["acc"]  # negative ⇒ G=4 worse
        # Bootstrap CI on the cell-level diff using reported CI half-width as sigma
        # Treat each cell as a Gaussian with mean=acc, sd=half-CI
        sigma4 = max((g4["ci_hi"] - g4["ci_lo"]) / 3.92, 1e-3)
        sigma32 = max((g32["ci_hi"] - g32["ci_lo"]) / 3.92, 1e-3)
        rng = np.random.default_rng(SEED + T)
        n = 1000
        a_samp = rng.normal(g4["acc"], sigma4, n)
        b_samp = rng.normal(g32["acc"], sigma32, n)
        diff_samp = a_samp - b_samp
        lo, hi = np.percentile(diff_samp, [2.5, 97.5])
        retention = g4["acc"] / g32["acc"] if g32["acc"] > 0 else float("nan")
        for eps in eps_list:
            # TOST: equivalent iff (lo > -eps) and (hi < eps)
            tost_pass = (lo > -eps) and (hi < eps)
            out.append(
                {
                    "T_tokens": T,
                    "G_a": 4,
                    "G_b": 32,
                    "G4_acc": g4["acc"],
                    "G32_acc": g32["acc"],
                    "abs_diff": diff,
                    "retention": retention,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "epsilon": eps,
                    "tost_equivalent": "yes" if tost_pass else "no",
                }
            )
    return out


# ---------------------------------------------------------------------------
# C. Peak shift
# ---------------------------------------------------------------------------


def peak_shift(rows):
    out = []
    budgets = sorted({r["T"] for r in rows})
    for T in budgets:
        cell = [r for r in rows if r["T"] == T]
        peak = max(cell, key=lambda r: r["acc"])
        # Compute Kendall's tau between G and acc on log scale
        gs = np.array([c["G"] for c in cell], dtype=float)
        ac = np.array([c["acc"] for c in cell], dtype=float)
        # Pearson on log G vs acc
        if len(gs) > 1:
            r_log = float(np.corrcoef(np.log(gs), ac)[0, 1])
        else:
            r_log = float("nan")
        out.append(
            {
                "T_tokens": T,
                "argmax_G": peak["G"],
                "argmax_acc": peak["acc"],
                "G4_acc": next(c["acc"] for c in cell if c["G"] == 4),
                "G32_acc": next(c["acc"] for c in cell if c["G"] == 32),
                "r_logG_acc": r_log,
                "direction": "increasing" if r_log > 0 else "decreasing",
            }
        )
    return out


# ---------------------------------------------------------------------------
# D. Wu 2025 log-linear fit
# ---------------------------------------------------------------------------


def wu_loglinear(rows):
    """For each (G, budget), fit acc = a*log10(T) + b; report slope+intercept.

    Then solve log10(T) at which G=4 retention = 0.976 ⇒ acc_G4 = 0.976 * acc_G32.
    """
    out = []
    Gs = sorted({r["G"] for r in rows})
    for G in Gs:
        cell = [r for r in rows if r["G"] == G]
        cell = sorted(cell, key=lambda r: r["T"])
        if len(cell) < 2:
            continue
        x = np.log10([c["T"] for c in cell])
        y = np.array([c["acc"] for c in cell])
        # Simple OLS
        n = len(x)
        sx, sy = x.sum(), y.sum()
        sxx, sxy = (x * x).sum(), (x * y).sum()
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        intercept = (sy - slope * sx) / n
        # R^2
        yhat = slope * x + intercept
        ss_res = float(((y - yhat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        # Solve for T at acc = 0.5, 0.7, 0.8, 0.9
        for tgt in [0.5, 0.7, 0.8, 0.9]:
            if slope != 0:
                log10T = (tgt - intercept) / slope
                Tstar = 10**log10T
            else:
                Tstar = float("nan")
            out.append(
                {
                    "G": G,
                    "slope": slope,
                    "intercept": intercept,
                    "R2": r2,
                    "target_acc": tgt,
                    "log10T_at_target": log10T if slope != 0 else float("nan"),
                    "T_at_target_M": Tstar / 1e6 if slope != 0 else float("nan"),
                }
            )
    return out


# ---------------------------------------------------------------------------
# E. Literature comparison: Wu 2025 G=2~G=16 vs our G=4~G=32
# ---------------------------------------------------------------------------


def literature_compare(rows):
    """The Wu 2025 paper (arXiv:2510.00977) reports that G=2 and G=16 produce
    equivalent models at 97.6% retention.  Our broader-scale data tests the
    analogous G=4 vs G=32 claim.

    For each budget T, compute the analogous retention R(G=4, T) = acc(G=4,T)/acc(G=32,T)
    alongside the Wu 97.6% anchor.
    """
    out = []
    budgets = sorted({r["T"] for r in rows})
    for T in budgets:
        g4 = next(r for r in rows if r["T"] == T and r["G"] == 4)
        g32 = next(r for r in rows if r["T"] == T and r["G"] == 32)
        # Wu 2025 G=2~G=16 retention would be measured on a 0.97-bounded cell — we
        # can't read it directly, but the 97.6% headline is in the brief.  Compare.
        retention_4_32 = g4["acc"] / g32["acc"]
        # The Wu 2025 retention was on GSM8K, ours is on a separate cell; the claim
        # is the same threshold (97.6%) tested on a different G_a/G_b pair.
        out.append(
            {
                "T_tokens": T,
                "wu_2025_paper": "arXiv:2510.00977",
                "wu_2025_Ga_over_Gb": "2 / 16",
                "wu_2025_retention": 0.976,
                "ours_Ga_over_Gb": "4 / 32",
                "ours_Ga_acc": g4["acc"],
                "ours_Gb_acc": g32["acc"],
                "ours_retention": retention_4_32,
                "above_wu_threshold": "yes" if retention_4_32 >= 0.976 else "no",
                "gap_to_wu": retention_4_32 - 0.976,
            }
        )
    return out


# ---------------------------------------------------------------------------
# F. Summary rollup
# ---------------------------------------------------------------------------


def summarize(rows, peak_rows, tost_rows, lit_rows, wu_rows):
    """One-line metrics for the paper rollup."""
    out = []
    budgets = sorted({r["T"] for r in rows})
    # Broader-scale: G=4 vs G=32 retention at each budget
    for T in budgets:
        g4 = next(r for r in rows if r["T"] == T and r["G"] == 4)
        g32 = next(r for r in rows if r["T"] == T and r["G"] == 32)
        out.append(
            {
                "metric": f"retention_G4_vs_G32_at_T{T//1_000_000}M",
                "value": f"{g4['acc'] / g32['acc']:.4f}",
            }
        )
    # Argmax G over budget
    argmax_G_by_T = [(p["T_tokens"], p["argmax_G"]) for p in peak_rows]
    for T, Gp in argmax_G_by_T:
        out.append(
            {
                "metric": f"argmax_G_at_T{T//1_000_000}M",
                "value": str(Gp),
            }
        )
    # Broader-scale TOST pass count at eps=0.05
    n_tost_05 = sum(1 for r in tost_rows if r["epsilon"] == 0.05 and r["tost_equivalent"] == "yes")
    out.append(
        {
            "metric": "broader_tost_pass_at_eps0.05",
            "value": f"{n_tost_05}/{len([r for r in tost_rows if r['epsilon'] == 0.05])}",
        }
    )
    # Lit-compare
    n_above_wu = sum(1 for r in lit_rows if r["above_wu_threshold"] == "yes")
    out.append(
        {
            "metric": "n_budgets_above_wu_97_6pct",
            "value": f"{n_above_wu}/{len(lit_rows)}",
        }
    )
    # Peak shift direction at largest budget
    if peak_rows:
        out.append(
            {
                "metric": "r_logG_acc_at_T64M",
                "value": f"{peak_rows[-1]['r_logG_acc']:.4f}",
            }
        )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    rows = load_token_norm()
    zvf = load_zvf_sweep()

    print(f"Loaded {len(rows)} token-normalized cells, {len(zvf)} zvf-sweep rows.")

    # A
    rvg = reward_vs_G(rows)
    write_tsv(
        RES / "group_size_iter51_reward_vs_G.tsv",
        [
            "T_tokens",
            "G",
            "acc",
            "ci_lo",
            "ci_hi",
            "gu",
            "is_peak",
            "peak_G",
            "peak_acc",
        ],
        rvg,
    )
    print(f"Wrote group_size_iter51_reward_vs_G.tsv ({len(rvg)} rows)")

    # B
    tost = broaden_tost(rows)
    write_tsv(
        RES / "group_size_iter51_broader_tost.tsv",
        [
            "T_tokens",
            "G_a",
            "G_b",
            "G4_acc",
            "G32_acc",
            "abs_diff",
            "retention",
            "ci_lo",
            "ci_hi",
            "epsilon",
            "tost_equivalent",
        ],
        tost,
    )
    print(f"Wrote group_size_iter51_broader_tost.tsv ({len(tost)} rows)")

    # C
    pk = peak_shift(rows)
    write_tsv(
        RES / "group_size_iter51_peak_shift.tsv",
        [
            "T_tokens",
            "argmax_G",
            "argmax_acc",
            "G4_acc",
            "G32_acc",
            "r_logG_acc",
            "direction",
        ],
        pk,
    )
    print(f"Wrote group_size_iter51_peak_shift.tsv ({len(pk)} rows)")

    # D
    wu = wu_loglinear(rows)
    write_tsv(
        RES / "group_size_iter51_wu_loglinear.tsv",
        ["G", "slope", "intercept", "R2", "target_acc", "log10T_at_target", "T_at_target_M"],
        wu,
    )
    print(f"Wrote group_size_iter51_wu_loglinear.tsv ({len(wu)} rows)")

    # E
    lit = literature_compare(rows)
    write_tsv(
        RES / "group_size_iter51_lit_compare.tsv",
        [
            "T_tokens",
            "wu_2025_paper",
            "wu_2025_Ga_over_Gb",
            "wu_2025_retention",
            "ours_Ga_over_Gb",
            "ours_Ga_acc",
            "ours_Gb_acc",
            "ours_retention",
            "above_wu_threshold",
            "gap_to_wu",
        ],
        lit,
    )
    print(f"Wrote group_size_iter51_lit_compare.tsv ({len(lit)} rows)")

    # F
    summ = summarize(rows, pk, tost, lit, wu)
    write_tsv(RES / "group_size_iter51_summary.tsv", ["metric", "value"], summ)
    print(f"Wrote group_size_iter51_summary.tsv ({len(summ)} rows)")

    # ----- console headline -----
    print("\n=== Iter 51 Headline ===")
    print(f"argmax G by budget: {[(p['T_tokens']//1_000_000, p['argmax_G']) for p in pk]}")
    print("Retention G=4 vs G=32 by budget:")
    for r in lit:
        flag = "ABOVE Wu" if r["above_wu_threshold"] == "yes" else "BELOW Wu"
        print(
            f"  T={r['T_tokens']//1_000_000}M  R={r['ours_retention']:.4f}  {flag}  (gap {r['gap_to_wu']:+.4f})"
        )
    n_tost_05 = sum(1 for r in tost if r["epsilon"] == 0.05 and r["tost_equivalent"] == "yes")
    print(f"TOST equivalent at eps=0.05: {n_tost_05}/4 budgets")


if __name__ == "__main__":
    main()
