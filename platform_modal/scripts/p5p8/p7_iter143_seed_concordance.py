"""Iter 143 — P7 Inter-Seed FIRE-Decision Concordance on Growing N10 Panel.

Vein (brief vein (c) extended to the GROWING-panel layer, novel):
prior iters measured fire-COUNT stability (iter-99: CV(savings) at τ=0.30
= 0.124 across 5 seeds; iter-135: τ-flip band [0.55,0.80]; iter-139:
across-seed bootstrap CI on per-seed Δr) but never the DECISION-LEVEL
concordance between seeds. Iter-143 computes the per-τ Cohen's-κ-style
fire-decision concordance on every (k, τ) sub-panel of the GROWING
n10_seed_expansion panel:

  - For each τ ∈ {0.30, 0.40, ..., 0.90} (7 τ-points):
    * FIRE_p,t = 1[zvf_p,t ≥ τ] for each (seed, step) pair.
    * For each pair (i, j) of the k seeds, compute Cohen's κ
      on the 15-step FIRE vector (binary agreement above chance).
    * Mean κ, descriptive SE(κ), SD(κ) over the C(k,2) pairs.
  - GROWING-PANEL ANALYSIS (the brief's literal "growing panel" vein):
    * k = 2, 3, 4, 5: the FIRST k seeds, in the order [42, 179, 316, 453, 590].
    * Track mean κ as k grows; test convergence.

Hypotheses (all falsifiable):
  H1: At canonical τ = 0.65 (unified-bank calibrated), mean κ ≥ 0.40 across
      the full 5-seed panel (Landis-Koch "fair agreement" threshold).
  H2: κ(k=5) ≥ κ(k=2) − 0.10 at τ = 0.65 — κ does not DEGRADE as the panel
      grows, i.e., adding seeds doesn't produce a fundamentally less
      concordant trigger signal.
  H3: τ* = argmax_τ mean κ on the 5-seed panel lies in the iter-135
      "stable τ band" [0.55, 0.80]. This is the GROWING-panel analogue of
      the unified-bank's τ recommendation.
  H4: Retired: CV is unsuitable for a signed statistic. Report SD instead.

Outputs:
  platform_hybrid/experiments/results/p5p8/p7_iter143_pair_kappa.tsv     (7 τ × 10 pairs = 70 rows)
  platform_hybrid/experiments/results/p5p8/p7_iter143_growing_kappa.tsv  (7 τ × 4 k-levels = 28 rows)
  platform_hybrid/experiments/results/p5p8/p7_iter143_summary.json       (H1-H4 verdicts + headline)
  platform_hybrid/experiments/results/p5p8/p7_iter143_summary.tsv       (7 τ × {mean_k, se_k, cv_k, n_pairs})

Stdlib only.
"""

import json
import math
import statistics
from itertools import combinations
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))  # platform_modal/scripts, for _paths
from _paths import REPO_ROOT

WORK = REPO_ROOT
N10_DIR = WORK / "platform_hybrid/experiments/results/n10_seed_expansion"
OUT_DIR = WORK / "platform_hybrid/experiments/results/p5p8/corrected-review19-v3"

TAU_GRID = (0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
SEED_ORDER = (42, 179, 316, 453, 590)  # panel order, used for growing-k slices
N_STEPS = 15
CANONICAL_TAU = 0.65
LANDIS_KOCH_FAIR = 0.40  # "fair agreement" lower bound


# ---------- I/O ----------


def load_seeds():
    """Require the full panel, unique aligned step IDs and finite probabilities."""
    out = {}
    expected_ids = None
    for seed in SEED_ORDER:
        path = N10_DIR / f"n10_grpo_s{seed}.json"
        with open(path) as handle:
            log = json.load(handle)["step_log"]
        if len(log) != N_STEPS:
            raise ValueError(f"seed {seed}: expected {N_STEPS} steps, got {len(log)}")
        ids = [row["step"] for row in log]
        if len(set(ids)) != N_STEPS:
            raise ValueError(f"seed {seed}: duplicate step IDs")
        log = sorted(log, key=lambda row: row["step"])
        ids = [row["step"] for row in log]
        if expected_ids is not None and ids != expected_ids:
            raise ValueError(f"seed {seed}: mismatched step IDs")
        expected_ids = ids
        values = [float(row["zvf"]) for row in log]
        if any(not math.isfinite(z) or not 0 <= z <= 1 for z in values):
            raise ValueError(f"seed {seed}: invalid ZVF")
        out[seed] = values
    return out


# ---------- Cohen's κ ----------


def cohens_kappa(f1, f2):
    """Binary Cohen's κ on two equal-length 0/1 vectors.

    κ = (p_o − p_e) / (1 − p_e)
    where p_o = fraction agreeing, p_e = P(both 1)*P(both 1) + P(both 0)*P(both 0).
    """
    n = len(f1)
    if n == 0 or len(f2) != n:
        raise ValueError("kappa requires nonempty aligned vectors")
    agree = sum(1 for a, b in zip(f1, f2, strict=False) if a == b)
    p_o = agree / n
    p1_pos = sum(f1) / n
    p2_pos = sum(f2) / n
    p1_neg = 1.0 - p1_pos
    p2_neg = 1.0 - p2_pos
    p_e = p1_pos * p2_pos + p1_neg * p2_neg
    if p_e >= 1.0 - 1e-12:
        return None  # chance agreement is one; kappa is undefined
    return (p_o - p_e) / (1.0 - p_e)


# ---------- Per-τ per-k analysis ----------


def fire_vector(zvfs, tau):
    return [1 if z >= tau else 0 for z in zvfs]


def all_pair_kappas(panel_zvfs, tau):
    """Return dict {(s_i, s_j): kappa} for all C(k,2) pairs of panel_zvfs."""
    seeds = list(panel_zvfs.keys())
    out = {}
    for s_i, s_j in combinations(seeds, 2):
        f1 = fire_vector(panel_zvfs[s_i], tau)
        f2 = fire_vector(panel_zvfs[s_j], tau)
        out[(s_i, s_j)] = cohens_kappa(f1, f2)
    return out


def summarize_kappas(kdict):
    """Undefined pairs are excluded and counted; signed kappa uses SD, not CV."""
    vals = [v for v in kdict.values() if v is not None]
    n = len(vals)
    mean = statistics.mean(vals) if n else None
    sd = statistics.pstdev(vals) if n else None
    return {
        "mean": mean,
        "se": sd / n**0.5 if n else None,
        "sd": sd,
        "cv": None,
        "n_pairs": n,
        "n_undefined": len(kdict) - n,
    }


def pair_diagnostics(panel, tau):
    rows = []
    for a, b in combinations(panel, 2):
        f1, f2 = fire_vector(panel[a], tau), fire_vector(panel[b], tau)
        k = cohens_kappa(f1, f2)
        rows.append(
            {
                "tau": tau,
                "seed_i": a,
                "seed_j": b,
                "kappa": k,
                "status": "DEFINED" if k is not None else "UNDEFINED",
                "raw_agreement": sum(x == y for x, y in zip(f1, f2, strict=False)) / len(f1),
                "prevalence_i": statistics.mean(f1),
                "prevalence_j": statistics.mean(f2),
                "n_steps": len(f1),
            }
        )
    return rows


# ---------- Per-τ full-panel analysis ----------


def full_panel_kappas(panel_zvfs):
    """For each τ in TAU_GRID, compute pair-kappa on the full 5-seed panel."""
    out = {}
    for tau in TAU_GRID:
        out[tau] = all_pair_kappas(panel_zvfs, tau)
    return out


# ---------- Growing-panel analysis ----------


def growing_panel_kappas(panel_zvfs):
    """For each k in 2..5, restrict to the FIRST k seeds and compute
    pair-kappa for each τ. Returns {(k, tau): {pair: kappa}}."""
    out = {}
    sorted_seeds = sorted(panel_zvfs.keys())  # SEED_ORDER is already sorted
    for k in range(2, len(sorted_seeds) + 1):
        sub = {s: panel_zvfs[s] for s in sorted_seeds[:k]}
        for tau in TAU_GRID:
            out[(k, tau)] = all_pair_kappas(sub, tau)
    return out


# ---------- Write outputs ----------


def write_pair_kappa(full_kappas):
    out = OUT_DIR / "p7_iter143_pair_kappa.tsv"
    with open(out, "w") as f:
        cols = ["tau", "seed_i", "seed_j", "kappa", "n_steps"]
        f.write("\t".join(cols) + "\n")
        for tau in TAU_GRID:
            for (s_i, s_j), k in sorted(full_kappas[tau].items()):
                f.write(f"{tau}\t{s_i}\t{s_j}\t{k}\t{N_STEPS}\n")
    print(f"wrote {out} ({len(TAU_GRID) * 10} rows)")


def write_summary(full_kappas):
    out = OUT_DIR / "p7_iter143_summary.tsv"
    with open(out, "w") as f:
        cols = ["tau", "mean_kappa", "se_kappa", "sd_kappa", "n_pairs", "n_undefined"]
        f.write("\t".join(cols) + "\n")
        for tau in TAU_GRID:
            s = summarize_kappas(full_kappas[tau])
            f.write(
                f"{tau}\t{s['mean']}\t{s['se']}\t{s['sd']}\t{s['n_pairs']}\t{s['n_undefined']}\n"
            )
    print(f"wrote {out} ({len(TAU_GRID)} rows)")


def write_growing(growing_kappas):
    out = OUT_DIR / "p7_iter143_growing_kappa.tsv"
    with open(out, "w") as f:
        cols = ["k", "tau", "mean_kappa", "se_kappa", "sd_kappa", "n_pairs", "n_undefined"]
        f.write("\t".join(cols) + "\n")
        for (k, tau), kdict in sorted(growing_kappas.items()):
            s = summarize_kappas(kdict)
            f.write(
                f"{k}\t{tau}\t{s['mean']}\t{s['se']}\t{s['sd']}\t{s['n_pairs']}\t{s['n_undefined']}\n"
            )
    print(f"wrote {out} ({len(growing_kappas)} rows)")


# ---------- Headline hypotheses ----------


def headline(panel_zvfs, full_kappas, growing_kappas):
    stats = {tau: summarize_kappas(pairs) for tau, pairs in full_kappas.items()}
    eligible = [tau for tau, row in stats.items() if row["mean"] is not None]
    best = max(eligible, key=lambda tau: stats[tau]["mean"]) if eligible else None
    canonical = stats[CANONICAL_TAU]
    k2 = summarize_kappas(growing_kappas[(2, CANONICAL_TAU)])["mean"]
    k5 = summarize_kappas(growing_kappas[(5, CANONICAL_TAU)])["mean"]
    summary = {
        "protocol": "corrected-review19-v3",
        "iter": 143,
        "status": "DEFINED" if eligible else "NO_DEFINED_KAPPA",
        "best_tau": best,
        "canonical_summary_row": canonical,
        "per_tau": stats,
        "n_steps_per_seed": {k: len(v) for k, v in panel_zvfs.items()},
        "pair_diagnostics": [r for tau in TAU_GRID for r in pair_diagnostics(panel_zvfs, tau)],
        "H1_mean_kappa_at_canonical_ge_040": {
            "pass": canonical["mean"] >= LANDIS_KOCH_FAIR if canonical["mean"] is not None else None
        },
        "H2_kappa_k5_ge_kappa_k2_minus_010": {
            "pass": k5 >= k2 - 0.10 if k2 is not None and k5 is not None else None
        },
        "H3_best_tau_in_iter135_band": {"pass": 0.55 <= best <= 0.80 if best is not None else None},
        "H4_cv_at_best_tau_le_030": {"pass": None, "status": "RETIRED_SIGNED_STATISTIC_USE_SD"},
        "growing_panel_table": {
            str(key): summarize_kappas(value) for key, value in growing_kappas.items()
        },
    }
    with open(OUT_DIR / "p7_iter143_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    return summary


def main():
    panel_zvfs = load_seeds()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"loaded {len(panel_zvfs)} seeds × {N_STEPS} steps each")
    full_kappas = full_panel_kappas(panel_zvfs)
    growing_kappas = growing_panel_kappas(panel_zvfs)
    write_pair_kappa(full_kappas)
    write_summary(full_kappas)
    write_growing(growing_kappas)
    headline(panel_zvfs, full_kappas, growing_kappas)


if __name__ == "__main__":
    main()
