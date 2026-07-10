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
    * Mean κ, SE(κ), CV(κ) over the C(k,2) pairs.
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
  H4: At τ*, CV(κ across pairs) ≤ 0.30 — the κ recommendation is
      STABLE across seed pairs (no single seed-pair dominates).

Outputs:
  experiments/results/p5p8/p7_iter143_pair_kappa.tsv     (7 τ × 10 pairs = 70 rows)
  experiments/results/p5p8/p7_iter143_growing_kappa.tsv  (7 τ × 4 k-levels = 28 rows)
  experiments/results/p5p8/p7_iter143_summary.json       (H1-H4 verdicts + headline)
  experiments/results/p5p8/p7_iter143_summary.tsv       (7 τ × {mean_k, se_k, cv_k, n_pairs})

Stdlib only.
"""
import json
import statistics
from itertools import combinations
from pathlib import Path

WORK = Path("/home/claude/tinker-rl-lab-minimax")
N10_DIR = WORK / "experiments/results/n10_seed_expansion"
OUT_DIR = WORK / "experiments/results/p5p8"

TAU_GRID = (0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
SEED_ORDER = (42, 179, 316, 453, 590)  # panel order, used for growing-k slices
N_STEPS = 15
CANONICAL_TAU = 0.65
LANDIS_KOCH_FAIR = 0.40  # "fair agreement" lower bound


# ---------- I/O ----------

def load_seeds():
    """Load {seed: [zvf per step]} for the 5 N10 GRPO seeds, padded/truncated
    to N_STEPS=15."""
    out = {}
    for s in SEED_ORDER:
        path = N10_DIR / f"n10_grpo_s{s}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing seed file: {path}")
        d = json.load(open(path))
        log = d["step_log"]
        zvfs = [row["zvf"] for row in log[:N_STEPS]]
        # Pad if short
        if len(zvfs) < N_STEPS:
            zvfs = zvfs + [zvfs[-1]] * (N_STEPS - len(zvfs))
        out[s] = zvfs
    return out


# ---------- Cohen's κ ----------

def cohens_kappa(f1, f2):
    """Binary Cohen's κ on two equal-length 0/1 vectors.

    κ = (p_o − p_e) / (1 − p_e)
    where p_o = fraction agreeing, p_e = P(both 1)*P(both 1) + P(both 0)*P(both 0).
    """
    n = len(f1)
    if n == 0 or len(f2) != n:
        return 0.0
    agree = sum(1 for a, b in zip(f1, f2) if a == b)
    p_o = agree / n
    p1_pos = sum(f1) / n
    p2_pos = sum(f2) / n
    p1_neg = 1.0 - p1_pos
    p2_neg = 1.0 - p2_pos
    p_e = p1_pos * p2_pos + p1_neg * p2_neg
    if p_e >= 1.0 - 1e-12:
        # degenerate: everyone agrees (kappa undefined → return 1.0 if perfect)
        if p_o >= 1.0 - 1e-12:
            return 1.0
        return 0.0
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
    """Return {mean, se, cv, n_pairs} over a {pair: kappa} dict."""
    vals = list(kdict.values())
    if not vals:
        return {"mean": 0.0, "se": 0.0, "cv": 0.0, "n_pairs": 0}
    m = statistics.mean(vals)
    se = statistics.pstdev(vals) / (len(vals) ** 0.5) if len(vals) > 1 else 0.0
    cv = (statistics.pstdev(vals) / m) if m > 1e-12 else 0.0
    return {
        "mean": round(m, 4),
        "se": round(se, 4),
        "cv": round(cv, 4),
        "n_pairs": len(vals),
    }


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
        cols = ["tau", "mean_kappa", "se_kappa", "cv_kappa", "n_pairs"]
        f.write("\t".join(cols) + "\n")
        for tau in TAU_GRID:
            s = summarize_kappas(full_kappas[tau])
            f.write(f"{tau}\t{s['mean']}\t{s['se']}\t{s['cv']}\t{s['n_pairs']}\n")
    print(f"wrote {out} ({len(TAU_GRID)} rows)")


def write_growing(growing_kappas):
    out = OUT_DIR / "p7_iter143_growing_kappa.tsv"
    with open(out, "w") as f:
        cols = ["k", "tau", "mean_kappa", "se_kappa", "cv_kappa", "n_pairs"]
        f.write("\t".join(cols) + "\n")
        for (k, tau), kdict in sorted(growing_kappas.items()):
            s = summarize_kappas(kdict)
            f.write(f"{k}\t{tau}\t{s['mean']}\t{s['se']}\t{s['cv']}\t{s['n_pairs']}\n")
    print(f"wrote {out} ({len(growing_kappas)} rows)")


# ---------- Headline hypotheses ----------

def headline(panel_zvfs, full_kappas, growing_kappas):
    # H1: canonical τ=0.65 mean κ ≥ 0.40
    canonical = summarize_kappas(full_kappas[CANONICAL_TAU])
    h1_pass = canonical["mean"] >= LANDIS_KOCH_FAIR

    # H2: κ(k=5) ≥ κ(k=2) − 0.10 at canonical τ=0.65
    k5 = summarize_kappas(growing_kappas[(5, CANONICAL_TAU)])
    k2 = summarize_kappas(growing_kappas[(2, CANONICAL_TAU)])
    h2_pass = k5["mean"] >= (k2["mean"] - 0.10)

    # H3: τ* = argmax_τ mean κ on full 5-seed panel lies in iter-135 band
    iter135_band = (0.55, 0.80)
    best_tau = max(TAU_GRID, key=lambda t: summarize_kappas(full_kappas[t])["mean"])
    h3_pass = iter135_band[0] <= best_tau <= iter135_band[1]

    # H4: at τ*, CV(κ across pairs) ≤ 0.30
    h4_cv = summarize_kappas(full_kappas[best_tau])["cv"]
    h4_pass = h4_cv <= 0.30

    # Growing-panel convergence table
    growing_table = {}
    for k in (2, 3, 4, 5):
        growing_table[k] = {
            tau: summarize_kappas(growing_kappas[(k, tau)])["mean"]
            for tau in TAU_GRID
        }

    summary = {
        "iter": 143,
        "pillar": "P7",
        "vein": "Inter-seed FIRE-decision Cohen's κ concordance on the GROWING n10_seed_expansion panel",
        "panel": "n10_seed_expansion/n10_grpo_s{42,179,316,453,590}.json",
        "n_seeds": len(panel_zvfs),
        "n_steps_per_seed": N_STEPS,
        "tau_grid": list(TAU_GRID),
        "canonical_tau": CANONICAL_TAU,
        "landis_koch_fair_threshold": LANDIS_KOCH_FAIR,
        "iter135_stable_band": list(iter135_band),
        "H1_mean_kappa_at_canonical_ge_040": {
            "tau": CANONICAL_TAU,
            "mean_kappa": canonical["mean"],
            "se_kappa": canonical["se"],
            "cv_kappa": canonical["cv"],
            "n_pairs": canonical["n_pairs"],
            "pass": bool(h1_pass),
        },
        "H2_kappa_k5_ge_kappa_k2_minus_010": {
            "tau": CANONICAL_TAU,
            "kappa_k2": k2["mean"],
            "kappa_k5": k5["mean"],
            "delta_k5_minus_k2": round(k5["mean"] - k2["mean"], 4),
            "pass": bool(h2_pass),
        },
        "H3_best_tau_in_iter135_band": {
            "best_tau": best_tau,
            "best_mean_kappa": summarize_kappas(full_kappas[best_tau])["mean"],
            "iter135_band": list(iter135_band),
            "pass": bool(h3_pass),
        },
        "H4_cv_at_best_tau_le_030": {
            "best_tau": best_tau,
            "cv_kappa": h4_cv,
            "pass": bool(h4_pass),
        },
        "growing_panel_table": growing_table,
        "canonical_summary_row": canonical,
    }
    out = OUT_DIR / "p7_iter143_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out}")

    print()
    print("=== Per-τ Cohen's κ (full 5-seed panel) ===")
    for tau in TAU_GRID:
        s = summarize_kappas(full_kappas[tau])
        marker = " <-- canonical" if tau == CANONICAL_TAU else ""
        print(f"  τ={tau:.2f}  mean_k={s['mean']:+.4f}  se={s['se']:.4f}  "
              f"cv={s['cv']:.3f}  n_pairs={s['n_pairs']}{marker}")
    print()
    print("=== Growing-panel mean κ at canonical τ = 0.65 ===")
    for k in (2, 3, 4, 5):
        s = summarize_kappas(growing_kappas[(k, CANONICAL_TAU)])
        print(f"  k={k}: mean_k={s['mean']:+.4f}  n_pairs={s['n_pairs']}")
    print()
    print(f"H1 (mean κ ≥ 0.40 at τ=0.65): {h1_pass}  "
          f"(observed mean_k={canonical['mean']:+.4f})")
    print(f"H2 (κ(k=5) ≥ κ(k=2)−0.10 at τ=0.65): {h2_pass}  "
          f"(observed κ(k=2)={k2['mean']:+.4f}, κ(k=5)={k5['mean']:+.4f})")
    print(f"H3 (best τ in iter-135 band [0.55,0.80]): {h3_pass}  "
          f"(observed best_tau={best_tau:.2f})")
    print(f"H4 (CV(κ) ≤ 0.30 at best_tau): {h4_pass}  "
          f"(observed cv={h4_cv:.3f} at τ={best_tau})")
    return summary


def main():
    panel_zvfs = load_seeds()
    print(f"loaded {len(panel_zvfs)} seeds × {N_STEPS} steps each")
    full_kappas = full_panel_kappas(panel_zvfs)
    growing_kappas = growing_panel_kappas(panel_zvfs)
    write_pair_kappa(full_kappas)
    write_summary(full_kappas)
    write_growing(growing_kappas)
    headline(panel_zvfs, full_kappas, growing_kappas)


if __name__ == "__main__":
    main()