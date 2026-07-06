#!/usr/bin/env python3
"""P5P8-SYNTH D21 (iter 196 JOB B): per-reward-decile cross-pillar decision
concordance on N2 same-stack step cells.

Fresh 21st density domain (NOT in any prior D1..D20 SYNTH row). D20 measured
aggregate cross-pillar Spearman ρ (4 N2 methods × 40 steps) and found a
two-cluster structure {P5, P8} (gift wins) ↔ {P6, P7} (areal wins) with
NEGATIVE cross-cluster ρ. D21 lifts the lens to per-decile granularity:
stratify the 160 (method, step) cells into reward-deciles (10 deciles
based on per-step reward_mean across all 160 cells), then compute
within-decile Spearman ρ across all 6 pillar pairs.

This is the natural extension of D20: does the two-cluster structure
break down in specific deciles, or is it uniform across the reward
distribution?

Pipeline:
  1. Load n2_metrics.tsv (160 rows: 4 methods × 40 steps).
  2. Per-(method, step) compute 4 pillar headliners:
     - P5: mean reward
     - P6: -ZVF (lower ZVF risk = higher P6 score)
     - P7: reward / std(reward)  (signal-to-noise)
     - P8: reward / mean_len    (reward per token)
  3. Stratify the 160 cells into 10 reward-deciles based on reward_mean.
  4. Per-(decile, pillar-pair) compute Spearman ρ across the 4 methods
     (16 cells per decile).
  5. Bootstrap CIs (B=2000) on each ρ by resampling (method, step) cells
     within each decile.
  6. 5 falsifiable hypotheses:
     H1: At least 3/10 deciles have positive P5↔P6 ρ (concordance
         breaks down within deciles, refuting D20's negative
         aggregate result).
     H2: The D20 cross-cluster structure holds in ≥ 7/10 deciles
         (uniform two-cluster).
     H3: P5↔P8 ρ is positive in ≥ 9/10 deciles (P5-P8 cluster is
         robust to decile stratification).
     H4: Low-reward deciles (1-3) show higher |ρ| variance than
         high-reward deciles (8-10).
     H5: P7↔P8 ρ is positive in ≥ 6/10 deciles (operational pillars
         agree on cross-decile basis).

Outputs:
  synth_iter196_d21_per_decile.tsv    10 rows (per-decile summary)
  synth_iter196_d21_per_pair.tsv      60 rows (6 pairs × 10 deciles)
  synth_iter196_d21_summary.json      H1..H5 verdicts + headline

This drives the D20 operational note "EXTEND to D21 per-decile
decision-concordance next iter" to a fully validated row.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)
N2 = ROOT / "experiments" / "results" / "n2_reward_tensor_resume" / "n2_metrics.tsv"

METHODS = ["grpo", "aero", "gift", "areal"]
N_STEPS = 40
N_DECILES = 10
N_BOOT = 2000
RNG = np.random.default_rng(20260706)


def spearman(x, y):
    """Spearman ρ via rank correlation (no scipy)."""
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    n = len(x)
    mx = rx.mean()
    my = ry.mean()
    num = ((rx - mx) * (ry - my)).sum()
    den = np.sqrt(((rx - mx) ** 2).sum() * ((ry - my) ** 2).sum())
    return float(num / den) if den > 0 else 0.0


def main():
    rows = []
    with N2.open() as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            rows.append(r)
    print(f"Loaded {len(rows)} rows from n2_metrics.tsv", flush=True)

    # Per-(method, step) compute pillar headliners
    per_method_step = {}
    for r in rows:
        m = r["method"]
        s = int(r["step"])
        rew_mean = float(r["reward_mean"])
        zvf = float(r["zvf"])
        mean_len = float(r["mean_len"])
        # P7: reward / std(reward) proxy — use cv_len as noise proxy:
        # higher CV = noisier = lower signal-to-noise → P7 = rew / (1 + cv)
        cv_len = float(r["cv_len"])
        per_method_step[(m, s)] = {
            "P5_reward": rew_mean,
            "P6_minus_zvf": -zvf,  # higher P6 score = lower ZVF risk
            "P7_snr": rew_mean / (1.0 + cv_len),  # signal-to-noise proxy
            "P8_reward_per_len": rew_mean / max(1.0, mean_len),  # reward per token
        }

    # Stratify 160 (method, step) cells into 10 reward-deciles
    reward_means = np.array([per_method_step[(m, s)]["P5_reward"]
                              for m in METHODS for s in range(N_STEPS)])
    decile_edges = np.quantile(reward_means, np.linspace(0, 1, N_DECILES + 1))
    decile_edges[0] -= 1e-9  # ensure first decile includes min
    decile_edges[-1] += 1e-9

    cells_with_decile = []
    for m in METHODS:
        for s in range(N_STEPS):
            cell = per_method_step[(m, s)]
            dec = int(np.searchsorted(decile_edges[1:-1], cell["P5_reward"]))
            cells_with_decile.append((m, s, dec))

    # Per-decile: rank methods on each pillar (4 methods → ranks 1..4)
    decile_data = {d: {"methods": [], "scores": {p: [] for p in ["P5", "P6", "P7", "P8"]}}
                   for d in range(N_DECILES)}
    for m, s, d in cells_with_decile:
        decile_data[d]["methods"].append(m)
        cell = per_method_step[(m, s)]
        # Use mean across the 4 steps per method within the decile (so each
        # method contributes ONE point per decile for the per-decile ρ).
        # Strategy: aggregate within (method, decile) first.
        for p in ["P5", "P6", "P7", "P8"]:
            key = f"{p}_" + {
                "P5": "reward", "P6": "minus_zvf", "P7": "snr", "P8": "reward_per_len"
            }[p]
            decile_data[d]["scores"][p].append((m, cell[key]))

    # Aggregate per (method, decile) → one value per method per pillar per decile
    agg = {d: {p: {} for p in ["P5", "P6", "P7", "P8"]} for d in range(N_DECILES)}
    for d in range(N_DECILES):
        for p in ["P5", "P6", "P7", "P8"]:
            scores = decile_data[d]["scores"][p]
            for m, sc in scores:
                agg[d][p].setdefault(m, []).append(sc)
            for m in METHODS:
                if m in agg[d][p]:
                    agg[d][p][m] = float(np.mean(agg[d][p][m]))

    # Per-(decile, pillar-pair) Spearman ρ on the 4 methods
    pairs = [("P5", "P6"), ("P5", "P7"), ("P5", "P8"),
             ("P6", "P7"), ("P6", "P8"), ("P7", "P8")]

    def rho_for_decile_pair(d, p1, p2, sample=None):
        # Use one value per method (the mean within decile); with only 4 methods
        # bootstrap by resampling (method, step) cells within the decile.
        v1 = [agg[d][p1][m] for m in METHODS]
        v2 = [agg[d][p2][m] for m in METHODS]
        return spearman(np.array(v1), np.array(v2))

    per_decile_summary = []
    per_pair_rows = []
    pair_decile_rhos = {pair: [] for pair in pairs}
    for d in range(N_DECILES):
        decile_cells = [(m, s) for m, s, dd in cells_with_decile if dd == d]
        # Bootstrap per-decile ρ by resampling (method, step) cells within decile
        boot_rhos = {pair: [] for pair in pairs}
        for b in range(N_BOOT):
            # Resample 4 methods × N_STEPS-within-decile? Too few cells.
            # Better: resample steps within decile (each method has its own
            # sub-sample of step cells).
            boot_sample = []
            for m in METHODS:
                m_cells = [s for mm, s in decile_cells if mm == m]
                if len(m_cells) == 0:
                    boot_sample.append((m, []))
                    continue
                idx = RNG.integers(0, len(m_cells), size=len(m_cells))
                boot_sample.append((m, [m_cells[i] for i in idx]))
            # Compute per-method mean over the resampled steps
            v = {p: {} for p in ["P5", "P6", "P7", "P8"]}
            for m, ss in boot_sample:
                for p in ["P5", "P6", "P7", "P8"]:
                    key = f"{p}_" + {
                        "P5": "reward", "P6": "minus_zvf", "P7": "snr", "P8": "reward_per_len"
                    }[p]
                    v[p][m] = float(np.mean([per_method_step[(m, s)][key] for s in ss])) \
                        if ss else agg[d][p].get(m, 0.0)
            for pair in pairs:
                p1, p2 = pair
                v1 = np.array([v[p1][m] for m in METHODS])
                v2 = np.array([v[p2][m] for m in METHODS])
                boot_rhos[pair].append(spearman(v1, v2))

        # Point estimate (no bootstrap)
        point_rhos = {}
        for pair in pairs:
            p1, p2 = pair
            v1 = np.array([agg[d][p1][m] for m in METHODS])
            v2 = np.array([agg[d][p2][m] for m in METHODS])
            point_rhos[pair] = spearman(v1, v2)

        # Mean method scores per pillar (for narrative)
        mean_scores = {p: {m: agg[d][p][m] for m in METHODS}
                       for p in ["P5", "P6", "P7", "P8"]}

        # Per-decile method-rank per pillar (rank 1 = highest score)
        ranks = {}
        for p in ["P5", "P6", "P7", "P8"]:
            sorted_m = sorted(METHODS, key=lambda m: -agg[d][p][m])
            ranks[p] = {m: r + 1 for r, m in enumerate(sorted_m)}

        per_decile_summary.append({
            "decile": d + 1,
            "n_cells": len(decile_cells),
            "mean_reward_in_decile": float(np.mean([agg[d]["P5"][m] for m in METHODS])),
            "point_rhos": point_rhos,
            "boot_ci": {pair: {
                "lo": float(np.quantile(boot_rhos[pair], 0.025)),
                "hi": float(np.quantile(boot_rhos[pair], 0.975)),
                "mean": float(np.mean(boot_rhos[pair])),
            } for pair in pairs},
            "mean_scores_per_method": mean_scores,
            "ranks_per_method": ranks,
            "best_method_per_pillar": {
                p: min(METHODS, key=lambda m: ranks[p][m]) for p in ["P5", "P6", "P7", "P8"]
            },
        })
        for pair in pairs:
            p1, p2 = pair
            ci = per_decile_summary[-1]["boot_ci"][pair]
            per_pair_rows.append({
                "decile": d + 1,
                "p1": p1, "p2": p2,
                "rho": point_rhos[pair],
                "lo": ci["lo"],
                "hi": ci["hi"],
                "ci_positive": bool(ci["lo"] > 0),
                "ci_negative": bool(ci["hi"] < 0),
            })
            pair_decile_rhos[pair].append(point_rhos[pair])

    # ----- TSV: per-decile summary -----
    out_dec = RES / "synth_iter196_d21_per_decile.tsv"
    with out_dec.open("w") as f:
        f.write("decile\tn_cells\tmean_reward\tbest_P5\tbest_P6\tbest_P7\tbest_P8\t"
                "rho_P5P6\tlo_P5P6\thi_P5P6\trho_P5P8\tlo_P5P8\thi_P5P8\n")
        for d_data in per_decile_summary:
            f.write(f"{d_data['decile']}\t{d_data['n_cells']}\t"
                    f"{d_data['mean_reward_in_decile']:.4f}\t"
                    f"{d_data['best_method_per_pillar']['P5']}\t"
                    f"{d_data['best_method_per_pillar']['P6']}\t"
                    f"{d_data['best_method_per_pillar']['P7']}\t"
                    f"{d_data['best_method_per_pillar']['P8']}\t"
                    f"{d_data['point_rhos'][('P5','P6')]:.4f}\t"
                    f"{d_data['boot_ci'][('P5','P6')]['lo']:.4f}\t"
                    f"{d_data['boot_ci'][('P5','P6')]['hi']:.4f}\t"
                    f"{d_data['point_rhos'][('P5','P8')]:.4f}\t"
                    f"{d_data['boot_ci'][('P5','P8')]['lo']:.4f}\t"
                    f"{d_data['boot_ci'][('P5','P8')]['hi']:.4f}\n")

    # ----- TSV: per-pair rho with CIs -----
    out_pair = RES / "synth_iter196_d21_per_pair.tsv"
    with out_pair.open("w") as f:
        f.write("decile\tp1\tp2\trho\tlo\thi\tci_positive\tci_negative\n")
        for row in per_pair_rows:
            f.write(f"{row['decile']}\t{row['p1']}\t{row['p2']}\t{row['rho']:.4f}\t"
                    f"{row['lo']:.4f}\t{row['hi']:.4f}\t"
                    f"{int(row['ci_positive'])}\t{int(row['ci_negative'])}\n")

    # ----- Hypotheses -----
    # H1: at least 3/10 deciles have CI-positive P5↔P6 ρ (concordance breaks
    #     down within deciles — refuting D20's aggregate NEGATIVE result)
    h1_count = sum(1 for r in per_pair_rows
                   if r["p1"] == "P5" and r["p2"] == "P6" and r["ci_positive"])
    h1 = h1_count >= 3

    # H2: D20 two-cluster structure {P5, P8} vs {P6, P7} holds in ≥ 7/10 deciles
    # Two-cluster holds if both within-cluster ρ > 0 (P5↔P8 AND P6↔P7)
    # AND at least one cross-cluster ρ < 0 (P5↔P6 OR P5↔P7).
    h2_count = 0
    for d in range(N_DECILES):
        d_data = per_decile_summary[d]
        within_p5p8 = d_data["boot_ci"][("P5", "P8")]["lo"] > 0
        within_p6p7 = d_data["boot_ci"][("P6", "P7")]["lo"] > 0
        cross_p5p6 = d_data["boot_ci"][("P5", "P6")]["hi"] < 0
        cross_p5p7 = d_data["boot_ci"][("P5", "P7")]["hi"] < 0
        if within_p5p8 and within_p6p7 and (cross_p5p6 or cross_p5p7):
            h2_count += 1
    h2 = h2_count >= 7

    # H3: P5↔P8 ρ CI-positive in ≥ 9/10 deciles (P5-P8 cluster is robust)
    h3_count = sum(1 for r in per_pair_rows
                   if r["p1"] == "P5" and r["p2"] == "P8" and r["ci_positive"])
    h3 = h3_count >= 9

    # H4: low-reward deciles (1-3) show higher |ρ| variance than high-reward (8-10)
    low_var = np.mean([abs(pair_decile_rhos[pair][d])
                       for pair in pairs for d in range(3)])
    high_var = np.mean([abs(pair_decile_rhos[pair][d])
                        for pair in pairs for d in range(7, 10)])
    h4 = low_var > high_var

    # H5: P7↔P8 ρ CI-positive in ≥ 6/10 deciles (operational pillars agree)
    h5_count = sum(1 for r in per_pair_rows
                   if r["p1"] == "P7" and r["p2"] == "P8" and r["ci_positive"])
    h5 = h5_count >= 6

    summary = {
        "data": {"n_cells": len(cells_with_decile), "n_deciles": N_DECILES,
                 "n_methods": len(METHODS), "n_steps": N_STEPS},
        "hypotheses": {
            "H1_ge_3_deciles_P5P6_ci_positive": bool(h1),
            "H2_two_cluster_in_ge_7_deciles": bool(h2),
            "H3_P5P8_ci_positive_in_ge_9_deciles": bool(h3),
            "H4_low_deciles_higher_rho_variance": bool(h4),
            "H5_P7P8_ci_positive_in_ge_6_deciles": bool(h5),
        },
        "h1_count_P5P6_positive_deciles": h1_count,
        "h2_count_two_cluster_deciles": h2_count,
        "h3_count_P5P8_positive_deciles": h3_count,
        "h5_count_P7P8_positive_deciles": h5_count,
        "low_decile_rho_abs_mean": float(low_var),
        "high_decile_rho_abs_mean": float(high_var),
        "rho_per_decile": {
            f"decile_{d_data['decile']}": {
                f"{pair[0]}_{pair[1]}": {
                    "point": d_data["point_rhos"][pair],
                    "ci_lo": d_data["boot_ci"][pair]["lo"],
                    "ci_hi": d_data["boot_ci"][pair]["hi"],
                } for pair in pairs
            } for d_data in per_decile_summary
        },
        "best_method_per_decile_pillar": {
            f"decile_{d_data['decile']}": d_data["best_method_per_pillar"]
            for d_data in per_decile_summary
        },
        "ci_per_decile_pair": {
            f"decile_{d_data['decile']}_{pair[0]}_{pair[1]}": d_data["boot_ci"][pair]
            for d_data in per_decile_summary for pair in pairs
        },
    }
    summary["verdict"] = sum(int(v) for v in summary["hypotheses"].values())
    out_sum = RES / "synth_iter196_d21_summary.json"
    out_sum.write_text(json.dumps(summary, indent=2, default=float))
    print(f"Wrote {out_sum}", flush=True)
    print(f"H1..H5: {summary['hypotheses']}", flush=True)
    print(f"h1_count={h1_count}, h2_count={h2_count}, "
          f"h3_count={h3_count}, h5_count={h5_count}", flush=True)
    print(f"low_var={low_var:.4f}, high_var={high_var:.4f}", flush=True)
    print(f"verdict={summary['verdict']}/5", flush=True)


if __name__ == "__main__":
    main()