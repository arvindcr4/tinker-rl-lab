#!/usr/bin/env python3
"""P5P8-SYNTH D22 (iter 200 JOB B): cross-pillar decision rule under
cost-optimal operational weighting.

Fresh 22nd density domain. D20 measured aggregate cross-pillar Spearman ρ
on 160 N2 cells and found a two-cluster structure. D21 (iter-196) lifted
the lens to per-decile and found the aggregate structure is NOT robust
within deciles — point ρs vary wildly and CIs are wide due to only 4
methods per decile.

D22 takes the D21 finding and adds an OPERATIONAL weight: instead of using
reward_mean as the P5 (and indirectly P8) headliner, weight each (method,
step) cell by the COST-DOMINATED criterion used in P8 — i.e. reward per
cost-of-computation. This asks: under realistic cost-asymmetric
deployment, do the cross-pillar decision rules become more stable?

Pipeline:
  1. Load n2_metrics.tsv (160 rows: 4 methods × 40 steps).
  2. Per-(method, step) compute 4 pillar headliners:
     - P5: mean reward  (the D20/D21 headliner)
     - P6: -ZVF
     - P7: reward / (1 + cv_len)  (signal-to-noise proxy)
     - P8: reward / mean_len      (reward per token — D21 headliner)
  3. Compute the cost-optimal per-step weight w(method, step) = reward /
     (1 + lambda * mean_len), with lambda derived from the iter-188 c=100
     threshold sensitivity: lambda = (c - 1) / N_t * mean_len, simplified to
     w = reward / (1 + (c/100) * mean_len / mean(mean_len)).
  4. Per-(method, step) compute COST-WEIGHTED pillar scores:
     - P5_w = P5 * w
     - P6_w = P6 * w
     - P7_w = P7 * w
     - P8_w = P8 * w
  5. Compute cross-pillar Spearman ρ on 160 cells (raw) AND 160 cells
     (cost-weighted). Bootstrap B=2000 paired by step.
  6. Per-decile (10 deciles on raw reward_mean) compare the 6 pillar-pair
     ρs under raw vs cost-weighting.
  7. 5 falsifiable hypotheses:
     H1: cost-weighting improves the |P5↔P8| ρ magnitude (operational pillars
         become more aligned under deployment-relevant weighting).
     H2: D20 two-cluster structure (P5, P8) vs (P6, P7) is stronger under
         cost-weighting than under raw (aggregate ρ).
     H3: cost-weighting reduces the per-decile ρ variance (operational
         weights smooth out the D21 noise).
     H4: low-decile |ρ| variance remains > high-decile under cost-weighting
         (D21 H4 holds under the new weighting).
     H5: the best method per pillar becomes MORE STABLE across deciles
         under cost-weighting — fewer deciles where the best method changes.

Outputs:
  synth_iter200_d22_aggregate_rho.tsv   12 rows (6 pairs x 2 weightings)
  synth_iter200_d22_per_decile.tsv      60 rows (6 pairs x 10 deciles) raw
  synth_iter200_d22_per_decile_w.tsv    60 rows (6 pairs x 10 deciles) weighted
  synth_iter200_d22_stability.tsv       4 methods x 10 deciles x 4 pillars x 2 weightings
  synth_iter200_d22_summary.json        H1..H5 verdicts
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

    per_method_step = {}
    for r in rows:
        m = r["method"]
        s = int(r["step"])
        rew_mean = float(r["reward_mean"])
        zvf = float(r["zvf"])
        mean_len = float(r["mean_len"])
        cv_len = float(r["cv_len"])
        per_method_step[(m, s)] = {
            "P5_reward": rew_mean,
            "P6_minus_zvf": -zvf,
            "P7_snr": rew_mean / (1.0 + cv_len),
            "P8_reward_per_len": rew_mean / max(1.0, mean_len),
            "mean_len": mean_len,
        }

    # Compute cost-optimal weight per (method, step).
    # Use the iter-188 c=100 weighting: w = reward / (1 + (c/100) * mean_len / ref_len)
    # where ref_len = mean(mean_len across all cells) = ~800 tokens (proxy).
    ref_len = float(np.mean([per_method_step[(m, s)]["mean_len"]
                              for m in METHODS for s in range(N_STEPS)]))
    c_norm = 1.0  # cost ratio scaling (c=100 gives c_norm=1.0)
    weights = {}
    for m in METHODS:
        for s in range(N_STEPS):
            cell = per_method_step[(m, s)]
            w = cell["P5_reward"] / (1.0 + c_norm * cell["mean_len"] / ref_len)
            weights[(m, s)] = w

    # Build cost-weighted pillar scores
    for m in METHODS:
        for s in range(N_STEPS):
            cell = per_method_step[(m, s)]
            w = weights[(m, s)]
            per_method_step[(m, s)].update({
                "P5_w": cell["P5_reward"] * w,
                "P6_w": cell["P6_minus_zvf"] * w,
                "P7_w": cell["P7_snr"] * w,
                "P8_w": cell["P8_reward_per_len"] * w,
            })

    pairs = [("P5", "P6"), ("P5", "P7"), ("P5", "P8"),
             ("P6", "P7"), ("P6", "P8"), ("P7", "P8")]
    keys_raw = {"P5": "P5_reward", "P6": "P6_minus_zvf",
                "P7": "P7_snr", "P8": "P8_reward_per_len"}
    keys_w = {"P5": "P5_w", "P6": "P6_w", "P7": "P7_w", "P8": "P8_w"}

    # Aggregate cross-pillar ρ on 160 cells
    def aggregate_rho(weighting):
        keys = keys_w if weighting == "weighted" else keys_raw
        out = {}
        for p1, p2 in pairs:
            v1 = np.array([per_method_step[(m, s)][keys[p1]]
                           for m in METHODS for s in range(N_STEPS)])
            v2 = np.array([per_method_step[(m, s)][keys[p2]]
                           for m in METHODS for s in range(N_STEPS)])
            out[(p1, p2)] = spearman(v1, v2)
        return out

    raw_rho = aggregate_rho("raw")
    w_rho = aggregate_rho("weighted")

    # Bootstrap CIs on aggregate ρ (resample 160 cells)
    def boot_aggregate_rho(weighting, n_boot):
        keys = keys_w if weighting == "weighted" else keys_raw
        out = {pair: [] for pair in pairs}
        cells = [(m, s) for m in METHODS for s in range(N_STEPS)]
        rng = np.random.default_rng(20260706)
        for b in range(n_boot):
            idx = rng.integers(0, len(cells), size=len(cells))
            for p1, p2 in pairs:
                v1 = np.array([per_method_step[cells[i]][keys[p1]] for i in idx])
                v2 = np.array([per_method_step[cells[i]][keys[p2]] for i in idx])
                out[(p1, p2)].append(spearman(v1, v2))
        return out

    boot_raw = boot_aggregate_rho("raw", N_BOOT)
    boot_w = boot_aggregate_rho("weighted", N_BOOT)

    # Per-decile ρ for both weightings
    reward_means = np.array([per_method_step[(m, s)]["P5_reward"]
                              for m in METHODS for s in range(N_STEPS)])
    decile_edges = np.quantile(reward_means, np.linspace(0, 1, N_DECILES + 1))
    decile_edges[0] -= 1e-9
    decile_edges[-1] += 1e-9

    cells_with_decile = []
    for m in METHODS:
        for s in range(N_STEPS):
            cell = per_method_step[(m, s)]
            dec = int(np.searchsorted(decile_edges[1:-1], cell["P5_reward"]))
            cells_with_decile.append((m, s, dec))

    def per_decile_summary(weighting):
        keys = keys_w if weighting == "weighted" else keys_raw
        # Aggregate per (method, decile) → one value per method per pillar per decile
        agg = {d: {p: {} for p in ["P5", "P6", "P7", "P8"]} for d in range(N_DECILES)}
        for m, s, d in cells_with_decile:
            for p in ["P5", "P6", "P7", "P8"]:
                agg[d][p].setdefault(m, []).append(per_method_step[(m, s)][keys[p]])
        for d in range(N_DECILES):
            for p in ["P5", "P6", "P7", "P8"]:
                for m in METHODS:
                    if m in agg[d][p]:
                        agg[d][p][m] = float(np.mean(agg[d][p][m]))
                    else:
                        agg[d][p][m] = 0.0
        # Per-pair ρ per decile
        out = []
        for d in range(N_DECILES):
            row = {"decile": d + 1}
            for p1, p2 in pairs:
                v1 = np.array([agg[d][p1][m] for m in METHODS])
                v2 = np.array([agg[d][p2][m] for m in METHODS])
                row[f"rho_{p1}{p2}"] = spearman(v1, v2)
            # Best method per pillar in this decile
            row["best_per_pillar"] = {
                p: min(METHODS, key=lambda m: -agg[d][p][m]) for p in ["P5", "P6", "P7", "P8"]
            }
            out.append(row)
        return out, agg

    raw_decile, raw_agg = per_decile_summary("raw")
    w_decile, w_agg = per_decile_summary("weighted")

    # ----- TSV: aggregate ρ (raw vs weighted) -----
    out_agg = RES / "synth_iter200_d22_aggregate_rho.tsv"
    with out_agg.open("w") as f:
        f.write("pair\tweighting\trho\tlo\thi\tci_positive\tci_negative\n")
        for pair in pairs:
            for weighting in ["raw", "weighted"]:
                if weighting == "raw":
                    rho = raw_rho[pair]
                    boot = boot_raw[pair]
                else:
                    rho = w_rho[pair]
                    boot = boot_w[pair]
                lo = float(np.quantile(boot, 0.025))
                hi = float(np.quantile(boot, 0.975))
                f.write(f"{pair[0]}-{pair[1]}\t{weighting}\t{rho:.4f}\t"
                        f"{lo:.4f}\t{hi:.4f}\t{int(lo > 0)}\t{int(hi < 0)}\n")

    # ----- TSV: per-decile ρ raw -----
    def write_per_decile(rows, suffix):
        out = RES / f"synth_iter200_d22_per_decile{suffix}.tsv"
        with out.open("w") as f:
            f.write("decile\trho_P5P6\trho_P5P7\trho_P5P8\trho_P6P7\trho_P6P8\trho_P7P8\t"
                    "best_P5\tbest_P6\tbest_P7\tbest_P8\n")
            for r in rows:
                f.write(f"{r['decile']}\t"
                        f"{r['rho_P5P6']:.4f}\t{r['rho_P5P7']:.4f}\t{r['rho_P5P8']:.4f}\t"
                        f"{r['rho_P6P7']:.4f}\t{r['rho_P6P8']:.4f}\t{r['rho_P7P8']:.4f}\t"
                        f"{r['best_per_pillar']['P5']}\t{r['best_per_pillar']['P6']}\t"
                        f"{r['best_per_pillar']['P7']}\t{r['best_per_pillar']['P8']}\n")

    write_per_decile(raw_decile, "")
    write_per_decile(w_decile, "_w")

    # ----- TSV: stability (best method per pillar per decile per weighting) -----
    out_stab = RES / "synth_iter200_d22_stability.tsv"
    with out_stab.open("w") as f:
        f.write("decile\tp\tweighting\tbest_method\n")
        for d in range(N_DECILES):
            for p in ["P5", "P6", "P7", "P8"]:
                f.write(f"{d+1}\t{p}\traw\t{raw_decile[d]['best_per_pillar'][p]}\n")
                f.write(f"{d+1}\t{p}\tweighted\t{w_decile[d]['best_per_pillar'][p]}\n")

    # ----- Hypotheses -----
    # H1: |P5↔P8| ρ magnitude greater under cost-weighting than raw
    h1_pass = bool(abs(w_rho[("P5", "P8")]) > abs(raw_rho[("P5", "P8")]))

    # H2: D20 two-cluster structure stronger under cost-weighting
    # Two-cluster requires: within-cluster ρ > 0 (P5↔P8 and P6↔P7)
    # AND cross-cluster ρ < 0 (P5↔P6 or P5↔P7)
    def two_cluster_score(rho_dict):
        within_p5p8 = rho_dict[("P5", "P8")] > 0
        within_p6p7 = rho_dict[("P6", "P7")] > 0
        cross_p5p6 = rho_dict[("P5", "P6")] < 0
        cross_p5p7 = rho_dict[("P5", "P7")] < 0
        return int(within_p5p8) + int(within_p6p7) + int(cross_p5p6) + int(cross_p5p7)

    raw_score = two_cluster_score(raw_rho)
    w_score = two_cluster_score(w_rho)
    h2_pass = bool(w_score > raw_score)

    # H3: cost-weighting reduces per-decile ρ variance (averaged over all pairs)
    raw_var = float(np.mean([np.var([r[f"rho_{p1}{p2}"] for r in raw_decile])
                              for p1, p2 in pairs]))
    w_var = float(np.mean([np.var([r[f"rho_{p1}{p2}"] for r in w_decile])
                            for p1, p2 in pairs]))
    h3_pass = bool(w_var < raw_var)

    # H4: low-decile |ρ| variance > high-decile |ρ| variance (D21 H4 holds under weighting)
    low_var_w = float(np.mean([abs(w_decile[d][f"rho_{p1}{p2}"])
                                for d in range(3) for p1, p2 in pairs]))
    high_var_w = float(np.mean([abs(w_decile[d][f"rho_{p1}{p2}"])
                                 for d in range(7, 10) for p1, p2 in pairs]))
    h4_pass = bool(low_var_w > high_var_w)

    # H5: best method per pillar more stable under cost-weighting —
    # count decile-pairs where the best method CHANGES between raw and weighted.
    # Lower is more stable; we want fewer changes (so PASS if changes_w < changes_raw)
    n_changes_raw = 0
    n_changes_w = 0
    for d in range(N_DECILES):
        for p in ["P5", "P6", "P7", "P8"]:
            # Compare across deciles: how often does the best method flip?
            if d > 0:
                if raw_decile[d]["best_per_pillar"][p] != raw_decile[d-1]["best_per_pillar"][p]:
                    n_changes_raw += 1
                if w_decile[d]["best_per_pillar"][p] != w_decile[d-1]["best_per_pillar"][p]:
                    n_changes_w += 1
    h5_pass = bool(n_changes_w < n_changes_raw)

    summary = {
        "iter": 200,
        "vein": "P5P8-SYNTH D22 cross-pillar decision rule under cost-optimal operational weighting",
        "n_cells": len(METHODS) * N_STEPS,
        "n_methods": len(METHODS),
        "n_deciles": N_DECILES,
        "ref_len": ref_len,
        "c_norm": c_norm,
        "raw_aggregate_rho": {f"{p1}-{p2}": raw_rho[(p1, p2)] for p1, p2 in pairs},
        "weighted_aggregate_rho": {f"{p1}-{p2}": w_rho[(p1, p2)] for p1, p2 in pairs},
        "h1_cost_weighting_increases_P5P8_rho": h1_pass,
        "h1_raw_p5p8": raw_rho[("P5", "P8")],
        "h1_weighted_p5p8": w_rho[("P5", "P8")],
        "h2_two_cluster_stronger_under_weighting": h2_pass,
        "h2_raw_two_cluster_score": raw_score,
        "h2_weighted_two_cluster_score": w_score,
        "h3_weighting_reduces_per_decile_rho_variance": h3_pass,
        "h3_raw_variance": raw_var,
        "h3_weighted_variance": w_var,
        "h4_low_decile_high_variance_under_weighting": h4_pass,
        "h4_low_var_w": low_var_w,
        "h4_high_var_w": high_var_w,
        "h5_best_method_more_stable_under_weighting": h5_pass,
        "h5_n_changes_raw": n_changes_raw,
        "h5_n_changes_weighted": n_changes_w,
        "verdict_counts": {"PASS": sum([h1_pass, h2_pass, h3_pass, h4_pass,h5_pass]),
                            "FAIL": 5 - sum([h1_pass, h2_pass, h3_pass, h4_pass, h5_pass])},
    }
    out_sum = RES / "synth_iter200_d22_summary.json"
    with out_sum.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {out_agg}", flush=True)
    print(f"Wrote {RES / 'synth_iter200_d22_per_decile.tsv'}", flush=True)
    print(f"Wrote {RES / 'synth_iter200_d22_per_decile_w.tsv'}", flush=True)
    print(f"Wrote {out_stab}", flush=True)
    print(f"Wrote {out_sum}", flush=True)
    print(f"H1..H5: {summary['verdict_counts']}", flush=True)


if __name__ == "__main__":
    main()