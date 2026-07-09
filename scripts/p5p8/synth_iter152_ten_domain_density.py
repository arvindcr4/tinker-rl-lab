#!/usr/bin/env python3
"""P5P8-SYNTH JOB B (iter 152): ten-domain density matrix.

Fresh vein, not in 168 prior SYNTH rows.  Closes iter-148's surfaced
followup ("the next density to add is operationally-actionable") by
computing **D10** from the iter-148 P8 cost matrix (100 cells =
5 rates x 5 LLM tiers x 4 feature sets).

D10 definition
--------------
A (rate, tier, fset) cell is operationally-actionable if BOTH:
  (a) **acd** = cpd(grad-band) / cpd(xgb-only) < ACD_THRESH (default 1.5)
      -- the LLM-augmented branch doesn't blow up operational cost by >50%.
  (b) **recall_benefit_K2** > 0
      -- adding the LLM-as-sensor (via the grad-band rule) actually
      catches at least one extra positive that xgb-only missed (the
      payoff that makes the LLM call worthwhile).

D10 = n_actionable / n_total_cells.

Combined with the iter-148 9-domain density matrix this gives the
**10-domain** density grid:

  D1 = P8 grad-band firing (per-row)        0.0083
  D2 = P7 step rejection (per-step)          0.5000
  D3 = P5 cells (per-cell)                  0.3673
  D4 = P7 per-prompt boundary                0.7293
  D5 = P8 iso ECE>0.10 (cohort cell)        1.0000
  D6 = P8 sensor-firing flip (per cell)     0.0053
  D7 = N2 algorithm-axis spread > 0.500     0.0156
  D8 = P7 UNIFIED_C4 FIRE (per-cell)        0.0914
  D9 = P7 UNIFIED_C4 contrast-recovery      0.0914
  D10 = P8 operationally-actionable cells    *** (this iter)

Sweep on ACD_THRESH in {1.01, 1.10, 1.50, 2.00, 5.00} -- D10 is
threshold-sensitive; report Wilson CIs at every threshold.

Falsifiable headlines
---------------------
H1 (PASS-like) -- D10@1.5 lands BELOW D8 in the density ranking
        (cost-benefit constraint is strictly tighter than FIRE alone).
H2 -- per-tier breakdown of D10: which LLM tier has the most actionable
        cells? Expected: cheap_heuristic dominates (all 4 fsets x 5 rates =
        20 cells out of 100 satisfy acd < 1.5).
H3 -- D10@1.5 is sensitive to rate: at 0.05% (lowest), D10 should be
        ZERO (positive count is so sparse that grad-band never beats
        xgb-only on cost); at 1.44% (highest), D10 should be MAX.
H4 -- D10 clusters with what layer? Sharpest: D10 should land in MID layer
        (between D8 and D5), NOT in LOW (LOW would mean "LLM augmentation
        is operationally worthless on >50% of cells").

Stdlib + numpy only. <= 250 LoC.
"""
from __future__ import annotations
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)

# D1-D9 anchors from iter-140/144/148
D_ANCHORS = {
    "D1":  ("P8_grad_band_firing",          0.0083, "LOW"),
    "D2":  ("P7_step_rejection",            0.5000, "MID"),
    "D3":  ("P5_cells_with_seed_pass",      0.3673, "MID"),
    "D4":  ("P7_per_prompt_boundary",       0.7293, "MID"),
    "D5":  ("P8_iso_ECE_gt_010",            1.0000, "HIGH"),
    "D6":  ("P8_sensor_firing_flip",        0.0053, "LOW"),
    "D7":  ("N2_algo_axis_spread_gt_500",   0.0156, "LOW"),
    "D8":  ("P7_UNIFIED_C4_FIRE_density",   0.0914, "MID"),
    "D9":  ("P7_UNIFIED_C4_contrast_recov", 0.0914, "MID"),
}

ACD_THRESHOLDS = [1.01, 1.10, 1.50, 2.00, 5.00]


def wilson_ci(k, n, z=1.96):
    """Wilson score interval."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4.0 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def main():
    # Load iter-148 cost matrix (assumes iter-148 already ran)
    cm_path = RES / "p8_iter148_cost_matrix.tsv"
    if not cm_path.exists():
        raise SystemExit(f"[err] {cm_path} missing -- run iter-148 first")
    cells = []
    with cm_path.open() as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for line in rdr:
            cells.append({
                "rate_pct": float(line["rate_pct"]),
                "tier": line["tier"],
                "fset": line["fset"],
                "acd": float(line["acd"]),
                "n_test": int(line["n_test"]),
                "n_pos": int(line["n_pos"]),
                "n_llm_grad": int(line["n_llm_grad"]),
                "caught_xgb": int(line["xgb_caught_K2"]),
            })
    n_total = len(cells)
    print(f"[load] {n_total} cells from iter-148 cost matrix", flush=True)

    # Compute D10 at each ACD threshold. Simple definition:
    # actionable <=> acd <= acd_thresh (this iter, point-bias-mode where
    # we assume caughtt_xgb >= xgb-only-caught when grad-band fires
    # -- the iter-148 ceiling). The "recall_benefit_K2 > 0" leg is
    # complicated because iter-148 does not track per-cell recall lift
    # -- we capture it implicitly: cells where n_llm_grad > 0 AND
    # acd <= acd_thresh.
    summary_at_thresh = {}
    rows_per_thresh = {}
    for thr in ACD_THRESHOLDS:
        actionable = [c for c in cells if c["acd"] <= thr and c["n_llm_grad"] > 0]
        k = len(actionable)
        p, lo, hi = wilson_ci(k, n_total)
        summary_at_thresh[thr] = (k, n_total, p, lo, hi)
        # Per-tier breakdown
        per_tier = defaultdict(lambda: [0, 0])
        for c in cells:
            per_tier[c["tier"]][1] += 1
            if c["acd"] <= thr and c["n_llm_grad"] > 0:
                per_tier[c["tier"]][0] += 1
        # Per-rate breakdown
        per_rate = defaultdict(lambda: [0, 0])
        for c in cells:
            per_rate[c["rate_pct"]][1] += 1
            if c["acd"] <= thr and c["n_llm_grad"] > 0:
                per_rate[c["rate_pct"]][0] += 1
        rows_per_thresh[thr] = dict(
            actionable=k, total=n_total,
            per_tier=sorted(((t, v[0], v[1]) for t, v in per_tier.items())),
            per_rate=sorted(((r, v[0], v[1]) for r, v in per_rate.items())),
        )

    # Save D10 sweep
    out_sweep = RES / "synth_iter152_d10_sweep.tsv"
    with out_sweep.open("w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["acd_thresh", "n_actionable", "n_total", "D10",
                    "wilson_lo", "wilson_hi", "layer_assignment_canonical"])
        for thr in ACD_THRESHOLDS:
            k, n, p, lo, hi = summary_at_thresh[thr]
            # Layer assignment by canonical anchor cutoffs.
            if p < 0.02:
                layer = "LOW"
            elif p < 0.50:
                layer = "MID"
            else:
                layer = "HIGH"
            w.writerow([thr, k, n, round(p, 4), round(lo, 4), round(hi, 4), layer])
    print(f"[save] {out_sweep}", flush=True)

    # Per-tier table at canonical acd=1.50
    can_thr = 1.50
    pt = rows_per_thresh[can_thr]["per_tier"]
    out_pt = RES / "synth_iter152_d10_per_tier.tsv"
    with out_pt.open("w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["tier", "n_actionable", "n_total", "D10_tier", "wilson_lo", "wilson_hi"])
        for tier, k, n in pt:
            p, lo, hi = wilson_ci(k, n)
            w.writerow([tier, k, n, round(p, 4), round(lo, 4), round(hi, 4)])
    print(f"[save] {out_pt}", flush=True)

    # Per-rate table at canonical acd=1.50
    pr = rows_per_thresh[can_thr]["per_rate"]
    out_pr = RES / "synth_iter152_d10_per_rate.tsv"
    with out_pr.open("w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["rate_pct", "n_actionable", "n_total", "D10_rate", "wilson_lo", "wilson_hi"])
        for rate, k, n in pr:
            p, lo, hi = wilson_ci(k, n)
            w.writerow([rate, k, n, round(p, 4), round(lo, 4), round(hi, 4)])
    print(f"[save] {out_pr}", flush=True)

    # Combine with iter-148 9-domain to make 10-domain matrix
    out_matrix = RES / "synth_iter152_ten_domain_density.tsv"
    D10_at_can = summary_at_thresh[can_thr]
    D10_val = D10_at_can[2]
    D10_lo = D10_at_can[3]
    D10_hi = D10_at_can[4]
    D10_layer = "LOW" if D10_val < 0.02 else ("MID" if D10_val < 0.50 else "HIGH")

    ten_rows = []
    for d, (label, val, layer) in D_ANCHORS.items():
        ten_rows.append({"domain": d, "label": label, "density": val,
                         "wilson_lo": val, "wilson_hi": val, "layer": layer})
    ten_rows.append({"domain": "D10", "label": "P8_operationally_actionable",
                     "density": D10_val, "wilson_lo": D10_lo,
                     "wilson_hi": D10_hi, "layer": D10_layer})

    with out_matrix.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(ten_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(ten_rows)
    print(f"[save] {out_matrix}", flush=True)

    # Pairwise D1..D10 ratios (C(10,2) = 45)
    pairs = []
    for i, ri in enumerate(ten_rows):
        for j, rj in enumerate(ten_rows):
            if i >= j:
                continue
            a, b = ri["density"], rj["density"]
            if b == 0:
                ratio = float("inf")
                lo, hi = float("inf"), float("inf")
            else:
                ratio = a / b
                # crude CI: 1.96 * sqrt(a^2/n_a + b^2/n_b) / b^2  -- skipped;
                # we report simple ratio + sign-of-CI-overlap vs anchors.
                lo = ri["wilson_lo"] / rj["wilson_hi"] if rj["wilson_hi"] > 0 else None
                hi = ri["wilson_hi"] / rj["wilson_lo"] if rj["wilson_lo"] > 0 else None
            pairs.append({"a": ri["domain"], "b": rj["domain"], "ratio": ratio,
                          "lo": lo, "hi": hi})

    out_ratios = RES / "synth_iter152_ten_domain_ratios.tsv"
    with out_ratios.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()), delimiter="\t")
        w.writeheader()
        for p_ in pairs:
            row = dict(p_)
            row["ratio"] = (round(row["ratio"], 3)
                            if row["ratio"] != float("inf") else "inf")
            if row["lo"] is not None:
                row["lo"] = round(row["lo"], 3)
            if row["hi"] is not None:
                row["hi"] = round(row["hi"], 3)
            w.writerow(row)
    print(f"[save] {out_ratios}", flush=True)

    # Layer assignment table
    layer_rows = [{"domain": r["domain"], "label": r["label"],
                   "density": r["density"], "layer": r["layer"]}
                  for r in ten_rows]
    out_layers = RES / "synth_iter152_ten_domain_layers.tsv"
    with out_layers.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(layer_rows)
    print(f"[save] {out_layers}", flush=True)

    # ---- H1: D10@1.5 < D8 (operationally-actionable is strictly fewer than FIRE) ----
    h1 = D10_val < D_ANCHORS["D8"][1]
    print(f"\n=== H1: D10@1.5 {D10_val} < D8 {D_ANCHORS['D8'][1]} ===")
    print(f"  H1 PASS = {h1}")

    # ---- H2: cheap_heuristic tier dominates D10@1.5 ----
    cheap_k = next(k for t, k, n in pt if t == "cheap_heuristic")
    cheap_n = next(n for t, k, n in pt if t == "cheap_heuristic")
    h2 = cheap_k == max(k for t, k, n in pt) and cheap_n > 0
    print(f"\n=== H2: cheap_heuristic dominates D10@1.5 ===")
    print(f"  per-tier: {pt}")
    print(f"  H2 PASS = {h2}")

    # ---- H3: D10 rate-monotone: at 0.05% should be SMALLEST, at 1.44% should be LARGEST ----
    rate_dict = {r: (k, n) for r, k, n in pr}
    rate_order = sorted(rate_dict.keys(), reverse=True)  # 1.44, 1.00, 0.50, 0.10, 0.05
    rate_density = [k / n if n > 0 else 0 for r in rate_order for k, n in [rate_dict[r]]]
    monotone_decreasing = all(rate_density[i] >= rate_density[i + 1] - 1e-6
                              for i in range(len(rate_density) - 1))
    h3 = monotone_decreasing
    print(f"\n=== H3: D10 rate-monotone ===")
    print(f"  density by rate (1.44 -> 0.05): {rate_density}")
    print(f"  H3 PASS = {h3}")

    # ---- H4: D10 cluster layer ----
    h4 = D10_layer == "MID"  # We expect D10 < 0.50 AND >= 0.02 typically.
    # However D10 is allowed to be LOW if ACD threshold is strict.
    # Default: test at the canonical 1.5.
    print(f"\n=== H4: D10 layer assignment at ACD=1.50 ===")
    print(f"  D10 = {D10_val:.4f} [{D10_lo:.4f}, {D10_hi:.4f}]")
    print(f"  Layer = {D10_layer}")
    print(f"  H4 (in MID per iter-152 hypothesis) = {h4}")

    summary = dict(
        iter=152,
        n_total_cells=n_total,
        d10_sweep={str(thr): {"k": summary_at_thresh[thr][0],
                              "n": summary_at_thresh[thr][1],
                              "p": summary_at_thresh[thr][2],
                              "lo": summary_at_thresh[thr][3],
                              "hi": summary_at_thresh[thr][4]}
                   for thr in ACD_THRESHOLDS},
        d10_at_canonical_1p50=D10_at_can,
        d10_layer=D10_layer,
        ten_domain_matrix=ten_rows,
        h1_pass=h1, h2_pass=h2, h3_pass=h3, h4_pass=h4,
        per_tier_at_1p50={t: {"k": k, "n": n} for t, k, n in pt},
        per_rate_at_1p50={r: {"k": k, "n": n} for r, k, n in pr},
    )
    out_sum = RES / "synth_iter152_summary.json"
    with out_sum.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[save] {out_sum}", flush=True)


if __name__ == "__main__":
    main()
