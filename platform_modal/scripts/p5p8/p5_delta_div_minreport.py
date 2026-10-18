#!/usr/bin/env python3
"""P5 JOB B / SYNTH (iter 80): yield-aware MIN-REPORT v2.1 axis.

Fresh vein (not in 93 prior rows). Closes the iter-76 mint recommendation:
"yield-aware MIN-REPORT item that records iter-66 row 77
measured_yield_residual (delta_div) directly". This iter adds a single
PROPOSED v2.1 item to MIN-REPORT:

  Item 13: zvf_yield_residual (float in [-1, 1], the structural
           anti-herding bonus delta_div = ZVF_obs - ZVF_iid).

This item is computed per-cell from the iter-73 G + p = mean_reward axis
NO additional harvest needed: cells.tsv already records (G, mean_reward,
zvf) and so the new item is a deterministic transform of stack data.

Measures (bootstrap B=2000, seed 20260705):

  H1. H_bits uplift: does Item 13 contribute measurable Shannon
      information to the v2.0 fingerprint? (item must have n_unique >= 2
      and H_bits > 0 on the live 98-cell corpus.)
  H2. Coupling: does Item 13 STRENGTHEN the iter-65 row 76
      Hamming x |deltaZVF| Spearman rho? Decompose into (a) full v2.1
      fingerprint, (b) v2.1 minus v1 (truly-new-only), and compare.
  H3. Per-cell badge uplift: deterministic; one Item 13 of weight 20
      lifts each cell's badge by +20 points; 95% CI degenerate.
  H4. Per-cell scalar coupling: item13 alone vs |dzvf| Spearman,
      versus v1 fingerprint alone, to isolate the marginal contribution.

Outputs (5 files):
  platform_hybrid/experiments/results/p5p8/p5_delta_div_minreport.tsv
  platform_hybrid/experiments/results/p5p8/p5_delta_div_minreport_boot.tsv
  platform_hybrid/experiments/results/p5p8/p5_delta_div_minreport_summary.json

Stdlib only. <=290 lines.
"""
from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

from _p5p7_common import spearman_v1 as spearman

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from _review02_contracts import finite_json
from _paths import REPO_ROOT
ROOT = REPO_ROOT / "platform_hybrid"
CELLS = ROOT / "experiments" / "results" / "mega_20260704" / "cells.tsv"
MANIFEST_DIR = ROOT / "experiments" / "results" / "mega_20260704" / "manifests"
OUT = ROOT / "experiments" / "results" / "p5p8"
OUT.mkdir(parents=True, exist_ok=True)
RNG = random.Random(20260705)
N_BOOT = 2000
N_PAIRS = 2000
ITEM13_WEIGHT = 20.0
V1_ITEMS = ["loss_form", "ref_policy_kl", "sampler_backend_precision",
            "per_step_zvf_path", "group_size_schedule", "heldout_split",
            "decontamination_notes"]
V2_STACK = [("model_family", 10), ("task_slice", 10), ("G", 5),
            ("temperature", 5), ("seed", 5)]


def load_cells():
    rows = []
    with CELLS.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            r["G"] = int(r["G"])
            r["temperature"] = float(r["temperature"])
            r["seed"] = int(r["seed"])
            for k in ("mean_reward", "zvf", "pcd", "mean_completion_len"):
                try:
                    r[k] = float(r[k]) if r[k] not in ("", None) else None
                except (TypeError, ValueError):
                    r[k] = None
            rows.append(r)
    return rows


def load_manifests():
    out = {}
    for jf in MANIFEST_DIR.glob("*.json"):
        try:
            with jf.open() as f:
                d = json.load(f)
        except Exception as exc:
            print(f"warn: bad json {jf}: {exc}")
            continue
        out[d.get("cell_id", jf.stem)] = d
    return out


def shannon(values):
    if not values:
        return 0.0
    n = len(values)
    h = 0.0
    for c in Counter(values).values():
        if c == 0:
            continue
        p = c / n
        h -= p * math.log2(p)
    return h


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def delta_div(c, reward_unit='percent'):
    """Observed telemetry minus homogeneous pooled Bernoulli null.

    Legacy corpus declares percent reward units. Override explicitly for a
    fraction corpus; never infer units, replace missing reward with zero, or
    clamp invalid data. This residual also captures prompt heterogeneity.
    """
    reward, zvf_obs = c.get('mean_reward'), c.get('zvf')
    if reward is None or zvf_obs is None:
        return None
    unit = c.get('reward_unit', reward_unit)
    if unit not in {'percent', 'fraction'}:
        raise ValueError('explicit reward unit percent or fraction required')
    p = float(reward) / (100 if unit == 'percent' else 1)
    G = c['G']
    if not math.isfinite(p) or not 0 <= p <= 1 or not math.isfinite(zvf_obs) or not 0 <= zvf_obs <= 1:
        raise ValueError('invalid reward/ZVF units or missingness')
    if isinstance(G, bool) or int(G) != G or G < 1:
        raise ValueError('positive integer group size required')
    return float(zvf_obs - (p**G + (1-p)**G))


def main():
    print("# === P5 yield-aware MIN-REPORT v2.1 axis (JOB B / SYNTH iter 80) ===")
    cells = load_cells()
    manifests = load_manifests()
    n = len(cells)
    print(f"# loaded {n} cells, {len(manifests)} manifests")

    # --- Compute Item 13 (zvf_yield_residual) for each cell ---
    item13_vals = []
    for c in cells:
        d = delta_div(c)
        item13_vals.append(d if d is not None else float("nan"))
    n13_present = sum(1 for v in item13_vals if not math.isnan(v))
    print(f"# Item13 populated on {n13_present}/{n} cells")

    # --- Per-item H_bits (H1) ---
    per_item = []
    for item in V1_ITEMS:
        vals = [str(manifests.get(c["cell_id"], {}).get(item, "MISSING")) for c in cells]
        n_unique = len(set(vals))
        h = shannon(vals)
        per_item.append({"item": item, "axis": "v1", "n_unique": n_unique, "H_bits": h, "weight": 10})
    for axis, w in V2_STACK:
        vals = [str(c.get(axis, "MISSING")) for c in cells]
        n_unique = len(set(vals))
        h = shannon(vals)
        per_item.append({"item": axis, "axis": "v2_stack", "n_unique": n_unique, "H_bits": h, "weight": w})
    # Item 13
    vals13 = [round(v, 4) if not math.isnan(v) else "NaN" for v in item13_vals]
    n_unique13 = len(set(vals13))
    h13 = shannon([str(v) for v in vals13])
    per_item.append({"item": "zvf_yield_residual", "axis": "v2.1_yield", "n_unique": n_unique13,
                     "H_bits": h13, "weight": ITEM13_WEIGHT})

    h_v1 = sum(p["H_bits"] for p in per_item if p["axis"] == "v1")
    h_v2_stack = sum(p["H_bits"] for p in per_item if p["axis"] == "v2_stack")
    h_v2_total = h_v1 + h_v2_stack
    h_v21_total = h_v2_total + h13
    print(f"# H_bits: v1={h_v1:.4f} + v2_stack={h_v2_stack:.4f} = v2_total={h_v2_total:.4f}; +Item13={h13:.4f} -> v2.1={h_v21_total:.4f}")

    # --- Build fingerprints (H2) ---
    def build_fingerprint(c, axis_set):
        out = []
        if "v1" in axis_set:
            for item in V1_ITEMS:
                out.append(str(manifests.get(c["cell_id"], {}).get(item, "MISSING")))
        if "v2_stack" in axis_set:
            for axis, _ in V2_STACK:
                out.append(str(c.get(axis, "MISSING")))
        if "v2.1_yield" in axis_set:
            d = delta_div(c)
            out.append(round(d, 4) if d is not None else "NaN")
        return tuple(out)

    # Sample N_PAIRS cell-pairs; compute Hamming x |dzvf| Spearman for each axis-set
    def rho_for_axis_set(axis_set):
        d_zvf = []
        hamming_d = []
        all_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                all_pairs.append((i, j))
        sampled = random.Random(20260705).sample(all_pairs, min(N_PAIRS, len(all_pairs)))
        for i, j in sampled:
            z_i = cells[i].get("zvf"); z_j = cells[j].get("zvf")
            if z_i is None or z_j is None:
                continue
            d_zvf.append(abs(z_i - z_j))
            f_i = build_fingerprint(cells[i], axis_set)
            f_j = build_fingerprint(cells[j], axis_set)
            hamming_d.append(hamming(f_i, f_j))
        # also include v1-fingerprint alone
        return spearman(hamming_d, d_zvf), len(hamming_d)

    rho_v1, n_v1 = rho_for_axis_set({"v1"})
    rho_v2_stack, n_v2 = rho_for_axis_set({"v1", "v2_stack"})
    rho_v21, n_v21 = rho_for_axis_set({"v1", "v2_stack", "v2.1_yield"})
    print(f"# Spearman(Hamming, |dzvf|): v1={rho_v1:.4f} (n={n_v1}); v2_stack={rho_v2_stack:.4f} (n={n_v2}); v2.1={rho_v21:.4f} (n={n_v21})")

    # Fixed-campaign census, with outcome-derived descriptors. No population
    # entropy/correlation intervals or predictive claims are identifiable here.
    diffs = [rho_v21 - rho_v2_stack]
    diff_lo = diff_hi = h_diff_lo = h_diff_hi = float('nan')
    boot = [dict(metric=metric, point=float(point), ci_lo=float('nan'), ci_hi=float('nan'),
                 **{'ci_p2.5': float('nan'), 'ci_p97.5': float('nan')})
            for metric, point in [('H_bits_item13', h13),
                                  ('delta_rho_v21_minus_v2stack', diffs[0]),
                                  ('H_bits_uplift_v2.1_minus_v2', h13),
                                  ('badge_uplift_per_cell_item13', ITEM13_WEIGHT)]]

    # --- Save outputs ---
    tsv_path = OUT / "p5_delta_div_minreport.tsv"
    with tsv_path.open("w") as f:
        keys = ["item", "axis", "n_unique", "H_bits", "weight"]
        f.write("\t".join(keys) + "\n")
        for p in per_item:
            f.write("\t".join(str(p[k]) for k in keys) + "\n")
    boot_path = OUT / "p5_delta_div_minreport_boot.tsv"
    with boot_path.open("w") as f:
        keys = ["metric", "point", "ci_lo", "ci_hi"]
        f.write("\t".join(keys) + "\n")
        for b in boot:
            f.write("\t".join(str(b[k]) for k in keys) + "\n")
    summary = {
        "n_cells": n, "n_item13_present": n13_present,
        "statistic_version": "pro-review-02-v2",
        "entropy_estimand": "sum_of_marginal_entropies; descriptive diversity only",
        "construct_warning": "delta_div contains observed ZVF. Association with differences in the same ZVF is outcome leakage, not independent predictive validation. Pooled homogeneous Bernoulli null does not adjust for prompt difficulty.",
        "inference_status": "fixed_campaign_descriptive_only",
        "H_bits_v1": h_v1, "H_bits_v2_stack": h_v2_stack,
        "H_bits_v2_total": h_v2_total, "H_bits_item13": h13,
        "H_bits_v21_total": h_v21_total,
        "spearman_rho_v1": rho_v1, "spearman_rho_v2_stack": rho_v2_stack,
        "spearman_rho_v21": rho_v21,
        "delta_rho_v21_minus_v2stack_pt": float(sum(diffs) / len(diffs)),
        "delta_rho_v21_minus_v2stack_ci_lo": float(diff_lo),
        "delta_rho_v21_minus_v2stack_ci_hi": float(diff_hi),
        "item13_n_unique": n_unique13,
        "item13_per_cell_uplift_weight": ITEM13_WEIGHT,
        "verdict": ("VARYING_DESCRIPTIVE" if h13 > 0 else "CONSTANT_DESCRIPTIVE"),
        "key_finding": (f"Item 13 contributes +{h13:.4f} bits to the v2.0 fingerprint; "
                        f"Spearman coupling to |dzvf| changes by "
                        f"{sum(diffs) / len(diffs):+.4f} (descriptive; no inferential interval).")
    }
    (OUT / "p5_delta_div_minreport_summary.json").write_text(json.dumps(finite_json(summary), indent=2, default=str, allow_nan=False))
    print("# summary:", json.dumps(finite_json(summary), indent=2, default=str, allow_nan=False))
    print("# === iter 80 JOB B complete; outputs in platform_hybrid/experiments/results/p5p8/ ===")


if __name__ == "__main__":
    main()
