#!/usr/bin/env python3
"""P5 MIN-REPORT v2.0 stack-axis extension prototype (iter 73).

Fresh vein: addresses the iter-72 mint recommendation to prototype a
yield-aware MIN-REPORT item that records stack axes directly.  This iter
takes a SHORTER, MORE CONCRETE route than the iter-66 delta_div item:
it adds the 5 stack axes that cells.tsv already records as Items 8-12 of a
proposed v2.0 MIN-REPORT and quantifies, on the live 98-cell corpus:

  H1 (per-item information uplift, falsifiable): Sum(H_bits) over the 5
      new stack-axis items MUST exceed the v1 placebo budget of 4-of-7
      items.  H_v2 = H_v1 + sum_axes.

  H2 (fingerprint x measured-telemetry coupling, falsifiable): does the
      v2 fingerprint STRENGTHEN the iter-65 row 76 Spearman rho
      (hamming, |dzvf|) over the v1 fingerprint?  Critical: v1 already
      encodes G (via group_size_schedule) and task_slice (via
      heldout_split), so the v2 axes are PARTIALLY REDUNDANT with v1.
      Decompose H2 into (a) full v2 fingerprint, (b) v2 "truly-new"
      fingerprint (3 axes: model_family, temperature, seed that v1 does
      NOT encode).

  H3 (per-cell badge uplift, deterministic): badge uplift is a
      deterministic function of the corpus because every cell populates
      all 5 v2 axes from cells.tsv.  95% bootstrap CI on the per-cell
      uplift is degenerate.

  H4 (per-axis Spearman contribution, falsifiable): which of the 5 v2
      axes individually drive the Hamming x |dzvf| correlation?
      Decompose Spearman rho(hamming, |dzvf|) into per-axis rho for
      each of the 5 axes individually (1-D Hamming).

Method:
  - Load cells.tsv (98 rows) and the 98 manifests.
  - Build v1 fingerprint = 7-tuple of v1 item values.
  - Build v2 fingerprint = 12-tuple = v1 + 5 stack axes.
  - Build v2_truly_new = v1 + 3 axes not encoded in v1.
  - Per-item Shannon H on the v2 axis set.
  - Hamming distance on each fingerprint; Spearman rho with |delta zvf|
    on 2000 random cell-pairs (replicates iter-65 H3 protocol).
  - Per-axis 1-D Spearman: each axis's individual rho(hamming_1d, |dzvf|).
  - Badge uplift = sum(weight_new * 1.0) over v2 axes; deterministic.

Outputs:
  platform_hybrid/experiments/results/p5p8/p5_stack_axis_minreport.tsv
  platform_hybrid/experiments/results/p5p8/p5_stack_axis_minreport_boot.tsv
  platform_hybrid/experiments/results/p5p8/p5_stack_axis_minreport_summary.json
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))
from _review02_contracts import (
    sampled_pairs,
    cell_distance_spearman_ci,
    finite_json,
    spearman_coefficient as spearman,
)
from _minreport_contract import ITEM_ALIASES
from p5_iter205_manifest_minreport_subfield_audit import normalized_min_report
from minreport_auditor import score_manifest
from _paths import REPO_ROOT

ROOT = REPO_ROOT / "platform_hybrid"
MANIFEST_DIR = ROOT / "experiments" / "results" / "mega_20260704" / "manifests"
CELLS_TSV = ROOT / "experiments" / "results" / "mega_20260704" / "cells.tsv"
OUT_DIR = ROOT / "experiments" / "results" / "p5p8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = random.Random(20260705)
N_BOOT = 5000
N_PAIRS = 2000

# 7-item MIN-REPORT standard (mirrors p5_manifest_outcome_coupling.py)
V1_ITEMS = [
    "loss_form",
    "ref_policy_kl",
    "sampler_backend_precision",
    "per_step_zvf_path",
    "group_size_schedule",
    "heldout_split",
    "decontamination_notes",
]
# auditor weights, identical to platform_modal/scripts/p5p8/minreport_auditor.py
V1_WEIGHTS = {
    "loss_form": 10,
    "ref_policy_kl": 10,
    "sampler_backend_precision": 20,
    "per_step_zvf_path": 20,
    "group_size_schedule": 10,
    "heldout_split": 10,
    "decontamination_notes": 20,
}

# 5 stack-axis v2 candidates proposed as Items 8-12.  Each item is sourced
# directly from cells.tsv (which already records these axes for every cell);
# the v2 extension therefore costs ZERO additional harvest effort.
V2_STACK_ITEMS = [
    ("model_family", 10),  # 2 unique: Llama, Qwen -- TRULY NEW (v1 has no model axis)
    ("task_slice", 10),  # 3 unique -- PARTIALLY redundant w/ v1 item 6 (heldout_split)
    ("G", 5),  # 5 unique -- PARTIALLY redundant w/ v1 item 5 (group_size_schedule)
    ("temperature", 5),  # 2 unique: 0.6, 1.0 -- TRULY NEW
    ("seed", 5),  # 2 unique: 0, 1 -- TRULY NEW (nuisance control)
]
V2_TOTAL_WEIGHT = sum(w for _, w in V2_STACK_ITEMS)
V2_TRULY_NEW = [
    ("model_family", 10),
    ("temperature", 5),
    ("seed", 5),
]
V2_TRULY_NEW_WEIGHT = sum(w for _, w in V2_TRULY_NEW)


def load_cells() -> list[dict]:
    rows = []
    with CELLS_TSV.open() as f:
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


def load_manifests() -> dict[str, dict]:
    out = {}
    for jf in sorted(MANIFEST_DIR.glob("*.json")):
        try:
            with jf.open() as f:
                d = json.load(f)
        except Exception as exc:
            print(f"warn: bad json {jf}: {exc}", file=sys.stderr)
            continue
        out[d.get("cell_id", jf.stem)] = d
    return out


def shannon_entropy_bits(values: list) -> float:
    if not values:
        return 0.0
    n = len(values)
    counts = Counter(values)
    h = 0.0
    for c in counts.values():
        if c == 0:
            continue
        p = c / n
        h -= p * math.log2(p)
    return h


def hamming(a: tuple, b: tuple) -> int:
    return sum(1 for x, y in zip(a, b, strict=False) if x != y)


def main():
    print("Iter 73 P5: MIN-REPORT v2.0 stack-axis extension prototype")
    print("=" * 60)

    cells = load_cells()
    manifests = load_manifests()
    n_cells = len(cells)
    print(f"Loaded {n_cells} cells, {len(manifests)} manifests")

    canonical_items = {legacy: canonical for canonical, legacy in ITEM_ALIASES.items()}
    normalized = {
        c["cell_id"]: normalized_min_report(manifests.get(c["cell_id"], {})) for c in cells
    }

    def item_content(cell, item):
        value = normalized[cell["cell_id"]][canonical_items.get(item, item)]
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    # ------------------------------------------------------------------
    # PART 1: per-item Shannon H on v1 (7) and v2 (12 = 7 + 5 stack axes)
    # ------------------------------------------------------------------
    per_item_v1 = []
    for item in V1_ITEMS:
        vals = []
        for c in cells:
            vals.append(item_content(c, item))
        n_unique = len(set(vals))
        h = shannon_entropy_bits(vals)
        is_cell_id = item == "per_step_zvf_path"
        if is_cell_id and n_unique > 1:
            cls = "CELL_IDENTIFIER"
            stack_bits = 0.0
        elif n_unique <= 1:
            cls = "CONSTANT_DESCRIPTIVE"
            stack_bits = 0.0
        else:
            cls = "VARYING_STACK_DESCRIPTOR"
            stack_bits = h
        per_item_v1.append(
            {
                "schema": "v1",
                "item": item,
                "n_unique": n_unique,
                "H_bits": h,
                "stack_discriminative_bits": stack_bits,
                "classification": cls,
            }
        )

    per_item_v2 = []
    for item, weight in V2_STACK_ITEMS:
        vals = [str(c[item]) for c in cells]
        n_unique = len(set(vals))
        h = shannon_entropy_bits(vals)
        per_item_v2.append(
            {
                "schema": "v2",
                "item": item,
                "weight": weight,
                "n_unique": n_unique,
                "H_bits": h,
                "stack_discriminative_bits": h,
                "classification": "VARYING_STACK_DESCRIPTOR",
            }
        )

    h_v1_total = sum(p["H_bits"] for p in per_item_v1)
    h_v1_stack = sum(p["stack_discriminative_bits"] for p in per_item_v1)
    h_v2_stack_axes = sum(p["H_bits"] for p in per_item_v2)
    h_v2_total = h_v1_total + h_v2_stack_axes
    h_v2_stack_total = h_v1_stack + h_v2_stack_axes

    print(f"H_bits(v1 total)        = {h_v1_total:.4f}")
    print(f"H_bits(v1 stack-discrim) = {h_v1_stack:.4f}")
    print(f"H_bits(v2 stack axes)   = {h_v2_stack_axes:.4f}")
    print(f"H_bits(v2 total)        = {h_v2_total:.4f}")
    print(f"H_bits(v2 stack-discrim) = {h_v2_stack_total:.4f}")
    print()

    # ------------------------------------------------------------------
    # PART 2: fingerprints, Hamming distance, Spearman with |delta zvf|
    # ------------------------------------------------------------------
    def fp_v1(c):
        return tuple(item_content(c, k) for k in V1_ITEMS)

    def fp_v2(c):
        return fp_v1(c) + tuple(str(c[k]) for k, _ in V2_STACK_ITEMS)

    def fp_v2_truly_new(c):
        return fp_v1(c) + tuple(str(c[k]) for k, _ in V2_TRULY_NEW)

    cell_ids = [c["cell_id"] for c in cells]
    cell_map = {c["cell_id"]: c for c in cells}
    rng_pairs = random.Random(20260705)
    pairs = sampled_pairs(n_cells, n_cells * (n_cells - 1) // 2, seed=20260705)
    if not pairs:
        raise ValueError("at least two cells required for pair-distance diagnostics")
    deltas = {
        "zvf": [abs(cell_map[cell_ids[j]]["zvf"] - cell_map[cell_ids[i]]["zvf"]) for i, j in pairs],
        "reward": [
            abs(cell_map[cell_ids[j]]["mean_reward"] - cell_map[cell_ids[i]]["mean_reward"])
            for i, j in pairs
        ],
        "pcd": [abs(cell_map[cell_ids[j]]["pcd"] - cell_map[cell_ids[i]]["pcd"]) for i, j in pairs],
        "mean_len": [
            abs(
                cell_map[cell_ids[j]]["mean_completion_len"]
                - cell_map[cell_ids[i]]["mean_completion_len"]
            )
            for i, j in pairs
        ],
    }

    h_v1 = [hamming(fp_v1(cell_map[cell_ids[i]]), fp_v1(cell_map[cell_ids[j]])) for i, j in pairs]
    h_v2 = [hamming(fp_v2(cell_map[cell_ids[i]]), fp_v2(cell_map[cell_ids[j]])) for i, j in pairs]
    h_v2tn = [
        hamming(fp_v2_truly_new(cell_map[cell_ids[i]]), fp_v2_truly_new(cell_map[cell_ids[j]]))
        for i, j in pairs
    ]

    rho_v1 = {ch: spearman(h_v1, d) for ch, d in deltas.items()}
    rho_v2 = {ch: spearman(h_v2, d) for ch, d in deltas.items()}
    rho_v2tn = {ch: spearman(h_v2tn, d) for ch, d in deltas.items()}

    print("Spearman rho(hamming, |dX|):")
    for ch in ["zvf", "reward", "pcd", "mean_len"]:
        print(f"  v1: rho(hamming_v1, |d{ch}|) = {rho_v1[ch]:+.4f}")
        print(
            f"  v2: rho(hamming_v2, |d{ch}|) = {rho_v2[ch]:+.4f}  "
            f"(delta = {rho_v2[ch] - rho_v1[ch]:+.4f})"
        )
        print(
            f"  v2_tn: rho(hamming_v2_tn, |d{ch}|) = {rho_v2tn[ch]:+.4f}  "
            f"(delta = {rho_v2tn[ch] - rho_v1[ch]:+.4f})"
        )
    print()

    # ------------------------------------------------------------------
    # PART 3: per-cell badge uplift (deterministic).
    # ------------------------------------------------------------------
    def badge_v1(c) -> float:
        return score_manifest(manifests.get(c["cell_id"], {}))["badge"]

    badges_v1 = [badge_v1(c) for c in cells]
    badges_v2 = [b + V2_TOTAL_WEIGHT for b in badges_v1]
    badges_v2tn = [b + V2_TRULY_NEW_WEIGHT for b in badges_v1]
    print(
        f"badge v1:      mean={statistics.mean(badges_v1):.3f}  range=[{min(badges_v1):.3f}, {max(badges_v1):.3f}]"
    )
    print(
        f"badge v2:      mean={statistics.mean(badges_v2):.3f}  range=[{min(badges_v2):.3f}, {max(badges_v2):.3f}]"
    )
    print(
        f"badge v2_tn:   mean={statistics.mean(badges_v2tn):.3f}  range=[{min(badges_v2tn):.3f}, {max(badges_v2tn):.3f}]"
    )
    print(
        f"badge uplift v2: mean=+{V2_TOTAL_WEIGHT}  (degenerate, every cell populates all 5 axes)"
    )
    print(f"badge uplift v2_tn: mean=+{V2_TRULY_NEW_WEIGHT}  (degenerate)")
    print()

    # ------------------------------------------------------------------
    # PART 4: per-axis 1-D Spearman contribution
    # For each of the 5 v2 axes, build a 1-D Hamming vector (just that axis)
    # and measure Spearman rho with |dzvf|.  Tells us which axis drives
    # the correlation.
    # ------------------------------------------------------------------
    per_axis_rho = {}
    for item, _ in V2_STACK_ITEMS:
        vals = [str(c[item]) for c in cells]
        idx = {v: i for i, v in enumerate(sorted(set(vals)))}
        h_1d = [1 if vals[i] != vals[j] else 0 for i, j in pairs]
        per_axis_rho[item] = {
            "n_unique": len(idx),
            "rho_zvf": spearman(h_1d, deltas["zvf"]),
            "rho_reward": spearman(h_1d, deltas["reward"]),
            "rho_pcd": spearman(h_1d, deltas["pcd"]),
        }
    print("Per-axis 1-D Hamming Spearman rho with |dzvf|:")
    for item in [it for it, _ in V2_STACK_ITEMS]:
        d = per_axis_rho[item]
        print(
            f"  {item:14s}  n_unique={d['n_unique']}  "
            f"rho_zvf={d['rho_zvf']:+.4f}  rho_reward={d['rho_reward']:+.4f}"
        )
    print()

    # ------------------------------------------------------------------
    # PART 5: bootstrap CIs on the headline numbers.
    # ------------------------------------------------------------------
    n = n_cells

    # Recompute both distance arrays on the same resampled CELLS. The cell
    # independence assumption is explicit; correlated experimental blocks need
    # a block inventory before these can be interpreted as population intervals.
    def cell_ci(fp, outcome):
        return cell_distance_spearman_ci(
            [fp(c) for c in cells], [c[outcome] for c in cells], N_BOOT, 20260705
        )

    boot_ci = {}
    for label, fp, distances in [
        ("v1", fp_v1, h_v1),
        ("v2", fp_v2, h_v2),
        ("v2tn", fp_v2_truly_new, h_v2tn),
    ]:
        for channel, outcome in [("zvf", "zvf"), ("reward", "mean_reward")]:
            boot_ci[f"rho_{label}_{channel}"] = (
                spearman(distances, deltas[channel]),
                *cell_ci(fp, outcome),
            )

    # ------------------------------------------------------------------
    # OUTPUTS
    # ------------------------------------------------------------------
    tsv_path = OUT_DIR / "p5_stack_axis_minreport.tsv"
    with tsv_path.open("w") as f:
        f.write(
            "schema\titem\tweight\tn_unique\tH_bits\tstack_discriminative_bits\tclassification\n"
        )
        for p in per_item_v1:
            f.write(
                f"v1\t{p['item']}\t{V1_WEIGHTS[p['item']]}\t{p['n_unique']}\t"
                f"{p['H_bits']:.6f}\t{p['stack_discriminative_bits']:.6f}\t"
                f"{p['classification']}\n"
            )
        for p in per_item_v2:
            f.write(
                f"v2\t{p['item']}\t{p['weight']}\t{p['n_unique']}\t"
                f"{p['H_bits']:.6f}\t{p['stack_discriminative_bits']:.6f}\t"
                f"{p['classification']}\n"
            )
    print(f"wrote {tsv_path}")

    boot_path = OUT_DIR / "p5_stack_axis_minreport_boot.tsv"
    with boot_path.open("w") as f:
        f.write("metric\tpoint\tci_lo\tci_hi\tstatus\n")
        for k, (point, lo, hi) in boot_ci.items():
            f.write(
                f"{k}\t{point:+.4f}\t{lo:+.4f}\t{hi:+.4f}\t{'ok' if math.isfinite(point) else 'insufficient_data'}\n"
            )
        # H_bits per v2 axis
        for item, _ in V2_STACK_ITEMS:
            vals = [str(c[item]) for c in cells]
            h = shannon_entropy_bits(vals)
            f.write(f"H_bits_{item}\t{h:.4f}\tNA\tNA\tdescriptive\n")
    print(f"wrote {boot_path}")

    def show_rho(value):
        return (
            f"{value:+.4f}"
            if math.isfinite(value)
            else "undefined (insufficient_data: constant or missing ranks)"
        )

    h2_finite = all(math.isfinite(r["zvf"]) for r in (rho_v1, rho_v2, rho_v2tn))
    h2_description = (
        f"Observed rho difference: all axes={rho_v2['zvf'] - rho_v1['zvf']:+.4f}; "
        f"new axes={rho_v2tn['zvf'] - rho_v1['zvf']:+.4f}. Descriptive association, not independent explanatory information."
        if h2_finite
        else "insufficient_data: at least one fingerprint/outcome rank vector is constant; correlation differences are undefined."
    )
    summary = {
        "statistic_version": "pro-review-02-v2.1",
        "entropy_estimand": "sum_of_marginal_entropies; fixed-campaign descriptive diversity, not joint information",
        "correlation_uncertainty": "cell-bootstrap sensitivity assuming independent cells; does not cover training-seed or evaluation-example populations",
        "n_pairs": len(pairs),
        "reporting_contract": "canonical typed content; badge uses minreport_auditor",
        "correlation_status": {
            k: "ok" if math.isfinite(v[0]) else "insufficient_data" for k, v in boot_ci.items()
        },
        "iter": 73,
        "n_cells": n_cells,
        "v1_total_h_bits": h_v1_total,
        "v1_stack_discriminative_h_bits": h_v1_stack,
        "v2_stack_axes_h_bits": h_v2_stack_axes,
        "v2_total_h_bits": h_v2_total,
        "v2_stack_discriminative_h_bits": h_v2_stack_total,
        "v2_truly_new_h_bits": sum(
            shannon_entropy_bits([str(c[k]) for c in cells]) for k, _ in V2_TRULY_NEW
        ),
        "v1_constant_descriptive_items": sum(
            1 for p in per_item_v1 if p["classification"] == "CONSTANT_DESCRIPTIVE"
        ),
        "v1_varying_items": sum(
            1 for p in per_item_v1 if p["classification"] == "VARYING_STACK_DESCRIPTOR"
        ),
        "v2_total_weight_pts": V2_TOTAL_WEIGHT,
        "v2_truly_new_weight_pts": V2_TRULY_NEW_WEIGHT,
        "spearman_v1": rho_v1,
        "spearman_v2": rho_v2,
        "spearman_v2_tn": rho_v2tn,
        "per_axis_1d_rho": per_axis_rho,
        "badge_v1_mean": statistics.mean(badges_v1),
        "badge_v2_mean": statistics.mean(badges_v2),
        "badge_v2_tn_mean": statistics.mean(badges_v2tn),
        "badge_uplift_v2_pts": V2_TOTAL_WEIGHT,
        "badge_uplift_v2_tn_pts": V2_TRULY_NEW_WEIGHT,
        "boot_ci": {k: list(v) for k, v in boot_ci.items()},
        "fingerprint_length_v1": len(V1_ITEMS),
        "fingerprint_length_v2": len(V1_ITEMS) + len(V2_STACK_ITEMS),
        "fingerprint_length_v2_tn": len(V1_ITEMS) + len(V2_TRULY_NEW),
        "verdict_H1": (
            "v2 stack axes add {:.2f} marginal-entropy bits; v2 sum of marginal entropies = "
            "{:.2f} bits (vs v1 {:.2f} bits), +{:.1f}% uplift".format(
                h_v2_stack_axes,
                h_v2_total,
                h_v1_total,
                100 * (h_v2_total - h_v1_total) / max(h_v1_total, 1e-9),
            )
        ),
        "verdict_H2": h2_description,
        "verdict_H3": (
            "badge uplift is deterministic +%d pts (v2) / +%d pts "
            "(v2_tn) on every cell; bootstrap CI on per-cell "
            "uplift is degenerate." % (V2_TOTAL_WEIGHT, V2_TRULY_NEW_WEIGHT)
        ),
        "verdict_H4_per_axis": (
            "Per-axis 1-D Hamming rho with |dzvf|: "
            + ", ".join(f"{k}={show_rho(per_axis_rho[k]['rho_zvf'])}" for k, _ in V2_STACK_ITEMS)
        ),
    }
    summary_path = OUT_DIR / "p5_stack_axis_minreport_summary.json"
    summary_path.write_text(json.dumps(finite_json(summary), indent=2, allow_nan=False))
    print(f"wrote {summary_path}")
    print()
    print("Summary:")
    print(f"  H1: {summary['verdict_H1']}")
    print(f"  H2: {summary['verdict_H2']}")
    print(f"  H3: {summary['verdict_H3']}")
    print(f"  H4: {summary['verdict_H4_per_axis']}")


if __name__ == "__main__":
    main()
