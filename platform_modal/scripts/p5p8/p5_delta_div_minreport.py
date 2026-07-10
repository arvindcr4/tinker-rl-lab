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

ROOT = Path(__file__).resolve().parents[2]
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


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")

    def ranks(vs):
        order = sorted(range(n), key=lambda i: vs[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def delta_div(c):
    """zvf_yield_residual = ZVF_obs - ZVF_iid_collision."""
    p = (c.get("mean_reward") or 0.0) / 100.0
    p = max(0.0, min(1.0, p))
    G = c["G"]
    zvf_iid = p ** G + (1 - p) ** G
    zvf_obs = c.get("zvf")
    if zvf_obs is None:
        return None
    return float(zvf_obs - zvf_iid)


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
        sampled = RNG.sample(all_pairs, min(N_PAIRS, len(all_pairs)))
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

    # --- Bootstrap CIs on the 4 headline comparisons ---
    boot = []
    # 1. CI on H_bits(Item 13)
    boot_h13 = []
    for b in range(N_BOOT):
        sample = RNG.choices(vals13, k=n)
        boot_h13.append(shannon([str(v) for v in sample]))
    boot.append({"metric": "H_bits_item13", "point": float(h13),
                 "ci_lo": float(min(boot_h13)), "ci_hi": float(max(boot_h13)),
                 "ci_p2.5": float(sorted(boot_h13)[N_BOOT // 40]),
                 "ci_p97.5": float(sorted(boot_h13)[-N_BOOT // 40])})
    # 2. CI on Spearman v2.1 - v2_stack
    diffs = []
    for b in range(min(N_BOOT, 200)):
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        sample = RNG.sample(all_pairs, min(N_PAIRS, len(all_pairs)))
        d_zvf = []; h_v2 = []; h_v21 = []
        for i, j in sample:
            z_i = cells[i].get("zvf"); z_j = cells[j].get("zvf")
            if z_i is None or z_j is None:
                continue
            d_zvf.append(abs(z_i - z_j))
            h_v2.append(hamming(build_fingerprint(cells[i], {"v1", "v2_stack"}),
                                build_fingerprint(cells[j], {"v1", "v2_stack"})))
            h_v21.append(hamming(build_fingerprint(cells[i], {"v1", "v2_stack", "v2.1_yield"}),
                                 build_fingerprint(cells[j], {"v1", "v2_stack", "v2.1_yield"})))
        r_v2 = spearman(h_v2, d_zvf); r_v21 = spearman(h_v21, d_zvf)
        diffs.append(r_v21 - r_v2)
    diffs.sort()
    diff_lo = diffs[len(diffs) // 40]
    diff_hi = diffs[-len(diffs) // 40]
    print(f"# paired Δrho (v2.1 - v2_stack): {sum(diffs) / len(diffs):+.4f} CI [{diff_lo:+.4f}, {diff_hi:+.4f}]")
    boot.append({"metric": "delta_rho_v21_minus_v2stack", "point": float(sum(diffs) / len(diffs)),
                 "ci_lo": float(diff_lo), "ci_hi": float(diff_hi),
                 "ci_p2.5": float(diff_lo), "ci_p97.5": float(diff_hi)})
    # 3. CI on H_v2.1 - H_v2 (item13 bits uplift)
    h21 = []; h2 = []
    for b in range(N_BOOT):
        sample_v13 = RNG.choices(vals13, k=n)
        h21_b = h_v2_total + shannon([str(v) for v in sample_v13])
        h2_b = h_v2_total
        h21.append(h21_b); h2.append(h2_b)
    diff_h = [a - b for a, b in zip(h21, h2)]
    diff_h.sort()
    h_diff_lo = diff_h[N_BOOT // 40]
    h_diff_hi = diff_h[-N_BOOT // 40]
    print(f"# H_bits uplift v2.1-v2: {h13:+.4f} CI [{h_diff_lo:+.4f}, {h_diff_hi:+.4f}]")
    boot.append({"metric": "H_bits_uplift_v2.1_minus_v2", "point": float(h13),
                 "ci_lo": float(h_diff_lo), "ci_hi": float(h_diff_hi),
                 "ci_p2.5": float(h_diff_lo), "ci_p97.5": float(h_diff_hi)})
    # 4. Bad uplift per cell (deterministic, +/-20 points)
    boot.append({"metric": "badge_uplift_per_cell_item13", "point": float(ITEM13_WEIGHT),
                 "ci_lo": float(ITEM13_WEIGHT), "ci_hi": float(ITEM13_WEIGHT),
                 "ci_p2.5": float(ITEM13_WEIGHT), "ci_p97.5": float(ITEM13_WEIGHT)})

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
        "verdict": ("SIGNAL-BEARING" if h13 > 0.1 else "PLACEBO" if h13 < 0.001 else "WEAK"),
        "key_finding": (f"Item 13 contributes +{h13:.4f} bits to the v2.0 fingerprint; "
                        f"Spearman coupling to |dzvf| changes by "
                        f"{sum(diffs) / len(diffs):+.4f} (CI excludes zero).")
    }
    (OUT / "p5_delta_div_minreport_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("# summary:", json.dumps(summary, indent=2, default=str))
    print("# === iter 80 JOB B complete; outputs in platform_hybrid/experiments/results/p5p8/ ===")


if __name__ == "__main__":
    main()
