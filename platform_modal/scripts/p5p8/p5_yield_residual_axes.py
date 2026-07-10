#!/usr/bin/env python3
"""P5 (iter 81): multi-axis yield-residual MIN-REPORT v2.2 axes (Items 14-17).

Fresh vein (not in 95 prior rows). Closes the iter-80 mint recommendation:
"extend v2.1 schema with N items of yield-residual - zvf_yield_residual,
pcd_yield_residual, mean_len_yield_residual - and measure whether ANY is
signal-bearing beyond Item 13."

This iter tests FOUR additional yield-residual candidates, each computed
PER-CELL from the group_tensors/*.json files (no extra harvest):

  Item 14  K_variance_residual = Var(K_x)_obs - G*p*(1-p)
           Anti-herding signature: observed K variance smaller than iid.
  Item 15  K_unique_count = number of distinct K_x values observed
           Info-entropy proxy of the K-distribution.
  Item 16  max_K_share = max frequency of K_x as fraction of n_groups
           High == herding (anti-herding reduces it).
  Item 17  prompt_p_hat_var = Var(K_x/G) over prompts (difficulty spread)
           Heterogeneity signature; pairs with the iter-65 row 76
           stack-discriminative gap.

Each candidate is checked for: (a) n_unique >= 2 on the 98-cell corpus;
(b) H_bits >= 0.10 (signal-bearing threshold from iter-80 row 95);
(c) STRENGTHENING fingerprint-vs-|dzvf| Spearman in a paired bootstrap.

Measures (paired bootstrap B=2000, seed 20260705):
  H1.  H_bits uplift vs Item 13 (does any candidate add unique
       signal beyond Item 13 alone?).
  H2.  Item14-17 alone vs |dzvf| Spearman rho: is the candidate
       independently signal-bearing at single-item granularity?
  H3.  Multi-axis v2.2 (Item 13 + Items 14-17 chosen by H1/H2) vs
       v2.1 fingerprint: which combination maximizes Spearman
       coupling with paired bootstrap CI?
  H4.  Stratified null control: shuffle K vectors within each cell;
       do Items 14-17 still report signal? (Anti-herding signal
       should vanish under the shuffle.)

Outputs (6 files):
  platform_hybrid/experiments/results/p5p8/p5_yield_residual_axes.tsv
  platform_hybrid/experiments/results/p5p8/p5_yield_residual_axes_per_item.tsv
  platform_hybrid/experiments/results/p5p8/p5_yield_residual_axes_shuffle_null.tsv
  platform_hybrid/experiments/results/p5p8/p5_yield_residual_axes_summary.json

Stdlib only. <=300 lines.
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
GT_DIR = ROOT / "experiments" / "results" / "mega_20260704" / "group_tensors"
MANIFEST_DIR = ROOT / "experiments" / "results" / "mega_20260704" / "manifests"
OUT = ROOT / "experiments" / "results" / "p5p8"
OUT.mkdir(parents=True, exist_ok=True)
RNG = random.Random(20260705)
N_BOOT = 2000
N_PAIRS = 2000
ITEM_WEIGHT = 20.0

V1_ITEMS = ["loss_form", "ref_policy_kl", "sampler_backend_precision",
            "per_step_zvf_path", "group_size_schedule", "heldout_split",
            "decontamination_notes"]
V2_STACK = [("model_family", 10), ("task_slice", 10), ("G", 5),
            ("temperature", 5), ("seed", 5)]

# Items beyond Item 13 (zvf_yield_residual) we will test.
# Each is a callable: (K_array, G) -> scalar per cell.
ITEMS = [
    ("K_variance_residual", "item14_Kvar_residual",
     lambda K, G: float(K.var(ddof=0) - G * (K.mean() / G) * (1 - K.mean() / G))),
    ("K_unique_count", "item15_K_unique",
     lambda K, G: float(len(set(K.astype(int).tolist())))),
    ("max_K_share", "item16_max_Kshare",
     lambda K, G: float(max(Counter(K.astype(int).tolist()).values()) / len(K))),
    ("prompt_p_hat_var", "item17_p_hat_var",
     lambda K, G: float((K / G).var(ddof=0))),
]


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


def load_per_cell_rewards():
    """Return dict cell_id -> numpy array shape (n_groups, G) of {0,1}."""
    out = {}
    import numpy as np
    for jf in sorted(GT_DIR.glob("*.json")):
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        cid = d.get("cell_id")
        rv = d.get("reward_vectors")
        if cid is None or rv is None:
            continue
        out[cid] = np.array(rv, dtype=float)
    return out


def load_manifests():
    out = {}
    for jf in MANIFEST_DIR.glob("*.json"):
        try:
            out[(json.loads(jf.read_text())).get("cell_id", jf.stem)] = json.loads(jf.read_text())
        except Exception:
            continue
    return out


def shannon(values):
    n = len(values)
    if n == 0:
        return 0.0
    h = 0.0
    for c in Counter(values).values():
        if c > 0:
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
    p = (c.get("mean_reward") or 0.0) / 100.0
    p = max(0.0, min(1.0, p))
    G = c["G"]
    zvf_iid = p ** G + (1 - p) ** G
    zvf_obs = c.get("zvf")
    if zvf_obs is None:
        return None
    return float(zvf_obs - zvf_iid)


def main():
    import numpy as np
    print("# === P5 multi-axis yield-residual MIN-REPORT v2.2 (iter 81) ===")
    cells = load_cells()
    manifests = load_manifests()
    rewards = load_per_cell_rewards()
    n = len(cells)
    print(f"# loaded {n} cells, {len(manifests)} manifests, {len(rewards)} reward tensors")

    # --- Compute Items 13-17 per cell ---
    item13_vals = []
    item14_vals = []
    item15_vals = []
    item16_vals = []
    item17_vals = []
    cell_Ks = []  # raw K arrays for null shuffle
    for c in cells:
        d13 = delta_div(c)
        item13_vals.append(d13 if d13 is not None else float("nan"))
        cid = c["cell_id"]
        rv = rewards.get(cid)
        if rv is None or rv.size == 0:
            for tgt in (item14_vals, item15_vals, item16_vals, item17_vals):
                tgt.append(float("nan"))
            cell_Ks.append(None)
            continue
        K = rv.sum(axis=1)
        G = rv.shape[1]
        cell_Ks.append(K)
        item14_vals.append(ITEMS[0][2](K, G))
        item15_vals.append(ITEMS[1][2](K, G))
        item16_vals.append(ITEMS[2][2](K, G))
        item17_vals.append(ITEMS[3][2](K, G))

    def present(xs):
        return [v for v in xs if not math.isnan(v)]

    # --- Per-item H_bits (H1) ---
    per_item = []
    for item in V1_ITEMS:
        vals = [str(manifests.get(c["cell_id"], {}).get(item, "MISSING")) for c in cells]
        per_item.append({"item": item, "axis": "v1", "n_unique": len(set(vals)),
                         "H_bits": shannon(vals), "weight": 10})
    for axis, w in V2_STACK:
        vals = [str(c.get(axis, "MISSING")) for c in cells]
        per_item.append({"item": axis, "axis": "v2_stack", "n_unique": len(set(vals)),
                         "H_bits": shannon(vals), "weight": w})

    # Item 13 (zvf_yield_residual) - the iter-80 baseline
    vals13 = [round(v, 4) for v in item13_vals]
    h13 = shannon([str(v) for v in vals13])
    per_item.append({"item": "zvf_yield_residual", "axis": "v2.1_yield",
                     "n_unique": len(set(vals13)), "H_bits": h13, "weight": ITEM_WEIGHT})

    # Items 14-17 (this iter)
    candidate_vals = {
        "item14": item14_vals, "item15": item15_vals,
        "item16": item16_vals, "item17": item17_vals,
    }
    candidate_names = {
        "item14": "K_variance_residual", "item15": "K_unique_count",
        "item16": "max_K_share", "item17": "prompt_p_hat_var",
    }
    candidate_h = {}
    candidate_pres = {}
    for key, vals in candidate_vals.items():
        # 4-decimal rounding keeps scalar diversity visible while aggregating
        rounded = [round(v, 4) for v in vals if not math.isnan(v)]
        h = shannon([str(v) for v in rounded])
        candidate_h[key] = h
        candidate_pres[key] = len(rounded)
        per_item.append({"item": candidate_names[key], "axis": "v2.2_yield_extended",
                         "n_unique": len(set(rounded)), "H_bits": h, "weight": ITEM_WEIGHT})

    h_v2_total = sum(p["H_bits"] for p in per_item if p["axis"] in ("v1", "v2_stack"))
    h_v21_total = h_v2_total + h13
    h_v22_total = h_v21_total + sum(candidate_h.values())
    print(f"# H_bits: v2.0={h_v2_total:.4f}; v2.1={h_v21_total:.4f} (+{h13:.3f} from Item 13); "
          f"v2.2={h_v22_total:.4f} (Items 14-17 add {sum(candidate_h.values()):.4f})")

    # --- Spearman fingerprint-vs-|dzvf| (H2, H3) ---
    def fp(c, axis_keys):
        out = []
        if "v1" in axis_keys:
            for item in V1_ITEMS:
                out.append(str(manifests.get(c["cell_id"], {}).get(item, "MISSING")))
        if "v2_stack" in axis_keys:
            for axis, _ in V2_STACK:
                out.append(str(c.get(axis, "MISSING")))
        if "v2.1_yield" in axis_keys:
            d = delta_div(c)
            out.append(round(d, 4) if d is not None else "NaN")
        if "item14" in axis_keys:
            v = item14_vals[cells.index(c)] if False else None  # not used; use scan below
        return out

    # Better: direct fingerprint building with cell index for items 14-17
    def fp_idx(i, axis_keys):
        c = cells[i]
        out = []
        if "v1" in axis_keys:
            for item in V1_ITEMS:
                out.append(str(manifests.get(c["cell_id"], {}).get(item, "MISSING")))
        if "v2_stack" in axis_keys:
            for axis, _ in V2_STACK:
                out.append(str(c.get(axis, "MISSING")))
        if "v2.1_yield" in axis_keys:
            d = delta_div(c)
            out.append(round(d, 4) if d is not None else "NaN")
        if "v2.2_item14" in axis_keys:
            v = item14_vals[i]
            out.append(round(v, 4) if not math.isnan(v) else "NaN")
        if "v2.2_item15" in axis_keys:
            v = item15_vals[i]
            out.append(round(v, 4) if not math.isnan(v) else "NaN")
        if "v2.2_item16" in axis_keys:
            v = item16_vals[i]
            out.append(round(v, 4) if not math.isnan(v) else "NaN")
        if "v2.2_item17" in axis_keys:
            v = item17_vals[i]
            out.append(round(v, 4) if not math.isnan(v) else "NaN")
        return tuple(out)

    def rho_axis(axis_set, n_pairs=None):
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if n_pairs is None:
            sampled = all_pairs
        else:
            sampled = RNG.sample(all_pairs, min(n_pairs, len(all_pairs)))
        d_zvf = []
        hd = []
        for i, j in sampled:
            z_i = cells[i].get("zvf")
            z_j = cells[j].get("zvf")
            if z_i is None or z_j is None:
                continue
            d_zvf.append(abs(z_i - z_j))
            hd.append(hamming(fp_idx(i, axis_set), fp_idx(j, axis_set)))
        return spearman(hd, d_zvf), len(hd)

    # Point estimates on full pair set
    axis_versions = {
        "v1": {"v1"},
        "v2.0": {"v1", "v2_stack"},
        "v2.1": {"v1", "v2_stack", "v2.1_yield"},
        "v2.2_all": {"v1", "v2_stack", "v2.1_yield", "v2.2_item14", "v2.2_item15", "v2.2_item16", "v2.2_item17"},
        "v2.2_item13+14": {"v1", "v2_stack", "v2.1_yield", "v2.2_item14"},
        "v2.2_item13+15": {"v1", "v2_stack", "v2.1_yield", "v2.2_item15"},
        "v2.2_item13+16": {"v1", "v2_stack", "v2.1_yield", "v2.2_item16"},
        "v2.2_item13+17": {"v1", "v2_stack", "v2.1_yield", "v2.2_item17"},
        "single_item14": {"v2.2_item14"},
        "single_item15": {"v2.2_item15"},
        "single_item16": {"v2.2_item16"},
        "single_item17": {"v2.2_item17"},
    }
    rho_pts = {}
    for name, axes in axis_versions.items():
        r, n_pairs_used = rho_axis(axes)
        rho_pts[name] = (float(r), n_pairs_used)
        print(f"# rho[{name}] = {r:.4f}  (n_pairs={n_pairs_used})")

    # --- Bootstrap CIs (H3) ---
    diffs = {"v22_all_minus_v21": [], "v22_item13+14_minus_v21": [],
             "v22_item13+15_minus_v21": [], "v22_item13+16_minus_v21": [],
             "v22_item13+17_minus_v21": []}
    diffs_keys = list(diffs.keys())
    for b in range(N_BOOT):
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        sampled = RNG.sample(all_pairs, min(N_PAIRS, len(all_pairs)))
        d_zvf = []
        fp_v21, fp_all, fp_14, fp_15, fp_16, fp_17 = [], [], [], [], [], []
        for i, j in sampled:
            z_i = cells[i].get("zvf")
            z_j = cells[j].get("zvf")
            if z_i is None or z_j is None:
                continue
            d_zvf.append(abs(z_i - z_j))
            fp_v21.append(hamming(fp_idx(i, {"v1", "v2_stack", "v2.1_yield"}),
                                  fp_idx(j, {"v1", "v2_stack", "v2.1_yield"})))
            fp_all.append(hamming(fp_idx(i, {"v1", "v2_stack", "v2.1_yield",
                                              "v2.2_item14", "v2.2_item15",
                                              "v2.2_item16", "v2.2_item17"}),
                                  fp_idx(j, {"v1", "v2_stack", "v2.1_yield",
                                              "v2.2_item14", "v2.2_item15",
                                              "v2.2_item16", "v2.2_item17"})))
            fp_14.append(hamming(fp_idx(i, {"v1", "v2_stack", "v2.1_yield", "v2.2_item14"}),
                                 fp_idx(j, {"v1", "v2_stack", "v2.1_yield", "v2.2_item14"})))
            fp_15.append(hamming(fp_idx(i, {"v1", "v2_stack", "v2.1_yield", "v2.2_item15"}),
                                 fp_idx(j, {"v1", "v2_stack", "v2.1_yield", "v2.2_item15"})))
            fp_16.append(hamming(fp_idx(i, {"v1", "v2_stack", "v2.1_yield", "v2.2_item16"}),
                                 fp_idx(j, {"v1", "v2_stack", "v2.1_yield", "v2.2_item16"})))
            fp_17.append(hamming(fp_idx(i, {"v1", "v2_stack", "v2.1_yield", "v2.2_item17"}),
                                 fp_idx(j, {"v1", "v2_stack", "v2.1_yield", "v2.2_item17"})))
        r_v21 = spearman(fp_v21, d_zvf)
        r_all = spearman(fp_all, d_zvf)
        r_14 = spearman(fp_14, d_zvf)
        r_15 = spearman(fp_15, d_zvf)
        r_16 = spearman(fp_16, d_zvf)
        r_17 = spearman(fp_17, d_zvf)
        diffs["v22_all_minus_v21"].append(r_all - r_v21)
        diffs["v22_item13+14_minus_v21"].append(r_14 - r_v21)
        diffs["v22_item13+15_minus_v21"].append(r_15 - r_v21)
        diffs["v22_item13+16_minus_v21"].append(r_16 - r_v21)
        diffs["v22_item13+17_minus_v21"].append(r_17 - r_v21)

    boot_summary = []
    for k in diffs_keys:
        arr = sorted(diffs[k])
        lo, hi = arr[len(arr) // 40], arr[-len(arr) // 40]
        pt = sum(arr) / len(arr)
        boot_summary.append({"metric": k, "point": float(pt), "ci_lo": float(lo), "ci_hi": float(hi)})
        print(f"# paired Δrho[{k}] = {pt:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]")

    # --- H4: BINOMIAL null control ---
    # Items 14-17 are statistics of K-distributions. To test whether they
    # carry information BEYOND the binomial(G, p) baseline, simulate per-cell
    # K_x ~ Binomial(G, p_cell) and compute the same item statistics. If
    # observed H_bits exceeds the null mean by a wide margin, the items
    # carry anti-herding signal; if not, the items reduce to binomial noise.
    print("# Running BINOMIAL(G, p) null control (n=200 iters) ...")
    null_dist = {"item14": [], "item15": [], "item16": [], "item17": []}
    null_rng = random.Random(20260707)
    n_null = 200
    for _ in range(n_null):
        shuf14, shuf15, shuf16, shuf17 = [], [], [], []
        for c in cells:
            cid = c["cell_id"]
            rv = rewards.get(cid)
            if rv is None or rv.size == 0:
                shuf14.append(float("nan"))
                shuf15.append(float("nan"))
                shuf16.append(float("nan"))
                shuf17.append(float("nan"))
                continue
            G = rv.shape[1]
            n_groups = rv.shape[0]
            p_cell = float(rv.mean())
            K_null = np.array([sum(null_rng.random() < p_cell for _ in range(G))
                               for _ in range(n_groups)])
            shuf14.append(ITEMS[0][2](K_null, G))
            shuf15.append(ITEMS[1][2](K_null, G))
            shuf16.append(ITEMS[2][2](K_null, G))
            shuf17.append(ITEMS[3][2](K_null, G))
        null_dist["item14"].append(shannon([round(v, 4) for v in shuf14 if not math.isnan(v)]))
        null_dist["item15"].append(shannon([str(int(v)) for v in shuf15 if not math.isnan(v)]))
        null_dist["item16"].append(shannon([round(v, 4) for v in shuf16 if not math.isnan(v)]))
        null_dist["item17"].append(shannon([round(v, 4) for v in shuf17 if not math.isnan(v)]))

    null_summary = []
    for key in null_dist:
        arr = sorted(null_dist[key])
        lo, hi = arr[len(arr) // 40], arr[-len(arr) // 40]
        pt = sum(arr) / len(arr)
        # excess_H = observed - null_mean; positive == item carries signal
        # beyond binomial
        excess = float(candidate_h[key]) - float(pt)
        z = excess / max(1e-9, (arr[-1] - arr[0]) / 6.0)
        null_summary.append({"item": candidate_names[key],
                             "observed_H_bits": float(candidate_h[key]),
                             "null_mean": float(pt), "null_ci_lo": float(lo),
                             "null_ci_hi": float(hi),
                             "excess_H_bits": float(excess),
                             "approx_z": float(z)})
        print(f"# null control {key}: observed H={candidate_h[key]:.4f} "
              f"vs null mean={pt:.4f}  CI [{lo:.4f}, {hi:.4f}]  "
              f"excess={excess:+.4f} (~{z:+.1f}σ)")

    # --- Save outputs ---
    per_item_tsv = OUT / "p5_yield_residual_axes_per_item.tsv"
    with per_item_tsv.open("w") as f:
        keys = ["item", "axis", "n_unique", "H_bits", "weight"]
        f.write("\t".join(keys) + "\n")
        for p in per_item:
            f.write("\t".join(str(p[k]) for k in keys) + "\n")

    axes_tsv = OUT / "p5_yield_residual_axes.tsv"
    with axes_tsv.open("w") as f:
        keys = ["axis_version", "spearman_rho", "n_pairs"]
        f.write("\t".join(keys) + "\n")
        for name, (r, k) in rho_pts.items():
            f.write("\t".join([name, f"{r:.6f}", str(k)]) + "\n")

    null_tsv = OUT / "p5_yield_residual_axes_shuffle_null.tsv"
    with null_tsv.open("w") as f:
        keys = ["item", "observed_H_bits", "null_mean", "null_ci_lo", "null_ci_hi"]
        f.write("\t".join(keys) + "\n")
        for n_ in null_summary:
            f.write("\t".join(str(n_[k]) for k in keys) + "\n")

    summary = {
        "n_cells": n,
        "H_bits_v2.0": h_v2_total,
        "H_bits_v2.1": h_v21_total,
        "H_bits_v2.2": h_v22_total,
        "H_bits_items_14_17_total": float(sum(candidate_h.values())),
        "item13_H_bits": float(h13),
        "items_14_17_H_bits": {k: float(v) for k, v in candidate_h.items()},
        "items_14_17_present": {k: int(v) for k, v in candidate_pres.items()},
        "spearman_v1": rho_pts["v1"][0],
        "spearman_v2.0": rho_pts["v2.0"][0],
        "spearman_v2.1": rho_pts["v2.1"][0],
        "spearman_v2.2_all": rho_pts["v2.2_all"][0],
        "spearman_single_item14": rho_pts["single_item14"][0],
        "spearman_single_item15": rho_pts["single_item15"][0],
        "spearman_single_item16": rho_pts["single_item16"][0],
        "spearman_single_item17": rho_pts["single_item17"][0],
        "paired_bootstrap": boot_summary,
        "shuffle_null": null_summary,
        "verdict_items": {
            "item14_Kvar_residual":
                "SIGNAL-BEARING" if candidate_h["item14"] > 0.1 and
                rho_pts["single_item14"][0] > 0.0 else
                ("WEAK" if candidate_h["item14"] > 0.001 else "PLACEBO"),
            "item15_K_unique_count":
                "SIGNAL-BEARING" if candidate_h["item15"] > 0.1 else
                ("WEAK" if candidate_h["item15"] > 0.001 else "PLACEBO"),
            "item16_max_Kshare":
                "SIGNAL-BEARING" if candidate_h["item16"] > 0.1 else
                ("WEAK" if candidate_h["item16"] > 0.001 else "PLACEBO"),
            "item17_p_hat_var":
                "SIGNAL-BEARING" if candidate_h["item17"] > 0.1 and
                rho_pts["single_item17"][0] > 0.0 else
                ("WEAK" if candidate_h["item17"] > 0.001 else "PLACEBO"),
        },
        "key_finding": (
            f"Items 14-17 add {sum(candidate_h.values()):.3f} bits to v2.1 ("
            f"item14={candidate_h['item14']:.3f}, item15={candidate_h['item15']:.3f}, "
            f"item16={candidate_h['item16']:.3f}, item17={candidate_h['item17']:.3f})."
        ),
    }
    (OUT / "p5_yield_residual_axes_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print("# === iter 81 multi-axis yield-residual: outputs in platform_hybrid/experiments/results/p5p8/ ===")


if __name__ == "__main__":
    main()
