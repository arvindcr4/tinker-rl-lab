#!/usr/bin/env python3
"""P5 — MIN-REPORT coverage x measured-telemetry multivariate coupling on the
98-cell mega corpus (iter 65 fresh vein).

Three falsifiable sub-claims:

  H1 (badge -> seed-pair reproducibility):
     Across the 49 cell-groups with >=2 seeds, higher manifest badge
     predicts LOWER |delta_reward| between seeds (within-group dispersion).
     Spearman rho with bootstrap 95% CI.

  H2 (per-item information content on the LIVE corpus):
     For each of the 7 MIN-REPORT items, compute unique-value count,
     Shannon entropy (bits), and whether the field is a "placebo"
     (single value or n/a placeholder across the entire 98-cell corpus).
     Reports the live-campaign info distribution.

  H3 (joint manifest fingerprint -> measured telemetry):
     Build a per-cell manifest fingerprint (7 items, each value
     discretised), compute Manhattan / Hamming pairwise distance between
     cells, and test whether close-fingerprint cells have closer
     measured telemetry (mean_reward, zvf, pcd, mean_completion_len) than
     far-fingerprint cells. Permutation p-value.

Outputs:
  experiments/results/p5p8/p5_manifest_outcome_coupling.tsv
  experiments/results/p5p8/p5_manifest_outcome_coupling_boot.tsv
  experiments/results/p5p8/p5_manifest_outcome_coupling_summary.json
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = ROOT / "experiments" / "results" / "mega_20260704" / "manifests"
CELLS_TSV = ROOT / "experiments" / "results" / "mega_20260704" / "cells.tsv"
OUT_DIR = ROOT / "experiments" / "results" / "p5p8"

RNG = random.Random(20260705)
N_BOOT = 5000

# 7-item MIN-REPORT standard (matches paper_P5 / auditor)
ITEMS = [
    "loss_form",
    "ref_policy_kl",
    "sampler_backend_precision",
    "per_step_zvf_path",
"group_size_schedule",
    "heldout_split",
    "decontamination_notes",
]

# Weight schedule mirrors scripts/p5p8/minreport_auditor.py exactly
ITEM_WEIGHTS = {
    "loss_form": 10,
    "ref_policy_kl": 10,
    "sampler_backend_precision": 20,
    "per_step_zvf_path": 20,
    "group_size_schedule": 10,
    "heldout_split": 10,
    "decontamination_notes": 20,
}


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
        except Exception as e:
            print(f"warn: bad json {jf}: {e}", file=sys.stderr)
            continue
        cid = d.get("cell_id", jf.stem)
        out[cid] = d
    return out


def shannon_entropy_bits(values: list[str]) -> float:
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


def badge_for(manifest: dict) -> tuple[float, dict[str, float]]:
    """Mirror the auditor's per-item base*0.5*(1+sub_cov) score."""
    if manifest is None:
        return 0.0, {}
    per_item = {}
    for k, w in ITEM_WEIGHTS.items():
        v = manifest.get(k)
        if v is None or str(v).strip() == "":
            base = 0.0
        elif str(v).startswith("n/a"):
            base = 0.5
        elif k == "per_step_zvf_path":
            base = 1.0  # a path is always concrete when present
        else:
            base = 1.0
        per_item[k] = w * base * (0.5 + 0.5 * 0.5)  # sub_cov~0.5 default
    # Special: per_step_zvf_path baseline — full weight when path present
    if manifest.get("per_step_zvf_path"):
        per_item["per_step_zvf_path"] = ITEM_WEIGHTS["per_step_zvf_path"]
    return sum(per_item.values()), per_item


def fingerprint(manifest: dict) -> tuple:
    if manifest is None:
        return tuple("MISSING" for _ in ITEMS)
    return tuple(str(manifest.get(k, "MISSING")) for k in ITEMS)


def hamming(a: tuple, b: tuple) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def manhattan(a: tuple, b: tuple) -> int:
    return hamming(a, b)


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rho without scipy: rank-transform then Pearson."""
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
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def bootstrap_ci(values: list[float], stat_fn, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    n = len(values)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    boots = []
    for _ in range(n_boot):
        sample = [values[RNG.randrange(n)] for _ in range(n)]
        boots.append(stat_fn(sample))
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot) - 1]
    return stat_fn(values), lo, hi


def permutation_p(diff_obs: float, group_a: list[float], group_b: list[float],
                   n_perm: int = 5000) -> float:
    """Two-sided permutation p-value for the difference in means."""
    combined = group_a + group_b
    n_a = len(group_a)
    n = len(combined)
    abs_diffs = []
    for _ in range(n_perm):
        RNG.shuffle(combined)
        ma = statistics.mean(combined[:n_a])
        mb = statistics.mean(combined[n_a:])
        abs_diffs.append(abs(ma - mb))
    abs_diffs.sort()
    rank = sum(1 for x in abs_diffs if x >= abs(diff_obs))
    return (rank + 1) / (n_perm + 1)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    manifests = load_manifests()

    # ----- Attach manifest + badge to each cell
    enriched = []
    for c in cells:
        m = manifests.get(c["cell_id"])
        b, _ = badge_for(m)
        c2 = dict(c)
        c2["badge"] = b
        c2["has_manifest"] = m is not None
        c2["fingerprint"] = fingerprint(m)
        enriched.append(c2)

    # ===== H2: per-item information content on the LIVE 98-cell corpus =====
    h2_rows = []
    for item in ITEMS:
        vals = [str(manifests.get(c["cell_id"], {}).get(item, "MISSING")) for c in cells]
        n_unique = len(set(vals))
        h_bits = shannon_entropy_bits(vals)
        n_na = sum(1 for v in vals if v.startswith("n/a") or v == "MISSING")
        h2_rows.append({
            "item": item,
            "n_unique_values": n_unique,
            "entropy_bits": round(h_bits, 4),
            "n_na_or_missing": n_na,
            "fraction_na": round(n_na / len(vals), 4),
            "max_entropy_possible_bits": round(math.log2(len(vals)), 4),
            "placebo": n_unique <= 1,
        })
    # Total info budget
    total_bits = sum(r["entropy_bits"] for r in h2_rows)
    info_share = {r["item"]: round(r["entropy_bits"] / total_bits, 4)
                  if total_bits > 0 else 0.0 for r in h2_rows}

    # ===== H1: badge vs seed-pair reproducibility =====
    cell_groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in enriched:
        cell_groups[(c["model"], c["task_slice"], c["G"], c["temperature"])].append(c)

    h1_rows = []
    h1_pair_diffs = []  # (badge, |delta_reward|)
    for key, group in cell_groups.items():
        if len(group) < 2:
            continue
        seeds = sorted({c["seed"] for c in group})
        if len(seeds) < 2:
            continue
        # Pair every distinct seed pair
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                ci = next(c for c in group if c["seed"] == seeds[i])
                cj = next(c for c in group if c["seed"] == seeds[j])
                if ci["mean_reward"] is None or cj["mean_reward"] is None:
                    continue
                d = abs(ci["mean_reward"] - cj["mean_reward"])
                h1_pair_diffs.append((ci["badge"], d, key))
                h1_rows.append({
                    "model": key[0], "task": key[1], "G": key[2],
                    "temperature": key[3], "seed_i": seeds[i], "seed_j": seeds[j],
                    "badge": ci["badge"],  # same group -> same badge
                    "abs_delta_reward": round(d, 6),
                })

    # Spearman on (badge, |delta_reward|)
    if h1_pair_diffs:
        bs = [x[0] for x in h1_pair_diffs]
        ds = [x[1] for x in h1_pair_diffs]
        rho = spearman(bs, ds)
        # bootstrap CI on rho
        rho_boots = []
        n = len(bs)
        for _ in range(N_BOOT):
            idx = [RNG.randrange(n) for _ in range(n)]
            xs = [bs[i] for i in idx]
            ys = [ds[i] for i in idx]
            r = spearman(xs, ys)
            if not math.isnan(r):
                rho_boots.append(r)
        rho_boots.sort()
        rho_lo = rho_boots[int(0.025 * len(rho_boots))]
        rho_hi = rho_boots[int(0.975 * len(rho_boots)) - 1]
    else:
        rho, rho_lo, rho_hi = float("nan"), float("nan"), float("nan")

    # ===== H3: joint fingerprint -> measured telemetry (close-fingerprint
    # cells have closer telemetry than far-fingerprint cells) =====
    # Use 1000 random unordered cell-pairs (10 choose 2 over 98 is 4753,
    # manageable but we sample for speed)
    all_pairs = []
    n_cells = len(enriched)
    for _ in range(min(2000, n_cells * (n_cells - 1) // 2)):
        i, j = RNG.sample(range(n_cells), 2)
        all_pairs.append((i, j))

    # Build observed: per-pair (hamming, |delta_mean_reward|, |delta_zvf|, |delta_pcd|)
    obs_deltas = []
    obs_distances = []
    for i, j in all_pairs:
        ci, cj = enriched[i], enriched[j]
        if ci["mean_reward"] is None or cj["mean_reward"] is None:
            continue
        d_h = hamming(ci["fingerprint"], cj["fingerprint"])
        d_r = abs(ci["mean_reward"] - cj["mean_reward"])
        d_z = abs((ci["zvf"] or 0.0) - (cj["zvf"] or 0.0))
        d_p = abs((ci["pcd"] or 0.0) - (cj["pcd"] or 0.0))
        obs_distances.append(d_h)
        obs_deltas.append((d_r, d_z, d_p))

    if obs_deltas:
        # Spearman: hamming vs |delta_telemetry|
        rho_r = spearman(obs_distances, [x[0] for x in obs_deltas])
        rho_z = spearman(obs_distances, [x[1] for x in obs_deltas])
        rho_p = spearman(obs_distances, [x[2] for x in obs_deltas])
        # Close-vs-far: median split, compare |delta_telemetry|
        med = statistics.median(obs_distances)
        close_d = [d for h, d in zip(obs_distances, obs_deltas) if h <= med]
        far_d = [d for h, d in zip(obs_distances, obs_deltas) if h > med]
        # For each telemetry channel, permutation test close_mean - far_mean
        def m(seq): return statistics.mean(seq) if seq else float("nan")
        diff_r_obs = m([x[0] for x in close_d]) - m([x[0] for x in far_d])
        diff_z_obs = m([x[1] for x in close_d]) - m([x[1] for x in far_d])
        diff_p_obs = m([x[2] for x in close_d]) - m([x[2] for x in far_d])
        p_r = permutation_p(diff_r_obs, [x[0] for x in close_d], [x[0] for x in far_d])
        p_z = permutation_p(diff_z_obs, [x[1] for x in close_d], [x[1] for x in far_d])
        p_p = permutation_p(diff_p_obs, [x[2] for x in close_d], [x[2] for x in far_d])
    else:
        rho_r = rho_z = rho_p = float("nan")
        diff_r_obs = diff_z_obs = diff_p_obs = float("nan")
        p_r = p_z = p_p = float("nan")

    # ===== Per-item outcome correlation (informative items → telemetry) =====
    # For each item with >=2 unique values, split cells by value and
    # compare mean_reward / zvf / pcd between groups via permutation p.
    item_outcome_rows = []
    for item in ITEMS:
        vals = [str(manifests.get(c["cell_id"], {}).get(item, "MISSING")) for c in enriched]
        uniq = sorted(set(vals))
        if len(uniq) < 2:
            item_outcome_rows.append({
                "item": item, "n_unique": len(uniq),
                "p_value_mean_reward": float("nan"),
                "p_value_zvf": float("nan"),
                "p_value_pcd": float("nan"),
                "note": "placebo (<=1 unique value on live corpus)",
            })
            continue
        # Group cells by value
        per_val: dict[str, list[dict]] = defaultdict(list)
        for c, v in zip(enriched, vals):
            per_val[v].append(c)
        # Use Kruskal-Wallis-like approximation: max abs pairwise mean diff
        # vs permutation null
        def ch(scalar):
            groups = [[c[scalar] for c in per_val[v] if c[scalar] is not None]
                      for v in uniq]
            groups = [g for g in groups if g]
            if len(groups) < 2:
                return float("nan")
            max_abs = 0.0
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    max_abs = max(max_abs, abs(statistics.mean(groups[i]) - statistics.mean(groups[j])))
            combined = [v for g in groups for v in g]
            n_each = [len(g) for g in groups]
            null = []
            for _ in range(2000):
                RNG.shuffle(combined)
                idx = 0
                ms = []
                for ne in n_each:
                    ms.append(statistics.mean(combined[idx:idx + ne]))
                    idx += ne
                mmax = 0.0
                for i in range(len(ms)):
                    for j in range(i + 1, len(ms)):
                        mmax = max(mmax, abs(ms[i] - ms[j]))
                null.append(mmax)
            null.sort()
            rank = sum(1 for x in null if x >= max_abs)
            return (rank + 1) / (len(null) + 1)

        item_outcome_rows.append({
            "item": item,
            "n_unique": len(uniq),
            "p_value_mean_reward": round(ch("mean_reward"), 4),
            "p_value_zvf": round(ch("zvf"), 4),
            "p_value_pcd": round(ch("pcd"), 4),
            "note": "",
        })

    # ===== Write outputs =====
    # p5_manifest_outcome_coupling.tsv — H1 + H3 + per-item outcome rows
    out_tsv = OUT_DIR / "p5_manifest_outcome_coupling.tsv"
    with out_tsv.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["section", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "value"])
        for r in h1_rows:
            w.writerow(["H1_pair", r["model"], r["task"], r["G"], r["temperature"],
                        r["seed_i"], r["seed_j"],
                        f"badge={r['badge']:.1f},|dR|={r['abs_delta_reward']:.6f}", ""])
        for r in h2_rows:
            share = info_share[r["item"]]
            w.writerow(["H2_info", r["item"], f"n_unique={r['n_unique_values']}",
                        f"H={r['entropy_bits']:.4f}", f"Hmax={r['max_entropy_possible_bits']:.4f}",
                        f"n_na={r['n_na_or_missing']}", f"frac_na={r['fraction_na']:.4f}",
                        f"share={share:.4f}", f"placebo={r['placebo']}"])
        for r in item_outcome_rows:
            w.writerow(["H3_perm", r["item"], f"n_unique={r['n_unique']}",
                        f"p_reward={r['p_value_mean_reward']}",
                        f"p_zvf={r['p_value_zvf']}",
                        f"p_pcd={r['p_value_pcd']}",
                        r["note"], ""])

    # p5_manifest_outcome_coupling_boot.tsv — bootstrap summary
    out_boot = OUT_DIR / "p5_manifest_outcome_coupling_boot.tsv"
    with out_boot.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["claim", "statistic", "point_estimate", "ci_low", "ci_high", "p_value", "n"])
        w.writerow(["H1_spearman_badge_vs_|dR|", "spearman_rho", round(rho, 4),
                    round(rho_lo, 4), round(rho_hi, 4), "", len(h1_pair_diffs)])
        w.writerow(["H3_spearman_hamming_vs_|dR|", "spearman_rho", round(rho_r, 4),
                    "", "", "", len(obs_deltas)])
        w.writerow(["H3_spearman_hamming_vs_|dZVF|", "spearman_rho", round(rho_z, 4),
                    "", "", "", len(obs_deltas)])
        w.writerow(["H3_spearman_hamming_vs_|dPCD|", "spearman_rho", round(rho_p, 4),
                    "", "", "", len(obs_deltas)])
        w.writerow(["H3_close_vs_far_|dR|", "mean_diff", round(diff_r_obs, 6),
                    "", "", round(p_r, 4), len(close_d) + len(far_d)])
        w.writerow(["H3_close_vs_far_|dZVF|", "mean_diff", round(diff_z_obs, 6),
                    "", "", round(p_z, 4), len(close_d) +len(far_d)])
        w.writerow(["H3_close_vs_far_|dPCD|", "mean_diff", round(diff_p_obs, 6),
                    "", "", round(p_p, 4), len(close_d) + len(far_d)])

    # p5_manifest_outcome_coupling_summary.json
    summary = {
        "n_cells": len(enriched),
        "n_cell_groups_with_seeds": len(cell_groups),
        "n_pairwise_seed_comparisons": len(h1_pair_diffs),
        "n_sampled_cell_pairs": len(obs_deltas),
        "h1": {
            "claim": "Higher manifest badge -> lower |delta_reward| across seeds (within cell-group).",
            "spearman_rho": round(rho, 4),
            "ci95_low": round(rho_lo, 4),
            "ci95_high": round(rho_hi, 4),
            "ci_excludes_zero": (rho_lo > 0 or rho_hi < 0) if not math.isnan(rho) else False,
            "n_pairs": len(h1_pair_diffs),
        },
        "h2": {
            "claim": "Per-item info content on live 98-cell corpus (bits).",
            "items": [
                {**r, "info_share": info_share[r["item"]]}
                for r in h2_rows
            ],
            "total_bits": round(total_bits, 4),
            "n_placebo_items": sum(1 for r in h2_rows if r["placebo"]),
        },
        "h3": {
            "claim": "Joint manifest fingerprint (Hamming) -> |delta_telemetry|.",
            "spearman_hamming_vs_d_reward": round(rho_r, 4),
            "spearman_hamming_vs_d_zvf": round(rho_z, 4),
            "spearman_hamming_vs_d_pcd": round(rho_p, 4),
            "close_vs_far_mean_diff_reward": round(diff_r_obs, 6),
            "close_vs_far_p_value_reward": round(p_r, 4),
            "close_vs_far_mean_diff_zvf": round(diff_z_obs, 6),
            "close_vs_far_p_value_zvf": round(p_z, 4),
            "close_vs_far_mean_diff_pcd": round(diff_p_obs, 6),
            "close_vs_far_p_value_pcd": round(p_p, 4),
            "per_item_outcome_perm": item_outcome_rows,
        },
    }
    with (OUT_DIR / "p5_manifest_outcome_coupling_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # ----- Console digest
    print(f"[p5_manifest_outcome_coupling] cells={len(enriched)} "
          f"pairs={len(h1_pair_diffs)} cell_pairs_sampled={len(obs_deltas)}")
    print(f"H1  spearman(badge, |dR|) = {rho:+.3f} "
          f"[{rho_lo:+.3f}, {rho_hi:+.3f}]  n={len(h1_pair_diffs)}")
    print(f"H2  total bits={total_bits:.3f} across 7 items")
    for r in h2_rows:
        print(f"   {r['item']:32s} n_unique={r['n_unique_values']:>3d}  "
              f"H={r['entropy_bits']:.3f} bits  share={info_share[r['item']]:.3f}  "
              f"placebo={r['placebo']}")
    print(f"H3  spearman(hamming, |dR|) = {rho_r:+.3f}, "
          f"close-far p_reward={p_r:.3f} zvf={p_z:.3f} pcd={p_p:.3f}")
    print(f"   per-item perm-p (reward / zvf / pcd):")
    for r in item_outcome_rows:
        print(f"     {r['item']:32s} p=({r['p_value_mean_reward']}, "
              f"{r['p_value_zvf']}, {r['p_value_pcd']})  {r['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())