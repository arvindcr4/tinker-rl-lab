#!/usr/bin/env python3
"""P5 MANIFEST SUFFICIENCY + MANIFEST PREDICTIVE POWER AUDIT (iter 189).

Fresh P5 vein, not in 198 prior P5 rows. Closes brief vein (a) at the
**manifest self-sufficiency + manifest predictive power** layer:

Two complementary questions:

VEIN A — MANIFEST SUFFICIENCY
   Given ONLY the manifest JSON fields (no cells.tsv), can the manifest
   UNIQUELY IDENTIFY every cell? The manifest has 8 fields; 3 are
   stack-conditioning fields (group_size_schedule, heldout_split,
   decontamination_notes) and 3 are CONSTANT (loss_form=n/a-sampling,
   ref_policy_kl=n/a, sampler_backend_precision=tinker-closed).

   The fact that 3/8 manifest fields are constants means the manifest
   only discriminates on at most 5 axes (G × task_slice × decontamination).
   But cells.tsv has 5 discriminating axes (model, G, task_slice,
   temperature, seed) = 2*5*3*2*2 = 120 cell-slots; the live corpus
   has 98 cells covering 98 of these 120 slots.

   We ask: what is the MINIMAL SUBSET of (manifest + cells.tsv) fields
   that uniquely identifies all 98 cells? And: how many cells can the
   manifest ALONE uniquely identify?

VEIN B — MANIFEST PREDICTIVE POWER
   Compute η² (variance-explained) of the 3 discriminating manifest
   fields (group_size_schedule=G, heldout_split=task_slice,
   decontamination_notes=correlated-with-task) on the 4 telemetry
   channels in cells.tsv (zvf, mean_reward, pcd, mean_completion_len).
   This quantifies: how much of cells.tsv telemetry is "stack-driven"
   (i.e., predictable from manifest fields alone)?

   Compare to the iter-5 mega η² decomposition (5 axes × 5 channels,
   row 11): iter-5 found stack axes explain 73-93% of variance in every
   telemetry channel; seed explains 0.0-0.15%. iter-189 specifically
   asks: how much do MANIFEST FIELDS explain (no model, no temperature,
   no seed in the manifest).

5 falsifiable hypotheses (set BEFORE measurement)
-------------------------------------------------
H1 manifest discriminating fields produce ≥ 10 size-1 equivalence
   classes (cells uniquely identifiable from manifest alone)
H2 manifest discriminating fields populate ≥ 14/15 effective
   equivalence classes (5 G × 3 task_slice; decontam is correlated
   with task_slice, so the effective design has 15 classes not 30)
H3 minimal-fields-to-identify-all-98 ≤ 5 (i.e., ≤ 5 fields suffice)
   {model, G, task_slice, temperature, seed} = 5 fields
H4 MANIFEST-FIELDS η²(zvf) > MANIFEST-FIELDS η²(pcd)
   (zvf is more G-driven than pcd which has more model-dependence)
H5 manifest + temperature + seed extend η²(mean_completion_len)
   by >= 5 pp over manifest-alone (the "missing 2 stack axes"
   contribute more to mean_completion_len than to mean_reward)

Outputs
-------
- platform_hybrid/experiments/results/p5p8/p5_iter189_minimal_field_set.tsv
  (rows: greedy-add minimal-field-set growth; col: incremental
   discrimination gain)
- platform_hybrid/experiments/results/p5p8/p5_iter189_manifest_sufficiency.tsv
  (98 rows: per-cell uniqueness under each candidate field-set)
- platform_hybrid/experiments/results/p5p8/p5_iter189_eta2_by_field_group.tsv
  (rows: 2 field_groups × 4 channels × 3 stats [η², ω², ε²])
- platform_hybrid/experiments/results/p5p8/p5_iter189_h5_eta2_lift.tsv
  (4 rows: per-channel η² lift from adding temp/seed to manifest)
- platform_hybrid/experiments/results/p5p8/p5_iter189_summary.json
  (H1-H5 verdicts + per-cell-level findings + bootstrap CIs)
"""
from __future__ import annotations
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

# ---------- config ----------
WORKTREE = Path("/home/claude/tinker-rl-lab-minimax")
MANIFEST_DIR = WORKTREE / "platform_hybrid/experiments/results/mega_20260704/manifests"
CELLS_TSV = WORKTREE / "platform_hybrid/experiments/results/mega_20260704/cells.tsv"
OUT_DIR = WORKTREE / "platform_hybrid/experiments/results/p5p8"

MANIFEST_FIELDS = [
    "loss_form", "ref_policy_kl", "sampler_backend_precision",
    "group_size_schedule", "heldout_split", "decontamination_notes",
]
CELLS_FIELDS = ["model", "task_slice", "G", "temperature", "seed"]
TELEMETRY = ["zvf", "mean_reward", "pcd", "mean_completion_len"]

# ---------- helpers ----------
def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def eta_squared(y, groups):
    """Compute eta-squared (one-way): SS_between / SS_total.
    groups: dict[group_key -> list[float]]"""
    grand_mean = sum(y) / len(y) if y else 0
    ss_total = sum((yi - grand_mean) ** 2 for yi in y)
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    n_total = len(y)
    for g, vals in groups.items():
        n_g = len(vals)
        mean_g = sum(vals) / n_g if n_g > 0 else 0
        ss_between += n_g * (mean_g - grand_mean) ** 2
    return ss_between / ss_total


def omega_squared(y, groups):
    """Bias-corrected eta-squared: (SS_between - (k-1)*MSE) / (SS_total + MSE)."""
    grand_mean = sum(y) / len(y) if y else 0
    ss_total = sum((yi - grand_mean) ** 2 for yi in y)
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    ss_within = 0.0
    k = len(groups)
    n_total = len(y)
    for g, vals in groups.items():
        n_g = len(vals)
        if n_g == 0:
            continue
        mean_g = sum(vals) / n_g
        ss_between += n_g * (mean_g - grand_mean) ** 2
        ss_within += sum((yi - mean_g) ** 2 for yi in vals)
    df_within = n_total - k
    if df_within <= 0:
        return 0.0
    mse = ss_within / df_within
    return (ss_between - (k - 1) * mse) / (ss_total + mse)


def epsilon_squared(y, groups):
    """Epsilon-squared (Kelley): (SS_between - (k-1)*MSE) / SS_total (no add)."""
    grand_mean = sum(y) / len(y) if y else 0
    ss_total = sum((yi - grand_mean) ** 2 for yi in y)
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    ss_within = 0.0
    k = len(groups)
    n_total = len(y)
    for g, vals in groups.items():
        n_g = len(vals)
        if n_g == 0:
            continue
        mean_g = sum(vals) / n_g
        ss_between += n_g * (mean_g - grand_mean) ** 2
        ss_within += sum((yi - mean_g) ** 2 for yi in vals)
    df_within = n_total - k
    if df_within <= 0:
        return 0.0
    mse = ss_within / df_within
    return (ss_between - (k - 1) * mse) / ss_total


# ---------- load data ----------
def load_manifests():
    """Load all manifests → dict[cell_id -> manifest_dict]."""
    out = {}
    for path in sorted(MANIFEST_DIR.glob("*.json")):
        d = json.load(open(path))
        out[d["cell_id"]] = d
    return out


def load_cells():
    """Load cells.tsv → dict[cell_id -> cells_row]."""
    out = {}
    with open(CELLS_TSV) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[row["cell_id"]] = row
    return out


def group_by(rows, key_fn):
    """Group list of rows by key_fn. Returns dict[key -> list]."""
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    return dict(groups)


def parse_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


# ---------- bootstrap ----------
def bootstrap_paired_diff(a_vals, b_vals, n_boot=2000, seed=20260706):
    """Paired bootstrap CI on mean(a - b)."""
    import random
    random.seed(seed)
    n = min(len(a_vals), len(b_vals))
    if n == 0:
        return (0.0, 0.0, 0.0)
    diffs = [a_vals[i] - b_vals[i] for i in range(n)]
    obs_mean = sum(diffs) / n
    boots = []
    for _ in range(n_boot):
        sample = [diffs[random.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot) - 1]
    return (obs_mean, lo, hi)


# ---------- main analysis ----------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifests = load_manifests()
    cells = load_cells()
    n_cells = len(cells)
    n_manifests = len(manifests)
    assert n_cells == n_manifests == 98, f"Got {n_cells} cells, {n_manifests} manifests"

    # ---- VEIN A: minimal-field-set analysis ----
    # Compute per-cell signature under candidate field-sets
    candidate_sets = {
        "manifest_all_8": [
            "cell_id", "loss_form", "ref_policy_kl", "sampler_backend_precision",
            "per_step_zvf_path", "group_size_schedule", "heldout_split",
            "decontamination_notes",
        ],
        "manifest_discriminating_3": [
            "group_size_schedule", "heldout_split", "decontamination_notes",
        ],
        "manifest_plus_cells_5": [
            "model", "G", "task_slice", "temperature", "seed",
        ],
        "cells_all_5": [
            "model", "task_slice", "G", "temperature", "seed",
        ],
        "cells_minus_seed_4": [
            "model", "task_slice", "G", "temperature",
        ],
        "cells_minus_temp_4": [
            "model", "task_slice", "G", "seed",
        ],
        "cells_minus_model_4": [
            "task_slice", "G", "temperature", "seed",
        ],
    }

    def signature(cell_id, field_list):
        m = manifests[cell_id]
        c = cells[cell_id]
        parts = []
        for f in field_list:
            if f in m:
                parts.append(f"{f}={m[f]}")
            elif f in c:
                parts.append(f"{f}={c[f]}")
        return "|".join(parts)

    sufficiency_rows = []
    for cell_id in sorted(cells):
        for cset_name, flds in candidate_sets.items():
            sig = signature(cell_id, flds)
            sufficiency_rows.append({
                "cell_id": cell_id,
                "field_set": cset_name,
                "n_fields": len(flds),
                "signature": sig,
            })

    # Per (field_set) — count equivalence classes and cells covered uniquely
    uniq_stats = {}
    for cset_name, flds in candidate_sets.items():
        sigs = [signature(c, flds) for c in sorted(cells)]
        cnt = Counter(sigs)
        n_unique = sum(1 for s in sigs if cnt[s] == 1)
        uniq_stats[cset_name] = {
            "n_fields": len(flds),
            "n_classes": len(cnt),
            "n_unique_cells": n_unique,
            "max_class_size": max(cnt.values()),
            "min_class_size": min(cnt.values()),
        }
    print("Field-set sufficiency:")
    for cs, st in uniq_stats.items():
        print(f"  {cs}: n_classes={st['n_classes']} n_unique={st['n_unique_cells']} max={st['max_class_size']}")

    # Greedy add — find minimal field-set that uniquely identifies all 98 cells
    # Strategy: start with cells_all_5; drop one field at a time and check
    pool = ["model", "task_slice", "G", "temperature", "seed"]
    greedy_rows = []
    for k in range(1, len(pool) + 1):
        # All subsets of size k
        from itertools import combinations
        best = None
        for sub in combinations(pool, k):
            flds = list(sub)
            sigs = [signature(c, flds) for c in sorted(cells)]
            cnt = Counter(sigs)
            n_unique = sum(1 for s in sigs if cnt[s] == 1)
            if n_unique == n_cells:
                if best is None:
                    best = (flds, n_unique)
        if best:
            greedy_rows.append({
                "k_fields": k,
                "fields": "+".join(best[0]),
                "n_unique_cells": best[1],
                "covers_all": best[1] == n_cells,
            })
        else:
            greedy_rows.append({
                "k_fields": k,
                "fields": "n/a",
                "n_unique_cells": 0,
                "covers_all": False,
            })
    print("\nGreedy minimal-field-set growth (subsets that cover all 98):")
    for r in greedy_rows:
        print(f"  k={r['k_fields']}: {r['fields']} unique={r['n_unique_cells']} all?={r['covers_all']}")

    # ---- VEIN B: η² of manifest fields on telemetry ----
    # For each (field_group, channel) compute η² + ω² + ε²
    field_groups = {
        "manifest_discriminating_3": ["group_size_schedule", "heldout_split", "decontamination_notes"],
        "cells_all_5": ["model", "task_slice", "G", "temperature", "seed"],
        "manifest_plus_temp_seed": ["group_size_schedule", "heldout_split", "decontamination_notes",
                                     "temperature", "seed"],
    }

    eta2_rows = []
    for fg_name, flds in field_groups.items():
        # Aggregate all (group_factor) levels by joining across fields
        rows_for_fg = []
        for cell_id in sorted(cells):
            m = manifests[cell_id]
            c = cells[cell_id]
            r = {}
            for f in flds:
                if f in m:
                    r[f] = m[f]
                else:
                    r[f] = c[f]
            # Join into single group key
            r["group_key"] = "|".join(f"{f}={r[f]}" for f in flds)
            for ch in TELEMETRY:
                v = parse_float(c[ch])
                if v is not None:
                    r2 = {**r, "channel": ch, "value": v}
                    rows_for_fg.append(r2)
        for ch in TELEMETRY:
            ch_rows = [r for r in rows_for_fg if r["channel"] == ch]
            y = [r["value"] for r in ch_rows]
            groups = defaultdict(list)
            for r in ch_rows:
                groups[r["group_key"]].append(r["value"])
            eta2 = eta_squared(y, groups)
            omg2 = omega_squared(y, groups)
            eps2 = epsilon_squared(y, groups)
            eta2_rows.append({
                "field_group": fg_name,
                "n_fields": len(flds),
                "channel": ch,
                "n_obs": len(y),
                "n_groups": len(groups),
                "eta_sq": round(eta2, 6),
                "omega_sq": round(omg2, 6),
                "epsilon_sq": round(eps2, 6),
            })
    print("\nη² by field_group × channel:")
    for r in eta2_rows:
        print(f"  {r['field_group']:30s} {r['channel']:25s} η²={r['eta_sq']:.4f} ω²={r['omega_sq']:.4f} ε²={r['epsilon_sq']:.4f}")

    # ---- VEIN B-extended: H5 lift from adding temp+seed ----
    # Compare manifest_discriminating_3 vs manifest_plus_temp_seed η² per channel
    h5_rows = []
    base = {r["channel"]: r["eta_sq"] for r in eta2_rows if r["field_group"] == "manifest_discriminating_3"}
    ext = {r["channel"]: r["eta_sq"] for r in eta2_rows if r["field_group"] == "manifest_plus_temp_seed"}
    for ch in TELEMETRY:
        b = base.get(ch, 0)
        e = ext.get(ch, 0)
        h5_rows.append({
            "channel": ch,
            "eta2_manifest_alone": round(b, 6),
            "eta2_manifest_plus_temp_seed": round(e, 6),
            "lift_pp": round((e - b) * 100, 3),
            "lift_relative": round((e - b) / max(b, 1e-9), 4) if b > 0 else None,
        })
    print("\nH5 η² lift (manifest + temp + seed):")
    for r in h5_rows:
        print(f"  {r['channel']:25s} base={r['eta2_manifest_alone']:.4f} ext={r['eta2_manifest_plus_temp_seed']:.4f} +{r['lift_pp']:.2f}pp")

    # ---- Hypotheses ----
    # H1: manifest discriminating fields produce ≥ 10 size-1 classes
    h1_unique = uniq_stats["manifest_discriminating_3"]["n_unique_cells"]
    h1_pass = h1_unique >= 10

    # H2: manifest discriminating fields populate ≥ 14/15 effective classes
    h2_classes = uniq_stats["manifest_discriminating_3"]["n_classes"]
    h2_pass = h2_classes >= 14

    # H3: minimal-fields-to-identify-all-98 ≤ 5
    h3_min_k = None
    for r in greedy_rows:
        if r["covers_all"]:
            h3_min_k = r["k_fields"]
            break
    h3_pass = h3_min_k is not None and h3_min_k <= 5

    # H4: η²(zvf) > η²(pcd) for manifest_discriminating_3
    eta2_zvf = next(r["eta_sq"] for r in eta2_rows
                    if r["field_group"] == "manifest_discriminating_3" and r["channel"] == "zvf")
    eta2_pcd = next(r["eta_sq"] for r in eta2_rows
                    if r["field_group"] == "manifest_discriminating_3" and r["channel"] == "pcd")
    h4_pass = eta2_zvf > eta2_pcd

    # H5: manifest + temperature + seed extend η²(mean_completion_len) by >= 5pp
    h5_lift = next(r["lift_pp"] for r in h5_rows if r["channel"] == "mean_completion_len")
    h5_pass = h5_lift >= 5.0

    print(f"\n=== HYPOTHESES ===")
    print(f"H1 manifest_discriminating_3 → ≥ 10/98 unique cells: {h1_unique}/98 {'PASS' if h1_pass else 'FAIL'}")
    print(f"H2 manifest_discriminating_3 → ≥ 14/15 effective classes: {h2_classes}/15 {'PASS' if h2_pass else 'FAIL'}")
    print(f"H3 minimal-fields ≤ 5 covers all 98: k={h3_min_k} {'PASS' if h3_pass else 'FAIL'}")
    print(f"H4 η²(zvf)={eta2_zvf:.4f} > η²(pcd)={eta2_pcd:.4f}: {'PASS' if h4_pass else 'FAIL'}")
    print(f"H5 η²(mean_completion_len) lift = {h5_lift:.2f}pp >= 5pp: {'PASS' if h5_pass else 'FAIL'}")

    # ---- Save outputs ----
    # 1. Minimal-field-set growth
    with open(OUT_DIR / "p5_iter189_minimal_field_set.tsv", "w") as f:
        w = csv.DictWriter(f, fieldnames=["k_fields", "fields", "n_unique_cells", "covers_all"],
                            delimiter="\t")
        w.writeheader()
        for r in greedy_rows:
            w.writerow(r)

    # 2. Manifest sufficiency per field-set
    with open(OUT_DIR / "p5_iter189_manifest_sufficiency.tsv", "w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["field_set", "n_fields", "n_classes", "n_unique_cells",
                    "max_class_size", "min_class_size"])
        for cs, st in uniq_stats.items():
            w.writerow([cs, st["n_fields"], st["n_classes"], st["n_unique_cells"],
                        st["max_class_size"], st["min_class_size"]])

    # 3. η² table
    with open(OUT_DIR / "p5_iter189_eta2_by_field_group.tsv", "w") as f:
        w = csv.DictWriter(f, fieldnames=["field_group", "n_fields", "channel", "n_obs",
                                          "n_groups", "eta_sq", "omega_sq", "epsilon_sq"],
                            delimiter="\t")
        w.writeheader()
        for r in eta2_rows:
            w.writerow(r)

    # 4. H5 lift
    with open(OUT_DIR / "p5_iter189_h5_eta2_lift.tsv", "w") as f:
        w = csv.DictWriter(f, fieldnames=["channel", "eta2_manifest_alone",
                                          "eta2_manifest_plus_temp_seed", "lift_pp",
                                          "lift_relative"], delimiter="\t")
        w.writeheader()
        for r in h5_rows:
            w.writerow(r)

    # 5. JSON summary
    summary = {
        "iter": 189,
        "pillar": "P5",
        "n_cells": n_cells,
        "n_manifests": n_manifests,
        "manifest_field_cardinality": {
            "loss_form": 1, "ref_policy_kl": 1, "sampler_backend_precision": 1,
            "group_size_schedule": 5, "heldout_split": 3, "decontamination_notes": 2,
            "per_step_zvf_path": 98, "cell_id": 98,
        },
        "uniq_stats": uniq_stats,
        "greedy_min_k": h3_min_k,
        "hypotheses": {
            "H1": {"claim": "manifest_discriminating_3 → ≥ 10/98 unique cells",
                   "observed": h1_unique, "verdict": "PASS" if h1_pass else "FAIL"},
            "H2": {"claim": "manifest_discriminating_3 → ≥ 14/15 effective classes",
                   "observed": h2_classes, "verdict": "PASS" if h2_pass else "FAIL"},
            "H3": {"claim": "minimal-fields ≤ 5 covers all 98",
                   "observed_min_k": h3_min_k, "verdict": "PASS" if h3_pass else "FAIL"},
            "H4": {"claim": "η²(zvf) > η²(pcd) for manifest_discriminating_3",
                   "eta2_zvf": eta2_zvf, "eta2_pcd": eta2_pcd,
                   "verdict": "PASS" if h4_pass else "FAIL"},
            "H5": {"claim": "η²(mean_completion_len) lift >= 5pp",
                   "lift_pp": h5_lift, "verdict": "PASS" if h5_pass else "FAIL"},
        },
        "headline_findings": [
            f"F1: 3 manifest fields are CONSTANT (loss_form=n/a-sampling, ref_policy_kl=n/a, sampler_backend_precision=tinker-closed), so manifest alone only discriminates on 3 fields (G, task_slice, decontamination)",
            f"F2: manifest_discriminating_3 produces {uniq_stats['manifest_discriminating_3']['n_classes']} equivalence classes (max size {uniq_stats['manifest_discriminating_3']['max_class_size']}); only {h1_unique}/98 cells uniquely identifiable from manifest alone",
            f"F3: minimal-fields-to-identify-all-98 = k={h3_min_k} (the standard {len(['model', 'task_slice', 'G', 'temperature', 'seed'])}-tuple)",
            f"F4: MANIFEST-FIELDS η² on telemetry — zvf={eta2_zvf:.3f}, pcd={eta2_pcd:.3f}, mean_reward={next(r['eta_sq'] for r in eta2_rows if r['field_group']=='manifest_discriminating_3' and r['channel']=='mean_reward'):.3f}, mean_completion_len={next(r['eta_sq'] for r in eta2_rows if r['field_group']=='manifest_discriminating_3' and r['channel']=='mean_completion_len'):.3f}",
            f"F5: adding temperature+seed lifts η²(mean_completion_len) by {h5_lift:.2f}pp; the manifest is missing 2 of the 5 stack-conditioning axes",
        ],
    }
    with open(OUT_DIR / "p5_iter189_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs written to {OUT_DIR}/p5_iter189_*")
    print(f"Total hypotheses: {sum(1 for h in summary['hypotheses'].values() if h['verdict']=='PASS')}/5 PASS")


if __name__ == "__main__":
    main()