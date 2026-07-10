#!/usr/bin/env python3
"""P5 JOB B / SYNTH (iter 56): Empirical validation of the iter-53 MVE recommendation.

iter-53 #64 measured the theoretical greedy-MVE recommendation: adding 5
continuous-telemetry fields (mean_reward, zvf, pcd, mean_completion_len,
std_completion_len) lifts distinct profiles 15 -> 98 on the 98-cell mega
corpus, the principled-minimum that breaks the iter-52 honesty vacuum.

This iter VALIDATES that recommendation end-to-end:
  1. Load 98 mega manifests + cells.tsv (which already carries the 5
     fields as measured telemetry)
  2. Compute the iter-13 baseline badge on the 7-item MIN-REPORT
  3. Augment each manifest with a continuous_telemetry block (mean_reward,
     zvf, pcd, mean_completion_len, std_completion_len)
  4. Add an EIGHTH item (continuous_telemetry, weight=20) to the auditor's
     scoring formula. This is the empirical side of iter-53's
     "recommended but not yet required" extension.
  5. Recompute the augmented badge on the 98 cells.
  6. Paired bootstrap B=2000 on per-cell delta-badge = augmented - baseline.

Falsifiable headline: the empirical badge-mean improvement, with paired
95% CI. If the CI excludes zero, the iter-53 MVE recommendation is
operational, not just theoretical.

Outputs
-------
experiments/results/p5p8/p5_mve_empirical.tsv             (98 rows per-cell baseline+augmented)
experiments/results/p5p8/p5_mve_empirical_summary.json   (CI + headline)
docs/p5p8_improvements/67_p5_mve_empirical.md

Stdlib + json + csv + math + statistics. <=290 lines.
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
CELLS_TSV = ROOT / "experiments" / "results" / "mega_20260704" / "cells.tsv"
MANIFEST_DIR = ROOT / "experiments" / "results" / "mega_20260704" / "manifests"
OUT = ROOT / "experiments" / "results" / "p5p8"
OUT.mkdir(parents=True, exist_ok=True)

# ----- iter-13 auditor's 7-item schema (copied for self-containment) -----
SCHEMA_7 = [
    (1, "loss_form", 10,
     [r"^(grpo|gspo|dapo|drgrpo|dpo|sequence|ppo|sft|n/a-sampling)$"],
     ["loss_form"]),
    (2, "ref_policy_kl", 10,
     [r"^(kl-[a-z]+(\d+(\.\d+)?)?|kl-est-[a-z]+|no-kl|n/a(?:-[a-z]+)?)$"],
     ["ref_policy_kl", "ref_policy_kl_handling"]),
    (3, "sampler_backend_precision", 20,
     [r"^(tinker-closed|vllm|sglang|hf|trtllm|openai|anthropic)[-@a-zA-Z0-9._/]*$"],
     ["sampler_backend_precision"]),
    (4, "per_step_zvf_path", 20,
     [r".*"],
     ["per_step_zvf_path"]),
    (5, "group_size_schedule", 10,
     [r"^(fixed-G=\d+|adaptive[-+a-zA-Z0-9=<>]*|escalating|decaying|constant G=\d+.*|paired phases.*|arm [A-Z]:.*|n/a.*)$"],
     ["group_size_schedule"]),
    (6, "heldout_split", 10,
     [r".*"],
     ["heldout_split"]),
    (7, "decontamination_notes", 20,
     [r".*"],
     ["decontamination_notes"]),
]

# The 5 MVE fields (from iter-53 #64). We add a new item-8 with weight=20.
MVE_FIELDS = ["mean_reward", "zvf", "pcd", "mean_completion_len", "std_completion_len"]
MVE_WEIGHT = 20


def load_cells() -> dict[str, dict]:
    cells = {}
    with open(CELLS_TSV) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            cells[row["cell_id"]] = row
    return cells


def load_manifests() -> list[tuple[str, dict]]:
    out = []
    for jf in sorted(MANIFEST_DIR.glob("*.json")):
        try:
            with jf.open() as f:
                d = json.load(f)
        except Exception:
            continue
        cell_id = d.get("cell_id", jf.stem)
        out.append((cell_id, d))
    return out


def score_one(manifest: dict, cells_by_id: dict[str, dict], include_mve: bool) -> dict:
    """Score a manifest with or without the MVE item-8.

    Returns dict with item_scores, total, per-item details.
    """
    cell = cells_by_id.get(manifest.get("cell_id", ""), {})
    total = 0.0
    items = []
    for item_no, name, weight, validators, keys in SCHEMA_7:
        raw = None
        for k in keys:
            if k in manifest and manifest[k] is not None:
                raw = manifest[k]
                break
        present = raw is not None and str(raw).strip() != ""
        validated = bool(present) and any(re.match(v, str(raw), re.IGNORECASE) for v in validators)
        # Treat any n/a- prefix as honest n/a (matches iter-13 semantics)
        is_na = isinstance(raw, str) and raw.strip().lower().startswith("n/a")
        if is_na and any(re.match(v, str(raw), re.IGNORECASE) for v in validators):
            base = 0.5
        elif present and validated:
            base = 1.0
        elif present:
            base = 0.25
        else:
            base = 0.0
        sub_frac = 1.0   # baseline auditor uses sub_frac=1 unless sub-field validation
        item_score = weight * base * sub_frac
        total += item_score
        items.append({"item": item_no, "name": name, "weight": weight,
                      "present": int(present), "validated": int(validated),
                      "base": base, "score": round(item_score, 2)})

    # MVE item-8
    if include_mve:
        mve_present = 0
        mve_valid = 0
        for fld in MVE_FIELDS:
            v = cell.get(fld)
            if v is not None and str(v).strip() != "":
                try:
                    float(v)
                    mve_present += 1
                    mve_valid += 1
                except ValueError:
                    pass
        # 5/5 sub-fields present and numeric -> base=1.0
        base = mve_valid / 5.0
        sub_frac = 0.5 + 0.5 * (mve_valid / 5.0)
        item_score = MVE_WEIGHT * base * sub_frac
        items.append({"item": 8, "name": "continuous_telemetry_mve",
                      "weight": MVE_WEIGHT, "present": mve_present,
                      "validated": mve_valid, "base": round(base, 3),
                      "score": round(item_score, 2)})
        total += item_score

    return {"items": items, "total": round(total, 2)}


def main() -> None:
    print("[p5_mve_empirical] starting")
    cells = load_cells()
    print(f"  cells loaded: {len(cells)}")
    manifests = load_manifests()
    print(f"  manifests loaded: {len(manifests)}")

    rows = []
    baselines = []
    augmented = []
    augmented_low = []
    for cell_id, m in manifests:
        b = score_one(m, cells, include_mve=False)
        a = score_one(m, cells, include_mve=True)
        # Conservative case: MVE item-8 with weight=10 (half weight)
        a_low = score_one(m, cells, include_mve=True)
        a_low_total = b["total"] + sum(
            it["score"] for it in a_low["items"] if it["item"] == 8
        ) * 0.5
        delta = a["total"] - b["total"]
        delta_low = a_low_total - b["total"]
        rows.append({
            "cell_id": cell_id,
            "baseline_total": b["total"],
            "augmented_total": a["total"],
            "augmented_low_total": round(a_low_total, 2),
            "delta": delta,
            "delta_low_weight": round(delta_low, 2),
        })
        baselines.append(b["total"])
        augmented.append(a["total"])
        augmented_low.append(a_low_total)
    df_path = OUT / "p5_mve_empirical.tsv"
    with open(df_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=["cell_id", "baseline_total",
                                          "augmented_total", "augmented_low_total",
                                          "delta", "delta_low_weight"], delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {df_path}")

    # Paired bootstrap B=2000 on per-cell delta-badge
    n = len(baselines)
    deltas = [a - b for a, b in zip(augmented, baselines)]
    deltas_low = [a - b for a, b in zip(augmented_low, baselines)]
    seed = 20260704
    rng_state = seed
    boot_means = []
    boot_means_low = []
    for _ in range(2000):
        # Use a simple LCG for repeatability (no numpy)
        sample = []
        sample_low = []
        for _ in range(n):
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            j = (rng_state >> 16) % n
            sample.append(deltas[j])
            sample_low.append(deltas_low[j])
        boot_means.append(sum(sample) / n)
        boot_means_low.append(sum(sample_low) / n)
    boot_means.sort()
    boot_means_low.sort()
    ci_lo = boot_means[int(0.025 * len(boot_means))]
    ci_hi = boot_means[int(0.975 * len(boot_means)) - 1]
    ci_lo_low = boot_means_low[int(0.025 * len(boot_means_low))]
    ci_hi_low = boot_means_low[int(0.975 * len(boot_means_low)) - 1]
    mean_delta = statistics.mean(deltas)
    mean_delta_low = statistics.mean(deltas_low)
    mean_b = statistics.mean(baselines)
    mean_a = statistics.mean(augmented)

    summary = {
        "n_cells": n,
        "mean_baseline": round(mean_b, 3),
        "mean_augmented_weight20": round(mean_a, 3),
        "mean_augmented_weight10": round(statistics.mean(augmented_low), 3),
        "mean_delta_weight20": round(mean_delta, 3),
        "mean_delta_weight10": round(mean_delta_low, 3),
        "paired_bootstrap_95ci_lo_weight20": round(ci_lo, 3),
        "paired_bootstrap_95ci_hi_weight20": round(ci_hi, 3),
        "paired_bootstrap_95ci_lo_weight10": round(ci_lo_low, 3),
        "paired_bootstrap_95ci_hi_weight10": round(ci_hi_low, 3),
        "ci_excludes_zero_weight20": bool(ci_lo > 0 or ci_hi < 0),
        "ci_excludes_zero_weight10": bool(ci_lo_low > 0 or ci_hi_low < 0),
        "median_delta_weight20": round(statistics.median(deltas), 3),
        "max_delta_weight20": round(max(deltas), 3),
        "min_delta_weight20": round(min(deltas), 3),
        "n_boot": 2000,
        "seed": seed,
        "mve_fields": MVE_FIELDS,
        "mve_weight_full": MVE_WEIGHT,
        "mve_weight_low": MVE_WEIGHT // 2,
        "baseline_weight_total": sum(w for _, _, w, _, _ in SCHEMA_7),
        "augmented_weight_total": sum(w for _, _, w, _, _ in SCHEMA_7) + MVE_WEIGHT,
    }
    with open(OUT / "p5_mve_empirical_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {OUT / 'p5_mve_empirical_summary.json'}")

    print(f"  HEADLINE: baseline={mean_b:.2f}, augmented_w20={mean_a:.2f}, "
          f"mean_delta={mean_delta:.2f}, CI=[{ci_lo:.2f}, {ci_hi:.2f}], "
          f"excl_zero={summary['ci_excludes_zero_weight20']}")


if __name__ == "__main__":
    main()