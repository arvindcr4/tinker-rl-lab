"""
P5 iter-105 — Live manifest per-field coverage audit (v2, post-iter-97)
=========================================================================

Fresh vein (not in 120 prior rows):
  - iter 01/14 audited boolean presence per MIN-REPORT item
  - iter 53 audited sub-field structured coverage (0% on all 12 sub-fields)
  - iter 81 audited per-cell yield-residual axes (15.86 bits)
  - iter 97 audited schema-vs-corpus-schema mismatch (axes_are_absent)
  - iter 114 (same dataset) extended iter 97 with discrimination-count
  - **this iter (105)** audits **per-field per-value classification** at the
    98-manifest level: each of the 8 declared manifest keys is classified
    per cell as PRESENT_CONCRETE / PRESENT_NA / PRESENT_PATH /
    PRESENT_KEYWORD / MISSING, with bootstrap CIs and a per-key inventory
    of unique values seen. Then a quantitative cross-ref to the MIN-REPORT
    Item 2 KL problem: count how many manifests literally report the
    `n/a` sentinel vs an empty string vs null vs a concrete value.

Outputs (under platform_hybrid/experiments/results/p5p8/):
  - p5_iter105_per_field_class.tsv     (per-cell per-field classification, 98x8 = 784 rows)
  - p5_iter105_per_field_summary.tsv   (per-field aggregate over 98 cells, ~10 rows)
  - p5_iter105_unique_values.tsv       (per-field unique-value inventory + frequency)
  - p5_iter105_item2_kl_inventory.tsv  (every literal value of ref_policy_kl by category)
  - p5_iter105_summary.json            (machine-readable summary)

Method (stdlib only):
  - load each manifest, classify each top-level key
  - aggregate coverage + bootstrap CIs (B=2000, seed=20260705)
  - emit machine-readable artefact set

Usage:
  python platform_modal/scripts/p5p8/p5_iter105_live_field_coverage.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict

random.seed(20260705)

N_BOOT = 2000
SEED = 20260705

MANIFEST_DIR = "platform_hybrid/experiments/results/mega_20260704/manifests"
OUT_DIR = "platform_hybrid/experiments/results/p5p8"

# Eight declared manifest keys (from iter 97 + iter 105 inspection)
EXPECTED_KEYS = [
    "cell_id",
    "loss_form",
    "ref_policy_kl",
    "sampler_backend_precision",
    "per_step_zvf_path",
    "group_size_schedule",
    "heldout_split",
    "decontamination_notes",
]


def classify(key: str, value: object) -> str:
    """Classify a manifest value into a fine-grained category.

    Categories:
      MISSING          - key absent from manifest
      PRESENT_NULL     - value is JSON null
      PRESENT_EMPTY    - value is empty string
      PRESENT_PATH     - value is a string starting with '/'
      PRESENT_KL_CONCRETE - value is a string starting with 'kl-' (concrete KL value)
      PRESENT_NA       - value is the literal sentinel 'n/a' or 'n/a-*'
      PRESENT_G_CONCRETE - key is group_size_schedule and value is 'fixed-G=<NUM>'
      PRESENT_KEYWORD  - everything else (including 'fixed-G=2' style with non-numeric)
    """
    if value is None:
        return "PRESENT_NULL"
    if not isinstance(value, str):
        return "PRESENT_KEYWORD"
    v = value.strip()
    if v == "":
        return "PRESENT_EMPTY"
    if v.startswith("/"):
        return "PRESENT_PATH"
    if key == "ref_policy_kl" and v.startswith("kl-"):
        return "PRESENT_KL_CONCRETE"
    if v == "n/a" or v.startswith("n/a-"):
        return "PRESENT_NA"
    return "PRESENT_KEYWORD"


def load_manifests() -> list[dict]:
    paths = sorted(glob.glob(os.path.join(MANIFEST_DIR, "*.json")))
    out = []
    for p in paths:
        try:
            with open(p) as f:
                m = json.load(f)
            m["__path__"] = os.path.basename(p)
            out.append(m)
        except Exception as e:
            print(f"WARN: could not parse {p}: {e}", file=sys.stderr)
    return out


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    manifests = load_manifests()
    print(f"loaded {len(manifests)} manifests from {MANIFEST_DIR}")

    # 1. per-cell per-field classification
    per_cell_rows: list[dict] = []
    for m in manifests:
        cid = m.get("cell_id", m.get("__path__", "?"))
        for key in EXPECTED_KEYS:
            present = key in m
            value = m.get(key, None)
            cls = classify(key, value) if present else "MISSING"
            per_cell_rows.append({
                "cell_id": cid,
                "field": key,
                "present_in_manifest": "Y" if present else "N",
                "classification": cls,
                "raw_value": "" if not present else json.dumps(value)[:80],
            })
    pc_path = os.path.join(OUT_DIR, "p5_iter105_per_field_class.tsv")
    with open(pc_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell_id", "field", "present_in_manifest",
                                          "classification", "raw_value"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(per_cell_rows)
    print(f"wrote {pc_path} ({len(per_cell_rows)} rows)")

    # 2. per-field summary
    cls_by_field: dict[str, Counter] = defaultdict(Counter)
    for row in per_cell_rows:
        cls_by_field[row["field"]][row["classification"]] += 1

    n = len(manifests)
    summary_rows = []
    for field in EXPECTED_KEYS:
        c = cls_by_field[field]
        for cls, count in sorted(c.items()):
            summary_rows.append({
                "field": field,
                "classification": cls,
                "count": count,
                "fraction": round(count / n, 4) if n else 0.0,
            })
        # bootstrap CI on the missing fraction
        missing_idx = [1 if r["classification"] == "MISSING" else 0
                       for r in per_cell_rows if r["field"] == field]
        boot_missing = []
        for _ in range(N_BOOT):
            sample = [missing_idx[random.randrange(n)] for _ in range(n)]
            boot_missing.append(sum(sample) / n)
        boot_missing.sort()
        ci_lo = boot_missing[int(0.025 * N_BOOT)]
        ci_hi = boot_missing[int(0.975 * N_BOOT)]
        concrete_count = c.get("PRESENT_KL_CONCRETE", 0) + c.get("PRESENT_G_CONCRETE", 0) \
            + c.get("PRESENT_PATH", 0) + c.get("PRESENT_KEYWORD", 0)
        concrete_frac = concrete_count / n if n else 0.0
        summary_rows.append({
            "field": field,
            "classification": f"__missing_frac [95% CI]",
            "count": "",
            "fraction": round(ci_lo, 4),
        })

    sum_path = os.path.join(OUT_DIR, "p5_iter105_per_field_summary.tsv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["field", "classification",
                                          "count", "fraction"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(summary_rows)
    print(f"wrote {sum_path} ({len(summary_rows)} rows)")

    # 3. unique-value inventory per field
    inv_rows: list[dict] = []
    val_by_field: dict[str, Counter] = defaultdict(Counter)
    for m in manifests:
        for k in EXPECTED_KEYS:
            v = m.get(k, "__ABSENT__")
            if isinstance(v, str):
                v = v.strip()
            val_by_field[k][json.dumps(v)[:60] if not isinstance(v, str) else v] += 1
    for field, counter in val_by_field.items():
        for value, count in sorted(counter.items(), key=lambda kv: -kv[1]):
            inv_rows.append({"field": field, "value": value, "count": count})
    inv_path = os.path.join(OUT_DIR, "p5_iter105_unique_values.tsv")
    with open(inv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["field", "value", "count"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(inv_rows)
    print(f"wrote {inv_path} ({len(inv_rows)} rows)")

    # 4. Item 2 (ref_policy_kl) inventory - find every literal value
    kl_inventory_rows: list[dict] = []
    for m in manifests:
        cid = m.get("cell_id", m.get("__path__", "?"))
        v = m.get("ref_policy_kl", None)
        kl_inventory_rows.append({
            "cell_id": cid,
            "key_present": "Y" if "ref_policy_kl" in m else "N",
            "raw_value": "" if v is None else str(v)[:80],
            "classification": classify("ref_policy_kl", v) if "ref_policy_kl" in m else "MISSING",
        })
    kl_path = os.path.join(OUT_DIR, "p5_iter105_item2_kl_inventory.tsv")
    with open(kl_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell_id", "key_present", "raw_value",
                                          "classification"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(kl_inventory_rows)
    print(f"wrote {kl_path} ({len(kl_inventory_rows)} rows)")

    # 5. machine-readable summary
    summary = {
        "iter": 105,
        "n_manifests": n,
        "manifest_dir": MANIFEST_DIR,
        "n_boot": N_BOOT,
        "seed": SEED,
        "expected_keys": EXPECTED_KEYS,
        "per_field_summary": {
            field: {
                "n_present_key": sum(1 for r in per_cell_rows
                                    if r["field"] == field and r["present_in_manifest"] == "Y"),
                "classification_distribution": dict(cls_by_field[field]),
                "n_unique_values": len(val_by_field[field]),
            }
            for field in EXPECTED_KEYS
        },
        "item2_kl_verdict": {
            "literal_na_count": sum(1 for r in kl_inventory_rows
                                    if r["classification"] == "PRESENT_NA"),
            "concrete_kl_count": sum(1 for r in kl_inventory_rows
                                     if r["classification"] == "PRESENT_KL_CONCRETE"),
            "missing_key_count": sum(1 for r in kl_inventory_rows
                                     if r["classification"] == "MISSING"),
            "items_with_item2_validated": 0,
        },
    }

    sm_path = os.path.join(OUT_DIR, "p5_iter105_summary.json")
    with open(sm_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"wrote {sm_path}")

    # CLI summary
    print()
    print("== HEADLINE SUMMARY (iter 105) ==")
    print(f"n_manifests = {n}; expected_keys = {len(EXPECTED_KEYS)}")
    for field in EXPECTED_KEYS:
        dist = dict(cls_by_field[field])
        print(f"  {field:30s}  " + " ".join(
            f"{k}={v}" for k, v in sorted(dist.items(), key=lambda kv: -kv[1])
        ))
    print()
    print(f"Item 2 ref_policy_kl  PRESENT_NA   = {summary['item2_kl_verdict']['literal_na_count']}/{n}")
    print(f"Item 2 ref_policy_kl  PRESENT_KL_C = {summary['item2_kl_verdict']['concrete_kl_count']}/{n}")
    print(f"Item 2 ref_policy_kl  MISSING_key  = {summary['item2_kl_verdict']['missing_key_count']}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
