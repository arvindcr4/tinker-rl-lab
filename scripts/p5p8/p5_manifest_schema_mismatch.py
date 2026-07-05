#!/usr/bin/env python3
"""P5 iter 97 — MIN-REPORT manifest schema vs cells.tsv schema mismatch audit.

The brief's vein (a) calls for a measured coverage table of the MIN-REPORT
schema against the live mega-campaign manifests, including missing/ambiguous
fields. Prior iterations audited (i) boolean presence per item (iter 01/14),
(ii) sub-field structured coverage (iter 53), and (iii) information-budget
contribution per item (iter 65). This iter audits a different angle:
**schema-vs-corpus-schema mismatch** — for each stack axis present in
cells.tsv but absent from the manifest, quantify the missing-field gap and
its impact on the manifest's discriminative power.

Pipeline (≤300 LoC, stdlib only):
  PART A — Schema mismatch table (cells.tsv vs manifest schema):
    For each of 5 stack axes (model_family, task_slice, G, temperature, seed)
    and 11 telemetry channels, classify:
      - manifest_captures: bool — is there a manifest key that captures this axis?
      - recoverable_via_cell_id: bool — can the value be parsed from cell_id?
      - manifest_equivalent: str — the manifest key (if any) that carries the
        same semantic content, e.g. group_size_schedule <-> G.
  PART B — Per-axis coverage gap with bootstrap CI:
    For each axis, fraction of unique cells.tsv values that are captured
    (or recoverable) by the manifest schema.
  PART C — Cross-reference with P6 registry schema:
    For each field in registry/schema.json stack_record, report whether it
    is present in the live MIN-REPORT manifest schema.
  PART D — Manifest's per-cell discriminative power:
    Two distinctness metrics:
      (1) n_distinct_minrep_str — number of distinct (7-item) manifest strings
      (2) n_distinct_with_cell_extension — distinct (7-item + cell-extracted)
    The gap is the **augmentation value** of recovering cell_id info into
    the manifest schema.
  PART E — Missing/ambiguous fields table (the brief's deliverable):
    Two outputs:
      (i)  missing_fields.tsv — fields in cells.tsv/registry that the
           manifest schema omits
      (ii) ambiguous_fields.tsv — manifest keys whose value-string is
           not uniquely machine-parseable into the underlying axis value
"""
import json, csv, glob, os, math, random, re
from collections import defaultdict, Counter

ROOT = "/home/claude/tinker-rl-lab-minimax"
MAN_DIR = f"{ROOT}/experiments/results/mega_20260704/manifests"
CELLS = f"{ROOT}/experiments/results/mega_20260704/cells.tsv"
OUT = f"{ROOT}/experiments/results/p5p8"
SCHEMA = f"{ROOT}/registry/schema.json"

random.seed(20260705)


def shannon(counter):
    n = sum(counter.values())
    if n == 0:
        return 0.0
    h = 0.0
    for v in counter.values():
        if v > 0:
            p = v / n
            h -= p * math.log2(p)
    return h


def main():
    # ---- Load manifest schema (from disk, not registry) ----
    sample_manifest = json.load(open(f"{MAN_DIR}/Qwen-Qwen3-5-4B_gsm8k_easy_G2_t0.6_s0_923b060d59.json"))
    manifest_keys = sorted(sample_manifest.keys())
    # ---- Load all manifests ----
    manifests = []
    for p in sorted(glob.glob(f"{MAN_DIR}/*.json")):
        with open(p) as f:
            m = json.load(f)
        m["_path"] = p
        m["_name"] = os.path.basename(p).replace(".json", "")
        manifests.append(m)
    # ---- Load cells.tsv ----
    with open(CELLS) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    cell_axis_cols = ["model_family", "task_slice", "G", "temperature", "seed"]
    tel_cols = ["n_groups", "mean_reward", "zvf", "pcd", "mean_completion_len",
                "std_completion_len", "sampled_tokens", "cumulative_sampled_tokens",
                "sample_errors"]

    # ---- PART A: manifest <-> cells.tsv axis mapping ----
    manifest_equiv = {
        "model_family": None,        # NOT in manifest
        "task_slice": "heldout_split",  # semantically equivalent
        "G": "group_size_schedule",     # semantically equivalent (fixed-G=N)
        "temperature": None,            # NOT in manifest
        "seed": None,                   # NOT in manifest
    }
    # ----------------------------------------------------------------
    # PART A rows
    # ----------------------------------------------------------------
    schema_table = []
    for axis in cell_axis_cols:
        n_unique_cells = len(set(r[axis] for r in rows))
        equiv_key = manifest_equiv[axis]
        if equiv_key is None:
            schema_table.append({
                "axis": axis,
                "manifest_key": "<MISSING>",
                "captured": False,
                "recoverable_via_cell_id": True,
                "n_unique_cells": n_unique_cells,
                "n_unique_manifest_equiv": 0,
                "captured_fraction": 0.0,
                "equiv_classifier": "ABSENT",
            })
        else:
            # try to recover values from manifest
            cell_vals = set(r[axis] for r in rows)
            man_vals = set(m.get(equiv_key) for m in manifests)
            # group_size_schedule values: "fixed-G=2" — extract N from string
            if equiv_key == "group_size_schedule":
                def extract(man_v, cell_v):
                    m = re.match(r"fixed-G=(\d+)", str(man_v))
                    if not m:
                        return False
                    try:
                        return int(m.group(1)) == int(cell_v)
                    except Exception:
                        return False
                # fraction of cells whose G matches the manifest's group_size_schedule
                n_match = sum(1 for r in rows for m in manifests
                              if r["cell_id"] in m["_name"] and extract(m[equiv_key], r["G"]))
            elif equiv_key == "heldout_split":
                n_match = sum(1 for r in rows for m in manifests
                              if r["cell_id"] in m["_name"] and str(m[equiv_key]) == str(r["task_slice"]))
            else:
                n_match = 0
            frac = n_match / len(rows) if rows else 0.0
            schema_table.append({
                "axis": axis,
                "manifest_key": equiv_key,
                "captured": True,
                "recoverable_via_cell_id": False,
                "n_unique_cells": n_unique_cells,
                "n_unique_manifest_equiv": len(man_vals),
                "captured_fraction": round(frac, 4),
                "equiv_classifier": "EQUIV_PRESENT",
            })
    # ----------------------------------------------------------------
    # PART B: per-axis coverage with bootstrap CI on captured_fraction
    # ----------------------------------------------------------------
    B = 2000
    boot_rows = []
    for entry in schema_table:
        if entry["equiv_classifier"] != "ABSENT":
            # captured_fraction is a point estimate; bootstrap on a synthetic
            # indicator vector where each manifest either matches or not
            ind = []
            if entry["manifest_key"] == "group_size_schedule":
                for r in rows:
                    mp = next((m for m in manifests if r["cell_id"] in m["_name"]), None)
                    if mp:
                        mm = re.match(r"fixed-G=(\d+)", str(mp[entry["manifest_key"]]))
                        ind.append(1 if mm and int(mm.group(1)) == int(r[entry["axis"]]) else 0)
                    else:
                        ind.append(0)
            elif entry["manifest_key"] == "heldout_split":
                for r in rows:
                    mp = next((m for m in manifests if r["cell_id"] in m["_name"]), None)
                    if mp:
                        ind.append(1 if str(mp[entry["manifest_key"]]) == str(r[entry["axis"]]) else 0)
                    else:
                        ind.append(0)
            else:
                ind = []
            # bootstrap CI
            boots = []
            for _ in range(B):
                s = sum(ind[random.randrange(len(ind))] for _ in range(len(ind)))
                boots.append(s / len(ind))
            boots.sort()
            lo = boots[int(0.025 * B)]
            hi = boots[int(0.975 * B) - 1]
        else:
            lo, hi = 0.0, 0.0
        entry["captured_fraction_ci_lo"] = round(lo, 4)
        entry["captured_fraction_ci_hi"] = round(hi, 4)
        entry["captured_fraction_ci_excludes_zero"] = (lo > 0.0) if entry["captured_fraction"] > 0 else False
        boot_rows.append(entry)

    # ----------------------------------------------------------------
    # PART C: registry schema cross-reference
    # ----------------------------------------------------------------
    with open(SCHEMA) as f:
        reg = json.load(f)
    stack_props = reg["$defs"]["stack_record"]["properties"].keys()
    reg_keys = sorted(stack_props)
    manifest_present = set(manifest_keys) - {"_path", "_name"}
    reg_only = sorted(set(reg_keys) - manifest_present)
    manifest_only = sorted(manifest_present - set(reg_keys))
    in_both = sorted(manifest_present & set(reg_keys))

    # ----------------------------------------------------------------
    # PART D: per-cell discriminative power augmentation
    # ----------------------------------------------------------------
    # baseline: 7-item manifest string EXCLUDING cell_id and per_step_zvf_path
    # (cell_id is a unique pointer; per_step_zvf_path is unique per cell)
    # these act as cell identifiers, not stack descriptors
    descriptor_keys = sorted(set(manifest_present) - {"cell_id", "per_step_zvf_path"})
    def manifest_str(m):
        return "|".join(str(m.get(k, "<NA>")) for k in descriptor_keys)
    base_strs = [manifest_str(m) for m in manifests]
    n_distinct_base = len(set(base_strs))

    # extended: 7-item manifest + recovered cell_id axes (G, task_slice, temperature, seed, model_family)
    def extended_str(m):
        # parse cell_id like Qwen-Qwen3-5-4B_gsm8k_easy_G2_t0.6_s0_xxx
        cid = m["_name"]
        # recover model_family from prefix before first '_'
        # parse G, temperature, seed
        def recover_g(cid):
            mm = re.search(r"_G(\d+)_", cid)
            return mm.group(1) if mm else "?"
        def recover_temp(cid):
            mm = re.search(r"_t([\d.]+)_", cid)
            return mm.group(1) if mm else "?"
        def recover_seed(cid):
            mm = re.search(r"_s(\d+)_", cid)
            return mm.group(1) if mm else "?"
        def recover_task(cid):
            # task is between model_family and _G
            mm = re.search(r"_(gsm8k_\w+|humaneval_subset)_", cid)
            return mm.group(1) if mm else "?"
        def recover_model_family(cid):
            # model_family is the leading prefix before the task
            mm = re.search(r"^(.+?)_(gsm8k_\w+|humaneval_subset)_", cid)
            return mm.group(1) if mm else "?"
        ext = "|".join(str(m.get(k, "<NA>")) for k in descriptor_keys)
        ext += f"|{recover_model_family(cid)}|{recover_task(cid)}|{recover_g(cid)}|{recover_temp(cid)}|{recover_seed(cid)}"
        return ext
    ext_strs = [extended_str(m) for m in manifests]
    n_distinct_ext = len(set(ext_strs))

    # ----------------------------------------------------------------
    # PART E: missing/ambiguous fields tables
    # ----------------------------------------------------------------
    missing_rows = []
    for axis in cell_axis_cols + tel_cols:
        n_unique = len(set(str(r[axis]) for r in rows))
        # is this axis in the manifest schema at all?
        in_manifest = (manifest_equiv.get(axis) is not None)
        # is it recoverable from cell_id?
        recoverable = True  # cell_id encodes (model_family, task_slice, G, temp, seed)
        missing_rows.append({
            "field": axis,
            "field_class": "stack_axis" if axis in cell_axis_cols else "telemetry",
            "in_manifest_schema": in_manifest,
            "recoverable_via_cell_id": recoverable,
            "n_unique_values_in_corpus": n_unique,
            "gap_class": "OK" if in_manifest else ("RECOVERABLE" if recoverable else "MISSING"),
        })

    ambiguous_rows = []
    # for each manifest key, evaluate "ambiguity" = how many cells.tsv axes are
    # mixed into the same key value
    for k in sorted(manifest_present):
        if k in ("cell_id", "per_step_zvf_path"):
            continue
        vals = set(str(m.get(k)) for m in manifests)
        # group_size_schedule: does one value uniquely identify the G axis?
        if k == "group_size_schedule":
            # parse the G value
            g_vals = set()
            for v in vals:
                mm = re.match(r"fixed-G=(\d+)", v)
                if mm:
                    g_vals.add(int(mm.group(1)))
            ambiguous_rows.append({
                "field": k,
                "n_unique_manifest_values": len(vals),
                "encodes_axes": "G",
                "uniquely_parseable": (len(vals) == 5 and all(re.match(r"fixed-G=\d+", v) for v in vals)),
                "parseable_to_axis": "G (clean)",
            })
        elif k == "heldout_split":
            ambiguous_rows.append({
                "field": k,
                "n_unique_manifest_values": len(vals),
                "encodes_axes": "task_slice",
                "uniquely_parseable": all(v in {"gsm8k_easy", "gsm8k_hard", "humaneval_subset"} for v in vals),
                "parseable_to_axis": "task_slice (clean)",
            })
        elif k == "decontamination_notes":
            # does this also encode task_slice?
            ambiguous_rows.append({
                "field": k,
                "n_unique_manifest_values": len(vals),
                "encodes_axes": "task_slice (implicit)",
                "uniquely_parseable": False,
                "parseable_to_axis": "task_slice (implicit via prefix)",
            })
        elif k == "loss_form":
            ambiguous_rows.append({
                "field": k,
                "n_unique_manifest_values": len(vals),
                "encodes_axes": "none",
                "uniquely_parseable": True,
                "parseable_to_axis": "placeholder",
            })
        elif k == "ref_policy_kl":
            ambiguous_rows.append({
                "field": k,
                "n_unique_manifest_values": len(vals),
                "encodes_axes": "none",
                "uniquely_parseable": True,
                "parseable_to_axis": "placeholder (n/a)",
            })
        elif k == "sampler_backend_precision":
            ambiguous_rows.append({
                "field": k,
                "n_unique_manifest_values": len(vals),
                "encodes_axes": "openness",
                "uniquely_parseable": True,
                "parseable_to_axis": "openness (single value)",
            })

    # ----------------------------------------------------------------
    # Write outputs
    # ----------------------------------------------------------------
    schema_path = f"{OUT}/p5_iter97_schema_mismatch.tsv"
    with open(schema_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(boot_rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in boot_rows:
            w.writerow(r)

    missing_path = f"{OUT}/p5_iter97_missing_fields.tsv"
    with open(missing_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(missing_rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in missing_rows:
            w.writerow(r)

    ambiguous_path = f"{OUT}/p5_iter97_ambiguous_fields.tsv"
    with open(ambiguous_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ambiguous_rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in ambiguous_rows:
            w.writerow(r)

    summary = {
        "n_manifests": len(manifests),
        "n_cells_tsv_rows": len(rows),
        "schema_mismatch_table": boot_rows,
        "missing_fields_table": missing_rows,
        "ambiguous_fields_table": ambiguous_rows,
        "registry_cross_reference": {
            "n_registry_keys": len(reg_keys),
            "n_manifest_keys": len(manifest_present),
            "in_both": in_both,
            "registry_only_keys": reg_only,
            "manifest_only_keys": manifest_only,
        },
        "augmentation_value": {
            "n_distinct_minrep_strings": n_distinct_base,
            "n_distinct_with_cell_extension": n_distinct_ext,
            "n_cells": len(manifests),
            "augmentation_delta": n_distinct_ext - n_distinct_base,
            "augmentation_pct": round((n_distinct_ext - n_distinct_base) / len(manifests) * 100, 2),
        },
    }
    summary_path = f"{OUT}/p5_iter97_schema_mismatch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote: {schema_path}")
    print(f"wrote: {missing_path}")
    print(f"wrote: {ambiguous_path}")
    print(f"wrote: {summary_path}")
    print()
    print("=== H1 (axis coverage gap) ===")
    for r in boot_rows:
        print(f"  {r['axis']:14s} manifest_key={r['manifest_key']:25s} "
              f"captured_frac={r['captured_fraction']:.3f} "
              f"CI=[{r['captured_fraction_ci_lo']:.3f}, {r['captured_fraction_ci_hi']:.3f}] "
              f"class={r['equiv_classifier']}")
    print()
    print("=== H2 (augmentation value) ===")
    print(f"  baseline distinct manifest strings: {n_distinct_base}")
    print(f"  extended (with cell-extracted axes): {n_distinct_ext}")
    print(f"  delta = {n_distinct_ext - n_distinct_base}")
    print()
    print("=== H3 (registry cross-reference) ===")
    print(f"  registry keys: {len(reg_keys)}")
    print(f"  manifest keys: {len(manifest_present)}")
    print(f"  in both: {len(in_both)}")
    print(f"  registry-only (would extend MIN-REPORT): {reg_only}")
    print(f"  manifest-only: {manifest_only}")


if __name__ == "__main__":
    main()