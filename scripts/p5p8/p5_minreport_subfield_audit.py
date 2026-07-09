#!/usr/bin/env python3
"""P5-53 -- MIN-REPORT sub-field completeness audit + minimum-viable-extension (MVE)
recommendation on the live 98-cell mega corpus.

Two PARTs on the same data:
  - 98 MIN-REPORT manifests in experiments/results/mega_20260704/manifests/
  - the live ledger experiments/results/mega_20260704/cells.tsv

PART A: SUB-FIELD COVERAGE
  The seven MIN-REPORT items decompose into 20 sub-fields across the seven
  rows of Table sec:p5-minreport (5+3+4+2+2+3+2 = 21; we report 20 because
  Item 1's "clip values and asymmetry" is one logical field that takes
  two concrete values -- clip_low and clip_high -- for clarity).

    Item 1 (Loss form)             : {ratio_level, clip_low, clip_high,
                                       token_mask, advantage_norm,
                                       dynamic_sampling}
    Item 2 (Reference policy & KL)  : {ref_snapshot, kl_coef, kl_estimator}
    Item 3 (Sampler/backend/prec)  : {sampler_backend, sampling_engine,
                                       precision, decoding_parameters}
    Item 4 (Per-step ZVF/GU)       : {zvf_traj, GU_per_step}
    Item 5 (Group-size schedule)   : {G_value, schedule_form}
    Item 6 (Held-out split)        : {split_name, split_size, disjointness}
    Item 7 (Decontam/parser probe) : {decontam_check, parser_probe}

  Every manifest yields exactly 7 string values (one per item, indexed by
  Item #), but the 20 sub-fields are populated only when the item value is
  structured (not "n/a"). For each sub-field this iter measures:
    coverage_pct = (% of manifests where the sub-field can be parsed out)
    n_unique     = number of distinct values observed
    H_bits       = Shannon entropy (base 2)
  A sub-field with coverage_pct = 100 and H_bits = 0 is "honest but vacuous".

PART B: MINIMUM-VIABLE-EXTENSION (MVE) ANALYSIS
  Since iter-52 #63 established that all 98 cells yield
  H(MIN-REPORT items) ~= 0 (3 of 7 items are vacuous by entropy), this iter
  asks: which CANDIDATE extension fields from the live cells.tsv schema would,
  if included alongside the seven items, raise per-cell distinguishability?
  For each candidate we compute:
    H(field)                       -- field's intrinsic entropy in bits
    n_distinct_minrep_alone        -- distinct MIN-REPORT-strings (baseline)
    n_distinct_with_extension      -- distinct (MIN-REPORT X, field-value) tuples
    delta_distinct                 -- gain in distinct profiles
    delta_distinct_pct             -- gain / n_cells
  Greedy MVE: smallest subset of candidates (in rank order) that lifts
  distinct profiles to >= n // 2 (the "at least half the cells uniquely
  identified" bar).

Outputs:
  experiments/results/p5p8/p5_minreport_subfield_audit.tsv          (PART A per-sub-field)
  experiments/results/p5p8/p5_minreport_subfield_audit_per_item.tsv (PART A per-item aggregate)
  experiments/results/p5p8/p5_minreport_subfield_audit_summary.json (PART A summary)
  experiments/results/p5p8/p5_minreport_mve.tsv                     (PART B per-extension table)
  experiments/results/p5p8/p5_minreport_mve_summary.json            (PART B summary)
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CELLS = ROOT / "experiments" / "results" / "mega_20260704" / "cells.tsv"
MANIFESTS = ROOT / "experiments" / "results" / "mega_20260704" / "manifests"
OUT_DIR = ROOT / "experiments" / "results" / "p5p8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- the 7 MIN-REPORT items and their sub-fields (from sec:p5-stack) -----
SUBFIELDS: dict[str, list[str]] = {
    "loss_form":                  # Item 1
        ["ratio_level", "clip_low", "clip_high", "token_mask",
         "advantage_norm", "dynamic_sampling"],
    "ref_policy_kl":              # Item 2
        ["ref_snapshot", "kl_coef", "kl_estimator"],
    "sampler_backend_precision":  # Item 3
        ["sampler_backend", "sampling_engine", "precision",
         "decoding_parameters"],
    "per_step_zvf_path":          # Item 4
        ["zvf_traj", "GU_per_step"],
    "group_size_schedule":        # Item 5
        ["G_value", "schedule_form"],
    "heldout_split":# Item 6
        ["split_name", "split_size", "disjointness"],
    "decontamination_notes":      # Item 7
        ["decontam_check", "parser_probe"],
}

# Items where the value is ALWAYS a structured string (file path / schedule /
# heldout name / decontam notes) -- all sub-fields under such an item are
# implicitly covered when the value is non-empty. Items where the value can
# be a sentinel (n/a) have an opt-out:
SENTINEL_PREFIXES = {
    "loss_form":                ["n/a"],   # e.g. "n/a-sampling"
    "ref_policy_kl":            ["n/a"],   # e.g. "n/a-no-kl"
    # Items 3/4/5/6/7: any non-empty value is structured
}


def _covered_subfields(item_key: str, value: str) -> list[str]:
    """Return the sub-fields implicitly populated by this item value.

    Sentinel ("n/a" + optional reason) items contribute zero sub-fields;
    any non-empty structured value implicitly populates all of the item's
    sub-fields. This is a CONSERVATIVE reading of the manifest format:
    the current manifests store ONE STRING per item, so the sub-fields
    cannot be parsed out independently -- they exist as a conjunction
    that is either all-present (when the item string is structured) or
    all-absent (when it's a sentinel).
    """
    if value is None:
        return []
    v = str(value).strip()
    if not v:
        return []
    prefixes = SENTINEL_PREFIXES.get(item_key, [])
    head = v.split("-", 1)[0]
    if head in prefixes:
        return []
    return SUBFIELDS[item_key]


# ----- PART A -----
def part_a() -> dict:
    manifests = sorted(MANIFESTS.glob("*.json"))
    if not manifests:
        print(f"ERROR: no manifests found under {MANIFESTS}", file=sys.stderr)
        sys.exit(1)
    n = len(manifests)
    print(f"PART A: scanning {n} manifests under {MANIFESTS.name}/")

    all_subs: list[str] = [s for sub in SUBFIELDS.values() for s in sub]
    sub_covered: dict[str, int] = {sf: 0 for sf in all_subs}
    item_present: dict[str, int] = {it: 0 for it in SUBFIELDS}
    item_sub_covered: dict[str, set[str]] = {it: set() for it in SUBFIELDS}
    sub_value_counter: dict[str, Counter] = {sf: Counter() for sf in all_subs}
    for mpath in manifests:
        with open(mpath) as f:
            d = json.load(f)
        for item in SUBFIELDS:
            value = d.get(item, "")
            subs = _covered_subfields(item, value)
            if value is not None and str(value).strip():
                item_present[item] += 1
            for sf in subs:
                sub_covered[sf] += 1
                item_sub_covered[item].add(sf)
                sub_value_counter[sf][value] += 1

    # per-sub-field entropy. For Item-4/5/6/7 we can compute H over the
    # DISTINCT item values (which is what the manifest carries); for Item-1/2/3
    # we only see sentinel strings, so the unique-value count is just "n/a-*"
    # and H ~= 0.
    table_rows: list[dict] = []
    n_subs = len(all_subs)
    for sf in all_subs:
        covered = sub_covered[sf]
        cov_pct = 100.0 * covered / n
        # entropy over covered/uncovered
        if covered == 0:
            H = 0.0
            n_unique_when_covered = 0
        elif covered == n:
            # all manifest items structurally carry this sub-field; H of the
            # (covered, not-covered) pair is 0; but we can still compute H over
            # the distinct item values that populate it (always >= 1)
            counts = sub_value_counter[sf]
            if len(counts) == 1:
                H = 0.0
                n_unique_when_covered = 1
            else:
                H = _h_from_counts(counts)
                n_unique_when_covered = len(counts)
        else:
            p = covered / n
            H = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
            n_unique_when_covered = 2  # "covered" or "not covered"
        vacuous = (covered == n and H == 0.0) or covered == 0
        table_rows.append({
            "sub_field": sf,
            "covered_n": covered,
            "coverage_pct": round(cov_pct, 2),
            "n_unique_when_covered": n_unique_when_covered,
            "H_bits": round(H, 4),
            "vacuous": vacuous,
        })

    # rank sort: zero-coverage first (most-missing), then vacuous, then
    # informative
    table_rows.sort(key=lambda r: (r["coverage_pct"], r["H_bits"], r["sub_field"]))

    # per-item aggregates
    per_item_rows: list[dict] = []
    for item in SUBFIELDS:
        per_item_rows.append({
            "item": item,
            "n_subfields_declared": len(SUBFIELDS[item]),
            "n_subfields_covered": len(item_sub_covered[item]),
            "item_present_pct": round(100.0 * item_present[item] / n, 2),
        })

    out_tsv = OUT_DIR / "p5_minreport_subfield_audit.tsv"
    with open(out_tsv, "w", newline="") as f:
        cols = ["sub_field", "covered_n", "coverage_pct",
                "n_unique_when_covered", "H_bits", "vacuous"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in table_rows:
            w.writerow(r)

    out_item_tsv = OUT_DIR / "p5_minreport_subfield_audit_per_item.tsv"
    with open(out_item_tsv, "w", newline="") as f:
        cols = ["item", "n_subfields_declared", "n_subfields_covered",
                "item_present_pct"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(per_item_rows)

    summary = {
        "n_manifests_scanned": n,
        "n_top_items": len(SUBFIELDS),
        "n_subfields_total": n_subs,
        "n_subfields_zero_coverage": sum(1 for r in table_rows if r["covered_n"] == 0),
        "n_subfields_full_coverage_vacuous": sum(
            1 for r in table_rows
            if r["covered_n"] == n and r["H_bits"] == 0.0
        ),
        "n_subfields_nontrivial": sum(
            1 for r in table_rows
            if r["covered_n"] > 0 and r["H_bits"] > 0
        ),
        "per_item": per_item_rows,
    }
    out_json = OUT_DIR / "p5_minreport_subfield_audit_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print(f"  -> {out_tsv.relative_to(ROOT)} ({len(table_rows)} rows)")
    print(f"  -> {out_item_tsv.relative_to(ROOT)} ({len(per_item_rows)} rows)")
    print(f"  -> {out_json.relative_to(ROOT)}")
    print(f"  zero-coverage sub-fields:    "
          f"{summary['n_subfields_zero_coverage']}/{n_subs}")
    print(f"  vacuous (covered, H=0):      "
          f"{summary['n_subfields_full_coverage_vacuous']}/{n_subs}")
    print(f"  informative (covered, H>0):  "
          f"{summary['n_subfields_nontrivial']}/{n_subs}")
    return summary


# ----- PART B -----
EXTENSION_CANDIDATES = [
    "model_family", "task_slice", "G", "temperature", "seed",
    "n_groups", "sample_errors",
    "mean_reward", "zvf", "pcd", "mean_completion_len", "std_completion_len",
]


def _h_from_counts(counts: Counter) -> float:
    n = sum(counts.values())
    if n == 0:
        return 0.0
    H = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / n
        H -= p * math.log2(p)
    return H


def _resolve_manifest_path(p: str) -> Path:
    """Resolve a cells.tsv manifest path that may have been written under the
    /home/claude/tinker-rl-lab/ root (the original root) or our local one."""
    p = p.strip()
    if p.startswith("/home/claude/tinker-rl-lab/experiments/"):
        rel = p[len("/home/claude/tinker-rl-lab/experiments/"):]
        return ROOT / "experiments" / rel
    return Path(p)


def part_b() -> dict:
    if not CELLS.exists():
        print(f"ERROR: cells ledger not at {CELLS}", file=sys.stderr)
        sys.exit(1)
    with open(CELLS) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        rows = list(rdr)
    n = len(rows)
    print(f"PART B: {n} cells from cells.tsv")

    # baseline: distinct (MIN-REPORT-7-string) profiles.
    # We compute TWO baselines:
    #  (A) joined string including per_step_zvf_path    -- "raw" identity
    #  (B) joined string EXCLUDING per_step_zvf_path    -- "content only"
    # because Item 4's value is a unique file path per cell and dominates (A).
    sig_minrep_raw: list[str] = []
    sig_minrep_content: list[str] = []
    for row in rows:
        mpath = _resolve_manifest_path(row["manifest_path"])
        try:
            with open(mpath) as f:
                d = json.load(f)
        except Exception:
            d = {}
        keys_content = [k for k in SUBFIELDS if k != "per_step_zvf_path"]
        sig_minrep_raw.append("|".join(str(d.get(k, "")) for k in SUBFIELDS))
        sig_minrep_content.append("|".join(str(d.get(k, "")) for k in keys_content))

    raw_counter = Counter(sig_minrep_raw)
    content_counter = Counter(sig_minrep_content)
    n_distinct_raw = len(raw_counter)
    n_distinct_content = len(content_counter)
    H_raw = _h_from_counts(raw_counter)
    H_content = _h_from_counts(content_counter)
    print(f"  RAW baseline (incl. Item 4 file path): {n_distinct_raw}/{n}  "
          f"H={H_raw:.4f} bits")
    print(f"  CONTENT baseline (excl. Item 4 file path): "
          f"{n_distinct_content}/{n}  H={H_content:.4f} bits")

    # the content baseline is the right test for "what does the standard
    # tell us about a cell, ignoring its unique trajectory file?" --
    # structurally analogous to iter-52 #63's per-item averaged H.
    n_distinct_minrep = n_distinct_content
    H_minrep = H_content
    sig_minrep = sig_minrep_content

    table_rows: list[dict] = []
    sig_with_ext: dict[str, list[str]] = {}
    for fld in EXTENSION_CANDIDATES:
        sig_with_ext[fld] = [
            sig_minrep[i] + "||" + fld + "=" + str(rows[i].get(fld, ""))
            for i in range(n)
        ]
        counter_with = Counter(sig_with_ext[fld])
        n_distinct_with = len(counter_with)
        H_with = _h_from_counts(counter_with)
        delta_distinct = n_distinct_with - n_distinct_minrep
        cells_unique_with = sum(1 for c in counter_with.values() if c == 1)
        table_rows.append({
            "extension_field": fld,
            "H_field_minrep": round(H_minrep, 4),
            "n_distinct_minrep_alone": n_distinct_minrep,
            "n_distinct_with_extension": n_distinct_with,
            "delta_distinct": delta_distinct,
            "delta_distinct_pct": round(100.0 * delta_distinct / n, 2),
            "cells_with_unique_profile": cells_unique_with,
            "H_with_extension": round(H_with, 4),
            "delta_H_bits": round(H_with - H_minrep, 4),
            "rank_by_delta_distinct": None,
        })

    # rank by delta_distinct descending
    table_rows.sort(key=lambda r: -r["delta_distinct"])
    for i, r in enumerate(table_rows, 1):
        r["rank_by_delta_distinct"] = i

    # MVE greedy: smallest cardinality achieving n_distinct >= n/2
    def profile_count(field_subset: list[str]) -> tuple[int, float]:
        profiles: list[tuple] = []
        for i in range(n):
            profiles.append(tuple(sig_with_ext[f][i] for f in field_subset))
        c = Counter(profiles)
        return len(c), _h_from_counts(c)

    chosen: list[str] = []
    chosen_set: list[str] = []
    for r in table_rows:
        chosen.append(r["extension_field"])
        n_dist, H = profile_count(chosen)
        if n_dist >= n // 2:
            chosen_set = list(chosen)
            break
    if not chosen_set:
        chosen_set = list(chosen)

    n_dist_mve, H_mve = profile_count(chosen_set)
    print(f"  MVE (smallest cardinality, n_distinct >= n/2): {chosen_set} "
          f"-> {n_dist_mve}/{n} distinct, H={H_mve:.4f} bits")

    out_tsv = OUT_DIR / "p5_minreport_mve.tsv"
    with open(out_tsv, "w", newline="") as f:
        cols = ["rank_by_delta_distinct", "extension_field",
                "H_field_minrep", "n_distinct_minrep_alone",
                "n_distinct_with_extension", "delta_distinct",
                "delta_distinct_pct", "cells_with_unique_profile",
                "H_with_extension", "delta_H_bits"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in table_rows:
            w.writerow(r)

    summary = {
        "n_cells": n,
        "raw_incl_path_n_distinct": n_distinct_raw,
        "content_excl_path_n_distinct": n_distinct_content,
        "baseline": {
            "raw_incl_path_n_distinct": n_distinct_raw,
            "raw_incl_path_H": round(H_raw, 4),
            "content_excl_path_n_distinct": n_distinct_content,
            "content_excl_path_H": round(H_content, 4),
            "n_distinct_minrep_alone": n_distinct_minrep,
            "H_minrep": round(H_minrep, 4),
            "frac_unique": round(n_distinct_minrep / n, 4),
        },
        "mve": {
            "extension_fields": chosen_set,
            "n_added_fields": len(chosen_set),
            "n_distinct_with_extension": n_dist_mve,
            "H_with_extension": round(H_mve, 4),
            "frac_unique_with_extension": round(n_dist_mve / n, 4),
            "distinct_uplift_pct": round(
                100.0 * (n_dist_mve - n_distinct_minrep) / n, 2),
        },
        "falsifiable_headline": (
            f"CONTENT baseline (MIN-REPORT excluding the unique-per-cell "
            f"per_step_zvf_path) yields {n_distinct_content}/{n} distinct "
            f"profiles (H={H_content:.4f} bits). "
            f"Adding the {len(chosen_set)} MVE fields {chosen_set} lifts the "
            f"distinct profile count to {n_dist_mve} (H={H_mve:.4f} bits, "
            f"a {(n_dist_mve / max(1, n_distinct_content)):.2f}x lift in "
            f"distinct profiles)."
        ),
        "per_field_rows": table_rows,
    }
    out_json = OUT_DIR / "p5_minreport_mve_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"  -> {out_tsv.relative_to(ROOT)}")
    print(f"  -> {out_json.relative_to(ROOT)}")
    return summary


def main() -> int:
    a = part_a()
    b = part_b()
    n_cells = b.get("n_cells", 0)
    print("\n=== ITER 53 -- P5 SUB-FIELD + MVE HEADLINES ===")
    print(
        f"PART A: {a['n_subfields_zero_coverage']}/{a['n_subfields_total']} zero-coverage; "
        f"{a['n_subfields_full_coverage_vacuous']}/{a['n_subfields_total']} vacuous; "
        f"{a['n_subfields_nontrivial']}/{a['n_subfields_total']} informative."
    )
    print(
        f"PART B: MVE = {b['mve']['extension_fields']} -> "
        f"{b['mve']['n_distinct_with_extension']}/{n_cells} distinct "
        f"({b['mve']['frac_unique_with_extension']:.3f}) vs "
        f"{b['baseline']['n_distinct_minrep_alone']}/{n_cells} MIN-REPORT-alone (content baseline)."
    )
    print(
        f"PART B: MVE = {b['mve']['extension_fields']} -> "
        f"{b['mve']['n_distinct_with_extension']}/{b['baseline']['n_distinct_minrep_alone']+b['mve']['distinct_uplift_pct']*98/100:.0f} distinct "
        f"({b['mve']['frac_unique_with_extension']:.3f}) vs "
        f"{b['baseline']['n_distinct_minrep_alone']}/{b['baseline']['n_distinct_minrep_alone'] + (98 - b['baseline']['n_distinct_minrep_alone'])} MIN-REPORT-alone."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
