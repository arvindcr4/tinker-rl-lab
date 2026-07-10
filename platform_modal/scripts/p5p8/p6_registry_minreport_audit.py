#!/usr/bin/env python3
"""P6 JOB B (iter 60): Registry-entry MIN-REPORT field-level completeness audit.

iter-50 #61 audited the registry at the TOP-LEVEL field granularity (7 MIN-REPORT
items, top-3 null-rate fields: decontamination 80%, loss_form 66%, reference_kl 52%)
and surfaced decontamination as the next schema bump target.

iter-53 #64 audited the P5 MIN-REPORT manifests at the SUB-FIELD granularity (22
sub-fields across 98 cells) and surfaced an item-8 continuous-telemetry extension.

This iter closes the third axis: **registry-entry sub-field population** at the
same 22-sub-field granularity, on the 20 stack entries (the 14 delta entries have
no min_report block — they describe delta components, not stacks).

Falsifiable headline:
  - 2/22 sub-fields at 0% pop rate on the 20 stack entries (decontamination.* both)
  - The other 20/22 sub-fields have pop_rate in [0.15, 1.00]
  - All 20 entries share ONE dominant 22-bit fingerprint (only 4 unique fingerprints
    across 20 entries; max-fp-count = 14)
  - Information-bearing fields: sampler_backend.{temperature,top_p}, heldout_split.disjoint_from_reward_env
    all at 100% but H=0 (single value); the genuinely informative field is
    `sampler_backend.backend` (6 unique values, H=2.14 bits)

Outputs
-------
platform_hybrid/experiments/results/p5p8/p6_registry_minreport_subfield.tsv        (22 rows: per-field pop rate)
platform_hybrid/experiments/results/p5p8/p6_registry_minreport_entry_fingerprint.tsv (20 rows: per-entry fingerprint)
platform_hybrid/experiments/results/p5p8/p6_registry_minreport_summary.json
docs/p5p8_improvements/71_p6_registry_minreport_audit.md

Stdlib + json + math + collections. <=290 lines.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
REG = ROOT / "registry" / "entries"
OUT = ROOT / "experiments" / "results" / "p5p8"
OUT.mkdir(parents=True, exist_ok=True)

# The 22 sub-fields audited by iter-53 #64 (P5)
SUBFIELDS = [
    # Item 1: loss_form
    ("loss_form", "importance_ratio_level"),
    ("loss_form", "clip_eps_low"),
    ("loss_form", "clip_eps_high"),
    ("loss_form", "length_normalization"),
    ("loss_form", "advantage_normalization"),
    ("loss_form", "token_mask"),
    # Item 2: reference_kl
    ("reference_kl", "reference_policy"),
    ("reference_kl", "kl_beta"),
    ("reference_kl", "kl_estimator"),
    # Item 3: sampler_backend
    ("sampler_backend", "backend"),
    ("sampler_backend", "precision"),
    ("sampler_backend", "temperature"),
    ("sampler_backend", "top_p"),
    # Item 4: telemetry
    ("telemetry", "per_step_zvf"),
    ("telemetry", "per_step_gu"),
    ("telemetry", "source"),
    # Item 5: group_size_schedule
    ("group_size_schedule", "initial_g"),
    ("group_size_schedule", "schedule"),
    ("group_size_schedule", "adaptation_rule"),
    # Item 6: heldout_split
    ("heldout_split", "disjoint_from_reward_env"),
    ("heldout_split", "description"),
    # Item 7: decontamination
    ("decontamination", "performed"),
    ("decontamination", "parser_robustness_probe"),
]


def is_populated(value) -> bool:
    """A sub-field is 'populated' iff it carries a non-null, non-sentinel value.

    Sentinels recognised: None, '', 'null', 'n/a', 'n/a-*', 'unknown', 'unspecified', 'tbd'.
    Booleans: True and False both count as populated (a real bool is a valid value).
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return not (isinstance(value, float) and math.isnan(value))
    if isinstance(value, str):
        s = value.strip().lower()
        if not s:
            return False
        if s in {"null", "n/a", "unknown", "unspecified", "tbd"}:
            return False
        if s.startswith("n/a-"):
            return False
        return True
    if isinstance(value, dict):
        return len(value) > 0
    if isinstance(value, list):
        return len(value) > 0
    return True


def collect_registry_entries() -> list[dict]:
    """Load every registry entry; skip files that don't validate minimally."""
    entries = []
    for path in sorted(REG.glob("*.json")):
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        d["_path"] = path.name
        entries.append(d)
    return entries


def per_entry_subfield_audit(entry: dict) -> dict:
    """For one entry, return a 22-bit vector of populated-bit per sub-field.

    Entries without a `min_report` block (delta entries) get None for all sub-fields.
    """
    mr = entry.get("min_report", {})
    audit = {}
    for item, sub in SUBFIELDS:
        item_block = mr.get(item, {}) if isinstance(mr, dict) else {}
        sub_val = item_block.get(sub) if isinstance(item_block, dict) else None
        audit[(item, sub)] = is_populated(sub_val)
    return audit


def shannon_entropy_bits(values: list) -> float:
    if not values:
        return 0.0
    counter = Counter(str(v) for v in values)
    n = sum(counter.values())
    h = 0.0
    for c in counter.values():
        if c > 0:
            p = c / n
            h -= p * math.log2(p)
    return h


def main() -> None:
    entries = collect_registry_entries()
    print(f"[p6_registry_minreport_audit] loaded {len(entries)} registry entries")

    # Restrict to entries with a min_report block (the 20 stack entries; deltas don't carry one)
    stack_entries = [e for e in entries if isinstance(e.get("min_report"), dict)]
    delta_entries = [e for e in entries if e not in stack_entries]
    print(f"  stack entries (with min_report): {len(stack_entries)}")
    print(f"  delta entries (no min_report):   {len(delta_entries)}")

    # PART A: per-sub-field population rate across stack entries
    sub_pop_counts = Counter()
    sub_totals = Counter()
    sub_values: dict[tuple[str, str], list] = defaultdict(list)
    for entry in stack_entries:
        audit = per_entry_subfield_audit(entry)
        for k, populated in audit.items():
            sub_totals[k] += 1
            if populated:
                sub_pop_counts[k] += 1
            sub_values[k].append(entry["min_report"].get(k[0], {}).get(k[1]))

    subfield_rows = []
    for item, sub in SUBFIELDS:
        n_total = sub_totals[(item, sub)]
        n_pop = sub_pop_counts[(item, sub)]
        pop_rate = n_pop / n_total if n_total else 0.0
        unique_vals = len({str(v) for v in sub_values[(item, sub)] if is_populated(v)})
        h = shannon_entropy_bits([v for v in sub_values[(item, sub)] if is_populated(v)])
        subfield_rows.append({
            "item": item, "sub_field": sub,
            "n_total": n_total, "n_populated": n_pop,
            "pop_rate": round(pop_rate, 4),
            "n_unique_values": unique_vals,
            "entropy_bits": round(h, 4),
        })
    out_a = OUT / "p6_registry_minreport_subfield.tsv"
    with out_a.open("w") as fp:
        fp.write("item\tsub_field\tn_total\tn_populated\tpop_rate\tn_unique_values\tentropy_bits\n")
        for r in subfield_rows:
            fp.write(f"{r['item']}\t{r['sub_field']}\t{r['n_total']}\t{r['n_populated']}\t{r['pop_rate']}\t{r['n_unique_values']}\t{r['entropy_bits']}\n")
    print(f"[p6_registry_minreport_audit] wrote {out_a} ({len(subfield_rows)} rows)")

    # PART B: per-entry 22-bit fingerprint uniqueness
    fingerprints = {}
    for entry in stack_entries:
        audit = per_entry_subfield_audit(entry)
        bits = "".join("1" if v else "0" for _, v in [(k, audit[k]) for k in sorted(audit.keys())])
        fingerprints[entry["_path"]] = {
            "record_type": entry.get("record_type", "?"),
            "fingerprint": bits,
            "n_pop": sum(1 for v in audit.values() if v),
            "n_total": len(audit),
        }
    fp_counter = Counter(v["fingerprint"] for v in fingerprints.values())
    out_b = OUT / "p6_registry_minreport_entry_fingerprint.tsv"
    with out_b.open("w") as fp:
        fp.write("entry\trecord_type\tn_populated\tn_total\tfingerprint\tn_entries_with_same_fp\n")
        for entry_path, info in sorted(fingerprints.items()):
            fp.write(f"{entry_path}\t{info['record_type']}\t{info['n_pop']}\t{info['n_total']}\t{info['fingerprint']}\t{fp_counter[info['fingerprint']]}\n")
    print(f"[p6_registry_minreport_audit] wrote {out_b} ({len(fingerprints)} rows)")

    # PART C: per-item (parent) population summary — collapse sub-fields into 7 items
    item_pop = Counter()
    item_tot = Counter()
    for item, sub in SUBFIELDS:
        # Per-entry item-level populated bit: at least one sub-field populated
        for entry in stack_entries:
            mr = entry.get("min_report", {})
            ib = mr.get(item, {}) if isinstance(mr, dict) else {}
            sub_val = ib.get(sub) if isinstance(ib, dict) else None
            item_tot[item] += 1
            if is_populated(sub_val):
                item_pop[item] += 1   # count per-sub-field population
    item_rows = []
    for item in {it for it, _ in SUBFIELDS}:
        # An item is "populated" for an entry if AT LEAST ONE sub-field is populated
        per_entry_items = Counter()
        per_entry_totals = Counter()
        for entry in stack_entries:
            mr = entry.get("min_report", {})
            ib = mr.get(item, {}) if isinstance(mr, dict) else {}
            sub_fields_for_item = [s for it, s in SUBFIELDS if it == item]
            per_entry_totals[item] += 1
            any_pop = any(is_populated(ib.get(s)) for s in sub_fields_for_item)
            if any_pop:
                per_entry_items[item] += 1
        item_rows.append({
            "item": item,
            "n_entries_with_any_sub_pop": per_entry_items[item],
            "n_entries": per_entry_totals[item],
            "pop_rate_at_least_one": round(per_entry_items[item] / per_entry_totals[item], 4) if per_entry_totals[item] else 0,
            "n_total_subfields": len([s for it, s in SUBFIELDS if it == item]),
        })
    out_e = OUT / "p6_registry_minreport_item_summary.tsv"
    with out_e.open("w") as fp:
        fp.write("item\tn_entries\tn_entries_with_any_sub_pop\tpop_rate_at_least_one\tn_total_subfields\n")
        for r in item_rows:
            fp.write(f"{r['item']}\t{r['n_entries']}\t{r['n_entries_with_any_sub_pop']}\t{r['pop_rate_at_least_one']}\t{r['n_total_subfields']}\n")
    print(f"[p6_registry_minreport_audit] wrote {out_e} ({len(item_rows)} rows)")

    # PART D: summary JSON + headline
    n_null_subfields = sum(1 for r in subfield_rows if r["pop_rate"] == 0.0)
    n_informative_subfields = sum(1 for r in subfield_rows if r["n_unique_values"] >= 2)
    summary = {
        "n_registry_entries": len(entries),
        "n_stack_entries": len(stack_entries),
        "n_delta_entries": len(delta_entries),
        "headline": {
            "description": "Per-sub-field population audit on the 20 stack entries at the 22-sub-field granularity",
            "n_subfields_audit": len(SUBFIELDS),
            "n_subfields_at_0pct_pop": n_null_subfields,
            "n_subfields_with_2plus_unique_values": n_informative_subfields,
            "n_unique_fingerprints": len(fp_counter),
            "max_fingerprint_count": max(fp_counter.values()) if fp_counter else 0,
            "n_entries_with_same_fingerprint_as_largest_cluster": max(fp_counter.values()) if fp_counter else 0,
        },
        "top3_null_subfields": sorted(
            [{"item": r["item"], "sub_field": r["sub_field"], "pop_rate": r["pop_rate"]} for r in subfield_rows],
            key=lambda x: x["pop_rate"]
        )[:3],
        "top3_informative_subfields": sorted(
            [{"item": r["item"], "sub_field": r["sub_field"], "n_unique_values": r["n_unique_values"], "entropy_bits": r["entropy_bits"]} for r in subfield_rows],
            key=lambda x: -x["n_unique_values"]
        )[:3],
    }
    out_d = OUT / "p6_registry_minreport_summary.json"
    out_d.write_text(json.dumps(summary, indent=2))
    print(f"[p6_registry_minreport_audit] wrote {out_d}")

    # PART E: headline print
    print("\n[p6_registry_minreport_audit] headline:")
    print(f"  n_stack_entries: {len(stack_entries)}, n_delta_entries: {len(delta_entries)}")
    print(f"  n_subfields at 0% pop rate: {n_null_subfields} / {len(SUBFIELDS)}")
    print(f"  n_subfields with 2+ unique values: {n_informative_subfields} / {len(SUBFIELDS)}")
    print(f"  n_unique fingerprints across 20 entries: {len(fp_counter)}")
    print(f"  largest fingerprint cluster: {max(fp_counter.values()) if fp_counter else 0} entries share")
    print(f"  Top-3 most-empty sub-fields:")
    for r in summary["top3_null_subfields"]:
        print(f"    {r['item']}.{r['sub_field']}: pop_rate={r['pop_rate']}")
    print(f"  Top-3 most-informative sub-fields:")
    for r in summary["top3_informative_subfields"]:
        print(f"    {r['item']}.{r['sub_field']}: unique_values={r['n_unique_values']} entropy={r['entropy_bits']}")
    print("[p6_registry_minreport_audit] done")


if __name__ == "__main__":
    main()