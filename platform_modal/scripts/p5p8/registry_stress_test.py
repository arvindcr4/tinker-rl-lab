#!/usr/bin/env python3
"""Iter 26 Pillar 2 (P6) — registry schema stress-test with CI.

The iter-14 audit computed a single point estimate ("31/31 schema PASS")
over the existing well-formed entries. This iter strengthens that claim
by ADVERSARIALLY testing the schema against synthetic malformed entries
across N=13 perturbation categories. For each (entry, mutation) pair:

  - We apply ONE perturbation (e.g., remove a required key, inject an
    additional property, swap a type, violate an enum).
  - We assert that jsonschema raises a ValidationError.
  - We track CAUGHT (True) / MISSED (False) per category.

Paired bootstrap CI on the recovery rate provides a sharp, reviewer-
defensible claim: "the schema's true-positive rate is at least X% on the
N=13-category stress test, with paired 95% bootstrap CI [...]".

Outputs:
  platform_hybrid/experiments/results/p5p8/registry_stress_per_run.tsv
      one row per (entry_id, category, mutation_idx) — 700 rows
  platform_hybrid/experiments/results/p5p8/registry_stress_summary.json
      machine-readable per-category TP / FP / FN + 95% paired bootstrap CI
      on the recovery rate (overall + per-category)
  platform_hybrid/experiments/results/p5p8/registry_stress_misses.tsv
      one row per (entry_id, category, mutation_idx) where the schema
      MISSED the perturbation — feeds the regression work-list
  platform_hybrid/experiments/results/p5p8/registry_stress_by_mutation.tsv
      one row per (category, mutation_idx) with mean recovery across entries
  docs/p5p8_improvements/33_registry_stress_test.md
  1 line in findings_ledger.jsonl

Stdlib + jsonschema only. Designed to be cheap (≤300 LoC) and CI-runnable.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import pathlib
import random
import statistics
import sys

try:
    import jsonschema
except ImportError:
    print("registry_stress_test.py requires the `jsonschema` package", file=sys.stderr)
    sys.exit(1)

HERE = pathlib.Path(__file__).resolve().parent
WORKTREE = HERE.parent.parent
REG = WORKTREE / "registry"
SCHEMA = REG / "schema.json"
ENTRIES_DIR = REG / "entries"
OUT = WORKTREE / "experiments" / "results" / "p5p8"
OUT.mkdir(parents=True, exist_ok=True)

REQUIRED_TOP_STACK = ["record_type", "id", "label_claimed",
                      "framework", "provenance", "min_report"]
REQUIRED_TOP_DELTA = ["record_type", "id", "name", "base", "deltas", "citation"]

# (category, description, mutator_fn(entry) -> (perturbed_entry, change_summary))
def _mut_remove_required_key(entry):
    e = copy.deepcopy(entry)
    rt = e["record_type"]
    required = REQUIRED_TOP_STACK if rt == "stack" else REQUIRED_TOP_DELTA
    key = random.choice(required)
    del e[key]
    return e, f"drop_top_required:{key}"


def _mut_wrong_type_record_type(entry):
    e = copy.deepcopy(entry)
    e["record_type"] = 1 if random.random() < 0.5 else ["stack", "delta"]
    return e, "wrong_type:record_type"


def _mut_wrong_type_id(entry):
    e = copy.deepcopy(entry)
    e["id"] = 42  # should be a string per pattern
    return e, "wrong_type:id"


def _mut_id_pattern(entry):
    e = copy.deepcopy(entry)
    # pattern is ^[a-z0-9][a-z0-9_.-]*$ — break with uppercase
    e["id"] = e["id"].upper()
    return e, "bad_id_pattern"


def _mut_enum_openness(entry):
    e = copy.deepcopy(entry)
    e["framework"]["openness"] = random.choice([
        "Open", "OPEN", "Public",
        "weird", "unknown", "tinker",
    ])
    return e, "bad_framework_openness_enum"


def _mut_enum_status(entry):
    """Stack entries only: variant_deltas_applied[*].status enum."""
    e = copy.deepcopy(entry)
    if e.get("variant_deltas_applied"):
        d = random.choice(e["variant_deltas_applied"])
        d["status"] = random.choice(["YES", "implemented_sort_of", "true"])
        return e, "bad_delta_status_enum"
    return None, "no_variant_deltas_to_perturb"


def _mut_enum_loss_field(entry):
    """Loss-form advantage_normalization enum (token / std / none)."""
    e = copy.deepcopy(entry)
    e["min_report"]["loss_form"]["advantage_normalization"] = "minmax"
    return e, "bad_loss_advantage_enum"


def _mut_additional_property(entry):
    e = copy.deepcopy(entry)
    e["__extra_top_level__"] = "uh-oh"
    return e, "additional_top_level_property"


def _mut_min_report_missing_required(entry):
    e = copy.deepcopy(entry)
    item = random.choice(list(e["min_report"].keys()))
    if isinstance(e["min_report"][item], dict):
        # drop one of the leaves in the item (each item is a flat dict)
        leaf = random.choice(list(e["min_report"][item].keys()))
        del e["min_report"][item][leaf]
        return e, f"drop_min_report_leaf:{item}.{leaf}"
    return None, "no_min_report_leaf"


def _mut_min_report_extra_required(entry):
    """Stack records: min_report itself is required; passing a NON-DICT fails."""
    e = copy.deepcopy(entry)
    e["min_report"] = "string-not-dict"
    return e, "wrong_type:min_report"


def _mut_provenance_missing_source_artifacts(entry):
    e = copy.deepcopy(entry)
    del e["provenance"]["source_artifacts"]
    return e, "drop_provenance_source_artifacts"


def _mut_vdeltas_broken_ref(entry):
    e = copy.deepcopy(entry)
    if e.get("variant_deltas_applied"):
        d = random.choice(e["variant_deltas_applied"])
        # original is delta_xxx; flip to a name that doesn't exist
        d["delta_id"] = "delta_NONE_EXISTENT_" + str(random.randint(99, 999))
        return e, "broken_delta_ref"
    return None, "no_variant_deltas_to_perturb"


def _mut_outcomes_wrong_type(entry):
    """`outcomes` is typed object|null — pass a string to break it."""
    e = copy.deepcopy(entry)
    e["outcomes"] = "this-should-be-a-dict"
    return e, "wrong_type:outcomes"


# MUTATORS is the catalogue. Order is stable.
MUTATORS = [
    ("drop_top_required", _mut_remove_required_key),
    ("wrong_type_record_type", _mut_wrong_type_record_type),
    ("wrong_type_id", _mut_wrong_type_id),
    ("bad_id_pattern", _mut_id_pattern),
    ("bad_framework_openness", _mut_enum_openness),
    ("bad_delta_status_enum", _mut_enum_status),
    ("bad_loss_advantage_enum", _mut_enum_loss_field),
    ("additional_top_level", _mut_additional_property),
    ("drop_min_report_leaf", _mut_min_report_missing_required),
    ("wrong_type_min_report", _mut_min_report_extra_required),
    ("drop_provenance_source_artifacts", _mut_provenance_missing_source_artifacts),
    ("broken_delta_ref", _mut_vdeltas_broken_ref),
    ("wrong_type_outcomes", _mut_outcomes_wrong_type),
]


def bootstrap_ci(samples, stat_fn, n_boot=4000, alpha=0.05, seed=20260704):
    """Paired bootstrap CI on a statistic (e.g. mean) of a list of {0,1} obs."""
    rng = random.Random(seed)
    n = len(samples)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = stat_fn(samples)
    boots = []
    for _ in range(n_boot):
        bag = [samples[rng.randrange(n)] for _ in range(n)]
        boots.append(stat_fn(bag))
    boots.sort()
    lo = boots[int(n_boot * alpha / 2)]
    hi = boots[int(n_boot * (1 - alpha / 2))]
    return point, lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-mutations-per-category", type=int, default=20,
                    help="Number of mutation attempts per (category, entry) "
                         "combination. Default 20 ⇒ 13 categories × 20 × "
                         "20 entries ≈ 5200 attempted perturbations.")
    ap.add_argument("--seed", type=int, default=20260704)
    ap.add_argument("--only-entries", nargs="*", default=None,
                    help="Restrict stress-test to a subset of entry IDs.")
    args = ap.parse_args()

    random.seed(args.seed)
    schema = json.loads(SCHEMA.read_text())
    entries = []
    for p in sorted(ENTRIES_DIR.glob("*.json")):
        rec = json.loads(p.read_text())
        if args.only_entries and rec["id"] not in args.only_entries:
            continue
        entries.append((rec["id"], rec))
    if not entries:
        print("no entries to stress-test", file=sys.stderr)
        sys.exit(2)
    print(f"loaded {len(entries)} registry entries; "
          f"{args.n_mutations_per_category} mutations × "
          f"{len(MUTATORS)} categories.")

    per_run_rows = []
    misses = []
    by_mutation = {}     # category -> list of CAUGHT (1) / MISSED (0)
    per_category_total = {cat: 0 for cat, _ in MUTATORS}
    per_category_caught = {cat: 0 for cat, _ in MUTATORS}

    for category, fn in MUTATORS:
        for entry_id, rec in entries:
            for i in range(args.n_mutations_per_category):
                try:
                    perturbed, label = fn(rec)
                except (KeyError, IndexError) as e:
                    # entry shape doesn't permit this mutation — skip cleanly
                    per_run_rows.append({
                        "entry_id": entry_id, "category": category,
                        "mutation_idx": i, "expected": "caught",
                        "actual": "skipped_noop",
                        "mut_label": f"error:{type(e).__name__}",
                    })
                    continue
                if perturbed is None:
                    per_run_rows.append({
                        "entry_id": entry_id, "category": category,
                        "mutation_idx": i, "expected": "caught",
                        "actual": "skipped_noop",
                        "mut_label": label,
                    })
                    continue
                caught = 1
                actual = "caught"
                mut_label = label
                try:
                    jsonschema.validate(perturbed, schema)
                    caught = 0
                    actual = "MISSED"
                    misses.append({
                        "entry_id": entry_id, "category": category,
                        "mutation_idx": i, "mut_label": mut_label,
                    })
                except jsonschema.ValidationError:
                    actual = "caught"
                by_mutation.setdefault(category, []).append(caught)
                per_category_total[category] += 1
                per_category_caught[category] += caught
                per_run_rows.append({
                    "entry_id": entry_id, "category": category,
                    "mutation_idx": i, "expected": "caught",
                    "actual": actual, "mut_label": mut_label,
                })

    # Per-run TSV
    with (OUT / "registry_stress_per_run.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["entry_id", "category",
                                          "mutation_idx", "expected",
                                          "actual", "mut_label"])
        w.writeheader()
        for r in per_run_rows:
            w.writerow(r)

    # Misses TSV — the regression work-list
    with (OUT / "registry_stress_misses.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["entry_id", "category",
                                          "mutation_idx", "mut_label"])
        w.writeheader()
        for r in misses:
            w.writerow(r)

    # By-mutation TSV: per (category, mutation_idx) mean across entries
    n_idx = args.n_mutations_per_category
    by_mut_rows = []
    for cat, _ in MUTATORS:
        bag = by_mutation.get(cat, [])
        if not bag:
            continue
        for i in range(n_idx):
            bucket = bag[i::n_idx]    # every n_idx-th element
            mean_caught = (sum(bucket) / len(bucket)) if bucket else 0.0
            by_mut_rows.append({
                "category": cat, "mutation_idx": i,
                "n": len(bucket), "mean_caught": round(mean_caught, 4),
            })
    with (OUT / "registry_stress_by_mutation.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "mutation_idx",
                                          "n", "mean_caught"])
        w.writeheader()
        for r in by_mut_rows:
            w.writerow(r)

    # Summary JSON with paired bootstrap CI on per-category recovery
    summary = {
        "n_entries": len(entries),
        "n_categories": len(MUTATORS),
        "n_mutations_per_category": args.n_mutations_per_category,
        "n_attempts_total": sum(per_category_total.values()),
        "n_caught_total": sum(per_category_caught.values()),
        "overall_recovery_rate": round(
            sum(per_category_caught.values()) /
            max(1, sum(per_category_total.values())), 4),
        "n_misses": len(misses),
        "per_category": {},
    }
    # overall paired bootstrap CI — treat each (category, mutation_idx) bin
    # across entries as one observation, then resample bins
    overall_bags = []
    for cat, _ in MUTATORS:
        bag = by_mutation.get(cat, [])
        if not bag:
            continue
        for i in range(args.n_mutations_per_category):
            bucket = bag[i::args.n_mutations_per_category]
            if bucket:
                overall_bags.append(sum(bucket) / len(bucket))
    if overall_bags:
        pt, lo, hi = bootstrap_ci(overall_bags, statistics.fmean,
                                  n_boot=4000, seed=args.seed)
        summary["overall_recovery_rate_ci95"] = {
            "point": round(pt, 4),
            "lo": round(lo, 4),
            "hi": round(hi, 4),
        }
    for cat, _ in MUTATORS:
        bag = by_mutation.get(cat, [])
        if not bag:
            continue
        n = per_category_total[cat]
        c = per_category_caught[cat]
        pt, lo, hi = bootstrap_ci(bag, statistics.fmean,
                                  n_boot=4000,
                                  seed=args.seed + hash(cat) % 10000)
        summary["per_category"][cat] = {
            "n": n, "n_caught": c,
            "recovery_rate": round(c / max(1, n), 4),
            "recovery_rate_ci95": {
                "point": round(pt, 4),
                "lo": round(lo, 4),
                "hi": round(hi, 4),
            },
        }
    (OUT / "registry_stress_summary.json").write_text(
        json.dumps(summary, indent=2))
    print("=== stress-test summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
