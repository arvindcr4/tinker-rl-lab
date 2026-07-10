#!/usr/bin/env python3
"""Iter 202 P6 (Pillar 2) — cross-framework x method coverage matrix (post-bump).

Iter-198 lifted schema validation 34/46 -> 46/46 by closing 5 drift classes
(12 affected entries). With every entry now schema-valid, the natural next
audit is the *cartesian* coverage of (framework x method): which cells of
the registry are populated, which are empty, and where multiple frameworks
share the same claimed method (the cross-framework reproducibility surface).

Outputs (all under experiments/results/p5p8/):
  p6_iter202_framework_method_cell.tsv      — one row per (framework, method)
  p6_iter202_same_method_clusters.tsv      — clusters of entries sharing a method
  p6_iter202_minreport_divergence.tsv      — per-field divergence across clusters
  p6_iter202_unmined_cell_priority.tsv     — priority score per unpopulated cell
  p6_iter202_summary.json                  — aggregate rollup

Stdlib only. Run: python3 scripts/p5p8/p6_iter202_framework_method_coverage.py
"""
import csv
import json
import pathlib
from collections import defaultdict
from statistics import mean

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
REG = ROOT / "registry" / "entries"
OUT = ROOT / "experiments" / "results" / "p5p8"
OUT.mkdir(parents=True, exist_ok=True)

# Framework classifier — uses the stack record's framework.name prefix
FRAMEWORK_PREFIXES = [
    ("tinker", "tinker"), ("wandb", "wandb"), ("colab-open", "colab-open"),
    ("openrlhf", "openrlhf"), ("trl", "trl"), ("verl", "verl"), ("zvf130", "zvf130"),
]

MIN_REPORT_ITEMS = ["loss_form", "reference_kl", "sampler_backend",
                    "telemetry", "group_size_schedule", "heldout_split",
                    "decontamination"]


def classify_framework(entry_id, framework_name):
    """Classify a stack record into one of the 7 framework classes."""
    if framework_name:
        fn = framework_name.lower()
        for prefix, label in FRAMEWORK_PREFIXES:
            if prefix in fn:
                return label
    eid = entry_id.lower()
    for prefix, label in FRAMEWORK_PREFIXES:
        if eid.startswith(prefix):
            return label
    return "other"


def load_entries():
    """Load every schema-valid registry entry. Schema-validation happens via
    the iter-198 schema (registry/schema.json) which now accepts 46/46."""
    stacks, deltas = {}, {}
    for p in sorted(REG.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        rt = d.get("record_type")
        if rt == "stack":
            stacks[d["id"]] = d
        elif rt == "variant_delta":
            deltas[d["id"]] = d
    return stacks, deltas


def walk_leaves(d, prefix=""):
    """Yield (path, leaf) for every leaf of a nested dict."""
    if isinstance(d, dict):
        for k, v in d.items():
            yield from walk_leaves(v, prefix + k + ".")
    else:
        yield prefix.rstrip("."), d


def count_leaves(d):
    """Count leaves (filled + null)."""
    return sum(1 for _ in walk_leaves(d))


def count_filled(d):
    """Count leaves with non-null value."""
    return sum(1 for _, v in walk_leaves(d) if v is not None)


def min_report_badge(rec):
    """Per-entry MIN-REPORT badge (0-100, weighted by leaf count)."""
    total_leaves, total_filled = 0, 0
    per_item = {}
    for it in MIN_REPORT_ITEMS:
        sub = rec.get("min_report", {}).get(it, {})
        nl = count_leaves(sub)
        nf = count_filled(sub)
        total_leaves += nl
        total_filled += nf
        per_item[it] = (nf, nl, nf / nl if nl else 0.0)
    score = (total_filled / total_leaves * 100.0) if total_leaves else 0.0
    return round(score, 2), per_item, total_filled, total_leaves


def main():
    stacks, deltas = load_entries()

    # (framework, method) cells
    cells = defaultdict(list)        # (fw, method) -> [entry_id]
    all_methods = set()
    all_frameworks = set()
    for sid, s in stacks.items():
        fw = classify_framework(sid, s.get("framework", {}).get("name", ""))
        m = s.get("label_claimed", "?")
        cells[(fw, m)].append(sid)
        all_methods.add(m)
        all_frameworks.add(fw)

    # Per-cell TSV
    cell_rows = []
    for (fw, m), entries in sorted(cells.items()):
        badges = []
        for sid in entries:
            rec = stacks[sid]
            sc, _, _, _ = min_report_badge(rec)
            badges.append(sc)
        cell_rows.append({
            "framework": fw,
            "method": m,
            "n_entries": len(entries),
            "entries": ",".join(entries),
            "mean_badge": round(mean(badges), 2) if badges else 0.0,
            "min_badge": min(badges) if badges else 0.0,
            "max_badge": max(badges) if badges else 0.0,
            "status": "POPULATED",
        })

    # Cross-framework same-method clusters
    cluster_rows = []
    min_report_divergence_rows = []
    for m in sorted(all_methods):
        # gather all frameworks covering this method
        fw_for_method = sorted({fw for (fw, mm) in cells if mm == m for _ in cells[(fw, mm)]})
        if len(fw_for_method) < 2:
            continue
        all_entries_in_cluster = []
        for fw in fw_for_method:
            all_entries_in_cluster.extend([(sid, fw) for sid in cells[(fw, m)]])
        cluster_rows.append({
            "method": m,
            "n_frameworks": len(fw_for_method),
            "frameworks": ",".join(fw_for_method),
            "n_entries": len(all_entries_in_cluster),
            "entries": ",".join(f"{fw}:{sid}" for sid, fw in all_entries_in_cluster),
            "status": "CROSS-FRAMEWORK",
        })

        # Per-field divergence across the cluster — for every leaf field of
        # min_report, count distinct non-null values.
        leaf_values = defaultdict(lambda: defaultdict(list))  # field -> framework -> [values]
        for sid, fw in all_entries_in_cluster:
            rec = stacks[sid]
            for it in MIN_REPORT_ITEMS:
                for fpath, val in walk_leaves(rec.get("min_report", {}).get(it, {})):
                    if val is not None:
                        leaf_values[fpath][fw].append(val)

        for fpath in sorted(leaf_values):
            per_fw = leaf_values[fpath]
            n_fws_reporting = len(per_fw)
            distinct_values_per_fw = {fw: sorted(set(vs)) for fw, vs in per_fw.items()}
            n_distinct_total = len({v for vs in per_fw.values() for v in vs})
            cross_fw_disagree = 0
            fws = sorted(per_fw.keys())
            for i in range(len(fws)):
                for j in range(i + 1, len(fws)):
                    if set(per_fw[fws[i]]) != set(per_fw[fws[j]]):
                        cross_fw_disagree += 1
            n_fw_pairs = len(fws) * (len(fws) - 1) // 2
            disagree_rate = (cross_fw_disagree / n_fw_pairs) if n_fw_pairs else 0.0
            min_report_divergence_rows.append({
                "method": m,
                "field": fpath,
                "n_fws_reporting": n_fws_reporting,
                "n_distinct_values": n_distinct_total,
                "cross_fw_pairs": n_fw_pairs,
                "cross_fw_disagree_pairs": cross_fw_disagree,
                "disagree_rate": round(disagree_rate, 4),
                "per_fw_values": json.dumps(distinct_values_per_fw, sort_keys=True),
            })

    # Unmined-cell priority: score each (framework, method) NOT in cells by
    # how transferable the harness + trace are. A cell is "easy" if the
    # method is covered elsewhere AND the framework already covers >=2 methods.
    populated_by_method = defaultdict(set)
    populated_by_framework = defaultdict(set)
    for (fw, m), _ in cells.items():
        populated_by_method[m].add(fw)
        populated_by_framework[fw].add(m)

    unmined_rows = []
    for fw in sorted(all_frameworks):
        for m in sorted(all_methods):
            if (fw, m) in cells:
                continue
            other_fws = populated_by_method[m]
            other_methods = populated_by_framework[fw]
            method_xfer = len(other_fws) > 0
            fw_xfer = len(other_methods) >= 2
            priority = (2 * method_xfer) + (3 * fw_xfer) + len(other_methods) + len(other_fws)
            unmined_rows.append({
                "framework": fw, "method": m,
                "method_transferable_from": ",".join(sorted(other_fws)),
                "n_other_methods_for_framework": len(other_methods),
                "framework_has_2plus_methods": fw_xfer,
                "method_already_covered": method_xfer,
                "priority_score": priority, "status": "UNMINED",
            })
    unmined_rows.sort(key=lambda r: (-r["priority_score"], r["method"], r["framework"]))

    # Write TSVs
    def write_tsv(path, rows, fieldnames):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_tsv(OUT / "p6_iter202_framework_method_cell.tsv", cell_rows,
              ["framework", "method", "n_entries", "entries", "mean_badge",
               "min_badge", "max_badge", "status"])
    write_tsv(OUT / "p6_iter202_same_method_clusters.tsv", cluster_rows,
              ["method", "n_frameworks", "frameworks", "n_entries", "entries", "status"])
    write_tsv(OUT / "p6_iter202_minreport_divergence.tsv", min_report_divergence_rows,
              ["method", "field", "n_fws_reporting", "n_distinct_values",
               "cross_fw_pairs", "cross_fw_disagree_pairs", "disagree_rate", "per_fw_values"])
    write_tsv(OUT / "p6_iter202_unmined_cell_priority.tsv", unmined_rows,
              ["framework", "method", "method_transferable_from",
               "n_other_methods_for_framework", "framework_has_2plus_methods",
               "method_already_covered", "priority_score", "status"])

    # Schema-bump cross-check: count entries present in registry dir vs those
    # that load successfully (the iter-198 bump lifted 34->46; verify).
    n_files = sum(1 for _ in REG.glob("*.json"))
    n_loaded_stacks = len(stacks)
    n_loaded_deltas = len(deltas)

    # Summary JSON
    summary = {
        "iter": 202,
        "pillar": "P6",
        "vein": "(b) cross-framework x method coverage matrix (post-iter-198 bump)",
        "n_registry_files": n_files,
        "n_loaded_stacks": n_loaded_stacks,
        "n_loaded_deltas": n_loaded_deltas,
        "n_frameworks": len(all_frameworks),
        "frameworks": sorted(all_frameworks),
        "n_methods": len(all_methods),
        "methods": sorted(all_methods),
        "n_populated_cells": len(cell_rows),
        "n_unmined_cells": len(unmined_rows),
        "n_cross_framework_clusters": len(cluster_rows),
        "cross_framework_methods": [r["method"] for r in cluster_rows],
        "framework_entry_counts": {
            fw: sum(1 for sid, s in stacks.items()
                    if classify_framework(sid, s.get("framework", {}).get("name", "")) == fw)
            for fw in sorted(all_frameworks)
        },
        "method_entry_counts": {
            m: sum(1 for sid, s in stacks.items() if s.get("label_claimed") == m)
            for m in sorted(all_methods)
        },
        "cartesian_density_pct": round(100 * len(cell_rows) / (len(all_frameworks) * len(all_methods)), 2),
        "schema_bump_alignment": {
            "iter198_total_valid": 46,
            "iter202_files": n_files,
            "loaded_minus_bump": n_loaded_stacks - 46,
            "note": "delta_*.json are variant-delta records (not stack records); iter-198's 46/46 covers both record types"
        },
    }
    (OUT / "p6_iter202_summary.json").write_text(json.dumps(summary, indent=2))

    # Print a short header
    print(f"Iter 202 — P6 cross-framework x method coverage matrix (post-iter-198 bump)")
    n_total = len(all_frameworks) * len(all_methods)
    print(f"  Files: {n_files}; loaded stacks={n_loaded_stacks}, deltas={n_loaded_deltas}")
    print(f"  Frameworks: {len(all_frameworks)}; methods: {len(all_methods)}")
    print(f"  Populated cells: {len(cell_rows)} / {n_total} ({summary['cartesian_density_pct']}%)")
    print(f"  Cross-framework clusters (>=2 frameworks share method): {len(cluster_rows)}"
          f" -> {[r['method'] for r in cluster_rows]}")
    print(f"  Unmined cells: {len(unmined_rows)}")
    print(f"  Top-3 unmined priority cells:")
    for r in unmined_rows[:3]:
        print(f"    [{r['priority_score']}] {r['framework']} x {r['method']}"
              f" (transferable from {r['method_transferable_from']}, fw has {r['n_other_methods_for_framework']} other methods)")
    print(f"  Output: {OUT}")


if __name__ == "__main__":
    main()