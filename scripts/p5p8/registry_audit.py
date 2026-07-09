#!/usr/bin/env python3
"""GRPO-Registry CI-style auditor.

Performs a complete health check on registry/entries/*.json:
  1. Schema validation (jsonschema Draft 2020-12).
  2. Per-leaf MIN-REPORT coverage (full / partial / unreported) for stacks.
  3. Cross-reference: every variant_deltas_applied[*].delta_id must point to a
     real registry/entries/delta_*.json record.
  4. Per-framework aggregate: by framework name and by openness tier
     (open / managed / closed).
  5. Per-variant-delta aggregate: how many stacks claim each named delta and
     in what status mix.
  6. Exports:
       experiments/results/p5p8/registry_audit.tsv         (one row per entry × item)
       experiments/results/p5p8/registry_audit_summary.json (machine-readable rollup)
       experiments/results/p5p8/figures/registry_coverage.{png,pdf}

Stdlib + jsonschema + matplotlib. Designed to be cheap and to fit in CI.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter, defaultdict

try:
    import jsonschema
except ImportError as exc:  # pragma: no cover
    print("registry_audit.py requires the `jsonschema` package", file=sys.stderr)
    raise

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = pathlib.Path(__file__).resolve().parent
REG = HERE.parent.parent / "registry"
SCHEMA = REG / "schema.json"
ENTRIES = REG / "entries"
OUT = HERE.parent.parent / "experiments" / "results" / "p5p8"
FIGS = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

ITEMS = ["loss_form", "reference_kl", "sampler_backend", "telemetry",
         "group_size_schedule", "heldout_split", "decontamination"]


def load_all():
    recs = {"stack": {}, "variant_delta": {}}
    for p in sorted(ENTRIES.glob("*.json")):
        r = json.loads(p.read_text())
        recs[r["record_type"]][r["id"]] = r
    return recs


def leaf_coverage(d):
    """Returns (filled, total) over leaves of a one-level dict.

    Each MIN-REPORT item is a flat dict in this registry (no nesting below the
    item level). Following the registry convention: null = UNREPORTED;
    explicit false / 0.0 / "none" = reported-as-absent (counts as filled
    because the auditor scores honesty over configuration virtue). See
    `registry/query.py:item_score` for the equivalent in the shipped CLI.
    """
    if isinstance(d, dict):
        leaves = list(d.values())
        if not leaves:
            return 0, 0
        filled = sum(1 for v in leaves if v is not None)
        return filled, len(leaves)
    # leaf (shouldn't occur in this registry)
    return (1 if d is not None else 0), 1


def badge_items(rec):
    out = {}
    for it in ITEMS:
        f, t = leaf_coverage(rec["min_report"][it])
        out[it] = (f / t) if t else 0.0
    return out


def variant_deltas_seen(rec):
    return rec.get("variant_deltas_applied", []) or []


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    schema = json.loads(SCHEMA.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    recs = load_all()
    delta_ids = set(recs["variant_delta"].keys())

    # Schema validation pass
    schema_results = {}
    for kind in ("stack", "variant_delta"):
        for rid, rec in recs[kind].items():
            errors = sorted(validator.iter_errors(rec), key=lambda e: e.path)
            schema_results[rid] = {
                "record_type": kind,
                "schema_ok": len(errors) == 0,
                "n_errors": len(errors),
                "errors": [f"{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in errors[:5]],
            }

    # Per-entry per-item coverage (stacks only)
    coverage_rows = []
    per_item_totals = defaultdict(lambda: [0, 0])  # filled, total
    framework_buckets = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    openness_buckets = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    badge_scores = {}
    for rid, rec in recs["stack"].items():
        scores = badge_items(rec)
        badge = round(100 * sum(scores.values()) / len(ITEMS))
        badge_scores[rid] = badge
        fw = rec["framework"]["name"]
        op = rec["framework"]["openness"]
        for it, s in scores.items():
            f, t = leaf_coverage(rec["min_report"][it])
            per_item_totals[it][0] += f
            per_item_totals[it][1] += t
            framework_buckets[fw][it][0] += f
            framework_buckets[fw][it][1] += t
            openness_buckets[op][it][0] += f
            openness_buckets[op][it][1] += t
            coverage_rows.append({
                "entry_id": rid,
                "framework": fw,
                "openness": op,
                "label_claimed": rec["label_claimed"],
                "item": it,
                "frac_reported": round(s, 3),
                "schema_ok": int(schema_results[rid]["schema_ok"]),
                "badge": badge,
            })

    # Variant-delta cross-reference
    delta_xref = defaultdict(list)  # delta_id -> list of (entry_id, status)
    broken_refs = []  # (entry_id, missing_delta_id)
    for rid, rec in recs["stack"].items():
        for d in variant_deltas_seen(rec):
            if d["delta_id"] not in delta_ids:
                broken_refs.append({"entry_id": rid, "delta_id": d["delta_id"]})
            delta_xref[d["delta_id"]].append({"entry_id": rid, "status": d["status"]})

    # Status mix per delta
    delta_status_mix = {}
    for did, refs in delta_xref.items():
        mix = Counter(r["status"] for r in refs)
        delta_status_mix[did] = {
            "n_claimants": len(refs),
            "status_mix": dict(mix),
            "unclaimed": did not in {r["id"] for r in recs["variant_delta"].values()},
        }

    # ---- Write TSV (one row per (entry, item))
    tsv_path = pathlib.Path(args.out_dir) / "registry_audit.tsv"
    with tsv_path.open("w") as f:
        cols = ["entry_id", "framework", "openness", "label_claimed",
                "item", "frac_reported", "schema_ok", "badge"]
        f.write("\t".join(cols) + "\n")
        for row in coverage_rows:
            f.write("\t".join(str(row[c]) for c in cols) + "\n")

    # ---- Write JSON summary
    summary = {
        "schema_validation": {
            "n_total": sum(len(recs[k]) for k in recs),
            "n_ok": sum(1 for r in schema_results.values() if r["schema_ok"]),
            "n_fail": sum(1 for r in schema_results.values() if not r["schema_ok"]),
            "failures": [{"entry_id": k, **{kk: vv for kk, vv in v.items() if kk != "record_type"}}
                         for k, v in schema_results.items() if not v["schema_ok"]],
        },
        "min_report_coverage": {
            "per_item": {
                it: {
                    "filled_leaves": per_item_totals[it][0],
                    "total_leaves": per_item_totals[it][1],
                    "frac": round(per_item_totals[it][0] / per_item_totals[it][1], 4) if per_item_totals[it][1] else 0.0,
                }
                for it in ITEMS
            },
            "per_framework": {
                fw: {
                    it: {
                        "filled_leaves": framework_buckets[fw][it][0],
                        "total_leaves": framework_buckets[fw][it][1],
                        "frac": round(framework_buckets[fw][it][0] / framework_buckets[fw][it][1], 4) if framework_buckets[fw][it][1] else 0.0,
                    }
                    for it in ITEMS
}
                for fw in framework_buckets
            },
            "per_openness": {
                op: {
                    it: {
                        "filled_leaves": openness_buckets[op][it][0],
                        "total_leaves": openness_buckets[op][it][1],
                        "frac": round(openness_buckets[op][it][0] / openness_buckets[op][it][1], 4) if openness_buckets[op][it][1] else 0.0,
                    }
                    for it in ITEMS
                }
                for op in openness_buckets
            },
            "per_entry": {
                rid: {
                    "frac_per_item": {it: round(s, 4) for it, s in badge_items(rec).items()},
                    "badge": badge_scores[rid],
                    "framework": rec["framework"]["name"],
                    "openness": rec["framework"]["openness"],
                    "label_claimed": rec["label_claimed"],
                }
                for rid, rec in recs["stack"].items()
            },
        },
        "variant_delta_xref": {
            "n_deltas": len(delta_ids),
            "n_claimed": len(delta_xref),
            "n_broken_refs": len(broken_refs),
            "broken_refs": broken_refs,
            "per_delta_status_mix": delta_status_mix,
        },
        "counts": {
            "n_stack_records": len(recs["stack"]),
            "n_variant_delta_records": len(recs["variant_delta"]),
        },
    }
    json_path = pathlib.Path(args.out_dir) / "registry_audit_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    # ---- Figure: per-item coverage, all entries
    items = ITEMS
    entry_ids = [r["id"] for r in recs["stack"].values()]
    matrix = []
    for rid in entry_ids:
        row = []
        for it in items:
            f, t = leaf_coverage(recs["stack"][rid]["min_report"][it])
            row.append(f / t if t else 0.0)
        matrix.append(row)
    matrix = list(zip(*matrix))  # transpose -> items × entries

    fig, ax = plt.subplots(figsize=(11.0, 4.0))
    cmap = plt.get_cmap("viridis")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(entry_ids)))
    ax.set_xticklabels([eid.replace("_qwen3.5-4b_gsm8k", "").replace("_qwen3-8b_gsm8k", "")
                        .replace("_e3", "").replace("zvf130_", "") for eid in entry_ids],
                       rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(items)
    ax.set_title("GRPO-Registry: per-entry MIN-REPORT coverage (fraction of leaves reported)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("fraction of leaves reported")
    fig.tight_layout()
    fig.savefig(FIGS / "registry_coverage.png", dpi=160)
    fig.savefig(FIGS / "registry_coverage.pdf")
    plt.close(fig)

    # ---- Console summary
    if not args.quiet:
        print(f"[registry-audit] schema: {summary['schema_validation']['n_ok']}/"
              f"{summary['schema_validation']['n_total']} PASS")
        if summary["schema_validation"]["n_fail"]:
            print("  FAIL:", summary["schema_validation"]["failures"])
        print(f"[registry-audit] variant-delta cross-ref: "
              f"{len(delta_xref)}/{len(delta_ids)} deltas claimed; "
              f"{len(broken_refs)} broken refs")
        print("[registry-audit] per-item coverage (overall):")
        for it in ITEMS:
            d = summary["min_report_coverage"]["per_item"][it]
            print(f"  {it:24s} {d['filled_leaves']:>3d}/{d['total_leaves']:<3d}  "
                  f"{100 * d['frac']:>5.1f}%")
        print(f"[registry-audit] wrote {tsv_path}")
        print(f"[registry-audit] wrote {json_path}")
        print(f"[registry-audit] wrote {FIGS / 'registry_coverage.png'}")


if __name__ == "__main__":
    main()