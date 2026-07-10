#!/usr/bin/env python3
"""P6 iter-98 per-component variant_deltas_applied (VDA) validator.

Vein: hybrid of iter-94 H3 follow-up (regenerate measured_block_audit.json
to close the delta_drgrpo stale-audit drift) and the per-component VDA
audit (mint vein "how much of the registry is actually verified vs
labelled").

Stdlib only. Outputs (5 files in experiments/results/p5p8/):

- p6_iter98_vda_per_stack.tsv     : per-stack record VDA summary (20 rows)
- p6_iter98_vda_per_component.tsv : per-(delta,component) coverage table
- p6_iter98_vda_status_dist.tsv   : corpus-wide VDA status distribution
- p6_iter98_vda_summary.json      : JSON summary + crosscheck
- registry/measured_block_audit_refresh.json (when --write-refresh)
"""
import argparse
import json
import pathlib
from collections import defaultdict, Counter

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REG = ROOT / "registry"
ENT = REG / "entries"
OUT = ROOT / "experiments" / "results" / "p5p8"

VDA_STATUSES = ("implemented", "surrogate", "absent", "unknown",
                "managed_by_tinker")
REPRODUCIBLE_STATUSES = ("implemented", "surrogate")


def _load_entry(p):
    try:
        return json.loads(p.read_text()), None
    except Exception as e:
        return None, f"json-parse-error:{e}"


def _flatten_measured(measured):
    panels, metrics = set(), set()
    for row in measured or []:
        if not isinstance(row, dict):
            continue
        p, m = row.get("panel"), row.get("metric")
        if p:
            panels.add(p)
        if m:
            metrics.add(m)
    return sorted(panels), sorted(metrics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--write-refresh", action="store_true",
                    help="write registry/measured_block_audit_refresh.json")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    # ---- load all entries ----
    entries = {}
    for p in sorted(ENT.glob("*.json")):
        rec, err = _load_entry(p)
        if err is None:
            entries[rec.get("id", p.stem)] = rec

    stack_records = [r for r in entries.values()
                     if r.get("record_type") == "stack"]
    delta_records = [r for r in entries.values()
                     if r.get("record_type") == "variant_delta"]
    print(f"[p6-iter98] loaded {len(entries)} entries "
          f"({len(stack_records)} stack + {len(delta_records)} variant_delta)")

    # =================================================================
    # JOB A — per-stack VDA audit
    # =================================================================
    status_keys = ("implemented", "surrogate", "absent", "unknown",
                   "managed_by_tinker")
    per_stack_rows = []
    for srec in stack_records:
        sc = Counter(vd.get("status", "unknown")
                     for vd in (srec.get("variant_deltas_applied") or []))
        n_total = sum(sc.values())
        per_stack_rows.append({
            "stack_id": srec["id"],
            "framework": (srec.get("framework") or {}).get("name", "?"),
            "openness": (srec.get("framework") or {}).get("openness", "?"),
            "label_claimed": srec.get("label_claimed", "?"),
            "n_components_declared": n_total,
            **{f"n_{k}": sc[k] for k in status_keys},
            "n_reproducible": sc["implemented"] + sc["surrogate"],
            "reproducible_fraction": round(
                (sc["implemented"] + sc["surrogate"]) / n_total
                if n_total else 0.0, 4),
        })

    pstk_tsv = OUT / "p6_iter98_vda_per_stack.tsv"
    cols = ["stack_id", "framework", "openness", "label_claimed",
            "n_components_declared", *status_keys,
            "n_reproducible", "reproducible_fraction"]
    with pstk_tsv.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in sorted(per_stack_rows,
                        key=lambda x: (-x["n_components_declared"],
                                       x["stack_id"])):
            f.write("\t".join([
                r["stack_id"], r["framework"], r["openness"],
                r["label_claimed"], str(r["n_components_declared"]),
                *(str(r[f"n_{k}"]) for k in status_keys),
                str(r["n_reproducible"]),
                f"{r['reproducible_fraction']:.4f}",
            ]) + "\n")
    print(f"[p6-iter98] JOB A wrote {pstk_tsv.name} "
          f"({len(per_stack_rows)} rows)")

    # JOB B — per-(delta_id, component) coverage table
    by_cell = defaultdict(Counter)
    cell_examples = defaultdict(list)
    for srec in stack_records:
        for vd in (srec.get("variant_deltas_applied") or []):
            did = vd.get("delta_id", "?")
            comp = vd.get("component", "?")
            st = vd.get("status", "unknown")
            by_cell[(did, comp)][st] += 1
            if len(cell_examples[(did, comp)]) < 3:
                cell_examples[(did, comp)].append(f"{srec['id']}({st})")

    pcomp_tsv = OUT / "p6_iter98_vda_per_component.tsv"
    pcols = ["delta_id", "component", "n_stacks_declaring", *status_keys,
             "reproducible_fraction", "example_stacks"]
    cell_rows = []
    for (did, comp), cnts in by_cell.items():
        n_total = sum(cnts.values())
        cell_rows.append({
            "delta_id": did, "component": comp,
            "n_stacks_declaring": n_total,
            **{k: cnts[k] for k in status_keys},
            "reproducible_fraction": round(
                (cnts["implemented"] + cnts["surrogate"]) / n_total, 4),
            "example_stacks": "|".join(cell_examples[(did, comp)][:3]),
        })
    with pcomp_tsv.open("w") as f:
        f.write("\t".join(pcols) + "\n")
        for r in sorted(cell_rows,
                        key=lambda x: (-x["n_stacks_declaring"],
                                       x["delta_id"], x["component"])):
            f.write("\t".join([
                r["delta_id"], r["component"],
                str(r["n_stacks_declaring"]),
                *(str(r[k]) for k in status_keys),
                f"{r['reproducible_fraction']:.4f}",
                r["example_stacks"],
            ]) + "\n")
    print(f"[p6-iter98] JOB B wrote {pcomp_tsv.name} "
          f"({len(cell_rows)} unique (delta,component) cells)")

    # =================================================================
    # JOB C — refresh measured_block_audit.json from live entries
    # =================================================================
    refreshed = {"n_entries": 0, "per_entry": [], "audit_date": "2026-07-05",
                 "audit_source": "platform_modal/scripts/p5p8/p6_iter98_vda_validator.py"}
    for drec in delta_records:
        measured = drec.get("measured") or []
        msig = [m for m in measured if isinstance(m, dict)]
        n_sig = sum(1 for m in msig if m.get("significant"))
        sup = sum(1 for m in msig if m.get("delta", 0) > 0 and m.get("significant"))
        con = sum(1 for m in msig if m.get("delta", 0) < 0 and m.get("significant"))
        n_src_ok = sum(1 for m in msig if m.get("source"))
        ns = [m.get("n", 0) for m in msig if "n" in m]
        panels, metrics = _flatten_measured(measured)
        refreshed["per_entry"].append({
            "delta_id": drec["id"], "name": drec.get("name", drec["id"]),
            "measured_count": len(measured), "validated_count": n_sig,
            "n_significant": n_sig,
            "pct_significant": round(100.0 * n_sig / max(1, len(measured)), 1),
            "min_n": min(ns) if ns else 0, "max_n": max(ns) if ns else 0,
            "supports": sup, "contradicts": con,
            "neutral": max(0, n_sig - sup - con),
            "unclaimed": max(0, len(measured) - n_sig),
            "n_src_resolved_ok": n_src_ok,
            "n_src_missing": len(measured) - n_src_ok,
            "panels": "|".join(panels), "metrics": "|".join(metrics),
            "citation_arxiv": (drec.get("citation") or {}).get("arxiv", ""),
        })
        refreshed["n_entries"] += 1

    # Cross-check vs the stale audit
    stale_path = REG / "measured_block_audit.json"
    crosscheck = {"stale_audit_path": str(stale_path.relative_to(ROOT)),
                  "refresh_audit_path": "registry/measured_block_audit_refresh.json",
                  "discrepancies": []}
    if stale_path.exists():
        stale = json.loads(stale_path.read_text())
        stale_by_id = {e["delta_id"]: e for e in stale.get("per_entry", [])}
        for r in refreshed["per_entry"]:
            sid = r["delta_id"]
            if sid in stale_by_id:
                smc = stale_by_id[sid].get("measured_count", 0)
                rmc = r["measured_count"]
                if smc != rmc:
                    crosscheck["discrepancies"].append({
                        "entry": sid,
                        "stale_audit_measured_count": smc,
                        "refreshed_measured_count": rmc,
                        "stale_audit_panels": stale_by_id[sid].get("panels", ""),
                        "refreshed_panels": r["panels"],
                        "interpretation": ("audit is STALE; entry has new measured rows"
                                           if rmc > smc else
                                           "entry has fewer rows than audit claims"),
                    })
    refresh_path = REG / "measured_block_audit_refresh.json"
    if args.write_refresh:
        refresh_path.write_text(json.dumps(refreshed, indent=2))
        crosscheck["refresh_audit_path"] = str(refresh_path.relative_to(ROOT))
        print(f"[p6-iter98] JOB C wrote {refresh_path.relative_to(ROOT)} "
              f"({refreshed['n_entries']} entries)")
    else:
        print(f"[p6-iter98] JOB C dry-run; pass --write-refresh to "
              f"write {refresh_path.relative_to(ROOT)}")
    # JOB D — corpus-wide status distribution table
    overall_status = Counter()
    overall_by_framework = defaultdict(Counter)
    overall_by_openness = defaultdict(Counter)
    overall_by_delta = defaultdict(Counter)
    for srec in stack_records:
        fw = (srec.get("framework") or {}).get("name", "?")
        op = (srec.get("framework") or {}).get("openness", "?")
        for vd in (srec.get("variant_deltas_applied") or []):
            st = vd.get("status", "unknown")
            did = vd.get("delta_id", "?")
            overall_status[st] += 1
            overall_by_framework[fw][st] += 1
            overall_by_openness[op][st] += 1
            overall_by_delta[did][st] += 1

    sd_tsv = OUT / "p6_iter98_vda_status_dist.tsv"
    cols = ["scope", "key", "n_total", *status_keys, "reproducible_fraction"]

    def _row(scope, key, cnts):
        n = sum(cnts.values()) or 1
        return [scope, key, str(sum(cnts.values())),
                *(str(cnts.get(k, 0)) for k in status_keys),
                f"{(cnts['implemented'] + cnts['surrogate']) / n:.4f}"]

    with sd_tsv.open("w") as f:
        f.write("\t".join(cols) + "\n")
        f.write("\t".join(_row("corpus", "ALL", overall_status)) + "\n")
        for scope_name, group in (("framework", overall_by_framework),
                                  ("openness", overall_by_openness),
                                  ("delta_id", overall_by_delta)):
            for key in sorted(group):
                f.write("\t".join(_row(scope_name, key, group[key])) + "\n")
    print(f"[p6-iter98] JOB D wrote {sd_tsv.name}")

    # Headline summary
    n_total_components = sum(overall_status.values())
    n_reproducible = overall_status["implemented"] + overall_status["surrogate"]
    n_unverified = (overall_status["unknown"]
                    + overall_status["managed_by_tinker"])
    n_declared_unable = overall_status["absent"]
    summary = {
        "n_stack_records": len(stack_records),
        "n_delta_records": len(delta_records),
        "n_total_vda_components_declared": n_total_components,
        **{k: overall_status[k] for k in status_keys},
        "reproducible_fraction": round(
            n_reproducible / max(1, n_total_components), 4),
        "unverified_fraction": round(
            n_unverified / max(1, n_total_components), 4),
        "explicit_absent_fraction": round(
            n_declared_unable / max(1, n_total_components), 4),
        "n_stale_audit_discrepancies": len(crosscheck["discrepancies"]),
        "audit_refresh_path": str(refresh_path.relative_to(ROOT)),
        "stale_audit_path": str(stale_path.relative_to(ROOT)),
    }
    summary_path = OUT / "p6_iter98_vda_summary.json"
    summary_path.write_text(json.dumps(
        {"summary": summary, "crosscheck": crosscheck,
         "overall_status": dict(overall_status)}, indent=2))

    print()
    print(f"[p6-iter98] HEADLINE: {n_total_components} VDA components "
          f"declared across {len(stack_records)} stack records")
    print(f"[p6-iter98]   reproducible (implemented+surrogate)="
          f"{n_reproducible} ({n_reproducible/n_total_components:.1%})")
    print(f"[p6-iter98]   unverified (unknown+managed_by_tinker)="
          f"{n_unverified} ({n_unverified/n_total_components:.1%})")
    print(f"[p6-iter98]explicit-absent (closed sampler can't run it)="
          f"{n_declared_unable} ({n_declared_unable/n_total_components:.1%})")
    print(f"[p6-iter98]   stale-audit discrepancies="
          f"{len(crosscheck['discrepancies'])}")
    for d in crosscheck["discrepancies"]:
        print(f"[p6-iter98]     - {d['entry']}: "
              f"stale={d['stale_audit_measured_count']} "
              f"refresh={d['refreshed_measured_count']}")
    print(f"[p6-iter98] wrote {pstk_tsv.name}, {pcomp_tsv.name}, "
          f"{sd_tsv.name}, {summary_path.name}")


if __name__ == "__main__":
    main()