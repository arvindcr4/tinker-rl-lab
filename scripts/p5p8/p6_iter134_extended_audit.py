#!/usr/bin/env python3
"""P6 iter-134 — extended audit: CI-method diversity + cross-panel companion.

Builds on iter-134 base audit (per-row field completeness) with:
  (i) CI-method diversity table — for each (ci_method.method, seed) tuple
      count rows, list entries.
  (ii) Seed-consistency audit — find rows with seed=None or seed!=20260705.
  (iii) Cross-panel companion TSV — for each (variant, metric) that appears in
        the iter-110 xpanel_verdict.tsv, emit one audit row keyed by delta_id.
  (iv) Empty-measured action gap — for each variant_delta entry with 0 measured
       rows, summarize expected_effects presence and blocking note.

Outputs:
  experiments/results/p5p8/p6_iter134_ci_method_diversity.tsv
  experiments/results/p5p8/p6_iter134_seed_inconsistency.tsv
  experiments/results/p5p8/p6_iter134_cross_panel_companion.tsv
  experiments/results/p5p8/p6_iter134_empty_action_gap.tsv
  experiments/results/p5p8/p6_iter134_summary.json
"""
import csv
import json
import pathlib
from collections import defaultdict, Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
REG = ROOT / "registry/entries"
OUT = ROOT / "experiments/results/p5p8"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    # Load all variant_delta entries
    vd = []
    for p in sorted(REG.glob("*.json")):
        d = json.loads(p.read_text())
        if d.get("record_type") != "variant_delta":
            continue
        d["_path"] = p
        d["_name"] = p.name
        vd.append(d)

    # (i) CI-method diversity
    ci_rows = []
    seed_incons = []
    shape_violations = []
    for e in vd:
        for i, m in enumerate(e.get("measured") or []):
            cm = m.get("ci_method")
            if cm is None:
                ci_rows.append({"id": e["id"], "row_idx": i, "ci_method": "null",
                                "n_boot": "", "seed": "", "ci_level": "",
                                "ci_shape": "null"})
                continue
            if not isinstance(cm, dict):
                # String-typed ci_method — schema violation ($defs/ci_method requires object|null)
                ci_rows.append({"id": e["id"], "row_idx": i,
                                "ci_method": str(cm),
                                "n_boot": "", "seed": "", "ci_level": "",
                                "ci_shape": "string_violation"})
                shape_violations.append({
                    "id": e["id"],
                    "row_idx": i,
                    "metric": m.get("metric", ""),
                    "panel": m.get("panel", ""),
                    "ci_method_string": str(cm),
                })
                continue
            ci_rows.append({
                "id": e["id"],
                "row_idx": i,
                "ci_method": cm.get("method", "none"),
                "n_boot": cm.get("n_boot", ""),
                "seed": cm.get("seed", ""),
                "ci_level": cm.get("ci_level", ""),
                "metric": m.get("metric", ""),
                "panel": m.get("panel", ""),
                "ci_shape": "object",
            })
            if cm.get("seed") in (None, "") or cm.get("seed") != 20260705:
                seed_incons.append({
                    "id": e["id"],
                    "row_idx": i,
                    "metric": m.get("metric", ""),
                    "panel": m.get("panel", ""),
                    "ci_method": cm.get("method", ""),
                    "seed": cm.get("seed", ""),
                    "note": "missing_seed" if cm.get("seed") in (None, "") else "non_canonical_seed",
                })

    # Aggregate ci_method diversity
    ci_div = Counter(r["ci_method"] for r in ci_rows)
    with (OUT / "p6_iter134_ci_method_diversity.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ci_method", "n_rows", "entries"])
        for cm, n in ci_div.most_common():
            ids = sorted(set(r["id"] for r in ci_rows if r["ci_method"] == cm))
            w.writerow([cm, n, ";".join(ids)])

    # Shape violation companion TSV
    with (OUT / "p6_iter134_ci_shape_violations.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "row_idx", "metric", "panel", "ci_method_string"])
        for r in shape_violations:
            w.writerow([r["id"], r["row_idx"], r["metric"], r["panel"], r["ci_method_string"]])

    with (OUT / "p6_iter134_seed_inconsistency.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "row_idx", "metric", "panel", "ci_method", "seed", "note"])
        for r in seed_incons:
            w.writerow([r["id"], r["row_idx"], r["metric"], r["panel"],
                        r["ci_method"], r["seed"], r["note"]])

    # (iii) Cross-panel companion — read iter-110 verdict + map to delta_id
    xp_v = OUT / "p6_iter110_xpanel_verdict.tsv"
    xp_rows = []
    if xp_v.exists():
        with xp_v.open() as f:
            rdr = csv.DictReader(f, delimiter="\t")
            for r in rdr:
                xp_rows.append(r)
    # Filter to (variant, panel=N2) so we get one row per (variant, metric_pair)
    cp_companion = []
    for r in xp_rows:
        v = r["variant"]
        mp = r["metric_pair"]
        if "zvf (N2)" not in mp:
            continue
        cp_companion.append({
            "id": f"delta_{v}",
            "variant": v,
            "metric_pair": mp,
            "verdict": r["verdict"],
            "n2_point": r["n2_point"],
            "n2_ci_lo": r["n2_ci_lo"],
            "n2_ci_hi": r["n2_ci_hi"],
            "n2_sig": r["n2_sig"],
            "zv_point": r["zv_point"],
            "zv_ci_lo": r["zv_ci_lo"],
            "zv_ci_hi": r["zv_ci_hi"],
            "zv_sig": r["zv_sig"],
        })
    with (OUT / "p6_iter134_cross_panel_companion.tsv").open("w", newline="") as f:
        if cp_companion:
            w = csv.DictWriter(f, fieldnames=list(cp_companion[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(cp_companion)
        else:
            f.write("id\tvariant\tmetric_pair\tverdict\n")

    # (iv) Empty-measured action gap
    empty = []
    for e in vd:
        if (e.get("measured") or []):
            continue
        ee = e.get("expected_effects") or []
        cit = e.get("citation", {})
        empty.append({
            "id": e["id"],
            "citation_bibkey": cit.get("bibkey", ""),
            "citation_arxiv": cit.get("arxiv", ""),
            "has_expected_effects": bool(ee),
            "n_expected_effects": len(ee),
            "ee_panels": ";".join(sorted(set(x.get("panel", "") for x in ee))),
            "blocking_note": (
                "expected_effects declared; only needs a same-stack run; cite a measured TSV"
                if ee else "no expected_effects; needs both prediction + same-stack run"
            ),
        })
    with (OUT / "p6_iter134_empty_action_gap.tsv").open("w", newline="") as f:
        if empty:
            w = csv.DictWriter(f, fieldnames=list(empty[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(empty)

    # Summary
    summary = {
        "iter": 134,
        "pillar": "P6",
        "n_variant_delta_entries": len(vd),
        "n_total_measured_rows": len(ci_rows),
        "ci_method_diversity": dict(ci_div),
        "n_seed_inconsistency_rows": len(seed_incons),
        "n_ci_shape_violations": len(shape_violations),
        "n_xpanel_companion_rows": len(cp_companion),
        "n_empty_measured_entries": len(empty),
        "n_empty_with_expected_effects": sum(1 for e in empty if e["has_expected_effects"]),
        "n_empty_without_expected_effects": sum(1 for e in empty if not e["has_expected_effects"]),
    }
    (OUT / "p6_iter134_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"WROTE ci_div={len(ci_div)} seed_incons={len(seed_incons)} "
          f"shape_viol={len(shape_violations)} xp_companion={len(cp_companion)} empty={len(empty)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()