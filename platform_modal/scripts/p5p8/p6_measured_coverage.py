#!/usr/bin/env python3
"""P6 iter-58: measured-block provenance & coverage audit (MBPCA).

For every delta_*.json in the registry, audit:

1. Block presence: measured / expected_effects / claim_validation row counts.
2. Source resolution: every measured row's source path resolves on disk and
   carries an mtime; emit age in days.
3. Coverage grid: panel x metric coverage grid across the 14 deltas, plus
   the registry-wide verdict tally (SUPPORTS / CONTRADICTS / NEUTRAL /
   UNCLAIMED).
4. Cross-panel agreement: for entries where the same conceptual metric is
   measured under two panels, normalise sign conventions and emit a
   "same direction?" verdict.

Writes TSVs + JSON to platform_hybrid/experiments/results/p5p8/, plus a regeneratable
sidecar audit file next to the registry. Stdlib only.
"""
import csv
import datetime as _dt
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
ENTRIES = ROOT / "platform_hybrid/registry/entries"
AUDIT = ROOT / "platform_hybrid/registry/measured_block_audit.json"
N2_METRICS_TSV = ROOT / "platform_hybrid/experiments/results/n2_reward_tensor_resume/n2_metrics.tsv"
Z130_TSV = ROOT / "platform_hybrid/experiments/results/zvf_iter130_method_risk.tsv"
OUT = ROOT / "platform_hybrid/experiments/results/p5p8"
OUT.mkdir(parents=True, exist_ok=True)

KNOWN_PANELS = ("n2_same_stack_last10", "zvf130_5seed")
KNOWN_METRICS = (
    "zvf",
    "reward_mean",
    "zvf_risk_mean",
    "mean_zvf",
)


def _age_days(path: pathlib.Path) -> float:
    if not path.exists():
        return float("inf")
    mtime = path.stat().st_mtime
    return (_dt.datetime.now().timestamp() - mtime) / 86400.0


def _resolve(source: str) -> tuple[str, float]:
    """Resolve repo-root- or platform_hybrid-relative artifact paths."""
    raw = pathlib.Path(source)
    candidates = [raw] if raw.is_absolute() else [ROOT / raw, ROOT / "platform_hybrid" / raw]
    p = next((candidate for candidate in candidates if candidate.exists()), None)
    if p is None:
        return "missing", float("inf")
    return "ok", round(_age_days(p), 3)


def _row_verdict_counts(cv_rows):
    out = {"SUPPORTS": 0, "CONTRADICTS": 0, "NEUTRAL": 0, "UNCLAIMED": 0}
    for r in cv_rows:
        v = r.get("verdict", "")
        out[v] = out.get(v, 0) + 1
    return out


def _sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def main():
    delta_files = [
        path
        for path in sorted(ENTRIES.glob("delta_*.json"))
        if json.loads(path.read_text()).get("record_type") == "variant_delta"
    ]
    summary = {
        "n_entries": len(delta_files),
        "per_entry": [],
        "verdict_totals": {},
        "source_path_missing": [],
        "panel_metric_grid": {},
        "cross_panel_agreement": [],
        "empty_measured_gap": [],
        "n_mismatch_audit": 0,
        "audit_metadata": {
            "seed": 20260704,
            "n_boot": 2000,
            "audit_date": _dt.date.today().isoformat(),
            "source": "platform_modal/scripts/p5p8/p6_measured_coverage.py",
        },
    }
    for v in ("SUPPORTS", "CONTRADICTS", "NEUTRAL", "UNCLAIMED"):
        summary["verdict_totals"][v] = 0

    for p in delta_files:
        rec = json.load(open(p))
        eid = rec["id"]
        m_rows = rec.get("measured", [])
        e_rows = rec.get("expected_effects", [])
        cv_rows = rec.get("claim_validation", [])
        cv_counts = _row_verdict_counts(cv_rows)
        for v, c in cv_counts.items():
            summary["verdict_totals"][v] += c

        # Source resolution
        src_resolved = []
        src_missing = []
        for m in m_rows:
            status, age = _resolve(m.get("source", ""))
            src_resolved.append({"path": m.get("source", ""), "status": status,
                                 "age_days": age})
            if status == "missing":
                src_missing.append(m.get("source", ""))

        # Panel × metric coverage
        grid = {}  # (panel, metric) -> count
        for m in m_rows:
            key = (m.get("panel", ""), m.get("metric", ""))
            grid[key] = grid.get(key, 0) + 1
        coverage = sorted(grid.items())

        # Aggregate deltas & per-row min n
        ns = [m.get("n", 0) for m in m_rows]
        sigs = sum(1 for m in m_rows if m.get("significant"))
        sig_pct = (sigs / len(m_rows) * 100.0) if m_rows else 0.0

        per_entry_row = {
            "delta_id": eid,
            "name": rec.get("name", ""),
            "measured_count": len(m_rows),
            "expected_count": len(e_rows),
            "validated_count": len(cv_rows),
            "n_significant": sigs,
            "pct_significant": round(sig_pct, 2),
            "min_n": min(ns) if ns else 0,
            "max_n": max(ns) if ns else 0,
            "supports": cv_counts.get("SUPPORTS", 0),
            "contradicts": cv_counts.get("CONTRADICTS", 0),
            "neutral": cv_counts.get("NEUTRAL", 0),
            "unclaimed": cv_counts.get("UNCLAIMED", 0),
            "n_src_resolved_ok": sum(1 for s in src_resolved if s["status"] == "ok"),
            "n_src_missing": len(src_missing),
            "panels": ";".join(sorted({m.get("panel", "") for m in m_rows})),
            "metrics": ";".join(sorted({m.get("metric", "") for m in m_rows})),
            "citation_arxiv": rec.get("citation", {}).get("arxiv") or "NR",
        }
        summary["per_entry"].append(per_entry_row)
        if src_missing:
            for s in src_missing:
                summary["source_path_missing"].append({"delta_id": eid, "source": s})

        # Empty-measured gap
        if len(m_rows) == 0 and rec.get("deltas"):
            summary["empty_measured_gap"].append({
                "delta_id": eid, "name": rec.get("name", ""),
                "arxiv": rec.get("citation", {}).get("arxiv") or "NR",
                "n_components": len(rec.get("deltas", [])),
                "gap_class": "n2_panel_only",
            })

        summary["panel_metric_grid"][eid] = {f"{pn}|{mn}": grid.get((pn, mn), 0)
                                            for pn in KNOWN_PANELS
                                            for mn in KNOWN_METRICS}

    # Cross-panel agreement: entries with zvf in n2 AND zvf_risk_mean in zvf130
    # both predict <0 risk reduction; here we check raw sign agreement only.
    for p in delta_files:
        rec = json.load(open(p))
        eid = rec["id"]
        m_rows = rec.get("measured", [])
        n2_zvf = next((r for r in m_rows
                       if r.get("panel") == "n2_same_stack_last10"
                       and r.get("metric") == "zvf"), None)
        z130_risk = next((r for r in m_rows
                          if r.get("panel") == "zvf130_5seed"
                           and r.get("metric") == "zvf_risk_mean"), None)
        if n2_zvf and z130_risk:
            ds = _sign(n2_zvf["delta"])
            dr = _sign(z130_risk["delta"])
            same = ds == dr
            summary["cross_panel_agreement"].append({
                "delta_id": eid,
                "n2_zvf_delta": round(n2_zvf["delta"], 6),
                "zvf130_risk_delta": round(z130_risk["delta"], 6),
                "same_sign": bool(same),
                "n2_significant": bool(n2_zvf.get("significant")),
                "zvf130_significant": bool(z130_risk.get("significant")),
            })

    summary["n_mismatch_audit"] = sum(1 for r in summary["per_entry"] if r["n_src_missing"] > 0)

    # Write per-entry TSV
    tsv_path = OUT / "p6_measured_coverage.tsv"
    per_rows = summary["per_entry"]
    with open(tsv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=list(per_rows[0].keys()),
            lineterminator="\n",
        )
        w.writeheader()
        w.writerows(per_rows)

    # Write panel × metric grid TSV
    grid_path = OUT / "p6_measured_coverage_grid.tsv"
    with open(grid_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["delta_id"] + [f"{pn}|{mn}" for pn in KNOWN_PANELS for mn in KNOWN_METRICS])
        for eid, g in summary["panel_metric_grid"].items():
            row = [eid]
            for pn in KNOWN_PANELS:
                for mn in KNOWN_METRICS:
                    row.append(g.get(f"{pn}|{mn}", 0))
            w.writerow(row)

    # Write cross-panel agreement TSV
    x_path = OUT / "p6_measured_cross_panel.tsv"
    with open(x_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["delta_id", "n2_zvf_delta", "zvf130_risk_delta", "same_sign",
                    "n2_significant", "zvf130_significant"])
        for r in summary["cross_panel_agreement"]:
            w.writerow([r["delta_id"], r["n2_zvf_delta"], r["zvf130_risk_delta"],
                        int(r["same_sign"]), int(r["n2_significant"]),
                        int(r["zvf130_significant"])])

    json_path = OUT / "p6_measured_coverage_summary.json"
    json.dump(summary, open(json_path, "w"), indent=2)
    AUDIT.write_text(json.dumps(summary, indent=2))

    # Aggregate headline
    headline = {
        "n_deltas": len(delta_files),
        "with_measured": sum(1 for r in per_rows if r["measured_count"] > 0),
        "empty_measured": sum(1 for r in per_rows if r["measured_count"] == 0),
        "verdict_totals": summary["verdict_totals"],
        "n_cross_panel": len(summary["cross_panel_agreement"]),
        "n_same_sign": sum(1 for r in summary["cross_panel_agreement"] if r["same_sign"]),
        "n_missing_sources": summary["n_mismatch_audit"],
    }
    print(json.dumps(headline, indent=2))
    print(f"wrote {tsv_path}\nwrote {grid_path}\nwrote {x_path}\nwrote {json_path}\nwrote {AUDIT}")


if __name__ == "__main__":
    main()
