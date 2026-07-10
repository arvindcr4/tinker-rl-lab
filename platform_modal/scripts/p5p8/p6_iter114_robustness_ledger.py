"""
Iter 114 — P6 Measured-Block Robustness Ledger.

For every measured[] row in registry/entries/delta_*.json, classify into a
robustness bucket by (effect / CI-half-width) signal strength and surface
load-bearing vs fragile vs underpowered rows.

Outputs (4 files):
  - p6_iter114_robustness_per_row.tsv   (1 row per measured[] entry, sorted by |effect|/CI_h)
  - p6_iter114_robustness_per_entry.tsv (1 row per delta entry, groundedness rollup)
  - p6_iter114_robustness_claim_matrix.tsv (entry × metric × panel measured-vs-expected)
  - p6_iter114_robustness_summary.json   (machine-readable headline counts)

Vein (a) — validate existing entries against measured behavior. Fresh angle:
characterize the *measurement quality* of the registry's evidence base,
not the directional claim-validation that iter-90/iter-46 already cover.

Definitions (stdlib only):
  effect      = |delta|
  ci_half     = (ci_high - ci_low) / 2
  snr         = effect / ci_half   (signal-to-noise ratio)
  bucket rules:
    POINT_ONLY    = ci_method.method == 'point_no_perseed_sd'
                  OR (ci_low == ci_high == delta)
    UNDERPOWERED  = n_obs < 8                  (paired-seed bootstrap at n<8 is fragile)
    LOAD_BEARING  = significant AND snr >= 5.0 (effect ≥ 5× CI half-width)
    FRAGILE_SIG   = significant AND snr < 5.0  (CI barely excludes 0)
    FRAGILE_NS    = NOT significant AND ci_half > 0 AND effect > 0
                  = NS but data trended in a direction (cannot falsify)
  groundedness_score = LOAD_BEARING * 1.0
                     + FRAGILE_SIG  * 0.5
                     + FRAGILE_NS   * 0.25
                     + UNDERPOWERED * 0.10
                     + POINT_ONLY   * 0.00
                     / total_rows

Run: python3 platform_modal/scripts/p5p8/p6_iter114_robustness_ledger.py
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ENTRIES_DIR = ROOT / "registry" / "entries"
OUT_DIR = ROOT / "experiments" / "results" / "p5p8"


def classify(measured_row):
    """Return (bucket, snr, ci_half) for a single measured[] row."""
    delta = abs(float(measured_row.get("delta", 0.0)))
    cl = float(measured_row.get("ci_low", 0.0))
    ch = float(measured_row.get("ci_high", 0.0))
    ci_half = (ch - cl) / 2.0
    sig = bool(measured_row.get("significant", False))
    n_obs = int(measured_row.get("n") or 0) if measured_row.get("n") is not None else 0
    method = (measured_row.get("ci_method") or {}).get("method") or ""

    # POINT_ONLY: collapsed CI (ci_low == ci_high == delta) or labelled
    if method == "point_no_perseed_sd" or (ci_half == 0.0 and delta > 0.0):
        return "POINT_ONLY", None, ci_half, n_obs, sig

    snr = delta / ci_half if ci_half > 0 else (float("inf") if delta > 0 else 0.0)
    if n_obs and n_obs < 8:
        return "UNDERPOWERED", snr, ci_half, n_obs, sig
    if sig and snr >= 5.0:
        return "LOAD_BEARING", snr, ci_half, n_obs, sig
    if sig and snr < 5.0:
        return "FRAGILE_SIG", snr, ci_half, n_obs, sig
    if (not sig) and ci_half > 0 and delta > 0:
        return "FRAGILE_NS", snr, ci_half, n_obs, sig
    return "DEGENERATE", snr, ci_half, n_obs, sig


BUCKET_WEIGHTS = {
    "LOAD_BEARING": 1.00,
    "FRAGILE_SIG":  0.50,
    "FRAGILE_NS":   0.25,
    "UNDERPOWERED": 0.10,
    "POINT_ONLY":   0.00,
    "DEGENERATE":   0.00,
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    deltas = sorted(ENTRIES_DIR.glob("delta_*.json"))
    if not deltas:
        print(f"no delta_*.json in {ENTRIES_DIR}", file=sys.stderr)
        sys.exit(2)

    per_row = []
    per_entry = defaultdict(lambda: {
        "name": "",
        "total_rows": 0,
        "load_bearing": 0,
        "fragile_sig": 0,
        "fragile_ns": 0,
        "underpowered": 0,
        "point_only": 0,
        "degenerate": 0,
        "groundedness_score": 0.0,
        "has_expected_effects": 0,
        "n_expected_effects": 0,
        "expected_unmeasured": 0,
        "theoretical_only": False,
        "n_panels": set(),
        "citation_arxiv": "",
    })
    claim_matrix = []  # measured × expected_effects overlap audit

    for dp in deltas:
        e = json.load(open(dp))
        eid = e.get("id") or dp.stem
        per_entry[eid]["name"] = e.get("name") or eid
        per_entry[eid]["citation_arxiv"] = (e.get("citation") or {}).get("arxiv") or ""
        exp = e.get("expected_effects") or []
        per_entry[eid]["has_expected_effects"] = int(bool(exp))
        per_entry[eid]["n_expected_effects"] = len(exp)
        measured_pairs = set()
        for m in (e.get("measured") or []):
            bucket, snr, ci_half, n, sig = classify(m)
            delta = float(m.get("delta", 0.0))
            ci_l = float(m.get("ci_low", 0.0))
            ci_h = float(m.get("ci_high", 0.0))
            metric = m.get("metric", "")
            panel = m.get("panel", "")
            measured_pairs.add((metric, panel))
            per_entry[eid]["total_rows"] += 1
            per_entry[eid][bucket.lower()] = per_entry[eid].get(bucket.lower(), 0) + 1
            per_entry[eid]["n_panels"].add(panel)
            ci_method_obj = m.get("ci_method") or {}
            per_row.append({
                "delta_id": eid,
                "metric": metric,
                "panel": panel,
                "delta": delta,
                "ci_low": ci_l,
                "ci_high": ci_h,
                "ci_half": ci_half,
                "abs_delta": abs(delta),
                "n": n,
                "significant": int(sig),
                "snr": "" if snr is None else f"{snr:.3f}",
                "ci_method": ci_method_obj.get("method") or "",
                "n_boot": ci_method_obj.get("n_boot") or "",
                "bucket": bucket,
                "source": m.get("source") or "",
            })
        # theoretical-only check: expected_effects declared but NO measured row
        for ef in exp:
            mpair = (ef.get("metric", ""), ef.get("panel", ""))
            if mpair not in measured_pairs:
                per_entry[eid]["expected_unmeasured"] += 1
                claim_matrix.append({
                    "delta_id": eid,
                    "metric": ef.get("metric", ""),
                    "panel": ef.get("panel", ""),
                    "predicted_sign": ef.get("predicted_sign", ""),
                    "measured": "no",
                    "observed_delta": "",
                    "rationale": ef.get("rationale", "")[:120],
                })
            else:
                claim_matrix.append({
                    "delta_id": eid,
                    "metric": ef.get("metric", ""),
                    "panel": ef.get("panel", ""),
                    "predicted_sign": ef.get("predicted_sign", ""),
                    "measured": "yes",
                    "observed_delta": "",  # filled below
                    "rationale": ef.get("rationale", "")[:120],
                })
        # round 2: fill observed_delta for claim_matrix where measured=yes
        # grouped by (delta, metric, panel) so we can attach a measured row value
        measured_lookup = {}
        for m in (e.get("measured") or []):
            measured_lookup[(m.get("metric", ""), m.get("panel", ""))] = m
        for cm in claim_matrix:
            if cm["delta_id"] == eid and cm["measured"] == "yes":
                key = (cm["metric"], cm["panel"])
                m = measured_lookup.get(key)
                if m is not None:
                    cm["observed_delta"] = float(m.get("delta", 0.0))

    # sort per_row by abs_delta descending (load-bearing first)
    per_row.sort(key=lambda r: r["abs_delta"], reverse=True)

    # ---------- write per-row TSV ----------
    p_row = OUT_DIR / "p6_iter114_robustness_per_row.tsv"
    with open(p_row, "w") as f:
        cols = ["delta_id", "metric", "panel", "delta", "ci_low", "ci_high",
                "ci_half", "abs_delta", "n", "significant", "snr",
                "ci_method", "n_boot", "bucket", "source"]
        f.write("\t".join(cols) + "\n")
        for r in per_row:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    # ---------- write per-entry TSV with groundedness_score ----------
    p_ent = OUT_DIR / "p6_iter114_robustness_per_entry.tsv"
    with open(p_ent, "w") as f:
        cols = ["delta_id", "name", "total_rows", "load_bearing", "fragile_sig",
                "fragile_ns", "underpowered", "point_only", "degenerate",
                "groundedness_score", "n_panels", "n_expected_effects",
                "expected_unmeasured", "theoretical_only", "citation_arxiv"]
        f.write("\t".join(cols) + "\n")
        rows_out = []
        for eid, agg in per_entry.items():
            tr = max(agg["total_rows"], 1)
            # per-entry stores bucket counts under LOWERCASE keys
            # e.g. "load_bearing" -> BUCKET_WEIGHTS["LOAD_BEARING"]
            score = 0.0
            for lc_key, weight in BUCKET_WEIGHTS.items():
                score += weight * agg.get(lc_key.lower(), 0)
            agg["groundedness_score"] = round(score / tr, 4)
            agg["theoretical_only"] = bool(agg["total_rows"] == 0 and agg["n_expected_effects"] > 0)
            rows_out.append((eid, agg))
        # sort by groundedness_score desc, then by total_rows desc
        rows_out.sort(key=lambda x: (-x[1]["groundedness_score"], -x[1]["total_rows"]))
        for eid, agg in rows_out:
            f.write("\t".join([
                eid,
                agg["name"],
                str(agg["total_rows"]),
                str(agg.get("load_bearing", 0)),
                str(agg.get("fragile_sig", 0)),
                str(agg.get("fragile_ns", 0)),
                str(agg.get("underpowered", 0)),
                str(agg.get("point_only", 0)),
                str(agg.get("degenerate", 0)),
                f"{agg['groundedness_score']:.4f}",
                str(len(agg["n_panels"])),
                str(agg["n_expected_effects"]),
                str(agg["expected_unmeasured"]),
                "1" if agg["theoretical_only"] else "0",
                agg["citation_arxiv"],
            ]) + "\n")

    # ---------- write claim-matrix TSV ----------
    p_mat = OUT_DIR / "p6_iter114_robustness_claim_matrix.tsv"
    with open(p_mat, "w") as f:
        cols = ["delta_id", "metric", "panel", "predicted_sign",
                "measured", "observed_delta", "rationale"]
        f.write("\t".join(cols) + "\n")
        for cm in claim_matrix:
            f.write("\t".join([
                cm["delta_id"],
                cm["metric"],
                cm["panel"],
                cm["predicted_sign"],
                cm["measured"],
                "" if cm["observed_delta"] == "" else f"{cm['observed_delta']:.6f}",
                cm["rationale"].replace("\t", " ").replace("\n", " "),
            ]) + "\n")

    # ---------- write summary JSON ----------
    by_bucket = defaultdict(int)
    for r in per_row:
        by_bucket[r["bucket"]] += 1
    summary = {
        "n_delta_entries": len(per_entry),
        "n_measured_rows": len(per_row),
        "n_claim_matrix_rows": len(claim_matrix),
        "by_bucket": dict(by_bucket),
        "bucket_weights": BUCKET_WEIGHTS,
        "robustness_thresholds": {
            "snr_load_bearing_min": 5.0,
            "n_underpowered_max": 7,
            "ci_method_point_only": "point_no_perseed_sd",
        },
        "entries_by_groundedness": [
            {
                "delta_id": eid,
                "name": agg["name"],
                "groundedness_score": agg["groundedness_score"],
                "total_rows": agg["total_rows"],
                "load_bearing": agg.get("load_bearing", 0),
                "theoretical_only": agg["theoretical_only"],
            }
            for eid, agg in rows_out
        ],
        "top_5_fragile_rows": [
            r for r in sorted(per_row, key=lambda r: float(r["snr"]) if r["snr"] != "" else 1e9)[:5]
        ],
        "top_5_load_bearing_rows": [
            r for r in sorted(per_row, key=lambda r: float(r["snr"]) if r["snr"] != "" else -1, reverse=True)[:5]
        ],
        "audit_source": "platform_modal/scripts/p5p8/p6_iter114_robustness_ledger.py",
        "audit_date": "2026-07-05",
    }
    p_sum = OUT_DIR / "p6_iter114_robustness_summary.json"
    with open(p_sum, "w") as f:
        json.dump(summary, f, indent=2)

    # ---------- console summary ----------
    print(f"=== P6 iter114 robustness ledger ===")
    print(f"delta entries:      {len(per_entry)}")
    print(f"measured rows:      {len(per_row)}  (claim matrix: {len(claim_matrix)})")
    print("by bucket:")
    for b in ["LOAD_BEARING", "FRAGILE_SIG", "FRAGILE_NS", "UNDERPOWERED",
             "POINT_ONLY", "DEGENERATE"]:
        print(f"  {b:13s}: {by_bucket.get(b, 0)}")
    print("top 5 by groundedness_score:")
    for eid, agg in rows_out[:5]:
        print(f"  {eid:18s} {agg['groundedness_score']:.4f}  "
              f"({agg['total_rows']} rows, "
              f"{agg.get('load_bearing', 0)} load-bearing)")
    print("bottom 3 by groundedness_score:")
    for eid, agg in rows_out[-3:]:
        print(f"  {eid:18s} {agg['groundedness_score']:.4f}  "
              f"({agg['total_rows']} rows, theoretical_only={agg['theoretical_only']})")
    print(f"per-row:    {p_row.relative_to(ROOT)}")
    print(f"per-entry:  {p_ent.relative_to(ROOT)}")
    print(f"claim-mtx:  {p_mat.relative_to(ROOT)}")
    print(f"summary:    {p_sum.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
