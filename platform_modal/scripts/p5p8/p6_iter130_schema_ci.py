#!/usr/bin/env python3
"""Review17-v2 canonical JSON Schema and finite-ordered-interval audit.
Empty inputs and invalid records fail the CLI. Completeness badges do not reward
significance counts and are not measures of scientific evidence quality."""
from __future__ import annotations
import json
from pathlib import Path

import sys
import jsonschema
sys.path.append(str(Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT
from review17_contract import validate_interval
ROOT = REPO_ROOT
ENTRIES = ROOT / "platform_hybrid/registry/entries"
SCHEMA = ROOT / "platform_hybrid/registry/schema.json"
OUT_DIR = ROOT / "platform_hybrid/experiments/results/p5p8/review17_v2"

# Required fields by record_type (mirrors registry/schema.json oneOf).
REQ_STACK = ["record_type", "id", "label_claimed", "framework", "provenance", "min_report"]
REQ_DELTA = ["record_type", "id", "name", "base", "deltas", "citation"]
MIN_REPORT_ITEMS = [
    "loss_form", "reference_kl", "sampler_backend", "telemetry",
    "group_size_schedule", "heldout_split", "decontamination",
]
MEASURED_DERIVED_REQD = ["metric", "panel", "base", "delta", "source"]
# The 9 delta entries the iter-128 audit found with stale method=point_no_perseed_sd
STALE_METHOD_VALUES = {"point_no_perseed_sd", "point_only"}

# ------------------------------------------------------------- schema walker


def _is_nullable(prop):
    """Return set of types if `prop` is the nullable helper, else None."""
    if not isinstance(prop, dict):
        return None
    t = prop.get("type")
    if isinstance(t, list) and "null" in t:
        return set(t)
    return None


def canonical_validate(rec, errs):
    schema = json.loads(SCHEMA.read_text())
    validator = jsonschema.validators.validator_for(schema)(schema)
    errs.extend(error.message for error in validator.iter_errors(rec))
    for row in rec.get("measured", []) if isinstance(rec.get("measured", []), list) else []:
        if not isinstance(row, dict):
            continue
        lo, hi = row.get("ci_low"), row.get("ci_high")
        if (lo is not None or hi is not None) and not validate_interval(lo, hi):
            errs.append("invalid measured interval")
    return errs


def valid_stack(rec, errs):
    return canonical_validate(rec, errs)


def valid_delta(rec, errs):
    return canonical_validate(rec, errs)


# ----------------------------------------------------------------- per-row health


def ci_overlaps_zero(row):
    lo, hi = row.get("ci_low"), row.get("ci_high")
    if not validate_interval(lo, hi):
        return None
    return lo <= 0 <= hi


def ci_width(row):
    lo, hi = row.get("ci_low"), row.get("ci_high")
    if not validate_interval(lo, hi):
        return None
    return hi - lo


def measured_health(rec):
    """Return dict: n_rows, n_sig, n_zero_ci, median_n, max_abs_delta,
    n_stale_method, n_point_zero_width, n_panels, n_metrics."""
    out = {
        "n_rows": 0, "n_sig": 0, "n_zero_ci": 0, "median_n": 0,
        "max_abs_delta": 0.0, "n_stale_method": 0, "n_point_zero_width": 0,
        "n_panels": 0, "n_metrics": 0,
    }
    rows = rec.get("measured") or []
    if not rows:
        return out
    out["n_rows"] = len(rows)
    panels = set()
    metrics = set()
    ns = []
    abs_d = []
    for r in rows:
        if r.get("significant") is True:
            out["n_sig"] += 1
        if ci_overlaps_zero(r) is True:
            out["n_zero_ci"] += 1
        w = ci_width(r)
        if w is None:
            out["n_point_zero_width"] += 1  # not in (lo,hi) form
        elif w == 0.0:
            out["n_point_zero_width"] += 1  # exact point CI (lo==hi)
        ns.append(r.get("n") or 0)
        d = r.get("delta")
        if isinstance(d, (int, float)):
            abs_d.append(abs(d))
        cm = (r.get("ci_method") or {}).get("method") if isinstance(r.get("ci_method"), dict) else None
        if isinstance(cm, str) and cm in STALE_METHOD_VALUES:
            out["n_stale_method"] += 1
        if r.get("panel"):
            panels.add(r["panel"])
        if r.get("metric"):
            metrics.add(r["metric"])
    out["n_panels"] = len(panels)
    out["n_metrics"] = len(metrics)
    out["median_n"] = sorted(ns)[len(ns) // 2] if ns else 0
    out["max_abs_delta"] = max(abs_d) if abs_d else 0.0
    return out


def badge(h):
    if h["n_rows"] == 0:
        return "D"
    if h["n_stale_method"] == 0 and h["n_rows"] >= 1:
        return "B"
    return "C"


# ----------------------------------------------------------------- CI walk


def process(entry_path):
    rec_id = entry_path.stem
    row = {
        "entry_id": rec_id,
        "parse_ok": 0, "schema_ok": 0, "badge": "?", "err": "",
        "n_measured": 0, "n_significant": 0, "n_stale_method": 0,
        "n_panels": 0, "n_metrics": 0, "median_n": 0, "max_abs_delta": 0.0,
        "label_or_name": "", "framework": "", "openness": "",
    }
    try:
        rec = json.loads(entry_path.read_text())
        row["parse_ok"] = 1
    except Exception as e:
        row["err"] = f"parse:{type(e).__name__}:{e}"
        return row
    errs = []
    if not isinstance(rec, dict):
        row["err"] = "record must be an object"
        return row
    rt = rec.get("record_type")
    if rt == "stack":
        valid_stack(rec, errs)
        row["label_or_name"] = rec.get("label_claimed", "") or ""
        fw = rec.get("framework", {}) or {}
        if not isinstance(fw, dict):
            fw = {}
        row["framework"] = fw.get("name", "") or ""
        row["openness"] = fw.get("openness", "") or ""
    elif rt == "variant_delta":
        valid_delta(rec, errs)
        row["label_or_name"] = rec.get("name", "") or ""
    else:
        errs.append(f"record_type={rt}")
    if errs:
        row["err"] = ";".join(errs[:6])
        return row
    row["schema_ok"] = 1 if not errs else 0
    h = measured_health(rec)
    for k, v in h.items():
        row[f"n_{k}"] if False else None  # silence
    row["n_measured"] = h["n_rows"]
    row["n_significant"] = h["n_sig"]
    row["n_stale_method"] = h["n_stale_method"]
    row["n_panels"] = h["n_panels"]
    row["n_metrics"] = h["n_metrics"]
    row["median_n"] = h["median_n"]
    row["max_abs_delta"] = round(h["max_abs_delta"], 6)
    row["badge"] = badge(h)
    return row


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(ENTRIES.glob("*.json")):
        rows.append(process(p))
    # write tsv
    cols = ["entry_id", "parse_ok", "schema_ok", "badge", "err",
            "n_measured", "n_significant", "n_stale_method",
            "n_panels", "n_metrics", "median_n", "max_abs_delta",
            "label_or_name", "framework", "openness"]
    with (OUT_DIR / "p6_iter130_schema_ci.tsv").open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    # summary json
    badges = {"A": 0, "B": 0, "C": 0, "D": 0, "?": 0}
    parse_ok = schema_ok = 0
    n_total_measured = 0
    n_total_significant = 0
    n_total_stale = 0
    by_type = {"stack": 0, "variant_delta": 0, "other": 0}
    for r in rows:
        badges[r["badge"]] = badges.get(r["badge"], 0) + 1
        if r["parse_ok"]:
            parse_ok += 1
        if r["schema_ok"]:
            schema_ok += 1
        n_total_measured += r["n_measured"]
        n_total_significant += r["n_significant"]
        n_total_stale += r["n_stale_method"]
        # better type inference: variant_delta = name present, no framework
        if r["framework"]:
            by_type["stack"] += 1
        elif r["entry_id"].startswith("delta_"):
            by_type["variant_delta"] += 1
        else:
            by_type["other"] += 1
    summary = {
        "n_entries": len(rows),
        "parse_ok": parse_ok,
        "schema_ok": schema_ok,
        "badges": badges,
        "by_type": by_type,
        "totals": {
            "measured_rows": n_total_measured,
            "significant_rows": n_total_significant,
            "stale_method_rows": n_total_stale,
        },
        "stale_action_required": n_total_stale > 0,
        "schema_health": "PASS" if rows and schema_ok == len(rows) else "FAIL",
    }
    (OUT_DIR / "p6_iter130_schema_ci.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    # patch plan — entries that need a ci_method patch
    patch_rows = [r for r in rows if r["n_stale_method"] > 0]
    with (OUT_DIR / "p6_iter130_schema_ci_patch_plan.tsv").open("w") as f:
        f.write("entry_id\tn_stale_method\tn_measured\taction\n")
        for r in patch_rows:
            f.write(f"{r['entry_id']}\t{r['n_stale_method']}\t{r['n_measured']}\t"
                    f"recompute paired-step bootstrap per iter-128; "
                    f"swap ci_method to bootstrap_paired_5seed\n")
    print(f"iter-130: validated {len(rows)} entries; parse_ok={parse_ok}/{len(rows)}; "
          f"schema_ok={schema_ok}/{len(rows)}; badges={badges}; "
          f"stale_method_rows={n_total_stale}")

    return 0 if rows and schema_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
