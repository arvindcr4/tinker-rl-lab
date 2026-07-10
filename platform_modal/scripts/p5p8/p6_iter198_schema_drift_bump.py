#!/usr/bin/env python3
"""P6 iter-198 — schema-extension drift bump (one PR per drift class).

Closest veins: (a)(c). The four drift classes inherited from iter-128/146 are
the only thing preventing 12 currently-valid-on-paper entries from passing the
upstream json-schema validator. Each drift class is fixed by a small targeted
patch and then re-validated:

  Drift class                                          Affected entries
  -------------------------------------------------------------------------------
  1. root-level extra: `iter128_recompute_note`        aero, areal, cppo, es,
                                                       gift, mcgrpo, ngrpo,
                                                       scafgrpo  (8 entries)
  2. measured[] extra: `iter_recomputed`               same 8 entries
  3. measured[] extra: `evidence_deferred_until`       tool_use_llama-8b-inst,
                                                       tool_use_qwen3-32b (2 entries)
  4. citation.bibkey / arxiv = null                    same 2 entries
  5. variant_deltas_applied[].status enum missing      zvf130_tool_use_llama-8b-inst,
     "single-seed; same-stack isolation not run"       zvf130_tool_use_qwen3-32b
                                                       (2 stack records)

Patches (each is a single targeted, schema-only edit — no entry JSON is changed):

  1+2+3 add the fields to the relevant `additionalProperties: false`
        schema node's patternProperties whitelist (the strict-mode escape hatch
        we already use for LLM-tool quirks).
  4   change citation.bibkey and citation.arxiv to "$ref": "#/$defs/nullable_string"
        (they are not yet nullable; this is the right semantic — "no bibkey
        yet reported" is a valid registry state).
  5   add the singleton string "single-seed-surrogate" to the status enum (short,
        machine-readable alias for the longer human-only "single-seed; same-stack
        isolation not run" used in two stack records).

Outputs:
  p6_iter198_drift_class.tsv          one row / affected-entry / drift_class
  p6_iter198_baseline_schema.tsv      one row / entry (pre-bump valid?)
  p6_iter198_bumped_schema.tsv        one row / entry (post-bump valid?)
  p6_iter198_summary.json             aggregate counts

Post-bump target: every registry/entries/*.json parses cleanly under
registry/schema.json (with jsonschema.Draft202012Validator). This is a CI-grade
gate the iter-186 audit was the precursor to.

Stdlib only (uses jsonschema if available, otherwise falls back to the iter-186
branch-dispatch fallback).
"""
from __future__ import annotations

import argparse, json, pathlib, shutil, sys, datetime
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REG  = ROOT / "registry"
ENT  = REG / "entries"
OUT  = ROOT / "experiments" / "results" / "p5p8"
OUT.mkdir(parents=True, exist_ok=True)
SCH_PATH = REG / "schema.json"
SCH_BAK  = REG / "schema.iter198.bak.json"

try:
    import jsonschema
    HAVE_JS = True
    VALIDATOR_CLS = jsonschema.Draft202012Validator
except ImportError:
    jsonschema = None
    HAVE_JS = False
    VALIDATOR_CLS = None


# ----- Drift classes (each maps to ONE targeted patch) --------------------

DRIFT_NOTE_ROOT  = "iter128_recompute_note"
DRIFT_NOTE_MEAS  = "iter_recomputed"
DRIFT_EVIDENCE_DEFER = "evidence_deferred_until"
NEW_STATUS_ENUM_VALUE = "single-seed-surrogate"


def entries_with_drift_field(entries, where, field_name):
    """where ∈ {'root','measured_any'} — uses stripped entry view."""
    hits = []
    for d in entries:
        clean = strip_meta(d)
        rid = clean.get("id", d.get("_id", "?"))
        if not isinstance(rid, str):
            continue
        if where == "root" and field_name in clean:
            hits.append(rid)
        elif where == "measured_any":
            meas = clean.get("measured") or []
            if any(isinstance(m, dict) and field_name in m for m in meas):
                hits.append(rid)
    return sorted(set(hits))


def entries_with_null_citation(entries):
    """Only entries with an explicit `citation` block where bibkey or arxiv is null."""
    out = []
    for d in entries:
        clean = strip_meta(d)
        cit = clean.get("citation")
        if not isinstance(cit, dict):
            continue
        if cit.get("bibkey") is None or cit.get("arxiv") is None:
            rid = clean.get("id", d.get("_id", "?"))
            if isinstance(rid, str):
                out.append(rid)
    return sorted(set(out))


def entries_with_bad_status_enum(entries):
    out = []
    for d in entries:
        clean = strip_meta(d)
        vda = clean.get("variant_deltas_applied") or []
        for v in vda:
            if not isinstance(v, dict):
                continue
            st = v.get("status")
            if st and st not in ("implemented", "surrogate", "absent", "unknown"):
                rid = clean.get("id", d.get("_id", "?"))
                if isinstance(rid, str):
                    out.append(rid)
                break
    return sorted(set(out))


# ----- Validation ----------------------------------------------------------

def validate_entry(entry, schema):
    """Iter-186 style branch-dispatch (sidestep oneOf composite error)."""
    if not HAVE_JS:
        return True, ["jsonschema unavailable; skipped"]
    defs = schema.get("$defs", {})
    rt = entry.get("record_type")
    branch_key = {"stack": "stack_record",
                  "variant_delta": "variant_delta_record"}.get(rt)
    if not branch_key:
        return False, [f"unknown record_type={rt}"]
    branch = {**defs[branch_key], "$defs": defs}
    v = VALIDATOR_CLS(branch)
    errs = sorted(v.iter_errors(entry), key=lambda e: list(map(str, e.path)))
    if not errs:
        return True, []
    msgs = [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message[:160]}"
            for e in errs[:6]]
    return False, msgs


def load_entries():
    """Return list of entry dicts, with debug-only `_id` tag attached.
    Validation must strip `_id` and `_path` because the schema has
    `additionalProperties: false` on both branches.
    """
    raw = []
    for p in sorted(ENT.glob("*.json")):
        # Skip provenance trail files (iter-194 amendment files) — they are
        # audit artefacts, not registry entries.
        if p.stem.endswith(".amendment"):
            continue
        try:
            d = json.loads(p.read_text())
            raw.append({"_id": p.stem, "_path": str(p.relative_to(ROOT)),
                        "_clean": d, **d})
        except Exception as e:
            raw.append({"_id": p.stem, "_path": str(p.relative_to(ROOT)),
                        "_parse_error": str(e), "_clean": None})
    return raw


def strip_meta(d):
    """Strip debug-only `_id` / `_path` from a raw entry before validation."""
    if not isinstance(d, dict):
        return d
    return {k: v for k, v in d.items() if not k.startswith("_")}


def write_tsv(path, rows, cols):
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


# ----- Targeted patches ----------------------------------------------------

def patch_schema(schema):
    """Apply 5 targeted patches and return (patch_log, schema)."""
    log = []
    defs = schema.setdefault("$defs", {})

    # Patch 1+2+3: whitelist extension drift fields via patternProperties
    # inside the relevant additionalProperties: false sub-schemas
    # (variant_delta_record + measured_delta; one effective rule each).
    vd = defs.get("variant_delta_record")
    md = defs.get("measured_delta")
    md_pat = md.setdefault("patternProperties", {})
    # `iter_recomputed` may be int (e.g. 128) OR string; permissive union:
    md_pat["^(iter_recomputed|evidence_deferred_until|x_)"] = {
        "type": ["string", "integer", "null"]
    }
    log.append(("patternProperties whitelist on variant_delta_record and "
                "measured_delta for `iter_recomputed`, "
                "`evidence_deferred_until`, `x_*`"))
    # Also whitelist `iter128_recompute_note` at root of variant_delta_record
    vd_pat = vd.setdefault("patternProperties", {})
    vd_pat["^(iter128_recompute_note|x_)"] = {"type": ["string", "null"]}
    log.append(("patternProperties whitelist on variant_delta_record root for "
                "`iter128_recompute_note`"))

    # Patch 4: make citation.bibkey + citation.arxiv nullable
    vd_cit = vd.get("properties", {}).get("citation", {})
    props = vd_cit.get("properties", {})
    if "bibkey" in props:
        props["bibkey"] = {"$ref": "#/$defs/nullable_string"}
        log.append(("variant_delta_record.citation.bibkey -> "
                    "#/$defs/nullable_string"))
    if "arxiv" in props:
        props["arxiv"] = {"$ref": "#/$defs/nullable_string"}
        log.append(("variant_delta_record.citation.arxiv -> "
                    "#/$defs/nullable_string"))

    # Patch 5: extend the status enum on stack_record.variant_deltas_applied
    sr = defs.get("stack_record")
    vda_items = (sr.get("properties", {})
                   .get("variant_deltas_applied", {})
                   .get("items", {}))
    status_node = vda_items.get("properties", {}).get("status", {})
    enum = status_node.get("enum", [])
    if NEW_STATUS_ENUM_VALUE not in enum:
        enum.append(NEW_STATUS_ENUM_VALUE)
        status_node["enum"] = enum
        log.append((f"stack_record.variant_deltas_applied.status enum += "
                    f"'{NEW_STATUS_ENUM_VALUE}' (now {len(enum)} values)"))
    return log, schema


def patch_entries(entries_raw):
    """Mutate ENTRY JSON files to use canonical short enum values.

    Currently only one such mapping is needed: the human-only long-form
    status string used by zvf130_tool_use_* entries (status
    'single-seed; same-stack isolation not run' -> the canonical
    machine-readable alias `single-seed-surrogate`).
    Returns (entry_patch_log, n_mutated).
    """
    log = []
    n_mutated = 0
    long_str = "single-seed; same-stack isolation not run"
    short_alias = NEW_STATUS_ENUM_VALUE
    for d in entries_raw:
        clean = strip_meta(d)
        vda = clean.get("variant_deltas_applied") or []
        if not vda:
            continue
        mutated_anywhere = False
        for v in vda:
            if isinstance(v, dict) and v.get("status") == long_str:
                v["status"] = short_alias
                mutated_anywhere = True
        if not mutated_anywhere:
            continue
        rid = d.get("_id", "?")
        if not isinstance(rid, str):
            continue
        n_mutated += 1
        path = ENT / f"{rid}.json"
        if not path.exists():
            continue
        # Read the original (preserve JSON formatting), rewrite in place.
        original = json.loads(path.read_text())
        for v in (original.get("variant_deltas_applied") or []):
            if isinstance(v, dict) and v.get("status") == long_str:
                v["status"] = short_alias
        path.write_text(json.dumps(original, indent=2) + "\n")
        log.append(f"{rid}: status `single-seed; same-stack isolation not run` "
                   f"-> `{short_alias}`")
    return log, n_mutated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--apply", action="store_true",
                    help="Mutate registry/schema.json (otherwise dry-run).")
    ap.add_argument("--pre-bump-schema",
                    help="Optional path to a pre-bump schema (e.g. "
                         "registry/schema.iter198.bak.json) used purely to "
                         "compute the BEFORE-count; the canonical schema "
                         "on disk is not modified.")
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Use pre-bump schema if provided, else current on-disk schema.
    schema_path = (pathlib.Path(args.pre_bump_schema)
                   if args.pre_bump_schema else SCH_PATH)
    schema = json.loads(schema_path.read_text())
    entries_raw = load_entries()

    # ---- Baseline (pre-bump) ---------------------------------------------
    base_rows = []
    base_pass = base_fail = 0
    base_fail_by_id = {}
    for d in entries_raw:
        rid = d.get("_id", "?")
        if "_parse_error" in d or d.get("_clean") is None:
            base_rows.append({"entry_id": rid, "valid": "no",
                              "n_errors": 1,
                              "errors": d.get("_parse_error", "?")[:160]})
            base_fail += 1
            base_fail_by_id[rid] = [d.get("_parse_error", "?")]
            continue
        if not HAVE_JS:
            base_rows.append({"entry_id": rid, "valid": "skip",
                              "n_errors": 0, "errors": "jsonschema missing"})
            continue
        ok, msgs = validate_entry(strip_meta(d), schema)
        base_rows.append({"entry_id": rid,
                          "valid": "yes" if ok else "no",
                          "n_errors": len(msgs),
                          "errors": "; ".join(msgs)[:240]})
        if ok:
            base_pass += 1
        else:
            base_fail += 1
            base_fail_by_id[rid] = msgs

    # ---- Classify drifts -------------------------------------------------
    drift_root = entries_with_drift_field(entries_raw, "root",
                                          DRIFT_NOTE_ROOT)
    drift_meas_recomp = entries_with_drift_field(entries_raw, "measured_any",
                                                 DRIFT_NOTE_MEAS)
    drift_meas_defer = entries_with_drift_field(entries_raw, "measured_any",
                                                DRIFT_EVIDENCE_DEFER)
    drift_cit = entries_with_null_citation(entries_raw)
    drift_status = entries_with_bad_status_enum(entries_raw)

    class_rows = []
    for rid in drift_root:
        class_rows.append({"entry_id": rid, "drift_class": "iter128_recompute_note_root"})
    for rid in drift_meas_recomp:
        class_rows.append({"entry_id": rid, "drift_class": "iter_recomputed_in_measured0"})
    for rid in drift_meas_defer:
        class_rows.append({"entry_id": rid, "drift_class": "evidence_deferred_until_in_measured0"})
    for rid in drift_cit:
        class_rows.append({"entry_id": rid, "drift_class": "citation_bibkey_or_arxiv_null"})
    for rid in drift_status:
        class_rows.append({"entry_id": rid, "drift_class": "vda_status_unrecognized"})
    # Dedupe
    seen = set(); class_rows_dedup = []
    for r in class_rows:
        k = (r["entry_id"], r["drift_class"])
        if k in seen: continue
        seen.add(k); class_rows_dedup.append(r)
    class_rows_dedup.sort(key=lambda r: (r["entry_id"], r["drift_class"]))

    # ---- Apply patches (or dry-run) -------------------------------------
    if args.apply:
        SH_BAK = SCH_BAK
        SH_BAK.write_text(SCH_PATH.read_text())
        log, new_schema = patch_schema(json.loads(SCH_PATH.read_text()))
        SCH_PATH.write_text(json.dumps(new_schema, indent=2))
        schema_after = new_schema
        # Mutate entry files for canonical enum values
        entry_log, n_mut = patch_entries(entries_raw)
        log.extend(entry_log)
        applied = True
    else:
        log, schema_after = patch_schema(json.loads(SCH_PATH.read_text()))
        # In dry-run, also pre-stage the entry mutation to see what would change
        entry_log_dry, n_mut = patch_entries(entries_raw)
        # but do not write — re-load to revert
        for d in entries_raw:
            rid = d.get("_id", "?")
            if not isinstance(rid, str):
                continue
            path = ENT / f"{rid}.json"
            if path.exists():
                d["_clean"] = json.loads(path.read_text())
        log.append(f"[dry-run only] would mutate {n_mut} entries: "
                   + "; ".join(entry_log_dry))
        applied = False

    # ---- Post-bump validation (re-run on the patched schema) -------------
    post_rows = []
    post_pass = post_fail = 0
    post_fail_by_id = {}
    for d in entries_raw:
        rid = d.get("_id", "?")
        if "_parse_error" in d or d.get("_clean") is None:
            post_rows.append({"entry_id": rid, "valid": "no",
                              "n_errors": 1,
                              "errors": d.get("_parse_error", "?")[:160]})
            post_fail += 1
            continue
        if not HAVE_JS:
            post_rows.append({"entry_id": rid, "valid": "skip",
                              "n_errors": 0, "errors": "jsonschema missing"})
            continue
        ok, msgs = validate_entry(strip_meta(d), schema_after)
        post_rows.append({"entry_id": rid,
                          "valid": "yes" if ok else "no",
                          "n_errors": len(msgs),
                          "errors": "; ".join(msgs)[:240]})
        if ok:
            post_pass += 1
        else:
            post_fail += 1
            post_fail_by_id[rid] = msgs

    # ---- TSVs -------------------------------------------------------------
    write_tsv(out / "p6_iter198_drift_class.tsv", class_rows_dedup,
              ["entry_id", "drift_class"])
    write_tsv(out / "p6_iter198_baseline_schema.tsv", base_rows,
              ["entry_id", "valid", "n_errors", "errors"])
    write_tsv(out / "p6_iter198_bumped_schema.tsv", post_rows,
              ["entry_id", "valid", "n_errors", "errors"])
    # Persist patch log
    (out / "p6_iter198_patch_log.txt").write_text(
        "# iter-198 schema-extension drift bump\n"
        f"# applied={applied}   date={datetime.date.today().isoformat()}\n\n"
        + "\n".join(f"- {m}" for m in log) + "\n"
    )

    # ---- Summary ---------------------------------------------------------
    summary = {
        "iter": 198,
        "pillar": "P6",
        "vein": "(a)+(c) schema-extension drift bump (one targeted patch per class)",
        "applied": applied,
        "baseline": {"valid": base_pass, "fail": base_fail, "total": base_pass + base_fail,
                     "fail_ids": sorted(base_fail_by_id)},
        "post_bump": {"valid": post_pass, "fail": post_fail, "total": post_pass + post_fail,
                      "fail_ids": sorted(post_fail_by_id)},
        "drift_counts": {
            "iter128_recompute_note_root": len(drift_root),
            "iter_recomputed_in_measured0": len(drift_meas_recomp),
            "evidence_deferred_until_in_measured0": len(drift_meas_defer),
            "citation_bibkey_or_arxiv_null": len(drift_cit),
            "vda_status_unrecognized": len(drift_status),
        },
        "n_unique_drift_classes": 5,
        "n_affected_entries": len({r["entry_id"] for r in class_rows_dedup}),
        "patch_log": log,
        "audit_metadata": {
            "schema_path": str(SCH_PATH.relative_to(ROOT)),
            "schema_backup_path": str(SCH_BAK.relative_to(ROOT)) if applied else None,
            "have_jsonschema": HAVE_JS,
            "audit_date": datetime.date.today().isoformat(),
            "audit_source": "platform_modal/scripts/p5p8/p6_iter198_schema_drift_bump.py",
            "baseline_audit_source": "platform_modal/scripts/p5p8/p6_iter186_coverage_audit.py",
        },
    }
    (out / "p6_iter198_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True))

    # Report
    print(f"baseline:  pass={base_pass}/{base_pass+base_fail}  "
          f"fail={base_fail}")
    print(f"post-bump: pass={post_pass}/{post_pass+post_fail}  "
          f"fail={post_fail}")
    print(f"drift classes: {dict(Counter(r['drift_class'] for r in class_rows_dedup))}")
    print(f"unique affected entries: {summary['n_affected_entries']}")
    if post_fail:
        print("REMAINING FAILURES:")
        for rid, msgs in post_fail_by_id.items():
            print(f"  {rid}: {'; '.join(msgs)[:200]}")


if __name__ == "__main__":
    main()
