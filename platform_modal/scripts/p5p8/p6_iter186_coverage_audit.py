#!/usr/bin/env python3
"""P6 iter-186 coverage audit. Three angles, four artifacts:

  1. variant_coverage: every delta_*.json classified by presence of
     measured[] / expected_effects[] / claim_validation[], cross-ref'd
     against zvf130_<id>.json risk-panel + tinker_*/wandb_* raw traces.
  2. stack_coverage: every stack record, min_report fill at 7-item + leaf.
  3. schema_validate: branch-dispatch (record_type → stack_record OR
     variant_delta_record) so the oneOf composite error doesn't drown the
     real additionalProperties / enum failures.

Outputs:
  p6_iter186_coverage.tsv          (one row / variant_delta)
  p6_iter186_stack_minreport.tsv   (one row / stack record)
  p6_iter186_schema_valid.tsv      (one row / entry)
  p6_iter186_coverage.json         (aggregates)
"""
import argparse, json, pathlib, sys
from collections import defaultdict, Counter

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REG  = ROOT / "registry"
ENT  = REG / "entries"
OUT  = ROOT / "experiments" / "results" / "p5p8"
OUT.mkdir(parents=True, exist_ok=True)

try:
    import jsonschema
    HAVE_JS = True
except ImportError:
    HAVE_JS = False


def load_schema():
    return json.loads((REG / "schema.json").read_text())


def load_entries():
    """Two parallel views of the same disk content:

      entries_raw — as parsed (with `_id` and `_path` tags) — used by the
        coverage reports, where we want the metadata visible.
      entries_clean — same content but with `_id` / `_path` / `_parse_error`
        stripped — used by the schema validator, where the schema's
        `additionalProperties: false` would otherwise flag the tags.
    """
    raw, clean = [], []
    for p in sorted(ENT.glob("*.json")):
        try:
            d = json.loads(p.read_text())
            tag = {"_id": p.stem, "_path": str(p.relative_to(ROOT))}
            raw.append({**tag, **d})
            clean.append(d)
        except Exception as e:
            tag = {"_id": p.stem, "_path": str(p.relative_to(ROOT)),
                   "_parse_error": str(e)}
            raw.append(tag)
            clean.append(tag)
    return raw, clean


def variant_coverage(entries):
    """For every delta_*.json: count measured[], expected[], claim_validation[],
    check zvf130 + raw-trace cross-reference."""
    zvf_ids = {p.stem.replace("zvf130_", "")
               for p in ENT.glob("zvf130_*.json")}
    raw_methods = defaultdict(list)
    for p in ENT.glob("tinker_*.json"):
        try:
            d = json.loads(p.read_text())
            lbl = (d.get("label_claimed") or "").lower()
            if lbl:
                raw_methods[lbl].append(p.stem)
        except Exception:
            pass
    for p in ENT.glob("wandb_*.json"):
        try:
            d = json.loads(p.read_text())
            lbl = (d.get("label_claimed") or "").lower()
            if lbl:
                raw_methods[lbl].append(p.stem)
        except Exception:
            pass
    # Map delta id → base label used by raw traces (e.g. delta_aero ↔ aero)
    delta_to_label = {d.get("id"): (d.get("name") or d.get("id", "")).lower()
                      for d in entries
                      if d.get("record_type") == "variant_delta"}
    rows = []
    for d in entries:
        if d.get("record_type") != "variant_delta":
            continue
        rid = d.get("id", d.get("_id", "?"))
        measured  = d.get("measured", []) or []
        expected  = d.get("expected_effects", []) or []
        claimed   = d.get("claim_validation", []) or []
        m_dicts = [m for m in measured if isinstance(m, dict)]
        uniq_measured = {(m.get("metric"), m.get("panel")) for m in m_dicts}
        panels = sorted({m.get("panel") for m in m_dicts})
        metric_set = sorted({m.get("metric") for m in m_dicts})
        short = rid.replace("delta_", "")
        zvf_crossref = "yes" if short in zvf_ids else "no"
        raw_traces   = ";".join(raw_methods.get(delta_to_label.get(rid, short.lower()), [])) or "none"
        if len(claimed) >= 3 and len(measured) >= 3:
            v = "FULL"
        elif len(measured) >= 1:
            v = "PARTIAL"
        elif len(expected) >= 1:
            v = "EXPECTED-ONLY"
        else:
            v = "BLANK"
        rows.append({
            "entry_id":      rid, "verdict": v,
            "n_measured":    len(measured), "n_measured_uniq": len(uniq_measured),
            "panels":        ";".join(panels) or "-",
            "metrics":       ";".join(metric_set) or "-",
            "n_expected":    len(expected), "n_claimed": len(claimed),
            "zvf130_xref":   zvf_crossref, "raw_traces": raw_traces,
            "n_panels_zvf":  len([p for p in panels if p and p.startswith("zvf130")]),
        })
    return rows


def stack_coverage(entries):
    """For every stack record: count min_report fill at item-level (7 items)
    and leaf-level (all leaves under min_report)."""
    ITEMS = ["loss_form", "reference_kl", "sampler_backend", "telemetry",
             "group_size_schedule", "heldout_split", "decontamination"]
    rows = []
    for d in entries:
        if d.get("record_type") != "stack":
            continue
        rid = d.get("id", d.get("_id", "?"))
        mr  = d.get("min_report") or {}
        item_scores = {}
        leaf_total = leaf_filled = 0
        for it in ITEMS:
            sub = mr.get(it) or {}
            leaves = [v for v in sub.values() if not isinstance(v, (dict, list))]
            if leaves:
                filled = sum(1 for v in leaves if v is not None)
                item_scores[it] = round(100 * filled / len(leaves))
                leaf_total += len(leaves); leaf_filled += filled
            else:
                item_scores[it] = -1
        leaf_pct = round(100 * leaf_filled / leaf_total) if leaf_total else 0
        rows.append({
            "entry_id": rid, "leaf_pct": leaf_pct,
            "items_full":    sum(1 for s in item_scores.values() if s == 100),
            "items_partial": sum(1 for s in item_scores.values() if 0 < s < 100),
            "items_zero":    sum(1 for s in item_scores.values() if s == 0),
            "items_absent":  sum(1 for s in item_scores.values() if s < 0),
            **{f"item_{k}": v for k, v in item_scores.items()},
        })
    return rows


def schema_validate(entries, schema):
    """Validate by dispatching on record_type to avoid oneOf composite errors.
    The top-level oneOf yields a single composite error whose .message is the
    whole data instance — useless for triage. We re-validate against the
    branch sub-schema with $defs preserved so $refs resolve."""
    defs = schema.get("$defs", {})
    rows = []
    if HAVE_JS:
        for d in entries:
            if "_parse_error" in d:
                rows.append({"entry_id": d.get("_id", "?"), "record_type": "?",
                             "valid": "no", "errors": d["_parse_error"],
                             "n_errors": 1})
                continue
            rt = d.get("record_type")
            if rt == "stack":
                branch_key = "stack_record"
            elif rt == "variant_delta":
                branch_key = "variant_delta_record"
            else:
                rows.append({"entry_id": d.get("id", d.get("_id", "?")),
                             "record_type": rt or "?",
                             "valid": "no",
                             "errors": f"unknown record_type={rt}",
                             "n_errors": 1})
                continue
            branch = {**defs[branch_key], "$defs": defs}
            v = jsonschema.Draft202012Validator(branch)
            errs = sorted(v.iter_errors(d), key=lambda e: list(map(str, e.path)))
            if not errs:
                rows.append({"entry_id": d.get("id", d.get("_id", "?")),
                             "record_type": rt, "valid": "yes",
                             "errors": "-", "n_errors": 0})
            else:
                msg = "; ".join(
                    f"{'/'.join(map(str,e.path)) or '<root>'}: {e.message[:160]}"
                    for e in errs[:4])
                if len(errs) > 4:
                    msg += f" (+{len(errs)-4} more)"
                rows.append({"entry_id": d.get("id", d.get("_id", "?")),
                             "record_type": rt, "valid": "no",
                             "errors": msg, "n_errors": len(errs)})
    else:
        # stdlib fallback: just record_type + presence of required-branch keys
        for d in entries:
            if "_parse_error" in d:
                rows.append({"entry_id": d["_id"], "record_type": "?",
                             "valid": "no", "errors": d["_parse_error"]})
                continue
            rt = d.get("record_type")
            req = {"stack": ["id", "label_claimed", "framework", "min_report"],
                   "variant_delta": ["id", "name", "base"]}.get(rt, [])
            missing = [k for k in req if k not in d]
            rows.append({
                "entry_id": d.get("id", d["_id"]),
"record_type": rt or "?",
                "valid": "yes" if not missing else "no",
                "errors": "missing " + ",".join(missing) if missing else "-",
            })
    return rows


def write_tsv(path, rows, cols):
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    entries_raw, entries_clean = load_entries()
    schema  = load_schema()

    vcov = variant_coverage(entries_raw)
    scov = stack_coverage(entries_raw)
    sval = schema_validate(entries_clean, schema)

    # Aggregate verdict totals
    v_verdict = Counter(r["verdict"] for r in vcov)
    s_valid   = Counter(r["valid"] for r in sval)
    n_var = len(vcov); n_stack = len(scov); n_total = n_var + n_stack

    # Method-level pairing: which variant_deltas have measured rows from
    # zvf130 cross-reference vs which have N2 cross-reference
    paired = []
    for r in vcov:
        has_zvf130 = r["zvf130_xref"] == "yes"
        has_raw = r["raw_traces"] != "none"
        if r["n_measured"] == 0:
            layer = "BLANK"
        elif has_zvf130 and r["n_panels_zvf"] >= 1:
            layer = "zvf130-measured"
        elif has_raw:
            layer = "raw-only-measured"
        else:
            layer = "measured-other"
        paired.append((r["entry_id"], r["verdict"], layer, has_zvf130, has_raw))

    summary = {
        "n_entries_total": len(entries_raw),
        "n_variant_delta": n_var,
        "n_stack":         n_stack,
        "variant_delta_verdict_totals": dict(v_verdict),
        "schema_valid_totals":          dict(s_valid),
        "have_jsonschema_pkg":          HAVE_JS,
        "method_layer_counts": dict(Counter(p[2] for p in paired)),
        "n_with_zvf130_xref":  sum(1 for p in paired if p[3]),
        "n_with_raw_traces":   sum(1 for p in paired if p[4]),
        "fully_blank_deltas":  [r["entry_id"] for r in vcov
                                if r["verdict"] == "BLANK"],
        "expected_only_deltas":[r["entry_id"] for r in vcov
                                if r["verdict"] == "EXPECTED-ONLY"],
        "audit_metadata": {
            "iter":   186,
            "pillar": "P6",
            "veins":  ["(b) coverage audit", "(c) schema validation"],
            "source": "platform_modal/scripts/p5p8/p6_iter186_coverage_audit.py",
            "seed":   20260706,
        },
    }
    (out / "p6_iter186_coverage.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True))

    # TSVs
    write_tsv(out / "p6_iter186_coverage.tsv", vcov,
              ["entry_id","verdict","n_measured","n_measured_uniq",
               "panels","metrics","n_expected","n_claimed",
               "zvf130_xref","raw_traces","n_panels_zvf"])
    write_tsv(out / "p6_iter186_stack_minreport.tsv", scov,
              ["entry_id","leaf_pct","items_full","items_partial",
               "items_zero","items_absent",
               "item_loss_form","item_reference_kl","item_sampler_backend",
               "item_telemetry","item_group_size_schedule",
               "item_heldout_split","item_decontamination"])
    write_tsv(out / "p6_iter186_schema_valid.tsv", sval,
              ["entry_id","record_type","valid","n_errors","errors"])

    print(f"variant_delta entries: {n_var}  verdicts={dict(v_verdict)}")
    print(f"stack records:    {n_stack}")
    print(f"schema valid totals: {dict(s_valid)}  (jsonschema={HAVE_JS})")
    print(f"blank:   {summary['fully_blank_deltas']}")
    print(f"expect-only: {summary['expected_only_deltas']}")
    print(f"layer_counts: {summary['method_layer_counts']}")
    print(f"xvref zvf130={summary['n_with_zvf130_xref']}/{n_var} "
          f"raw={summary['n_with_raw_traces']}/{n_var}")


if __name__ == "__main__":
    main()