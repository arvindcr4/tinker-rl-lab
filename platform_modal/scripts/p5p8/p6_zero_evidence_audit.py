#!/usr/bin/env python3
"""P6 iter-74: zero-measurement-evidence registry audit.

After iter-74's p6_drgrpo_measured_evidence.py patches delta_drgrpo.json,
catalog the remaining delta entries that have NO measured block, NO
expected_effects, NO claim_validation, and characterize what panel would
ground each. Output: experiments/results/p5p8/p6_zero_evidence_audit.tsv

Also builds the registry-level evidence summary:
  experiments/results/p5p8/p6_registry_evidence_summary.tsv
  one row per (delta_id) with has_measured / n_panels / n_verdicts / sources.

Then runs jsonschema validation on every entry + emits 14/14 PASS evidence.
"""
import csv
import json
import pathlib
import re

import jsonschema

ROOT = pathlib.Path(__file__).resolve().parents[2]
ENTRIES = ROOT / "registry" / "entries"
SCHEMA = ROOT / "registry" / "schema.json"
OUT = ROOT / "experiments/results/p5p8"

# characterization of the 4 unmeasured deltas: which panel would ground them
# and what's the closest existing data
CHARACTERIZATION = {
    "delta_dapo": {
        "n_components": 5,
        "core_claim": ("asymmetric clip + dynamic sampling + token-level loss + "
                       "overlong reward shaping + KL removal"),
        "needed_panel": ("N2-style same-stack 4-method paired tensor "
                         "(aero/gift/areal/dapo/grpo) — NOT in this worktree"),
        "closest_proxy": ("length_bias data (KL-removal acts on length); "
                          "currently we have only n2 metrics for aero/gift/areal"),
        "verdict_status": "OPEN",
    },
    "delta_gspo": {
        "n_components": 2,
        "core_claim": "sequence-level importance ratio + sequence-level clip",
        "needed_panel": ("N2 same-stack 4-method run with GSPO replacing GRPO; "
                         "would compare per-token vs per-sequence ratio gradient"),
        "closest_proxy": ("tinker_gspo_qwen3.5-4b_gsm8k.json (stack entry exists, "
                          "outcomes=null) — needs rollout tensor logging on GSPO"),
        "verdict_status": "OPEN",
    },
    "delta_liteppo": {
        "n_components": 4,
        "core_claim": ("lightweight PPO: remove value head + clip advantages; "
                       "use group-relative baseline as critic surrogate"),
        "needed_panel": ("N2 same-stack with LitePPO replacing GRPO — NOT in this "
                         "worktree"),
        "closest_proxy": ("no tinker LitePPO rollout log; would need a fresh "
                          "training run"),
        "verdict_status": "OPEN",
    },
    "delta_reinforce": {
        "n_components": 1,
        "core_claim": ("vanilla REINFORCE with group-relative baseline "
                       "(no clip, no KL, no value head)"),
        "needed_panel": ("N2 same-stack 5th method (REINFORCE) rollout tensor — "
                         "would isolate the clipping mechanism"),
        "closest_proxy": ("no tinker REINFORCE rollout log; the 4 N2 methods are "
                          "all PPO-family variants"),
        "verdict_status": "OPEN",
    },
}


def main():
    schema = json.loads(SCHEMA.read_text())

    # 1) zero-evidence audit
    zero_rows = []
    for p in sorted(ENTRIES.glob("delta_*.json")):
        d = json.loads(p.read_text())
        if d["record_type"] != "variant_delta":
            continue
        meas = d.get("measured") or []
        exp = d.get("expected_effects") or []
        val = d.get("claim_validation") or []
        n_comps = len(d.get("deltas", []))
        if not meas and not exp and not val:
            cid = d["id"]
            char = CHARACTERIZATION.get(cid, {})
            zero_rows.append({
                "delta_id": cid,
                "name": d.get("name", ""),
                "n_components": n_comps,
                "n_measured": 0,
                "n_expected": 0,
                "n_validated": 0,
                "core_claim": char.get("core_claim", "(unknown; see registry)"),
                "needed_panel": char.get("needed_panel", ""),
                "closest_proxy": char.get("closest_proxy", ""),
                "verdict_status": char.get("verdict_status", "OPEN"),
            })

    zero_path = OUT / "p6_zero_evidence_audit.tsv"
    with open(zero_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        cols = ["delta_id", "name", "n_components", "n_measured", "n_expected",
                "n_validated", "core_claim", "needed_panel", "closest_proxy",
                "verdict_status"]
        w.writerow(cols)
        for r in zero_rows:
            w.writerow([r[c] for c in cols])

    # 2) registry-level evidence summary
    summary_rows = []
    for p in sorted(ENTRIES.glob("delta_*.json")):
        d = json.loads(p.read_text())
        if d["record_type"] != "variant_delta":
            continue
        meas = d.get("measured") or []
        exp = d.get("expected_effects") or []
        val = d.get("claim_validation") or []
        panels = sorted({m["panel"] for m in meas}) if meas else []
        sup = sum(1 for v in val if v.get("verdict") == "SUPPORTS")
        con = sum(1 for v in val if v.get("verdict") == "CONTRADICTS")
        neu = sum(1 for v in val if v.get("verdict") == "NEUTRAL")
        unc = sum(1 for v in val if v.get("verdict") == "UNCLAIMED")
        yield_resid = d.get("measured_yield_residual")
        summary_rows.append({
            "delta_id": d["id"],
            "name": d.get("name", ""),
            "n_components": len(d.get("deltas", [])),
            "n_measured": len(meas),
            "n_expected": len(exp),
            "n_validated": len(val),
            "n_panels": len(panels),
            "panels": ";".join(panels),
            "supports": sup,
            "contradicts": con,
            "neutral": neu,
            "unclaimed": unc,
            "has_yield_residual": bool(yield_resid),
            "n_yield_metrics": (len([k for k in (yield_resid or {}) if isinstance(yield_resid.get(k), (int, float))
                                     and yield_resid[k] is not None]) if yield_resid else 0),
        })

    sum_path = OUT / "p6_registry_evidence_summary.tsv"
    with open(sum_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        cols = ["delta_id", "name", "n_components", "n_measured", "n_expected",
                "n_validated", "n_panels", "panels", "supports", "contradicts",
                "neutral", "unclaimed", "has_yield_residual", "n_yield_metrics"]
        w.writerow(cols)
        for r in summary_rows:
            w.writerow([r[c] for c in cols])

    # 3) schema validation
    schema_pass = 0
    schema_fail = 0
    schema_errs = []
    for p in sorted(ENTRIES.glob("*.json")):
        rec = json.loads(p.read_text())
        try:
            jsonschema.validate(rec, schema)
            schema_pass += 1
        except jsonschema.ValidationError as e:
            schema_fail += 1
            schema_errs.append((p.name, str(e.message)[:120]))

    # 4) overall findings
    n_total = len(summary_rows)
    n_with_measured = sum(1 for r in summary_rows if r["n_measured"] > 0)
    n_total_validated = sum(r["n_validated"] for r in summary_rows)
    n_total_supports = sum(r["supports"] for r in summary_rows)
    n_total_contradicts = sum(r["contradicts"] for r in summary_rows)
    n_total_neutral = sum(r["neutral"] for r in summary_rows)

    summary = {
        "n_delta_entries": n_total,
        "n_with_measured": n_with_measured,
        "n_zero_evidence": len(zero_rows),
        "total_measured_rows": sum(r["n_measured"] for r in summary_rows),
        "total_validated_rows": n_total_validated,
        "total_supports": n_total_supports,
        "total_contradicts": n_total_contradicts,
        "total_neutral": n_total_neutral,
        "schema_pass": schema_pass,
        "schema_fail": schema_fail,
        "schema_errs": schema_errs,
        "zero_evidence_tsv": str(zero_path),
        "summary_tsv": str(sum_path),
        "zero_evidence_deltas": [r["delta_id"] for r in zero_rows],
    }
    (OUT / "p6_zero_evidence_audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


if __name__ == "__main__":
    s = main()
    print(json.dumps(s, indent=2))