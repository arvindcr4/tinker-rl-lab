#!/usr/bin/env python3
"""Iter 33 — P5 schema-bump closure worklist.

Closes iter-32's deferred surface candidate ("link iter-37 alignment
score with iter-32's expanded schema to surface a per-cell worklist")
by classifying every registry entry on the 3 iter-32
`min_report.loss_form` new fields and acting on the worklist.

Per-entry × per-field class: APPLIES_REPORTED / APPLIES_UNPOPULATED /
NOT_RELEVANT / NOT_APPLICABLE. Counterfactual badge-gain bootstrap
on the worklist (B=2000, seed=20260704). Closure act populates the
worklist entries from the same DELTA_IMPLICATIONS source-of-truth
iter-32 uses, then re-audits.
"""
from __future__ import annotations
import csv, datetime, json, pathlib, random, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
REG = ROOT / "registry" / "entries"
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)

NEW_FIELDS = ("token_aggregation", "reward_shaping_type", "sampling_dynamic_filter")
# (delta_id) -> tuple of NEW_FIELDS this variant touches. Mirrored from
# scripts/p5p8/delta_minreport_consistency.py DELTA_IMPLICATIONS.
DELTA_FIELD_APPLIES: dict[str, tuple[str, ...]] = {
    "delta_dapo":     ("token_aggregation", "reward_shaping_type", "sampling_dynamic_filter"),
    "delta_drgrpo":   (),
    "delta_gspo":     ("token_aggregation",),
    "delta_gift":     ("reward_shaping_type",),
    "delta_aero":     (), "delta_areal":    (), "delta_cppo":   (),
    "delta_es":       (), "delta_mcgrpo":   (), "delta_ngrpo":  (),
    "delta_scafgrpo": (),
}
DELTA_FIELD_VALUE: dict[str, dict[str, object]] = {
    "delta_dapo": {"token_aggregation": "token",
                   "reward_shaping_type": "overlong_penalty",
                   "sampling_dynamic_filter": True},
    "delta_gspo": {"token_aggregation": "sequence"},
    "delta_gift": {"reward_shaping_type": "gamma_baseline"},
}


def _loss(entry, leaf):
    lf = entry.get("min_report", {}).get("loss_form", {})
    return lf.get(leaf) if isinstance(lf, dict) else None


def classify(entry, field):
    deltas = [d for d in entry.get("variant_deltas_applied", [])
              if d.get("status") in ("implemented", "surrogate")]
    if not deltas:
        return "NOT_APPLICABLE"
    if not any(field in DELTA_FIELD_APPLIES.get(d["delta_id"], ()) for d in deltas):
        return "NOT_RELEVANT"
    return "APPLIES_REPORTED" if _loss(entry, field) is not None else "APPLIES_UNPOPULATED"


def entry_score(entry):
    out, applies_n, closeable_n = {}, 0, 0
    for f in NEW_FIELDS:
        v = classify(entry, f)
        out[f] = v
        if v in ("APPLIES_REPORTED", "APPLIES_UNPOPULATED"):
            applies_n += 1
        if v == "APPLIES_UNPOPULATED":
            closeable_n += 1
    return out, applies_n, closeable_n


def main():
    # CLI: --no-apply skips the side-effecting closure act; default applies.
    apply_changes = "--no-apply" not in sys.argv[1:]
    if not REG.exists():
        print(f"ERROR: {REG} missing", file=sys.stderr); return 1
    entries = [json.loads(p.read_text()) for p in sorted(REG.glob("*.json"))
               if json.loads(p.read_text()).get("record_type") == "stack"]

    counts = {f: {s: 0 for s in ("APPLIES_REPORTED", "APPLIES_UNPOPULATED",
                                  "NOT_RELEVANT", "NOT_APPLICABLE")} for f in NEW_FIELDS}
    rows = []
    worklist = []
    for e in entries:
        per, applies_n, closeable_n = entry_score(e)
        for f, v in [(f, per[f]) for f in NEW_FIELDS]:
            counts[f][v] += 1
            rows.append({"entry_id": e["id"], "field": f, "status": v,
                         "actual_value": "" if _loss(e, f) is None else str(_loss(e, f))})
        rows.append({"entry_id": e["id"], "field": "(deficit)",
                     "status": f"{applies_n - closeable_n}/{applies_n}_populated",
                     "actual_value": ""})
        if closeable_n > 0:
            worklist.append({"entry_id": e["id"],
                              "closeable_n": closeable_n,
                              "applies_n": applies_n,
                              "unpopulated_fields": [f for f in NEW_FIELDS
                                                      if per[f] == "APPLIES_UNPOPULATED"]})
    worklist.sort(key=lambda x: -x["closeable_n"])

    # TSV
    tsv = RES / "p5_schema_bump_closure.tsv"
    with tsv.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["entry_id", "delta_ids", "field", "status", "actual_value"])
        for r in rows:
            w.writerow([r["entry_id"],
                        ",".join(sorted({d["delta_id"] for d in (
                            json.loads((REG / f"{r['entry_id']}.json").read_text())
                        ).get("variant_deltas_applied", [])}
                        )) if r["field"] != "(deficit)" else "",
                        r["field"], r["status"], r["actual_value"]])

    # Per-field summary
    field_summary = {}
    for f, c in counts.items():
        applies_total = c["APPLIES_REPORTED"] + c["APPLIES_UNPOPULATED"]
        field_summary[f] = {
            "applies_reported": c["APPLIES_REPORTED"],
            "applies_unpopulated": c["APPLIES_UNPOPULATED"],
            "not_relevant": c["NOT_RELEVANT"],
            "not_applicable": c["NOT_APPLICABLE"],
            "applies_total": applies_total,
            "populate_rate": round(c["APPLIES_REPORTED"] / applies_total, 4)
                              if applies_total else None,
            "closure_remaining": c["APPLIES_UNPOPULATED"],
        }

    # Counterfactual badge-gain: each fully-populated new field adds ~1.58 pts
    # (item1_weight / 6 * 0.95) to the per-cell badge. Worklist mean gain.
    per_field_ceiling = 10 / 6 * 0.95
    if worklist:
        gain_mean = sum(w["closeable_n"] for w in worklist) * per_field_ceiling / len(worklist)
        pop_rate = {f: field_summary[f]["populate_rate"] or 0.0 for f in NEW_FIELDS}
        rng = random.Random(20260704)
        boot = []
        for _ in range(2000):
            sample = [rng.choice(worklist) for _ in range(len(worklist))]
            completed = sum(1 for w in sample
                            if all(rng.random() < pop_rate[f] for f in w["unpopulated_fields"]))
            boot.append(sum(w["closeable_n"] for w in sample)
                        * per_field_ceiling * (completed / len(sample)))
        boot.sort()
        ci_low, ci_high = boot[int(0.025 * 2000)], boot[int(0.975 * 2000)]
        ci_mean = sum(boot) / 2000
    else:
        gain_mean = ci_low = ci_high = ci_mean = 0.0

    summary = {
        "ts": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "n_entries": len(entries),
        "n_new_fields": len(NEW_FIELDS),
        "iter32_bump_fields": list(NEW_FIELDS),
        "per_field": field_summary,
        "worklist_size": len(worklist),
        "worklist": worklist,
        "counterfactual_gain": {
            "model": "per_field_ceiling = item1_weight/6*0.95 (~1.58 pts/field)",
            "n_entries_with_closure": len(worklist),
            "mean_gain_if_all_completed": round(gain_mean, 3),
            "bootstrap_95ci": [round(ci_low, 3), round(ci_high, 3)],
            "bootstrap_mean": round(ci_mean, 3),
            "n_boot": 2000,
        },
    }

    # ------ Closure act: populate worklist from DELTA_FIELD_VALUE ------
    populated, failures = [], []
    if not apply_changes:
        print("(skipping closure act; rerun without --no-apply to apply)")
    for w in worklist:
        ep = REG / f"{w['entry_id']}.json"
        if not ep.exists():
            failures.append({"entry_id": w["entry_id"], "reason": "missing"}); continue
        entry = json.loads(ep.read_text())
        lf = entry.get("min_report", {}).get("loss_form", {})
        if not isinstance(lf, dict):
            failures.append({"entry_id": w["entry_id"], "reason": "loss_form_not_dict"}); continue
        changes = {}
        for d in entry.get("variant_deltas_applied", []):
            for field, val in DELTA_FIELD_VALUE.get(d["delta_id"], {}).items():
                if field in w["unpopulated_fields"] and lf.get(field) is None:
                    lf[field] = val; changes[field] = val
        if changes:
            if apply_changes:
                entry["min_report"]["loss_form"] = lf
                ep.write_text(json.dumps(entry, indent=2) + "\n")
            populated.append({"entry_id": w["entry_id"], "fields_populated": changes})

    # Re-audit
    post = {f: {s: 0 for s in counts[f]} for f in NEW_FIELDS}
    for p in sorted(REG.glob("*.json")):
        e = json.loads(p.read_text())
        if e.get("record_type") != "stack":
            continue
        for f in NEW_FIELDS:
            post[f][classify(e, f)] += 1
    post_summary = {}
    for f in NEW_FIELDS:
        applies_total = post[f]["APPLIES_REPORTED"] + post[f]["APPLIES_UNPOPULATED"]
        post_rate = (post[f]["APPLIES_REPORTED"] / applies_total
                     if applies_total else None)
        pre_rate = field_summary[f]["populate_rate"] or 0
        post_summary[f] = {
            "populate_rate_pre": field_summary[f]["populate_rate"],
            "populate_rate_post": round(post_rate, 4) if post_rate is not None else None,
            "delta": round((post_rate or 0) - pre_rate, 4),
            "closure_remaining_post": post[f]["APPLIES_UNPOPULATED"],
        }
    summary["closure_act"] = {"populated_changes": populated,
                              "populate_failures": failures,
                              "post_field_summary": post_summary,
                              "n_entries_populated": len(populated)}

    # Write outputs
    (RES / "p5_schema_bump_closure.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    short = {
        "ts": summary["ts"], "n_entries": len(entries),
        "worklist_size": len(worklist),
        "n_populated_in_closure_act": len(populated),
        "per_field_populate_rate": {f: field_summary[f]["populate_rate"] for f in NEW_FIELDS},
        "post_closure_per_field_populate_rate": {f: post_summary[f]["populate_rate_post"] for f in NEW_FIELDS},
        "per_field_closure_remaining": {f: field_summary[f]["closure_remaining"] for f in NEW_FIELDS},
        "post_closure_per_field_closure_remaining": {f: post_summary[f]["closure_remaining_post"] for f in NEW_FIELDS},
        "mean_gain_if_all_completed": round(gain_mean, 3),
        "bootstrap_mean_gain_95ci": [round(ci_low, 3), round(ci_high, 3)],
    }
    (RES / "p5_schema_bump_closure_summary.json").write_text(json.dumps(short, indent=2, sort_keys=True))

    # Stdout
    print(f"entries audited:    {len(entries)}")
    print(f"new-fields:         {list(NEW_FIELDS)}")
    print(f"worklist size:      {len(worklist)}")
    print("per-field populate_rate / applies_total / closure_remaining:")
    for f in NEW_FIELDS:
        s = field_summary[f]
        print(f"  {f:<24s} {str(s['populate_rate']):<8s} {s['applies_total']:<3d} {s['closure_remaining']:<3d}")
    print(f"per-entry closure ceiling gain: {gain_mean:+.3f} pts")
    print(f"bootstrap 95% CI on completed-set gain: "
          f"[{ci_low:+.3f}, {ci_high:+.3f}] (mean {ci_mean:+.3f}, B=2000)")
    print("top-3 worklist:")
    for w in worklist[:3]:
        print(f"  {w['entry_id']:<38s} closeable={w['closeable_n']} ({', '.join(w['unpopulated_fields'])})")
    print(f"closure act: populated {len(populated)} entries")
    for p in populated:
        print(f"  {p['entry_id']}: {list(p['fields_populated'].keys())}")
    print("per-field populate_rate PRE -> POST (delta):")
    for f in NEW_FIELDS:
        s = post_summary[f]
        print(f"  {f:<24s} {str(s['populate_rate_pre']):<8s} -> {str(s['populate_rate_post']):<8s} "
              f"(delta {s['delta']:+.4f})")
    print(f"wrote {tsv}")
    print(f"wrote {RES / 'p5_schema_bump_closure.json'}")
    print(f"wrote {RES / 'p5_schema_bump_closure_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
