#!/usr/bin/env python3
"""Iter 30 (P6 / Pillar 2) — variant-delta × MIN-REPORT consistency audit.

For every (entry, applied_delta, component) triple, the audit looks up
the component-specific MIN-REPORT implication (the field/value the
technique would yield if fully applied) and compares to the entry's
actual MIN-REPORT. Verdicts: MATCH / MISMATCH / MISSING_REPORT /
SURROGATE_OBS / NOT_APPLICABLE.

Outputs:
  experiments/results/p5p8/delta_minreport_consistency.tsv  (per-triple rows)
  experiments/results/p5p8/delta_minreport_consistency.json (summary)
"""
from __future__ import annotations
import json, pathlib, sys, datetime, warnings
from collections import defaultdict, Counter

warnings.filterwarnings("ignore", category=DeprecationWarning)

REG = pathlib.Path("registry/entries")
OUT = pathlib.Path("experiments/results/p5p8")
OUT.mkdir(parents=True, exist_ok=True)


# Hand-curated component-level implication table. For each (delta_id,
# component) we list the MIN-REPORT field → expected value mapping that
# the technique, if *fully* applied, would yield.
#
# Where a delta's component is *not* a MIN-REPORT-visible technique
# (e.g. DAPO's "overlong_reward_shaping" lives in a `reward` block that
# the registry does not currently expose) we list no implication, so the
# verdict is NOT_APPLICABLE.
DELTA_IMPLICATIONS: dict[str, dict[str, dict[str, object]]] = {
    "delta_dapo": {
        # explicit claims per the source paper (yu2025dapo / arXiv:2503.14476)
        "clip_higher": {
            "loss_form.clip_eps_low": 0.2,
            "loss_form.clip_eps_high": 0.28,
        },
        "dynamic_sampling": {
            # Iter-32 extension: sampling_dynamic_filter captures DAPO's
            # dynamic-sampling-on accuracy filter.
            "loss_form.sampling_dynamic_filter": True,
        },
        "token_level_loss": {
            # Iter-32 extension: token_aggregation captures DAPO's
            # token-level loss aggregation.
            "loss_form.token_aggregation": "token",
        },
        "overlong_reward_shaping": {
            # Iter-32 extension: reward_shaping_type captures DAPO's
            # overlong-reward shaping.
            "loss_form.reward_shaping_type": "overlong_penalty",
        },
        "kl_removed": {
            "reference_kl.kl_beta": 0.0,
            "reference_kl.kl_estimator": "none",
            "reference_kl.reference_policy": False,
        },
    },
    "delta_drgrpo": {
        # drgrpo removes two normalizations (lin2025drgrpo / arXiv:2503.20783)
        "length_normalization": {
            "loss_form.length_normalization": False,
        },
        "advantage_std_normalization": {
            "loss_form.advantage_normalization": "none",
        },
    },
    "delta_gspo": {
        # gspo moves importance ratio to sequence level (zhai2025gspo).
        "sequence_level_ratio": {
            "loss_form.importance_ratio_level": "sequence",
            # Iter-32 extension: token_aggregation matches the same notion.
            "loss_form.token_aggregation": "sequence",
        },
        "sequence_level_clip": {
            # clip bounds differ at sequence level — registry exposes
            # token-level clip, no sequence-level field.
        },
    },
    "delta_gift": {
        "gamma_likelihood_baseline": {
            # Iter-32 extension: reward_shaping_type captures GIFT's
            # gamma-likelihood baseline prior.
            "loss_form.reward_shaping_type": "gamma_baseline",
        },
    },
    "delta_aero": {
        "advantage_guided_evolution": {
            # entropy-guided advantage — no MIN-REPORT field captures
            # the entropy prior.
        },
    },
    "delta_areal": {
        "autoscaling_Rollout": {
            # rollout-budget decoupling — no MIN-REPORT field.
        },
    },
    "delta_cppo": {
        "continuity_penalty": {
            # continuity penalty on log-prob delta — the closest
            # MIN-REPORT field is length_normalization (True), but the
            # semantics are different (cppo is a log-prob smoothness
            # prior, not response-length normalisation). Mark as no
            # strict implication; we keep this entry here as
            # documentation that the audit was run.
        },
    },
    "delta_es": {
        "black_box_perturbation": {
            # ES replaces PG with central differences; no MIN-REPORT
            # field captures this.
        },
    },
    "delta_mcgrpo": {
        "mcts_rollout": {
            # no MIN-REPORT field.
        },
        "per_prompt_diversity_bonus": {
            # no MIN-REPORT field.
        },
    },
    "delta_ngrpo": {
        "per_prompt_normalization": {
            # no MIN-REPORT field.
        },
    },
    "delta_scafgrpo": {
        "scaffold_aware_advantage": {
            # no MIN-REPORT field.
        },
    },
}


def _get_field(entry: dict, field_path: str):
    if "." not in field_path:
        return None
    block, leaf = field_path.split(".", 1)
    block_d = entry.get("min_report", {}).get(block)
    if not isinstance(block_d, dict) or leaf not in block_d:
        return None
    return block_d.get(leaf)


def classify(entry: dict, status: str, expected) -> str:
    if not expected or status in ("absent", "unknown"):
        return "NOT_APPLICABLE"
    if status == "surrogate":
        any_obs = any(_get_field(entry, f) is not None for f in expected)
        return "SURROGATE_OBS" if any_obs else "NOT_APPLICABLE"
    rows = []
    for f, v in expected.items():
        actual = _get_field(entry, f)
        if actual is None:
            rows.append("MISSING_REPORT")
        else:
            eq = (isinstance(v, float) and isinstance(actual, (int, float))
                  and abs(actual - v) < 1e-9) or actual == v
            rows.append("MATCH" if eq else "MISMATCH")
    if any(r == "MISMATCH" for r in rows):
        return "MISMATCH"
    if any(r == "MISSING_REPORT" for r in rows):
        return "MISSING_REPORT"
    return "MATCH"


def main() -> int:
    if not REG.exists():
        print(f"ERROR: {REG} not found", file=sys.stderr)
        return 1
    entries = {}
    for p in sorted(REG.glob("*.json")):
        d = json.loads(p.read_text())
        if d.get("record_type") == "stack":
            entries[d["id"]] = d
    rows = []
    for entry_id, entry in sorted(entries.items()):
        for applied in entry.get("variant_deltas_applied", []):
            delta_id = applied["delta_id"]
            comp = applied["component"]
            status = applied["status"]
            impls = DELTA_IMPLICATIONS.get(delta_id, {}).get(comp, {})
            verdict = classify(entry, status, impls)
            if impls:
                # record one row per field for the TSV
                for field, expected in impls.items():
                    actual = _get_field(entry, field)
                    if status in ("absent", "unknown"):
                        per_field = "NOT_APPLICABLE"
                    elif status == "surrogate":
                        per_field = "SURROGATE_OBS" if actual is not None else "NOT_APPLICABLE"
                    elif actual is None:
                        per_field = "MISSING_REPORT"
                    else:
                        if isinstance(expected, float) and isinstance(actual, (int, float)):
                            per_field = "MATCH" if abs(actual - expected) < 1e-9 else "MISMATCH"
                        else:
                            per_field = "MATCH" if actual == expected else "MISMATCH"
                    rows.append({
                        "entry_id": entry_id,
                        "delta_id": delta_id,
                        "component": comp,
                        "status": status,
                        "field": field,
                        "expected_value": str(expected),
                        "actual_value": "" if actual is None else str(actual),
                        "verdict": per_field,
                    })
            else:
                # no MIN-REPORT implication for this component
                rows.append({
                    "entry_id": entry_id,
                    "delta_id": delta_id,
                    "component": comp,
                    "status": status,
                    "field": "(no_minreport_implication)",
                    "expected_value": "(none)",
                    "actual_value": "(none)",
                    "verdict": "NOT_APPLICABLE",
                })
    # TSV
    out_tsv = OUT / "delta_minreport_consistency.tsv"
    with out_tsv.open("w") as f:
        f.write("entry_id\tdelta_id\tcomponent\tstatus\tfield\texpected_value\tactual_value\tverdict\n")
        for r in rows:
            f.write("\t".join([
                r["entry_id"], r["delta_id"], r["component"], r["status"],
                r["field"], r["expected_value"], r["actual_value"], r["verdict"],
            ]) + "\n")
    # Summary
    by_verdict = Counter(r["verdict"] for r in rows)
    by_delta: dict[str, Counter] = defaultdict(Counter)
    by_entry: dict[str, Counter] = defaultdict(Counter)
    by_status: dict[str, Counter] = defaultdict(Counter)
    by_field: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_delta[r["delta_id"]][r["verdict"]] += 1
        by_entry[r["entry_id"]][r["verdict"]] += 1
        by_status[r["status"]][r["verdict"]] += 1
        if r["field"] != "(no_minreport_implication)":
            by_field[r["field"]][r["verdict"]] += 1
    # Implementation-honesty: among status=implemented triples, what
    # fraction end up MATCH vs MISSING_REPORT vs MISMATCH.
    impl_rows = [r for r in rows if r["status"] == "implemented"]
    n_impl = len(impl_rows)
    n_match = sum(1 for r in impl_rows if r["verdict"] == "MATCH")
    n_mismatch = sum(1 for r in impl_rows if r["verdict"] == "MISMATCH")
    n_missing = sum(1 for r in impl_rows if r["verdict"] == "MISSING_REPORT")
    # Component-coverage: of 21 (delta,component) pairs in the table, how
    # many have at least one MIN-REPORT implication?
    n_total_components = sum(len(v) for v in DELTA_IMPLICATIONS.values())
    n_with_implication = sum(
        1 for d in DELTA_IMPLICATIONS.values()
        for c, impls in d.items() if impls
    )
    summary = {
        "ts": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "n_entries_with_deltas": len({r["entry_id"] for r in rows}),
        "n_triples": len(rows),
        "n_implemented_triples": n_impl,
        "by_verdict": dict(by_verdict),
        "implementation_honesty": {
            "n_match": n_match,
            "n_mismatch": n_mismatch,
            "n_missing_report": n_missing,
            "match_rate": round(n_match / n_impl, 4) if n_impl else None,
            "mismatch_rate": round(n_mismatch / n_impl, 4) if n_impl else None,
            "missing_report_rate": round(n_missing / n_impl, 4) if n_impl else None,
            "consistent_rate": round((n_match + n_missing) / n_impl, 4) if n_impl else None,
        },
        "schema_exposure": {
            "total_components": n_total_components,
            "components_with_minreport_implication": n_with_implication,
            "exposure_rate": round(n_with_implication / n_total_components, 4) if n_total_components else None,
        },
        "by_status_verdict": {s: dict(c) for s, c in by_status.items()},
        "by_delta_verdict": {d: dict(c) for d, c in by_delta.items()},
        "by_entry_verdict": {e: dict(c) for e, c in by_entry.items()},
        "by_field_verdict": {f: dict(c) for f, c in by_field.items()},
        "implication_table": DELTA_IMPLICATIONS,
    }
    out_json = OUT / "delta_minreport_consistency.json"
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {out_tsv} ({len(rows)} rows)")
    print(f"wrote {out_json}")
    print(f"by_verdict: {dict(by_verdict)}")
    print(f"implementation honesty: {summary['implementation_honesty']}")
    print(f"schema exposure: {summary['schema_exposure']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
