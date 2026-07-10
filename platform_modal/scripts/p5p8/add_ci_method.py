#!/usr/bin/env python3
"""P6 JOB B (iter 28): drive ledger item 10 to validated.

Item 10 (proposed since iter 2): "add `outcomes.ci_method` to the schema so
future entries can self-report CI provenance (paired-bootstrap n_boot, seed)."
It was deferred repeatedly for fear a schema bump would force re-validating all
31 entries. This makes the addition *backward-compatible and optional*: a new
named property inside the already-permissive `outcomes` object, with no
`required` constraint, so every existing entry still validates unchanged.

We then truthfully populate `ci_method` on the 7 tinker entries whose outcomes
were derived from the N2 same-stack four-method run, whose CI methodology is the
paired bootstrap in `platform_modal/scripts/p5p8/registry_validate.py::bootstrap_paired_diff`
(n_boot=2000, seed=0, percentile 95%). Finally we re-validate all 31 entries.

Stdlib + jsonschema only. Idempotent (safe to re-run).
"""
import csv
import glob
import json
from pathlib import Path
import jsonschema

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
SCHEMA = ROOT / "registry/schema.json"
ENTRIES = sorted(glob.glob(str(ROOT / "registry/entries/*.json")))
OUT = ROOT / "experiments/results/p5p8"

# entries whose outcomes came from the N2 same-stack four-method tensors
N2_TINKER = {
    "tinker_aero_qwen3.5-4b_gsm8k.json", "tinker_areal_qwen3.5-4b_gsm8k.json",
    "tinker_dapo_qwen3.5-4b_gsm8k.json", "tinker_drgrpo_qwen3.5-4b_gsm8k.json",
    "tinker_gift_qwen3.5-4b_gsm8k.json", "tinker_grpo_qwen3.5-4b_gsm8k.json",
    "tinker_gspo_qwen3.5-4b_gsm8k.json",
}
CI_METHOD = {
    "method": "paired_bootstrap",
    "n_boot": 2000,
    "seed": 0,
    "ci_level": 0.95,
    "source": ("platform_modal/scripts/p5p8/registry_validate.py::bootstrap_paired_diff "
               "(percentile 2.5/97.5 over aligned per-step N2 tensors)"),
}


def patch_schema():
    s = json.loads(SCHEMA.read_text())
    defs = s["$defs"]
    # 1. add reusable ci_method definition (idempotent)
    defs["ci_method"] = {
        "type": ["object", "null"],
        "description": ("Optional self-reported provenance of the confidence "
                        "interval on this entry's outcomes/deltas (added iter 28, "
                        "backward-compatible optional field). null = unreported."),
        "properties": {
            "method": {"$ref": "#/$defs/nullable_string"},
            "n_boot": {"$ref": "#/$defs/nullable_integer"},
            "seed": {"$ref": "#/$defs/nullable_integer"},
            "ci_level": {"$ref": "#/$defs/nullable_number"},
            "source": {"$ref": "#/$defs/nullable_string"},
        },
        "additionalProperties": False,
    }
    # 2. reference it inside outcomes (optional; no `required`)
    oc = defs["stack_record"]["properties"]["outcomes"]
    oc["properties"]["ci_method"] = {"$ref": "#/$defs/ci_method"}
    SCHEMA.write_text(json.dumps(s, indent=2) + "\n")
    return s


def populate_entries():
    touched = []
    for f in ENTRIES:
        name = Path(f).name
        if name not in N2_TINKER:
            continue
        e = json.loads(Path(f).read_text())
        oc = e.get("outcomes")
        if not isinstance(oc, dict):
            continue
        if oc.get("ci_method") != CI_METHOD:
            oc["ci_method"] = CI_METHOD
            Path(f).write_text(json.dumps(e, indent=2) + "\n")
            touched.append(name)
    return touched


def validate_all(schema):
    validator = jsonschema.Draft202012Validator(schema)
    rows = []
    npass = 0
    for f in ENTRIES:
        name = Path(f).name
        e = json.loads(Path(f).read_text())
        errs = sorted(validator.iter_errors(e), key=str)
        ok = not errs
        npass += ok
        has_ci = isinstance(e.get("outcomes"), dict) and e["outcomes"].get("ci_method") is not None
        rows.append({"entry": name, "schema_pass": ok,
                     "has_ci_method": has_ci,
                     "first_error": (errs[0].message[:90] if errs else "")})
    return rows, npass


def main():
    schema = patch_schema()
    touched = populate_entries()
    rows, npass = validate_all(schema)
    n = len(rows)
    n_ci = sum(r["has_ci_method"] for r in rows)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "registry_ci_method_coverage.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(rows)
    summary = {"n_entries": n, "n_schema_pass": npass,
               "all_pass": npass == n, "n_self_report_ci_method": n_ci,
               "ci_method_entries": [r["entry"] for r in rows if r["has_ci_method"]],
               "touched_this_run": touched, "ci_method_value": CI_METHOD}
    with open(OUT / "registry_ci_method_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[ci_method] schema PASS {npass}/{n}; ci_method self-reported by {n_ci}/{n} entries")
    print(f"           touched this run: {touched}")
    assert npass == n, "REGRESSION: not all entries validate!"


if __name__ == "__main__":
    main()
