#!/usr/bin/env python3
"""Iter-70 P6 — extend measured_yield_residual / outcomes block with
controller_predicted_savings_per_rollout (closes registry→controller pipeline).

The iter-67 row 78 controller counterfactual produced a 60-row summary TSV
(4 methods x 3 triggers x 5 thresholds) of paired-step bootstrap estimates of
"saved/fire", "savings_per_rollout", and "cost_ratio" for the adaptive-G
controller on the N2 same-stack corpus. This script lifts that summary
into the registry as a new additive-optional block so a downstream
consumer can read controller predictions from the registry without
re-running iter-67.

Schema design (mirrors iter-28 ci_method, iter-62 outcomes.coverage,
iter-66 zvf_antiherding / measured_yield_residual patterns):
- All fields nullable, additionalProperties: false
- Backward-compatible: every existing entry validates unchanged
- Lives at:
    stack_record.outcomes.controller_predicted_savings_per_rollout
    variant_delta_record.controller_predicted_savings_per_rollout
- audit_source / audit_date stamp the producer script
"""
import argparse
import csv
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
WORKTREE = HERE.parent.parent
REGISTRY = WORKTREE / "registry" / "entries"
SUMMARY = WORKTREE / "experiments" / "results" / "p5p8" / "p7_antiherding_controller_cf_summary.tsv"
OUT_TSV = WORKTREE / "experiments" / "results" / "p5p8" / "p6_controller_predicted_savings.tsv"
OUT_JSON = WORKTREE / "experiments" / "results" / "p5p8" / "p6_controller_predicted_savings_summary.json"
SCHEMA = WORKTREE / "registry" / "schema.json"
AUDIT_DATE = "2026-07-05"
AUDIT_SOURCE = "platform_modal/scripts/p5p8/p6_controller_predicted_savings.py"

# Same N2 stack panel the iter-67 summary was produced on
PANEL = "n2_same_stack_40step"
G = 8
N_STEPS = 40

STACK_TO_ENTRY = {
    "grpo": "tinker_grpo_qwen3.5-4b_gsm8k",
    "aero": "tinker_aero_qwen3.5-4b_gsm8k",
    "areal": "tinker_areal_qwen3.5-4b_gsm8k",
    "gift": "tinker_gift_qwen3.5-4b_gsm8k",
}
DELTA_TO_ENTRY = {
    "aero": "delta_aero",
    "areal": "delta_areal",
    "gift": "delta_gift",
}


def load_summary():
    """Return list of dicts from the iter-67 summary TSV (60 rows)."""
    rows = []
    with SUMMARY.open() as f:
        rd = csv.DictReader(f, delimiter="\t")
        for r in rd:
            rows.append({k: v for k, v in r.items()})
    return rows


def per_method_rows(rows, method):
    return [r for r in rows if r["method"] == method]


def to_controller_block(rows_for_method):
    """Convert iter-67 TSV rows for one method → registry block (array)."""
    block = []
    for r in rows_for_method:
        block.append({
            "trigger": r["controller"],
            "threshold": float(r["threshold"]),
            "fires": int(r["fires"]),
            "saved": int(r["saved"]),
            "missed": int(r["missed"]),
            "saved_per_fire": float(r["saved_per_fire"]),
            "savings_per_rollout_pt": float(r["savings_per_rollout_pt"]),
            "savings_per_rollout_lo": float(r["savings_per_rollout_lo"]),
            "savings_per_rollout_hi": float(r["savings_per_rollout_hi"]),
            "cost_ratio_pt": float(r["cost_ratio_pt"]),
        })
    return block


def build_block_with_meta(rows_for_method):
    """Wrap the controller rows with audit metadata."""
    return {
        "panel": PANEL,
        "G": G,
        "n_steps": N_STEPS,
        "predictions": to_controller_block(rows_for_method),
        "audit_source": AUDIT_SOURCE,
        "audit_date": AUDIT_DATE,
    }


def patch_entry(entry_path, where, block):
    """Patch a single entry JSON. `where` is 'outcomes' (stack) or top-level (delta)."""
    data = json.loads(entry_path.read_text())
    if where == "outcomes":
        outcomes = data.get("outcomes") or {}
        outcomes["controller_predicted_savings_per_rollout"] = block
        data["outcomes"] = outcomes
    else:
        data["controller_predicted_savings_per_rollout"] = block
    entry_path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


def write_outputs(all_per_method):
    """Write the iter-70 outputs."""
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["method", "trigger", "threshold", "fires", "saved", "missed",
            "saved_per_fire", "savings_per_rollout_pt",
            "savings_per_rollout_lo", "savings_per_rollout_hi",
            "cost_ratio_pt"]
    with OUT_TSV.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for method, rows in all_per_method.items():
            for r in rows:
                row = {
                    "method": method,
                    "trigger": r["controller"],
                    "threshold": r["threshold"],
                    "fires": r["fires"],
                    "saved": r["saved"],
                    "missed": r["missed"],
                    "saved_per_fire": r["saved_per_fire"],
                    "savings_per_rollout_pt": r["savings_per_rollout_pt"],
                    "savings_per_rollout_lo": r["savings_per_rollout_lo"],
                    "savings_per_rollout_hi": r["savings_per_rollout_hi"],
                    "cost_ratio_pt": r["cost_ratio_pt"],
                }
                f.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")
    summary = {
        "panel": PANEL,
        "G": G,
        "n_steps": N_STEPS,
        "n_methods": len(all_per_method),
        "n_predictions_per_method": [len(v) for v in all_per_method.values()],
        "audit_source": AUDIT_SOURCE,
        "audit_date": AUDIT_DATE,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")


def patch_schema():
    """Add controller_predicted_savings_per_rollout to both record types.

    For stack_record.outcomes: add the property (additive, all-nullable).
    For variant_delta_record: add it as a sibling of measured_yield_residual.
    Uses the same additive-optional pattern as iter-28/62/66.
    """
    s = json.loads(SCHEMA.read_text())
    pred_props = {
        "panel": {"$ref": "#/$defs/nullable_string"},
        "G": {"$ref": "#/$defs/nullable_integer"},
        "n_steps": {"$ref": "#/$defs/nullable_integer"},
        "audit_source": {"$ref": "#/$defs/nullable_string"},
        "audit_date": {"$ref": "#/$defs/nullable_string"},
        "predictions": {
            "type": ["array", "null"],
            "description": "Per-(trigger, threshold) controller prediction. "
                           "Each element: trigger (str), threshold (num), "
                           "fires/saved/missed (int), saved_per_fire (num), "
                           "savings_per_rollout_pt/lo/hi (num, bootstrap "
                           "percentile), cost_ratio_pt (num). Source: "
                           "iter-67 p7_antiherding_controller_cf_summary.tsv.",
            "items": {
                "type": "object",
                "properties": {
                    "trigger": {"type": "string"},
                    "threshold": {"type": "number"},
                    "fires": {"type": "integer"},
                    "saved": {"type": "integer"},
                    "missed": {"type": "integer"},
                    "saved_per_fire": {"type": "number"},
                    "savings_per_rollout_pt": {"type": "number"},
                    "savings_per_rollout_lo": {"type": "number"},
                    "savings_per_rollout_hi": {"type": "number"},
                    "cost_ratio_pt": {"type": "number"},
                },
                "additionalProperties": False,
            },
        },
    }
    # Patch stack_record.outcomes
    stack_props = s["$defs"]["stack_record"]["properties"]["outcomes"]["properties"]
    if "controller_predicted_savings_per_rollout" not in stack_props:
        stack_props["controller_predicted_savings_per_rollout"] = {
            "type": ["object", "null"],
            "description": "Iter-70: per-(method × trigger × threshold) "
                           "controller savings predictions for the iter-51 "
                           "adaptive-G controller on this stack. All fields "
                           "nullable; null = not yet audited. Sourced from "
                           "iter-67 p7_antiherding_controller_cf_summary.tsv.",
            "properties": pred_props,
            "additionalProperties": False,
        }
    # Patch variant_delta_record (sibling of measured_yield_residual)
    delta_props = s["$defs"]["variant_delta_record"]["properties"]
    if "controller_predicted_savings_per_rollout" not in delta_props:
        delta_props["controller_predicted_savings_per_rollout"] = {
            "type": ["object", "null"],
            "description": "Iter-70: per-(trigger × threshold) controller "
                           "savings predictions for the variant on the N2 "
                           "same-stack corpus. Inherits from the variant's "
                           "measured method; predictions carry the iter-67 "
                           "saved_per_fire and savings_per_rollout paired-"
                           "bootstrap estimates.",
            "properties": pred_props,
            "additionalProperties": False,
        }
    SCHEMA.write_text(json.dumps(s, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Patch registry entries + schema (default: dry-run)")
    ap.add_argument("--validate", action="store_true",
                    help="Run canonical validator after patching")
    args = ap.parse_args()

    rows = load_summary()
    if len(rows) != 60:
        print(f"WARN: expected 60 rows from iter-67 summary, got {len(rows)}",
              file=sys.stderr)
    per_method = {m: per_method_rows(rows, m) for m in STACK_TO_ENTRY}

    # Validate each method has 15 predictions (3 triggers x 5 thresholds)
    for m, rs in per_method.items():
        if len(rs) != 15:
            print(f"WARN: method {m} has {len(rs)} rows, expected 15",
                  file=sys.stderr)

    write_outputs(per_method)

    if args.apply:
        # Patch schema
        patch_schema()
        # Patch stack entries (4)
        for method, eid in STACK_TO_ENTRY.items():
            block = build_block_with_meta(per_method[method])
            patch_entry(REGISTRY / f"{eid}.json", "outcomes", block)
        # Patch variant-delta entries (3: aero, areal, gift)
        for method, eid in DELTA_TO_ENTRY.items():
            block = build_block_with_meta(per_method[method])
            patch_entry(REGISTRY / f"{eid}.json", "top", block)
        print(f"Patched schema + 4 stack entries + 3 variant-delta entries.")
    else:
        print("Dry-run: would patch schema + 4 stack entries + 3 variant-delta entries.")
        print(f"Wrote {OUT_TSV.name} + {OUT_JSON.name}")

    if args.validate:
        # Run canonical validator
        import subprocess
        r = subprocess.run([sys.executable, "registry_validate.py"],
                           cwd=str(WORKTREE), capture_output=True, text=True)
        print(r.stdout[-2000:])
        print(r.stderr[-1000:])


if __name__ == "__main__":
    main()
