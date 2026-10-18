#!/usr/bin/env python3
"""Propose review17-v2 method-specific counterfactual controller savings.
Requires a complete unique method/threshold panel. Savings units are per rollout.
No keyed bootstrap evidence is supplied, so uncertainty stays explicitly unverified.
Does not claim realized financial savings or overwrite the registry."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))  # platform_modal/scripts, for _paths
from review17_contract import validate_proposals
from _paths import REPO_ROOT

ROOT = REPO_ROOT
RES = ROOT / "platform_hybrid/experiments" / "results" / "p5p8"
OUT = RES / "review17_v2"
REG = ROOT / "platform_hybrid/registry"
ENTRIES = REG / "entries"

METHODS = ["grpo", "aero", "areal", "gift"]
TAUS = [0.03, 0.04, 0.05, 0.06, 0.07]
SEED = 20260705


def load_joint_tsv():
    rows = []
    with (RES / "p7_joint_controller.tsv").open() as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            rows.append(r)
    return rows


def build_predictions():
    """One prediction entry per (method, tau)."""
    out = []
    seen = set()
    for r in load_joint_tsv():
        m = r["method"]
        t = float(r["tau"])
        if m not in METHODS or t not in TAUS:
            raise ValueError("Unexpected method/threshold")
        if (m, t) in seen:
            raise ValueError("Duplicate method/threshold")
        seen.add((m, t))
        # Joint controller breakdown: dualformer rollout saves + zvf ddiv-saved
        # (rollout_saves is the Dualformer G'=2 contribution on contrast prompts;
        #  zvf_saved is the ddiv-triage contribution on fired steps)
        n_contrast = int(r["n_contrast"])
        n_fired = int(r["n_fired_steps"])
        n_zvf_saved = int(r["n_zvf_saved"])
        n_rollout_saves = int(r["rollout_saves"])
        net_saves = int(r["net_saves"])
        g_total = int(r["g_total"])
        cost_ratio = float(r["cost_ratio"])
        # Per-rollout net_saves = (rollout_saves + zvf_saved) / g_total
        # (already what iter-70 row 82 normalises to "savings_per_rollout_pt")
        if g_total <= 0 or not math.isfinite(cost_ratio):
            raise ValueError("Invalid rollout denominator/cost ratio")
        per_rollout_pt = net_saves / g_total
        # Bootstrap CI from iter-72 row 85 boot file
        out.append(
            {
                "method": m,
                "evidence_status": "COUNTERFACTUAL_PREDICTED",
                "ci_method": None,
                "uncertainty_status": "UNVERIFIED_NO_KEYED_BOOTSTRAP",
                "trigger": "joint_dualformer_ddiv",
                "threshold": t,
                "n_contrast_prompts": n_contrast,
                "n_fired_steps": n_fired,
                "n_zvf_saved": n_zvf_saved,
                "n_rollout_saves": n_rollout_saves,
                "net_saves": net_saves,
                "g_total": g_total,
                "cost_ratio_pt": cost_ratio,
                "savings_per_rollout_pt": per_rollout_pt,
                "source_iter": "iter-72 row 85 joint controller",
            }
        )
    # Sort by (method, tau)
    expected = {(m, t) for m in METHODS for t in TAUS}
    if seen != expected:
        raise ValueError("Incomplete method/threshold panel")
    out.sort(key=lambda x: (x["method"], x["threshold"]))
    return out


def main():
    print("# === P5P8 SYNTH / P6 JOB B: joint-controller registry extension (iter 76) ===")
    preds = build_predictions()
    OUT.mkdir(parents=True, exist_ok=True)
    proposed = {}
    print(f"# assembled {len(preds)} joint-controller predictions")

    # Write TSV + JSON artifacts
    if preds:
        keys = list(preds[0].keys())
        lines = ["\t".join(keys)]
        for p in preds:
            lines.append(
                "\t".join(f"{p[k]:.6g}" if isinstance(p[k], float) else str(p[k]) for k in keys)
            )
        (OUT / "p6_joint_controller_extension.tsv").write_text("\n".join(lines) + "\n")
    (OUT / "p6_joint_controller_extension.json").write_text(
        json.dumps(
            {
                "n_predictions": len(preds),
                "predictions": preds,
                "panel": "n2_same_stack_40step_joint",
                "source_iter": "iter-72 row 85",
                "audit_source": "platform_modal/scripts/p5p8/p6_controller_joint_extension.py",
                "audit_date": "2026-07-05",
            },
            indent=2,
            default=str,
        )
    )

    # Patch the 4 stack entries with `joint_controller_predictions` block
    for m in METHODS:
        path = ENTRIES / f"tinker_{m}_qwen3.5-4b_gsm8k.json"
        e = json.loads(path.read_text())
        out = e.get("outcomes", {}) or {}
        # collect predictions by method
        method_preds = [p for p in preds if p["method"] == m]
        out["joint_controller_predictions"] = {
            "panel": "n2_same_stack_40step_joint",
            "G": 8,
            "n_steps": 40,
            "predictions": method_preds,
            "audit_source": "platform_modal/scripts/p5p8/p6_controller_joint_extension.py",
            "audit_date": "2026-07-05",
            "ci_method": None,
            "uncertainty_status": "UNVERIFIED_NO_KEYED_BOOTSTRAP",
        }
        e["outcomes"] = out
        proposed[path.stem] = e
        print(
            f"# proposed stack entry {path.name}: +joint_controller_predictions ({len(method_preds)} entries)"
        )

    # Patch the 3 delta entries with a `joint_controller_cost_ratio_per_tau` field
    for m in ["aero", "areal", "gift"]:
        path = ENTRIES / f"delta_{m}.json"
        e = json.loads(path.read_text())
        existing = e.get("controller_predicted_savings_per_rollout") or {}
        method_preds = [p for p in preds if p["method"] == m]
        # Append joint-controller fields to the existing schema-bounded block
        if not isinstance(existing, dict):
            existing = {}
        existing["joint_controller"] = {
            "panel": "n2_same_stack_40step_joint",
            "predictions": method_preds,
            "audit_source": "platform_modal/scripts/p5p8/p6_controller_joint_extension.py",
            "audit_date": "2026-07-05",
            "ci_method": None,
            "uncertainty_status": "UNVERIFIED_NO_KEYED_BOOTSTRAP",
        }
        e["controller_predicted_savings_per_rollout"] = existing
        # Clean up any stray 'outcomes' field that may have been set incorrectly
        # Preserve unrelated existing fields.
        proposed[path.stem] = e
        print(
            f"# proposed delta entry {path.name}: +joint_controller in controller_predicted_savings_per_rollout ({len(method_preds)} entries)"
        )

    validate_proposals(proposed, REG / "schema.json")
    (OUT / "p6_joint_controller_proposed_entries.json").write_text(
        json.dumps(proposed, indent=2, allow_nan=False) + "\n"
    )
    print("# === iter 76 JOB B / SYNTH complete ===")


if __name__ == "__main__":
    main()
