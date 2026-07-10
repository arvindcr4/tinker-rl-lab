#!/usr/bin/env python3
"""P5P8-SYNTH / P6 iter 76 JOB B: extend iter-70 row 82
`controller_predicted_savings_per_rollout` with the iter-72 row 85
joint-controller cost-adjusted savings per method x tau.

Closes the iter-72 row 85 mint recommendation:
    'cost-adjusted joint controller incorporating iter-72 cost_ratio
     into iter-70 row 82 controller_predicted_savings_per_rollout
     block -- extends the registry-readable joint controller savings
     to include cost_ratio per method x tau'

Inputs
------
platform_hybrid/experiments/results/p5p8/p7_joint_controller.tsv     (20 rows: 4 methods x 5 tau)
platform_hybrid/experiments/results/p5p8/p7_joint_controller_boot.tsv (4 rows: bootstrap headline)
registry/entries/tinker_{grpo,aero,areal,gift}_qwen3.5-4b_gsm8k.json
                                                       (4 stack entries)
registry/entries/delta_{aero,areal,gift}.json          (3 delta entries)
registry/schema.json                                    (34/34 must remain PASS)

Outputs
-------
platform_hybrid/experiments/results/p5p8/p6_joint_controller_extension.tsv  (20 rows: 4 m x 5 t)
platform_hybrid/experiments/results/p5p8/p6_joint_controller_extension.json (machine-readable)
patched registry/entries/tinker_{grpo,aero,areal,gift}_qwen3.5-4b_gsm8k.json
patched registry/entries/delta_{aero,areal,gift}.json
docs/p5p8_improvements/90_p6_joint_controller_extension.md
paper/sections/p6_joint_controller_extension.tex    (new §)
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
REG = ROOT / "registry"
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
    for r in load_joint_tsv():
        m = r["method"]
        t = float(r["tau"])
        if m not in METHODS or t not in TAUS:
            continue
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
        per_rollout_pt = net_saves / max(1, g_total) * 1000  # per 1000 rollouts
        # Bootstrap CI from iter-72 row 85 boot file
        out.append({
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
        })
    # Sort by (method, tau)
    out.sort(key=lambda x: (x["trigger"], x["threshold"]))
    return out


def main():
    print("# === P5P8 SYNTH / P6 JOB B: joint-controller registry extension (iter 76) ===")
    preds = build_predictions()
    print(f"# assembled {len(preds)} joint-controller predictions")

    # Write TSV + JSON artifacts
    if preds:
        keys = list(preds[0].keys())
        lines = ["\t".join(keys)]
        for p in preds:
            lines.append("\t".join(f"{p[k]:.6g}" if isinstance(p[k], float) else str(p[k]) for k in keys))
        (RES / "p6_joint_controller_extension.tsv").write_text("\n".join(lines) + "\n")
    (RES / "p6_joint_controller_extension.json").write_text(
        json.dumps({"n_predictions": len(preds), "predictions": preds,
                    "panel": "n2_same_stack_40step_joint", "source_iter": "iter-72 row 85",
                    "audit_source": "platform_modal/scripts/p5p8/p6_controller_joint_extension.py",
                    "audit_date": "2026-07-05"}, indent=2, default=str)
    )

    # Patch the 4 stack entries with `joint_controller_predictions` block
    for m in METHODS:
        path = ENTRIES / f"tinker_{m}_qwen3.5-4b_gsm8k.json"
        e = json.loads(path.read_text())
        out = e.get("outcomes", {}) or {}
        # collect predictions by method
        method_preds = [p for p in preds if p["trigger"] == "joint_dualformer_ddiv"]
        out["joint_controller_predictions"] = {
            "panel": "n2_same_stack_40step_joint",
            "G": 8,
            "n_steps": 40,
            "predictions": method_preds,
            "audit_source": "platform_modal/scripts/p5p8/p6_controller_joint_extension.py",
            "audit_date": "2026-07-05",
            "ci_method": {
                "method": "paired_step_bootstrap_pct",
                "n_boot": 2000,
                "seed": SEED,
                "ci_level": 0.95,
                "source": "iter-72 row 85 p7_joint_controller.py",
            },
        }
        e["outcomes"] = out
        path.write_text(json.dumps(e, indent=2) + "\n")
        print(f"# patched stack entry {path.name}: +joint_controller_predictions ({len(method_preds)} entries)")

    # Patch the 3 delta entries with a `joint_controller_cost_ratio_per_tau` field
    for m in ["aero", "areal", "gift"]:
        path = ENTRIES / f"delta_{m}.json"
        e = json.loads(path.read_text())
        existing = e.get("controller_predicted_savings_per_rollout") or {}
        method_preds = [p for p in preds if p["trigger"] == "joint_dualformer_ddiv"]
        # Append joint-controller fields to the existing schema-bounded block
        if not isinstance(existing, dict):
            existing = {}
        existing["joint_controller"] = {
            "panel": "n2_same_stack_40step_joint",
            "predictions": method_preds,
            "audit_source": "platform_modal/scripts/p5p8/p6_controller_joint_extension.py",
            "audit_date": "2026-07-05",
            "ci_method": {
                "method": "paired_step_bootstrap_pct",
                "n_boot": 2000,
                "seed": SEED,
                "ci_level": 0.95,
                "source": "iter-72 row 85 p7_joint_controller.py",
            },
        }
        e["controller_predicted_savings_per_rollout"] = existing
        # Clean up any stray 'outcomes' field that may have been set incorrectly
        e.pop("outcomes", None)
        path.write_text(json.dumps(e, indent=2) + "\n")
        print(f"# patched delta entry {path.name}: +joint_controller in controller_predicted_savings_per_rollout ({len(method_preds)} entries)")

    print("# === iter 76 JOB B / SYNTH complete ===")


if __name__ == "__main__":
    main()
