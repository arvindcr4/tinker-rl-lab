#!/usr/bin/env python3
"""P6 JOB B (iter 128): patch the 9 delta_*.json entries with recomputed
mean_zvf CIs from the iter-128 recompute run.

For each of the 9 GRPO-family methods, find the `mean_zvf` measured row
at panel zvf130_5seed, replace:
  - ci_low, ci_high  (with the bootstrap CI from iter-128 recompute)
  - ci_method         (from "point_no_perseed_sd" to
                       "bootstrap_paired_5seed")
and add a comment with the recompute provenance.

This is the literal operational recommendation (a) from iter-114 row
128: "recompute the 8 POINT_ONLY mean_zvf CIs from raw zvf_iter130*.tsv
(single 80-LoC script, zero harvest cost) — closes 24% of the gap".

Stdlib only. Idempotent.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"

DELTA_IDS = [
    "delta_aero", "delta_areal", "delta_cppo", "delta_es", "delta_gift",
    "delta_grpo", "delta_mcgrpo", "delta_ngrpo", "delta_scafgrpo",
]


def main():
    print("# === P6 JOB B (iter 128): patch delta_*.json with recomputed CIs ===")
    # Load recompute summary
    summary = json.loads(
        (RES / "p6_iter128_recompute_point_only_summary.json").read_text()
    )
    recomp_by_id = {r["delta_id"]: r for r in summary["recomputed"]}

    n_patched = 0
    n_total = 0
    n_ci_method_changed = 0
    for delta_id in DELTA_IDS:
        fp = ROOT / "registry" / "entries" / f"{delta_id}.json"
        if not fp.exists():
            print(f"# SKIP {delta_id}: not found")
            continue
        n_total += 1
        d = json.loads(fp.read_text())
        rec = recomp_by_id.get(delta_id)
        if rec is None:
            print(f"# SKIP {delta_id}: no recompute result")
            continue
        n_measured_changed = 0
        measured = d.get("measured", [])
        for row in measured:
            if (row.get("metric") == "mean_zvf"
                    and row.get("panel") == "zvf130_5seed"):
                old_method_obj = row.get("ci_method")
                old_method_name = (old_method_obj or {}).get("method") if isinstance(old_method_obj, dict) else None
                row["ci_low"] = round(rec["recomputed_delta_ci_lo"], 6)
                row["ci_high"] = round(rec["recomputed_delta_ci_hi"], 6)
                row["ci_method"] = {
                    "method": "bootstrap_paired_5seed",
                    "n_boot": 2000,
                    "seed": 20260705,
                    "ci_level": 0.95,
                    "source": "scripts/p5p8/p6_iter128_recompute_point_only.py"
                }
                row["iter_recomputed"] = 128
                if old_method_name == "point_no_perseed_sd":
                    n_ci_method_changed += 1
                n_measured_changed += 1
        # Add provenance comment at the entry root level
        d["iter128_recompute_note"] = (
            "Patched by iter 128 JOB B: recomputed mean_zvf CI on zvf130_5seed "
            "panel from raw zvf_iter130_risk_index.tsv (per-seed n=5) with "
            "paired bootstrap (B=2000, seed=20260705). Old ci_method="
            "point_no_perseed_sd (CI width=0) replaced with "
            "bootstrap_paired_5seed (mean CI width="
            f"{round(rec['recomputed_delta_ci_hi'] - rec['recomputed_delta_ci_lo'], 6)})."
        )
        # Write back
        fp.write_text(json.dumps(d, indent=2) + "\n")
        n_patched += 1
        print(f"# patched {delta_id}: "
              f"new_ci=[{rec['recomputed_delta_ci_lo']:.4f}, {rec['recomputed_delta_ci_hi']:.4f}] "
              f"({n_measured_changed} mean_zvf rows changed)")

    # Validate the patched entries with the canonical validator
    print(f"# summary: patched {n_patched}/{n_total} entries, "
          f"{n_ci_method_changed} ci_method upgraded point_no_perseed_sd->bootstrap_paired_5seed")

    # Cross-check that schema still passes (run registry_validate.py schema only)
    try:
        import subprocess
        result = subprocess.run(
            ["python3", str(ROOT / "scripts" / "p5p8" / "registry_validate.py"),
             "--schema-only"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60
        )
        schema_ok = "PASS" in result.stdout and result.returncode == 0
        print(f"# schema re-validation: {'PASS' if schema_ok else 'FAIL'}")
    except Exception as e:
        print(f"# schema re-validation: ERROR ({e})")


if __name__ == "__main__":
    main()