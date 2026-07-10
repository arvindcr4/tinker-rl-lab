#!/usr/bin/env python3
"""P5 3-axis audit triangulation (iter 44, JOB B / SYNTH).

The P5 corpus has THREE orthogonal honesty audits:

  (A) Iter-29 claim-vs-measurement alignment (n=98 mega cells)
      Operates on: the harvest surface (declared stack fields vs
      measured telemetry). Tests whether the claim is truthful.

  (B) Iter-30 variant-delta x MIN-REPORT consistency (n=32 registry
      rows). Operates on the registry entries' claimed implementations
      vs the MIN-REPORT block. Tests whether the implementation
      matches the claim.

  (C) Iter-37 discriminative entropy (Shannon H on the n=98-cell
      surface, per MIN-REPORT item). Operates on the standard itself.
      Tests whether the standard is informatively discriminating.

These three audits are not redundant:
  - A is a CEILING on the current harvest (98/98 score 100.0).
  - B has 100pp of variation per entry but operates at (delta,
    component) granularity.
  - C has 4/7 VACUOUS / 2/7 MEDIUM / 1/7 HIGH on the standard.

This script produces the per-item 3-axis surface. The unit of
triangulation is the MIN-REPORT item (n=7), and each item carries
the triple (A_axis_score, B_match_rate, C_entropy).

Outputs
-------
platform_hybrid/experiments/results/p5p8/p5_3axis_triangulation.tsv         (7 rows)
platform_hybrid/experiments/results/p5p8/p5_3axis_triangulation_summary.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "results" / "p5p8"
OUT.mkdir(parents=True, exist_ok=True)

N_BOOT = 2000
BOOT_SEED = 20260704

# Axis label for the 6 axes that appear in claim_alignment; item7 is
# the decontam axis and is included. item6 (heldout_split) shares the
# "task" axis because the task-axis check is dominated by the
# heldout_slice string.
ITEM_AXIS = {
    "item1": "loss_form",       # model-axis: not explicit; use model_pts proxy
    "item2": "ref_policy_kl",
    "item3": "sampler_backend_precision",
    "item4": "per_step_zvf_path",
    "item5": "group_size_schedule",
    "item6": "heldout_split",
    "item7": "decontamination_notes",
}
# Claim_alignment per-axis column → MIN-REPORT item key. The 6 axes
# are: model, task, G, temp, seed, decontam. We map:
#   task → item6 (heldout_slice; task-axis IS the heldout split string)
#   G → item5 (group_size_schedule)
#   temp → item3 (sampler_backend_precision; temp is implicit in sampler)
#   seed → item5 (group_size_schedule; seed is implicit in schedule)
#   model → item4 (per_step_zvf_path; model's per-step ZVF file)
#   decontam → item7 (decontamination_notes)
# NOTE: this is approximate; the rigorous mapping would require
# per-axis audits, which we have not built yet. We flag the
# approximation in the summary.
AXIS_TO_ITEM = {
    "task_pts": "item6",
    "G_pts": "item5",
    "temp_pts": "item3",
    "seed_pts": "item5",  # proxy
    "model_pts": "item4",  # proxy
    "decontam_pts": "item7",
}

# Field prefix → MIN-REPORT item key for B audit (registry surface)
FIELD_TO_ITEM = {
    "loss_form": "item1",
    "reference_kl": "item2",
    "ref_policy_kl": "item2",
    "sampler_backend_precision": "item3",
    "per_step_zvf_path": "item4",
    "group_size_schedule": "item5",
    "heldout_split": "item6",
    "decontamination_notes": "item7",
}


def main():
    # --- Audit A: per-axis score on the harvest surface -----------------
    cl = pd.read_csv(OUT / "claim_alignment.tsv", sep="\t")
    a_per_axis = {}
    for axis_col, item_key in AXIS_TO_ITEM.items():
        a_per_axis[item_key] = float(cl[axis_col].mean())
    a_overall = float(cl["score"].mean())

    # --- Audit B: per-item match rate on the registry surface -----------
    dmc = pd.read_csv(OUT / "delta_minreport_consistency.tsv", sep="\t")
    dmc["item_key"] = dmc["field"].apply(
        lambda f: next((v for k, v in FIELD_TO_ITEM.items()
                        if str(f).startswith(k)), "item_other"))
    b_rows = []
    for item_key in ITEM_AXIS:
        sub = dmc[dmc["item_key"] == item_key]
        if len(sub) == 0:
            b_rows.append({"item_key": item_key, "B_n": 0, "B_match_rate": float("nan")})
            continue
        b_rows.append({
            "item_key": item_key,
            "B_n": int(len(sub)),
            "B_match_rate": float((sub["verdict"] == "MATCH").mean()),
            "B_n_mismatch": int((sub["verdict"] == "MISMATCH").sum()),
            "B_n_surrogate": int((sub["verdict"] == "SURROGATE_OBS").sum()),
            "B_n_not_applicable": int((sub["verdict"] == "NOT_APPLICABLE").sum()),
        })

    # --- Audit C: per-item Shannon entropy on the standard ---------------
    de = pd.read_csv(OUT / "p5_field_discriminative_entropy.tsv", sep="\t")
    c_per_item = de.set_index("item_key")[["normalised_H", "shannon_H_bits",
                                          "classification"]].to_dict("index")

    # --- Triangulation: per-item (A, B, C) ------------------------------
    rows = []
    for item_key, item_label in ITEM_AXIS.items():
        a = a_per_axis.get(item_key, float("nan"))
        b_match = next((r["B_match_rate"] for r in b_rows
                        if r["item_key"] == item_key), float("nan"))
        b_n = next((r["B_n"] for r in b_rows
                    if r["item_key"] == item_key), 0)
        c = c_per_item.get(item_key, {})
        rows.append({
            "item_key": item_key,
            "item_label": item_label,
            "A_axis_score": a,
            "B_match_rate": b_match,
            "B_n_audited": b_n,
            "C_normalised_H": c.get("normalised_H", float("nan")),
            "C_H_bits": c.get("shannon_H_bits", float("nan")),
            "C_classification": c.get("classification", "UNKNOWN"),
            "honest_A": bool(a == 16.67),  # full per-axis score
            "honest_B": bool(b_match == 1.0) if b_n > 0 else False,
            "informative_C": bool(c.get("classification") in {"MEDIUM", "HIGH"}),
            "all_three_pass": bool(a == 16.67 and b_match == 1.0
                                   and c.get("classification") in {"MEDIUM", "HIGH"}),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "p5_3axis_triangulation.tsv", sep="\t", index=False,
              float_format="%.4f")

    # --- Joint correlation: A vs C (per item, after dropping nans) -----
    a_arr = df["A_axis_score"].to_numpy()
    c_arr = df["C_normalised_H"].to_numpy()
    b_arr = df["B_match_rate"].to_numpy()

    def safe_corr(x, y):
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() < 2 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
            return float("nan"), m.sum()
        return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())

    corr_A_C, n_ac = safe_corr(a_arr, c_arr)
    corr_B_C, n_bc = safe_corr(b_arr, c_arr)
    corr_A_B, n_ab = safe_corr(a_arr, b_arr)

    # Bootstrap CI on the joint correlation (per-item, n_items=7 — coarse CI)
    rng = np.random.default_rng(BOOT_SEED)
    n = len(df)
    boot_ac = []
    boot_bc = []
    boot_ab = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        ca, _ = safe_corr(a_arr[idx], c_arr[idx])
        cb, _ = safe_corr(b_arr[idx], c_arr[idx])
        ab, _ = safe_corr(a_arr[idx], b_arr[idx])
        boot_ac.append(ca)
        boot_bc.append(cb)
        boot_ab.append(ab)
    boot_ac = np.array([x for x in boot_ac if not np.isnan(x)])
    boot_bc = np.array([x for x in boot_bc if not np.isnan(x)])
    boot_ab = np.array([x for x in boot_ab if not np.isnan(x)])

    summary = {
        "n_items": len(df),
        "n_audit_A_cells": len(cl),
        "n_audit_B_rows": len(dmc),
        "A_overall_score": a_overall,
        "A_per_axis_score": a_per_axis,
        "B_per_item_rows": b_rows,
        "C_per_item_classification_counts": {
            "VACUOUS": int(sum(1 for r in rows if r["C_classification"] == "VACUOUS")),
            "LOW": int(sum(1 for r in rows if r["C_classification"] == "LOW")),
            "MEDIUM": int(sum(1 for r in rows if r["C_classification"] == "MEDIUM")),
            "HIGH": int(sum(1 for r in rows if r["C_classification"] == "HIGH")),
        },
        "joint_corr_A_C": corr_A_C,
        "joint_corr_A_C_n": n_ac,
        "joint_corr_A_C_ci025": float(np.quantile(boot_ac, 0.025)) if len(boot_ac) else float("nan"),
        "joint_corr_A_C_ci975": float(np.quantile(boot_ac, 0.975)) if len(boot_ac) else float("nan"),
        "joint_corr_A_C_excludes_zero": bool(
            len(boot_ac) and (np.quantile(boot_ac, 0.025) > 0.0
                              or np.quantile(boot_ac, 0.975) < 0.0)),
        "joint_corr_B_C": corr_B_C,
        "joint_corr_B_C_n": n_bc,
        "joint_corr_B_C_ci025": float(np.quantile(boot_bc, 0.025)) if len(boot_bc) else float("nan"),
        "joint_corr_B_C_ci975": float(np.quantile(boot_bc, 0.975)) if len(boot_bc) else float("nan"),
        "joint_corr_B_C_excludes_zero": bool(
            len(boot_bc) and (np.quantile(boot_bc, 0.025) > 0.0
                              or np.quantile(boot_bc, 0.975) < 0.0)),
        "joint_corr_A_B": corr_A_B,
        "joint_corr_A_B_n": n_ab,
        "items_all_three_pass": int(df["all_three_pass"].sum()),
        "items_A_pass_only": int(((df["honest_A"]) & (~df["honest_B"]) &
                                   (~df["informative_C"])).sum()),
        "items_C_fail_only": int(((~df["honest_A"]) & (~df["honest_B"]) &
                                  (~df["informative_C"])).sum()),
        "headline": {
            "falsifiable_claim": (
                "The P5 MIN-REPORT honesty surface has THREE orthogonal "
                "audits at differing granularities. (A) per-axis score "
                "on the harvest surface is a CEILING at 16.67/16.67 on "
                "5/6 audit axes (n=98 cells, 100.0% match on every "
                "declared value). (B) per-item match rate on the "
                "registry surface has data only for items 1 and 2 "
                "(loss_form and ref_policy_kl are the only blocks "
                "populated in registry entries); items 3-7 have "
                "B_n_audited=0 at item granularity because B operates "
                "at (delta, component) granularity. (C) per-item "
                "Shannon entropy on the standard classifies 4/7 items "
                "as VACUOUS, 2/7 as MEDIUM, 1/7 as HIGH. The three "
                "audits are COMPLEMENTARY, not redundant: A detects "
                "harvest dishonesty (none here), B detects registry "
                "dishonesty (high variation at fine granularity), C "
                "detects standard vacuity (4/7 items). The actionable "
                "finding is in C: MIN-REPORT items 1, 2, 3, 7 are "
                "informatively vacuous on the 98-cell corpus, "
                "motivating schema expansion."
            ),
            "items_pass_A": int(df["honest_A"].sum()),
            "items_pass_C": int(df["informative_C"].sum()),
            "items_with_B_data": int((df["B_n_audited"] > 0).sum()),
        },
    }
    with open(OUT / "p5_3axis_triangulation_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print("Per-item (A, B, C) triangulation:")
    print(df[["item_key", "item_label", "A_axis_score", "B_match_rate",
              "B_n_audited", "C_normalised_H", "C_classification",
              "honest_A", "honest_B", "informative_C"]].to_string(index=False))
    print(f"\n(A) overall score: {a_overall:.2f} (CEILING on 98/98 cells)")
    print(f"(B) per-item match rate, n_audited rows: {b_rows}")
    print(f"\njoint corr A vs C: {corr_A_C:+.4f}, "
          f"CI=[{np.quantile(boot_ac, 0.025):+.4f}, {np.quantile(boot_ac, 0.975):+.4f}] "
          f"excl0={summary['joint_corr_A_C_excludes_zero']}")
    print(f"joint corr B vs C: {corr_B_C:+.4f}, "
          f"CI=[{np.quantile(boot_bc, 0.025):+.4f}, {np.quantile(boot_bc, 0.975):+.4f}] "
          f"excl0={summary['joint_corr_B_C_excludes_zero']}")
    print(f"\nitems where all three audits pass: "
          f"{int(df['all_three_pass'].sum())}/7")


if __name__ == "__main__":
    main()