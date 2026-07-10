#!/usr/bin/env python3
"""P5/P6 JOB B (iter 36): triangulation of two independent MIN-REPORT audits.

The P5 corpus has TWO independent audits that measure different things on
overlapping surfaces:

(A) Iter-29 claim-vs-measurement alignment (n=98 mega cells, score 0-100)
    Operates on: the manifest surface (declared stack fields) vs measured
    telemetry. Tests whether the claim is *truthful*.

(B) Iter-30 variant-delta x MIN-REPORT consistency (n=32 registry rows)
    Operates on: the registry entries' claimed implementations vs the
    MIN-REPORT block. Tests whether the implementation matches the claim.

These two audits are not the same -- A is "what you said vs what you did"
on the harvest surface, B is "what you said vs what you wrote" on the
registry surface.

This script computes the CROSS-PAPER coupling:
  1. Per-entry registry match rate (B) -- is honest?
  2. Whether audit (A) is a ceiling (score=100 everywhere) -- is meaningful?
  3. Joint metric: correlate B_match_rate with per-entry MIN-REPORT coverage
     (n_audited_B) -- does more-audited entries have higher match?

Outputs
-------
platform_hybrid/experiments/results/p5p8/p5p6_audit_triangulation.tsv
platform_hybrid/experiments/results/p5p8/p5p6_audit_triangulation_summary.json

Stdlib + pandas. <=200 lines.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "results" / "p5p8"
OUT.mkdir(parents=True, exist_ok=True)

CL = OUT / "claim_alignment.tsv"
DMC = OUT / "delta_minreport_consistency.tsv"
N_BOOT = 2000
BOOT_SEED = 20260704


def main():
    cl = pd.read_csv(CL, sep="\t")
    dmc = pd.read_csv(DMC, sep="\t")
    print(f"Claim alignment (A): {len(cl)} cells, mean score={cl['score'].mean():.2f}")
    print(f"Variant-delta consistency (B): {len(dmc)} registry rows")

    # Per-entry match-rate from audit B
    entry_match = (
        dmc.groupby("entry_id")["verdict"]
        .apply(lambda s: (s == "MATCH").mean())
        .reset_index()
        .rename(columns={"verdict": "B_match_rate"})
    )
    n_audit = (
        dmc.groupby("entry_id")["verdict"].count().reset_index()
        .rename(columns={"verdict": "n_audited_B"})
    )
    n_mismatch = (
        dmc.groupby("entry_id")["verdict"]
        .apply(lambda s: (s == "MISMATCH").sum())
        .reset_index()
        .rename(columns={"verdict": "n_mismatch"})
    )
    n_surrogate = (
        dmc.groupby("entry_id")["verdict"]
        .apply(lambda s: (s == "SURROGATE_OBS").sum())
        .reset_index()
        .rename(columns={"verdict": "n_surrogate"})
    )
    entry_match = entry_match.merge(n_audit, on="entry_id")
    entry_match = entry_match.merge(n_mismatch, on="entry_id")
    entry_match = entry_match.merge(n_surrogate, on="entry_id")

    # Read registry entries for stack info (informative only)
    REG = ROOT / "registry" / "entries"
    entry_to_stack = {}
    for jf in sorted(REG.glob("*.json")):
        try:
            d = json.loads(jf.read_text())
            stack = d.get("stack", {})
            entry_to_stack[jf.stem] = {
                "model_family": stack.get("model_family"),
                "task_slice": stack.get("task_slice"),
            }
        except Exception:
            pass

    rows = []
    for _, r in entry_match.iterrows():
        eid = r["entry_id"]
        info = entry_to_stack.get(eid, {})
        rows.append({
            "entry_id": eid,
            "B_match_rate": r["B_match_rate"],
            "n_audited_B": r["n_audited_B"],
            "n_mismatch": r["n_mismatch"],
            "n_surrogate": r["n_surrogate"],
            "model_family": info.get("model_family"),
            "task_slice": info.get("task_slice"),
        })

    df = pd.DataFrame(rows).sort_values("B_match_rate", ascending=False)
    df.to_csv(OUT / "p5p6_audit_triangulation.tsv", sep="\t", index=False)

    # Joint metric: B_match_rate vs n_audited_B
    if df["B_match_rate"].std() > 0 and df["n_audited_B"].std() > 0:
        corr_match_vs_audited = float(
            df["B_match_rate"].corr(df["n_audited_B"])
        )
    else:
        corr_match_vs_audited = float("nan")

    rng = np.random.default_rng(BOOT_SEED)
    boot = []
    n = len(df)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        sub = df.iloc[idx]
        if sub["B_match_rate"].std() == 0 or sub["n_audited_B"].std() == 0:
            continue
        boot.append(sub["B_match_rate"].corr(sub["n_audited_B"]))
    boot = np.array(boot) if boot else np.array([corr_match_vs_audited])

    a_unique = int(cl["score"].nunique())
    a_ceiling = a_unique == 1 and float(cl["score"].iloc[0]) == 100.0

    summary = {
        "n_entries_audited_B": len(df),
        "n_cells_audited_A": len(cl),
        "B_match_rate_mean": float(df["B_match_rate"].mean()),
        "A_mean_score_overall": float(cl["score"].mean()),
        "A_unique_scores": a_unique,
        "A_is_ceiling": bool(a_ceiling),
        "joint_corr_B_match_vs_n_audited": corr_match_vs_audited,
        "joint_corr_ci025": float(np.quantile(boot, 0.025)),
        "joint_corr_ci975": float(np.quantile(boot, 0.975)),
        "joint_corr_excludes_zero": bool(
            np.quantile(boot, 0.025) > 0.0 or np.quantile(boot, 0.975) < 0.0
        ),
        "per_entry": rows,
        "headline": {
            "B_match_rate_min": float(df["B_match_rate"].min()),
            "B_match_rate_max": float(df["B_match_rate"].max()),
            "B_match_rate_range_pp": float(
                (df["B_match_rate"].max() - df["B_match_rate"].min()) * 100
            ),
            "A_ceiling": bool(a_ceiling),
            "joint_corr_significant": bool(
                np.quantile(boot, 0.025) > 0.0
                or np.quantile(boot, 0.975) < 0.0
            ),
        },
    }
    with open(OUT / "p5p6_audit_triangulation_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"\nB_match_rate: mean={df['B_match_rate'].mean():.3f}, "
          f"min={df['B_match_rate'].min():.3f}, max={df['B_match_rate'].max():.3f}")
    print(f"A_mean_score: overall={cl['score'].mean():.3f} "
          f"(unique values={a_unique}, ceiling={a_ceiling})")
    print(f"Joint correlation B_match vs n_audited: "
          f"corr={corr_match_vs_audited:+.4f}, "
          f"CI=[{np.quantile(boot, 0.025):+.4f}, "
          f"{np.quantile(boot, 0.975):+.4f}] excl0="
          f"{summary['joint_corr_excludes_zero']}")
    print(f"\nHeadline: {summary['headline']}")


if __name__ == "__main__":
    main()