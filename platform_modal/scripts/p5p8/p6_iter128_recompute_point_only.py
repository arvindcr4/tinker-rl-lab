#!/usr/bin/env python3
"""Recompute review17-v2 mean_zvf effects from explicit, complete seed pairs.
Uses numpy percentile bootstrap with its actual RNG seed and sample identities.
Writes prospective versioned results only; missing controls fail closed."""

from __future__ import annotations

from review17_contract import load_seed_metric, bootstrap, paired_result, source_hash, VERSION

import json
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))  # platform_modal/scripts, for _paths
from _paths import REPO_ROOT

ROOT = REPO_ROOT
RES = ROOT / "platform_hybrid/experiments" / "results" / "p5p8"
OUT = RES / "review17_v2"

SEED = 20260705
N_BOOT = 2000
DELTA_IDS = [
    "delta_aero",
    "delta_areal",
    "delta_cppo",
    "delta_es",
    "delta_gift",
    "delta_grpo",
    "delta_mcgrpo",
    "delta_ngrpo",
    "delta_scafgrpo",
]
METHOD_NAMES = {
    "delta_aero": "aero",
    "delta_areal": "areal",
    "delta_cppo": "cppo",
    "delta_es": "es",
    "delta_gift": "gift",
    "delta_grpo": "grpo",
    "delta_mcgrpo": "mcgrpo",
    "delta_ngrpo": "ngrpo",
    "delta_scafgrpo": "scafgrpo",
}


def load_risk_index():
    return load_seed_metric(
        ROOT / "platform_hybrid/experiments/results/zvf_iter130_risk_index.tsv", "mean_zvf"
    )


def boot_ci(values, B, seed):
    return bootstrap(values, B, seed)


def load_registry_delta(delta_id):
    """Read platform_hybrid/registry/entries/delta_<id>.json, return measured[] list."""
    fp = ROOT / "platform_hybrid/registry" / "entries" / f"{delta_id}.json"
    if not fp.exists():
        return None, None
    with fp.open() as f:
        d = json.load(f)
    return d.get("measured", []), d.get("zvf_risk_mean", None)


def write_tsv(name, rows, cols):
    if not rows:
        return
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append(
            "\t".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]) for c in cols)
        )
    (OUT / name).write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("# === P6 JOB B (iter 128): recompute 8 POINT_ONLY mean_zvf CIs ===")
    risk = load_risk_index()
    available_methods = sorted(risk.keys())
    print(f"# loaded risk_index for {len(available_methods)} methods: {available_methods}")

    rows = []
    summary = {
        "iter": 128,
        "timestamp": "2026-07-05",
        "n_deltas_recomputed": 0,
        "n_deltas_requested": len(DELTA_IDS),
        "recomputed": [],
        "drift_summary": {},
    }
    for delta_id in DELTA_IDS:
        mname = METHOD_NAMES[delta_id]
        if mname not in risk:
            print(f"# SKIP {delta_id}: no risk data for {mname}")
            continue
        per_seed = list(risk[mname].values())
        n_seeds = len(per_seed)
        # Compute per-seed mean_zvf aggregated to a single mean (for cross-check)
        mean_perseed = sum(per_seed) / n_seeds
        boot_mean, ci_lo, ci_hi = boot_ci(per_seed, N_BOOT, SEED)
        result = paired_result(risk.get("grpo", {}), risk[mname], N_BOOT, SEED)
        recomputed_delta = result["delta"]
        new_ci_lo, new_ci_hi = result["ci_low"], result["ci_high"]
        # Load the registry's existing mean_zvf claim
        measured, registry_zvf_risk = load_registry_delta(delta_id)
        existing_mean_zvf = None
        existing_ci = None
        existing_ci_method = None
        if measured:
            for row in measured:
                if row.get("metric") == "mean_zvf" and row.get("panel") == "zvf130_5seed":
                    existing_mean_zvf = float(row.get("delta", 0.0))
                    existing_ci = (float(row.get("ci_low", 0.0)), float(row.get("ci_high", 0.0)))
                    existing_ci_method = row.get("ci_method", "?")
                    break
        # The "delta" of the registry is itself a difference; for mean_zvf at
        # zvf130_5seed, the recorded "delta" is (method - grpo) mean_zvf.
        # Compute drift = (recomputed delta) - (existing delta)
        drift = recomputed_delta - (existing_mean_zvf if existing_mean_zvf is not None else 0.0)
        rec = {
            "delta_id": delta_id,
            "method": mname,
            "n_seeds": n_seeds,
            "result": result,
            "metric": "mean_zvf",
            "units": "fraction",
            "source_sha256": source_hash(
                ROOT / "platform_hybrid/experiments/results/zvf_iter130_risk_index.tsv"
            ),
            "recomputed_mean": boot_mean,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "recomputed_delta": recomputed_delta,
            "recomputed_delta_ci_lo": new_ci_lo,
            "recomputed_delta_ci_hi": new_ci_hi,
            "existing_delta": existing_mean_zvf if existing_mean_zvf is not None else 0.0,
            "existing_ci_low": existing_ci[0] if existing_ci else 0.0,
            "existing_ci_high": existing_ci[1] if existing_ci else 0.0,
            "existing_ci_method": existing_ci_method or "missing",
            "drift": drift,
            "ci_includes_existing": bool(
                existing_ci and new_ci_lo <= existing_ci[0] and new_ci_hi >= existing_ci[1]
            ),
        }
        rows.append(rec)
        summary["recomputed"].append(rec)
        summary["n_deltas_recomputed"] += 1

    if not rows:
        print("# ERROR: no rows produced")
        raise ValueError("No rows produced")

    # Headline synthesis
    n_recomp_ci_includes_old = sum(1 for r in rows if r["ci_includes_existing"])
    summary["n_recomp_ci_includes_old"] = n_recomp_ci_includes_old
    summary["n_recomp_total"] = len(rows)
    summary["drift_magnitude"] = {
        "max_abs_drift": float(max(abs(r["drift"]) for r in rows)),
        "min_abs_drift": float(min(abs(r["drift"]) for r in rows)),
        "mean_abs_drift": float(sum(abs(r["drift"]) for r in rows) / len(rows)),
    }
    summary["recompute_old_point_only_to_new_bootstrap"] = {
        r["delta_id"]: {
            "old_ci": [r["existing_ci_low"], r["existing_ci_high"]],
            "new_ci": [
                round(r["recomputed_delta_ci_lo"], 6),
                round(r["recomputed_delta_ci_hi"], 6),
            ],
            "ci_width_delta": round(
                (r["recomputed_delta_ci_hi"] - r["recomputed_delta_ci_lo"])
                - max(r["existing_ci_high"] - r["existing_ci_low"], 0.0),
                6,
            ),
        }
        for r in rows
    }
    summary["analysis_version"] = VERSION
    summary["operational_recommendation"] = (
        "Review complete paired-seed result objects; no automatic evidence-tier promotion."
    )

    write_tsv(
        "p6_iter128_recompute_point_only.tsv",
        rows,
        [
            "delta_id",
            "method",
            "n_seeds",
            "recomputed_mean",
            "ci_lo",
            "ci_hi",
            "recomputed_delta",
            "recomputed_delta_ci_lo",
            "recomputed_delta_ci_hi",
            "existing_delta",
            "existing_ci_low",
            "existing_ci_high",
            "existing_ci_method",
            "drift",
            "ci_includes_existing",
        ],
    )
    (OUT / "p6_iter128_recompute_point_only_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"# wrote {len(rows)} recompute rows -> p6_iter128_recompute_point_only.tsv")
    print(f"# CI-includes-old: {n_recomp_ci_includes_old}/{len(rows)}")
    print(f"# mean abs drift: {summary['drift_magnitude']['mean_abs_drift']:.4f}")
    print(f"# max abs drift: {summary['drift_magnitude']['max_abs_drift']:.4f}")
    for r in rows:
        print(
            f"# {r['delta_id']}: new CI [{r['recomputed_delta_ci_lo']:.4f}, {r['recomputed_delta_ci_hi']:.4f}] "
            f"old CI [{r['existing_ci_low']:.4f}, {r['existing_ci_high']:.4f}] "
            f"drift={r['drift']:+.4f}"
        )


if __name__ == "__main__":
    main()
