"""Run-level endpoint analysis under an explicit, matched hypothesis manifest.

Inputs must be independently identified empirical run receipts, never raw
trajectory steps or a list of numbers reconstructed from aggregate summaries.
This validates the contract; it cannot authenticate historical measurements.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from platform_hybrid.experiments.research_contracts import (
    EvidenceUnavailable,
    PROTOCOL_VERSION,
    measured_observations,
)

MATCH_KEYS = (
    "model_revision",
    "initialization_id",
    "adapter_config",
    "dataset_hash",
    "prompt_hash",
    "grader_version",
    "evaluation_ids_hash",
    "budget_kind",
    "budget",
)


def analyze_measured_runs(records, hypotheses, *, seed=42, n_boot=2000, alpha=0.05):
    if not hypotheses:
        raise EvidenceUnavailable("Declare the complete correction family")
    if not 0 < alpha < 1 or n_boot < 100:
        raise ValueError("Invalid analysis settings")
    ids = [h["hypothesis_id"] for h in hypotheses]
    if len(ids) != len(set(ids)):
        raise ValueError("Hypothesis IDs must be unique")
    comparisons = []
    for h in hypotheses:
        endpoint = h["endpoint"]
        arms = []
        for arm in (h["arm_a"], h["arm_b"]):
            selected = [
                r
                for r in records
                if r.get("arm") == arm and r.get("comparison_id") == h["comparison_id"]
            ]
            values = measured_observations(selected, endpoint)
            if len(values) < 2:
                raise EvidenceUnavailable("At least two independent runs per arm required")
            arms.append(selected)
        for key in MATCH_KEYS:
            vals = [json.dumps(r.get(key), sort_keys=True) for group in arms for r in group]
            if any(r.get(key) is None for group in arms for r in group) or len(set(vals)) != 1:
                raise EvidenceUnavailable(f"Missing/mismatched comparison contract: {key}")
        if {r["run_id"] for r in arms[0]} & {r["run_id"] for r in arms[1]}:
            raise EvidenceUnavailable("An independent run cannot appear in both arms")
        paired = h.get("paired", False)
        if paired:
            maps = [{r.get("pair_id"): r for r in group} for group in arms]
            if (
                any(None in m or len(m) != len(group) for m, group in zip(maps, arms))
                or maps[0].keys() != maps[1].keys()
            ):
                raise EvidenceUnavailable("Matched unique pair_id sets required")
            order = sorted(maps[0])
            a, b = [np.array([m[k][endpoint] for k in order], dtype=float) for m in maps]
            differences = b - a
            if np.var(differences) == 0:
                p = 1.0 if np.all(differences == 0) else 0.0
            else:
                p = float(stats.ttest_rel(b, a).pvalue)
            rng = np.random.default_rng(seed)
            draws = np.mean(rng.choice(differences, (n_boot, len(differences))), axis=1)
            test = "paired_run_t"
        else:
            a, b = [np.array([r[endpoint] for r in group], dtype=float) for group in arms]
            if np.var(a) == 0 and np.var(b) == 0:
                p = 1.0 if a[0] == b[0] else 0.0
            else:
                p = float(stats.ttest_ind(b, a, equal_var=False).pvalue)
            rng = np.random.default_rng(seed)
            draws = np.mean(rng.choice(b, (n_boot, len(b))), axis=1) - np.mean(
                rng.choice(a, (n_boot, len(a))), axis=1
            )
            test = "welch_independent_runs"
        if not math.isfinite(p):
            raise EvidenceUnavailable("Undefined inferential test")
        comparisons.append(
            dict(
                hypothesis_id=h["hypothesis_id"],
                endpoint=endpoint,
                n=[len(a), len(b)],
                effect=float(b.mean() - a.mean()),
                ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
                test=test,
                p_raw=p,
                interval_type="paired_run_bootstrap" if paired else "independent_run_bootstrap",
                run_ids=[[r["run_id"] for r in group] for group in arms],
                receipt_ids=[[r["receipt_id"] for r in group] for group in arms],
            )
        )
    for c in comparisons:
        c["p_corrected"] = min(1.0, c["p_raw"] * len(comparisons))
        c["reject"] = c["p_corrected"] < alpha
    return dict(
        protocol_version=PROTOCOL_VERSION,
        analysis_unit="independent_run",
        correction_scope="Bonferroni over supplied prespecified hypothesis family",
        family_size=len(comparisons),
        comparisons=comparisons,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipts", type=Path)
    p.add_argument("hypotheses", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = analyze_measured_runs(
        json.loads(args.receipts.read_text()), json.loads(args.hypotheses.read_text())
    )
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
