"""Recompute the draft's family sensitivity from retained source rows."""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

import pytest
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "platform_hybrid/experiments/results"


def test_bh_fourteen_of_sixteen_sensitivity_from_source_rows():
    with (RESULTS / "length_bias_mechanism_per_run.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    pvalues = []
    for task in sorted({row["task"] for row in rows}):
        arms = {
            algo: {row["seed"]: row for row in rows if row["task"] == task and row["algo"] == algo}
            for algo in ("grpo", "dr_grpo")
        }
        assert arms["grpo"].keys() == arms["dr_grpo"].keys()
        seeds = sorted(arms["grpo"])
        for metric in ("ols_dL_dR", "theil_sen_dL_dR", "spearman_L_R", "lag1_dL_autocorr"):
            pvalues.append(
                float(
                    ttest_rel(
                        [float(arms["dr_grpo"][seed][metric]) for seed in seeds],
                        [float(arms["grpo"][seed][metric]) for seed in seeds],
                    ).pvalue
                )
            )
    with (RESULTS / "length_bias_iter24_signflip.tsv").open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["win"] == "10":
                pvalues.extend([float(row["p_drift"]), float(row["p_vs_half_signflip"])])
    assert len(pvalues) == 16
    # The minimum BH-adjusted p equals min(m*p_(i)/i); all 120 families
    # are retained, including those omitting the two smallest p-values.
    minima = [
        min(14 * p / rank for rank, p in enumerate(sorted(family), 1))
        for family in itertools.combinations(pvalues, 14)
    ]
    assert len(minima) == 120
    assert min(minima) == pytest.approx(0.11554570078507856)
    assert max(minima) == pytest.approx(0.4872)
    assert all(value > 0.05 for value in minima)
