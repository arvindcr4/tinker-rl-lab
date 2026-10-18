"""Synthetic numeric-to-figure regressions; no historical artifact regeneration."""

import importlib
import json
import math
from pathlib import Path
import sys

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "platform_modal/scripts"
sys.path.insert(0, str(SCRIPTS))


def load(n):
    return importlib.import_module(f"group_size_iter{n}")


def iso_fixture():
    return pd.DataFrame(
        [
            {
                "budget_tokens": t,
                "G": g,
                "heldout_acc_mean": a,
                "heldout_acc_ci_low": a - 0.04,
                "heldout_acc_ci_high": a + 0.04,
            }
            for t, accs in [(1_000_000, [0.79, 0.8]), (4_000_000, [0.6, 0.8])]
            for g, a in zip([4, 32], accs, strict=True)
        ]
    )


def test_effect_size_does_not_invent_replicate_sd():
    m = load(123)
    results = m.effect_size(iso_fixture(), pd.DataFrame())
    assert results[0]["mean_difference"] == pytest.approx(0.01)
    assert results[1]["mean_difference"] == pytest.approx(0.2)
    assert all(r["cohens_d"] is None for r in results)
    assert all(r["effect_size_status"] == "NOT_IDENTIFIABLE_FROM_SUMMARY_CIS" for r in results)
    # Changing a CI width cannot change an observed effect's magnitude.
    iso = iso_fixture()
    iso["heldout_acc_ci_low"] = 0
    iso["heldout_acc_ci_high"] = 1
    assert m.effect_size(iso, pd.DataFrame()) == results
    with pytest.raises(ValueError, match="duplicate"):
        m.effect_size(pd.concat([iso, iso]), pd.DataFrame())


def test_iter123_figure_uses_numeric_effect_and_marks_missing(tmp_path, monkeypatch):
    m = load(123)
    results = m.effect_size(iso_fixture(), pd.DataFrame())
    for row in results:
        row["headline"] = "arbitrary prose with T=garbage = never parsed"
    original_close = m.plt.close
    monkeypatch.setattr(m.plt, "close", lambda *_: None)
    m.make_figure([], [], [], results, [], tmp_path / "effect.pdf")
    fig = m.plt.gcf()
    heights = [bar.get_height() for bar in fig.axes[3].patches]
    assert heights[:2] == pytest.approx([0.01, 0.2])
    assert all(math.isnan(v) for v in heights[2:])
    assert "Accuracy difference" in fig.axes[3].get_ylabel()
    assert not any("large effect" in line.get_label() for line in fig.axes[3].lines)
    original_close(fig)


def test_iter119_boundary_numeric_serialization_and_plot(tmp_path, monkeypatch):
    m = load(119)
    wu = m.wu_equivalence_boundary(iso_fixture(), pd.DataFrame())
    values = [r for r in wu if r["metric_kind"] == "wu_boundary_at_T"]
    assert values[0]["max_group_ratio"] == 0.125
    assert values[1]["max_group_ratio"] is None
    assert values[1]["boundary_status"] == "NO_PASSING_PAIR"
    m.write_tsv(tmp_path / "boundary.tsv", wu, ["metric_kind", "metric_key", "headline"])
    table = pd.read_csv(tmp_path / "boundary.tsv", sep="\t")
    assert table.loc[table.metric_key == "T=1e+06", "max_group_ratio"].iloc[0] == 0.125
    for row in values:
        row["headline"] = "changed prose"
    extrap = [
        {
            "metric_kind": "_extras",
            "headline": json.dumps(
                {
                    "T": [1e6, 4e6],
                    "R": [0.99, 0.75],
                    "slope": -0.1,
                    "intercept": 1.5,
                    "slope_lo": -0.2,
                    "slope_hi": 0,
                }
            ),
        }
    ]
    cross = [
        {
            "metric_kind": "_extras",
            "headline": json.dumps(
                {
                    "df_records": [
                        {
                            "experiment": "groupsize_zvf_sweep",
                            "library": "test",
                            "G": 4,
                            "mean_zvf": 0.3,
                            "last10_avg": 0.7,
                        }
                    ],
                    "rho_pooled": -0.5,
                }
            ),
        }
    ]
    original_close = m.plt.close
    monkeypatch.setattr(m.plt, "close", lambda *_: None)
    m.make_figure(extrap, cross, wu, [], tmp_path / "boundary.pdf")
    fig = m.plt.gcf()
    plotted = fig.axes[2].lines[0].get_ydata()
    assert plotted[0] == 0.125
    assert all(math.isnan(v) for v in plotted[1:])
    assert "Lower ZVF associated with higher accuracy" in fig.axes[1].get_title()
    original_close(fig)


@pytest.mark.parametrize(
    ("rho", "expected"),
    [
        (-0.5, "higher accuracy"),
        (0.5, "lower accuracy"),
        (0, "Zero rank"),
        (float("nan"), "unavailable"),
    ],
)
def test_correlation_sign_and_no_causal_claim(rho, expected):
    assert expected in load(119).zvf_association_label(rho)
