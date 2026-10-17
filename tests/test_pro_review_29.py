"""Regression reproductions for the supplied, truncated review29."""

import importlib
import math
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "platform_modal/scripts"
sys.path.insert(0, str(SCRIPTS))
from zvf_alarm_contract import (
    at_risk_transitions,
    classify_trace,
    first_alarm,
    kaplan_meier,
    select_operating_point,
)


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [([1, 2, 3], [3, 2, 1], -1), ([1, 2, 2], [1, 3, 3], 1), ([1, 2, 3], [4, 5, 6], 1)],
)
def test_spearman_rank_correlation(x, y, expected):
    assert importlib.import_module("zvf_iter70")._spearman(x, y) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("x", "y"), [([1] * 3, [1, 2, 3]), ([1, 2], [1, 2, 3]), ([1, 2, float("nan")], [1, 2, 3])]
)
def test_spearman_undefined(x, y):
    assert math.isnan(importlib.import_module("zvf_iter70")._spearman(x, y))


def trace(z, a):
    return {
        "steps": [{"step": i, "zvf": v, "heldout_acc": h} for i, (v, h) in enumerate(zip(z, a))]
    }


def test_endpoints_are_independent():
    m = importlib.import_module("zvf_iter78")
    high = m.per_step_features(trace([1.0] * 25, [0.8] * 25))
    low = m.per_step_features(trace([0.0] * 25, [0.01] * 25))
    assert high[-1]["zvf_stuck"] == 1
    assert m.find_failure(high) is None
    assert low[-1]["zvf_stuck"] == 0
    assert m.find_failure(low) == 0
    assert m.find_failure(high, "zvf_stuck") == 19


def test_feature_prefix_never_uses_future_baseline():
    m = importlib.import_module("zvf_iter78")
    prefix = trace([0.2] * 12, [0.8] * 12)
    longer = trace([0.2] * 12 + [1.0] * 88, [0.8] * 100)
    assert m.per_step_features(prefix) == m.per_step_features(longer)[:12]
    assert m.find_alarm(m.per_step_features(trace([1.0] * 9, [0.8] * 9)), 0.1) is None


def test_failure_only_one_at_risk_transition_and_gap_excluded():
    assert list(at_risk_transitions([0, 1, 2, 3], [0, 1, 1, 1])) == [(0, 0, True)]
    assert list(at_risk_transitions([0, 2, 3], [0, 1, 1])) == []


def test_censored_trace_not_in_future_risk_set():
    rows = [{"n_steps": 2, "t_fail": -1}, {"n_steps": 4, "t_fail": 3}]
    assert kaplan_meier(rows, [0, 1, 2, 3, 4]) == [
        (0, 1.0, 2),
        (1, 1.0, 2),
        (2, 1.0, 1),
        (3, 0.0, 1),
    ]


def test_alarm_decision_not_backdated_and_same_step_is_late():
    assert first_alarm([1, 1, 1], 0.5, 3) == 2
    t = {"step": [0, 1, 2], "zvf": [1, 1, 1], "accuracy_failure": [0, 0, 1]}
    assert classify_trace(t, 0.5, 3)[0] == "late_alarm"
    assert classify_trace(t, 0.5, 2)[0] == "correct_stop"


def test_headline_is_grid_minimum_with_declared_tie_break():
    rows = [
        {"cost": 3, "k_persist": 5, "tau_optimal_at_C10": 0.5},
        {"cost": 1, "k_persist": 3, "tau_optimal_at_C10": 0.2},
    ]
    assert select_operating_point(rows) == rows[1]


def test_leadtime_figure_does_not_fabricate_standard_error():
    m = importlib.import_module("zvf_iter78_fig")
    ax = Mock()
    m.panel_leadtime(
        ax, [{"method": "a", "mean_lead_time": "10", "min_lead_time": "1", "max_lead_time": "20"}]
    )
    assert "xerr" not in ax.barh.call_args.kwargs


def test_recovery_from_onset_not_episode_end_and_censored():
    m = importlib.import_module("zvf_iter90")
    tr = lambda z: [{"zvf": v, "ha": 0.8, "step": i} for i, v in enumerate(z)]
    assert m.detect_episodes(tr([1] * 7 + [0]))[0]["recovered"] is False
    assert m.detect_episodes(tr([1, 0]))[0]["recovered"] is True
    assert m.detect_episodes(tr([1, 1]))[0]["recovered"] is None
    assert m.detect_episodes(tr([1, 1]))[0]["censored"]


def test_pooled_correlation_does_not_invent_independent_clusters(tmp_path):
    m = importlib.import_module("zvf_iter70")
    result = m.write_severity_corr(
        [{"mean_zvf": i / 4, "severity": i} for i in range(4)], tmp_path / "a.tsv"
    )
    assert math.isnan(result["spearman_ci"][0])
    assert "CI unavailable" in (tmp_path / "a.tsv").read_text()


def test_cli_modules_have_no_import_time_execution():
    importlib.import_module("zvf_iter82")
    importlib.import_module("zvf_iter86")


@pytest.mark.parametrize(
    ("method", "panel"), [("grpo", "panel_grpo_trace"), ("aero", "panel_aero_trace")]
)
def test_figure_alarm_uses_versioned_recorded_composite(method, panel):
    producer = importlib.import_module("zvf_iter78")
    figure = importlib.import_module("zvf_iter78_fig")
    rows = producer.per_step_features(trace([1.0] * 25, [0.8] * 25))
    for row in rows:
        row.update(
            source="variance_mitigation",
            method=method,
            seed="0",
            composite_ews=producer.composite_ews(row),
        )
    ax = Mock()
    getattr(figure, panel)(ax, rows)
    alarms = [
        call.args[0]
        for call in ax.axvline.call_args_list
        if call.kwargs.get("label", "").startswith("alarm")
    ]
    assert alarms == [producer.find_alarm(rows, 0.5)] == [10]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 2.0])
def test_episode_nonfinite_or_invalid_inputs_reject(value):
    m = importlib.import_module("zvf_iter90")
    with pytest.raises(ValueError, match="finite ZVF"):
        m.detect_episodes([{"step": 0, "zvf": value, "ha": 0.8}])


def test_all_censored_main_does_not_publish_zero_rate(tmp_path, monkeypatch):
    m = importlib.import_module("zvf_iter90")
    monkeypatch.setattr(m, "OUT", tmp_path)
    monkeypatch.setattr(m, "FIG", tmp_path)
    monkeypatch.setattr(
        m, "load_traces", lambda: {("a", 0): [{"step": i, "zvf": 1.0, "ha": 0.8} for i in range(2)]}
    )
    result = m.main()
    assert result["n_censored"] == 1
    assert result["pooled_recovery_rate"] is None
    assert result["best_method_recovery"] is None
    assert result["by_method_recovery"] == {"a": None}


def test_feature_generation_rejects_unobserved_step_gaps():
    m = importlib.import_module("zvf_iter78")
    with pytest.raises(ValueError, match="consecutive"):
        m.per_step_features(
            {
                "steps": [
                    {"step": 0, "zvf": 0.0, "heldout_acc": 0.8},
                    {"step": 2, "zvf": 1.0, "heldout_acc": 0.0},
                ]
            }
        )


def test_groupsize_loader_does_not_repeat_terminal_training_reward(tmp_path, monkeypatch):
    import json

    m = importlib.import_module("zvf_iter78")
    monkeypatch.setattr(m, "RESULTS", str(tmp_path))
    (tmp_path / "groupsize_zvf_sweep.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "seed": 0,
                        "group_size": 2,
                        "last10_avg": 0.01,
                        "step_log": [{"step": 0, "zvf": 1.0, "mean_reward": 0.01}],
                    }
                ]
            }
        )
    )
    row = m.load_groupsize_sweep()[0]
    assert row["steps"][0]["heldout_acc"] is None
    with pytest.raises(ValueError, match="per-step heldout"):
        m.per_step_features(row)
