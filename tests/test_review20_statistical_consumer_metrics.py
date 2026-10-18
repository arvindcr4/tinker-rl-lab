"""Offline regressions for Review20 R16/R17/R18/R19 statistical consumers."""
import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import kendalltau

SCRIPTS = Path(__file__).resolve().parents[1] / "platform_modal/scripts/p5p8"
sys.path.insert(0, str(SCRIPTS))
cohort = importlib.import_module("p8_cohort_calibration_parity")
tau = importlib.import_module("p7_tau_seed_decision_stability")
regret = importlib.import_module("p8_decision_regret")
calibration = importlib.import_module("p8_calibration_cis")


def test_eta_squared_uses_between_and_total_row_residual_variance():
    groups = [0, 0, 1, 1]
    # Equal, nonzero cohort means with real within-cohort variance explain none.
    assert cohort.calibration_gap_eta_squared([0.1, 0.3, 0.1, 0.3], [0] * 4, groups) == 0
    # Opposite cohort gaps with no within-cohort variance explain all variance.
    assert cohort.calibration_gap_eta_squared([0.2, 0.2, 0.8, 0.8], [0, 0, 1, 1], groups) == 1
    assert cohort.calibration_gap_eta_squared([0, 0.2, 0.4, 0.6], [0] * 4, groups) == pytest.approx(0.8)
    assert cohort.calibration_gap_eta_squared([0.2] * 4, [0] * 4, groups) is None
    assert cohort.calibration_gap_eta_squared([], [], []) is None
    with pytest.raises(ValueError, match="align"):
        cohort.calibration_gap_eta_squared([0.2], [], [])


def test_eta_squared_is_invariant_to_row_and_group_labels():
    scores = np.array([0.1, 0.6, 0.4, 0.9, 0.2])
    labels = np.array([0, 0, 1, 1, 0])
    groups = np.array([0, 0, 1, 1, 1])
    reference = cohort.calibration_gap_eta_squared(scores, labels, groups)
    order = [4, 0, 3, 1, 2]
    assert cohort.calibration_gap_eta_squared(scores[order], labels[order], 9 - groups[order]) == pytest.approx(reference)


def test_all_negative_cohorts_survive_point_and_bootstrap_outputs(tmp_path, monkeypatch):
    data = tmp_path / "test.csv"
    headers = [f"V{i}" for i in range(1, 21)] + ["V_mean", "V_std", "V_max", "V_min", "Amount", "Time", "Class"]
    with data.open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for i in range(30):
            writer.writerow([0] * 20 + [i, i + 1, i + 2, 0, i * 10, i * 100, 0])
    monkeypatch.setattr(cohort, "TRAIN", str(data))
    monkeypatch.setattr(cohort, "TEST", str(data))
    monkeypatch.setattr(cohort, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(cohort, "B", 8)
    monkeypatch.setattr(cohort, "fit_gbt", lambda X, y, **kwargs: len(X[0]))
    monkeypatch.setattr(cohort, "predict_proba", lambda width, X: np.full(len(X), 0.2 if width == 20 else 0.3))
    cohort.main()
    summary = json.loads((tmp_path / "p8_cohort_calibration_summary.json").read_text())
    assert len(summary["cells"]) == 26
    for cell in summary["cells"]:
        score = 0.2 if cell["tree"] == "XGB-20raw" else 0.3
        assert cell["pos"] == 0
        assert cell["ece10"] == pytest.approx(score)
        assert cell["brier"] == pytest.approx(score**2)
        assert cell["calib_gap"] == pytest.approx(score)
        assert cell["recall_at_K_2pct"] is None
    for name, n_strata in (("v_mean", 5), ("amount", 5), ("time", 3)):
        result = summary["H2_24full_minus_20raw_ece_delta"][name]
        assert result["n_strata"] == n_strata
        assert [result[k] for k in ("obs_mean_delta", "ci_lo", "ci_hi")] == pytest.approx([0.1] * 3)
    assert cohort.recall_at_k_pct([0.1, 0.2], [0, 0], 2) == (None, 1)


@pytest.mark.parametrize(("a", "b"), [
    ([0, 0, 1], [0, 0, 1]),
    ([0, 0, 1], [1, 1, 0]),
    ([0, 0, 1, 2], [0, 1, 1, 2]),
    ([0, 1, 2, 3], [2, 0, 3, 1]),
])
def test_kendall_tau_b_matches_independent_tie_aware_reference(a, b):
    assert tau.kendall_tau(a, b) == pytest.approx(kendalltau(a, b, variant="b").statistic)


def test_kendall_constant_or_tiny_input_is_explicitly_undefined():
    assert tau.kendall_tau([1, 1, 1], [0, 1, 2]) is None
    assert tau.kendall_tau([1, 1], [1, 1]) is None
    assert tau.kendall_tau([], []) is None
    assert tau.kendall_tau([1], [2]) is None
    with pytest.raises(ValueError, match="equal lengths"):
        tau.kendall_tau([1], [1, 2])


@pytest.mark.parametrize(("choices", "expected"), [([8, 8], 0), ([16, 16], -100), ([4, 4], 50), ([4, 16], -25)])
def test_savings_are_baseline_minus_actual(choices, expected):
    result = tau.rollout_accounting(choices)
    assert result["baseline_rollouts"] == 16
    assert result["actual_rollouts"] == sum(choices)
    assert result["saved_rollouts"] == 16 - sum(choices)
    assert result["net_savings_pct"] == expected


def test_tau_main_uses_real_actions_savings_and_narrative(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tau, "SEEDS", [1, 2])
    monkeypatch.setattr(tau, "OUT", tmp_path)
    monkeypatch.setattr(tau, "load_seed", lambda seed: {
        "seed": seed, "zvf": [0.1, 0.7, 0.9, 1.0], "heldout_acc": 0.5,
        "mean_zvf": 0.675, "last10_zvf": 0.675,
    })
    original_bootstrap = tau.bootstrap_ci
    monkeypatch.setattr(tau, "bootstrap_ci", lambda values, **kwargs: original_bootstrap(values, B=8))
    tau.main()
    summary = json.loads((tmp_path / "p7_tau_seed_stability_summary.json").read_text())
    assert summary["cross_seed_kendall_tau_zvf_trajectory"]["mean"] == 1
    assert summary["cross_seed_kendall_tau_zvf_trajectory"]["variant"] == "tau-b"
    for row in summary["two_threshold_grid"]:
        assert row["de_esc_active"] is (row["de_esc_fires_total"] > 0)
        assert row["net_savings_pct"] == row["savings_pct_mean"]
        assert row["net_savings_pct"] == pytest.approx(100 * (row["baseline_rollouts"] - row["actual_rollouts"]) / row["baseline_rollouts"])
    output = capsys.readouterr().out
    assert "agreement is Jaccard=1.000, full step agreement=1.000" in output
    assert "COMPLETELY DIFFERENT" not in output


def test_regret_plot_payloads_have_actual_models_and_distinct_units(monkeypatch):
    monkeypatch.setattr(regret, "C_INV_GRID", [0.5])
    monkeypatch.setattr(regret, "L_GRID", [100, 250])
    rows = [
        {"C_inv": 0.5, "L": loss, "model": model, "regret_per_dec": value}
        for loss, raw, full in [(100, 2, 0.5), (250, 0, 0.5)]
        for model, value in [("XGB-20raw", raw), ("XGB-24full", full), ("XGB-4sensor", 9)]
    ]
    payload = regret.regret_plot_payloads(rows)
    np.testing.assert_allclose(payload["raw_regret"], [[2, 0]])
    np.testing.assert_allclose(payload["full_regret"], [[0.5, 0.5]])
    np.testing.assert_allclose(payload["sensor_closure"], [[1.5, -0.5]])
    np.testing.assert_allclose(payload["fraction_captured"], [[0.75, np.nan]])


def test_regret_plot_draws_the_validated_payloads(tmp_path, monkeypatch):
    monkeypatch.setattr(regret, "C_INV_GRID", [0.5])
    monkeypatch.setattr(regret, "L_GRID", [100])
    rows = [{"C_inv": 0.5, "L": 100, "model": name, "regret_per_dec": r}
            for name, r in [("XGB-20raw", 2), ("XGB-24full", 0.5)]]
    from matplotlib.axes import Axes
    original = Axes.imshow
    seen = []

    def record(self, values, *args, **kwargs):
        seen.append(np.array(values))
        return original(self, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", record)
    regret.plot_grid(rows, tmp_path / "regret.png", tmp_path / "regret.pdf")
    assert [float(values[0, 0]) for values in seen] == [2, 0.5, 1.5, 0.75]
    assert (tmp_path / "regret.png").stat().st_size > 0
    assert (tmp_path / "regret.pdf").stat().st_size > 0


def test_cost_table_keeps_duration_separate_from_unknown_dollars():
    rows = calibration.cost_accounting_rows({"infer_time_sec_per_10k": 0.006})
    tree, sensor, _, hybrid = rows
    assert tree["tree_inference_s_batch"] == 0.006
    assert tree["cost_usd_batch"] is None
    assert tree["cpu_usd_per_hour"] is None
    assert sensor["cost_usd_batch"] == 0.0035
    assert hybrid["tree_rows"] == 10_000
    assert hybrid["llm_invocations"] == 1_000
    assert hybrid["llm_cost_usd_batch"] == 3.5
    assert hybrid["cost_usd_batch"] is None
    assert hybrid["llm_input_tokens_batch"] == 120_000
    assert hybrid["llm_output_tokens_batch"] == 5_000
    assert "$3.5 / 10000 rows" in hybrid["comment"]
    assert "unavailable" in hybrid["cost_status"]


@pytest.mark.parametrize("field", ["inference_us_per_10k", "inf_us_10k"])
def test_explicit_cpu_price_converts_seconds_to_dollars_and_adds_hybrid_cost(field):
    rows = calibration.cost_accounting_rows({field: 6_000}, cpu_usd_per_hour=2)
    assert rows[0]["tree_inference_s_batch"] == pytest.approx(0.006)
    assert rows[0]["cost_usd_batch"] == pytest.approx(0.006 * 2 / 3600)
    assert rows[-1]["cost_usd_batch"] == pytest.approx(3.5 + rows[0]["cost_usd_batch"])
    assert rows[-1]["cost_usd_per_decision"] == pytest.approx(rows[-1]["cost_usd_batch"] / 10_000)


def test_missing_timing_does_not_invent_measurement_or_tree_price():
    tree = calibration.cost_accounting_rows({}, cpu_usd_per_hour=2)[0]
    assert tree["tree_inference_s_batch"] is None
    assert tree["cost_usd_batch"] is None
    assert tree["timing_status"] == "unavailable"
    zero = calibration.cost_accounting_rows({"infer_time_sec_per_10k": 0}, cpu_usd_per_hour=2)[0]
    assert zero["tree_inference_s_batch"] == 0
    assert zero["cost_usd_batch"] == 0


def test_consumers_use_versioned_output_directories():
    for path in (cohort.OUT_DIR, tau.OUT, regret.OUT, calibration.RES):
        assert Path(path).name == "corrected-review20-v2"
