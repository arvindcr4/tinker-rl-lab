"""Synthetic-only regression tests for review27-v3; no evidence regenerated."""

import importlib
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "platform_modal/scripts"))


def mod(n):
    return importlib.import_module(f"scaling_law_iter{n}")


def test_piecewise_constant_predict_and_forecast():
    m = mod(89)
    y = np.full(12, 0.6)
    fit = m.fit_3seg(y)
    assert all(abs(m._predict_at(fit, i) - 0.6) < 1e-10 for i in range(16))
    assert m.loocv(y, "3seg")["cv_rmse"] < 1e-10
    assert m.kfold_forecast(y)["3seg"]["rmse"] < 1e-10


def test_piecewise_continuous_original_coordinates():
    m = mod(89)
    x = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 12.0])
    y = 0.2 + 0.03 * x
    fit = m.fit_family("3seg", y, x)
    assert np.max(abs(fit["pred"] - y)) < 1e-10
    for key in ["cp1", "cp2"]:
        c = fit["params"][key]
        assert abs(m._predict_at(fit, c - 1e-8) - m._predict_at(fit, c + 1e-8)) < 1e-8
    assert m.loocv(np.arange(12) * 0.03 + 0.2, "1seg_ols")["cv_rmse"] < 1e-10


@pytest.mark.parametrize("n", [93, 97])
def test_ar_intercept_and_no_future_conditioning(n):
    m = mod(n)
    y = [0.1]
    for _ in range(20):
        y.append(0.15 + 0.7 * y[-1])
    y = np.array(y)
    t = np.arange(len(y), dtype=float)
    p = m._fit_ar1(y)
    assert p[:2] == pytest.approx([0.15, 0.7], abs=1e-9)
    assert m._predict_one_at_step(
        np.array([0, 1, 3]), 2, p, "ar1", np.array([0.1, 0.22, 999])
    ) == pytest.approx(0.304)
    assert m._loocv_rmse(y, t, "ar1")[0] < 1e-10
    assert m._forecast_last_k(y, t, "ar1")[0] < 1e-10
    # A missing time point may not create a fake AR pair.
    with pytest.raises(ValueError, match="insufficient consecutive AR pairs"):
        m._fit_ar1(y[:4], np.array([0, 1, 3, 4]))


def test_profile_and_peak_time_coordinates():
    m = mod(73)
    t = np.linspace(10, 30, 30)
    y = 0.8 * (-np.expm1(-0.1 * t))
    f = m.closed_form_saturation(t, y)
    assert f["rss"] < 1e-8
    assert f["r_max"] == pytest.approx(0.8, rel=1e-3)
    y = m.peak_decay(t, 0.8, 17, 0.04)
    f = m.nemotron_peak_decay(t, y)
    assert f["t_peak"] == pytest.approx(17, abs=1e-3)
    assert f["rss"] < 1e-8


@pytest.mark.parametrize("family", ["fit_saturation", "fit_power"])
def test_compute_unit_invariance(family):
    m = mod(81)
    x = np.arange(1, 20, dtype=float)
    y = 0.8 * (-np.expm1(-0.5 * x)) if family == "fit_saturation" else 0.2 * x**0.2
    a, ra = getattr(m, family)(x, y)
    b, rb = getattr(m, family)(x * 0.004096, y)
    assert ra == pytest.approx(rb, abs=1e-12)
    fn = m.saturation if family == "fit_saturation" else m.power_law
    assert fn(x, *a) == pytest.approx(fn(x * 0.004096, *b), abs=1e-8)


def test_likelihood_counts_and_no_invalid_winner():
    from _review27_models import information_criteria, best_candidate, signed_phases

    assert information_criteria(1, 20, 3) == pytest.approx(mod(73)._aic_bic(1, 20, 2))
    assert information_criteria(1, 20, 3) == pytest.approx(mod(81).aic_bic(1, 20, 2))
    assert best_candidate({"a": float("nan"), "b": float("inf")}) == "no_valid_candidate"
    assert signed_phases([0.1, 0.4, 0.05])
    assert not signed_phases([-0.1, -0.4, -0.05])
    assert not signed_phases([0, 0, 0])


def test_bootstrap_deterministic_and_ordered():
    m = mod(89)
    y = np.full(12, 0.6)
    a = m.bootstrap_3seg(y, 8)
    b = m.bootstrap_3seg(y, 8)
    assert a == b
    assert a["n_boot_succeeded"] == 8
    m = mod(93)
    y = np.array([0.2 + 0.01 * i for i in range(20)])
    a = m._bootstrap_phi_ci(y, 30)
    assert a == m._bootstrap_phi_ci(y, 30)
    assert a[0] > 0.9


def test_paired_hierarchical_comparison():
    m = mod(77)
    y = np.array([0.2, 0.3, 0.35, 0.4, 0.42, 0.43, 0.44, 0.45])
    result = m.hierarchical_r_max({"x": y}, ["x"])["per_anchor"]["x"]
    assert "suffix_mse" in result
    assert "constant_last_mse" in result
    h = m.prefix_horizon(np.arange(1, 9.0), y)
    at = next(r for r in h["per_k"] if r[0] == h["k_best_sat"])
    assert h["best_r2_const"] == at[2]


def test_short_bootstrap_schema_and_explicit_paths(tmp_path):
    m = mod(89)
    result = m.bootstrap_3seg(np.array([0.1, 0.2, 0.3]), 3)
    assert result["n_boot_succeeded"] == 0
    assert result["n_boot_failed"] == 3
    assert np.isnan(result["canon_order_rate"])
    from _review27_models import configure_paths

    traces = tmp_path / "traces"
    traces.mkdir()
    ns = {"TRACE_DIR": traces, "RESULTS_DIR": tmp_path / "old"}
    configure_paths(ns, ["--output-dir", str(tmp_path / "new")])
    assert ns["RESULTS_DIR"] == tmp_path / "new"
    assert not (tmp_path / "old").exists()


def test_ar_forecast_respects_offset():
    for n in (93, 97):
        m = mod(n)
        assert m._predict_ar1(
            np.array([0.2]), [0.1, 0.5, 0.01], [12, 14], origin=10
        ) == pytest.approx([0.2, 0.2])
        assert m._predict_ar1(np.array([0.1]), [0.1, 0.5, 0.01], [12], origin=10) == pytest.approx(
            [0.175]
        )
