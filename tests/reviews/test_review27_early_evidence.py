"""Regression fixtures use temporary evidence/output roots only."""

import csv
import hashlib
import importlib
import json
import math

import numpy as np
import pytest

from platform_modal.scripts._review27_rank import load_reward_trace, observed_trace


def module(n):
    return importlib.import_module(f"platform_modal.scripts.scaling_law_iter{n}")


def test_explicit_trace_identity(tmp_path):
    raw = json.dumps({"run_id": "run-1", "reward_trace": [0, 0.2, 0.4, 0.6]}).encode()
    (tmp_path / "trace.json").write_bytes(raw)
    row = dict(
        trace_file="trace.json",
        trace_sha256=hashlib.sha256(raw).hexdigest(),
        run_id="run-1",
        n_steps=4,
    )
    values, identity = observed_trace(row, tmp_path)
    assert values.tolist() == [0, 0.2, 0.4, 0.6]
    assert identity["trace_sha256"] == row["trace_sha256"]
    assert observed_trace({"n_steps": 4}, tmp_path)[0] is None
    for changed, match in (
        ({"trace_sha256": "bad"}, "hash"),
        ({"run_id": "other"}, "identity"),
        ({"n_steps": 3}, "length"),
        ({"trace_file": "../outside.json"}, "escapes"),
    ):
        with pytest.raises(ValueError, match=match):
            observed_trace({**row, **changed}, tmp_path)


@pytest.mark.parametrize("values", [[0, None, 1], [0, float("nan"), 1], [0, True, 1]])
def test_trace_loader_never_compresses_invalid_steps(tmp_path, values):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"reward_trace": values}))
    with pytest.raises(
        ValueError,
        match="ordered reward trace cannot contain missing or boolean steps|ordered reward trace must be finite",
    ):
        load_reward_trace(path)


def test_failed_fit_is_not_perfect_bound_hit(monkeypatch):
    m = module(61)

    def fail(*args, **kwargs):
        raise RuntimeError("optimizer failed")

    monkeypatch.setattr(m, "curve_fit", fail)
    assert all(math.isnan(v) for v in m._fit_sat([1, 2, 3], [0.1, 0.2, 0.3]))
    assert math.isnan(m._zvf_proxy([])["fraction_extreme_step_means"])
    assert m._zvf_proxy([0.5, 0.5])["fraction_extreme_step_means"] == 0


def test_phase_objective_continuity_and_signed_shape():
    m = module(65)
    t = np.arange(1.0, 13.0)
    y = 0.1 + 0.01 * t + 0.09 * np.maximum(0, t - 4) - 0.1 * np.maximum(0, t - 9)
    i, j, sse, slopes, intercepts, residuals, means, stds = m._fit_3piece(t, y)
    assert (i, j) == (3, 8)
    assert sse < 1e-20
    assert slopes[0] * t[i] + intercepts[0] == pytest.approx(slopes[1] * t[i] + intercepts[1])
    assert slopes[1] * t[j] + intercepts[1] == pytest.approx(slopes[2] * t[j] + intercepts[2])
    assert m._classify(slopes, residuals, len(t), i, j, means, stds) == "three-phase"
    assert m._phase_conformity((0, -0.1, 0), (0.8, 0.5, 0.2), (0.0, 0.1, 0.0)) < 3
    assert (
        m._classify((0.01, 0.05, -0.02), (0, 0, 0), 12, 3, 8, (0.2, 0.6, 0.5), (0, 0, 0)) == "drift"
    )
    good = m._bootstrap_boundaries(y, n_boot=3)
    assert good == m._bootstrap_boundaries(y, n_boot=3)
    assert set(good) == set(m._bootstrap_boundaries(np.array([0.1, 0.2]), n_boot=3))
    assert good["cp1_lo"] == good["cp1_hi"] == i
    assert good["cp2_lo"] == good["cp2_hi"] == j


def test_initial_positive_acf_does_not_return_full_ess():
    m = module(69)
    tau = m._tau_int(m._acf(np.arange(8.0), 7))
    assert tau > 1
    assert m._eff_n(8, tau) < 8
    assert math.isnan(m._tau_int(m._acf(np.ones(8), 7)))


def write_tsv(path, rows):
    with path.open("w") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_writer_reversals_and_latex_rows(tmp_path, monkeypatch):
    m = module(61)
    monkeypatch.setattr(m, "PAPER_SEC", tmp_path)
    strata = tmp_path / "strata.tsv"
    preds = tmp_path / "preds.tsv"
    write_tsv(strata, [dict(stratum="HIGH_zvf", n_anchors=2, n_anchors_attempted=3)])
    for verdict in ("PASS", "FAIL"):
        write_tsv(preds, [dict(prediction_id="P1_test", pass_fail=verdict)])
        output = m.write_section(None, strata, None, None, preds, None).read_text()
        assert f"P1\\_test: {verdict}" in output
        assert "2 \\\\\n" in output
    m53 = module(53)
    monkeypatch.setattr(m53, "PAPER_SEC", tmp_path)
    for flag in (True, False):
        result = m53.write_paper_section(*([0.5] * 14), [], [], flag, flag, flag, [1, 2])
        assert result.read_text().count("PASS" if flag else "FAIL") == 3


@pytest.mark.parametrize("number", [61, 65, 69])
def test_synthetic_pipeline_writes_only_explicit_output(tmp_path, monkeypatch, number):
    m = module(number)
    source = tmp_path / "source"
    trace_dir = source / "experiments/tinker-runs/results"
    trace_dir.mkdir(parents=True)
    models = {}
    for idx in range(6):
        name = f"Fixture-{idx}"
        fname = f"{idx}.json"
        t = np.arange(1.0, 13.0 + idx)
        rewards = (0.3 + 0.05 * idx) * (1 - np.exp(-(0.1 + 0.03 * idx) * t))
        (trace_dir / fname).write_text(
            json.dumps(
                dict(
                    reward_trace=rewards.tolist(),
                    step_log=[dict(reward=float(r), zvf=0.05 if idx % 2 else 0.3) for r in rewards],
                )
            )
        )
        models[name] = dict(file=fname, params=2.0**idx, arch="dense" if idx % 2 else "moe")
    if number == 65:
        (trace_dir / "short.json").write_text(json.dumps(dict(reward_trace=[0.1, 0.2])))
        models = {"Short-first": dict(file="short.json", params=1, arch="dense"), **models}
    monkeypatch.setattr(m, "MODELS", models)
    if hasattr(m, "N_BOOT"):
        monkeypatch.setattr(m, "N_BOOT", 2)
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "out"
    m.main(output_dir=output, source_root=source)
    assert list(output.glob("*.tsv"))
    assert list((output / "sections").glob("*.tex"))
    assert before == {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    if number == 61:
        checks = list(
            csv.DictReader((output / "scaling_law_iter61_predictions.tsv").open(), delimiter="\t")
        )
        assert checks[-1]["pass_fail"] == "UNAVAILABLE"


def test_failed_stratum_fit_excludes_denominator(tmp_path, monkeypatch):
    m = module(61)
    monkeypatch.setattr(m, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(m, "MODELS", {"one": dict(file="unused")})
    monkeypatch.setattr(m, "_load_trace", lambda name: [0.1, 0.2, 0.3])
    monkeypatch.setattr(m, "_fit_sat", lambda *args: (math.nan,) * 4)
    path = tmp_path / "proxy.tsv"
    write_tsv(path, [dict(model="one", zvf_per_step_mean=0.2)])
    rows = list(csv.DictReader(m.artefact_stratum_fit(path).open(), delimiter="\t"))
    assert rows[0]["n_anchors_attempted"] == "1"
    assert rows[0]["n_anchors"] == "0"
    assert math.isnan(float(rows[0]["frac_at_bound"]))


def test_spectral_saturation_fits_original_space():
    m = module(69)
    t = np.arange(1.0, 21.0)
    maximum, rate, _ = m._fit_saturation(0.7 * (-np.expm1(-0.2 * t)))
    assert maximum == pytest.approx(0.7, abs=1e-5)
    assert rate == pytest.approx(0.2, abs=1e-5)


def test_rank_pipeline_preserves_ties_and_unavailable(tmp_path):
    m = module(53)
    source = tmp_path / "source"
    results = source / "experiments/results"
    results.mkdir(parents=True)
    out = tmp_path / "out"
    out.mkdir()
    loo = [
        dict(
            model=f"Fixture-{i}",
            R_max_actual=0.5,
            R_max_predicted=0.5,
            residual=0,
            abs_residual=0,
            params_B=2**i,
            log10_P=float(i),
            log10_C=float(i) + 1,
        )
        for i in range(6)
    ]
    phase = [
        dict(
            model=r["model"],
            phase_classifier="other",
            peak_step=3,
            peak_frac=0.5,
            n_steps=6,
            mean_reward=0.5,
            var_reward=0.1,
        )
        for r in loo
    ]
    write_tsv(out / "scaling_law_iter49_loo_residuals.tsv", loo)
    write_tsv(out / "scaling_law_iter45_compute_proxy_v2.tsv", loo)
    write_tsv(results / "scaling_law_iter33_phase_score.tsv", phase)
    m.main(output_dir=out, source_root=source)
    ranks = list(csv.DictReader((out / "scaling_law_iter53_rank.tsv").open(), delimiter="\t"))
    assert all(float(r["rank_actual"]) == 3.5 for r in ranks)
    assert all(float(r["rank_diff"]) == 0 for r in ranks)
    assert (out / "sections/scaling_law_iter53_review27_v3.tex").read_text().count(
        "UNAVAILABLE"
    ) == 3
