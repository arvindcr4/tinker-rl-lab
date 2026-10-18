import importlib.util
from pathlib import Path
import subprocess
import sys
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "platform_modal/scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("budget", [1000000, 4000000, 16000000, 64000000])
@pytest.mark.parametrize("old_block", ["", "\n# ITER_63_RETENTION_FITS\nold data\n"])
def test_iter63_preserves_every_historical_byte(tmp_path, budget, old_block):
    m = load("group_size_iter63")
    m.EFFECT_TSV = tmp_path / "historical.tsv"
    original = f"budget\tmetric\n{budget}.0\t0.4\n" + old_block
    m.EFFECT_TSV.write_text(original)
    m.append_to_effect_tsv(
        [(budget, 0.8, 0.7, 0.9)],
        [0],
        (0.8, 0, 1, (0, 0, 0)),
        (0.8, 0, 1, (0, 0, 0)),
        (0.8, 0.8, 1e6, 1, (0, 0, 0), (0, 0, 0)),
        False,
        False,
    )
    assert m.EFFECT_TSV.read_text() == original
    output = tmp_path / "group_size_iter63_retention_fits_v2.tsv"
    assert len(output.read_text().splitlines()) == 2


@pytest.mark.parametrize("destination", ["existing", "source", "child"])
def test_anonymizer_refuses_unsafe_destinations(tmp_path, destination):
    dest = tmp_path / "valuable"
    dest.mkdir()
    (dest / "keep").write_text("precious")
    out = {
        "existing": dest,
        "source": SCRIPTS.parents[1],
        "child": SCRIPTS.parents[1] / "unsafe-new-copy",
    }[destination]
    proc = subprocess.run(
        ["bash", str(SCRIPTS / "anonymize.sh"), str(out)], cwd=tmp_path, capture_output=True
    )
    assert proc.returncode != 0
    assert (dest / "keep").read_text() == "precious"


def test_tost_direction():
    m = load("group_size_iter43")
    assert m.tost_pvalue(0, -0.01, 0.01, 0.05) < 0.05
    assert m.tost_pvalue(-0.1, -0.11, -0.09, 0.05) > 0.95


@pytest.mark.parametrize(("small", "expected"), [(0.4, False), (0.79, True)])
def test_native_retention_direction(small, expected):
    m = load("group_size_iter135")
    runs = [
        {"seed": seed, "group_size": g, "heldout_acc": value}
        for seed in [1, 2, 3]
        for g, value in [(2, small), (16, 0.8)]
    ]
    result = m.native_wu_claim({"runs": runs})
    assert f"wu_native_claim_holds={expected}" in result[1]["headline"]


def test_nonmonotone_crossing():
    m = load("group_size_iter59")
    assert (
        1
        < m.find_T_for_acc(
            [{"G": 4, "T": t, "acc": a} for t, a in [(1, 0.4), (4, 0.8), (16, 0.6)]], 4, 0.7
        )
        < 4
    )


def test_no_threshold_crossing():
    m = load("group_size_iter39")
    assert m.t_critical_exponential(0.99, 1) == float("inf")


def test_impossible_fixed_marginal():
    m = load("group_size_iter23")
    rows = m.anti_herding_simulation()
    assert (
        rows.loc[rows.p_eff == 0.8, "feasible_alpha"].iloc[0] == "unattainable_at_fixed_marginal_p"
    )


def test_contamination_requires_real_inputs(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "contamination_check.py")], cwd=tmp_path, capture_output=True
    )
    assert result.returncode != 0
    assert b"actual --train-texts" in result.stderr


def test_paired_seed_means_ignore_input_order():
    m = load("group_size_iter15")
    runs = [
        {"group_size": g, "seed": s, "step_log": [{"mean_reward": v + g / 100}]}
        for g in m.G_VALUES
        for s, v in [(1, 0.2), (2, 0.8), (3, 0.5)]
    ]
    a = m.per_seed_last10_mean_reward({"runs": runs})
    b = m.per_seed_last10_mean_reward({"runs": list(reversed(runs))})
    for g in m.G_VALUES:
        assert (a[g] == b[g]).all()
    with pytest.raises(ValueError, match="duplicate group/seed 2/1"):
        m.per_seed_last10_mean_reward({"runs": runs + [runs[0]]})


def test_reward_quadratic_matches_seeds():
    m = load("group_size_iter135")
    runs = [
        {
            "group_size": g,
            "seed": s,
            "mean_reward_train": v + g / 100,
            "last10_avg": v + g / 100,
            "mean_zvf": 0.4,
        }
        for g in [2, 4, 8, 16]
        for s, v in [(1, 0.2), (2, 0.5), (3, 0.7)]
    ]
    a = m.reward_quadratic({"runs": runs})
    b = m.reward_quadratic({"runs": list(reversed(runs))})
    assert a == b


def test_checksum_missing_and_short_corpora_not_pass(tmp_path):
    m = load("contamination_check")
    (tmp_path / "result.json").write_text("{}")
    assert m.check_results_integrity(str(tmp_path))["flag"] == "INCOMPLETE"
    assert m.check_ngram_overlap([], [])["flag"] == "INCOMPLETE"
    assert m.check_ngram_overlap(["same short"], ["same short"])["flag"] == "FAIL"
    assert m.check_ngram_overlap(["a b"], ["c d"])["flag"] == "INCOMPLETE"


def test_yaml_validation_never_prints_partial_rows(tmp_path, capsys):
    m = load("_reviewer_points_extract")
    file = tmp_path / "review.yaml"
    file.write_text("weaknesses:\n  - id: first\n    marker: ok\n  - id: second\n")
    assert m.main(str(file)) == 1
    assert capsys.readouterr().out == ""


def test_retention_model_recovers_exponential():
    import numpy as np

    m = load("group_size_iter39")
    t = np.array([1e6, 4e6, 16e6, 64e6])
    y = 0.6 + 0.4 * np.exp(-t / 10e6)
    r, tau, sse = m.fit_exponential(t, y)
    assert r == pytest.approx(0.6, abs=1e-5)
    assert tau == pytest.approx(10e6, rel=1e-5)
    assert sse < 1e-12


def test_convergence_uses_logged_steps(tmp_path):
    import csv

    m = load("group_size_elevated")
    m.RESULTS = tmp_path
    path = m.write_convergence_tsv(
        {
            "runs": [
                {
                    "group_size": 4,
                    "seed": 1,
                    "step_log": [
                        {"step": 10, "mean_reward": 0.1},
                        {"step": 20, "mean_reward": 0.6},
                    ],
                }
            ]
        }
    )
    rows = list(csv.DictReader(path.open(), delimiter="\t"))
    assert rows[0]["first_step"] == "20"
    assert rows[-1]["first_step"] == "never"
    assert rows[-1]["observation_horizon"] == "20"


def test_native_retention_reference_self_is_identically_one():
    m = load("group_size_iter27")
    runs = [
        {"group_size": g, "seed": s, "step_log": [{"mean_reward": v}]}
        for g in m.G_VALUES
        for s, v in [(1, 0.3), (2, 0.8), (3, 0.6)]
    ]
    df = m.per_g_master({"runs": runs})
    assert (df["retention_ci_low"] == 1).all()
    assert (df["retention_ci_high"] == 1).all()


def test_pareto_sign_not_trend():
    import pandas as pd

    m = load("group_size_iter23")
    rows = [
        {"G": g, "T_tokens": t, "acc": a} for t in [1, 4, 16, 64] for g, a in [(4, 0.3), (32, 0.8)]
    ]
    result = m.pareto_crossings(pd.DataFrame(rows)).iloc[0]
    assert result["direction"] == "G=32 dominates G=4"
    assert result["T_crossover_tokens"] == 1


def test_unbounded_crossing_interval_preserved():
    import numpy as np

    m = load("group_size_iter39")
    assert m.bootstrap_ci(np.array([1, 2, float("inf")]))[1] == float("inf")


def test_ratio_above_one_not_clipped(tmp_path):
    m = load("group_size_iter63")
    m.EFFECT_TSV = tmp_path / "historical.tsv"
    m.append_to_effect_tsv(
        [(1e6, 1.2, 1.1, 1.3)],
        [0],
        (1.2, 0, 1, (0, 0, 0)),
        (1.2, 0, 1, (0, 0, 0)),
        (1.2, 1.2, 1e6, 1, (0, 0, 0), (0, 0, 0)),
        False,
        False,
    )
    assert (
        "1.2" in (tmp_path / "group_size_iter63_retention_fits_v2.tsv").read_text().splitlines()[1]
    )


def test_anonymization_scans_binary_names_and_symlinks(tmp_path):
    m = load("_anonymization_validate")
    (tmp_path / "innocent.bin").write_bytes(b"\x00ArvindCR\x00")
    (tmp_path / "arvindcr4.txt").write_text("ordinary")
    (tmp_path / "link").symlink_to(tmp_path / "innocent.bin")
    errors = m.validate(tmp_path)
    assert len(errors) == 3
    assert any("content" in e for e in errors)
    assert any("filename" in e for e in errors)
    assert any("symlink" in e for e in errors)


def test_threshold_refits_loaded_evidence():
    m = load("group_size_iter135")
    rows = [
        {"T": 10**t, "G": g, "acc": (0.9 - 0.1 * t if g == 4 else 1)}
        for t in [1, 2, 3]
        for g in [4, 32]
    ]
    out = m.threshold_tstar(rows)
    assert "T=10000" in out[0]["headline"]
    assert "input_sha256=" in out[0]["headline"]
    changed = [dict(row, acc=row["acc"] - 0.1) if row["G"] == 4 else row for row in rows]
    assert m.threshold_tstar(changed)[0]["headline"] != out[0]["headline"]
    with pytest.raises(ValueError, match="duplicate budget/group"):
        m.threshold_tstar(rows + rows)


def test_compute_cost_scenario_and_gu_association():
    m = load("group_size_iter115")
    rows = [
        {
            "budget_tokens": t,
            "G": g,
            "heldout_acc_mean": 0.1 * (i + 1) * (1 if g == 4 else 2),
            "gu_estimate": (i + 1 if g == 4 else 1),
        }
        for i, t in enumerate([100, 1000, 10000])
        for g in [4, 32]
    ]
    cost = m.compute_cost(rows, tokens_per_gpu_hour=100, usd_per_million_tokens=3)
    assert all(
        r["usd_per_million_tokens"] == 3 and r["cost_kind"] == "unvalidated_scenario" for r in cost
    )
    assert all("extrapolated_G4" in r for r in cost)
    assert m.zvf_linkage(rows)[-1]["budget_tokens"] == "SPEARMAN_GU_RATIO"
    with pytest.raises(ValueError, match="invalid cost scenario"):
        m.compute_cost(rows, tokens_per_gpu_hour=0)


def test_isog_separates_observed_and_projected(tmp_path):
    import csv

    m = load("group_size_elevated")
    m.RESULTS = tmp_path
    path = m.write_isog_sizing_tsv(
        {
            "runs": [
                {"group_size": g, "step_log": [{"mean_reward": 0.5, "zvf": 0.8}]} for g in [2, 4]
            ]
        }
    )
    with path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert rows
    assert all(r["G_min_observed"] == "unavailable" for r in rows)
    assert all("G_min_projected" in r for r in rows)
