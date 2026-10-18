import math
import pytest
import importlib.util
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[2] / "platform_modal/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_observations_are_distinct_and_reject_duplicates():
    m = load("length_bias_iter136_fig")
    rows = [{"task": "t", "algo": "grpo", "seed": s, "value": v} for s, v in [(1, 0.1), (2, 0.9)]]
    result = m.seed_observations(rows, {"h": ("value", "smaller")})
    assert result[("t", "grpo", 1, "h")] == 0.1
    assert result[("t", "grpo", 2, "h")] == 0.9
    with pytest.raises(ValueError, match="duplicate observation"):
        m.seed_observations(rows + rows, {"h": ("value", "smaller")})


def test_separate_output_grains_and_sparse_row(tmp_path):
    m = load("length_bias_iter56")
    m.RES = tmp_path
    m.write_grains(
        "cumulative_tax",
        [{"seed": 1, "tax": 0}, {"seed": 2, "tax": 1, "R_first": 0.3}],
        [{"metric": "tax", "n_pairs": 2, "ci_lo": 0}],
    )
    run = (tmp_path / "length_bias_iter56_cumulative_tax_per_run_v2.tsv").read_text()
    comparison = (tmp_path / "length_bias_iter56_cumulative_tax_comparison_v2.tsv").read_text()
    assert "R_first" in run
    assert "n_pairs" not in run
    assert "n_pairs" in comparison
    assert "seed" not in comparison
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    ("a", "b", "expected"), [([1, 1], [2, 2], -1), ([2, 2], [1, 1], 1), ([1, 1], [1, 1], 0)]
)
def test_signed_area(a, b, expected):
    assert load("length_bias_iter68").trajectory_auc(a, b) == expected


@pytest.mark.parametrize(
    ("reward", "identified"),
    [([4, 1, 0, 1, 4], False), ([-4, -1, 0, -1, -4], True), ([1, 1, 1, 1, 1], False)],
)
def test_quadratic_requires_negative_curvature(reward, identified):
    fit = load("length_bias_iter60").fit_quadratic_R_of_L(reward, [-2, -1, 0, 1, 2])
    assert fit["optimum_identified"] is identified
    assert math.isfinite(fit["L_star"]) is identified


@pytest.mark.parametrize("z", [float("nan"), float("inf"), -0.1, 1.1])
def test_missing_zvf_not_high(z):
    assert load("length_bias_iter52")._zvf_bin(z) == "unavailable"


def test_auc_step_grid_and_summary_not_double_counted():
    m = load("length_bias_iter68")
    m.B = 100
    assert m.trajectory_auc([2, 2, 2], [1, 1, 1], [0, 2, 7]) == 7
    assert m.signed_auc_summary([3, 3, 3])[:3] == (-3, -3, -3)
    with pytest.raises(ValueError, match="steps must increase strictly"):
        m.trajectory_auc([1, 2], [1, 2], [2, 1])
    with pytest.raises(ValueError, match="paired trajectories must have equal lengths"):
        m.trajectory_auc([1, 2], [1])


def test_first_divergence_is_descriptive_not_self_contrast():
    m = load("length_bias_iter68")
    m.B = 100
    runs = {
        (algo, seed): [
            {"mean_comp_len": v + offset, "mean_reward": 0.1 * i, "step": i * 2}
            for i, v in enumerate([2, 5, 3, 7, 4])
        ]
        for algo, offset in [("a", 0), ("b", 1)]
        for seed in [1, 2, 3]
    }
    row = m.analyse("synthetic", runs, "a", "b")["summary"][0]
    assert math.isnan(row["p_le0"])
    assert math.isnan(row["mean_diff"])
    assert math.isfinite(row["estimate"])


def test_curvature_writer_reports_unidentified_optima(tmp_path, monkeypatch):
    import csv

    m = load("length_bias_iter60")
    m.RES = tmp_path
    m.N_BOOT = 30

    def synthetic(path, task):
        return [
            {
                "task": task,
                "algo": algo,
                "seed": seed,
                "L": [1, 2, 3, 4, 5],
                "R": [0.01 * x * x for x in [1, 2, 3, 4, 5]],
                "t": [0, 1, 2, 3, 4],
            }
            for algo in ["grpo", "dr_grpo"]
            for seed in [1, 2, 3]
        ]

    monkeypatch.setattr(m, "load_runs", synthetic)
    m.main()
    with (tmp_path / "length_bias_iter60_pair_coverage_v2.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    optima = [r for r in rows if r["metric"] == "L_star"]
    assert len(optima) == 2
    assert all(r["n_excluded"] == "3" and r["n_valid"] == "0" for r in optima)
    with (tmp_path / "length_bias_iter60_curvature_v2.tsv").open() as f:
        assert all(r["optimum_identified"] == "False" for r in csv.DictReader(f, delimiter="\t"))
