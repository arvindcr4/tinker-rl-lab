"""Versioned measurement, statistic, and reporting contract regressions."""

import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from utils import stats


def contract(seeds=(1, 2)):
    return {
        "schema": "stats-measurement-v2",
        "metric": "accuracy",
        "unit": "fraction",
        "split": "heldout",
        "endpoint_step": 30,
        "expected_seeds": list(seeds),
        "protocol_id": "tiny-v2",
    }


def row(score=0.8, step=30):
    return {
        "accuracy": score,
        "step": step,
        "unit": "fraction",
        "split": "heldout",
        "protocol_id": "tiny-v2",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "empty",
        "missing_seed",
        "extra_seed",
        "early",
        "reordered",
        "missing_step",
        "unit",
        "split",
        "protocol",
        "mixed_metric",
        "percent",
    ],
)
def test_seed_measurements_must_share_one_contract(mutation):
    rows = {1: [row()], 2: [row(0.6)]}
    if mutation == "empty":
        rows[2] = []
    elif mutation == "missing_seed":
        rows.pop(2)
    elif mutation == "extra_seed":
        rows[3] = [row()]
    elif mutation == "early":
        rows[2][0]["step"] = 29
    elif mutation == "reordered":
        rows[2] = [row(step=30), row(step=29)]
    elif mutation == "missing_step":
        rows[2][0].pop("step")
    elif mutation == "mixed_metric":
        rows[2][0]["reward/mean"] = rows[2][0].pop("accuracy")
    elif mutation == "percent":
        rows[2][0]["accuracy"] = 80
    else:
        rows[2][0][mutation if mutation != "protocol" else "protocol_id"] = "different"
    with pytest.raises(ValueError, match="seed|metric|step|contract|measurement"):
        stats.contract_scores(rows, contract())


@pytest.mark.parametrize(
    "rows",
    [
        {1: [{"accuracy": 0.8}], 2: [{"eval/percent_correct": 80.0}]},
        {1: [{"accuracy": 0.8}], 2: []},
        {1: [{"accuracy": 0.8, "step": 30}], 2: [{"accuracy": 0.8, "step": 29}]},
    ],
)
def test_even_legacy_exploratory_mode_never_mixes_or_omits_seeds(rows):
    with pytest.raises(ValueError, match="seed|metric|step|measurement"):
        stats.contract_scores(rows)


def test_contract_cli_records_population_metric_and_endpoint(tmp_path):
    root, output = tmp_path / "data", tmp_path / "output"
    for seed, score in ((1, 0.2), (2, 0.6)):
        directory = root / "demo" / f"seed_{seed}"
        directory.mkdir(parents=True)
        (directory / "metrics.jsonl").write_text(json.dumps(row(score)) + "\n")
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract()))
    stats.main(
        [
            "--results-dir",
            str(root),
            "--output-dir",
            str(output),
            "--contract",
            str(contract_path),
            "--bootstrap-samples",
            "50",
        ]
    )
    record = pd.read_csv(output / "results_table.csv").iloc[0]
    assert record["mean"] == pytest.approx(0.4)
    assert record["metric"] == "accuracy"
    assert record["unit"] == "fraction"
    assert record["endpoint_step"] == 30
    assert record["protocol_id"] == "tiny-v2"
    assert record["expected_seeds"] == record["included_seeds"] == "[1, 2]"
    assert record["excluded_seeds"] == "[]"
    assert record["population_status"] == "contract_complete"


def test_missing_measurement_contract_rejected_by_default(tmp_path):
    with pytest.raises(SystemExit) as error:
        stats.main(["--results-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")])
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("values", [[0, 0, 0, 1, 2, 3, 100, 100], [0, 0, 0, 2, 20], [1, 2, 3], [2]])
@pytest.mark.parametrize("tau", [0, 0.1, 0.25, 0.49])
def test_iqm_uses_independent_sorted_rank_oracle(values, tau):
    ranked = sorted(values)
    trim = int(len(ranked) * tau)
    middle = ranked[trim : len(ranked) - trim]
    expected = sum(middle) / len(middle)
    assert stats.compute_iqm(values, tau) == expected
    if values == [0, 0, 0, 1, 2, 3, 100, 100] and tau == 0.25:
        assert expected == 1.5


@pytest.mark.parametrize("tau", [-0.1, 0.5, float("nan"), float("inf"), True])
def test_iqm_rejects_invalid_trim(tau):
    with pytest.raises(ValueError, match="tau"):
        stats.compute_iqm([1, 2], tau)


@pytest.mark.parametrize("values", [[float("inf")], [float("nan")], [[1, 2]], [], 1.0])
def test_bootstrap_rejects_nonfinite_empty_or_multidimensional_units(values):
    with pytest.raises(ValueError, match="requires"):
        stats.compute_bootstrap_ci(values)


@pytest.mark.parametrize("count", [0, -1, True, 1.5, None, "3"])
def test_bootstrap_rejects_invalid_resample_count(count):
    with pytest.raises(ValueError, match="positive integer"):
        stats.compute_bootstrap_ci([1, 2], n_bootstrap=count)


def test_new_reporting_does_not_claim_zero_seed_uncertainty(tmp_path):
    table = stats.generate_results_table(
        {"only_one": np.array([0.8])}, str(tmp_path / "result.tex")
    )
    assert table.iloc[0]["95% CI"] == "unavailable (n < 2)"
    assert "SE unavailable" in table.iloc[0]["Accuracy (mean ± SE)"]
    # Historical numerical helper remains explicitly separate from report policy.
    assert stats.compute_bootstrap_ci([0.8]) == (0.8, 0.8, 0.8)


def test_hash_seed_metadata_never_claims_runtime_activation():
    script = """import json, random, sys
sys.modules['torch'] = None
sys.modules['tensorflow'] = None
from utils.seed import set_global_seed
info = set_global_seed(42)
print(json.dumps({'hash': hash('review09-startup'), 'random': random.random(), 'info': info}))
"""

    def run(seed):
        output = subprocess.check_output(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            text=True,
        )
        return json.loads(output)

    first, again, other = run(1), run(1), run(2)
    assert first == again
    assert first["hash"] != other["hash"]
    assert first["random"] == other["random"]
    assert first["info"]["python_hash_seed"] is None
    assert first["info"]["python_hash_seed_environment"] == "1"
    assert first["info"]["python_hash_seed_requested"] == "42"
    assert first["info"]["python_hash_seed_status"] == "unverified_startup_configuration"


def test_nonempty_experiment_beside_empty_experiment_is_not_silently_omitted(tmp_path):
    root = tmp_path / "data"
    (root / "empty").mkdir(parents=True)
    (root / "valid/seed_1").mkdir(parents=True)
    (root / "valid/seed_1/metrics.jsonl").write_text('{"accuracy": .8}'.replace(".8", "0.8"))
    with pytest.raises(SystemExit):
        stats.main(
            [
                "--results-dir",
                str(root),
                "--legacy-exploratory",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    assert not (tmp_path / "out/results_table.csv").exists()


def test_independent_signflip_multiplicity_and_nondegenerate_tost_decisions():
    from tools import audit_thesis_numbers as audit

    assert audit.signflip_p([1, 1, 1, 1, 1]) == 2 / 32
    raw = [0.01, 0.04, 0.03]
    assert audit.holm(raw) == pytest.approx([0.03, 0.06, 0.06])
    assert [p < 0.05 for p in raw] == [True, True, True]
    assert [p < 0.05 for p in audit.holm(raw)] == [True, False, False]
    # df=3 Student-t integral, fixed independent numerical reference.
    t = math.sqrt(120)
    expected = 0.5 - (math.atan(t / math.sqrt(3)) + t * math.sqrt(3) / (t * t + 3)) / math.pi
    assert audit.tost_p([-0.02, -0.01, 0.01, 0.02], 0.1) == pytest.approx(expected, abs=1e-12)
