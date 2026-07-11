from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS_NEXT = REPO / "zvf-program" / "experiments-next"


def load_script_module(name: str):
    path = EXPERIMENTS_NEXT / f"{name}.py"
    sys.path.insert(0, str(EXPERIMENTS_NEXT))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXPERIMENTS_NEXT))


quality = load_script_module("analyze_rollout_quality")
seed_aggregate = load_script_module("aggregate_seed_audits")
passk = load_script_module("passk_eval")
passk_compare = load_script_module("compare_passk_results")


def synthetic_pool(with_lengths: bool = True):
    prompts = [
        {"idx": 0, "rewards": [0.0, 1.0], "p_hat": 0.5},
        {"idx": 1, "rewards": [0.0, 0.0], "p_hat": 0.0},
    ]
    if with_lengths:
        prompts[0]["token_counts"] = [10, 20]
        prompts[1]["token_counts"] = [30, 40]
    return {
        "kind": "zvf_pool",
        "status": "complete",
        "tag": "synthetic",
        "model": "test/model",
        "split": "test",
        "seed": 42,
        "temperature": 1.0,
        "top_p": 1.0,
        "rollouts_per_prompt": 2,
        "prompts": prompts,
    }


def test_quality_audit_uses_prompt_level_signal_and_length_diagnostics():
    audit = quality.analyze(synthetic_pool(), n_bootstrap=100, seed=7)
    core = audit["core_metrics"]

    assert core["pass_at_1"] == pytest.approx(0.25)
    assert core["zero_variance_prompt_rate"] == pytest.approx(0.5)
    assert core["active_advantage_fraction"] == pytest.approx(0.5)
    assert core["mean_group_reward_variance"] == pytest.approx(0.125)
    assert audit["prompt_clustered_bootstrap_95_ci"]["pass_at_1"][
        "resampling_unit"
    ] == "prompt"

    length = audit["length_diagnostics"]
    assert length["available"] is True
    assert length["prompt_coverage"] == 1.0
    assert length["mean_tokens_correct"] == pytest.approx(20.0)
    assert length["mean_tokens_incorrect"] == pytest.approx(80.0 / 3.0)
    assert length["length_predictive_auc"] == pytest.approx(1.0 / 3.0)


def test_legacy_pool_never_imputes_missing_lengths():
    audit = quality.analyze(
        synthetic_pool(with_lengths=False), n_bootstrap=100, seed=7
    )
    assert audit["length_diagnostics"]["available"] is False
    assert "lack token_counts" in audit["length_diagnostics"]["reason"]


def test_seed_aggregate_requires_distinct_compatible_evaluation_seeds():
    base = quality.analyze(synthetic_pool(), n_bootstrap=100, seed=7)
    audits = []
    for evaluation_seed, pass_at_1 in zip((42, 43, 44), (0.2, 0.3, 0.4)):
        audit = copy.deepcopy(base)
        audit["source"]["seed"] = evaluation_seed
        audit["core_metrics"]["pass_at_1"] = pass_at_1
        audits.append(audit)

    result = seed_aggregate.aggregate(audits, seed=9, n_bootstrap=100)
    assert result["n_evaluation_seeds"] == 3
    assert result["core_metrics"]["pass_at_1"]["mean"] == pytest.approx(0.3)
    assert "not independent training-seed" in result["methodology"]["scope_warning"]

    audits[2]["source"]["seed"] = 43
    with pytest.raises(ValueError, match="duplicate evaluation seeds"):
        seed_aggregate.aggregate(audits, seed=9, n_bootstrap=100)


def test_pool_builder_dry_run_never_requires_credentials():
    completed = subprocess.run(
        [
            sys.executable,
            "build_pool.py",
            "--model",
            "test/model",
            "--prompts",
            "2",
            "--rollouts",
            "2",
            "--tag",
            "unit-test-dry-run",
            "--dry-run",
        ],
        cwd=EXPERIMENTS_NEXT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "DRY RUN" in completed.stdout


def test_passk_summary_and_offline_audit_are_problem_clustered(tmp_path):
    point, intervals = passk.summarize_pass_at_k(
        [0, 2], n=2, ks=[1, 2], n_bootstrap=100, seed=3
    )
    assert point == {"1": pytest.approx(0.5), "2": pytest.approx(0.5)}
    assert intervals["1"]["resampling_unit"] == "problem"

    source = tmp_path / "passk.json"
    output = tmp_path / "audit.json"
    source.write_text(json.dumps({
        "kind": "passk_eval",
        "status": "complete",
        "tag": "synthetic",
        "model": "test/model",
        "which": "base",
        "split": "test",
        "seed": 42,
        "n_per_problem": 2,
        "ks": [1, 2],
        "per_problem_c": [0, 2],
    }))
    completed = subprocess.run(
        [
            sys.executable,
            "passk_eval.py",
            "--from-result",
            str(source),
            "--out",
            str(output),
            "--bootstrap",
            "100",
        ],
        cwd=EXPERIMENTS_NEXT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text())
    assert payload["pass_at_k"]["1"] == pytest.approx(0.5)
    assert payload["pass_at_k_95_ci"]["2"]["resampling_unit"] == "problem"


def test_paired_passk_comparison_requires_verified_matching_prompts():
    base = {
        "kind": "passk_eval",
        "status": "complete",
        "model": "test/model",
        "which": "base",
        "split": "test",
        "seed": 42,
        "n_problems": 2,
        "n_per_problem": 2,
        "ks": [1, 2],
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 32,
        "max_prompt_tokens": 64,
        "prompt_fingerprints": ["a", "b"],
        "per_problem_c": [0, 1],
    }
    post = copy.deepcopy(base)
    post["which"] = "postrl"
    post["per_problem_c"] = [1, 2]

    result = passk_compare.compare(base, post, n_bootstrap=100, seed=5)
    assert result["pairing_verified_by_prompt_fingerprint"] is True
    assert result["metrics"]["1"]["paired_delta"] == pytest.approx(0.5)
    assert result["metrics"]["1"]["resampling_unit"] == "paired problem"

    post["prompt_fingerprints"] = ["a", "different"]
    with pytest.raises(ValueError, match="prompt fingerprints differ"):
        passk_compare.compare(base, post, n_bootstrap=100, seed=5)
