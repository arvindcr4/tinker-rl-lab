"""Offline synthetic counterexamples, never benchmark results."""

import json
import pytest
from eval_apex_agents import validate_native_score
from e1_swe_bench_pro_full_eval import _load_image_map
from e9_mle_bench_streaming import _money, LaunchGateError
from e13_openreward_games_local_runner import (
    parse_split_manifest,
    prove_seed_separation,
    build_receipt,
    ProgrammaticRewardVerifier,
    EpisodeRecord,
)
from e11_verilog_eval_local_runner import sweep_all_problems


@pytest.mark.parametrize("score", [None, True, "1", float("nan"), float("inf"), -float("inf")])
def test_native_score_rejects_malformed(score):
    with pytest.raises(RuntimeError):
        validate_native_score({"scoring_results": {"final_score": score}})


@pytest.mark.parametrize("grades", [[], {"scoring_results": []}, {}])
def test_native_score_rejects_bad_schema(grades):
    with pytest.raises(RuntimeError):
        validate_native_score(grades)


def test_native_score_preserves_native_scale():
    assert validate_native_score({"scoring_results": {"final_score": 75.5}}) == 75.5


@pytest.mark.parametrize(
    "record", [None, [], {"instance_id": "x", "immutable_uri": "r@sha256:x", "digest": "sha256:x"}]
)
def test_image_record_rejected(tmp_path, record):
    p = tmp_path / "images.json"
    p.write_text(json.dumps({"images": [record]}))
    with pytest.raises(RuntimeError):
        _load_image_map(p)


def test_image_inventory_exact(tmp_path):
    p = tmp_path / "images.json"
    digest = "sha256:" + "a" * 64
    p.write_text(
        json.dumps(
            {"images": [{"instance_id": "x", "immutable_uri": "r@" + digest, "digest": digest}]}
        )
    )
    assert set(_load_image_map(p, ["x"])) == {"x"}
    for ids in (["y"], ["x", "y"], ["x", "x"], []):
        with pytest.raises(RuntimeError):
            _load_image_map(p, ids)


@pytest.mark.parametrize("amount", ["NaN", "Infinity", "-Infinity", True])
def test_money_finite(amount):
    with pytest.raises(LaunchGateError):
        _money(amount, "budget")


def manifest(split, seed):
    return parse_split_manifest(
        {
            "environment": "EnvCommons/wordle",
            "split": split,
            "source_revision": "a" * 40,
            "synthetic": True,
            "tasks": [{"id": split, "env_id": "wordle", "seed": seed}],
        }
    )


def test_disjoint_wrong_wordle_ranges():
    assert not prove_seed_separation(manifest("train", 10001), manifest("test", 10002)).holds
    assert not prove_seed_separation(manifest("train", 1), manifest("test", 2)).holds


def test_rejected_episode_stays_in_population():
    train, evaluation = manifest("train", 1), manifest("test", 10001)
    outcome = ProgrammaticRewardVerifier().verify(
        EpisodeRecord(evaluation.tasks[0], 1, False, None)
    )
    receipt = build_receipt(
        lane="E13",
        suite="wordle",
        status="BLOCKED",
        separation=prove_seed_separation(train, evaluation),
        outcomes=[outcome],
    )
    assert (
        receipt["assigned_task_count"]
        == receipt["episodes_verified"]
        == receipt["episodes_rejected"]
        == 1
    )
    assert receipt["score"] is None


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_workers_fail_before_reading_checkout(tmp_path, workers):
    with pytest.raises(ValueError, match="workers"):
        sweep_all_problems(
            tmp_path, iverilog=tmp_path, vvp=tmp_path, verilator=tmp_path, workers=workers
        )


@pytest.mark.parametrize(
    "dirty", [" M grading/runner/main.py", "M  grading/runner/main.py", "?? grading/extra.py"]
)
def test_native_checkout_rejects_dirty_files(tmp_path, monkeypatch, dirty):
    import eval_apex_agents as apex

    for name in ("README.md", "LICENSE", "grading/runner/main.py", "grading/pyproject.toml"):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")

    def output(command, **kwargs):
        return dirty if "status" in command else apex.ARCHIPELAGO_REVISION

    monkeypatch.setattr(apex.subprocess, "check_output", output)
    assert apex._native_verifier_gate(tmp_path).status == "BLOCKED"
