"""Small offline adversarial runner fixtures. No model, Docker, or provider calls."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest
import swe_bench_pro_eval as swe
import public_omni_math_native as omni
import public_portfolio_native as portfolio
import public_core_bench_native as core
import run_harbor_via_tinker_bridge as harbor


@pytest.mark.parametrize(
    "extra",
    [
        {"max_completion_tokens": 8192},
        {"truncate_prompt_tokens": 1},
        {"chat_template_kwargs": {"enable_thinking": True}},
    ],
)
def test_omni_rejects_protocol_overrides(monkeypatch, extra):
    monkeypatch.setattr(omni, "validate_manifest", lambda x: None)
    monkeypatch.setattr(omni, "validate_provenance", lambda *x: None)
    monkeypatch.setattr(omni.durable, "validate_wandb", lambda x: None)
    with pytest.raises(portfolio.ContractError, match="protocol overrides"):
        omni.create_contract({}, "actor", {}, {}, extra)


@pytest.mark.parametrize(
    "extra", [{"max_completion_tokens": 8192}, {"truncate_prompt_tokens": 1}, {"seed": 10}]
)
def test_portfolio_rejects_protocol_overrides(monkeypatch, extra):
    monkeypatch.setattr(portfolio, "validate_manifest", lambda x: None)
    monkeypatch.setattr(portfolio, "validate_provenance", lambda x: None)
    monkeypatch.setattr(portfolio, "validate_wandb", lambda x: None)
    sampling = {
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 2048,
        "seed": 1,
        "n": 1,
        "extra_body": extra,
    }
    with pytest.raises(portfolio.ContractError, match="protocol overrides"):
        portfolio.create_contract({}, {}, {}, sampling, "http://localhost")


def test_harbor_rejects_overrides_before_health_or_credentials(monkeypatch):
    monkeypatch.setattr(harbor, "fetch_health", lambda *a: pytest.fail("must not query health"))
    with pytest.raises(harbor.LaunchGateError, match="overrides"):
        harbor.main(["--lane", "E4", "--agent", "claude-code"])


def test_harbor_does_not_inherit_unrelated_credentials(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "synthetic-secret")
    env = harbor.build_harbor_env("bridge-key", harbor.LANES["E4"])
    assert "ANTHROPIC_API_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env


def test_conflicting_patch_aliases_rejected(tmp_path):
    path = tmp_path / "patches.json"
    path.write_text(json.dumps([{"instance_id": "one", "patch": "a", "model_patch": "b"}]))
    assert not swe.validate_patch_artifact(path, ["one"])["valid"]


def run_synthetic(tmp_path, monkeypatch, result_payload=None, timeout=False, patch="first"):
    path = tmp_path / "input.json"
    path.write_text(json.dumps([{"instance_id": "one", "patch": patch}]))
    receipt = {
        "launchable": True,
        "blockers": [],
        "native_environment": {"isolated": {"python": "/usr/bin/python"}},
        "benchmark": {"selected_task_ids": ["one"]},
    }
    monkeypatch.setattr(
        swe,
        "inspect_local_evaluator",
        lambda p: {"clean": True, "revision": swe.EVALUATOR_REVISION, "runner_sha256": "a" * 64},
    )

    def run(command, **kw):
        out = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--output_dir=")))
        started = list(out.glob("attempt-*.json"))
        assert len(started) >= 1
        assert json.loads(started[-1].read_text())["status"] == "STARTED"
        if timeout:
            raise subprocess.TimeoutExpired(command, 1, output=b"partial")
        (out / "eval_results.json").write_text(json.dumps(result_payload))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(swe, "_run", run)
    return swe.run_evaluation(
        receipt=receipt,
        runtime_dir=tmp_path,
        evaluator=tmp_path,
        rows=[{"instance_id": "one"}],
        patch_path=path,
    )


@pytest.mark.parametrize("payload", [{}, {"one": "false"}, {"one": False, "extra": True}])
def test_native_result_coverage_and_boolean_contract(tmp_path, monkeypatch, payload):
    result = run_synthetic(tmp_path, monkeypatch, payload)
    assert result["status"] == "FAILED"
    assert result["score"] is None


def test_native_false_is_a_valid_fixed_denominator_failure(tmp_path, monkeypatch):
    result = run_synthetic(tmp_path, monkeypatch, {"one": False})
    assert result["status"] == "COMPLETE"
    assert result["score"]["accuracy"] == 0
    assert result["evidence_class"] == "provided_patch_verification_only"


def test_changed_patch_uses_different_output_identity(tmp_path, monkeypatch):
    one = run_synthetic(tmp_path, monkeypatch, {"one": True})
    two = run_synthetic(tmp_path, monkeypatch, {"one": False}, patch="changed")
    assert one["evaluation_id"] != two["evaluation_id"]


def test_timeout_is_durable_and_does_not_release_uncertain_reservation(tmp_path, monkeypatch):
    result = run_synthetic(tmp_path, monkeypatch, timeout=True)
    assert result["launches_any_job"]
    assert result["status"] == "FAILED"
    assert result["outstanding_reservation"]
    assert result["launch"]["stdout"] == "partial"
    terminal = Path(result["launch"]["output_dir"]) / ("attempt-" + result["attempt_id"] + ".json")
    assert json.loads(terminal.read_text())["status"] == "FAILED"


def test_core_terminal_cache_cannot_replace_retained_evidence(tmp_path, monkeypatch):
    task = {"capsule_id": "one", "field": "f", "language": "python", "capsule_title": "title"}
    contract = {"model": {}, "wandb": {}, "agent": {}}
    manifest = {"tasks": [task]}
    monkeypatch.setattr(core, "prepared_manifest", lambda *a: manifest)
    monkeypatch.setattr(core, "create_contract", lambda *a: contract)
    directory = tmp_path / "tasks" / "one"
    directory.mkdir(parents=True)

    def write(path, value):
        path.write_text(json.dumps(value))

    write(tmp_path / "run_contract.json", contract)
    submission = {"task": task, "contract_sha256": core.fingerprint(contract)}
    write(directory / "submission.json", submission)
    raw_row = {**task, **{key: 0 for key in core.GRADE_KEYS}, "result_report": {}}
    raw = {"capsule_results": [raw_row]}
    write(directory / "native_result_file.json", raw)
    raw_hash = core.digest((directory / "native_result_file.json").read_bytes())
    execution = {
        "capsule_id": "one",
        "submission_sha256": core.fingerprint(submission),
        "benchmark_level": core.LEVEL,
        "source_revision": core.REVISION,
        "terminal": True,
        "outcome": "completed",
        "model_provenance_sha256": core.fingerprint({}),
        "native_results_sha256": raw_hash,
        "artifacts_sha256": {},
        "report_status": "missing_or_invalid",
    }
    terminal = {
        "capsule_id": "one",
        "submission_sha256": core.fingerprint(submission),
        "native_results_sha256": raw_hash,
        "execution_receipt": execution,
        "native_result": {**raw_row, "tampered": True},
        "original_report": {},
    }
    write(directory / "terminal.json", terminal)
    with pytest.raises(portfolio.ContractError, match="cache differs"):
        core.score(tmp_path, tmp_path, tmp_path, tmp_path / "out.json")


@pytest.mark.parametrize("mutation", ["protocol", "source_hash"])
def test_omni_recomputed_self_hash_does_not_validate_changed_protocol(
    tmp_path, monkeypatch, mutation
):
    monkeypatch.setattr(omni, "EXPECTED_TOTAL", 1)
    monkeypatch.setattr(omni, "sources", lambda p: {"actor_prompt_source": "synthetic"})
    rows = [
        {
            "problem": "one plus one",
            "answer": "2",
            "domain": "math",
            "difficulty": 1,
            "source": "synthetic",
            "solution": "2",
        }
    ]
    monkeypatch.setattr(omni, "dataset_rows", lambda p: rows)
    omni.prepare(tmp_path, tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    if mutation == "protocol":
        manifest["protocol"]["actor_max_tokens"] = 1
    else:
        manifest["tasks"][0]["source_row_sha256"] = "f" * 64
    path.write_text(json.dumps(manifest))
    keys = json.loads((tmp_path / "reference_rows.json").read_text())
    keys["manifest_sha256"] = omni.fingerprint(manifest)
    (tmp_path / "reference_rows.json").write_text(json.dumps(keys))
    with pytest.raises(portfolio.ContractError, match="protocol drift|source row hash drift"):
        omni.verified_prepared(tmp_path, tmp_path)


def test_dirty_evaluator_is_not_a_pinned_snapshot(tmp_path, monkeypatch):
    (tmp_path / "swe_bench_pro_eval.py").write_text("modified grader")
    monkeypatch.setattr(
        swe,
        "_run",
        lambda cmd, **kw: SimpleNamespace(
            returncode=0,
            stdout=" M swe_bench_pro_eval.py\n" if cmd[1] == "status" else swe.EVALUATOR_REVISION,
        ),
    )
    assert swe.inspect_local_evaluator(tmp_path)["clean"] is False


def test_old_canary_cannot_approve_reviewed_sources():
    import public_omni_actor_launch_preparation as planner

    with pytest.raises(ValueError, match="new separately authorized canary"):
        planner.source_modules()
