from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "zvf-program/next-submission/run_preflight.py"
SPEC = importlib.util.spec_from_file_location("next_submission_preflight", PATH)
assert SPEC is not None and SPEC.loader is not None
PREFLIGHT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREFLIGHT
SPEC.loader.exec_module(PREFLIGHT)

SECURE_PATH = ROOT / "zvf-program/next-submission/secure_exec_preflight.py"
SECURE_SPEC = importlib.util.spec_from_file_location("next_submission_secure_exec", SECURE_PATH)
assert SECURE_SPEC is not None and SECURE_SPEC.loader is not None
SECURE = importlib.util.module_from_spec(SECURE_SPEC)
sys.modules[SECURE_SPEC.name] = SECURE
SECURE_SPEC.loader.exec_module(SECURE)


def valid_payload():
    request = {
        "task": "gsm8k",
        "arm": "contrast_early_stop_g2_to_g8",
        "seed": 211,
        "fingerprint": "a" * 64,
        "stack_fingerprint": "b" * 64,
        "source_commit": "c" * 40,
        "protocol_sha256": "d" * 64,
        "gpu": "L4",
        "provider": "huggingface_jobs",
        "hardware_flavor": "l4x1",
        "runtime_packages": list(PREFLIGHT.PACKAGE_PINS),
        "tracking": {
            "hf_repo_prefix": "arvindcr4/tinker-rl-next-preflight",
            "wandb_project": "tinker-rl-lab",
            "wandb_entity": None,
            "wandb_group": "next-submission-preflight",
        },
        "tracking_packages": {"trackio": "0.34.0"},
    }
    audit = {
        "task_id": "gsm8k",
        "arm_id": "contrast_early_stop_g2_to_g8",
        "seed": 211,
        "status": "complete",
        "evidence_tier": "preflight_not_scientific_evidence",
        "heldout_correct": 4,
        "heldout_n": 8,
        "heldout_accuracy": 0.5,
        "charged_generated_tokens": 100,
        "generated_rollouts": 10,
        "rollout_groups": 2,
        "updated_groups": 1,
        "all_wrong_fraction": 0.5,
        "all_correct_fraction": 0.0,
        "mixed_fraction": 0.5,
    }
    run_config = {
        "task": request["task"],
        "arm": request["arm"],
        "seed": request["seed"],
        "unit_fingerprint": request["fingerprint"],
        "stack_fingerprint": request["stack_fingerprint"],
        "source_commit": request["source_commit"],
        "protocol_sha256": request["protocol_sha256"],
        "max_steps": 1,
        "heldout_n": 8,
        "provider": request["provider"],
        "hardware_flavor": request["hardware_flavor"],
        "tracking_backends": ["wandb", "trackio"],
        "tracking": {
            "wandb_project": request["tracking"]["wandb_project"],
            "wandb_entity": request["tracking"]["wandb_entity"],
            "wandb_group": request["tracking"]["wandb_group"],
        },
    }
    trace = [
        {"index": index, "correct": index < 4, "completion_sha256": f"{index:064x}"}
        for index in range(8)
    ]
    source_files = {
        path.name: PREFLIGHT.sha256_file(path)
        for path in (PREFLIGHT.REMOTE_SCRIPT, PREFLIGHT.SAMPLER, PREFLIGHT.TRL_ADAPTER)
    }
    manifest = {
        "schema_version": "aiml-next-preflight-run-v1",
        "status": "complete",
        "evidence_class": "preflight-not-evidence",
        "audit_record": audit,
        "run_config": copy.deepcopy(run_config),
        "heldout_trace": trace,
        "source_files": source_files,
        "runtime_versions": {
            "trl": "1.8.0",
            "transformers": "5.13.1",
            "jinja2": "3.1.6",
            "datasets": "4.8.5",
            "peft": "0.19.1",
            "torchao": "0.17.0",
            "wandb": "0.28.0",
            "huggingface_hub": "1.26.0",
            "trackio": "0.34.0",
            "gpu": "NVIDIA L4",
        },
    }
    result = {
        "schema_version": "aiml-next-preflight-result-v1",
        "status": "completed",
        "evidence_class": "preflight-not-evidence",
        "audit_record": audit,
        "run_config": run_config,
        "runtime_versions": manifest["runtime_versions"],
    }
    return manifest, result, request


def test_valid_intervention_preflight_contract_passes():
    PREFLIGHT.validate_manifest(*valid_payload())


def test_preflight_cannot_be_promoted_to_scientific_evidence():
    manifest, result, request = copy.deepcopy(valid_payload())
    result["evidence_class"] = "confirmatory"
    with pytest.raises(RuntimeError, match="promoted"):
        PREFLIGHT.validate_manifest(manifest, result, request)


def test_intervention_generation_formula_is_enforced():
    manifest, result, request = copy.deepcopy(valid_payload())
    manifest["audit_record"]["generated_rollouts"] = 16
    result["audit_record"]["generated_rollouts"] = 16
    with pytest.raises(RuntimeError, match="G2-to-G8"):
        PREFLIGHT.validate_manifest(manifest, result, request)


def test_source_hash_drift_is_rejected():
    manifest, result, request = copy.deepcopy(valid_payload())
    manifest["source_files"]["contrast_sampler.py"] = "0" * 64
    with pytest.raises(RuntimeError, match="source hashes"):
        PREFLIGHT.validate_manifest(manifest, result, request)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda manifest, result, request: manifest["audit_record"].update(status="running"),
            "status",
        ),
        (
            lambda manifest, result, request: manifest["audit_record"].update(task_id="math500"),
            "identity",
        ),
        (
            lambda manifest, result, request: manifest["audit_record"].update(updated_groups=99),
            "updated-group",
        ),
        (
            lambda manifest, result, request: manifest["audit_record"].update(
                charged_generated_tokens=1
            ),
            "charged generated tokens",
        ),
    ],
)
def test_semantically_invalid_audit_receipts_are_rejected(mutation, match):
    manifest, result, request = copy.deepcopy(valid_payload())
    mutation(manifest, result, request)
    result["audit_record"] = copy.deepcopy(manifest["audit_record"])
    with pytest.raises(RuntimeError, match=match):
        PREFLIGHT.validate_manifest(manifest, result, request)


def test_result_run_config_must_match_manifest():
    manifest, result, request = copy.deepcopy(valid_payload())
    result["run_config"]["task"] = "math500"
    with pytest.raises(RuntimeError, match="run configs differ"):
        PREFLIGHT.validate_manifest(manifest, result, request)


def test_observed_gpu_and_versions_must_match_request():
    manifest, result, request = copy.deepcopy(valid_payload())
    manifest["runtime_versions"]["gpu"] = "NVIDIA T4"
    result["runtime_versions"] = copy.deepcopy(manifest["runtime_versions"])
    with pytest.raises(RuntimeError, match="does not match requested"):
        PREFLIGHT.validate_manifest(manifest, result, request)

    manifest, result, request = copy.deepcopy(valid_payload())
    manifest["runtime_versions"]["trl"] = "1.9.0"
    result["runtime_versions"] = copy.deepcopy(manifest["runtime_versions"])
    with pytest.raises(RuntimeError, match="package mismatch"):
        PREFLIGHT.validate_manifest(manifest, result, request)


def test_provider_specific_runtime_pins_are_enforced():
    manifest, result, request = copy.deepcopy(valid_payload())
    request["runtime_packages"] = [
        "jinja2==3.1.7" if pin.startswith("jinja2==") else pin
        for pin in request["runtime_packages"]
    ]
    with pytest.raises(RuntimeError, match="package mismatch"):
        PREFLIGHT.validate_manifest(manifest, result, request)


def test_result_log_parser_uses_last_marker():
    payload = {"schema_version": "aiml-next-preflight-result-v1"}
    lines = ["noise", "NEXT_PREFLIGHT_RESULT " + __import__("json").dumps(payload)]
    assert PREFLIGHT.result_from_log(lines) == payload


def test_missing_result_marker_falls_back_to_remote_recovery(monkeypatch):
    _, expected, request = valid_payload()
    calls = []

    def recover(credentials, recovered_request):
        calls.append((credentials, recovered_request))
        return expected

    monkeypatch.setattr(PREFLIGHT, "recover_result_from_remote", recover)
    result, recovery = PREFLIGHT.result_from_log_or_remote(
        ["scientific child completed without a streamed marker"],
        {"HF_TOKEN": "hf", "WANDB_API_KEY": "wandb"},
        request,
    )

    assert result == expected
    assert calls == [
        (
            {"HF_TOKEN": "hf", "WANDB_API_KEY": "wandb"},
            request,
        )
    ]
    assert recovery == {
        "reason": "remote output did not contain a NEXT_PREFLIGHT_RESULT record",
        "source": "exact-private-hf-commit-and-finished-wandb-run",
    }


def test_failure_summary_is_bounded_and_strips_ansi():
    lines = [f"line {index}" for index in range(20)] + ["\x1b[31mquota rejected\x1b[0m"]
    summary = PREFLIGHT.failure_summary(lines, limit=3)
    assert summary == "line 18 | line 19 | quota rejected"


def test_hardware_retry_archives_incompatible_receipt(tmp_path):
    existing = {"fingerprint": "a" * 64, "status": "failed", "gpu": "A100"}

    def write(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    path = PREFLIGHT.archive_incompatible_result(
        output_dir=tmp_path,
        unit="gsm8k__adaptive__s211",
        existing=existing,
        new_fingerprint="b" * 64,
        atomic_json=write,
    )

    assert path == tmp_path / "results/history/gsm8k__adaptive__s211__aaaaaaaaaaaa.json"
    assert __import__("json").loads(path.read_text(encoding="utf-8")) == existing


def test_colab_plan_uses_bounded_setup_and_stages_secrets_after_environment_check():
    args = PREFLIGHT.parse_args(["--task", "gsm8k", "--arm", "contrast_early_stop_g2_to_g8"])
    plan = PREFLIGHT.build_execution_plan(
        args,
        session="exact-session",
        secret_source="<secret>",
        invocation_source="<request>",
    )

    assert all("install" not in command for command in plan)
    assert str(PREFLIGHT.SETUP_SCRIPT) in plan[4]
    assert plan[4][-2:] == ["--timeout", "1800"]
    assert plan[5][-2:] == ["<request>", "/content/next-preflight-request.json"]
    assert str(PREFLIGHT.ENVIRONMENT_CHECK) in plan[6]
    assert plan[7][-2:] == ["<secret>", "/content/.next-preflight-secrets.json"]
    assert str(PREFLIGHT.SECURE_EXEC) in plan[8]
    assert plan[9][-3:] == ["stop", "--session", "exact-session"]


def test_cleanup_retries_until_server_enumeration_proves_absence(monkeypatch, tmp_path):
    responses = iter(
        [
            SimpleNamespace(returncode=1, stdout="transient stop error\n"),
            SimpleNamespace(returncode=0, stdout="exact-session A100 BUSY\n"),
            SimpleNamespace(returncode=0, stdout="stopped\n"),
            SimpleNamespace(returncode=0, stdout="[colab] No active sessions found on server.\n"),
        ]
    )
    monkeypatch.setattr(PREFLIGHT.subprocess, "run", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(PREFLIGHT.time, "sleep", lambda seconds: None)

    log_path = tmp_path / "cleanup.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        receipt = PREFLIGHT.stop_session_verified(
            "oauth2",
            "exact-session",
            log_handle,
            attempts=2,
            delay_seconds=0,
        )

    assert receipt == {
        "attempts": 2,
        "stop_return_codes": [1, 0],
        "sessions_return_codes": [0, 0],
        "session_absent_verified": True,
    }


def test_secure_exec_deletes_files_and_keeps_secrets_out_of_persistent_environment(
    capsys, monkeypatch, tmp_path
):
    secret_path = tmp_path / "secrets.json"
    request_path = tmp_path / "request.json"
    script_path = tmp_path / "child.py"
    secret_path.write_text(
        json.dumps({"HF_TOKEN": "child-hf-token", "WANDB_API_KEY": "child-wandb-key"}),
        encoding="utf-8",
    )
    request_path.write_text(
        json.dumps({"expected_gpu": "A100", "script_args": ["expected-argument"]}),
        encoding="utf-8",
    )
    script_path.write_text(
        "import os, sys\n"
        "assert os.environ['HF_TOKEN'] == 'child-hf-token'\n"
        "assert os.environ['WANDB_API_KEY'] == 'child-wandb-key'\n"
        "assert sys.argv[1:] == ['expected-argument']\n"
        "print('NEXT_PREFLIGHT_RESULT streamed-marker', flush=True)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_TOKEN", "parent-hf-token")
    monkeypatch.setenv("WANDB_API_KEY", "parent-wandb-key")

    assert (
        SECURE.main(
            secret_path=secret_path,
            request_path=request_path,
            script_path=script_path,
        )
        == 0
    )
    assert not secret_path.exists()
    assert not request_path.exists()
    assert os.environ["HF_TOKEN"] == "parent-hf-token"
    assert os.environ["WANDB_API_KEY"] == "parent-wandb-key"
    assert "NEXT_PREFLIGHT_RESULT streamed-marker" in capsys.readouterr().out


def test_recovery_reconstructs_result_only_from_complete_private_artifacts(monkeypatch, tmp_path):
    manifest, _, request = valid_payload()
    commit = "e" * 40
    repo = PREFLIGHT.expected_hf_repo(request)
    manifest["wandb"] = {
        "run_id": "run12345",
        "run_url": "https://wandb.ai/test-entity/tinker-rl-lab/runs/run12345",
    }
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class FakeApi:
        private = True

        def model_info(self, repo_id, *, files_metadata):
            assert repo_id == repo
            assert files_metadata is True
            return SimpleNamespace(
                private=self.private,
                sha=commit,
                siblings=[
                    SimpleNamespace(rfilename="run_manifest.json"),
                    SimpleNamespace(rfilename="final/adapter_model.safetensors"),
                ],
            )

    fake_api = FakeApi()
    monkeypatch.setattr(PREFLIGHT, "HfApi", lambda token: fake_api)
    monkeypatch.setattr(PREFLIGHT, "hf_hub_download", lambda **kwargs: str(manifest_path))

    recovered = PREFLIGHT.recover_result_from_remote(
        {"HF_TOKEN": "hf", "WANDB_API_KEY": "wandb"},
        request,
    )

    assert recovered["status"] == "completed"
    assert recovered["evidence_class"] == "preflight-not-evidence"
    assert recovered["remote"] == {
        "hf_repo": repo,
        "hf_commit": commit,
        "hf_manifest_path": "run_manifest.json",
        "hf_final_adapter_path": "final/adapter_model.safetensors",
        "wandb_run_id": "run12345",
        "wandb_run_url": manifest["wandb"]["run_url"],
    }

    fake_api.private = False
    with pytest.raises(RuntimeError, match="non-private"):
        PREFLIGHT.recover_result_from_remote(
            {"HF_TOKEN": "hf", "WANDB_API_KEY": "wandb"},
            request,
        )


def test_remote_verification_requires_exact_private_hf_commit_and_finished_wandb(
    monkeypatch, tmp_path
):
    manifest, result, request = valid_payload()
    commit = "e" * 40
    repo = PREFLIGHT.expected_hf_repo(request)
    result["remote"] = {
        "hf_repo": repo,
        "hf_commit": commit,
        "wandb_run_id": "run12345",
        "wandb_run_url": "https://wandb.ai/test-entity/tinker-rl-lab/runs/run12345",
    }
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class FakeApi:
        private = True

        def file_exists(self, **kwargs):
            return kwargs["revision"] == commit

        def model_info(self, repo_id, *, revision, files_metadata):
            assert repo_id == repo
            assert revision == commit
            assert files_metadata is True
            return SimpleNamespace(
                private=self.private,
                sha=commit,
                siblings=[
                    SimpleNamespace(rfilename="run_manifest.json"),
                    SimpleNamespace(rfilename="final/adapter_model.safetensors"),
                ],
            )

    fake_api = FakeApi()
    monkeypatch.setattr(PREFLIGHT, "HfApi", lambda token: fake_api)
    monkeypatch.setattr(PREFLIGHT, "hf_hub_download", lambda **kwargs: str(manifest_path))
    monkeypatch.setattr(
        PREFLIGHT,
        "verify_wandb_run",
        lambda *args, **kwargs: {
            "entity": "test-entity",
            "project": "tinker-rl-lab",
            "run_id": "run12345",
            "state": "finished",
            "run_url": result["remote"]["wandb_run_url"],
        },
    )

    _, verification = PREFLIGHT.verify_remote(
        {"HF_TOKEN": "hf", "WANDB_API_KEY": "wandb"},
        result,
        request,
    )
    assert verification["hf_private"] is True
    assert verification["hf_commit"] == commit
    assert verification["wandb"]["state"] == "finished"

    fake_api.private = False
    with pytest.raises(RuntimeError, match="not private"):
        PREFLIGHT.verify_remote(
            {"HF_TOKEN": "hf", "WANDB_API_KEY": "wandb"},
            result,
            request,
        )
