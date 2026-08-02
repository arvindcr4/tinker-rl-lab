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
        "decoder": copy.deepcopy(PREFLIGHT.DECODER_CONTRACT),
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
        "completion_clipped_fraction": 0.0,
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
        "decoder": copy.deepcopy(request["decoder"]),
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
        (
            lambda manifest, result, request: manifest["audit_record"].update(
                completion_clipped_fraction=1.0
            ),
            "clipping",
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


def test_validate_args_accepts_recovery_without_task_and_arm():
    args = PREFLIGHT.parse_args(["--recover-request", "request.json"])
    PREFLIGHT.validate_args(args)


def test_validate_args_rejects_mixed_launch_and_recovery_modes():
    args = PREFLIGHT.parse_args(
        ["--task", "gsm8k", "--arm", "grpo_g8", "--recover-request", "request.json"]
    )
    with pytest.raises(SystemExit, match="recovery mode uses the request artifact identity"):
        PREFLIGHT.validate_args(args)


def test_recovery_mode_reconstructs_completed_receipt_from_request(monkeypatch, tmp_path):
    manifest, result, request = valid_payload()
    request.update(
        {
            "schema_version": "aiml-next-preflight-request-v1",
            "status": "launching",
            "provider": "colab",
            "hardware_flavor": "A100",
            "gpu": "A100",
            "auth_strategy": "oauth2",
            "colab_cli_version": "0.6.0",
            "session": "exact-session",
            "hf_repo": "arvindcr4/tinker-rl-next-preflight-gsm8k-grpo_g8-s211-aaaaaaaa",
        }
    )
    output_dir = tmp_path / "preflight"
    request_dir = output_dir / "requests"
    request_dir.mkdir(parents=True)
    _, _, request_path, log_path = PREFLIGHT.result_paths(output_dir, request)
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(request), encoding="utf-8")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("transport ended before marker\n", encoding="utf-8")

    def read_json(path):
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def atomic_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        PREFLIGHT,
        "load_e1_helpers",
        lambda: SimpleNamespace(
            read_json=read_json,
            atomic_json=atomic_json,
            load_credentials=lambda: {"HF_TOKEN": "hf", "WANDB_API_KEY": "wandb"},
        ),
    )
    monkeypatch.setattr(PREFLIGHT, "verify_tracking_credentials", lambda credentials, hf_repo_prefix: {"hf_identity": "arvindcr4", "wandb_identity": "tester"})
    monkeypatch.setattr(PREFLIGHT, "result_from_log_or_remote", lambda lines, credentials, recovered_request: (result, {"source": "exact-private-hf-commit-and-finished-wandb-run", "reason": "missing marker"}))
    monkeypatch.setattr(PREFLIGHT, "verify_remote", lambda credentials, recovered_result, recovered_request: (manifest, {"hf_private": True, "hf_repo": recovered_request["hf_repo"], "hf_commit": "e" * 40, "hf_files": ["run_manifest.json", "final/adapter_model.safetensors"], "wandb": {"state": "finished"}}))
    monkeypatch.setattr(PREFLIGHT.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="[colab] No active sessions found on server.\n"))

    args = PREFLIGHT.parse_args(
        ["--recover-request", str(request_path), "--output-dir", str(output_dir)]
    )
    PREFLIGHT.validate_args(args)
    status = PREFLIGHT.recover_request_artifact(args)

    receipt = json.loads((output_dir / "results" / "gsm8k__contrast_early_stop_g2_to_g8__s211.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert receipt["status"] == "completed"
    assert receipt["recovery"]["source"] == "exact-private-hf-commit-and-finished-wandb-run"
    assert receipt["cleanup"]["session_absent_verified"] is True
    assert receipt["request_path"] == str(request_path)


def test_recovery_mode_stops_live_session_and_records_failed_recovery(monkeypatch, tmp_path):
    _, _, request = valid_payload()
    request.update(
        {
            "schema_version": "aiml-next-preflight-request-v1",
            "status": "launching",
            "provider": "colab",
            "hardware_flavor": "A100",
            "gpu": "A100",
            "auth_strategy": "oauth2",
            "colab_cli_version": "0.6.0",
            "session": "exact-session",
            "hf_repo": "arvindcr4/tinker-rl-next-preflight-gsm8k-grpo_g8-s211-aaaaaaaa",
        }
    )
    output_dir = tmp_path / "preflight"
    _, _, request_path, log_path = PREFLIGHT.result_paths(output_dir, request)
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(request), encoding="utf-8")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("transport ended before marker\n", encoding="utf-8")

    def read_json(path):
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def atomic_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        PREFLIGHT,
        "load_e1_helpers",
        lambda: SimpleNamespace(
            read_json=read_json,
            atomic_json=atomic_json,
            load_credentials=lambda: {"HF_TOKEN": "hf", "WANDB_API_KEY": "wandb"},
        ),
    )
    monkeypatch.setattr(PREFLIGHT, "verify_tracking_credentials", lambda credentials, hf_repo_prefix: {"hf_identity": "arvindcr4", "wandb_identity": "tester"})
    monkeypatch.setattr(PREFLIGHT, "result_from_log_or_remote", lambda lines, credentials, recovered_request: (_ for _ in ()).throw(RuntimeError("no remote artifacts found")))
    responses = iter(
        [
            SimpleNamespace(returncode=0, stdout="exact-session A100 BUSY\n"),
            SimpleNamespace(returncode=0, stdout="stopped\n"),
            SimpleNamespace(returncode=0, stdout="[colab] No active sessions found on server.\n"),
        ]
    )
    monkeypatch.setattr(PREFLIGHT.subprocess, "run", lambda *args, **kwargs: next(responses))

    args = PREFLIGHT.parse_args(
        ["--recover-request", str(request_path), "--output-dir", str(output_dir)]
    )
    status = PREFLIGHT.recover_request_artifact(args)

    receipt = json.loads((output_dir / "results" / "gsm8k__contrast_early_stop_g2_to_g8__s211.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert receipt["status"] == "failed"
    assert receipt["failed_step"] == "recovery"
    assert receipt["cleanup"]["attempts"] == 1
    assert receipt["cleanup"]["session_absent_verified"] is True
    assert "no remote artifacts found" in receipt["error"]


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


def test_transport_interrupt_is_converted_to_a_failed_step():
    calls = []
    lines = []

    def run_logged(command, log_handle, output_lines):
        calls.append(command)
        if len(calls) == 1:
            return 0
        raise KeyboardInterrupt

    return_code, failed_step, allocation_started = PREFLIGHT.execute_commands(
        [["colab", "new"], ["colab", "exec"]],
        run_logged=run_logged,
        log_handle=object(),
        lines=lines,
    )

    assert return_code == 1
    assert failed_step == 1
    assert allocation_started is True
    assert lines == ["transport exception: KeyboardInterrupt:"]


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


def seam_payload(arm="contrast_early_stop_g2_to_g8", groups=6, mixed=1):
    """A seam_verification payload: the widened window, everything else frozen."""
    manifest, result, request = copy.deepcopy(valid_payload())
    cap, max_steps = PREFLIGHT.seam_window("seam_verification", arm)
    request["arm"] = arm
    request["preflight_class"] = "seam_verification"
    for config in (manifest["run_config"], result["run_config"]):
        config["arm"] = arm
        config["max_steps"] = max_steps
        config["optimizer_steps"] = max_steps
        config["preflight_class"] = "seam_verification"
        config["rollout_groups_cap"] = cap
    generated = groups * 2 + mixed * 6 if arm == "contrast_early_stop_g2_to_g8" else groups * 8
    for audit in (manifest["audit_record"], result["audit_record"]):
        audit["arm_id"] = arm
        audit["rollout_groups"] = groups
        audit["updated_groups"] = mixed
        audit["generated_rollouts"] = generated
        audit["charged_generated_tokens"] = max(generated, 100)
        audit["mixed_fraction"] = mixed / groups
        audit["all_wrong_fraction"] = 1 - mixed / groups
        audit["all_correct_fraction"] = 0.0
    return manifest, result, request


def test_launcher_and_remote_seam_windows_agree():
    """The launcher validates the window the remote run actually executes."""
    remote_dir = ROOT / "zvf-program/next-submission"
    remote_path = remote_dir / "remote_preflight.py"
    spec = importlib.util.spec_from_file_location("next_submission_remote", remote_path)
    assert spec is not None and spec.loader is not None
    remote = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(remote_dir))  # remote_preflight imports its siblings
    try:
        spec.loader.exec_module(remote)
    finally:
        sys.path.remove(str(remote_dir))
    assert remote.SEAM_ROLLOUT_GROUP_CAP == PREFLIGHT.SEAM_ROLLOUT_GROUP_CAP
    assert remote.SEAM_OPTIMIZER_STEP_CAP == PREFLIGHT.SEAM_OPTIMIZER_STEP_CAP
    assert remote.PREFLIGHT_CLASSES == PREFLIGHT.PREFLIGHT_CLASSES
    for preflight_class in PREFLIGHT.PREFLIGHT_CLASSES:
        for arm in PREFLIGHT.ARMS:
            assert remote.seam_window(preflight_class, arm) == PREFLIGHT.seam_window(
                preflight_class, arm
            )


def test_seam_window_stays_within_the_amendment_caps():
    assert PREFLIGHT.seam_window("matrix_infrastructure", "grpo_g8") == (2, 1)
    assert PREFLIGHT.seam_window("seam_verification", "grpo_g8") == (24, 12)
    assert PREFLIGHT.seam_window("seam_verification", "contrast_early_stop_g2_to_g8") == (48, 24)
    for arm in PREFLIGHT.ARMS:
        cap, steps = PREFLIGHT.seam_window("seam_verification", arm)
        assert steps <= PREFLIGHT.SEAM_OPTIMIZER_STEP_CAP
        assert cap < 60  # one confirmatory unit is 30 steps x 2 groups


def test_valid_seam_verification_preflight_contract_passes():
    PREFLIGHT.validate_manifest(*seam_payload())


def test_seam_verification_rollout_groups_cannot_exceed_the_cap():
    manifest, result, request = seam_payload(groups=6)
    for audit in (manifest["audit_record"], result["audit_record"]):
        audit["rollout_groups"] = 49
        audit["updated_groups"] = 1
        audit["mixed_fraction"] = 1 / 49
        audit["all_wrong_fraction"] = 48 / 49
        audit["generated_rollouts"] = 49 * 2 + 6
        audit["charged_generated_tokens"] = 200
    with pytest.raises(RuntimeError, match="cap"):
        PREFLIGHT.validate_manifest(manifest, result, request)


def test_seam_verification_cannot_run_the_matrix_window_max_steps():
    manifest, result, request = seam_payload()
    manifest["run_config"]["max_steps"] = 1
    result["run_config"]["max_steps"] = 1
    with pytest.raises(RuntimeError, match="run config mismatch"):
        PREFLIGHT.validate_manifest(manifest, result, request)


def test_seam_receipt_does_not_overwrite_the_matrix_receipt(tmp_path):
    base = {"task": "gsm8k", "arm": "grpo_g8", "seed": 223, "fingerprint": "f" * 64}
    matrix_unit, matrix_path, _, _ = PREFLIGHT.result_paths(tmp_path, base)
    seam_unit, seam_path, _, _ = PREFLIGHT.result_paths(
        tmp_path, {**base, "preflight_class": "seam_verification"}
    )
    assert matrix_unit == "gsm8k__grpo_g8__s223"
    assert seam_unit == "gsm8k__grpo_g8__s223__seam_verification"
    assert matrix_path != seam_path


def test_preflight_class_defaults_to_matrix_infrastructure():
    args = PREFLIGHT.parse_args(["--task", "gsm8k", "--arm", "grpo_g8", "--seed", "223"])
    assert args.preflight_class == "matrix_infrastructure"
    manifest, result, request = copy.deepcopy(valid_payload())
    request.pop("preflight_class", None)
    PREFLIGHT.validate_manifest(manifest, result, request)
