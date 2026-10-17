"""CPU-only reproductions for review 05. No services, models or retained result writes."""

from __future__ import annotations
import copy
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

ROOT = Path(__file__).resolve().parents[1]
ZVF = ROOT / "zvf-program"
for directory in (
    "audit",
    "sweep",
    "next-submission",
    "e1_wave10",
    "jev_lab",
    "e13_balrog",
    "zvf-triage/src",
):
    sys.path.insert(0, str(ZVF / directory))
from tests._shared_fakes import load_module
from zvf_triage import ZVFCallback, ZVFController
from zvf_triage import core
from zvf_triage.integrations.verl import VERLZVFAdapter

AUDIT = importlib.import_module("aggregate_audit")
CAMPAIGN = importlib.import_module("run_colab_e1_campaign")
CONTRACT = importlib.import_module("audit_evidence_contract")
SWEEP = importlib.import_module("run_sweep")
MATCHED = importlib.import_module("matched_compute")
ANALYSIS = importlib.import_module("head_to_head_analysis")
GCP = load_module("review05_gcp", ZVF / "next-submission/run_gcp_preflight.py")
REMOTE = load_module("review05_remote", ZVF / "next-submission/remote_preflight.py")
BPG = importlib.import_module("bounded_procgroup")
DRIVER = load_module("review05_e1_driver", ZVF / "e1_wave10/driver.py")
JEV = importlib.import_module("jev_judge")
USE = importlib.import_module("usecases")
HS = load_module("review05_supervisor", ZVF / "e13_balrog/hosted_supervisor.py")


AUDIT_EXECUTION = {
    "mode": "confirmatory",
    "model": "Qwen/Qwen3-8B",
    "task": "GSM8K",
    "max_steps": 30,
    "heldout_n": 2,
    "max_completion_length": 1024,
    "batch_prompts": 4,
}


def audit_pair():
    record = dict(
        arm="grpo",
        seed=11,
        evidence_class="confirmatory",
        fingerprint="a" * 64,
        stack_fingerprint="b" * 64,
        heldout_n=2,
        heldout_score=0.5,
        last10_reward=0.4,
        mean_zvf=0.2,
        mean_gu=0.8,
        wall_clock_seconds=1.0,
        rollouts=16,
        collapse=False,
        treatment_changes=[],
        manifest_path="manifest.json",
        remote_verification={
            "hf_repo": "owner/repo",
            "hf_commit": "d" * 40,
            "wandb": {"state": "finished"},
        },
    )
    manifest = dict(
        evidence_class="confirmatory",
        audit_record=copy.deepcopy(record),
        run_config=dict(
            **AUDIT_EXECUTION,
            arm="grpo",
            seed=11,
            treatment_changes=[],
            unit_fingerprint="a" * 64,
            stack_fingerprint="b" * 64,
        ),
        heldout_trace=[dict(index=i, correct=i == 0, completion_sha256="c" * 64) for i in range(2)],
    )
    return record, manifest


@pytest.mark.parametrize(
    "field", ["last10_reward", "mean_zvf", "mean_gu", "wall_clock_seconds", "rollouts", "collapse"]
)
def test_audit_missing_required_measurement_is_rejected_everywhere(tmp_path, monkeypatch, field):
    record, manifest = audit_pair()
    record.pop(field)
    assert CONTRACT.record_errors(
        record, manifest, heldout_n=2, expected_execution=AUDIT_EXECUTION, expected_changes=[]
    )
    expected = dict(
        execution=AUDIT_EXECUTION, heldout_n=2, checkpoint_steps=[5], treatment_changes={"grpo": []}
    )
    monkeypatch.setattr(CAMPAIGN, "read_json", lambda _: manifest)
    assert (
        CAMPAIGN.validate_result(("grpo", 11), record, expected, stack_fingerprint="b" * 64)[0]
        is False
    )


def test_audit_rejects_pilot_and_wrong_identity_even_with_valid_trace():
    record, manifest = audit_pair()
    assert (
        CONTRACT.record_errors(
            record, manifest, heldout_n=2, expected_execution=AUDIT_EXECUTION, expected_changes=[]
        )
        == []
    )
    manifest["evidence_class"] = "pilot"
    manifest["audit_record"]["seed"] = 23
    assert (
        len(
            CONTRACT.record_errors(
                record,
                manifest,
                heldout_n=2,
                expected_execution=AUDIT_EXECUTION,
                expected_changes=[],
            )
        )
        >= 2
    )


@pytest.mark.parametrize(
    "field,value",
    [("mean_zvf", 5), ("mean_gu", 3), ("rollouts", True), ("heldout_score", float("nan"))],
)
def test_audit_impossible_values_are_not_finite_only_admissible(field, value):
    record, manifest = audit_pair()
    record[field] = value
    manifest["audit_record"][field] = value
    assert CONTRACT.record_errors(
        record, manifest, heldout_n=2, expected_execution=AUDIT_EXECUTION, expected_changes=[]
    )


def test_power_backend_handles_concentrated_chi_density():
    assert AUDIT._chi_radius_expectation(75, lambda radius: 1.0) == pytest.approx(1.0, abs=1e-10)
    from scipy.stats import t, nct

    for n in (3, 8, 76, 200):
        expected = nct.sf(t.ppf(0.95, n - 1), n - 1, 0.3 * math.sqrt(n))
        assert AUDIT.paired_t_power_one_sided(0.3, n, 0.05) == pytest.approx(expected, abs=1e-12)


def test_recovery_cannot_bypass_attempt_limit():
    assert (
        CAMPAIGN.choose_launches(
            [("grpo", 11)],
            set(),
            set(),
            {"grpo:11": 100},
            capacity=1,
            max_attempts=3,
            recovery_kinds={("grpo", 11): "evaluation-recovery"},
        )
        == []
    )


def test_matched_launchers_share_effective_treatments_and_budget():
    cfg = MATCHED.load_config()
    cfg["sweep_design"]["matched_compute"].update(
        arms=["greso", "zvf_ci_gated", "dapo"], base_batch_prompts=4, max_tokens=256
    )
    plan = MATCHED.build_plan(cfg)
    generic = {c["tag"]: c for c in SWEEP.expand_manifest(cfg) if c["block"] == "matched"}
    for command in plan["commands"]:
        cell = generic[command["tag"]]
        assert SWEEP.build_command(cfg, cell)[0] == command["command"]
        assert "--batch-prompts 4" in command["command"]
        assert "--max-tokens 256" in command["command"]
        if command["arm"] in ("greso", "zvf_ci_gated"):
            assert f"--prompt-filter {command['arm']}" in command["command"]
    cfg["sweep_design"]["matched_compute"]["arms"] = ["grpo"]
    assert [a["name"] for a in MATCHED.build_plan(cfg)["arms"]] == ["grpo"]


def test_missing_command_placeholder_is_an_error():
    cfg = MATCHED.load_config()
    rc = cfg["runner"][cfg["runner"]["kind"]]
    rc["command_template"] = rc["command_template"].replace("--prompt-filter {prompt_filter}", "")
    with pytest.raises(ValueError, match="omits"):
        MATCHED.build_plan(cfg)


def test_short_sweep_result_never_passes_resume_or_postrun(tmp_path, monkeypatch):
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {"status": "completed", "step_log": [{"step": 1, "reward": 0.5, "zvf": 0.5, "gu": 0.5}]}
        )
    )
    assert SWEEP.result_json_is_complete(result, 30) is False
    assert SWEEP.result_json_is_complete(result) is False
    # No subprocess is dispatched for stale preexisting evidence.
    mocked = Mock(side_effect=AssertionError("must not dispatch"))
    monkeypatch.setattr(SWEEP.subprocess, "run", mocked)
    row = SWEEP.run_one(
        "unused",
        result,
        dict(tag="x", block="matched", framework="fake", model_short="fake", steps=30),
    )
    assert row["status"] == "blocked-attempt-reconciliation"
    mocked.assert_not_called()


def test_summary_telemetry_preserves_values_and_cannot_drop_prompts():
    cb = ZVFCallback(drop_k=3)
    adapter = cb.as_trl_callback(num_generations=8)
    control = SimpleNamespace(should_training_stop=False)
    adapter.on_log(
        None,
        SimpleNamespace(global_step=1),
        control,
        logs={"frac_reward_zero_std": 0.9, "reward": 0.96},
    )
    decision = cb.decisions[-1]
    assert (decision.zvf, decision.mean_reward) == (0.9, 0.96)
    assert decision.observation_kind == "summary_only" and not control.should_training_stop
    for step in range(2, 6):
        adapter.on_log(
            None,
            SimpleNamespace(global_step=step),
            control,
            logs={"frac_reward_zero_std": 0.5, "reward": 0.5},
        )
    assert [d.zvf for d in cb.decisions[1:]] == [0.5] * 4
    adapter.on_log(
        None,
        SimpleNamespace(global_step=5),
        control,
        logs={"frac_reward_zero_std": 1.0, "reward": 0.0},
    )
    assert len(cb.decisions) == 5
    assert cb.dropped_prompts == []


def test_diagnostic_population_is_not_censored_by_recommendations():
    controller = ZVFController(drop_k=3, stop_k=3)
    ids = [0, 0, 1, 1]
    for _ in range(2):
        controller.step([0, 0, 1, 1], ids, prompt_ids=ids)
    for _ in range(3):
        decision = controller.step([0, 0, 0, 0], ids, prompt_ids=ids)
    assert decision.auto_stop and decision.zvf == 1 and decision.filtered_rewards == 0
    assert ZVFController().step([], []).auto_stop


def test_invalid_rewards_do_not_mutate_history():
    controller = ZVFController()
    with pytest.raises(ValueError):
        controller.step([0, float("nan")], [0, 0])
    assert controller.zvf_history == [] and controller._step == 0
    with pytest.raises(ValueError):
        core.zvf([float("inf")], [0])


def test_batch_positions_are_not_stable_prompts():
    controller = ZVFController(drop_k=1)
    assert controller.step([0, 0], [0, 0]).dropped_prompts == []
    assert controller.step([0, 1], [0, 0]).zvf == 0


def test_verl_fit_fallback_is_rejected():
    adapter = VERLZVFAdapter()
    adapter.verl_available = True
    trainer = SimpleNamespace(fit=lambda: None)
    with pytest.raises(AttributeError):
        adapter.install(trainer)
    trainer.fit()


@pytest.mark.parametrize(
    "response", ["#### 42.9", "#### 42/7", "#### not 42", "#### 4,2", "#### 42 blah"]
)
def test_integer_prefix_grading_counterexamples(response):
    assert REMOTE.gsm8k_reward(response, "#### 42") == 0


def test_strict_integer_grammar_and_last_marker():
    assert REMOTE.parse_marked_integer("reason #### 1 then #### +1,234") == 1234
    assert REMOTE.parse_marked_integer("#### -42") == -42


def test_gcp_auth_failures_are_not_verified_absence(monkeypatch):
    monkeypatch.setattr(
        GCP.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="authentication error", stderr=""),
    )
    assert GCP.instance_status("gcloud", instance="unit", project="p", zone="z") == "LOOKUP_FAILED"
    receipt = GCP.delete_instance("gcloud", instance="unit", project="p", zone="z")
    assert receipt["instance_absent_verified"] is False


def test_gcp_not_found_is_bound_to_exact_resource(monkeypatch):
    error = {
        "error": {
            "code": 404,
            "status": "NOT_FOUND",
            "resource": "projects/p/zones/z/instances/unit",
        }
    }
    monkeypatch.setattr(
        GCP.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout=json.dumps(error), stderr=""),
    )
    assert GCP.instance_status("gcloud", instance="unit", project="p", zone="z") == "ABSENT"
    assert GCP.instance_status("gcloud", instance="other", project="p", zone="z") == "LOOKUP_FAILED"


def test_gcp_alternate_framework_fails_before_allocation(monkeypatch):
    monkeypatch.setattr(
        GCP, "load_common", Mock(side_effect=AssertionError("should reject locally"))
    )
    with pytest.raises(ValueError, match="no allocation"):
        GCP.run_unit(SimpleNamespace(framework="verl"))


def test_stale_e1_result_is_never_returned(tmp_path):
    path = tmp_path / "result.json"
    path.write_text('{"old":true}')
    outcome = BPG.run_child(
        [sys.executable, "-c", "pass"],
        result_path=path,
        receipt_dir=tmp_path,
        deadline_epoch=__import__("time").time() + 3,
    )
    assert outcome["status"] == "UNKNOWN" and outcome["result"] is None


def test_native_worker_parser_forwards_evaluator_argv_and_records_failure(tmp_path):
    path = tmp_path / "result.json"
    process = subprocess.run(
        [
            sys.executable,
            str(ZVF / "e1_wave10/driver.py"),
            "--native-worker",
            "--native-result-path",
            str(path),
            "--native-stdout",
            str(tmp_path / "out"),
            "--native-stderr",
            str(tmp_path / "err"),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(7)",
        ],
        env={**os.environ, "TINKER_ATTEMPT_NONCE": "synthetic-attempt"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert process.returncode == 0, process.stderr  # receipt-writing worker succeeded
    result = json.loads(path.read_text())
    assert result["returncode"] == 7 and result["attempt_nonce"] == "synthetic-attempt"


def test_generation_dispatch_is_exclusively_claimed(tmp_path, monkeypatch):
    deployment = tmp_path / "deployment.json"
    deployment.write_text("{}")
    source = {"actor_task": {key: "test" for key in DRIVER.psml.ACTOR_FIELDS}, "files": {}}
    args = SimpleNamespace(
        max_tokens=16,
        temperature=0,
        seed=1,
        endpoint="fake",
        api_key=None,
        request_timeout=1,
        wandb_run_id="fake",
        deployment=deployment,
    )
    monkeypatch.setattr(
        DRIVER.psml, "build_actor_request", lambda *a, **k: {"model": "fake", "messages": []}
    )
    dispatch = Mock(side_effect=RuntimeError("uncertain network outcome"))
    monkeypatch.setattr(DRIVER, "post_chat_completion", dispatch)
    for _ in range(2):
        with pytest.raises((RuntimeError, FileExistsError)):
            DRIVER.generate_one(
                "id",
                source,
                {"served_model_id": "fake"},
                args,
                tmp_path / "attempts",
                lambda _: None,
            )
    assert dispatch.call_count == 1


def test_head_to_head_missing_data_and_cross_model_pairing(tmp_path):
    base = dict(status="completed", loss="grpo", seed=1, model_id="A")
    (tmp_path / "a.json").write_text(json.dumps(base))
    runs = ANALYSIS.load_results(tmp_path)
    assert runs[0]["loss_arm"] == "grpo"
    assert math.isnan(ANALYSIS.per_arm_summary(runs)["grpo"]["last10_avg_mean"])
    fields = dict(
        dataset="d",
        split="test",
        steps=30,
        group_size=8,
        batch_prompts=2,
        max_tokens=10,
        parser_version="v2",
        stack_fingerprint="hash",
        last10_avg=0.5,
    )
    assert (
        ANALYSIS.paired_comparison(
            [
                {**base, **fields, "loss_arm": "grpo"},
                {**base, **fields, "loss_arm": "dapo", "model_id": "B", "last10_avg": 0.6},
            ]
        )
        == []
    )
    with pytest.raises(ValueError, match="duplicate"):
        ANALYSIS.paired_comparison([{**base, **fields}, {**base, **fields}])
    assert ANALYSIS.sign_test([1, 1, 1])["p_value"] == 0.25


def test_healthcheck_reads_live_probe_paths(monkeypatch):
    def fake(args):
        assert Path(args[1]).is_file()
        return {"noul": 1 if "apple?" in args[-1] else 0}

    monkeypatch.setattr(JEV, "_call_jev", fake)
    assert JEV.healthcheck()["healthy"] is True


@pytest.mark.parametrize("status", ["INCOMPLETE", "UNSCORED", "UNGRADED", "NOT_TERMINAL"])
def test_negative_status_names_never_mean_complete(status):
    assert USE._keyword_status(json.dumps({"status": status})) == "in_progress"


def test_service_outage_persists_routing_policy_and_exact_state(tmp_path, monkeypatch):
    monkeypatch.setattr(USE, "healthcheck", Mock(side_effect=RuntimeError("offline")))
    state = "x" * 130000
    route = USE.judged(state, "Is this complete?", label="probe", receipt_dir=tmp_path)
    assert route.decision == "FALLBACK_OUTAGE"
    receipt = json.loads(Path(route.receipt_path).read_text())
    assert receipt["input"]["policy_version"] == "jev-routing-v2"
    assert __import__("base64").b64decode(receipt["state_bytes_base64"]).decode() == state
    assert receipt["resolved_model"] is None


def supervisor(tmp_path):
    clock = SimpleNamespace(wall=100.0, mono=50.0)
    sup = HS.HostedSupervisor(
        tmp_path, b"r" * 32, b"s" * 32, clock=lambda: (clock.wall, clock.mono)
    )
    sup.arm(dict(provider="modal", operation="FunctionCallCancel", object_id="fake", nonce="bound"))
    return sup, clock


def test_supervisor_monotonic_expiry_with_frozen_wall(tmp_path):
    sup, clock = supervisor(tmp_path)
    clock.mono += 4000
    assert sup.run_once()["status"] != "LIVE"


def test_requester_cannot_bypass_supervisor_renewal_admission(tmp_path):
    sup, clock = supervisor(tmp_path)
    forged = sup.build_renewal(1, "unbound", clock.wall + 4000)
    HS.write_once(sup.renewals_dir / "ren-0000000001.json", forged)
    assert sup.latest_verified_renewal(sup.load_lease()) is None


def test_manual_cleanup_assertion_is_not_provider_verification(tmp_path):
    sup, clock = supervisor(tmp_path)
    assert (
        sup.record_completion_observation(1.0, True, True, {})["status"]
        == "MANUAL_ATTESTATION_UNVERIFIED"
    )
    assert sup.terminal_receipt() is None


def test_mathematical_counterexamples_and_corrected_bound():
    assert 0.5 * 0.25**2 == 0.03125
    assert 0.5 * (1 / (1 + math.exp(-2.5)) - 0.5) > 0.1 * math.log(2)
    assert 0.6**2 + 0.4**2 > 0.5
    assert math.sqrt(7) / 8 > 0.33
    from scipy.optimize import brentq

    assert brentq(lambda p: p**4 + (1 - p) ** 4 - 0.5, 0.5, 1) == pytest.approx(0.8406250193)


@pytest.mark.parametrize(
    "changed,tokens,gradient,expected",
    [
        (False, 4, 1.0, False),
        (True, 0, 1.0, False),
        (True, 4, float("nan"), False),
        (True, 4, 1.0, True),
    ],
)
def test_optimizer_evidence_requires_measured_active_update(changed, tokens, gradient, expected):
    import torch
    from trl_sampler_adapter import make_optimizer_observer

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([gradient])
    trainer = SimpleNamespace(
        model=SimpleNamespace(parameters=lambda: [parameter]),
        _next_loss_tokens=tokens,
        _next_sampler_cumulative={"update_eligible_groups": 1},
    )
    observer = make_optimizer_observer(trainer, object, torch, stop_on_applied=True)
    state, control = SimpleNamespace(global_step=0), SimpleNamespace(should_training_stop=False)
    observer.on_pre_optimizer_step(None, state, control)
    if changed:
        with torch.no_grad():
            parameter.add_(0.25)
    observer.on_optimizer_step(None, state, control)
    observation = trainer._next_update_observations[-1]
    assert observation["update_applied"] is expected
    assert control.should_training_stop is expected
    assert observation["optimizer_step"] == 1
    assert trainer._next_loss_tokens == 0


def test_loss_token_accounting_excludes_environment_masked_slots():
    import torch
    from trl_sampler_adapter import build_sampler_trainer_class

    class FakeBase:
        def compute_loss(self, model, inputs, *args, **kwargs):
            return 0

    trainer = build_sampler_trainer_class(FakeBase)()
    trainer.compute_loss(
        None,
        {
            "completion_mask": torch.ones((2, 2)),
            "advantages": torch.tensor([1.0, 1.0]),
            "env_mask": torch.tensor([[0, 0], [1, 0]]),
        },
    )
    assert trainer._next_loss_tokens == 1


@pytest.mark.parametrize(
    "provider,gpu,flavor,terminal",
    [
        ("colab", "A100", "A100", False),
        ("gcp", "L4", "g2-standard-8", False),
        ("huggingface_jobs", "L4", "l4x1", True),
        ("kaggle", "T4", "T4x2", True),
    ],
)
def test_each_provider_shared_request_and_completed_contract(
    tmp_path, provider, gpu, flavor, terminal
):
    from tests.test_next_submission_preflight_gate import synthetic_matrix, GATE

    protocol, paths = synthetic_matrix(tmp_path)
    for path in paths:
        receipt = json.loads(path.read_text())
        original = receipt["identity_manifest"]
        original.update(provider=provider, gpu=gpu, hardware_flavor=flavor)
        original.pop("identity_manifest", None)
        request = GATE.common.seal_request(
            original,
            decoder=original["decoder"],
            tracking=original["tracking"],
            source_files=receipt["manifest"]["source_files"],
        )
        for payload in (receipt["payload"], receipt["manifest"]):
            payload["run_config"].update(
                provider=provider,
                hardware_flavor=flavor,
                unit_fingerprint=request["fingerprint"],
                stack_fingerprint=request["stack_fingerprint"],
            )
            payload["runtime_versions"]["gpu"] = f"NVIDIA {gpu}"
        completed = GATE.common.completed_receipt(
            request,
            result=receipt["payload"],
            manifest=receipt["manifest"],
            verification=receipt["remote_verification"],
            cleanup={
                "session_absent_verified": not terminal,
                "compute_terminal_verified": terminal,
                "resource_id": f"{provider}/unit",
                "provider_observation": "COMPLETED",
            },
            return_code=0,
        )
        path.write_text(json.dumps(completed))
    assert GATE.evaluate_matrix(protocol, paths)["confirmatory_execution_gate"] == "pass"
    # Provider entrypoints actually route through these shared constructors.
    module = (
        "run_preflight"
        if provider == "colab"
        else "run_hf_jobs_preflight"
        if provider == "huggingface_jobs"
        else f"run_{provider}_preflight"
    )
    source = (ZVF / "next-submission" / f"{module}.py").read_text()
    assert "seal_request(" in source and "completed_receipt(" in source


def test_http_actor_contract_and_actual_usage_without_loading_model(monkeypatch):
    import torch

    monkeypatch.setenv("E13_ACTOR_API_KEY", "synthetic-test-token")
    actor = load_module("review05_http_actor", ZVF / "e13_balrog/hf_actor_server.py")

    class Inputs(dict):
        def to(self, device):
            return self

    class Tokenizer:
        eos_token_id = 2

        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, *args, **kwargs):
            return Inputs(input_ids=torch.tensor([[10, 11, 12]]))

        def decode(self, *args, **kwargs):
            return "answer"

    actor.tok = Tokenizer()
    actor.model = SimpleNamespace(generate=lambda **kwargs: torch.tensor([[10, 11, 12, 20, 2]]))
    from fastapi.testclient import TestClient

    client = TestClient(actor.app)
    base = {
        "model": actor.REPO,
        "messages": [{"role": "user", "content": "question"}],
        "max_tokens": 2,
    }
    header = {"Authorization": "Bearer synthetic-test-token"}
    response = client.post("/v1/chat/completions", json=base, headers=header)
    assert response.status_code == 200
    assert response.json()["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    assert response.json()["choices"][0]["finish_reason"] == "stop"
    actor.model.generate = lambda **kwargs: torch.tensor([[10, 11, 12, 20, 21]])
    assert (
        client.post("/v1/chat/completions", json=base, headers=header).json()["choices"][0][
            "finish_reason"
        ]
        == "length"
    )
    for mutation in (
        {"model": "other"},
        {"temperature": 0.8},
        {"top_p": 0.9},
        {"stream": True},
        {"max_tokens": 0},
        {"stop": ["x"]},
    ):
        assert (
            client.post(
                "/v1/chat/completions", json={**base, **mutation}, headers=header
            ).status_code
            == 400
        )


def test_modal_http_entrypoint_is_registered_as_asgi_without_process_wait():
    import ast

    tree = ast.parse((ZVF / "e13_balrog/modal_actor05.py").read_text())
    serve = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "serve")
    decorators = [ast.unparse(n.func) for n in serve.decorator_list]
    assert decorators == ["app.function", "modal.asgi_app"]
    assert "wait(" not in ast.unparse(serve) and "return actor_app" in ast.unparse(serve)


def test_verifier_and_shaped_reward_contrast_remain_distinct():
    import numpy as np

    data = SimpleNamespace(
        batch={
            "token_level_scores": np.array([[1.0], [1.0]]),
            "token_level_rewards": np.array([[0.1], [0.2]]),
        },
        non_tensor_batch={"uid": np.array(["p", "p"])},
    )
    decision = VERLZVFAdapter().process_step(data)
    assert decision.verifier_zvf == 1 and decision.effective_training_reward_zvf == 0
    assert decision.applied_group_size is None


def test_sweep_resume_requires_same_attempt_and_current_evidence(tmp_path, monkeypatch):
    cfg = MATCHED.load_config()
    command = MATCHED.build_plan(cfg)["commands"][0]
    cell = {
        **command["cell"],
        "tag": command["tag"],
        "block": "matched",
        "framework": "fake",
        "model_short": "fake",
        "effective_spec": command["effective_spec"],
    }
    path = tmp_path / "cell.json"

    def fake(argv, **kwargs):
        assert kwargs["shell"] is False
        path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "attempt_nonce": kwargs["env"]["TINKER_ATTEMPT_NONCE"],
                    "effective_spec": cell["effective_spec"],
                    "step_log": [
                        {"step": i, "reward": 0.5, "zvf": 0.5, "gu": 0.5}
                        for i in range(cell["steps"])
                    ],
                    "resource_usage": {
                        "attempted_groups": 30,
                        "accepted_groups": 30,
                        "generated_tokens": 64,
                        "training_tokens": 32,
                        "optimizer_steps": cell["steps"],
                        "elapsed_gpu_seconds": 1.0,
                    },
                }
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(SWEEP.subprocess, "run", fake)
    assert SWEEP.run_one("fake --argument", path, cell)["status"] == "completed"
    assert SWEEP.result_json_is_complete(path, cell["steps"], cell["effective_spec"])
    data = json.loads(path.read_text())
    data["step_log"][0]["reward"] = 0.6
    path.write_text(json.dumps(data))
    assert not SWEEP.result_json_is_complete(path, cell["steps"], cell["effective_spec"])


def test_audit_failed_aggregate_cannot_exit_success_and_same_count_invalidates_cache(tmp_path):
    registration = tmp_path / "protocol.json"
    registration.write_text("{}")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    records = {("grpo", 11): {"heldout_score": 0.5, "manifest_path": "manifest.json"}}
    digest = lambda: CAMPAIGN.campaign_evidence_digest(
        registration, records, lambda unit: tmp_path / "record.json"
    )
    original = digest()
    records[("grpo", 11)]["heldout_score"] = 0.6
    assert digest() != original
    second = digest()
    manifest.write_text('{"changed":true}')
    assert digest() != second
    third = digest()
    registration.write_text('{"amended":true}')
    assert digest() != third
    assert CAMPAIGN.completion_exit_code(40, 40, False) == 1
    assert CAMPAIGN.completion_exit_code(40, 40, True) == 0


def test_ledger_requires_exact_status_retained_components_and_nonzero_failure():
    from ledger_contract import is_graded, components_match, verification_exit_code

    assert is_graded("GRADED") and not is_graded("UNGRADED")
    assert not components_match(None, 129)
    assert not components_match({"a": {"passes": 67}, "b": {"passes": 61}}, 129)
    assert components_match({"a": {"passes": 67}, "b": {"passes": 62}}, 129)
    assert verification_exit_code([{"status": "PASS"}, {"status": "FAIL"}]) == 1


def test_diagnostic_seeds_exclude_locked_confirmatory_and_reserve():
    import run_preflight

    protocol = json.loads(run_preflight.PROTOCOL.read_text())
    for seed in (211, 349):
        with pytest.raises(ValueError, match="disjoint"):
            run_preflight.validate_diagnostic_seed(seed, protocol)
    run_preflight.validate_diagnostic_seed(1009, protocol)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "preflight"),
        ("model", "unrelated/model"),
        ("max_steps", 1),
        ("heldout_n", 8),
        ("task", "unrelated"),
        ("batch_prompts", 1),
        ("max_completion_length", 32),
        ("arm", "dapo"),
        ("seed", 999),
        ("treatment_changes", ["unregistered_change"]),
        ("max_steps", None),
        ("batch_prompts", True),
    ],
)
def test_audit_execution_mutations_fail_real_aggregate_and_campaign(tmp_path, field, value):
    from test_aggregate_audit import ContractTests

    fixture = ContractTests()
    fixture.setUp()
    record = fixture.record(tmp_path)
    record["remote_verification"]["hf_checkpoint_steps"] = [5, 10, 15, 20, 25, 30]
    path = tmp_path / record["manifest_path"]
    record["manifest_path"] = str(path)
    manifest = json.loads(path.read_text())
    manifest["run_config"][field] = value
    if value is None:
        manifest["run_config"].pop(field)
    path.write_text(json.dumps(manifest))
    accepted, errors = AUDIT.validate_records(fixture.prereg, [(tmp_path / "record.json", record)])
    assert not accepted and errors
    valid, reason = CAMPAIGN.validate_result(
        ("grpo", 11), record, CAMPAIGN.load_contract()[1], stack_fingerprint="b" * 64
    )
    assert valid is False and reason


@pytest.mark.parametrize("location", ["audit_record", "run_config", "both"])
def test_audit_treatment_changes_must_match_in_every_manifest_location(tmp_path, location):
    from test_aggregate_audit import ContractTests

    fixture = ContractTests()
    fixture.setUp()
    record = fixture.record(tmp_path)
    record["remote_verification"]["hf_checkpoint_steps"] = [5, 10, 15, 20, 25, 30]
    path = tmp_path / record["manifest_path"]
    record["manifest_path"] = str(path)
    manifest = json.loads(path.read_text())
    for layer in ("audit_record", "run_config") if location == "both" else (location,):
        manifest[layer]["treatment_changes"] = ["unregistered_change"]
    path.write_text(json.dumps(manifest))
    assert not AUDIT.validate_records(fixture.prereg, [(tmp_path / "record.json", record)])[0]
    assert (
        CAMPAIGN.validate_result(
            ("grpo", 11), record, CAMPAIGN.load_contract()[1], stack_fingerprint="b" * 64
        )[0]
        is False
    )


@pytest.mark.parametrize("value", [False, None, 0, 1, "true", [], {}, "MISSING"])
def test_matrix_applied_updates_require_exact_finite_parameter_true(tmp_path, value):
    from tests.test_next_submission_preflight_gate import synthetic_matrix, GATE

    protocol, paths = synthetic_matrix(tmp_path)
    for path in paths:
        receipt = json.loads(path.read_text())
        for layer in ("payload", "manifest"):
            observation = receipt[layer]["audit_record"]["update_observations"][0]
            if value == "MISSING":
                observation.pop("finite_parameters")
            else:
                observation["finite_parameters"] = value
        path.write_text(json.dumps(receipt))
    with pytest.raises(GATE.PreflightGateError, match="finite parameters"):
        GATE.evaluate_matrix(protocol, paths)


def test_summary_only_collapse_does_not_apply_abort_but_raw_rewards_do():
    actions = []
    cb = ZVFCallback(on_collapse="abort", abort_fn=lambda decision: actions.append(decision.step))
    adapter = cb.as_trl_callback(num_generations=2)
    control = SimpleNamespace(should_training_stop=False)
    adapter.on_log(
        None,
        SimpleNamespace(global_step=1),
        control,
        logs={"frac_reward_zero_std": 1.0, "reward": 0.0},
    )
    assert cb.decisions[-1].observation_kind == "summary_only"
    assert cb.decisions[-1].signal == "noop"
    assert control.should_training_stop is False and cb.should_stop is False and actions == []
    adapter.on_log(None, SimpleNamespace(global_step=2), control, logs={"rewards": [0.0, 0.0]})
    assert cb.decisions[-1].observation_kind == "raw_rewards"
    assert control.should_training_stop is True and cb.should_stop is True and actions == [2]
