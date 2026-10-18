"""Offline/mocked regressions for review 10. Never invoke a real provider."""

import ast
import asyncio
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def gcp():
    return load("platform_gcp/run_unified_dispatch.py", "review10_gcp")


@pytest.fixture
def echo():
    return load("platform_colab/echoverse/echoverse_matrix_run.py", "review10_echo")


def test_gcp_refuses_secret_source_embedding(gcp):
    with pytest.raises(ValueError, match="Raw credential"):
        gcp.build_entry_script(
            framework="tinker", model="model", forwarded={"HF_TOKEN": "dummy-secret-never-real"}
        )
    script = gcp.build_entry_script(
        framework="tinker",
        model="model",
        secret_refs={"HF_TOKEN": "projects/project/secrets/hf/versions/2"},
    )
    assert "dummy-secret-never-real" not in script
    assert "os.environ[key] = value" in script
    with pytest.raises(
        ValueError,
        match="Expected an allowed environment name and versioned Secret Manager reference",
    ):
        gcp.build_entry_script(
            framework="tinker", model="model", secret_refs={"HF_TOKEN": "raw-dummy"}
        )


def test_failed_gcp_create_never_deletes_collision(gcp, monkeypatch):
    args = gcp.parse_args(
        ["--framework", "tinker", "--instance", "existing", "--repo-revision", "a" * 40]
    )
    calls, scripts = [], []
    monkeypatch.setattr(gcp, "gcloud_binary", lambda: "fake-gcloud")

    def fail(cmd, **kwargs):
        calls.append(cmd)
        scripts.append(
            Path(next(v.split("=", 2)[2] for v in cmd if v.startswith("--metadata-from-file=")))
        )
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(gcp.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        gcp.run_unit(args)
    assert len(calls) == 1
    assert "create" in calls[0]
    assert all(not p.exists() for p in scripts)


def test_gcp_receipt_saved_before_owned_delete(gcp, monkeypatch, tmp_path):
    args = gcp.parse_args(
        ["--framework", "tinker", "--repo-revision", "a" * 40, "--receipt-dir", str(tmp_path)]
    )
    monkeypatch.setattr(gcp, "gcloud_binary", lambda: "fake")
    state = {}

    def fake(cmd, **kwargs):
        if "create" in cmd:
            state["run_id"] = next(v.split("=", 2)[2] for v in cmd if v.startswith("--labels="))
            return SimpleNamespace(stdout='[{"id":"101"}]', returncode=0)
        if "describe" in cmd:
            return SimpleNamespace(
                stdout=json.dumps({"id": "101", "labels": {"dispatch-run": state["run_id"]}}),
                returncode=0,
            )
        assert "delete" in cmd
        assert len(list(tmp_path.glob("*.json"))) == 1
        state["deleted"] = True
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(gcp.subprocess, "run", fake)
    monkeypatch.setattr(
        gcp,
        "_wait_for_receipt",
        lambda *a: {
            "protocol": gcp.PROTOCOL,
            "run_id": a[4],
            "spec_hash": a[5],
            "status": "failed",
            "exit_code": 9,
        },
    )
    result = gcp.run_unit(args)
    assert state["deleted"]
    assert result["exit_code"] == 9
    assert result["status"] == "failed"


def test_gcp_does_not_accept_serial_substrings(gcp):
    with pytest.raises(RuntimeError, match="Serial markers"):
        gcp._wait_for_marker("fake", "owner/GCP_UNIFIED_DISPATCH_DONE", "project", "zone")


def test_gcp_expiry_deletes_disk_and_rejects_duration_injection(gcp):
    cmd = gcp.create_command(
        gcloud="fake",
        instance="x",
        project="p",
        zone="z",
        startup_script=Path("x"),
        max_run_duration="90m",
    )
    assert "--instance-termination-action=DELETE" in cmd
    with pytest.raises(
        ValueError, match="max\\-run\\-duration must be a positive duration such as 90m"
    ):
        gcp.build_startup_script("pass", max_run_duration="1m; touch injected")


def test_gcp_main_propagates_failure(gcp, monkeypatch, capsys):
    monkeypatch.setattr(gcp, "run_unit", lambda _: {"exit_code": 7})
    assert gcp.main(["--framework", "tinker"]) == 7


@pytest.mark.parametrize("failure_at", [0, 1, 2])
def test_echoverse_preserves_checkout(echo, monkeypatch, tmp_path, failure_at):
    (tmp_path / "uncommitted.txt").write_text("keep")
    calls = []

    def run(cmd, **kwargs):
        if "check-ref-format" in cmd:
            return 0, "branch"
        calls.append(cmd)
        return (1 if len(calls) - 1 == failure_at else 0), "mock failure"

    monkeypatch.setattr(echo, "_run_cmd", run)
    with pytest.raises(RuntimeError, match="preserved"):
        echo.ensure_repo(tmp_path, "unused", "branch")
    assert (tmp_path / "uncommitted.txt").read_text() == "keep"


def test_echoverse_public_config_uses_allowlist(echo):
    args = SimpleNamespace(
        agent_model="model",
        agent_api_key="dummy-secret-never-real",
        future_secret="dummy-secret-never-real",
        agent_base_url="https://dummy-secret-never-real",
    )
    assert echo.public_config(args) == {"agent_model": "model"}


def test_echoverse_wandb_never_receives_secret(echo, monkeypatch):
    captured = {}
    fake = SimpleNamespace(
        init=lambda **kw: captured.update(kw),
        log=lambda *a, **k: None,
        summary={},
        run=None,
        finish=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    args = SimpleNamespace(
        use_wandb=True,
        run_name_prefix="test",
        wandb_project="p",
        wandb_entity=None,
        agent_api_key="dummy-secret-never-real",
    )
    echo.summarize_and_log_wandb([], args)
    assert "dummy-secret-never-real" not in json.dumps(captured)


def test_echoverse_missing_harness_contract_fails_closed(echo, monkeypatch, tmp_path):
    monkeypatch.setattr(echo, "_run_cmd", lambda *a, **k: (0, "--num-tasks"))
    args = SimpleNamespace(
        agent="x",
        agent_model="m",
        agent_api_key="dummy",
        max_rounds=1,
        agent_base_url=None,
        agent_base_urls=None,
        concurrency=1,
        headful=False,
    )
    with pytest.raises(RuntimeError, match="task-ids-file"):
        echo.batch_run(tmp_path, "env", ["task-b"], 1, tmp_path, args, "run", "user")


def test_echoverse_failure_exit(echo, monkeypatch):
    monkeypatch.setattr(
        echo, "_parse_args", lambda _: SimpleNamespace(log_level="INFO", dry_run=False)
    )
    monkeypatch.setattr(echo, "run_matrix", lambda _: [SimpleNamespace(batch_rc=3, errors=0)])
    monkeypatch.setattr(echo, "summarize_and_log_wandb", lambda *a: None)
    assert echo.main([]) == 1


def test_correctness_beats_keyword_reward():
    from platform_local.trl_integrations.scientific import correctness_rewards

    assert correctness_rewards(["2", r"first then step step step \boxed{999}"], ["2", "2"]) == [
        1.0,
        0.0,
    ]
    with pytest.raises(ValueError, match="One gold answer is required for every completion"):
        correctness_rewards(["2"], None)


def test_collapse_is_per_verified_prompt_group():
    from platform_local.trl_integrations.scientific import group_diagnostics

    result = group_diagnostics([0.0] * 8 + [1.0] * 8, ["a"] * 8 + ["b"] * 8, 8)
    assert result == {"reward_variance": 0.0, "advantage_collapse_rate": 1.0}
    with pytest.raises(
        ValueError, match="Prompt groups must each contain exactly num_generations rewards"
    ):
        group_diagnostics([0, 1], ["a", "b"], 8)


def test_generated_model_name_cannot_inject_source(tmp_path):
    from platform_local.trl_integrations.config import TRLConfig
    from platform_local.trl_integrations.trainer import generate_trl_train_script

    model = 'test-model"""\nINJECTED_SENTINEL = 123\n"""'
    script = generate_trl_train_script(
        TRLConfig(
            execution_contract="scientific",
            model={"model_name": model},
            data={"train_data": ["dummy"]},
        ),
        tmp_path / "train.py",
    )
    tree = ast.parse(script)
    assert not any(isinstance(n, ast.Name) and n.id == "INJECTED_SENTINEL" for n in ast.walk(tree))
    assert "get_last_checkpoint" not in script
    assert "resume_from_checkpoint=resume" in script
    assert "Checkpoint manifest differs" in script
    assert "uuid.uuid4().hex" in script
    assert "trainer.state.global_step" in script
    assert script.index("set_seed(") < script.index("model = AutoModel")


def test_simulator_requires_explicit_opt_in():
    from platform_local.trl_integrations.config import TRLConfig
    from platform_local.trl_integrations.trainer import TRLTrainer

    with pytest.raises(RuntimeError, match="simulator"):
        asyncio.run(TRLTrainer(TRLConfig()).train_step(0))


def test_resolved_spec_matches_launcher_request():
    from platform_local.unified.launcher import UnifiedLauncher

    launcher = UnifiedLauncher()
    launcher.model, launcher.algorithm, launcher.epochs = "Requested/TestModel", "ppo", 7
    spec = launcher._resolve_spec()
    assert (spec.model, spec.algorithm, spec.training_steps) == ("Requested/TestModel", "ppo", 7)
    assert spec.protocol == "unified-resolved-v2"


def test_argv_boundaries_and_unsupported_adapters(monkeypatch):
    from platform_local.unified.backends.vast import VastBackend
    from platform_local.unified.canonical import load_spec

    spec = replace(load_spec(), model="/models/my model --not-an-option")
    plan = VastBackend().plan("trl", spec)
    assert plan.argv[plan.argv.index("--model") + 1] == spec.model
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("must reject before provider process")
    )
    with pytest.raises(NotImplementedError):
        VastBackend()._execute(plan, spec)


def test_gcp_frozen_rejects_overrides():
    from platform_local.unified.backends.gcp import GCPBackend
    from platform_local.unified.canonical import load_spec

    with pytest.raises(ValueError, match="algorithm"):
        GCPBackend().plan("trl", replace(load_spec(), algorithm="ppo"))


def test_skyrl_server_exit_cannot_claim_training(monkeypatch):
    from platform_local.unified.launcher import UnifiedLauncher

    monkeypatch.setenv("SKYRL_CHECKOUT", "/fake")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("must not launch idle server")
    )
    with pytest.raises(NotImplementedError, match="alone does not train"):
        UnifiedLauncher()._run_skyrl()


def test_canonical_missing_or_malformed_never_falls_back(monkeypatch, tmp_path):
    from platform_local.unified import canonical

    path = tmp_path / "missing.json"
    monkeypatch.setattr(canonical, "PREREGISTRATION", path)
    with pytest.raises(FileNotFoundError):
        canonical.load_spec()
    path.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported preregistration schema"):
        canonical.load_spec()


@pytest.mark.parametrize(
    ("jobs", "n"),
    [
        ([{"name": "dup", "cmd": "true"}] * 2, 2),
        ([{"name": "x", "cmd": "true"}], 0),
        ([{"name": "../escape", "cmd": "true"}], 1),
    ],
)
def test_queue_invalid_spec_never_spawns(jobs, n, monkeypatch, tmp_path):
    queue = load("platform_colab/publication_fixes/vm_queue.py", "review10_queue")
    monkeypatch.setattr(
        queue.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned invalid spec")
    )
    with pytest.raises(
        ValueError,
        match="Concurrency must be between 1 and 16|Job names must be unique safe basenames",
    ):
        queue.run_queue(jobs, n, tmp_path)


def test_queue_propagates_failure_and_tracks_all_jobs(tmp_path):
    queue = load("platform_colab/publication_fixes/vm_queue.py", "review10_queue")
    # Only local harmless shell builtins, no training/provider calls.
    assert (
        queue.run_queue(
            [{"name": "one", "cmd": "exit 3"}, {"name": "two", "cmd": "exit 0"}],
            2,
            tmp_path,
            poll_s=0.001,
        )
        == 1
    )
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["one"]["rc"] == 3
    assert status["two"]["state"] == "ok"
    assert "_all_done" in status


def test_weak_merge_evidence_is_rejected():
    gate = load("platform_colab/samestack_v2/legacy_verify_and_merge.py", "review10_gate")
    ref = {"n_eval": 4, "n_steps": 2, "model": "m"}
    result = {
        **ref,
        "arm": "ppo",
        "seed": 999,
        "group": 999,
        "smoke": False,
        "step_log": [],
        "pre_correct": [0] * 4,
        "post_correct": [0] * 4,
        "heldout_pre_acc": 1.0,
        "heldout_post_acc": 1.0,
        "execution_environment": {},
    }
    checks = gate.validate_result("grpo_g8_s123", result, {"pre": [""] * 4, "post": ["x"] * 4}, ref)
    assert not checks["filename_identity"]
    assert not checks["group_matches"]
    assert not checks["accuracy_recomputed"]
    assert not checks["substantive_environment"]
    assert not checks["completion_cardinality"]
    assert not checks["actual_steps"]


def test_valid_merge_requires_items_and_regrading():
    gate = load("platform_colab/samestack_v2/legacy_verify_and_merge.py", "review10_gate")
    ref = {"n_eval": 2, "n_steps": 1, "model": "m"}
    result = {
        **ref,
        "arm": "grpo_g8",
        "seed": 123,
        "group": 8,
        "smoke": False,
        "critic": None,
        "pre_correct": [1, 0],
        "post_correct": [1, 1],
        "heldout_pre_acc": 0.5,
        "heldout_post_acc": 1.0,
        "step_log": [{"step": 0}],
        "item_ids": ["a", "b"],
        "env": {"gpu": "fake", "python": "3", "torch": "2"},
    }
    texts = {"pre": [r"\boxed{2}", r"\boxed{9}"], "post": [r"\boxed{2}", r"\boxed{3}"]}
    items = [{"id": "a", "answer": "2"}, {"id": "b", "answer": "3"}]
    assert all(gate.validate_result("grpo_g8_s123", result, texts, ref, items).values())
    assert not all(gate.validate_result("grpo_g8_s123", result, texts, ref).values())


def test_staged_json_alone_is_not_complete(tmp_path):
    stage = load("platform_colab/samestack_v2/legacy_drive_sessions.py", "review10_stage")
    (tmp_path / "grpo_g8_s1.json").write_text("{}")
    assert not stage.bundle_complete(tmp_path, "grpo_g8_s1")


def test_failed_text_download_never_publishes_marker(monkeypatch, tmp_path):
    stage = load("platform_colab/samestack_v2/legacy_drive_sessions.py", "review10_stage")

    def download(*args, **kwargs):
        if args[-1].endswith(".json"):
            Path(args[-1]).write_text(
                json.dumps({"arm": "grpo_g8", "seed": 1, "smoke": False, "n_eval": 1})
            )
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(stage, "colab", download)
    with pytest.raises(stage.TransferIncomplete, match="Completion transfer failed"):
        stage.transfer_bundle("fake", "grpo_g8_s1", tmp_path)
    assert not list(tmp_path.glob("*.json"))


def test_existing_colab_session_not_stopped(monkeypatch, tmp_path):
    stage = load("platform_colab/samestack_v2/legacy_drive_sessions.py", "review10_stage")
    monkeypatch.setattr(
        stage.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="mine active\n", returncode=0),
    )
    calls = []
    monkeypatch.setattr(
        stage, "colab", lambda *a, **k: calls.append(a) or SimpleNamespace(returncode=0)
    )
    stage.main(["mine"])
    assert not any(a[0] in ("new", "stop") for a in calls)


def test_hf_integrity_requires_external_anchor_and_detects_tampering(tmp_path, monkeypatch):
    verify = load("platform_hf_spaces/defense_live_demo/offline_verify.py", "review10_verify")
    snapshot = tmp_path / "evidence_snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "snapshot_date": "2026-10-17",
                "claim_2_matched_budget": [{"run_id": "r1", "url": "https://example.test/run"}],
            }
        )
    )
    picture = tmp_path / "image.png"
    picture.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    monkeypatch.setattr(verify, "SNAPSHOT", snapshot)
    monkeypatch.setattr(verify, "IMAGES", (picture,))
    monkeypatch.delenv("EVIDENCE_MANIFEST_PATH", raising=False)
    monkeypatch.delenv("EVIDENCE_MANIFEST_SHA256", raising=False)
    assert verify.verify()["status"] == "FAIL"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"sha256": {p.name: verify.sha256(p) for p in (snapshot, picture)}, "run_ids": ["r1"]}
        )
    )
    anchor = verify.sha256(manifest)
    assert verify.verify(manifest, anchor)["status"] == "PASS"
    picture.write_bytes(b"arbitrary bytes")
    assert verify.verify(manifest, anchor)["status"] == "FAIL"
    assert verify.main([]) == 1


def test_hf_live_requires_index_and_exact_framework(monkeypatch):
    fetch = load("platform_hf_spaces/defense_live_demo/fetch_results.py", "review10_fetch")
    calls = []
    monkeypatch.setattr(
        fetch, "fetch_hf", lambda **k: calls.append(k) or {"source": "hf", "data": {}}
    )
    entry = {
        "framework": "trl",
        "task": "gsm8k",
        "model_revision": "a",
        "seed": 1,
        "protocol_hash": "p",
        "hf": {"repo_id": "dummy", "revision": "r"},
    }
    assert fetch.fetch_all("openrlhf", index=[entry])["status"] == "empty"
    assert not calls
    assert fetch.fetch_all("trl", index=[entry])["status"] == "unavailable"
    assert len(calls) == 1


def test_hosted_budget_is_static_by_default_and_persistent(tmp_path):
    policy = load("platform_hf_spaces/defense_live_demo/serving_policy.py", "review10_policy")
    assert policy.reserve_call("prompt", environ={}) == "live_inference_disabled"
    env = {
        "LIVE_INFERENCE_POLICY": "bounded_public",
        "LIVE_TOTAL_CALL_LIMIT": "1",
        "LIVE_BUDGET_DB": str(tmp_path / "quota.db"),
    }
    assert policy.reserve_call("x" * 4097, environ=env) == "invalid_input_size"
    assert policy.reserve_call("prompt", environ=env) is None
    assert policy.reserve_call("prompt", environ=env) == "shared_call_limit_reached"


def test_export_excludes_credentials_and_unknown_files(tmp_path):
    export = load("platform_local/blind_review/anonymize_code.py", "review10_export")
    for path in (
        ".env",
        ".env.local",
        "nested/.env",
        "data.unknown",
        "private.pem",
        "document.pdf",
    ):
        assert export._excluded(path)
    path = tmp_path / ".env"
    path.write_text("TOKEN=dummy-token-never-real")
    assert ".env" in export._post_scan(tmp_path)
    with pytest.raises(ValueError, match="commit"):
        export._git_archive_into(tmp_path / "out")


def test_export_rejects_tracked_symlink(monkeypatch, tmp_path):
    export = load("platform_local/blind_review/anonymize_code.py", "review10_export")
    monkeypatch.setattr(
        export.subprocess, "check_output", lambda *a, **k: b"120000 blob abc\tlink\x00"
    )
    with pytest.raises(ValueError, match="Symlinks"):
        export._git_archive_into(tmp_path, "a" * 40)


def test_source_root_calculations():
    export = load("platform_local/blind_review/anonymize_code.py", "review10_export")
    assert export.ROOT == ROOT
    builder = (ROOT / "platform_gcp/infra/gcp/build_webarena_image.sh").read_text()
    assert '"$(dirname "$0")"/../../..' in builder
    assert builder.index("test -d") < builder.index("gcloud compute instances create")


@pytest.mark.parametrize("script", ["run_one.sh", "run_coding.sh", "vast_run.sh"])
def test_vast_secret_argv_and_failure_guards(script):
    source = (ROOT / "platform_vast" / script).read_text()
    assert "key='${" not in source
    assert "token='${" not in source
    assert "--wandb_key" not in source
    assert 'exit "$overall_status"' in source
    assert "overall_status=$job_status" in source
    subprocess.run(["bash", "-n"], input=source, text=True, check=True)


def test_shell_dependency_constraints_keep_argument_boundaries(tmp_path):
    source = (ROOT / "platform_vast/run_coding.sh").read_text()
    line = next(line for line in source.splitlines() if '"trl>=' in line)
    script = "pip() { printf '%s\\n' \"$@\"; }; " + line.replace("2>&1 | tail -1", "")
    output = subprocess.check_output(["bash", "-c", script], cwd=tmp_path, text=True)
    assert "peft>=0.12.0" in output
    assert "accelerate>=0.33.0" in output
    assert not list(tmp_path.iterdir())


def test_mig_secret_script_has_no_xtrace_and_required_fetch_is_checked():
    source = (ROOT / "platform_gcp/infra/gcp/deploy_mig.sh").read_text()
    assert "set -eux" not in source
    assert "export TINKER_API_KEY=\\$(" not in source
    assert "Tinker credential is empty" in source
    assert "cd /opt/tinker-rl-lab/platform_hybrid" not in source


def test_ungrounded_paid_analysis_is_blocked():
    module = load("platform_local/analyze_papers_google_ai.py", "review10_analysis")
    with pytest.raises(ValueError, match="file partitions"):
        asyncio.run(module.analyze_part(SimpleNamespace(), 1))
    with pytest.raises(SystemExit, match="disabled"):
        asyncio.run(module.main())


def test_smoke_result_cannot_satisfy_full_request(tmp_path):
    module = load("platform_colab/publication_fixes/samestack_v2_colab.py", "review10_colab_run")
    legacy = tmp_path / "grpo_g8_s1.json"
    legacy.write_text(json.dumps({"smoke": True, "n_steps": 2, "n_eval": 32}))
    full = tmp_path / "samestack-colab-v2-validated-reuse" / "full"
    full.mkdir(parents=True)
    (full / legacy.name).write_text(legacy.read_text())
    with pytest.raises(ValueError, match="manifest does not match"):
        module.run_arm("grpo_g8", 1, str(tmp_path), smoke=False)


def test_matching_cached_result_requires_text_companion(tmp_path):
    module = load("platform_colab/publication_fixes/samestack_v2_colab.py", "review10_colab_run")
    full = tmp_path / "samestack-colab-v2-validated-reuse" / "full"
    full.mkdir(parents=True)
    result = {
        "protocol": "samestack-colab-v2-validated-reuse",
        "arm": "grpo_g8",
        "seed": 1,
        "smoke": False,
        "model": module.MODEL,
        "n_steps": module.N_STEPS,
        "n_eval": 1319,
        "group": 8,
        "n_gen": module.N_GEN,
        "lr": module.LR,
        "max_new": module.MAX_NEW,
        "k_epochs": module.K_EPOCHS,
    }
    (full / "grpo_g8_s1.json").write_text(json.dumps(result))
    with pytest.raises(FileNotFoundError):
        module.run_arm("grpo_g8", 1, str(tmp_path), smoke=False)


def test_hardware_labels_come_from_run_metadata():
    # Extract the pure helper without executing the analysis CLI or changing results.
    tree = ast.parse((ROOT / "platform_colab/publication_fixes/v2_final_analysis.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "hardware_label")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "hardware_helper", "exec"), ns)  # noqa: S102 - Execute only checked-in helper definitions to isolate optional imports.
    assert ns["hardware_label"]({"env": {"platform": "colab", "gpu": "L4"}}) == "L4 (colab)"
    with pytest.raises(ValueError, match="provenance"):
        ns["hardware_label"]({})


def test_matched_eval_releases_models_and_skips_final_reload():
    source = (ROOT / "platform_colab/publication_fixes/matched_eval_colab.py").read_text()
    assert source.index('os.replace(f"{out}/eval_heldout.json.tmp"') < source.index(
        "del model, base"
    )
    assert "if index + 1 < len(specs):" in source
    assert "model.unload()" not in source


def test_mig_controller_with_fake_cloud_commands(tmp_path):
    # Execute only local fixture CLIs, never the generated VM startup/training.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.jsonl"
    startup = tmp_path / "startup.sh"
    fake = f"""#!{sys.executable}
import json, os, pathlib, sys
args = sys.argv[1:]
with open({str(log)!r}, "a") as handle: handle.write(json.dumps([pathlib.Path(sys.argv[0]).name, *args])+"\\n")
if pathlib.Path(sys.argv[0]).name == "gsutil":
    if args[0] == "cat":
        parts = args[1].split("/")
        print(json.dumps({{"protocol":"webarena-shard-v2", "run_id":parts[-2], "shard":0, "exit_code":0, "task_manifest_sha256":os.environ["WEBARENA_TASK_MANIFEST_SHA256"], "repo_revision":os.environ["TINKER_REPO_REVISION"]}}))
    sys.exit(0)
for arg in args:
    if arg.startswith("--metadata-from-file=startup-script="):
        pathlib.Path({str(startup)!r}).write_text(pathlib.Path(arg.split("=",2)[2]).read_text())
if "describe" in args:
    print("template-id" if "instance-templates" in args else "group-id")
if "list-instances" in args: print("worker-0")
"""
    for command in ("gcloud", "gsutil"):
        path = bin_dir / command
        path.write_text(fake)
        path.chmod(0o755)
    import hashlib

    manifest = tmp_path / "tasks.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "webarena-service-contract-v1",
                "benchmark": "webarena_verified",
                "tasks": [{"id": "fixture.0", "required_sites": ["shopping"]}],
            }
        )
    )
    env = {
        "WEBARENA_TASK_MANIFEST": str(manifest),
        "WEBARENA_TASK_MANIFEST_SHA256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "PATH": str(bin_dir) + ":/tmp/tinker-review-venv/bin:/usr/bin:/bin",
        "GCP_PROJECT": "fixture",
        "BUCKET": "fixture",
        "NUM_WORKERS": "1",
        "MAX_RUN_SECONDS": "1",
        "TINKER_REPO_REVISION": "a" * 40,
    }
    subprocess.run(
        ["bash", str(ROOT / "platform_gcp/infra/gcp/deploy_mig.sh")],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    subprocess.run(["bash", "-n", str(startup)], check=True)
    source = startup.read_text()
    assert "--resume" in source
    assert "sleep infinity" not in source
    assert "set -eux" not in source
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert any("--stateful-metadata=worker-id=0" in call for call in calls)
    assert any("delete" in call and "managed" in call for call in calls)
    assert any("delete" in call and "instance-templates" in call for call in calls)


def test_export_scanner_failure_blocks_packaging(monkeypatch, tmp_path):
    export = load("platform_local/blind_review/anonymize_code.py", "review10_export")
    monkeypatch.setattr(export.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="gitleaks is required"):
        export._credential_scan(tmp_path)
    monkeypatch.setattr(export.shutil, "which", lambda _: "fixture-gitleaks")
    monkeypatch.setattr(
        export.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=2, stdout="dummy-redacted", stderr=""),
    )
    with pytest.raises(RuntimeError, match="credential scan failed"):
        export._credential_scan(tmp_path)


def test_echoverse_dry_run_never_serializes_env_credential(echo, monkeypatch, capsys):
    monkeypatch.setenv("ECHOVERSE_AGENT_API_KEY", "dummy-secret-never-real")
    assert echo.main(["--dry-run"]) == 0
    assert "dummy-secret-never-real" not in capsys.readouterr().out


def test_image_hash_mismatch_prevents_docker_load(tmp_path):
    import hashlib

    names = [
        "shopping_final_0712.tar",
        "shopping_admin_final_0719.tar",
        "postmill-populated-exposed-withimg.tar",
        "gitlab-populated-final-port8023.tar",
    ]
    work = tmp_path / "images"
    work.mkdir()
    for name in names:
        (work / name).write_bytes(b"tampered fixture")
    manifest = tmp_path / "checksums.txt"
    manifest.write_text("\n".join("a" * 64 + "  " + name for name in names) + "\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sudo = bin_dir / "sudo"
    sudo.write_text(
        '#!/bin/bash\nif [ "$1" = mkdir ]; then shift; exec /bin/mkdir "$@"; fi\necho forbidden-provider-operation >&2; exit 99\n'
    )
    sudo.chmod(0o755)
    env = {
        "PATH": str(bin_dir) + ":/usr/bin:/bin",
        "WORK": str(work),
        "WEBARENA_SHA256_MANIFEST": str(manifest),
        "WEBARENA_SHA256_ANCHOR": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "KIWIX_IMAGE": "ghcr.io/kiwix/kiwix-serve@sha256:" + "b" * 64,
    }
    proc = subprocess.run(
        ["bash", str(ROOT / "platform_gcp/infra/gcp/webarena-compose/load_images.sh")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "forbidden-provider-operation" not in proc.stderr
    assert "FAILED" in proc.stdout


def test_matched_eval_does_not_overlap_model_allocations(monkeypatch, tmp_path):
    import contextlib
    import types
    import weakref
    import numpy as np

    alive = weakref.WeakSet()
    allocations = []

    class Base:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            allocations.append(len(alive))
            model = cls()
            alive.add(model)
            return model

        def to(self, *_):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            return np.array([[1, 2]])

    class Wrapper:
        def __init__(self, base):
            self.base = base

        @classmethod
        def from_pretrained(cls, base, _):
            return cls(base)

        def eval(self):
            return self

        def named_parameters(self):
            return []

        def generate(self, **kwargs):
            return self.base.generate(**kwargs)

    class Batch(dict):
        def to(self, *_):
            return self

    class Tokenizer:
        pad_token_id = 0

        @classmethod
        def from_pretrained(cls, *_):
            return cls()

        def __call__(self, prompts, **kwargs):
            return Batch(input_ids=np.ones((len(prompts), 1), dtype=int))

        def batch_decode(self, rows, **kwargs):
            return [r"\boxed{2}"] * len(rows)

    fake_torch = types.SimpleNamespace(
        bfloat16="mock",
        no_grad=contextlib.nullcontext,
        cuda=types.SimpleNamespace(empty_cache=lambda: None),
        __version__="mock",
    )
    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=Base, AutoTokenizer=Tokenizer, __version__="mock"
    )
    fake_peft = types.SimpleNamespace(PeftModel=Wrapper, __version__="mock")
    common = types.SimpleNamespace(
        LORA={},
        MODEL="fixture",
        N_EVAL=1,
        MAX_TOK=1,
        gsm8k=lambda _: [("q", "2")],
        build_prompt=lambda q: q,
        reward_fn=lambda *a: 1,
        boxed_only_reward=lambda *a: 1,
    )
    for name, module in {
        "torch": fake_torch,
        "transformers": fake_transformers,
        "peft": fake_peft,
        "matched_common": common,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    module = load("platform_colab/publication_fixes/matched_eval_colab.py", "review10_eval")
    monkeypatch.setattr(module, "fix_tinker", lambda adapter: adapter)
    monkeypatch.setattr(
        module.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="mock gpu")
    )
    module.main(str(tmp_path), ["one=adapter1", "two=adapter2"])
    assert allocations == [0, 0]
    assert set(json.loads((tmp_path / "eval_heldout.json").read_text())) == {"one", "two", "_env"}


def test_gcp_ownership_mismatch_never_authorizes_delete(gcp, monkeypatch):
    monkeypatch.setattr(
        gcp.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=json.dumps({"id": "other", "labels": {"dispatch-run": "mine"}})
        ),
    )
    assert not gcp._owned("fake", "name", "p", "z", "mine", "101")


def test_gcp_rejects_wrong_receipt_identity(gcp, monkeypatch):
    monkeypatch.setattr(
        gcp.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "protocol": gcp.PROTOCOL,
                    "run_id": "other",
                    "spec_hash": "hash",
                    "status": "process_completed",
                    "exit_code": 0,
                }
            ),
        ),
    )
    with pytest.raises(ValueError, match="identity"):
        gcp._wait_for_receipt("fake", "name", "p", "z", "mine", "hash", timeout_s=1)


def test_colab_transfer_failure_preserves_new_session(monkeypatch, tmp_path):
    stage = load("platform_colab/samestack_v2/legacy_drive_sessions.py", "review10_stage")
    monkeypatch.setattr(stage, "STAGE", tmp_path)
    monkeypatch.setattr(
        stage.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="", returncode=0)
    )
    calls = []
    monkeypatch.setattr(
        stage, "colab", lambda *a, **k: calls.append(a) or SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(
        stage,
        "transfer_bundle",
        lambda *a: (_ for _ in ()).throw(stage.TransferIncomplete("fixture failed")),
    )
    with pytest.raises(stage.TransferIncomplete, match="fixture failed"):
        stage.main(["new-session", "grpo_g8:1"])
    assert any(a[0] == "new" for a in calls)
    assert not any(a[0] in ("exec", "stop") for a in calls)


@pytest.mark.parametrize("service", ["forum", "wikipedia", "map"])
def test_missing_webarena_services_rejected_before_allocation(tmp_path, service):
    import hashlib

    contract = load("platform_gcp/infra/gcp/webarena_contract.py", "review10_webarena")
    manifest = tmp_path / "tasks.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "webarena-service-contract-v1",
                "benchmark": "webarena_verified",
                "tasks": [{"id": "task.1", "required_sites": [service]}],
            }
        )
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="full benchmark deployment is blocked"):
        contract.validate(manifest, digest, "webarena_verified")


def test_webarena_manifest_pins_tasks_and_sites(tmp_path):
    import hashlib

    contract = load("platform_gcp/infra/gcp/webarena_contract.py", "review10_webarena")
    manifest = tmp_path / "tasks.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "webarena-service-contract-v1",
                "benchmark": "webarena_verified",
                "tasks": [
                    {"id": "task.2", "required_sites": ["gitlab"]},
                    {"id": "task.1", "required_sites": ["shopping"]},
                ],
            }
        )
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    result = contract.validate(manifest, digest, "webarena_verified")
    assert result["tasks"] == "task.2,task.1"
    assert result["protocol"] == "webarena-restricted-services-v2"
    with pytest.raises(ValueError, match="digest mismatch"):
        contract.validate(manifest, "wrong", "webarena_verified")
    source = (ROOT / "platform_gcp/infra/gcp/deploy_mig.sh").read_text()
    assert source.index("TASKS=$(python3") < source.index("gcloud config set")
    assert "--tasks all" not in source
    assert source.index("trap finish_shard EXIT") < source.index("# Required: Tinker key")


def test_verl_dryrun_wrapper_is_rejected_before_execution(monkeypatch):
    from platform_local.unified.launcher import UnifiedLauncher

    monkeypatch.setitem(sys.modules, "verl.trainer", None)
    with pytest.raises(NotImplementedError, match="dryrun/simulated"):
        UnifiedLauncher()._run_verl()


@pytest.mark.parametrize("framework", ["skyrl", "verl"])
def test_gcp_rejects_nontraining_drivers_before_allocation(gcp, monkeypatch, framework):
    monkeypatch.setattr(gcp, "gcloud_binary", lambda: pytest.fail("must not start provisioning"))
    with pytest.raises(NotImplementedError, match="before provisioning"):
        gcp.run_unit(gcp.parse_args(["--framework", framework]))


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf"), True, False])
def test_queue_nonfinite_or_boolean_timeout_never_spawns(timeout, monkeypatch, tmp_path):
    queue = load("platform_colab/publication_fixes/vm_queue.py", "review10_queue")
    monkeypatch.setattr(
        queue.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned invalid spec")
    )
    with pytest.raises(ValueError, match="finite and positive"):
        queue.run_queue([{"name": "test", "cmd": "exit 0", "timeout_s": timeout}], 1, tmp_path)


@pytest.mark.parametrize("signum", [1, 2, 15])
def test_queue_signal_stops_owned_job_group(tmp_path, signum):
    import os
    import shlex
    import signal
    import time

    pidfile = tmp_path / "child.pid"
    child = tmp_path / "harmless_child.py"
    child.write_text(
        "import os, pathlib, time\npathlib.Path("
        + repr(str(pidfile))
        + ").write_text(str(os.getpid()))\ntime.sleep(60)\n"
    )
    jobs = tmp_path / "jobs.json"
    jobs.write_text(
        json.dumps(
            [
                {
                    "name": "child",
                    "cmd": "exec " + shlex.join([sys.executable, str(child)]),
                    "timeout_s": 60,
                }
            ]
        )
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "platform_colab/publication_fixes/vm_queue.py"),
            str(jobs),
            "1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    child_pid = None
    try:
        deadline = time.monotonic() + 5
        while not pidfile.exists() and time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.01)
        assert pidfile.exists(), proc.communicate(timeout=1)
        child_pid = int(pidfile.read_text())
        proc.send_signal(signum)
        stdout, stderr = proc.communicate(timeout=5)
        assert proc.returncode == 128 + signum, (stdout, stderr)
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        status = json.loads((tmp_path / "logs/status.json").read_text())
        assert status["child"]["state"] == "cancelled"
        assert status["_interrupted_signal"] == signum
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("branch", ["-f", "--force", "@{-1}", ""])
def test_echoverse_branch_options_rejected_before_mutation(echo, monkeypatch, tmp_path, branch):
    original = tmp_path / "work.txt"
    original.write_text("UNCOMMITTED WORK")
    monkeypatch.setattr(
        echo, "_run_cmd", lambda *a, **k: pytest.fail("Git called for unsafe branch")
    )
    with pytest.raises(ValueError, match="branch"):
        echo.ensure_repo(tmp_path, "unused", branch)
    assert original.read_text() == "UNCOMMITTED WORK"


def test_merge_ignores_completion_marker_but_rejects_malformed_result(monkeypatch, tmp_path):
    import gzip
    import hashlib

    gate = load("platform_colab/samestack_v2/legacy_verify_and_merge.py", "review10_gate")
    stage, canon = tmp_path / "stage", tmp_path / "canonical"
    stage.mkdir()
    canon.mkdir()
    monkeypatch.setattr(gate, "STAGE", stage)
    monkeypatch.setattr(gate, "CANON", canon)
    monkeypatch.setattr(gate, "OUT", tmp_path / "report.json")
    ref = {"n_eval": 1, "n_steps": 1, "model": "fixture"}
    result = {
        **ref,
        "arm": "grpo_g8",
        "seed": 123,
        "group": 8,
        "smoke": False,
        "critic": None,
        "pre_correct": [1],
        "post_correct": [1],
        "heldout_pre_acc": 1.0,
        "heldout_post_acc": 1.0,
        "step_log": [{"step": 0}],
        "item_ids": ["a"],
        "env": {"gpu": "fake", "python": "3", "torch": "2"},
    }
    (canon / "grpo_g8_s42.json").write_text(json.dumps(ref))
    (stage / "grpo_g8_s123.json").write_text(json.dumps(result))
    (stage / "grpo_g8_s123.complete.json").write_text(json.dumps({"protocol": "colab-staging-v2"}))
    with gzip.open(stage / "grpo_g8_s123_texts.json.gz", "wt") as handle:
        json.dump({"pre": [r"\boxed{2}"], "post": [r"\boxed{2}"]}, handle)
    items = tmp_path / "items.json"
    items.write_text(json.dumps({"items": [{"id": "a", "answer": "2"}]}))
    args = [
        "--item-manifest",
        str(items),
        "--item-manifest-sha256",
        hashlib.sha256(items.read_bytes()).hexdigest(),
    ]
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps({"calibration": "fixture", "mode": "stratified_by_recorded_hardware"})
    )
    assert gate.main(args + ["--merge", "--pooling-policy", str(policy)]) == 0
    assert list(json.loads(gate.OUT.read_text())["runs"]) == ["grpo_g8_s123"]
    (stage / "grpo_g8_s123.bad.json").write_text(json.dumps(result))
    assert gate.main(args) == 1
