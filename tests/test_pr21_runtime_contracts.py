"""Synthetic regressions; no providers, paid jobs, or empirical artifacts."""

# ruff: noqa: S102, S307 -- execute only inspected repository AST with synthetic globals
import ast
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def function(path, name, **namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_inventory_historical_and_duplicate_manifest(tmp_path):
    from platform_hybrid.registry.provenance.verify_inventory import verify
    import shutil

    source = ROOT / "platform_hybrid/registry/provenance"
    for pattern in ("*.provenance.json", "*.provenance.json.sig", "required_records.json"):
        for path in source.glob(pattern):
            shutil.copy(path, tmp_path)
    assert verify(tmp_path) == {"inventory_valid": True, "publication_accepted": False}
    manifest = tmp_path / "required_records.json"
    manifest.write_text('{"schema_version":1,"schema_version":1,"records":[]}')
    with pytest.raises(ValueError, match=".+"):
        verify(tmp_path)


def test_inventory_passes_trusted_key(tmp_path, monkeypatch):
    from platform_hybrid.registry.provenance import verify_inventory as inv

    (tmp_path / "a.provenance.json").write_text("{}")
    (tmp_path / "a.provenance.json.sig").write_text("{}")
    (tmp_path / "required_records.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "records": [{"path": "a.provenance.json", "signature_required": True}],
            }
        )
    )
    with pytest.raises(ValueError, match="trusted"):
        inv.verify(tmp_path)
    calls = []
    monkeypatch.setattr(inv.subprocess, "run", lambda args, **kw: calls.append(args))
    assert inv.verify(tmp_path, "/trusted/key")["publication_accepted"]
    assert len(calls) == 2
    assert all(c[-2:] == ["--trusted-key", "/trusted/key"] for c in calls)


@pytest.mark.parametrize("bad", [-2, "invalid", 1.5])
def test_quota_corrupt_counter(tmp_path, bad):
    mod = load("platform_hf_spaces/defense_live_demo/serving_policy.py")
    db_path = tmp_path / "quota.sqlite"
    with sqlite3.connect(db_path) as db:
        db.execute("CREATE TABLE budget (id INTEGER PRIMARY KEY, used INTEGER)")
        db.execute("INSERT INTO budget VALUES (1, ?)", (bad,))
    assert (
        mod.reserve_call(
            "hello",
            environ={
                "LIVE_INFERENCE_POLICY": "bounded_public",
                "LIVE_TOTAL_CALL_LIMIT": "10",
                "LIVE_BUDGET_DB": str(db_path),
            },
        )
        == "quota_unavailable"
    )


def test_help_cannot_bypass_contract():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "platform_gcp/infra/gcp/webarena_contract.py"),
            "missing.json",
            "digest",
            "--help",
            "--field",
            "tasks",
        ],
        capture_output=True,
    )
    assert result.returncode != 0


def test_pid_target_does_not_integrate_integral_again():
    mod = load("platform_hybrid/experiments/implementations/p7_zvf_pid.py")
    controller = mod.ZVF_PIDController(kd=0)
    controller.step(0.9)
    states = [controller.step(0.5) for _ in range(100)]
    assert len(set(states)) == 1
    assert states[0] < controller.max_g


def test_modal_walk_depth():
    mod = load("platform_hybrid/experiments/tinker-runs/modal_inventory.py")
    calls = []

    def ls(volume, path):
        calls.append(path)
        return [
            {"path": path.rstrip("/") + "/d", "type": "directory"},
            {"path": path.rstrip("/") + "/result.json", "type": "file"},
        ]

    mod.vol_ls = ls
    assert len(mod.walk("fake")) == mod.MAX_DEPTH + 1
    assert len(calls) == mod.MAX_DEPTH + 1


def test_legacy_backfill_not_duplicated(tmp_path):
    path = "platform_hybrid/experiments/tinker-runs/wandb_backfill.py"
    existing = function(path, "existing_names", ENTITY="test")
    api = SimpleNamespace(
        runs=lambda *a, **k: [
            SimpleNamespace(name="old", config={}),
            SimpleNamespace(name="new", config={"source_identity": "digest"}),
        ]
    )
    seen = existing(api, "project")
    assert seen == {"legacy-name:old", "digest"}
    artifact = tmp_path / "x.json"
    artifact.write_text("{}")
    log = function(path, "log_file", Path=Path, content_hash=lambda d: "fresh")
    assert log("project", "old", artifact, [], "", seen) is False


def test_numeric_token_metrics_preserved_secrets_redacted():
    mod = load("autoresearch/reason-260727-2155/audit_sanitizer.py")
    output = mod.sanitize(
        {
            "summary": {"train/num_tokens": 12, "eval/response_tokens": 4},
            "api_token": "secret-value",
            "token": "secret-value",
        }
    )
    assert output["summary"] == {"train/num_tokens": 12, "eval/response_tokens": 4}
    assert output["api_token"] == output["token"] == "<REDACTED>"


def test_offline_nonstring_url_fails(tmp_path, monkeypatch):
    mod = load("platform_hf_spaces/defense_live_demo/offline_verify.py")
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        json.dumps(
            {"snapshot_date": "2026-10-18", "claim_2_matched_budget": [{"run_id": "a", "url": 123}]}
        )
    )
    picture = tmp_path / "x.png"
    picture.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(mod, "SNAPSHOT", snapshot)
    monkeypatch.setattr(mod, "IMAGES", (picture,))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"sha256": {p.name: mod.sha256(p) for p in [snapshot, picture]}, "run_ids": ["a"]}
        )
    )
    assert mod.verify(manifest, mod.sha256(manifest))["status"] == "FAIL"


def test_dry_run_honors_output(tmp_path):
    target = tmp_path / "requested.tsv"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "platform_hybrid/experiments/bfclv4_tool_use.py"),
            "--dry-run",
            "--seeds",
            "1",
            "--steps",
            "2",
            "--out",
            str(target),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert target.exists()


def test_causal_launcher_blocks_before_allocation(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "platform_hybrid/experiments/run_causal_zvf.sh")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "service-resolved base-model revision unavailable" in result.stderr
    assert not list(tmp_path.iterdir())


def test_sft_run_identity_not_ppo():
    tree = ast.parse(
        (ROOT / "platform_hybrid/experiments/modal/modal_parallel_runner.py").read_text()
    )
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_ppo_gsm8k")
    exp = next(
        n
        for n in fn.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "exp" for t in n.targets)
    )
    ns = {"model_name": "Qwen/Qwen3-8B", "seed": 42}
    exec(compile(ast.Module(body=[exp], type_ignores=[]), "identity", "exec"), ns)
    assert ns["exp"] == "success_filtered_sft_best_of_4_gsm8k_Qwen3-8B_s42"


def test_p8_insufficient_runs_writes_explicit_nonresult(tmp_path):
    out = tmp_path / "detector.json"

    def rejected(*args, **kwargs):
        raise ValueError("At least two independent runs per class required")

    main = function(
        "platform_hybrid/experiments/openings/p8_detector.py",
        "main",
        load_steps=lambda: [{"label": 0, "method": "grpo", "run_id": "only"}],
        make_matrix=lambda *a: ([], []),
        FULL_FEATURES=[],
        REWARD_ONLY_FEATURES=[],
        cv_auroc=rejected,
        OUT_PATH=out,
        json=json,
    )
    main()
    data = json.loads(out.read_text())
    assert data["status"] == "not_reproducible"
    assert data["claim_eligible"] is False
    assert data["auroc_full_telemetry"] is None


def test_incomplete_heldout_retains_identity():
    tree = ast.parse((ROOT / "platform_hybrid/experiments/modal/modal_heldout_eval.py").read_text())
    result = next(
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Return)
        and isinstance(n.value, ast.Dict)
        and any(
            isinstance(v, ast.Constant) and v.value == "fixed_evaluation_incomplete"
            for v in n.value.values
        )
    )
    rec = SimpleNamespace(
        checkpoint="cp",
        model_short="m",
        seed=42,
        last10_avg=0.4,
        run_id="r",
        experiment_id="exp",
        source_file="file",
        wandb_run_url="https://example.org",
    )
    value = eval(
        compile(ast.Expression(result), "incomplete", "eval"),
        {"rec": rec, "model": "model", "n": 0, "per_problem": []},
    )
    assert value["experiment_id"] == "exp"
    assert value["model"] == "model"
    assert value["seed"] == 42
    assert value["run_id"] == "r"
    assert value["training_last10_avg"] == 0.4


def test_role_mapping_has_distinct_nonmatches():
    tree = ast.parse(
        (ROOT / "platform_hybrid/experiments/tinker-runs/tinker_wandb_registry.py").read_text()
    )
    role = next(
        v
        for n in ast.walk(tree)
        if isinstance(n, ast.Dict)
        for k, v in zip(n.keys, n.values, strict=False)
        if isinstance(k, ast.Constant) and k.value == "evidence_role"
    )
    roles = {
        c: eval(compile(ast.Expression(role), "role", "eval"), {"confidence": c})
        for c in ["exact", "candidate", "unresolved", "unmatched", "conflict_or_unverified"]
    }
    assert len(set(roles.values())) == 5
    assert roles["unmatched"] == "no_evidence"


def test_claim_manifest_uses_validation_failures():
    tree = ast.parse(
        (ROOT / "platform_hybrid/experiments/tinker-runs/build_claim_run_table.py").read_text()
    )
    value = next(
        v
        for n in ast.walk(tree)
        if isinstance(n, ast.Dict)
        for k, v in zip(n.keys, n.values, strict=False)
        if isinstance(k, ast.Constant) and k.value == "schema_valid"
    )
    for failures in ([], ["bad row"]):
        assert eval(
            compile(ast.Expression(value), "schema", "eval"),
            {"metadata": {"registry_validation_failures": failures}},
        ) == (not failures)


def test_campaign_empty_steps_not_counted_as_zero_loss():
    tree = ast.parse((ROOT / "platform_hybrid/experiments/openings/campaign.py").read_text())
    branch = next(
        n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "not data"
    )
    assert any(isinstance(n, ast.AugAssign) and n.target.id == "no_data_steps" for n in branch.body)
    assert not any(isinstance(n, ast.AugAssign) and n.target.id == "zero_loss" for n in branch.body)
    value = next(
        k.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        for k in n.keywords
        if k.arg == "near_zero_scalar_loss_fraction"
    )
    assert (
        eval(compile(ast.Expression(value), "metric", "eval"), {"zero_loss": 1, "loss_steps": 2})
        == 0.5
    )
    assert (
        eval(compile(ast.Expression(value), "metric", "eval"), {"zero_loss": 0, "loss_steps": 0})
        is None
    )


def test_compact_run_retains_scalar_config_aliases():
    run = SimpleNamespace(
        summary={},
        config={"model_name": "Qwen", "task_name": "gsm8k", "train_steps": 30, "model_object": {}},
        id="id",
        name="name",
        state="finished",
        created_at="now",
        group="group",
        job_type="train",
        tags=[],
    )
    compact = function(
        "autoresearch/reason-260727-2155/wandb_claim_audit.py",
        "compact_run",
        SUMMARY_TERMS=("reward",),
        CONFIG_TERMS=("model", "task", "step"),
        sanitize=lambda v: v,
    )
    assert compact(run)["config"] == {"model_name": "Qwen", "task_name": "gsm8k", "train_steps": 30}
