"""Synthetic offline evidence and terminal-state contract, not official parity."""

import json
import os
import sys
from pathlib import Path
import pytest
import webbench_eval as web


def task():
    return web.Task(1, "https://example.invalid", "fixture", "synthetic task")


def row(root):
    value = {
        "task_id": 1,
        "success": True,
        "verifier": "Halluminate/WebBench/native",
        "environment_revision": "a" * 40,
        "verifier_revision": "b" * 40,
    }
    for kind in ("screenshot", "dom", "trajectory"):
        path = root / (kind + ".fixture")
        path.write_bytes(("synthetic-" + kind).encode())
        value[kind + "_path"] = path.name
        value[kind + "_sha256"] = web.file_sha256(path)
    return value


@pytest.mark.parametrize("value", [True, False, "1", 1.0, None, -1])
def test_task_ids_never_coerce(value, tmp_path):
    with pytest.raises(web.WebBenchError):
        web.task_id_hash([value])
    path = tmp_path / "ids.json"
    path.write_text(json.dumps([value]))
    with pytest.raises(web.WebBenchError):
        web.load_task_id_manifest(path)


@pytest.mark.parametrize("value", ["1,1", "1,,2", "", "-1", "1.0", "true"])
def test_cli_rejects_bad_ids(value):
    with pytest.raises(web.WebBenchError):
        web._parse_task_ids([value])


@pytest.mark.parametrize("conflicting", [True, False])
def test_duplicate_verdicts_are_never_collapsed(tmp_path, conflicting):
    first = row(tmp_path)
    second = dict(first, success=not conflicting)
    with pytest.raises(web.WebBenchError, match="duplicate verdict"):
        web._validate_verdict_rows([first, second], [task()], tmp_path)


@pytest.mark.parametrize("mutation", ["missing", "string", "bool", "extra"])
def test_malformed_and_extra_verdict_ids(tmp_path, mutation):
    value = row(tmp_path)
    if mutation == "missing":
        value.pop("task_id")
    else:
        value["task_id"] = {"string": "1", "bool": True, "extra": 2}[mutation]
    with pytest.raises(web.WebBenchError):
        web._validate_verdict_rows([value], [task()], tmp_path)


def test_artifact_bytes_and_path_escape_are_rejected(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    value = row(root)
    web._validate_verdict_rows([value], [task()], root)
    (root / value["dom_path"]).write_text("tampered")
    with pytest.raises(web.WebBenchError, match="bytes"):
        web._validate_verdict_rows([value], [task()], root)
    value = row(root)
    outside = tmp_path / "outside"
    outside.write_text("external")
    (root / "escape").symlink_to(outside)
    value.update(dom_path="escape", dom_sha256=web.file_sha256(outside))
    with pytest.raises(web.WebBenchError, match="inside"):
        web._validate_verdict_rows([value], [task()], root)


def test_subprocess_timeout_cleans_own_process_group(tmp_path):
    with pytest.raises(web.WebBenchTimeout):
        web._run_command(
            [sys.executable, "-c", "import time; time.sleep(20)"],
            env=os.environ,
            cwd=tmp_path,
            timeout_seconds=0.05,
        )


def receipt():
    return {
        "status": "READY",
        "experiment": {
            "model_id": "fixture",
            "base_model_revision": "a" * 40,
            "model_binding": {"evaluated_hf_commit": "b" * 40},
        },
        "benchmark": {"manifest_hash": "c" * 64, "task_id_hash": "d" * 64},
    }


def launch(tmp_path, native):
    from unittest.mock import patch

    with (
        patch.object(web, "validate_wandb_receipt", return_value=[]),
        patch.object(web, "_require_enforced_budget"),
    ):
        return web.launch(
            receipt=receipt(),
            tasks=[task()],
            agent_command=["fake"],
            verifier_command=["fake"],
            worktree_root=tmp_path,
            output_dir=tmp_path,
            wandb_factory=lambda config: {},
            native_runner=native,
        )


def test_smoke_result_never_promotes_benchmark_claim(tmp_path):
    value = row(tmp_path)
    result = launch(tmp_path, lambda **kwargs: [value])
    assert result["status"] == "SMOKE_COMPLETED"
    assert result["subset_score"] == 1
    assert result["benchmark_score"] is None
    assert not result["evidence_boundary"]["score_is_claimable"]
    assert not result["billing_reconciliation"]["hard_spend_enforced"]


def test_timeout_produces_persisted_scoreless_terminal_receipt(tmp_path):
    def timeout(**kwargs):
        raise web.WebBenchTimeout("synthetic timeout")

    result = launch(tmp_path, timeout)
    assert result["status"] == "TIMED_OUT"
    assert result["score"] is None
    assert json.loads((tmp_path / "terminal_receipt.json").read_text())["status"] == "TIMED_OUT"


def test_timeout_terminates_descendants(tmp_path):
    import time

    marker = tmp_path / "child-survived"
    child = f'import time; from pathlib import Path; time.sleep(.5); Path({str(marker)!r}).write_text("bad")'
    parent = f'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",{child!r}]); time.sleep(20)'
    with pytest.raises(web.WebBenchTimeout):
        web._run_command(
            [sys.executable, "-c", parent], env=os.environ, cwd=tmp_path, timeout_seconds=0.15
        )
    time.sleep(0.6)
    assert not marker.exists()


def test_verilog_checkpoint_visibility_matches_across_validators():
    import pavlov_verilog_eval_adapter as adapter
    import pavlov_verilog_eval_receipt as receipt

    checkpoints = [
        {
            "repo_url": "https://huggingface.co/org/model",
            "revision": char * 40,
            "url": "https://huggingface.co/org/model/tree/" + char * 40,
            "stage": stage,
            "visibility": visibility,
            "safe_public_artifact": visibility == "public",
        }
        for stage, char, visibility in [
            ("initial", "a", "public"),
            ("periodic", "b", "private"),
            ("final", "c", "public"),
        ]
    ]
    for module, error in [
        (adapter, adapter.VerilogEvalBoundaryError),
        (receipt, receipt.VerilogEvalReceiptError),
    ]:
        valid = module._validate_hf_checkpoints(checkpoints)
        assert (
            next(item for item in valid if item["stage"] == "periodic")["safe_public_artifact"]
            is False
        )
        bad = [dict(item) for item in checkpoints]
        bad[1]["safe_public_artifact"] = True
        with pytest.raises(error, match="match visibility"):
            module._validate_hf_checkpoints(bad)


def test_unenforced_budget_blocks_before_any_webbench_side_effect(tmp_path):
    from unittest.mock import Mock

    wandb = Mock()
    native = Mock()
    tinker = Mock()
    with pytest.raises(web.WebBenchError, match="enforced reservation/settlement"):
        web.launch(
            receipt=receipt(),
            tasks=[task()],
            agent_command=["fake"],
            verifier_command=["fake"],
            worktree_root=tmp_path,
            output_dir=tmp_path,
            wandb_factory=wandb,
            native_runner=native,
            tinker_callback=tinker,
        )
    wandb.assert_not_called()
    native.assert_not_called()
    tinker.assert_not_called()


@pytest.mark.parametrize(
    "name",
    [
        "eval_pavlov_sdab",
        "modal_e1_swe_bench_pro_full",
        "modal_e1_swe_bench_pro_gpu_resume",
        "modal_e9_mle_bench_streaming",
    ],
)
def test_legacy_paid_runner_requires_real_budget_enforcement(name):
    # AST-load only the gate so optional Modal image construction is never executed.
    import ast

    source = Path(__file__).with_name(name + ".py")
    tree = ast.parse(source.read_text())
    gate = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_require_enforced_budget"
    )
    ns = {}
    exec(compile(ast.Module(body=[gate], type_ignores=[]), str(source), "exec"), ns)
    with pytest.raises(RuntimeError, match="enforced reservation/settlement"):
        ns["_require_enforced_budget"]()


def test_sdab_result_cannot_substitute_task_identity(monkeypatch):
    from flagship import test_eval_pavlov_sdab as fixture

    original = fixture._FakeRuntime.evaluate_task

    def substitute(self, *args, **kwargs):
        return {**original(self, *args, **kwargs), "task_id": "another-task"}

    monkeypatch.setattr(fixture._FakeRuntime, "evaluate_task", substitute)
    with pytest.raises(RuntimeError, match="result task identity mismatch"):
        fixture.PavlovSDABServiceLifetimeTests().test_completed_receipt_retains_tinker_service_run_id()


def test_sdab_selected_task_must_be_manifest_member(monkeypatch):
    from flagship import test_eval_pavlov_sdab as fixture

    monkeypatch.setattr(
        fixture._FakeRuntime,
        "list_tasks",
        lambda *args, **kwargs: [{"task_id": "outside-manifest"}],
    )
    with pytest.raises(RuntimeError, match="unique manifest members"):
        fixture.PavlovSDABServiceLifetimeTests().test_completed_receipt_retains_tinker_service_run_id()
