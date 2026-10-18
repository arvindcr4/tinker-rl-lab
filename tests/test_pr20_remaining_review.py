"""Offline adversarial regressions for the remaining PR20 review findings."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from samestack_fixtures import synthetic_run
from platform_hybrid.experiments import samestack_contract as contract
from platform_colab.samestack_v2 import drive_sessions as driver

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "reports/project_finish_2026-10-11/validation/samestack_v2"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize(
    ("field", "value"), [("step", True), ("step", 0.0), ("warm_step", False), ("warm_step", 0.0)]
)
def test_step_ids_are_typed(field, value):
    run, texts = synthetic_run("ppo")
    run["step_log" if field == "step" else "warm_log"][1 if value is True else 0][field] = value
    assert any("consecutive" in e for e in contract.validate_run(run, texts=texts)["errors"])


@pytest.mark.parametrize(
    ("key", "value"),
    [("num_beams", True), ("top_k", False), ("do_sample", 1), ("temperature", True)],
)
def test_generation_fields_reject_bool_number_coercion(key, value):
    run, texts = synthetic_run()
    run["manifest"]["generation"]["train"][key] = value
    assert any(
        f"generation.train.{key}" in e for e in contract.validate_run(run, texts=texts)["errors"]
    )


def test_missing_root_config_is_unknown_not_conflict():
    run, texts = synthetic_run()
    del run["manifest"]["effective_config"]["seed"]
    run["manifest"]["config_hash"] = contract.canonical_hash(run["manifest"]["effective_config"])
    report = contract.validate_run(run, texts=texts)
    assert "effective_config.seed unavailable" in report["unverified"]
    assert not any("contradicts root seed" in e for e in report["errors"])
    assert not report["verified"]


def test_missing_text_identity_is_unverified():
    run, texts = synthetic_run()
    del texts["test_item_ids"]
    report = contract.validate_run(run, texts=texts)
    assert not report["verified"]
    assert "text test_item_ids unavailable" in report["unverified"]


@pytest.mark.parametrize("number", ["١٢", "１２", "1e٢", "१२"])
def test_unicode_digits_rejected(number):
    from platform_hybrid.experiments.samestack_protocol import exact_number

    assert exact_number(number) is None


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Qwen/Qwen3-8B-base (shared harness)", "base"),
        ("Qwen/X-instruct(annotation)", "instruct"),
        ("X-baseball", None),
    ],
)
def test_checkpoint_annotations(name, expected):
    from platform_hybrid.registry.query import _variant

    assert _variant(name) == expected


def test_v1_unused_nan_is_accepted_consumed_nan_is_rejected(tmp_path):
    v1 = module(ANALYSIS / "v1_recompute.py", "pr20_v1")
    run = dict(
        arm="ppo",
        seed=42,
        n_eval=200,
        pre_correct=[0] * 200,
        post_correct=[1] * 200,
        heldout_post_acc=1.0,
        step_log=[{"zvf": float("nan")}],
    )
    path = tmp_path / "samestack_gsm8k_cot_full.json"
    path.write_text(json.dumps({"runs": [run]}))
    loaded = v1.load_runs(tmp_path)
    assert loaded[0]["step_log"][0]["zvf"] is None
    v1.validate_runs(loaded)
    run["heldout_post_acc"] = float("nan")
    path.write_text(json.dumps({"runs": [run]}))
    with pytest.raises(ValueError, match="stored post accuracy must be finite"):
        v1.validate_runs(v1.load_runs(tmp_path))


def test_validator_defaults_are_not_sticky_and_custom_output_required(tmp_path, monkeypatch):
    val = module(
        ROOT / "platform_modal/scripts/p5p8/p6_iter118_strict_validator.py", "pr20_validator"
    )
    import sys

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "query",
        SimpleNamespace(
            load=lambda **kw: calls.append(kw["registry"]) or {},
            registry_input_manifest=lambda _: {},
            registry_manifest_sha256=lambda _: "abc",
            VALIDATOR_VERSION="test",
        ),
    )
    assert val.main(["--registry", str(tmp_path), "--output", str(tmp_path / "out")]) == 1
    assert val.main(["--output", str(tmp_path / "out")]) == 1
    assert calls == [tmp_path.resolve(), (val.PROJECT_ROOT / "registry").resolve()]
    with pytest.raises(SystemExit):
        val.main(["--registry", str(tmp_path)])


def test_empty_expected_hash_is_mismatch(tmp_path, monkeypatch, capsys):
    val = module(
        ROOT / "platform_modal/scripts/p5p8/p6_iter118_strict_validator.py", "pr20_validator_hash"
    )
    import sys

    monkeypatch.setitem(
        sys.modules,
        "query",
        SimpleNamespace(
            load=lambda **_: {},
            registry_input_manifest=lambda _: {},
            registry_manifest_sha256=lambda _: "abc",
            VALIDATOR_VERSION="test",
        ),
    )
    assert val.main(["--input-sha256", "", "--output", str(tmp_path / "out")]) == 1
    assert "manifest mismatch" in capsys.readouterr().err


def source():
    return dict(
        status="observed",
        commit="a" * 40,
        dirty_hash="b" * 64,
        source_files={
            str(driver.RUNNER.relative_to(driver.ROOT)): contract.sha256_file(driver.RUNNER)
        },
    )


def test_unavailable_source_never_uploads(tmp_path):
    calls = []
    with pytest.raises(ValueError, match="observed source"):
        driver.drive_session(
            "mock",
            ["ppo:42"],
            stage=tmp_path,
            source={**source(), "status": "unavailable"},
            command=lambda *a, **k: calls.append(a),
        )
    assert not calls


def test_failed_exec_preserves_all_siblings_and_serial_timeout(tmp_path, monkeypatch):
    calls = []
    preserved = []

    def command(*args, **kwargs):
        calls.append(args)
        if args[0] == "exec":
            assert kwargs["check"] is False
            return SimpleNamespace(returncode=1)

    monkeypatch.setattr(
        driver, "preserve_run", lambda session, name, *a, **kw: preserved.append(name)
    )
    monkeypatch.setattr(driver, "validate_completion", lambda *a, **kw: {})
    with pytest.raises(RuntimeError, match="remote batch"):
        driver.drive_session(
            "mock",
            ["ppo:42", "grpo_g8:42"],
            stage=tmp_path,
            source=source(),
            parallel=2,
            command=command,
        )
    assert preserved == ["ppo_s42", "grpo_g8_s42"]
    execution = next(a for a in calls if a[0] == "exec")
    assert execution[execution.index("--timeout") + 1] == "28800"
    assert calls[-1][0] == "stop"


def test_failed_preservation_still_attempts_siblings(tmp_path, monkeypatch):
    calls = []
    preserved = []

    def preserve(session, name, *a, **kw):
        preserved.append(name)
        if name == "ppo_s42":
            raise OSError("retry transfer")

    monkeypatch.setattr(driver, "preserve_run", preserve)
    with pytest.raises(OSError, match="retry transfer"):
        driver.drive_session(
            "mock",
            ["ppo:42", "grpo_g8:42"],
            stage=tmp_path,
            source=source(),
            parallel=2,
            command=lambda *a, **kw: calls.append(a),
        )
    assert preserved == ["ppo_s42", "grpo_g8_s42"]
    assert not any(a[0] == "stop" for a in calls)


def test_logprob_allocation_only_completion_positions(monkeypatch):
    torch = pytest.importorskip("torch")
    from platform_hybrid.experiments.samestack_training import completion_log_probs
    import torch.nn.functional as F

    full = torch.tensor([[1, 2, 3, 4, 0]])
    logits = torch.randn(1, 5, 6, requires_grad=True)
    expected = (
        F.log_softmax(logits[:, :-1, :], dim=-1).gather(-1, full[:, 1:, None]).squeeze(-1)[:, 2:]
    )
    original = F.log_softmax
    shapes = []
    monkeypatch.setattr(
        F, "log_softmax", lambda x, **kw: shapes.append(tuple(x.shape)) or original(x, **kw)
    )
    result, _ = completion_log_probs(
        lambda **kw: SimpleNamespace(logits=logits), full, torch.ones_like(full), 3
    )
    assert shapes == [(1, 2, 6)]
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    assert logits.grad is not None


def test_equivalence_description_uses_plan_margin_and_justification(monkeypatch):
    monkeypatch.syspath_prepend(str(ANALYSIS))
    plan_module = module(ANALYSIS / "analysis_plan.py", "pr20_plan")
    plan = plan_module.default_plan()
    plan["equivalence_margin"] = 0.03
    plan["margin_justification"] = "Synthetic sensitivity rationale"
    message = plan_module.analyze_runs([], plan)["inference"]["equivalence"]
    assert "+/-3pp" in message
    assert "Synthetic sensitivity rationale" in message
    assert "no supplied" not in message
