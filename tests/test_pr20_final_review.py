"""Offline closure tests for schema, run identity and audit display review."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from samestack_fixtures import synthetic_run
from platform_hybrid.experiments import samestack_contract as contract

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "platform_hybrid/registry"
ANALYSIS = ROOT / "reports/project_finish_2026-10-11/validation/samestack_v2"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def schema_validator():
    from jsonschema import Draft202012Validator

    schema = json.loads((REG / "schema.json").read_text())
    return Draft202012Validator(schema)


@pytest.fixture
def registry_record():
    return json.loads((REG / "entries/tinker_dapo_qwen3.5-4b_gsm8k.json").read_text())


@pytest.mark.parametrize(
    "manifest",
    [
        {"revisions": []},
        {"revisions": {"model": {"revision": 42}}},
        {"train_item_ids": [True], "test_item_ids": ["one"]},
        {"test_item_ids": ["same", "same"]},
        {"decoding.top_k": True},
        {"generation": {"train": {"temperature": "warm"}}},
    ],
)
def test_schema_rejects_malformed_observed_manifest(schema_validator, registry_record, manifest):
    registry_record["effective_manifest"] = manifest
    assert list(schema_validator.iter_errors(registry_record))


def test_schema_accepts_typed_manifest(schema_validator, registry_record):
    registry_record["effective_manifest"] = {
        "revisions": {"model": {"id": "example", "revision": "abc"}},
        "train_item_ids": ["train:1"],
        "test_item_ids": ["test:1"],
        "decoding.top_k": 0,
    }
    assert not list(schema_validator.iter_errors(registry_record))


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(typo="unrecognized"),
        lambda p: p.update(
            coverage_provenance={"source": "x", "input_sha256": "bad", "computation_version": "v1"}
        ),
        lambda p: p.update(outcome_provenance={"mean": {"source": "x"}}),
        lambda p: p.update(generator_version=3),
    ],
)
def test_provenance_schema_rejects_bad_lineage(schema_validator, registry_record, change):
    change(registry_record["provenance"])
    assert list(schema_validator.iter_errors(registry_record))


def test_all_generated_provenance_shapes_validate(schema_validator, monkeypatch):
    monkeypatch.syspath_prepend(str(REG))
    import generate_entries as generator

    specs = {**generator.fc_family(), **generator.h2h_family(), **generator.e3_family()}
    for eid, spec in specs.items():
        old = json.loads((REG / "entries" / f"{eid}.json").read_text())
        assert not list(schema_validator.iter_errors(generator.build(eid, spec, old))), eid


@pytest.mark.parametrize("name", ["huggingface_hub", "safetensors", "tokenizers"])
def test_every_locked_package_prevents_inprocess_replacement(monkeypatch, name):
    from platform_colab.samestack_v2 import colab_run_arms as runner
    from platform_hybrid.experiments import samestack_dependencies as deps

    monkeypatch.setattr(deps, "runtime_versions", dict)
    monkeypatch.setitem(sys.modules, name, SimpleNamespace())
    monkeypatch.setattr(
        runner.subprocess, "check_call", lambda *a, **k: pytest.fail("must not install")
    )
    with pytest.raises(RuntimeError, match=name):
        runner.ensure_packages()
    assert set(deps.IMPORT_NAMES) == set(deps.LOCK)


def test_hardware_does_not_fabricate_training_tensor_precision(monkeypatch):
    from platform_hybrid.experiments import samestack_manifest as m

    torch = SimpleNamespace(
        cuda=SimpleNamespace(device_count=lambda: 0),
        version=SimpleNamespace(cuda=None),
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(version=lambda: None),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
        ),
        are_deterministic_algorithms_enabled=lambda: False,
    )
    monkeypatch.setattr(m.subprocess, "check_output", lambda *a, **k: "test driver")
    hardware = m.hardware_manifest(
        torch,
        SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa")),
        model_dtype="torch.bfloat16",
    )
    assert "logprob_dtype" not in hardware
    assert "value_head_dtype" not in hardware
    run, _ = synthetic_run()
    assert run["manifest"]["effective_config"]["precision"]["logprobs"] == "torch.float32"


@pytest.mark.parametrize("kind", ["template", "train", "test"])
def test_resume_identity_binds_template_and_ordered_items(kind):
    run, texts = synthetic_run()
    manifest = run["manifest"]
    if kind == "template":
        manifest["chat_template_hash"] = contract.canonical_hash("changed")
    else:
        key = kind + "_item_ids"
        manifest[key] = list(reversed(manifest[key]))
        manifest["ordered_" + kind + "_item_hash"] = contract.canonical_hash(manifest[key])
        if kind == "test":
            texts["test_item_ids"] = manifest[key]
        else:
            for target, index in [
                ("policy_item_ids", "policy"),
                ("critic_warmup_item_ids", "critic_warmup"),
            ]:
                manifest["schedules"][target] = [
                    [manifest[key][i] for i in row] for row in manifest["schedules"][index]
                ]
    report = contract.validate_run(run, texts=texts)
    assert any("run_id does not match" in message for message in report["errors"])
    assert not report["verified"]


def test_old_run_identity_version_is_not_silently_upgraded():
    run, texts = synthetic_run()
    del run["manifest"]["run_identity_version"]
    report = contract.validate_run(run, texts=texts)
    assert "run identity version 2 unavailable" in report["unverified"]
    assert not report["verified"]


def test_degenerate_wilcoxon_is_explicitly_unavailable(monkeypatch):
    monkeypatch.syspath_prepend(str(ANALYSIS))
    module = load_module(ANALYSIS / "v2_analysis.py", "pr20_final_analysis")
    records = [{"manifest": {"run_id": name}, "post_correct": [1, 0]} for name in ["a", "b"]]
    analysis = {
        "primary_contrasts": {
            "same": {"status": "available", "pairs": [{"a_run_id": "a", "b_run_id": "b"}]}
        },
        "exploratory_contrasts": {},
    }
    assert (
        module.item_diagnostics(records, analysis, B=2)["same"]["wilcoxon_items_seedavg_p"] is None
    )


def test_browser_assertion_helpers_reject_missing_and_wrong_values():
    module = load_module(ROOT / "submission/demo/web/tests/test_demo.py", "pr20_browser_checks")
    assert module.p_pair_matches("0.016 / 0.048", 0.016, 0.048)
    assert module.p_pair_matches("1.2e-5 / 0.000", 0.000012, 0.00000001) is False
    assert not module.p_pair_matches("0.016 / 0.040", 0.016, 0.048)
    assert not module.p_pair_matches("Combined TOST", 0.016, 0.048)
    assert not module.p_pair_matches("nan / nan", None, None)
    for audit in [{}, {"disagreements": None}, {"disagreements": True}, {"disagreements": "2"}]:
        assert not module.disagreement_disclosed(audit, ": 2 disagreements")
    assert module.disagreement_disclosed(
        {"disagreements": 2}, "3 stored answers: 2 disagreements; coverage incomplete"
    )
    assert not module.disagreement_disclosed({"disagreements": 2}, "12 disagreements")
