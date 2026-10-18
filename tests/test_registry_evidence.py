"""Fail-closed registry evidence, strict-validator freshness and source mutation tests."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REG = Path(__file__).resolve().parents[1] / "platform_hybrid" / "registry"
sys.path.insert(0, str(REG))
import query as q
import generate_entries as g


def known(value):
    return {"value": value, "status": "logged", "source": "fixture.json#observed"}


def stack():
    return {
        "id": "fixture",
        "label_claimed": "grpo",
        "model": None,
        "run_config": {field: known(f"value:{field}") for field in q.FIELDS},
        "variant_deltas_applied": [],
        "min_report": {item: {} for item in q.ITEMS},
    }


@pytest.fixture
def registry(tmp_path, monkeypatch):
    root = tmp_path / "repo" / "platform_hybrid" / "registry"
    (root / "entries").mkdir(parents=True)
    (root / "schema.json").write_text("{}")
    stale = root.parent / "experiments/results/p5p8/p6_iter118_strict_audit.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps({"findings": [], "n_stack_entries": 0, "n_deltas": 0}))
    producer = root.parents[1] / "platform_modal/scripts/p5p8/p6_iter118_strict_validator.py"
    producer.parent.mkdir(parents=True)
    producer.write_text("# isolated producer identity fixture\n")
    monkeypatch.setattr(q, "HERE", root)
    return root


def audit_for(registry, **extra):
    manifest = q.registry_input_manifest(registry)
    return {
        "input_manifest": manifest,
        "input_sha256": q.registry_manifest_sha256(manifest),
        "validator_version": q.VALIDATOR_VERSION,
        "validator_sha256": hashlib.sha256(
            (
                registry.parents[1] / "platform_modal/scripts/p5p8/p6_iter118_strict_validator.py"
            ).read_bytes()
        ).hexdigest(),
        "findings": [],
        "n_stack_entries": 0,
        "n_deltas": 0,
        **extra,
    }


def test_strict_producer_exit_seven_cannot_reuse_clean_old_audit(registry, monkeypatch, capsys):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=7, stdout="producer output\n", stderr="producer failed\n"
        ),
    )
    assert q.cmd_validate_strict(SimpleNamespace(include_warn=False)) == 7
    captured = capsys.readouterr()
    assert "producer output" in captured.out
    assert "producer failed" in captured.err
    assert "exited 7" in captured.err
    assert "CI GATE: PASS" not in captured.out


@pytest.mark.parametrize(
    "mode",
    [
        "missing",
        "malformed",
        "non_object",
        "wrong_hash",
        "wrong_manifest",
        "bad_findings",
        "input_changed",
    ],
)
def test_strict_rejects_missing_malformed_or_unbound_audit(registry, monkeypatch, mode):
    def producer(command, **kwargs):
        out = Path(command[command.index("--output") + 1])
        assert not out.exists()
        audit = audit_for(registry)
        if mode == "wrong_hash":
            audit["input_sha256"] = "0" * 64
        if mode == "wrong_manifest":
            audit["input_manifest"] = {}
        if mode == "bad_findings":
            audit["findings"] = [{"severity": "unexpected"}]
        if mode == "input_changed":
            (registry / "schema.json").write_text('{"title":"changed"}')
        if mode != "missing":
            out.write_text(
                "not json"
                if mode == "malformed"
                else "[]"
                if mode == "non_object"
                else json.dumps(audit)
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", producer)
    assert q.cmd_validate_strict(SimpleNamespace(include_warn=False)) == 1


def test_strict_success_uses_unique_bound_outputs_and_forwards_stderr(
    registry, monkeypatch, capsys
):
    paths = []

    def producer(command, **kwargs):
        out = Path(command[command.index("--output") + 1])
        paths.append(out)
        assert not out.exists()
        assert command[command.index("--input-sha256") + 1] == audit_for(registry)["input_sha256"]
        out.write_text(json.dumps(audit_for(registry)))
        return SimpleNamespace(returncode=0, stdout="", stderr="producer diagnostic\n")

    monkeypatch.setattr(subprocess, "run", producer)
    assert q.cmd_validate_strict(SimpleNamespace(include_warn=False)) == 0
    assert q.cmd_validate_strict(SimpleNamespace(include_warn=False)) == 0
    assert paths[0] != paths[1]
    assert not any(p.exists() for p in paths)
    assert "producer diagnostic" in capsys.readouterr().err


def test_real_strict_producer_writes_requested_bound_audit(tmp_path):
    registry = tmp_path / "registry"
    (registry / "entries").mkdir(parents=True)
    (registry / "schema.json").write_text("{}")
    output = tmp_path / "fresh/audit.json"
    producer = REG.parents[1] / "platform_modal/scripts/p5p8/p6_iter118_strict_validator.py"
    digest = q.registry_manifest_sha256(q.registry_input_manifest(registry))
    run = subprocess.run(
        [
            sys.executable,
            str(producer),
            "--registry",
            str(registry),
            "--output",
            str(output),
            "--input-sha256",
            digest,
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(output.read_text())["input_sha256"] == digest
    rejected = tmp_path / "must-not-exist.json"
    run = subprocess.run(
        [
            sys.executable,
            str(producer),
            "--registry",
            str(registry),
            "--output",
            str(rejected),
            "--input-sha256",
            "wrong",
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0
    assert not rejected.exists()


@pytest.mark.parametrize(
    ("model", "variant"),
    [
        ("Qwen/Qwen3-8B", None),
        ("x/Model-Base", "base"),
        ("x/Model-Instruct", "instruct"),
        ("x/Model-it", "instruct"),
    ],
)
def test_variant_requires_positive_name_evidence(model, variant):
    assert g.variant_rule(model)["value"] == variant


def test_min_report_does_not_invent_negative_evidence():
    spec = {"telemetry": (None, None, None)}
    report = g.to_min_report({}, {}, spec)
    assert report["decontamination"] == {"performed": None, "parser_robustness_probe": None}
    spec["decontamination"] = {"performed": known(False), "parser_robustness_probe": False}
    assert g.to_min_report({}, {}, spec)["decontamination"] == {
        "performed": False,
        "parser_robustness_probe": None,
    }
    assert report["group_size_schedule"]["schedule"] is None


def test_missing_launcher_does_not_establish_effective_defaults():
    for spec in g.h2h_family().values():
        rc = spec["run_config"]
        for field in (
            "decoding.top_p",
            "decoding.max_tokens",
            "train.sampler_refresh",
            "data.subset",
        ):
            assert rc[field]["status"] == "unknown", field
        assert rc["decoding.temperature"]["status"] == "logged"


def generated_fixture():
    old = json.loads((REG / "entries/colab-open_dapo_e3.json").read_text())
    return old, g.e3_family()["colab-open_dapo_e3"]


def test_outcomes_are_clean_deepcopy_and_manual_annotations_are_separate():
    old, spec = generated_fixture()
    old["outcomes"]["stale_statistic"] = 999
    old["outcomes"]["ci_method"] = {"method": "stale"}
    old["outcomes"]["controller_predicted_savings_per_rollout"] = {"made_up": 123}
    old["provenance"]["manual_annotations"] = {"review": "keep separately"}
    generated = g.build(old["id"], spec, old)
    for key in ("stale_statistic", "ci_method", "controller_predicted_savings_per_rollout"):
        assert key not in generated["outcomes"]
    assert generated["provenance"]["manual_annotations"] == {"review": "keep separately"}
    generated["outcomes"]["mean_zvf"] = 321
    assert spec["outcomes"]["mean_zvf"] != 321
    for key in spec["outcomes"]:
        provenance = generated["provenance"]["outcome_provenance"][key]
        assert "#" in provenance["source"]
        assert len(provenance["input_sha256"]) == 64
        assert provenance["computation_version"]


def test_versioned_coverage_separates_reporting_absence_and_implementation():
    old, spec = generated_fixture()
    spec["deltas"] = [("delta_dapo", "a", "absent", "documented missing implementation")]
    generated = g.build(old["id"], spec, old)
    coverage = generated["outcomes"]["coverage"]
    assert coverage["declared_delta_reporting_completeness_v2"] == 1
    assert coverage["declared_delta_implementation_fraction_v1"] == 0
    assert coverage["declared_delta_surrogate_fraction_v1"] == 0
    assert coverage["reported_leaf_coverage_v3"] < coverage["item_presence_coverage_v3"]
    assert coverage["coverage_version"] == q.COVERAGE_VERSION
    assert "min_report_coverage" not in coverage
    assert "declared_deltas_coverage" not in coverage


def source(tmp_path, monkeypatch, text):
    monkeypatch.setattr(g, "ROOT", tmp_path)
    (tmp_path / "fixture.py").write_text(text)
    return g.Src("fixture.py")


@pytest.mark.parametrize(("batch", "beta1", "beta2"), [(16, 0.8, 0.97), (24, 0.85, 0.96)])
def test_source_literals_drive_batch_and_betas(tmp_path, monkeypatch, batch, beta1, beta2):
    src = source(
        tmp_path,
        monkeypatch,
        f"args = GRPOConfig(generation_batch_size={batch}, num_generations=8)\n"
        f"opt = T.AdamParams(beta1={beta1}, beta2={beta2})\n",
    )
    assert src.prompts_per_step()["value"] == batch // 8
    assert src.optimizer_betas()["value"] == [beta1, beta2]
    with pytest.raises(SystemExit, match="source assumption mismatch"):
        src.cell(r"generation_batch_size=(\d+)", 8)
    with pytest.raises(SystemExit, match="source assumption mismatch"):
        src.cell(r"beta1=([\d.]+), beta2=([\d.]+)", [0.9, 0.95])


def test_nonliteral_or_nondivisible_batch_fails_closed(tmp_path, monkeypatch):
    src = source(
        tmp_path, monkeypatch, "args = GRPOConfig(generation_batch_size=9, num_generations=8)\n"
    )
    with pytest.raises(SystemExit, match="integer multiple"):
        src.prompts_per_step()
    src = source(tmp_path, monkeypatch, "opt = T.AdamParams(beta1=choose(), beta2=0.95)\n")
    with pytest.raises(SystemExit, match="nonliteral"):
        src.optimizer_betas()


@pytest.mark.parametrize(
    "field",
    [
        "runtime.gpu",
        "runtime.environment",
        "tokenizer.id",
        "tokenizer.revision",
        "data.revision",
        "data.ordered_example_identity",
        "implementation.identity",
        "decoding.top_k",
        "decoding.repetition_penalty",
    ],
)
def test_extended_catalogue_detects_changed_and_unknown_fields(field):
    a, b = stack(), stack()
    b["run_config"][field] = known("different")
    assert field in {f for f, *_ in q.stackdiff(a, b)["differs"]}
    b["run_config"][field] = {"value": None, "status": "unknown", "source": "not observed"}
    result = q.stackdiff(a, b)
    assert result["level"] == 0
    assert result["no_known_differences"]
    assert not result["evidence_complete"]
    assert not result["matching_requirements"]["satisfied"]


def test_effective_manifest_overrides_old_values_and_preserves_order():
    a, b = stack(), stack()
    manifest = {
        "environment_hash": "env-sha",
        "hardware": {"gpu_names": ["T4"]},
        "revisions": {
            "model": {"id": "model", "revision": "rev"},
            "tokenizer": {"id": "tokenizer", "revision": "tok-rev"},
            "dataset": {"id": "dataset", "revision": "data-rev"},
        },
        "generation": {"train": {"top_k": 0, "repetition_penalty": 1.0}},
        "train_item_ids": ["a", "b"],
        "test_item_ids": ["c", "d"],
        "source": {"commit": "code-rev", "source_files": {"train.py": "code-sha"}},
    }
    a["effective_manifest"] = manifest
    b["effective_manifest"] = copy.deepcopy(manifest)
    view = q.field_view(a)
    assert view["tokenizer.revision"]["value"] == "tok-rev"
    assert view["decoding.top_k"]["value"] == 0
    assert view["runtime.environment"]["value"] == "env-sha"
    assert view["implementation.identity"]["value"]["commit"] == "code-rev"
    b["effective_manifest"]["train_item_ids"].reverse()
    assert q.stackdiff(a, b)["differs"][0][0] == "data.ordered_example_identity"


def test_missing_checkpoint_revision_is_never_matching():
    a = stack()
    a["run_config"]["checkpoint.revision"] = g.unknown("unpinned")
    result = q.stackdiff(a, a)
    assert result["level"] == 0
    assert result["underspecified"]
    assert not result["evidence_complete"]
    assert not result["matching_requirements"]["satisfied"]


def test_generated_records_validate_against_additive_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((REG / "schema.json").read_text())
    for eid, spec in {**g.fc_family(), **g.h2h_family(), **g.e3_family()}.items():
        old = json.loads((REG / "entries" / f"{eid}.json").read_text())
        jsonschema.validate(g.build(eid, spec, old), schema)


def test_explicit_unknown_in_manifest_overrides_historical_assumption():
    rec = stack()
    rec["effective_manifest"] = {
        "revisions": {"model": {"revision": None}},
        "source": {"status": "unknown", "commit": None},
    }
    view = q.field_view(rec)
    assert view["checkpoint.revision"]["status"] == "unknown"
    assert view["implementation.identity"]["status"] == "unknown"
    assert not q.stackdiff(rec, rec)["matching_requirements"]["satisfied"]


def test_partial_example_identity_and_environment_name_are_not_complete_evidence():
    view = q.manifest_view({"train_item_ids": ["a"], "environment": "modal"})
    assert "data.ordered_example_identity" not in view
    assert "runtime.environment" not in view


def test_source_keyword_ambiguity_fails_closed(tmp_path, monkeypatch):
    src = source(
        tmp_path,
        monkeypatch,
        "a = T.AdamParams(beta1=0.9, beta2=0.95)\nb = T.AdamParams(beta1=0.8, beta2=0.96)\n",
    )
    with pytest.raises(SystemExit, match="expected one"):
        src.optimizer_betas()
