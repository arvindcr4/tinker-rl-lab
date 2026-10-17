"""Offline regressions for schema, atomic artifact preservation and strict JSON."""
import copy
import gzip
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

from samestack_fixtures import synthetic_run
from platform_hybrid.experiments import samestack_contract as contract
from platform_hybrid.experiments import samestack_dependencies as dependencies
from platform_colab.samestack_v2 import colab_run_arms as runner
from platform_colab.samestack_v2 import drive_sessions as driver
from platform_colab.samestack_v2 import verify_and_merge as verifier


def write_bundle(directory, *, smoke=True, arm="grpo_g8", seed=42, marker=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    record, texts = synthetic_run(arm, seed, smoke=smoke)
    name = f"{arm}_s{seed}"
    result = directory / f"{name}.json"
    result.write_text(json.dumps(record, allow_nan=False))
    with gzip.open(directory / f"{name}_texts.json.gz", "wt") as stream:
        json.dump(texts, stream, allow_nan=False)
    adapter = directory / "adapters" / name
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"SYNTHETIC MODEL BYTES, NOT A REAL MODEL")
    for file in ("adapter_config.json", "tokenizer_config.json", "tokenizer.json"):
        (adapter / file).write_text('{"synthetic": true}')
    for mode in ("train", "eval"):
        (adapter / f"{mode}_generation_config.json").write_text(json.dumps(record["manifest"]["generation"][mode]))
    if marker:
        contract.write_completion(result)
    return result, record, texts


@pytest.mark.parametrize("arm", contract.ARMS)
@pytest.mark.parametrize("smoke", [True, False])
def test_complete_synthetic_schema(arm, smoke):
    record, texts = synthetic_run(arm, smoke=smoke)
    result = contract.validate_run(record, filename=f"{arm}_s42.json", texts=texts)
    assert result["verified"], result


def test_review_c5_contradictory_reproducer_is_rejected(tmp_path):
    record, texts = synthetic_run("ppo_8x8", smoke=False)
    record.update(arm="WRONG", seed=999, group=0, prompts_per_step=999,
                  heldout_post_acc=1, execution_environment={})
    record.pop("manifest")
    texts["pre"] = []
    path = tmp_path / "ppo_8x8_s42.json"
    path.write_text(json.dumps(record))
    with gzip.open(tmp_path / "ppo_8x8_s42_texts.json.gz", "wt") as stream:
        json.dump(texts, stream)
    result = verifier.verify_stage(tmp_path)
    check = result["runs"][path.name]
    assert not result["all_verified"]
    assert not check["valid"]
    assert any("filename" in error for error in check["errors"])
    assert any("heldout_post_acc" in error for error in check["errors"])
    assert any("texts.pre" in error for error in check["errors"])


@pytest.mark.parametrize(("field", "mutation"), [
    ("pre_correct", lambda r: r["pre_correct"].pop()),
    ("post_correct", lambda r: r["post_correct"].__setitem__(0, 2)),
    ("pre_trunc", lambda r: r["pre_trunc"].__setitem__(0, True)),
    ("post_trunc", lambda r: r["post_trunc"].append(0)),
    ("heldout_pre_acc", lambda r: r.__setitem__("heldout_pre_acc", 0.5)),
    ("n_steps", lambda r: r.__setitem__("n_steps", 999)),
    ("group", lambda r: r.__setitem__("group", 0)),
    ("config_hash", lambda r: r["manifest"].__setitem__("config_hash", "0" * 64)),
    ("environment_hash", lambda r: r["manifest"].__setitem__("environment_hash", "0" * 64)),
    ("ordered", lambda r: r["manifest"]["test_item_ids"].reverse()),
    ("schedule", lambda r: r["manifest"]["schedules"]["policy"][0].reverse()),
    ("run_id", lambda r: r["manifest"].__setitem__("run_id", "0" * 64)),
    ("revision", lambda r: r["manifest"]["revisions"]["model"].__setitem__("revision", "0" * 40)),
])
def test_semantic_mutations_rejected(field, mutation):
    record, texts = synthetic_run()
    record = copy.deepcopy(record)
    mutation(record)
    result = contract.validate_run(record, texts=texts)
    assert not result["verified"], (field, result)
    assert not result["valid"], (field, result)


def test_missing_legacy_manifest_is_unverified_and_not_rewritten(tmp_path):
    record, texts = synthetic_run(smoke=False)
    record.pop("manifest")
    path = tmp_path / "grpo_g8_s42.json"
    path.write_text(json.dumps(record))
    original = path.read_bytes()
    result = contract.validate_run(record, texts=texts)
    assert result["valid"]
    assert not result["verified"]
    assert "legacy" in " ".join(result["unverified"])
    with pytest.raises(ValueError, match="not merging"):
        verifier.merge_verified(tmp_path, tmp_path / "canonical")
    assert path.read_bytes() == original


def test_empty_staging_not_success(tmp_path):
    report = verifier.verify_stage(tmp_path)
    assert report["staged_count"] == 0
    assert not report["all_verified"]
    with pytest.raises(ValueError, match="empty"):
        verifier.merge_verified(tmp_path, tmp_path / "canonical")


def test_duplicate_identity_rejected():
    record, texts = synthetic_run()
    seen = set()
    assert contract.validate_run(record, texts=texts, seen=seen)["verified"]
    assert "duplicate run identity" in contract.validate_run(record, texts=texts, seen=seen)["errors"]


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_json_rejected(value):
    with pytest.raises(ValueError, match="Non-finite"):
        contract.strict_json_loads('{"value": ' + value + '}')


def test_nan_literal_text_is_preserved_and_duplicate_keys_rejected():
    assert contract.strict_json_loads('{"text": "NaN Infinity"}')["text"] == "NaN Infinity"
    with pytest.raises(ValueError, match="Duplicate JSON"):
        contract.strict_json_loads('{"seed":42,"seed":999}')


def test_no_json_only_resume_and_all_artifacts_required(tmp_path):
    path, _, _ = write_bundle(tmp_path, marker=False)
    assert contract.load_completed(path) is None
    contract.write_completion(path)
    assert contract.load_completed(path) is not None
    (tmp_path / "adapters/grpo_g8_s42/tokenizer_config.json").unlink()
    assert contract.load_completed(path) is None


def test_changed_content_and_identity_invalidate_completion(tmp_path):
    path, record, _ = write_bundle(tmp_path)
    assert contract.load_completed(path, expected={"config_hash": "0" * 64}) is None
    assert contract.load_completed(path, expected={"source_commit": "0" * 40}) is None
    record["heldout_post_acc"] = 0.8
    path.write_text(json.dumps(record))
    assert contract.load_completed(path) is None


def test_download_failure_after_json_retries_and_preserves_bundle(tmp_path):
    remote, stage = tmp_path / "remote", tmp_path / "stage"
    source, _, _ = write_bundle(remote)
    downloaded, fail = [], [True]
    def fake(*args, **kwargs):
        assert args[0] == "download"
        relative = args[3].split(driver.PROTOCOL_ID + "/", 1)[1]
        downloaded.append(relative)
        if relative.endswith("_texts.json.gz") and fail[0]:
            fail[0] = False
            raise subprocess.CalledProcessError(1, "mock download")
        destination = Path(args[4])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote / relative, destination)
    with pytest.raises(subprocess.CalledProcessError):
        driver.preserve_run("SYNTHETIC", source.stem, stage, command=fake)
    assert not (stage / source.stem).exists()
    assert any(stage.glob(".grpo_g8_s42.partial-*"))
    driver.preserve_run("SYNTHETIC", source.stem, stage, command=fake)
    assert downloaded.count("grpo_g8_s42.json") == 2
    assert contract.validate_completion(stage / source.stem / source.name)


def test_session_does_not_stop_when_artifact_preservation_fails(tmp_path, monkeypatch):
    source = {"commit": "a" * 40, "dirty_hash": "b" * 64,
              "source_files": {str(driver.RUNNER.relative_to(driver.ROOT)): contract.sha256_file(driver.RUNNER)}}
    commands = []
    def fail(*args, **kwargs):
        raise OSError("synthetic interrupted transfer")
    monkeypatch.setattr(driver, "preserve_run", fail)
    with pytest.raises(OSError, match="interrupted"):
        driver.drive_session("SYNTHETIC", ["ppo:42"], stage=tmp_path, source=source,
                             command=lambda *args, **kwargs: commands.append(args))
    assert any(command[0] == "exec" for command in commands)
    assert not any(command[0] == "stop" for command in commands)


def test_staging_merge_keeps_sources_and_verified_inventory(tmp_path):
    stage, canonical = tmp_path / "stage", tmp_path / "canonical"
    path, _, _ = write_bundle(stage / "grpo_g8_s42", smoke=False)
    original = path.read_bytes()
    report = verifier.merge_verified(stage, canonical)
    assert report["all_verified"]
    assert path.read_bytes() == original
    assert contract.validate_completion(canonical / path.stem / path.name)


def test_installed_wrong_versions_are_not_accepted(monkeypatch):
    observed = dict(dependencies.LOCK, transformers="0.0.0")
    monkeypatch.setattr(dependencies, "runtime_versions", lambda: observed)
    with pytest.raises(ValueError, match="runtime lock mismatch"):
        runner.ensure_packages(install=False)


def test_imports_do_not_launch_or_stub_modal(monkeypatch):
    original = sys.modules.get("modal")
    def forbidden(*args, **kwargs):
        raise AssertionError("import must not start a service or install packages")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "check_call", forbidden)
    for module in (runner, driver, verifier):
        importlib.reload(module)
    assert sys.modules.get("modal") is original


def test_source_archive_hashes_checked_before_extract(tmp_path):
    archive = tmp_path / "source.zip"
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps({"status": "observed", "source_files": {"module.py": "0" * 64}}))
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("module.py", "print('SYNTHETIC')")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.prepare_bundle(archive, source_path, tmp_path / "target")
    assert not (tmp_path / "target/module.py").exists()


def test_remote_inventory_cannot_escape_run_directory():
    with pytest.raises(ValueError, match="unsafe"):
        driver._safe_inventory({"files": {"../outside": {}}}, "grpo_g8_s42")


def test_duplicate_canonical_identity_cannot_be_remerged_under_new_path(tmp_path):
    stage, canonical = tmp_path / "stage", tmp_path / "canonical"
    write_bundle(stage / "grpo_g8_s42", smoke=False)
    write_bundle(canonical / "different-location", smoke=False)
    report = verifier.verify_stage(stage, canonical)
    assert not report["all_verified"]
    assert any("duplicate" in error for row in report["runs"].values() for error in row["errors"])


def test_existing_json_never_skips_colab_execution(tmp_path, monkeypatch):
    existing = tmp_path / "ppo_s42.json"
    existing.write_text('{"incomplete": true}')
    source = {"commit": "a" * 40, "dirty_hash": "b" * 64,
              "source_files": {str(driver.RUNNER.relative_to(driver.ROOT)): contract.sha256_file(driver.RUNNER)}}
    commands = []
    monkeypatch.setattr(driver, "preserve_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, "validate_completion", lambda *args, **kwargs: {})
    driver.drive_session("SYNTHETIC", ["ppo:42"], stage=tmp_path, source=source,
                         command=lambda *args, **kwargs: commands.append(args))
    assert sum(command[0] == "exec" for command in commands) == 1
    assert commands[-1][0] == "stop"
    assert existing.read_text() == '{"incomplete": true}'


def test_partial_legacy_provenance_is_unverified_not_fabricated():
    record, texts = synthetic_run()
    record["manifest"] = {"environment": "colab"}
    texts.pop("test_item_ids")
    result = contract.validate_run(record, texts=texts)
    assert result["valid"], result
    assert not result["verified"]
    assert record["manifest"] == {"environment": "colab"}


def test_saved_generation_config_must_match_manifest(tmp_path):
    path, _, _ = write_bundle(tmp_path, marker=False)
    (tmp_path / "adapters/grpo_g8_s42/train_generation_config.json").write_text('{"top_k": 50}')
    with pytest.raises(ValueError, match="generation config contradicts"):
        contract.write_completion(path)


def test_stale_imported_source_requires_runtime_restart(tmp_path, monkeypatch):
    import types
    target = tmp_path / "target/helpers/fake.py"
    module = types.ModuleType("platform_hybrid.experiments.samestack_fake")
    module.__file__ = str(target)
    module.__samestack_source_hash__ = "0" * 64
    monkeypatch.setitem(sys.modules, module.__name__, module)
    archive, source_path = tmp_path / "source.zip", tmp_path / "source.json"
    data = b"# SYNTHETIC source"
    import hashlib
    source_path.write_text(json.dumps({"status": "observed", "source_files": {"helpers/fake.py": hashlib.sha256(data).hexdigest()}}))
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("helpers/fake.py", data)
    with pytest.raises(RuntimeError, match="restart"):
        runner.prepare_bundle(archive, source_path, tmp_path / "target")
    assert not target.exists()


@pytest.mark.parametrize("field", ["protocol_hash", "environment", "ordered_test_item_hash"])
def test_malformed_identity_returns_invalid_instead_of_crashing(field):
    record, texts = synthetic_run()
    record["manifest"][field] = [{"malformed": True}]
    result = contract.validate_run(record, texts=texts, seen=set())
    assert not result["valid"]


def test_mean_reward_cannot_silently_bypass_summary_recomputation():
    record, texts = synthetic_run()
    record["step_log"][0].pop("mean_reward")
    assert not contract.validate_run(record, texts=texts)["valid"]
