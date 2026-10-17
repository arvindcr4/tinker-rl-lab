"""Review08 regression fixtures. All keys/evidence are synthetic and temporary."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import statistics
import subprocess

import pytest

from platform_hybrid.registry.provenance import minreport as mr
from platform_hybrid.registry.provenance import sign


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))
    return {"path": path.name, "sha256": mr.digest(path.read_bytes())}


@pytest.fixture
def publication(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mr, "git_sha", lambda: "1" * 40)
    monkeypatch.setattr(mr, "worktree_identity", lambda: {
        "format": "git-tracked-worktree/1.0", "git_tree": "2" * 40,
        "tracked_diff_hash": mr.digest(b""), "dirty": False,
    })
    (tmp_path / "trainer.py").write_text("def reward(x):\n    return 1.0 if x else 0.0\n")
    min_report = {
        "loss_form": {"importance_ratio_level": "token", "clip_eps_low": None,
                      "clip_eps_high": None, "clipping_applied": False,
                      "length_normalization": False, "advantage_normalization": "std",
                      "token_mask": "completion", "token_aggregation": "token",
                      "reward_shaping_type": "none", "sampling_dynamic_filter": False},
        "reference_kl": {"reference_policy": False, "kl_beta": 0.0, "kl_estimator": "none"},
        "sampler_backend": {"backend": "synthetic-fixture", "precision": "float32", "temperature": 1.0, "top_p": 1.0},
        "telemetry": {"per_step_zvf": False, "per_step_gu": False, "source": "synthetic fixture"},
        "group_size_schedule": {"initial_g": 4, "schedule": "fixed", "adaptation_rule": "none"},
        "heldout_split": {"disjoint_from_reward_env": True, "description": "synthetic IDs", "n": 2},
        "decontamination": {"performed": True, "parser_robustness_probe": False},
    }
    runtime = {"code_file": "trainer.py", "data_ref": "synthetic-data", "min_report_rl": min_report}
    config_hash = mr.object_digest(runtime)
    split = write_json(tmp_path / "split.json", {"train_ids": ["t1", "t2"], "heldout_ids": ["h1", "h2"], "data_ref": "synthetic-data"})
    seeds, values = [0, 1, 2], [0.2, 0.4, 0.6]
    runs = [write_json(tmp_path / f"run-{seed}.json", {
        "run_id": "synthetic", "seed": seed, "status": "completed",
        "code_hash": mr.sha256_file("trainer.py"), "config_hash": config_hash,
        "split_sha256": split["sha256"], "heldout_metrics": {"accuracy": value},
    }) for seed, value in zip(seeds, values)]
    results = {"metric": "accuracy", "mean": statistics.mean(values), "sd": statistics.stdev(values),
               "ci": [0.1, 0.7], "confidence": 0.95,
               "per_seed": [{"seed": seed, "value": value} for seed, value in zip(seeds, values)]}
    analysis = write_json(tmp_path / "analysis.json", {
        **{key: results[key] for key in ("metric", "mean", "sd", "ci", "confidence")},
        "n": 3, "method": "synthetic fixture interval; no scientific conclusion",
        "input_run_sha256": [run["sha256"] for run in runs],
    })
    decontamination = write_json(tmp_path / "decontamination.json", {
        "method": "split-id-disjoint/1.0", "performed": True, "verified": True,
        "split_sha256": split["sha256"], "overlap_count": 0, "parser_robustness_probe": False,
    })
    config = {**runtime, "rigor": {"n_seeds": 3, "seeds": seeds, "reports_heldout": True,
                                 "heldout_disjoint": True, "reports_ci": True},
              "results": results, "evidence": {"runs": runs, "split": split,
                                                "analysis": analysis, "decontamination": decontamination}}
    write_json(tmp_path / "config.json", config)
    path = tmp_path / "record.json"
    record = mr.generate("synthetic", tmp_path / "config.json", path)
    key_dir = tmp_path / "keys"
    sign.keygen(key_dir=key_dir)
    public = key_dir / sign.PUB_ED.name
    return {"root": tmp_path, "path": path, "record": record, "key_dir": key_dir, "public": public}


def authenticated_rewrite(fixture, record):
    """Tests can authenticate bad evidence to demonstrate independent rejection."""
    fixture["path"].write_text(json.dumps(record))
    envelope = sign.sign_record(record, fixture["key_dir"])
    sign.sig_path(fixture["path"]).write_text(json.dumps(envelope))


def test_unsigned_diagnostic_never_awards_publication_acceptance(publication):
    report = mr.verify(publication["record"], strict=True)
    assert report["evidence_valid"], report["checks"]
    assert report["score_pct"] == 100
    assert not report["accepted"]
    assert not report["authenticated"]
    assert report["grade"] == "REJECTED"
    assert "UNAUTHENTICATED" in report["checks"][-1]["msg"]
    assert not mr.audit_file(publication["path"], publication["public"])["accepted"]


def test_file_publication_requires_trust_signature_and_evidence(publication):
    sign.sign_file(publication["path"], publication["key_dir"])
    report = mr.audit_file(publication["path"], publication["public"])
    assert report["accepted"], report["checks"]
    assert report["authenticated"]
    assert report["assurance"] == "trusted-ed25519"
    assert report["grade"] == "A"
    assert not mr.audit_file(publication["path"], None)["accepted"]
    assert not mr.audit_file(publication["path"], publication["public"], ("hmac-sha256",))["accepted"]


def test_sign_file_refuses_invalid_provenance(publication):
    publication["path"].write_text('{"run_id":"unsubstantiated"}')
    with pytest.raises(ValueError, match="not publication-ready"):
        sign.sign_file(publication["path"], publication["key_dir"])
    assert not sign.sig_path(publication["path"]).exists()


@pytest.mark.parametrize("change", [
    lambda record: record["rigor"].update(n_seeds=True),
    lambda record: record["rigor"].update(n_seeds=3, seeds=[42, 42, 42]),
    lambda record: record["rigor"].update(n_seeds=3, seeds=[0, 1]),
    lambda record: record["rigor"].update(seeds=[False, 1, 2]),
    lambda record: record.update(results={}),
    lambda record: record["min_report_rl"]["group_size_schedule"].update(initial_g=None),
    lambda record: record["min_report_rl"]["heldout_split"].update(n=None),
    lambda record: record["min_report_rl"]["heldout_split"].update(n=True),
    lambda record: record["min_report_rl"]["decontamination"].update(performed=False),
    lambda record: record["min_report_rl"]["decontamination"].update(verified=False),
    lambda record: record["min_report_rl"]["reference_kl"].update(kl_beta=1.0),
    lambda record: record["min_report_rl"]["loss_form"].update(clip_eps_low=0.0),
    lambda record: record["min_report_rl"].update(sampler_backend="unsupported-string"),
    lambda record: record["min_report_rl"].update(telemetry=["heldout"]),
    lambda record: record["evidence"].update(runs=[]),
    lambda record: record["results"].update(mean=0.999),
])
def test_authenticated_contradictory_evidence_rejected(publication, change):
    record = copy.deepcopy(publication["record"])
    change(record)
    authenticated_rewrite(publication, record)
    report = mr.audit_file(publication["path"], publication["public"])
    assert report["authenticated"]
    assert not report["accepted"]
    assert not report["evidence_valid"]
    assert report["grade"] != "A"


@pytest.mark.parametrize("field", ["git_sha", "code_hash", "reward_fn_hash", "config_hash", "resolved_config", "worktree"])
def test_integrity_checks_never_disappear_or_get_offset(publication, field):
    good_count = len(mr.verify(publication["record"])["checks"])
    record = copy.deepcopy(publication["record"])
    del record["provenance"][field]
    authenticated_rewrite(publication, record)
    report = mr.audit_file(publication["path"], publication["public"])
    assert len(report["checks"]) == good_count
    assert not report["accepted"]
    assert report["grade"] == "REJECTED"


@pytest.mark.parametrize("artifact_name", ["run-0.json", "split.json", "analysis.json", "decontamination.json"])
def test_changed_artifact_bytes_reject_publication(publication, artifact_name):
    sign.sign_file(publication["path"], publication["key_dir"])
    artifact = publication["root"] / artifact_name
    artifact.write_text(artifact.read_text() + "\n")
    assert not mr.audit_file(publication["path"], publication["public"])["accepted"]


@pytest.mark.parametrize(("artifact_name", "mutation"), [
    ("run-0.json", lambda doc: doc.update(status="running")),
    ("run-0.json", lambda doc: doc.update(seed=2)),
    ("run-0.json", lambda doc: doc.update(config_hash="sha256:" + "0" * 64)),
    ("split.json", lambda doc: doc.update(heldout_ids=["t1", "h2"])),
    ("analysis.json", lambda doc: doc.update(input_run_sha256=[])),
    ("analysis.json", lambda doc: doc.update(mean=0.99)),
    ("analysis.json", lambda doc: doc.update(ci=[0.7, 0.1])),
    ("decontamination.json", lambda doc: doc.update(verified=False)),
])
def test_authenticated_artifact_claims_still_require_consistency(publication, artifact_name, mutation):
    record = copy.deepcopy(publication["record"])
    path = publication["root"] / artifact_name
    value = sign.load_json(path)
    mutation(value)
    reference = write_json(path, value)
    for key in ("split", "analysis", "decontamination"):
        if record["evidence"][key]["path"] == artifact_name:
            record["evidence"][key] = reference
    for index, run in enumerate(record["evidence"]["runs"]):
        if run["path"] == artifact_name:
            record["evidence"]["runs"][index] = reference
    authenticated_rewrite(publication, record)
    report = mr.audit_file(publication["path"], publication["public"])
    assert report["authenticated"]
    assert not report["accepted"]


def test_missing_config_errors_and_unknowns_are_preserved(tmp_path):
    with pytest.raises(FileNotFoundError):
        mr.generate("missing", tmp_path / "no-config", tmp_path / "not-created")
    record = mr.generate("unknown", None, tmp_path / "draft")
    assert record["min_report_rl"]["heldout_split"]["disjoint_from_reward_env"] is None
    assert record["min_report_rl"]["heldout_split"]["n"] is None
    assert record["min_report_rl"]["decontamination"]["performed"] is None
    assert record["min_report_rl"]["telemetry"]["per_step_zvf"] is None
    assert record["provenance"]["code_hash"] is None
    with pytest.raises(FileExistsError):
        mr.generate("unknown", None, tmp_path / "draft")


def test_config_and_code_hashes_are_full_and_recomputed(publication):
    record = publication["record"]
    assert len(record["provenance"]["code_hash"]) == 71
    assert len(record["provenance"]["config_hash"]) == 71
    sign.sign_file(publication["path"], publication["key_dir"])
    source = publication["root"] / "trainer.py"
    source.write_text(source.read_text() + "\n# code outside reward changed\n")
    assert not mr.audit_file(publication["path"], publication["public"])["accepted"]


def test_git_sha_is_full_and_independent_of_callers_working_directory(tmp_path, monkeypatch):
    expected = subprocess.check_output(["git", "-C", str(mr.REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.chdir(tmp_path)
    assert mr.git_sha() == expected
    assert len(expected) in (40, 64)


def test_external_source_is_structured_failure(publication, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "code.py"
    outside.write_text("def reward(): return 1\n")
    record = copy.deepcopy(publication["record"])
    record["provenance"]["code_file"] = str(outside)
    report = mr.verify(record)
    assert not report["accepted"]
    assert any("provenance source missing" in check["msg"] for check in report["checks"])


def test_keygen_and_signing_never_replace_existing_anchor(publication):
    public = publication["public"]
    original = public.read_bytes()
    (publication["key_dir"] / sign.PRIV_ED.name).unlink()
    with pytest.raises(FileNotFoundError):
        sign.sign_record({"run_id": "x"}, publication["key_dir"])
    for force in (False, True):
        with pytest.raises((FileExistsError, ValueError)):
            sign.keygen(force=force, key_dir=publication["key_dir"])
    assert public.read_bytes() == original


def test_key_files_are_created_restrictively_under_permissive_umask(tmp_path, monkeypatch):
    directory = tmp_path / "keys"
    modes = []
    original_open = os.open
    def checked_open(path, flags, mode=0o777, **kwargs):
        modes.append((Path(path).name, flags, mode))
        return original_open(path, flags, mode, **kwargs)
    monkeypatch.setattr(sign.os, "open", checked_open)
    old = os.umask(0)
    try:
        fingerprint = sign.keygen(key_dir=directory)
    finally:
        os.umask(old)
    assert directory.stat().st_mode & 0o777 == 0o700
    assert (directory / sign.PRIV_ED.name).stat().st_mode & 0o777 == 0o600
    secret = next(entry for entry in modes if entry[0] == sign.PRIV_ED.name)
    assert secret[2] == 0o600
    assert secret[1] & os.O_EXCL
    assert (directory / (fingerprint.replace(":", "-") + ".pub")).exists()


def test_mismatched_keypair_is_rejected(publication, tmp_path):
    other = tmp_path / "other-keys"
    sign.keygen(key_dir=other)
    publication["public"].write_bytes((other / sign.PUB_ED.name).read_bytes())
    with pytest.raises(ValueError, match="does not match"):
        sign.sign_record({"run_id": "x"}, publication["key_dir"])


@pytest.mark.parametrize(("header", "value"), [
    ("spec", "P5-PROVENANCE-SIG/1.0"),
    ("canonicalization", "unversioned"),
    ("algorithm", "hmac-sha256"),
    ("key_id", "sha256:" + "0" * 64),
    ("digest_sha256", "0" * 64),
])
def test_protected_metadata_tampering_rejected(publication, header, value):
    record = {"run_id": "x"}
    envelope = sign.sign_record(record, publication["key_dir"])
    assert sign.verify_record(record, envelope, publication["public"])
    envelope["protected"][header] = value
    assert not sign.verify_record(record, envelope, publication["public"])


@pytest.mark.parametrize("document", [
    '{"run_id":"forged","run_id":"original"}',
    '{"nested":{"x":1,"x":2}}', '{"value":NaN}', '{"value":Infinity}',
    '{"value":-Infinity}', '{"value":1e999}', '{"value":"\\ud800"}',
])
def test_ambiguous_json_rejected_before_signing_and_verification(publication, document):
    publication["path"].write_text(document)
    with pytest.raises(ValueError, match="duplicate JSON property|nonfinite JSON number|finite, unambiguous|surrogates not allowed"):
        sign.sign_file(publication["path"], publication["key_dir"])
    assert not sign.verify_file(publication["path"], publication["public"])
    assert not mr.audit_file(publication["path"], publication["public"])["accepted"]


def test_duplicate_property_cannot_reuse_existing_signature(publication):
    record = {"run_id": "original"}
    envelope = sign.sign_record(record, publication["key_dir"])
    publication["path"].write_text('{"run_id":"forged","run_id":"original"}')
    sign.sig_path(publication["path"]).write_text(json.dumps(envelope))
    assert not sign.verify_file(publication["path"], publication["public"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), {1: "non-string-key"}])
def test_canonicalization_rejects_non_json_values(value):
    with pytest.raises(ValueError, match="only finite, unambiguous JSON values"):
        sign.canonical_bytes({"value": value})


def test_strict_envelope_no_added_unauthenticated_assurance(publication):
    envelope = sign.sign_record({"run_id": "x"}, publication["key_dir"])
    envelope["is_public_key_signature"] = False
    assert not sign.verify_record({"run_id": "x"}, envelope, publication["public"])


def test_no_automatic_hmac_fallback(publication, monkeypatch):
    monkeypatch.setattr(sign, "HAVE_ED25519", False)
    with pytest.raises(RuntimeError, match="no weaker fallback"):
        sign.sign_record({}, publication["key_dir"])


def test_verify_cli_extensionless_and_json_ancestor_preserve_input(publication):
    directory = publication["root"] / "ancestor.json"
    directory.mkdir()
    record = directory / "extensionless"
    record.write_bytes(publication["path"].read_bytes())
    before = record.read_bytes()
    assert mr.main(["verify", str(record), "--diagnostic"]) == 0
    assert record.read_bytes() == before
    audit = record.with_name(record.name + ".audit.json")
    assert audit.is_file()
    assert sign.load_json(audit)["accepted"] is False


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink", "signature", "source", "run", "trusted-key"])
def test_explicit_audit_output_cannot_destroy_evidence(publication, alias):
    path = publication["path"]
    before = path.read_bytes()
    if alias == "same":
        output = path
    elif alias in ("symlink", "hardlink"):
        output = publication["root"] / alias
        output.symlink_to(path) if alias == "symlink" else os.link(path, output)
    elif alias == "signature":
        sign.sign_file(path, publication["key_dir"])
        output = sign.sig_path(path)
    elif alias == "source":
        output = publication["root"] / "trainer.py"
    elif alias == "run":
        output = publication["root"] / "run-0.json"
    else:
        output = publication["public"]
    output_before = output.read_bytes()
    assert mr.main(["verify", str(path), "--out", str(output), "--trusted-key", str(publication["public"])]) == 1
    assert path.read_bytes() == before
    assert output.read_bytes() == output_before


def test_historical_record_is_read_only_and_never_publication_accepted():
    path = Path(__file__).resolve().parents[1] / "platform_hybrid/registry/provenance/campaign.provenance.json"
    before = path.read_bytes()
    record = sign.load_json(path)
    snapshot = copy.deepcopy(record)
    report = mr.verify(record)
    assert record == snapshot
    assert path.read_bytes() == before
    assert not report["schema_valid"]
    assert not report["accepted"]


def test_registry_and_provenance_use_identical_coverage(publication):
    from platform_hybrid.registry import query
    score, items = query.badge({"min_report": publication["record"]["min_report_rl"]})
    coverage = mr.reporting_coverage(publication["record"]["min_report_rl"])
    assert coverage["reported_leaf_coverage_v3"] == score / 100
    assert coverage["items"] == items


@pytest.mark.parametrize("value", [None, [], True, "bad", 1, {"min_report_rl": None},
                                   {"min_report_rl": {"loss_form": []}},
                                   {"provenance": {"code_file": "bad\x00path"}}])
def test_malformed_record_types_fail_without_crashing(value):
    assert not mr.verify(value)["accepted"]


def test_boolean_per_seed_value_is_not_a_numeric_result(publication):
    record = copy.deepcopy(publication["record"])
    record["results"]["per_seed"][0]["seed"] = False
    authenticated_rewrite(publication, record)
    assert not mr.audit_file(publication["path"], publication["public"])["accepted"]


def test_wrong_trusted_public_key_is_rejected(publication):
    other = publication["root"] / "different-identity"
    sign.keygen(key_dir=other)
    sign.sign_file(publication["path"], publication["key_dir"])
    assert not mr.audit_file(publication["path"], other / sign.PUB_ED.name)["accepted"]


def test_truncated_hashes_and_commit_cannot_pass(publication):
    record = copy.deepcopy(publication["record"])
    record["provenance"]["git_sha"] = "1" * 7
    record["provenance"]["code_hash"] = record["provenance"]["code_hash"][:23]
    record["provenance"]["config_hash"] = record["provenance"]["config_hash"][:23]
    authenticated_rewrite(publication, record)
    assert not mr.audit_file(publication["path"], publication["public"])["accepted"]


def test_existing_private_key_blocks_keygen_even_without_public(publication):
    private = publication["key_dir"] / sign.PRIV_ED.name
    before = private.read_bytes()
    publication["public"].unlink()
    with pytest.raises(FileExistsError):
        sign.keygen(key_dir=publication["key_dir"])
    assert private.read_bytes() == before


def test_minimum_seed_rule_never_claims_statistical_power(publication):
    report = mr.verify(publication["record"])
    assert not any("underpowered" in check["msg"].lower() for check in report["checks"])
    assert any("minimum distinct-seed" in check["msg"] for check in report["checks"])


@pytest.mark.parametrize("mutation", [
    lambda record: record["provenance"]["worktree"].update(dirty=0),
    lambda record: record["min_report_rl"]["sampler_backend"].update(backend=""),
    lambda record: record["min_report_rl"]["reference_kl"].update(kl_estimator="   "),
])
def test_empty_substantive_values_and_boolean_aliases_are_rejected(publication, mutation):
    record = copy.deepcopy(publication["record"])
    mutation(record)
    authenticated_rewrite(publication, record)
    assert not mr.audit_file(publication["path"], publication["public"])["accepted"]



def test_authenticated_boolean_config_cannot_equal_numeric_declaration(publication):
    record=copy.deepcopy(publication['record'])
    config=record['provenance']['resolved_config']
    config['min_report_rl']['sampler_backend']['temperature']=True
    record['provenance']['config_hash']=mr.object_digest(config)
    for reference in record['evidence']['runs']:
        path=publication['root']/reference['path'];doc=sign.load_json(path)
        doc['config_hash']=record['provenance']['config_hash']
        reference.update(write_json(path,doc))
    path=publication['root']/record['evidence']['analysis']['path'];doc=sign.load_json(path)
    doc['input_run_sha256']=[ref['sha256'] for ref in record['evidence']['runs']]
    record['evidence']['analysis'].update(write_json(path,doc))
    authenticated_rewrite(publication,record)
    audit=mr.audit_file(publication['path'],publication['public'])
    assert audit['authenticated'] and not audit['accepted']
    assert not audit['evidence_valid']
    with pytest.raises(ValueError,match='not publication-ready'):
        sign.sign_file(publication['path'],publication['key_dir'])
