"""Review09 adversarial boundaries, using only offline synthetic fixtures."""

from __future__ import annotations

import hashlib
import base64
import json
from pathlib import Path
import subprocess
import sys
import types
import zipfile

import pytest

from tools import anonymize_export as anon
from tools import audit_thesis_numbers as audit
from tools import build_pes_review_pack as pack
from tools import check_thesis_evidence as evidence
from utils import verify_results as verify

ROOT = Path(__file__).resolve().parents[2]


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("destination", ["root", "ancestor", "source", "unrelated", "symlink"])
def test_anonymizer_never_changes_existing_destination(tmp_path, monkeypatch, destination):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "tests"
    source.mkdir()
    sentinel = source / "sentinel"
    sentinel.write_bytes(b"original source")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    other = unrelated / "sentinel"
    other.write_bytes(b"unrelated")
    link = tmp_path / "link"
    link.symlink_to(unrelated, target_is_directory=True)
    monkeypatch.setattr(anon, "ROOT", root)
    out = {
        "root": root,
        "ancestor": tmp_path,
        "source": source,
        "unrelated": unrelated,
        "symlink": link,
    }[destination]
    with pytest.raises(ValueError, match="new dedicated"):
        anon.main(["--out", str(out)])
    assert sentinel.read_bytes() == b"original source"
    assert other.read_bytes() == b"unrelated"


@pytest.mark.parametrize("failure", ["read", "leak", "concurrent"])
def test_anonymizer_stages_and_preserves_prior_success(tmp_path, monkeypatch, failure):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "LICENSE").write_text("safe content")
    prior = tmp_path / "previous"
    prior.mkdir()
    (prior / "sentinel").write_bytes(b"previous export")
    out = tmp_path / "new"
    monkeypatch.setattr(anon, "ROOT", root)
    monkeypatch.setattr(anon, "tracked", lambda: ["LICENSE"])
    if failure == "read":
        monkeypatch.setattr(anon, "tracked", lambda: (_ for _ in ()).throw(OSError("read failed")))
        with pytest.raises(OSError, match="read failed|injected failure"):
            anon.main(["--out", str(out)])
    elif failure == "leak":
        monkeypatch.setattr(anon, "rewrite", lambda text, counts: text)
        (root / "LICENSE").write_text("arvindcr4")
        assert anon.main(["--out", str(out)]) == 1
    else:
        publish = anon.publish_new_directory

        def collision(staging, target):
            target.mkdir()
            (target / "other").write_bytes(b"concurrent owner")
            publish(staging, target)

        monkeypatch.setattr(anon, "publish_new_directory", collision)
        with pytest.raises(FileExistsError):
            anon.main(["--out", str(out)])
        assert (out / "other").read_bytes() == b"concurrent owner"
    assert (prior / "sentinel").read_bytes() == b"previous export"
    assert not list(tmp_path.glob(".anon-export-*"))
    if failure != "concurrent":
        assert not out.exists()


def test_anonymizer_publishes_validated_export(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "LICENSE").write_text("safe content")
    monkeypatch.setattr(anon, "ROOT", root)
    monkeypatch.setattr(anon, "tracked", lambda: ["LICENSE"])
    out = tmp_path / "export"
    assert anon.main(["--out", str(out)]) == 0
    assert (out / "LICENSE").read_text() == "safe content"
    assert json.loads((out / "ANON_MANIFEST.json").read_text())["n_included"] == 1


@pytest.mark.parametrize(
    "paths",
    [
        ["tools/arvind.py", "tools/Anonymous Author.py"],
        ["ANON_MANIFEST.json"],
    ],
)
def test_anonymizer_rejects_rewritten_or_reserved_destination_collisions(
    tmp_path, monkeypatch, paths
):
    root = tmp_path / "repo"
    root.mkdir()
    for index, name in enumerate(paths):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"original file {index}")
    monkeypatch.setattr(anon, "ROOT", root)
    monkeypatch.setattr(anon, "tracked", lambda: paths)
    prior = tmp_path / "prior-export"
    prior.mkdir()
    (prior / "sentinel").write_bytes(b"previous export")
    out = tmp_path / "new-export"
    with pytest.raises(ValueError, match="destination collision"):
        anon.main(["--all", "--out", str(out)])
    assert not out.exists()
    assert (prior / "sentinel").read_bytes() == b"previous export"
    for index, name in enumerate(paths):
        assert (root / name).read_text() == f"original file {index}"


@pytest.mark.parametrize(
    ("token", "source"),
    [
        ("GRPO 0.9, PPO 0.1", "GRPO 0.1, PPO 0.9"),
        ("1/190", "190/1"),
        ("[0.1, 0.9]", "[0.9, 0.1]"),
        ("90%", "90 kg"),
        ("0.9%", "10.9%"),
        ("0.9%", "-0.9%"),
        ("1 ms", "11 ms"),
        ("1 ms", "1 msec"),
        ("1 ms", "1 ms/s"),
        ("1 ms", "1 ms / s"),
        ("0.9%", "0.9%/day"),
        ("1 ms", "1 ms^2"),
    ],
)
def test_transcription_preserves_order_labels_and_units(token, source):
    assert not audit.in_source(token, source)


@pytest.mark.parametrize(
    ("token", "source"),
    [("0.9%", "10.9%"), ("1 ms", "11 ms"), ("1 ms", "1 ms/s"), ("0.9%", "0.9%/day")],
)
def test_full_evaluator_rejects_composite_numeric_substrings(tmp_path, monkeypatch, token, source):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "THESIS", "thesis")
    monkeypatch.setattr(audit, "CHAPTERS", {"ch06": "chapter.md"})
    (tmp_path / "thesis").mkdir()
    (tmp_path / "source.txt").write_text(f"Measurement = {source}")
    (tmp_path / "thesis/chapter.md").write_text(
        f"<!-- claim: measurement -->Measurement: {token}<!-- /claim: measurement -->"
    )
    (tmp_path / "thesis/claim_bindings.json").write_text(
        json.dumps(
            {
                "schema": "thesis-claim-bindings-v2",
                "claims": {"Measurement": [{"span": "measurement", "label": "Measurement"}]},
            }
        )
    )
    monkeypatch.setattr(
        audit, "claims", lambda: iter([("Measurement", "ch06", [token], "T", ["source.txt"], None)])
    )
    monkeypatch.setattr(audit, "claims_ext", lambda: iter(()))
    monkeypatch.setattr(audit, "prose_claims", lambda ch: iter(()))
    groups = audit.evaluate()
    assert groups[0]["tokens"][0]["evidence_result"] != "PASS"
    assert audit.overall_status(groups) != "PASS"


@pytest.mark.parametrize("mutation", ["swapped", "labels", "elsewhere", "duplicate", "unbound"])
def test_full_evaluator_binds_claim_to_its_own_span(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "THESIS", "thesis")
    monkeypatch.setattr(audit, "CHAPTERS", {"ch06": "chapter.md"})
    (tmp_path / "thesis").mkdir()
    chapter = "<!-- claim: grpo -->GRPO: 0.90<!-- /claim: grpo -->\n<!-- claim: ppo -->PPO: 0.10<!-- /claim: ppo -->"
    if mutation in {"swapped", "elsewhere"}:
        chapter = chapter.replace("0.90", "SWAP").replace("0.10", "0.90").replace("SWAP", "0.10")
    if mutation == "labels":
        chapter = chapter.replace("GRPO:", "PPO:")
    if mutation == "duplicate":
        chapter += chapter
    chapter += "\nCorrect-looking tokens elsewhere: 0.90 0.10"
    (tmp_path / "thesis/chapter.md").write_text(chapter)
    if mutation != "unbound":
        (tmp_path / "thesis/claim_bindings.json").write_text(
            json.dumps(
                {
                    "schema": "thesis-claim-bindings-v2",
                    "claims": {
                        "GRPO": [{"span": "grpo", "label": "GRPO"}],
                        "PPO": [{"span": "ppo", "label": "PPO"}],
                    },
                }
            )
        )
    monkeypatch.setattr(
        audit,
        "claims",
        lambda: iter(
            [
                ("GRPO", "ch06", ["0.90"], "R", [], lambda: [0.9]),
                ("PPO", "ch06", ["0.10"], "R", [], lambda: [0.1]),
            ]
        ),
    )
    monkeypatch.setattr(audit, "claims_ext", lambda: iter(()))
    monkeypatch.setattr(audit, "prose_claims", lambda ch: iter(()))
    groups = audit.evaluate()
    assert audit.overall_status(groups) != "PASS"
    assert groups[0]["status"] == ("UNBOUND" if mutation == "unbound" else "FAIL")


@pytest.mark.parametrize(
    ("loader", "filename", "contents"),
    [
        (audit.J, "bad.json", '{"score":0.25,"score":0.75}'),
        (audit.jsonl, "bad.jsonl", '{"score":0.25,"score":0.75}\n'),
    ],
)
def test_audit_loaders_reject_duplicate_keys(tmp_path, monkeypatch, loader, filename, contents):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    audit.text.cache_clear()
    loader.cache_clear()
    (tmp_path / filename).write_text(contents)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        loader(filename)


@pytest.mark.parametrize(
    ("field", "value"), [("correct", "false"), ("native_accepted", 1), ("correct", None)]
)
def test_e14_requires_real_booleans(tmp_path, monkeypatch, field, value):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "E14_DISP", "rows.jsonl")
    row = {"correct": False, "native_accepted": True, "actor_usage": {"completion_tokens": 1}}
    row[field] = value
    (tmp_path / "rows.jsonl").write_text(json.dumps(row))
    for reader in (audit.text, audit.jsonl, audit.e14):
        reader.cache_clear()
    with pytest.raises(ValueError, match="must be boolean"):
        audit.e14()


def test_replacement_rejects_overlapping_e1_tasks_with_reconciled_totals():
    path = ROOT / evidence.FINISH / "E1/result.json"
    d = json.loads(path.read_text())
    # Replace the existing single resolution with two matching-count resolutions
    # of one task in separate waves; aggregate arithmetic still agrees.
    parts = list(d["breakdown"].values())
    for part in parts:
        part["resolved"] = 1
        part["resolved_ids"] = ["same-task"]
        for denominator in ("graded", "attempted"):
            part[f"resolved_over_{denominator}"] = round(1 / part[denominator], 4)
    d.update(
        n_resolved=2, resolved_ids=["same-task", "same-task"], score=round(2 / d["n_attempted"], 4)
    )
    d["secondary_resolved_over_graded"]["score"] = round(2 / d["n_graded"], 4)
    with pytest.raises(ValueError, match="duplicate"):
        evidence.check_replacement("E1", d)


def test_replacement_rejects_impossible_e5_subset_even_when_totals_agree():
    d = json.loads((ROOT / evidence.FINISH / "E5/result.json").read_text())
    parts = list(d["subsets"].values())
    parts[0].update(n=1, passes=2, pass1=2.0)
    parts[1].update(n=d["n_attempted"] - 1, passes=d["n_success"] - 2)
    parts[1]["pass1"] = round(parts[1]["passes"] / parts[1]["n"], 4)
    with pytest.raises(ValueError, match="impossible subset"):
        evidence.check_replacement("E5", d)


def test_replacement_rejects_e9_success_on_failed_return_code():
    d = json.loads((ROOT / evidence.FINISH / "E9/result.json").read_text())
    passing = next(r for r in d["per_task"] if r["success"])
    failing = next(r for r in d["per_task"] if not r["success"])
    passing["rc"], failing["rc"] = 1, 0
    d["n_errors_or_timeouts"] = sum(r["rc"] != 0 for r in d["per_task"])
    d["n_graded"] = d["n_attempted"] - d["n_errors_or_timeouts"]
    with pytest.raises(ValueError, match="successful task has a nonzero return code"):
        evidence.check_replacement("E9", d)


@pytest.mark.parametrize(
    "name",
    ["gsm8k_qwen3_8b_g20_s42", "gsm8k_qwen3_8b_g2_unknown_s42", "prefix_gsm8k_qwen3_8b_g2_s42"],
)
def test_experiment_matching_never_uses_substrings(name):
    assert verify._match_key(name, {"gsm8k_qwen3_8b_g2": {}}) is None


@pytest.mark.parametrize(("last10", "peak"), [(1.02, 1.05), (-0.01, 0.5), (0.8, 0.7)])
@pytest.mark.parametrize("format", ["json", "cli", "legacy", "reference"])
def test_accuracy_invariants_identical_across_formats(tmp_path, last10, peak, format):
    expected = {"demo": {"last10": 0.5, "peak": 0.8, "seed": 42}}
    path = tmp_path / ("demo_s42.json" if format == "json" else "demo_s42.log")
    if format == "reference":
        expected["demo"].update(last10=last10, peak=peak)
    elif format == "json":
        path.write_text(json.dumps({"last10": last10, "peak": peak}))
    elif format == "cli":
        path.write_text(f"[grpo_cli] Seed 42 done.\navg_last10 : {last10}\npeak_reward : {peak}\n")
    else:
        path.write_text(f"Last-10 avg accuracy: {last10 * 100}%\nPeak accuracy: {peak * 100}%\n")
    try:
        rows, failures = verify.verify(tmp_path, expected, 1.0, 1.0)
    except ValueError:
        return
    assert failures > 0
    assert not any(row[-1] is True for row in rows)


def test_filename_payload_experiment_conflict_fails(tmp_path):
    (tmp_path / "first_s42.json").write_text(
        json.dumps({"experiment": "second_s42", "last10": 0.5, "peak": 0.8})
    )
    refs = {k: {"last10": 0.5, "peak": 0.8, "seed": 42} for k in ("first", "second")}
    rows, failures = verify.verify(tmp_path, refs, 0.05, 0.1)
    assert failures == 1
    assert rows[0][-1] is None


def bound_package():
    snapshot = {"thesis/chapter.md": b"GRPO: 0.90", "evidence.json": b'{"accuracy":0.9}'}
    audit_record = {
        "schema_version": "thesis-number-provenance-v2",
        "status": "PASS",
        "thesis_directory": "thesis",
        "source_bindings": [{"path": k, "sha256": digest(v)} for k, v in snapshot.items()],
        "groups": [
            {
                "chapter": "chapter.md",
                "status": "PASS",
                "tokens": [{"result": "PASS", "manuscript_binding": "matched"}],
                "artifacts": [
                    {"path": "evidence.json", "sha256": digest(snapshot["evidence.json"])}
                ],
            }
        ],
    }
    return audit_record, snapshot


@pytest.mark.parametrize("changed", ["status_only", "chapter", "evidence", "unbound", "missing"])
def test_packager_rejects_unbound_or_stale_audit(changed):
    a, snapshot = bound_package()
    if changed == "status_only":
        a = {"status": "PASS"}
    elif changed in ("chapter", "evidence"):
        snapshot["thesis/chapter.md" if changed == "chapter" else "evidence.json"] += b"changed"
    elif changed == "unbound":
        a["groups"][0]["tokens"][0]["manuscript_binding"] = "missing"
    else:
        snapshot.pop("evidence.json")
    with pytest.raises(ValueError, match="audit|source|binding|module|v2"):
        pack.validate_audit(a, snapshot)


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_packager_failed_rebuild_preserves_archive(tmp_path, monkeypatch, failure):
    target = tmp_path / "review.zip"
    before = b"previous successful archive"
    target.write_bytes(before)

    def fail(*args, **kwargs):
        raise OSError("injected failure")

    if failure == "write":
        monkeypatch.setattr(zipfile.ZipFile, "writestr", fail)
    else:
        monkeypatch.setattr(pack.os, "replace", fail)
    with pytest.raises(OSError, match="read failed|injected failure"):
        pack.publish_archive(target, {"one": b"1", "two": b"2"})
    assert target.read_bytes() == before
    assert not list(tmp_path.glob(".review-pack-*"))


def test_packager_writes_exact_validated_snapshot(tmp_path):
    a, snapshot = bound_package()
    pack.validate_audit(a, snapshot)
    target = tmp_path / "review.zip"
    pack.publish_archive(target, snapshot)
    with zipfile.ZipFile(target) as archive:
        assert {name: archive.read(name) for name in archive.namelist()} == snapshot


@pytest.mark.parametrize(
    "text",
    [
        "Installation instructions superseded\n" + "filler\n" * 80 + "DAPO DISAPPEARS",
        "This dependency version is invalid\nDAPO DISAPPEARS",
        "<!-- historical-dapo-2026-08-02:start -->\nDAPO DISAPPEARS\n<!-- historical-dapo-2026-08-02:end -->\nDAPO DISAPPEARS",
    ],
)
def test_stale_banner_never_exempts_current_verdict(tmp_path, text):
    from tools import check_stale_verdicts as gate

    path = tmp_path / "note.md"
    path.write_text(text)
    assert len(gate.find_unmarked_occurrences(path, "note.md")) == 1


def test_failed_module_load_restores_only_replaced_key(tmp_path):
    from tests._shared_fakes import load_module, temporary_modules

    path = tmp_path / "bad.py"
    path.write_text("raise RuntimeError('broken load')")
    original = types.ModuleType("review09_original")
    sys.modules["review09_failure"] = original
    try:
        with pytest.raises(RuntimeError):
            load_module("review09_failure", path)
        assert sys.modules["review09_failure"] is original
        imported = types.ModuleType("review09_genuine")
        with temporary_modules({"review09_failure": None}):
            sys.modules["review09_genuine"] = imported
        assert sys.modules["review09_genuine"] is imported
        assert sys.modules["review09_failure"] is original
    finally:
        sys.modules.pop("review09_failure", None)
        sys.modules.pop("review09_genuine", None)


def test_lightweight_tools_import_without_ml_stack():
    code = """import sys
class BlockML:
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'torch', 'numpy', 'pandas', 'transformers'}:
            raise ImportError('ML stack intentionally absent')
sys.meta_path.insert(0, BlockML())
import tests.conftest
from tools import anonymize_export, audit_thesis_numbers, check_thesis_evidence, build_pes_review_pack
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_audit_json_nonfinite_exception_does_not_extend_to_other_fields(value):
    with pytest.raises(ValueError, match="non-finite"):
        audit.strict_json_loads(json.dumps({"score": value}))
    with pytest.raises(ValueError, match="non-finite"):
        audit.strict_json_loads(
            json.dumps({"runs": [{"algo": "ppo", "step_log": [{"mean_reward": value}]}]}),
            f"{audit.RES}/samestack_ppo_grpo.json",
        )


def test_historical_ppo_nan_exception_is_field_and_path_specific():
    source = json.dumps({"runs": [{"algo": "ppo", "step_log": [{"zvf": float("nan")}]}]})
    assert audit.strict_json_loads(source, f"{audit.RES}/samestack_ppo_grpo.json")["runs"]
    with pytest.raises(ValueError, match="non-finite"):
        audit.strict_json_loads(source, "unrelated.json")


@pytest.mark.parametrize("case", ["missing", "skip", "failed"])
def test_required_scientific_collection_gate_catches_missing_or_skipped_modules(tmp_path, case):
    from tools.check_required_test_report import check_report

    report = tmp_path / "junit.xml"
    inner = {
        "missing": "",
        "skip": '<testcase classname="tests.required"><skipped/></testcase>',
        "failed": '<testcase classname="tests.required"><failure/></testcase>',
    }[case]
    report.write_text("<testsuites><testsuite>" + inner + "</testsuite></testsuites>")
    with pytest.raises(ValueError, match="audit|source|binding|module|v2"):
        check_report(report, ["required"])


def test_unknown_experiment_beside_valid_result_does_not_get_omitted(tmp_path):
    for name in ("demo", "demo_unknown"):
        (tmp_path / f"{name}_s42.json").write_text(json.dumps({"last10": 0.5, "peak": 0.8}))
    rows, failures = verify.verify(
        tmp_path, {"demo": {"last10": 0.5, "peak": 0.8, "seed": 42}}, 0.05, 0.1
    )
    assert len(rows) == 2
    assert failures == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("last10", 1.02),
        ("last10", 0.4),
        ("last10", True),
        ("peak_reward", 1.05),
        ("peak_reward", 0.7),
        ("peak_reward", "0.8"),
    ],
)
def test_result_metric_aliases_cannot_hide_invalid_or_conflicting_values(tmp_path, field, value):
    record = {"last10_avg": 0.5, "peak": 0.8, field: value}
    (tmp_path / "demo_s42.json").write_text(json.dumps(record))
    rows, failures = verify.verify(
        tmp_path, {"demo": {"last10": 0.5, "peak": 0.8, "seed": 42}}, 0.05, 0.1
    )
    assert failures == 1
    assert rows[0][-1] is None


def test_result_identical_metric_aliases_are_unambiguous(tmp_path):
    record = {"last10_avg": 0.5, "last10": 0.5, "peak": 0.8, "peak_reward": 0.8}
    (tmp_path / "demo_s42.json").write_text(json.dumps(record))
    rows, failures = verify.verify(
        tmp_path, {"demo": {"last10": 0.5, "peak": 0.8, "seed": 42}}, 0.05, 0.1
    )
    assert failures == 0
    assert rows[0][-1] is True


def test_anonymizer_rejects_dangling_destination_symlink(tmp_path):
    link = tmp_path / "destination"
    link.symlink_to(tmp_path / "not-yet-existing")
    with pytest.raises(ValueError, match="new dedicated"):
        anon.main(["--out", str(link)])
    assert link.is_symlink()
    assert not link.exists()


def test_audit_hashes_the_manuscript_bytes_it_actually_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "THESIS", "thesis")
    monkeypatch.setattr(audit, "CHAPTERS", {"ch06": "chapter.md"})
    directory = tmp_path / "thesis"
    directory.mkdir()
    original = b"<!-- claim: grpo -->GRPO: 0.25<!-- /claim: grpo -->"
    (directory / "chapter.md").write_bytes(original)
    (directory / "claim_bindings.json").write_text(
        json.dumps(
            {
                "schema": "thesis-claim-bindings-v2",
                "claims": {"GRPO": [{"span": "grpo", "label": "GRPO"}]},
            }
        )
    )

    def compute():
        (directory / "chapter.md").write_text("concurrent edit: GRPO 0.75")
        return [0.25]

    monkeypatch.setattr(
        audit, "claims", lambda: iter([("GRPO", "ch06", ["0.25"], "R", [], compute)])
    )
    monkeypatch.setattr(audit, "claims_ext", lambda: iter(()))
    monkeypatch.setattr(audit, "prose_claims", lambda ch: iter(()))
    assert audit.main(["--check"]) == 0
    report = json.loads((directory / "provenance_audit.json").read_text())
    sources = {item["path"]: item["sha256"] for item in report["source_bindings"]}
    assert sources["thesis/chapter.md"] == digest(original)
    assert sources["thesis/chapter.md"] != digest((directory / "chapter.md").read_bytes())


def test_serving_recomputation_uses_the_bytes_bound_by_artifact_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "THESIS", "thesis")
    monkeypatch.setattr(audit, "CHAPTERS", {"ch06": "chapter.md"})
    monkeypatch.setattr(audit, "E1_DIR", "e1")
    monkeypatch.setattr(audit, "E13_RAW", "absent/*.raw")
    monkeypatch.setattr(audit, "finish_reasons", lambda _: [])
    artifact = "e1/tasks/t1/generation.json"
    path = tmp_path / artifact
    path.parent.mkdir(parents=True)
    original = b'{"response_tokens": 1}'
    path.write_bytes(original)
    directory = tmp_path / "thesis"
    directory.mkdir()
    token = "1/1 Tinker and 0/0 vLLM"
    (directory / "chapter.md").write_text(
        f"<!-- claim: serving -->Serving: {token}<!-- /claim: serving -->"
    )
    (directory / "claim_bindings.json").write_text(
        json.dumps(
            {
                "schema": "thesis-claim-bindings-v2",
                "claims": {"Serving": [{"span": "serving", "label": "Serving"}]},
            }
        )
    )

    def compute():
        path.write_text('{"response_tokens": 8192}')
        return [audit.serving_rows()[0]]

    monkeypatch.setattr(
        audit, "claims", lambda: iter([("Serving", "ch06", [token], "R", [artifact], compute)])
    )
    monkeypatch.setattr(audit, "claims_ext", lambda: iter(()))
    monkeypatch.setattr(audit, "prose_claims", lambda ch: iter(()))
    groups = audit.evaluate()
    assert groups[0]["artifacts"][0]["sha256"] == digest(original)
    assert groups[0]["tokens"][0]["recomputed"] == "0/1 Tinker and 0/0 vLLM"
    assert audit.overall_status(groups) == "FAIL"


@pytest.mark.parametrize("layer", ["outer", "body"])
def test_response_journals_reject_duplicate_json_keys(tmp_path, monkeypatch, layer):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    body = '{"choices": [], "choices": [{"finish_reason": "length"}]}'
    if layer == "outer":
        source = '{"response": {}, "response": {"choices": []}}'
    else:
        source = json.dumps({"raw_body_base64": base64.b64encode(body.encode()).decode()})
    (tmp_path / "journal.jsonl").write_text(source)
    audit.text.cache_clear()
    with pytest.raises(ValueError, match="duplicate JSON key"):
        audit.finish_reasons("journal.jsonl")
