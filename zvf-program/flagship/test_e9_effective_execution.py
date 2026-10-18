"""Synthetic exact-byte contract tests; no upstream authority or model execution."""
import hashlib
import json
from dataclasses import replace

import pytest
import e9_mle_bench_streaming as core


ORIGINAL = b"import unused\nprint('fixture')\n"
EFFECTIVE = b"print('fixture')\n"
SUBMISSION = b"id,prediction\r\na,0.1\r\nb,0.9\r\n"
GRADE = b'{"score": 0.5, "valid_submission": true}'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def policy():
    return core.RepairPolicy(
        repair_id="synthetic-remove-unused-import-v1", competition_id="synthetic",
        source_sha256=digest(ORIGINAL), effective_sha256=digest(EFFECTIVE),
        allowed_error_signature="No module named 'unused'", changes_task_semantics=False,
        review_reference="synthetic-fixture-only-not-official-policy",
        columns=("id", "prediction"), identifier_columns=("id",),
        ordered_identifiers=(("a",), ("b",)), finite_columns=("prediction",))


def record(monkeypatch, selected=None):
    monkeypatch.setattr(core, "REPAIR_POLICIES", (selected or policy(),))
    return core.authorize_program_repair(competition_id="synthetic", original=ORIGINAL,
        effective=EFFECTIVE, error="ModuleNotFoundError: No module named 'unused'")


def binding(monkeypatch, **changes):
    repair = record(monkeypatch)
    args = dict(competition_id="synthetic", original=ORIGINAL, effective=EFFECTIVE,
                submission=SUBMISSION, native_grade=GRADE, repairs=[repair],
                grader_input={"competition_id": "synthetic", "submission_sha256": digest(SUBMISSION),
                              "grader_revision": "synthetic-local-fixture"})
    args.update(changes)
    return core.bind_effective_execution(**args)


def test_exact_program_csv_and_native_input_binding(monkeypatch):
    result = binding(monkeypatch)
    assert result["original_program_sha256"] == digest(ORIGINAL)
    assert result["effective_program_sha256"] == digest(EFFECTIVE)
    assert result["submission_sha256"] == digest(SUBMISSION)
    assert result["native_grade_sha256"] == digest(GRADE)
    assert result["repairs"][0]["changes_task_semantics"] is False


@pytest.mark.parametrize("changes", [
    {"competition_id": "different"}, {"original": ORIGINAL + b"\n"},
    {"effective": EFFECTIVE + b"\n"}, {"error": "different failure"},
])
def test_repair_preconditions_bind_exact_task_source_output_and_failure(monkeypatch, changes):
    monkeypatch.setattr(core, "REPAIR_POLICIES", (policy(),))
    args = dict(competition_id="synthetic", original=ORIGINAL, effective=EFFECTIVE,
                error="No module named 'unused'")
    args.update(changes)
    with pytest.raises(core.LaunchGateError):
        core.authorize_program_repair(**args)


def test_empty_registry_never_promotes_historical_pattern_recognition():
    assert core.REPAIR_POLICIES == ()
    with pytest.raises(core.LaunchGateError, match="no unique reviewed"):
        core.authorize_program_repair(competition_id="synthetic", original=ORIGINAL,
            effective=EFFECTIVE, error="No module named 'unused'")


def test_semantic_repair_requires_explicit_policy_decision(monkeypatch):
    with pytest.raises(core.LaunchGateError, match="benchmark-policy decision"):
        record(monkeypatch, replace(policy(), changes_task_semantics=True))
    approved = replace(policy(), changes_task_semantics=True,
                       benchmark_policy_decision="synthetic-test-decision-not-upstream-approval")
    assert record(monkeypatch, approved)["changes_task_semantics"] is True


@pytest.mark.parametrize("submission", [
    b"prediction,id\n0.1,a\n0.9,b\n", b"id,prediction\na,0.1\n",
    b"id,prediction\nb,0.9\na,0.1\n", b"id,prediction\na,0.1\na,0.9\n",
    b"id,prediction\na,nan\nb,0.9\n", b"id,prediction\na,inf\nb,0.9\n",
    b"id,prediction\na,0.1,extra\nb,0.9\n", b"\xff",
])
def test_repaired_csv_contract_rejects_columns_rows_ids_order_and_nonfinite(submission):
    with pytest.raises(core.LaunchGateError, match="submission contract"):
        core.validate_repair_submission(submission, policy())


def test_same_csv_values_different_bytes_cannot_substitute_native_input(monkeypatch):
    with pytest.raises(core.LaunchGateError, match="exact submission bytes"):
        binding(monkeypatch, submission=SUBMISSION.replace(b"\r\n", b"\n"))


@pytest.mark.parametrize("changes", [{"repairs": []}, {"grader_input": None},
    {"native_grade": b'{"score": 0.5, "valid_submission": false}'}, {"submission": None}])
def test_no_success_without_chain_submission_and_explicit_native_validity(monkeypatch, changes):
    with pytest.raises(core.LaunchGateError):
        binding(monkeypatch, **changes)


def test_unchanged_failed_program_is_bound_without_fake_grade():
    result = core.bind_effective_execution(competition_id="synthetic", original=ORIGINAL,
        effective=ORIGINAL, submission=None,
        native_grade=json.dumps({"score": None, "valid_submission": False}).encode(),
        grader_input=None, repairs=[])
    assert result["submission_sha256"] is None
    assert result["repairs"] == []


def test_missing_submission_can_retain_invalid_native_grade():
    grade = b'{"score": null, "valid_submission": false, "submission_exists": false}'
    result = core.bind_effective_execution(competition_id="synthetic", original=ORIGINAL,
        effective=ORIGINAL, submission=None, native_grade=grade, repairs=[],
        grader_input={"competition_id": "synthetic", "submission_sha256": None,
                      "grader_revision": "synthetic-fixture"})
    assert result["submission_sha256"] is None
    assert result["native_grade_sha256"] == digest(grade)


@pytest.mark.parametrize("mutate", [False, True])
def test_production_native_grade_boundary_records_exact_input_and_rejects_mutation(tmp_path, monkeypatch, mutate):
    pytest.importorskip("modal")
    import modal_e9_mle_bench_streaming as runner
    submission = tmp_path / "submission.csv"
    submission.write_bytes(SUBMISSION)
    grade_path = tmp_path / "native_grade.json"
    def local_grader(command, **kwargs):
        assert command[-2] == str(submission)
        assert submission.read_bytes() == SUBMISSION
        grade_path.write_bytes(GRADE)
        if mutate:
            submission.write_bytes(SUBMISSION.replace(b"0.1", b"0.2"))
    monkeypatch.setattr(runner, "_run", local_grader)
    if mutate:
        with pytest.raises(RuntimeError, match="changed during grading"):
            runner._grade(tmp_path, "synthetic", submission, grade_path)
        assert not grade_path.with_suffix(".input.json").exists()
    else:
        assert runner._grade(tmp_path, "synthetic", submission, grade_path) == json.loads(GRADE)
        inputs = json.loads(grade_path.with_suffix(".input.json").read_text())
        assert inputs["submission_sha256"] == digest(SUBMISSION)
        assert inputs["competition_id"] == "synthetic"
        assert inputs["grader_revision"] == runner.MLE_BENCH_COMMIT
        assert len(inputs["grader_script_sha256"]) == 64
