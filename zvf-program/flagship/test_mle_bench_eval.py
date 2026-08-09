from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from . import mle_bench_eval as runner  # noqa: F401
else:
    try:
        from . import mle_bench_eval as runner
    except ImportError:
        _TEST_DIR = Path(__file__).resolve().parent
        if str(_TEST_DIR) not in sys.path:
            sys.path.insert(0, str(_TEST_DIR))
        runner = import_module("mle_bench_eval")


_PINNED_CHECKOUT = Path(__file__).resolve().parents[2] / "outputs/e9_mle_bench/mle-bench-source"


def _fake_checkout(root: Path) -> Path:
    """A minimal MLE-bench-shaped checkout so tests never need the real one."""

    (root / "mlebench/competitions/tiny-comp").mkdir(parents=True)
    (root / "mlebench/competitions/huge-comp").mkdir(parents=True)
    (root / "experiments/splits").mkdir(parents=True)
    (root / "environment").mkdir(parents=True)

    (root / "mlebench/registry.py").write_text("# registry\n", encoding="utf-8")
    (root / "LICENSE").write_text("MIT\n", encoding="utf-8")
    (root / "environment/Dockerfile").write_text("FROM ubuntu:20.04\n", encoding="utf-8")
    (root / "environment/grading_server.py").write_text("# server\n", encoding="utf-8")

    (root / "experiments/splits/split75.txt").write_text("huge-comp\ntiny-comp\n", encoding="utf-8")
    (root / "experiments/splits/low.txt").write_text("tiny-comp\n", encoding="utf-8")
    (root / "experiments/splits/medium.txt").write_text("huge-comp\n", encoding="utf-8")
    (root / "experiments/splits/high.txt").write_text("", encoding="utf-8")

    with (root / "experiments/competition_categories.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["competition_id", "category", "dataset_size_GB", "EnabledDate", "DeadlineDate", "Complexity"])
        writer.writerow(["huge-comp", "Image Classification", "120.5", "", "", "Medium"])
        writer.writerow(["tiny-comp", "Text Classification", "0.002", "", "", "Low"])

    for comp, grader in (("tiny-comp", "log-loss"), ("huge-comp", "auc-roc")):
        comp_dir = root / "mlebench/competitions" / comp
        (comp_dir / "config.yaml").write_text(
            "id: {c}\n"
            "name: {c}\n"
            "grader:\n"
            "  name: {g}\n"
            "  grade_fn: mlebench.competitions.{c}.grade:grade\n".format(c=comp, g=grader),
            encoding="utf-8",
        )
        (comp_dir / "grade.py").write_text("def grade(s, a):\n    return 0.0\n", encoding="utf-8")
        (comp_dir / "prepare.py").write_text("def prepare(raw, public, private):\n    return None\n", encoding="utf-8")
        (comp_dir / "checksums.yaml").write_text("zip: deadbeef\n", encoding="utf-8")
        (comp_dir / "leaderboard.csv").write_text("teamId,score\n1,0.1\n2,0.2\n", encoding="utf-8")

    return root


class TaskIdHashTests(unittest.TestCase):
    def test_task_id_hash_is_order_independent(self) -> None:
        self.assertEqual(runner.task_ids_sha256(["b", "a"]), runner.task_ids_sha256(["a", "b"]))

    def test_task_id_hash_ignores_blank_and_whitespace_entries(self) -> None:
        self.assertEqual(runner.task_ids_sha256(["a", "", " b ", "\n"]), runner.task_ids_sha256(["a", "b"]))

    def test_task_id_hash_changes_when_the_task_set_changes(self) -> None:
        self.assertNotEqual(runner.task_ids_sha256(["a", "b"]), runner.task_ids_sha256(["a", "c"]))

    def test_task_id_hash_matches_the_documented_algorithm(self) -> None:
        import hashlib

        expected = hashlib.sha256(b"a\nb\n").hexdigest()
        self.assertEqual(runner.task_ids_sha256(["b", "a"]), expected)

    def test_sha256_file_returns_none_for_a_missing_file(self) -> None:
        self.assertIsNone(runner.sha256_file(Path("/nonexistent/definitely/not/here.txt")))


class SurveyTests(unittest.TestCase):
    def test_survey_ranks_by_recorded_download_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkout = _fake_checkout(Path(tmp))
            payload = runner.survey(checkout)

        self.assertEqual(payload["sized_competition_count"], 2)
        self.assertEqual(payload["smallest"]["competition_id"], "tiny-comp")
        self.assertEqual([row["competition_id"] for row in payload["ranking"]], ["tiny-comp", "huge-comp"])
        self.assertTrue(payload["sized_covers_eval_split"])

    def test_survey_reports_the_total_download_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = runner.survey(_fake_checkout(Path(tmp)))
        self.assertAlmostEqual(payload["total_dataset_size_gb"], 120.502, places=3)

    def test_survey_hashes_the_size_table_it_derived_the_pick_from(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = runner.survey(_fake_checkout(Path(tmp)))
        self.assertIsNotNone(payload["source"]["size_table_sha256"])
        self.assertIsNotNone(payload["source"]["eval_split_sha256"])

    def test_survey_on_the_pinned_checkout_finds_all_seventy_five_competitions(self) -> None:
        if not (_PINNED_CHECKOUT / "mlebench/registry.py").is_file():
            self.skipTest("pinned MLE-bench checkout is not present")
        payload = runner.survey(_PINNED_CHECKOUT)
        self.assertEqual(payload["eval_split_competition_count"], 75)
        self.assertEqual(payload["sized_competition_count"], 75)
        self.assertTrue(payload["sized_covers_eval_split"])
        self.assertEqual(payload["smallest"]["competition_id"], "spooky-author-identification")


class SplitManifestTests(unittest.TestCase):
    def test_split_manifest_hashes_every_split_and_its_task_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = runner.split_manifest(_fake_checkout(Path(tmp)))

        self.assertEqual(manifest["task_id_kind"], "competition_id")
        self.assertEqual(manifest["eval_split"]["count"], 2)
        self.assertEqual(manifest["eval_split"]["task_ids_sha256"], runner.task_ids_sha256(["tiny-comp", "huge-comp"]))
        self.assertEqual(manifest["complexity_splits"]["low"]["count"], 1)
        self.assertIsNone(manifest["complexity_splits"]["high"]["task_ids_sha256"])

    def test_split_manifest_does_not_claim_per_sample_task_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = runner.split_manifest(_fake_checkout(Path(tmp)))
        self.assertIsNone(manifest["per_sample_task_hashes"])


class VerifierIdentityTests(unittest.TestCase):
    def test_verifier_identity_pins_grader_leaderboard_and_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkout = _fake_checkout(Path(tmp))
            verifier = runner.verifier_identity(checkout, "tiny-comp")

        self.assertEqual(verifier["grader_name"], "log-loss")
        self.assertEqual(verifier["grade_fn"], "mlebench.competitions.tiny-comp.grade:grade")
        self.assertEqual(verifier["leaderboard_team_count"], 2)
        self.assertIsNotNone(verifier["expected_data_checksums_sha256"])
        self.assertTrue(verifier["resolved"])

    def test_unresolved_git_lfs_leaderboard_pointer_fails_the_verifier_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkout = _fake_checkout(Path(tmp))
            (checkout / "mlebench/competitions/tiny-comp/leaderboard.csv").write_text(
                "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 10\n", encoding="utf-8"
            )
            verifier = runner.verifier_identity(checkout, "tiny-comp")

        self.assertTrue(verifier["leaderboard_is_unresolved_lfs_pointer"])
        self.assertIsNone(verifier["leaderboard_team_count"])
        self.assertFalse(verifier["resolved"])


class DatasetStateTests(unittest.TestCase):
    def test_dataset_state_is_unprepared_when_the_answers_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = runner.dataset_state(Path(tmp), "tiny-comp")
        self.assertFalse(state["prepared"])
        self.assertIsNone(state["private_answers_sha256"])

    def test_dataset_state_is_prepared_and_hashed_when_both_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tiny-comp/prepared"
            (root / "private").mkdir(parents=True)
            (root / "public").mkdir(parents=True)
            (root / "private/test.csv").write_text("id,a\n1,1\n", encoding="utf-8")
            (root / "public/sample_submission.csv").write_text("id,a\n1,0\n", encoding="utf-8")
            state = runner.dataset_state(Path(tmp), "tiny-comp")

        self.assertTrue(state["prepared"])
        self.assertIsNotNone(state["private_answers_sha256"])


class KaggleRuleTests(unittest.TestCase):
    def test_rule_state_is_unknown_without_a_probe_and_carries_the_url(self) -> None:
        state = runner.kaggle_rule_acceptance("spooky-author-identification")
        self.assertIsNone(state["accepted"])
        self.assertEqual(state["rules_url"], "https://www.kaggle.com/c/spooky-author-identification/rules")

    def test_a_failed_probe_records_rules_as_not_accepted(self) -> None:
        state = runner.kaggle_rule_acceptance("x", {"rules_accepted": False, "error": "403 Forbidden"})
        self.assertFalse(state["accepted"])
        self.assertEqual(state["accepted_evidence"], "403 Forbidden")

    def test_check_rules_output_shape_is_accepted_as_a_probe(self) -> None:
        state = runner.kaggle_rule_acceptance(
            "x", {"accepted": True, "download_endpoint_ok": True, "metadata_endpoint_ok": True}
        )
        self.assertTrue(state["accepted"])
        self.assertEqual(state["verified_via"], "download endpoint (the only gated one)")

    def test_metadata_only_success_does_not_count_as_acceptance(self) -> None:
        state = runner.kaggle_rule_acceptance(
            "x", {"accepted": False, "download_endpoint_ok": False, "metadata_endpoint_ok": True}
        )
        self.assertFalse(state["accepted"])

    def test_rule_state_warns_that_the_metadata_endpoint_is_not_a_check(self) -> None:
        state = runner.kaggle_rule_acceptance("x")
        self.assertIn("competitions files", state["verification_warning"])
        self.assertIn("NOT a valid acceptance check", state["verification_warning"])

    def test_check_rules_reports_not_accepted_when_only_metadata_succeeds(self) -> None:
        """The exact false positive: metadata lists files, download is still 403."""

        class FakeApi:
            config_values = {"username": "someone"}

            def competition_list_files(self, competition):
                return ["train.zip"]

            def competition_download_files(self, competition, path, quiet, force):
                raise RuntimeError(
                    "(403) Forbidden: You must accept this competition's rules before "
                    "you'll be able to download files."
                )

        with patch.dict(sys.modules, {"mlebench.utils": type("M", (), {"authenticate_kaggle_api": lambda: FakeApi()})}):
            result = runner.check_rules_accepted("spooky-author-identification")

        self.assertTrue(result["metadata_endpoint_ok"])
        self.assertFalse(result["download_endpoint_ok"])
        self.assertFalse(result["accepted"])
        self.assertTrue(result["blocked_on_rules"])

    def test_check_rules_reports_accepted_only_when_the_download_succeeds(self) -> None:
        class FakeApi:
            config_values = {"username": "someone"}

            def competition_list_files(self, competition):
                return ["train.zip"]

            def competition_download_files(self, competition, path, quiet, force):
                return None

        with patch.dict(sys.modules, {"mlebench.utils": type("M", (), {"authenticate_kaggle_api": lambda: FakeApi()})}):
            result = runner.check_rules_accepted("spooky-author-identification")

        self.assertTrue(result["accepted"])
        self.assertTrue(result["download_endpoint_ok"])


class DockerDigestTests(unittest.TestCase):
    def test_a_missing_image_yields_no_digest_rather_than_an_exception(self) -> None:
        with patch.object(runner.subprocess, "run") as mocked:
            mocked.return_value = type("R", (), {"returncode": 1, "stdout": "", "stderr": "No such image"})()
            result = runner.docker_image_digest("mlebench-env")
        self.assertFalse(result["present"])
        self.assertIsNone(result["digest"])

    def test_a_present_image_reports_its_repo_digest(self) -> None:
        payload = json.dumps({"RepoDigests": ["mlebench-env@sha256:abc"], "Id": "sha256:local", "Architecture": "amd64", "Os": "linux"})
        with patch.object(runner.subprocess, "run") as mocked:
            mocked.return_value = type("R", (), {"returncode": 0, "stdout": payload, "stderr": ""})()
            result = runner.docker_image_digest("mlebench-env")
        self.assertTrue(result["present"])
        self.assertEqual(result["digest"], "mlebench-env@sha256:abc")
        self.assertEqual(result["digest_kind"], "repo_digest")

    def test_an_unavailable_docker_binary_fails_closed(self) -> None:
        with patch.object(runner.subprocess, "run", side_effect=OSError("no docker")):
            result = runner.docker_image_digest("mlebench-env")
        self.assertFalse(result["present"])
        self.assertIsNone(result["digest"])


class FailClosedReceiptTests(unittest.TestCase):
    def _receipt(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = _fake_checkout(Path(tmp) / "checkout")
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            return runner.build_receipt(
                checkout=checkout,
                competition_id=kwargs.pop("competition_id", None),
                data_dir=kwargs.pop("data_dir", data_dir),
                observed_at="2026-08-09",
                inspect_docker=False,
                **kwargs,
            )

    def test_receipt_defaults_to_the_smallest_competition(self) -> None:
        receipt = self._receipt()
        self.assertEqual(receipt["competition_binding"]["competition_id"], "tiny-comp")
        self.assertAlmostEqual(receipt["competition_binding"]["recorded_download_size_gb"], 0.002)

    def test_receipt_is_blocked_and_scoreless_when_gates_fail(self) -> None:
        receipt = self._receipt()
        self.assertEqual(receipt["status"], "BLOCKED")
        self.assertFalse(receipt["runnable_now"])
        self.assertIsNone(receipt["metrics"]["score"])
        self.assertFalse(receipt["metrics"]["is_model_score"])

    def test_blockers_name_every_failing_gate(self) -> None:
        receipt = self._receipt()
        for gate in (
            "dataset_license_accepted",
            "competition_data_prepared",
            "container_image_digest_present",
            "model_submission_artifact_present",
            "contamination_disjointness_receipt",
        ):
            self.assertIn(gate, receipt["blockers"])

    def test_revision_and_split_gates_pass_on_a_well_formed_checkout(self) -> None:
        receipt = self._receipt()
        self.assertTrue(receipt["gates"]["upstream_revision_pinned"])
        self.assertTrue(receipt["gates"]["split_manifest_resolved"])
        self.assertTrue(receipt["gates"]["verifier_identity_resolved"])

    def test_status_is_ready_only_when_every_required_gate_passes(self) -> None:
        gates = {name: True for name in runner.REQUIRED_GATES}
        self.assertEqual(runner.fail_closed_status(gates), "READY")
        for name in runner.REQUIRED_GATES:
            broken = dict(gates)
            broken[name] = False
            self.assertEqual(runner.fail_closed_status(broken), "BLOCKED", name)

    def test_a_missing_gate_key_is_treated_as_a_failure(self) -> None:
        self.assertEqual(runner.fail_closed_status({}), "BLOCKED")

    def test_harness_validation_never_promotes_itself_to_a_suite_score(self) -> None:
        harness = {"label": "harness_validation", "is_model_score": False, "status": "PASS", "suite_score": None}
        receipt = self._receipt(harness=harness)
        self.assertEqual(receipt["harness_validation"]["status"], "PASS")
        self.assertIsNone(receipt["metrics"]["score"])
        self.assertEqual(receipt["status"], "BLOCKED")

    def test_receipt_records_the_dataset_licence_position(self) -> None:
        receipt = self._receipt()
        position = receipt["authoritative_public_source"]["license_position"]
        self.assertEqual(position["repository_code"], "MIT")
        self.assertEqual(position["competition_datasets"], "not_covered_by_repository_license")
        self.assertFalse(position["acceptance_is_automatable"])

    def test_receipt_pins_the_upstream_revision(self) -> None:
        receipt = self._receipt()
        self.assertEqual(receipt["authoritative_public_source"]["revision"], runner.UPSTREAM_COMMIT)
        self.assertEqual(len(runner.UPSTREAM_COMMIT), 40)

    def test_verifier_only_variant_never_satisfies_the_container_gate(self) -> None:
        payload = json.dumps({"RepoDigests": [], "Id": "sha256:abc", "Architecture": "amd64", "Os": "linux"})
        with tempfile.TemporaryDirectory() as tmp:
            checkout = _fake_checkout(Path(tmp) / "checkout")
            with patch.object(runner, "docker_image_digest") as mocked:
                mocked.side_effect = lambda image: (
                    {"image": image, "present": False, "digest": None, "error": "no such image"}
                    if image == runner.REQUIRED_IMAGE
                    else json.loads(payload) | {"image": image, "present": True, "digest": "sha256:abc"}
                )
                receipt = runner.build_receipt(checkout=checkout, data_dir=Path(tmp), observed_at="2026-08-09")

        variant = receipt["environment"]["verifier_only_variant"]
        self.assertTrue(variant["present"])
        self.assertFalse(variant["satisfies_container_gate"])
        self.assertFalse(receipt["gates"]["container_image_digest_present"])
        self.assertEqual(receipt["status"], "BLOCKED")

    def test_lane_status_is_partial_when_only_the_harness_ran(self) -> None:
        gates = {name: False for name in runner.REQUIRED_GATES}
        self.assertEqual(runner.lane_status(gates, {"status": "PASS"}), "PARTIAL")
        self.assertEqual(runner.lane_status(gates, None), "BLOCKED")
        self.assertEqual(runner.lane_status({n: True for n in runner.REQUIRED_GATES}, None), "RUNNING")

    def test_every_failing_gate_names_the_receipt_that_would_clear_it(self) -> None:
        receipt = self._receipt()
        details = {entry["gate"]: entry for entry in receipt["blocker_details"]}
        self.assertEqual(set(details), set(receipt["blockers"]))
        for entry in details.values():
            self.assertTrue(entry["missing"])
            self.assertTrue(entry["external_receipt"])
            self.assertTrue(entry["action"])

    def test_licence_blocker_points_at_the_kaggle_rules_page(self) -> None:
        receipt = self._receipt()
        details = {entry["gate"]: entry for entry in receipt["blocker_details"]}
        self.assertIn("kaggle.com/c/tiny-comp/rules", details["dataset_license_accepted"]["action"])

    def test_receipt_declares_no_paid_side_effects(self) -> None:
        receipt = self._receipt()
        self.assertEqual(
            receipt["safety"],
            {"paid_calls": False, "tinker_launched": False, "wandb_published": False, "checkpoints_pushed": False},
        )


class HarnessFixtureTests(unittest.TestCase):
    def test_fixture_matches_the_schema_prepare_py_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = runner.build_spooky_fixture(Path(tmp), rows=25, seed=1)
            answers = Path(meta["answers_path"]).read_text(encoding="utf-8").splitlines()

        self.assertEqual(answers[0], "id,EAP,HPL,MWS")
        self.assertEqual(len(answers), 26)
        for row in answers[1:]:
            values = [int(v) for v in row.split(",")[1:]]
            self.assertEqual(sum(values), 1)

    def test_fixture_is_labelled_synthetic_not_official_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = runner.build_spooky_fixture(Path(tmp), rows=5)
        self.assertEqual(meta["provenance"], "synthetic_schema_conformant_fixture")

    def test_fixture_is_deterministic_for_a_given_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = runner.build_spooky_fixture(Path(tmp) / "a", rows=40, seed=7)["answers_sha256"]
            b = runner.build_spooky_fixture(Path(tmp) / "b", rows=40, seed=7)["answers_sha256"]
            c = runner.build_spooky_fixture(Path(tmp) / "c", rows=40, seed=8)["answers_sha256"]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_harness_validate_refuses_competitions_without_a_fixture_builder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.harness_validate(_PINNED_CHECKOUT, "aerial-cactus-identification", Path(tmp), Path(tmp))
        self.assertEqual(result["status"], "BLOCKED")
        self.assertFalse(result["is_model_score"])

    def test_harness_validate_result_is_never_a_model_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.harness_validate(_PINNED_CHECKOUT, "not-a-real-competition", Path(tmp), Path(tmp))
        self.assertEqual(result["label"], "harness_validation")
        self.assertFalse(result["is_model_score"])
        self.assertIsNone(result["suite_score"])


class UpstreamReproductionTests(unittest.TestCase):
    def test_reads_the_recorded_sample_submission_score(self) -> None:
        if not (_PINNED_CHECKOUT / "tests/constants.py").is_file():
            self.skipTest("pinned MLE-bench checkout is not present")
        self.assertAlmostEqual(
            runner.upstream_sample_submission_score(_PINNED_CHECKOUT, "spooky-author-identification"),
            1.08468,
        )

    def test_nan_entries_return_none_rather_than_a_bogus_float(self) -> None:
        if not (_PINNED_CHECKOUT / "tests/constants.py").is_file():
            self.skipTest("pinned MLE-bench checkout is not present")
        self.assertIsNone(
            runner.upstream_sample_submission_score(_PINNED_CHECKOUT, "us-patent-phrase-to-phrase-matching")
        )

    def test_unknown_competition_returns_none(self) -> None:
        if not (_PINNED_CHECKOUT / "tests/constants.py").is_file():
            self.skipTest("pinned MLE-bench checkout is not present")
        self.assertIsNone(runner.upstream_sample_submission_score(_PINNED_CHECKOUT, "not-a-competition"))

    def test_missing_constants_file_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(runner.upstream_sample_submission_score(Path(tmp), "anything"))


class InvalidControlTests(unittest.TestCase):
    def test_control_breaks_the_sum_to_one_constraint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "sample_submission.csv"
            source.write_text("id,EAP,HPL,MWS\nid1,0.4,0.3,0.3\n", encoding="utf-8")
            control = runner.build_invalid_control(source, Path(tmp) / "controls")
            rows = control.read_text(encoding="utf-8").splitlines()

        self.assertEqual(rows[0], "id,EAP,HPL,MWS")
        values = [float(v) for v in rows[1].split(",")[1:]]
        # Sum-to-one is the constraint this control reliably violates (2.5 != 1).
        # A [0, 1] range violation is not guaranteed for every source row, so it
        # is deliberately not asserted here.
        self.assertAlmostEqual(sum(values), 2.5)
        self.assertNotAlmostEqual(sum(values), 1.0)

    def test_control_is_written_outside_the_prepared_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prepared = Path(tmp) / "prepared/public"
            prepared.mkdir(parents=True)
            source = prepared / "sample_submission.csv"
            source.write_text("id,a,b\nid1,0.5,0.5\n", encoding="utf-8")
            before = source.read_text(encoding="utf-8")
            control = runner.build_invalid_control(source, Path(tmp) / "controls")

        self.assertNotIn(str(prepared), str(control))
        self.assertEqual(before, "id,a,b\nid1,0.5,0.5\n")

    def test_control_preserves_non_numeric_id_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "s.csv"
            source.write_text("id,a\nid00042,0.5\n", encoding="utf-8")
            control = runner.build_invalid_control(source, Path(tmp) / "c")
            self.assertTrue(control.read_text(encoding="utf-8").splitlines()[1].startswith("id00042,"))


class CliTests(unittest.TestCase):
    def test_survey_subcommand_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkout = _fake_checkout(Path(tmp) / "checkout")
            out = Path(tmp) / "survey.json"
            code = runner.main(["--checkout", str(checkout), "survey", "--out", str(out)])
            payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(payload["smallest"]["competition_id"], "tiny-comp")

    def test_receipt_subcommand_writes_a_blocked_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkout = _fake_checkout(Path(tmp) / "checkout")
            out = Path(tmp) / "receipt.json"
            code = runner.main(
                ["--checkout", str(checkout), "receipt", "--data-dir", tmp, "--no-docker", "--out", str(out)]
            )
            payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "BLOCKED")
        self.assertIsNone(payload["metrics"]["score"])


if __name__ == "__main__":
    unittest.main()
