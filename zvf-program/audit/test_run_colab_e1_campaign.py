import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "run_colab_e1_campaign", HERE / "run_colab_e1_campaign.py"
)
CAMPAIGN = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(CAMPAIGN)


class E1CampaignTests(unittest.TestCase):
    def setUp(self):
        self.expected = {
            "heldout_n": 500,
            "checkpoint_steps": [5, 10, 15, 20, 25, 30],
            "treatment_changes": {"grpo": []},
        }
        self.good = {
            "arm": "grpo",
            "seed": 11,
            "evidence_class": "confirmatory",
            "heldout_n": 500,
            "treatment_changes": [],
            "fingerprint": "u" * 64,
            "stack_fingerprint": "s" * 64,
            "heldout_score": 0.5,
            "manifest_path": "manifests/grpo-seed-11.json",
            "remote_verification": {
                "hf_checkpoint_steps": [5, 10, 15, 20, 25, 30],
                "hf_repo": "owner/private-repo",
                "hf_commit": "commit",
                "wandb": {"state": "finished"},
            },
        }
        self.good_manifest = {
            "heldout_trace": [
                {
                    "index": index,
                    "correct": index % 2 == 0,
                    "completion_sha256": f"{index:064x}",
                }
                for index in range(500)
            ]
        }

    def test_strict_result_validation_requires_remote_reconciliation(self):
        with mock.patch.object(CAMPAIGN, "read_json", return_value=self.good_manifest):
            valid, reason = CAMPAIGN.validate_result(
                ("grpo", 11), self.good, self.expected, stack_fingerprint="s" * 64
            )
        self.assertTrue(valid, reason)
        broken = {**self.good, "remote_verification": {}}
        with mock.patch.object(CAMPAIGN, "read_json", return_value=self.good_manifest):
            valid, reason = CAMPAIGN.validate_result(
                ("grpo", 11), broken, self.expected, stack_fingerprint="s" * 64
            )
        self.assertFalse(valid)
        self.assertIn("wandb_finished", reason)
        self.assertIn("hf_checkpoints", reason)

    def test_result_validation_rejects_unhashed_or_duplicate_evidence(self):
        unhashed = {
            "heldout_trace": [
                {"index": index, "correct": index % 2 == 0}
                for index in range(500)
            ]
        }
        with mock.patch.object(CAMPAIGN, "read_json", return_value=unhashed):
            valid, reason = CAMPAIGN.validate_result(
                ("grpo", 11), self.good, self.expected, stack_fingerprint="s" * 64
            )
        self.assertFalse(valid)
        self.assertIn("heldout_hashes", reason)

        duplicate = {"heldout_trace": [dict(row) for row in self.good_manifest["heldout_trace"]]}
        duplicate["heldout_trace"][1]["completion_sha256"] = duplicate["heldout_trace"][0][
            "completion_sha256"
        ]
        with mock.patch.object(CAMPAIGN, "read_json", return_value=duplicate):
            valid, reason = CAMPAIGN.validate_result(
                ("grpo", 11), self.good, self.expected, stack_fingerprint="s" * 64
            )
        self.assertFalse(valid)
        self.assertIn("unique_completion_hashes", reason)

    def test_process_parser_only_accepts_confirmatory_unit_runner(self):
        command = (
            "python3 /repo/zvf-program/audit/run_colab_e1_confirmatory.py "
            "--mode confirmatory --arm dapo --seed 53"
        )
        self.assertEqual(CAMPAIGN.parse_process_unit(command), ("dapo", 53))
        self.assertEqual(
            CAMPAIGN.parse_process_unit(
                "python3 run_colab_e1_confirmatory.py --mode confirmatory --seed 37"
            ),
            ("grpo", 37),
        )
        self.assertIsNone(
            CAMPAIGN.parse_process_unit(command.replace("confirmatory", "preflight"))
        )
        self.assertIsNone(CAMPAIGN.parse_process_unit("python3 unrelated.py"))
        self.assertIsNone(
            CAMPAIGN.parse_process_unit(
                "tmux new-session -d -s e1-campaign "
                "python3 run_colab_e1_confirmatory.py "
                "--mode confirmatory --seed 53"
            )
        )

    def test_process_parser_counts_evaluation_recovery_as_an_active_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "request.json"
            source.write_text('{"arm":"gspo","seed":71}\n')
            command = (
                f"python3 {CAMPAIGN.EVALUATION_RUNNER} "
                f"--source-request {source} --eval-batch-size 8"
            )
            self.assertEqual(CAMPAIGN.parse_process_unit(command), ("gspo", 71))

    def test_recovery_uses_verified_step30_even_after_assignment_failure(self):
        unit = ("grpo", 89)
        result_path = CAMPAIGN.UNIT_RESULTS / "e1__grpo__s89__confirmatory.json"
        source_path = Path("/tmp/source-request.json")
        result = {"status": "failed", "failed_step": 6, "request_path": str(source_path)}
        source = {
            "mode": "confirmatory",
            "arm": "grpo",
            "seed": 89,
            "hf_repo": "owner/repo",
        }
        with mock.patch.object(CAMPAIGN, "read_json") as read:
            read.side_effect = lambda path: result if path == result_path else source
            self.assertEqual(
                CAMPAIGN.recovery_source_request(
                    unit, checkpoint_exists=lambda repo: repo == "owner/repo"
                ),
                source_path,
            )

        completed_result = {**result, "status": "completed", "failed_step": None}
        with mock.patch.object(CAMPAIGN, "read_json") as read:
            read.side_effect = (
                lambda path: completed_result if path == result_path else source
            )
            self.assertEqual(
                CAMPAIGN.recovery_source_request(
                    unit, checkpoint_exists=lambda repo: repo == "owner/repo"
                ),
                source_path,
            )
            self.assertIsNone(
                CAMPAIGN.recovery_source_request(
                    unit, checkpoint_exists=lambda _repo: False
                )
            )

        assignment_failure = {**result, "failed_step": 0}
        with mock.patch.object(CAMPAIGN, "read_json") as read:
            read.side_effect = (
                lambda path: assignment_failure if path == result_path else source
            )
            self.assertEqual(
                CAMPAIGN.recovery_source_request(
                    unit, checkpoint_exists=lambda repo: repo == "owner/repo"
                ),
                source_path,
            )

    def test_recovery_follows_evaluation_wrapper_to_confirmatory_request(self):
        unit = ("grpo", 89)
        result_path = CAMPAIGN.UNIT_RESULTS / "e1__grpo__s89__confirmatory.json"
        wrapper_path = Path("/tmp/evaluation-request.json")
        source_path = Path("/tmp/source-request.json")
        records = {
            result_path: {
                "status": "failed",
                "failed_step": 0,
                "request_path": str(wrapper_path),
            },
            wrapper_path: {"source_request": str(source_path)},
            source_path: {
                "mode": "confirmatory",
                "arm": "grpo",
                "seed": 89,
                "fingerprint": "f" * 64,
                "hf_repo": "owner/repo",
            },
        }
        with mock.patch.object(CAMPAIGN, "read_json", side_effect=records.get):
            self.assertEqual(
                CAMPAIGN.recovery_source_request(
                    unit, checkpoint_exists=lambda repo: repo == "owner/repo"
                ),
                source_path,
            )
            self.assertEqual(
                CAMPAIGN.failed_source_request(
                    unit, checkpoint_exists=lambda repo: repo == "owner/repo"
                ),
                source_path,
            )

    def test_preassignment_failure_is_retryable_without_consuming_unit_budget(self):
        self.assertTrue(
            CAMPAIGN.is_preassignment_failure(
                {"status": "failed", "failed_step": 0}
            )
        )
        self.assertFalse(
            CAMPAIGN.is_preassignment_failure(
                {"status": "failed", "failed_step": 1}
            )
        )
        self.assertFalse(
            CAMPAIGN.is_preassignment_failure(
                {"status": "completed", "failed_step": 0}
            )
        )

    def test_transient_provider_failure_is_retryable_without_consuming_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "unit.log"
            log.write_text(
                "Traceback\n"
                "huggingface_hub.errors.HfHubHTTPError: 504 Server Error\n"
            )
            record = {
                "status": "failed",
                "failed_step": 7,
                "error": "colab CLI returned non-zero",
                "log_path": str(log),
            }
            self.assertTrue(CAMPAIGN.is_transient_provider_failure(record))

            log.write_text("RuntimeError: Connection was lost.\n")
            self.assertTrue(CAMPAIGN.is_transient_provider_failure(record))

            log.write_text(
                "Fatal Python error: init_sys_streams: can't initialize sys "
                "standard streams\n"
                "OSError: [Errno 9] Bad file descriptor\n"
            )
            self.assertTrue(CAMPAIGN.is_transient_provider_failure(record))

            log.write_text(
                "Bad request for commit endpoint:\n"
                "Private repository storage limit reached, please upgrade "
                "your plan to increase your private storage limit\n"
            )
            self.assertTrue(CAMPAIGN.is_transient_provider_failure(record))

            log.write_text("new retry replaced the per-run log\n")
            campaign_log = Path(tmp) / "campaign.log"
            campaign_log.write_text(
                "Private repository storage limit reached, please upgrade "
                "your plan to increase your private storage limit\n"
            )
            self.assertTrue(
                CAMPAIGN.is_transient_provider_failure(
                    record, extra_log_path=campaign_log
                )
            )

            campaign_log.write_text(
                "[old] campaign attempt 1 kind=evaluation-recovery\n"
                "HfHubHTTPError: 504 Server Error\n"
                "[new] campaign attempt 2 kind=evaluation-recovery\n"
                "RuntimeError: invalid recovered rollout count for aero: 436.0\n"
            )
            self.assertFalse(
                CAMPAIGN.is_transient_provider_failure(
                    record, extra_log_path=campaign_log
                )
            )

            log.write_text(
                "HTTP Error 504 thrown while requesting HEAD\n"
                "RuntimeError: CUDA out of memory\n"
            )
            self.assertFalse(CAMPAIGN.is_transient_provider_failure(record))

        self.assertFalse(CAMPAIGN.is_transient_provider_failure(None))
        self.assertFalse(
            CAMPAIGN.is_transient_provider_failure(
                {"status": "completed", "error": "504 Server Error"}
            )
        )

    def test_failure_credit_identity_changes_only_for_a_new_failed_result(self):
        unit = ("grpo", 89)
        record = {
            "status": "failed",
            "completed_at": "2026-07-16T08:38:19+00:00",
            "failed_step": 7,
            "request_path": "/tmp/request.json",
        }
        credit_id = CAMPAIGN.failure_credit_id(unit, record)
        self.assertEqual(credit_id, CAMPAIGN.failure_credit_id(unit, dict(record)))
        self.assertNotEqual(
            credit_id,
            CAMPAIGN.failure_credit_id(
                unit, {**record, "completed_at": "2026-07-16T09:38:19+00:00"}
            ),
        )
        self.assertIsNone(
            CAMPAIGN.failure_credit_id(unit, {**record, "status": "completed"})
        )

    def test_active_backoff_uses_latest_future_deadline(self):
        self.assertEqual(
            CAMPAIGN.active_backoff_until(
                [900.0, None, 1200.0, 1001.0], now_epoch=1000.0
            ),
            1200.0,
        )
        self.assertIsNone(
            CAMPAIGN.active_backoff_until(
                [900.0, None, 1000.0], now_epoch=1000.0
            )
        )

    def test_aggregate_refresh_is_allow_incomplete_and_requires_json_output(self):
        completed = mock.Mock(returncode=0, stdout='{"status":"INCOMPLETE"}\n')
        report = {
            "status": "INCOMPLETE",
            "validated_units": 5,
            "required_units": 40,
            "errors": [],
            "missing_units": [{}] * 35,
            "verdicts_emitted": False,
        }
        with (
            mock.patch.object(CAMPAIGN.subprocess, "run", return_value=completed) as run,
            mock.patch.object(CAMPAIGN, "read_json", return_value=report),
        ):
            CAMPAIGN.regenerate_aggregate(validated_units=5, required_units=40)
        command = run.call_args.args[0]
        self.assertIn("--allow-incomplete", command)
        self.assertEqual(command[command.index("--output") + 1], str(CAMPAIGN.AGGREGATE_OUTPUT))

        with (
            mock.patch.object(CAMPAIGN.subprocess, "run", return_value=completed),
            mock.patch.object(CAMPAIGN, "read_json", return_value=None),
            self.assertRaisesRegex(RuntimeError, "valid JSON"),
        ):
            CAMPAIGN.regenerate_aggregate(validated_units=5, required_units=40)

        with (
            mock.patch.object(CAMPAIGN.subprocess, "run", return_value=completed),
            mock.patch.object(
                CAMPAIGN, "read_json", return_value={**report, "validated_units": 4}
            ),
            self.assertRaisesRegex(RuntimeError, "disagrees with campaign"),
        ):
            CAMPAIGN.regenerate_aggregate(validated_units=5, required_units=40)

    def test_partial_checkpoint_preserves_original_source_request(self):
        unit = ("grpo", 71)
        result_path = CAMPAIGN.UNIT_RESULTS / "e1__grpo__s71__confirmatory.json"
        source_path = Path("/tmp/original-request.json")
        result = {"status": "failed", "failed_step": 6, "request_path": str(source_path)}
        source = {
            "mode": "confirmatory",
            "arm": "grpo",
            "seed": 71,
            "fingerprint": "f" * 64,
            "hf_repo": "owner/repo",
        }
        with mock.patch.object(CAMPAIGN, "read_json") as read:
            read.side_effect = lambda path: result if path == result_path else source
            self.assertEqual(
                CAMPAIGN.failed_source_request(
                    unit, checkpoint_exists=lambda repo: repo == "owner/repo"
                ),
                source_path,
            )

        too_early = {**result, "failed_step": 0}
        with mock.patch.object(CAMPAIGN, "read_json") as read:
            read.side_effect = lambda path: too_early if path == result_path else source
            self.assertIsNone(
                CAMPAIGN.failed_source_request(
                    unit, checkpoint_exists=lambda _repo: False
                )
            )

    def test_recovery_plan_prefers_evaluation_after_verified_step30(self):
        unit = ("gspo", 11)
        result_path = CAMPAIGN.UNIT_RESULTS / "e1__gspo__s11__confirmatory.json"
        source_path = Path("/tmp/gspo-source-request.json")
        records = {
            result_path: {
                "status": "failed",
                "failed_step": 6,
                "request_path": str(source_path),
            },
            source_path: {
                "mode": "confirmatory",
                "arm": "gspo",
                "seed": 11,
                "fingerprint": "f" * 64,
                "hf_repo": "owner/repo",
            },
        }
        with mock.patch.object(CAMPAIGN, "read_json", side_effect=records.get):
            self.assertEqual(
                CAMPAIGN.recovery_plan(
                    unit,
                    checkpoint5_exists=lambda _repo: True,
                    checkpoint30_exists=lambda _repo: True,
                ),
                ("evaluation-recovery", source_path),
            )
            self.assertEqual(
                CAMPAIGN.recovery_plan(
                    unit,
                    checkpoint5_exists=lambda _repo: True,
                    checkpoint30_exists=lambda _repo: False,
                ),
                ("exact-source-checkpoint-resume", source_path),
            )

        completed_records = {
            result_path: {
                "status": "completed",
                "request_path": str(source_path),
            },
            source_path: records[source_path],
        }
        with mock.patch.object(CAMPAIGN, "read_json", side_effect=completed_records.get):
            self.assertEqual(
                CAMPAIGN.recovery_plan(
                    unit,
                    checkpoint5_exists=lambda _repo: True,
                    checkpoint30_exists=lambda _repo: True,
                ),
                ("evaluation-recovery", source_path),
            )

    def test_verified_recovery_bypasses_consumed_attempt_budget(self):
        unit = ("dapo", 131)
        self.assertEqual(
            CAMPAIGN.choose_launches(
                [unit],
                completed=set(),
                active=set(),
                attempts={"dapo:131": 3},
                capacity=1,
                max_attempts=3,
                recovery_kinds={unit: "evaluation-recovery"},
            ),
            [unit],
        )

    def test_active_process_scan_excludes_zombie_wrappers(self):
        stdout = (
            "101 Z python3 run_colab_e1_confirmatory.py --mode confirmatory --seed 37\n"
            "102 Ss python3 run_colab_e1_confirmatory.py --mode confirmatory --seed 53\n"
        )
        completed = mock.Mock(stdout=stdout)
        with mock.patch.object(CAMPAIGN.subprocess, "run", return_value=completed):
            self.assertEqual(CAMPAIGN.active_runner_processes(), {102: ("grpo", 53)})

    def test_remote_colab_assignment_count_uses_server_session_rows(self):
        output = (
            "[?] endpoint-1 | Hardware: CPU | Variant: DEFAULT\n"
            "[?] endpoint-2 | Hardware: A100 | Variant: GPU\n"
        )
        self.assertEqual(CAMPAIGN.parse_colab_assignment_count(output), 2)
        self.assertEqual(
            CAMPAIGN.parse_colab_assignment_count(
                "[colab] No active sessions found on server.\n"
            ),
            0,
        )
        completed = mock.Mock(returncode=0, stdout=output)
        with mock.patch.object(CAMPAIGN.subprocess, "run", return_value=completed):
            self.assertEqual(CAMPAIGN.remote_colab_assignment_count(), 2)

    def test_colab_session_parser_preserves_named_and_unnamed_rows(self):
        output = (
            "[e1-resume-dapo-s23-c0d539] endpoint-1 | Hardware: A100 | Variant: GPU\n"
            "[?] endpoint-2 | Hardware: A100 | Variant: GPU\n"
        )
        self.assertEqual(
            CAMPAIGN.parse_colab_session_names(output),
            ({"e1-resume-dapo-s23-c0d539"}, True),
        )

    def test_active_transport_parser_handles_python_cli_launcher(self):
        output = (
            "202 101 S /venv/bin/python /Users/me/.local/bin/colab --auth=oauth2 "
            "exec --session e1-resume-dapo-s23-c0d539 --file secure.py\n"
            "203 102 S /venv/bin/python /Users/me/.local/bin/colab exec "
            "--session=e1-con-dapo-s89-c0e809 --file secure.py\n"
            "204 103 S /venv/bin/python unrelated.py\n"
        )
        completed = mock.Mock(stdout=output)
        with mock.patch.object(CAMPAIGN.subprocess, "run", return_value=completed):
            self.assertEqual(
                CAMPAIGN.active_colab_exec_transports({101, 102, 103}),
                {
                    101: (202, "e1-resume-dapo-s23-c0d539"),
                    102: (203, "e1-con-dapo-s89-c0e809"),
                },
            )

    def test_stale_transport_requires_three_named_absence_polls(self):
        active = {101: ("dapo", 23)}
        missing: dict[int, int] = {}
        with (
            mock.patch.object(
                CAMPAIGN,
                "active_colab_exec_transports",
                return_value={101: (202, "e1-resume-dapo-s23-c0d539")},
            ),
            mock.patch.object(CAMPAIGN.os, "kill") as kill,
        ):
            self.assertEqual(
                CAMPAIGN.reap_missing_remote_transports(
                    active,
                    remote_names=set(),
                    has_unnamed_remote=False,
                    missing_polls=missing,
                ),
                [],
            )
            self.assertEqual(
                CAMPAIGN.reap_missing_remote_transports(
                    active,
                    remote_names=set(),
                    has_unnamed_remote=False,
                    missing_polls=missing,
                ),
                [],
            )
            reaped = CAMPAIGN.reap_missing_remote_transports(
                active,
                remote_names=set(),
                has_unnamed_remote=False,
                missing_polls=missing,
            )
        self.assertEqual(reaped, [(101, 202, "e1-resume-dapo-s23-c0d539")])
        kill.assert_called_once_with(202, CAMPAIGN.signal.SIGTERM)

    def test_stale_transport_counter_resets_for_live_or_unnamed_session(self):
        active = {101: ("dapo", 23)}
        missing = {101: 2}
        with mock.patch.object(
            CAMPAIGN,
            "active_colab_exec_transports",
            return_value={101: (202, "e1-resume-dapo-s23-c0d539")},
        ):
            CAMPAIGN.reap_missing_remote_transports(
                active,
                remote_names={"e1-resume-dapo-s23-c0d539"},
                has_unnamed_remote=False,
                missing_polls=missing,
            )
            self.assertEqual(missing, {})
            missing[101] = 2
            CAMPAIGN.reap_missing_remote_transports(
                active,
                remote_names=set(),
                has_unnamed_remote=True,
                missing_polls=missing,
            )
        self.assertEqual(missing, {})

    def test_hf_launch_preflight_requires_authenticated_responsive_api(self):
        api = mock.Mock()
        CAMPAIGN.verify_hf_launch_ready(api)
        api.whoami.assert_called_once_with()
        with self.assertRaisesRegex(RuntimeError, "Hugging Face login"):
            CAMPAIGN.verify_hf_launch_ready(None)

    def test_latest_assignment_failure_drives_global_backoff(self):
        units = [("grpo", 89), ("grpo", 107), ("grpo", 131)]
        records = {
            CAMPAIGN.UNIT_RESULTS / "e1__grpo__s89__confirmatory.json": {
                "status": "failed",
                "failed_step": 0,
                "completed_at": "2026-07-15T15:20:00+00:00",
            },
            CAMPAIGN.UNIT_RESULTS / "e1__grpo__s107__confirmatory.json": {
                "status": "failed",
                "failed_step": 6,
                "completed_at": "2026-07-15T15:30:00+00:00",
            },
            CAMPAIGN.UNIT_RESULTS / "e1__grpo__s131__confirmatory.json": {
                "status": "failed",
                "failed_step": 0,
                "completed_at": "2026-07-15T15:25:00Z",
            },
        }
        with mock.patch.object(CAMPAIGN, "read_json", side_effect=records.get):
            actual = CAMPAIGN.latest_assignment_failure_epoch(units)
        expected = CAMPAIGN.datetime.fromisoformat(
            "2026-07-15T15:25:00+00:00"
        ).timestamp()
        self.assertEqual(actual, expected)

    def test_launch_selection_avoids_completed_active_and_exhausted_units(self):
        units = [("grpo", 11), ("grpo", 23), ("dapo", 11), ("dapo", 23)]
        launches = CAMPAIGN.choose_launches(
            units,
            completed={("grpo", 11)},
            active={("grpo", 23)},
            attempts={"dapo:11": 3},
            capacity=2,
            max_attempts=3,
        )
        self.assertEqual(launches, [("dapo", 23)])
        self.assertEqual(
            CAMPAIGN.choose_launches(
                units,
                completed=set(),
                active=set(),
                attempts={},
                capacity=0,
                max_attempts=3,
            ),
            [],
        )

    def test_launch_selection_honors_retry_backoff(self):
        units = [("grpo", 89), ("grpo", 107)]
        launches = CAMPAIGN.choose_launches(
            units,
            completed=set(),
            active=set(),
            attempts={"grpo:89": 1, "grpo:107": 1},
            capacity=2,
            max_attempts=3,
            last_attempt_epoch={"grpo:89": 950.0, "grpo:107": 0.0},
            now_epoch=1000.0,
            retry_backoff_seconds=900,
        )
        self.assertEqual(launches, [("grpo", 107)])

    def test_launch_selection_rotates_past_credited_provider_failure(self):
        units = [("dapo", 37), ("dapo", 71), ("dapo", 89)]
        launches = CAMPAIGN.choose_launches(
            units,
            completed=set(),
            active=set(),
            attempts={"dapo:37": 0, "dapo:71": 0, "dapo:89": 0},
            capacity=2,
            max_attempts=3,
            last_attempt_epoch={"dapo:37": 950.0, "dapo:71": 800.0},
            now_epoch=2000.0,
            retry_backoff_seconds=900,
        )
        self.assertEqual(launches, [("dapo", 89), ("dapo", 71)])

    def test_launch_selection_prioritizes_hub_proven_recovery(self):
        units = [("gspo", 11), ("gspo", 23), ("dapo", 107), ("dapo", 131)]
        launches = CAMPAIGN.choose_launches(
            units,
            completed=set(),
            active=set(),
            attempts={"gspo:11": 1, "gspo:23": 0, "dapo:107": 1, "dapo:131": 1},
            capacity=3,
            max_attempts=3,
            last_attempt_epoch={
                "gspo:11": 900.0,
                "gspo:23": 950.0,
                "dapo:107": 800.0,
                "dapo:131": 850.0,
            },
            now_epoch=2000.0,
            retry_backoff_seconds=0,
            recovery_kinds={
                ("gspo", 11): "evaluation-recovery",
                ("dapo", 107): "exact-source-checkpoint-resume",
                ("dapo", 131): "exact-source-checkpoint-resume",
            },
        )
        self.assertEqual(
            launches,
            [("gspo", 11), ("dapo", 107), ("dapo", 131)],
        )

    def test_retry_limit_does_not_exhaust_a_newly_active_final_attempt(self):
        units = [("grpo", 89), ("grpo", 107)]
        self.assertEqual(
            CAMPAIGN.exhausted_units(
                units,
                completed=set(),
                active={("grpo", 89)},
                attempts={"grpo:89": 3, "grpo:107": 3},
                max_attempts=3,
            ),
            [("grpo", 107)],
        )

    def test_frozen_contract_expands_to_forty_units(self):
        units, expected = CAMPAIGN.load_contract()
        self.assertEqual(len(units), 40)
        self.assertEqual(len(set(units)), 40)
        self.assertEqual(expected["heldout_n"], 500)


if __name__ == "__main__":
    unittest.main()
