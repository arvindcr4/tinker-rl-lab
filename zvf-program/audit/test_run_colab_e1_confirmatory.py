import importlib.util
import copy
import io
from pathlib import Path
import sys
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "run_colab_e1_confirmatory", HERE / "run_colab_e1_confirmatory.py"
)
CONFIRMATORY = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(CONFIRMATORY)

REMOTE_DIR = HERE.parent / "colab-experiments"
sys.path.insert(0, str(REMOTE_DIR))
evaluator_spec = importlib.util.spec_from_file_location(
    "e1_evaluate_checkpoint", REMOTE_DIR / "e1_evaluate_checkpoint.py"
)
EVALUATOR = importlib.util.module_from_spec(evaluator_spec)
assert evaluator_spec.loader is not None
evaluator_spec.loader.exec_module(EVALUATOR)


class RemoteManifestValidationTests(unittest.TestCase):
    def setUp(self):
        self.fingerprint = "f" * 64
        self.result = {
            "audit_record": {
                "heldout_n": 2,
                "heldout_score": 0.5,
                "stack_fingerprint": "s" * 64,
            },
            "remote": {"wandb_run_id": "run123"},
        }
        self.manifest = {
            "schema_version": "e1-colab-confirmatory-run-v1",
            "evidence_class": "confirmatory",
            "run_config": {
                "unit_fingerprint": self.fingerprint,
                "stack_fingerprint": "s" * 64,
            },
            "audit_record": self.result["audit_record"],
            "remote_checkpoint_steps": [5, 10],
            "heldout_trace": [
                {"index": 0, "correct": True, "completion_sha256": "a" * 64},
                {"index": 1, "correct": False, "completion_sha256": "b" * 64},
            ],
            "wandb": {"run_id": "run123"},
        }

    def validate(self, manifest):
        return CONFIRMATORY.validate_remote_manifest(
            manifest,
            self.result,
            self.fingerprint,
            [5, 10],
            "confirmatory",
        )

    def test_complete_unique_hashed_manifest_is_accepted(self):
        self.assertIsNone(self.validate(self.manifest))

    def test_missing_hash_is_rejected(self):
        broken = copy.deepcopy(self.manifest)
        broken["heldout_trace"][1].pop("completion_sha256")
        with self.assertRaisesRegex(RuntimeError, "lacks completion hashes"):
            self.validate(broken)

    def test_duplicate_hash_is_rejected(self):
        broken = copy.deepcopy(self.manifest)
        broken["heldout_trace"][1]["completion_sha256"] = "a" * 64
        with self.assertRaisesRegex(RuntimeError, "not unique"):
            self.validate(broken)


class RecoveryRolloutTests(unittest.TestCase):
    def test_fixed_sampling_arms_recover_frozen_rollout_count(self):
        for arm in ("grpo", "gspo", "drgrpo"):
            with self.subTest(arm=arm):
                self.assertEqual(EVALUATOR.recovered_rollout_count([], arm), 480)

    def test_dapo_recovery_uses_checkpoint_cumulative_rollouts(self):
        history = [
            {"dapo/rollouts_cumulative": 480.0},
            {"train/dapo/rollouts_cumulative": 1728.0},
        ]
        self.assertEqual(EVALUATOR.recovered_rollout_count(history, "dapo"), 1728)

    def test_aero_recovery_uses_checkpoint_cumulative_rollouts(self):
        history = [{"aero/rollouts_cumulative": 436.0}]
        self.assertEqual(EVALUATOR.recovered_rollout_count(history, "aero"), 436)

    def test_aero_recovery_rejects_counts_outside_frozen_generation_bounds(self):
        for count in (359.0, 481.0):
            with self.subTest(count=count), self.assertRaisesRegex(
                RuntimeError, "invalid recovered rollout count"
            ):
                EVALUATOR.recovered_rollout_count(
                    [{"aero/rollouts_cumulative": count}], "aero"
                )

    def test_dynamic_recovery_fails_closed_without_rollout_telemetry(self):
        with self.assertRaisesRegex(RuntimeError, "lacks required"):
            EVALUATOR.recovered_rollout_count([], "dapo")

    def test_complete_hashed_progress_is_not_rewound(self):
        progress = {
            "next_index": 2,
            "correct": 1,
            "updated_at": "before",
            "trace": [
                {"index": 0, "correct": True, "completion_sha256": "a" * 64},
                {"index": 1, "correct": False, "completion_sha256": "b" * 64},
            ],
        }
        self.assertIsNone(EVALUATOR.rewind_unhashed_suffix(progress))
        self.assertEqual(progress["next_index"], 2)
        self.assertEqual(progress["correct"], 1)

    def test_unhashed_recovery_suffix_is_rewound_for_exact_replay(self):
        progress = {
            "next_index": 4,
            "correct": 3,
            "updated_at": "before",
            "trace": [
                {"index": 0, "correct": True, "completion_sha256": "a" * 64},
                {"index": 1, "correct": False, "completion_sha256": "b" * 64},
                {"index": 2, "correct": True},
                {"index": 3, "correct": True},
            ],
        }
        self.assertEqual(EVALUATOR.rewind_unhashed_suffix(progress), 2)
        self.assertEqual(progress["next_index"], 2)
        self.assertEqual(progress["correct"], 1)
        self.assertEqual([row["index"] for row in progress["trace"]], [0, 1])
        self.assertNotEqual(progress["updated_at"], "before")
        self.assertEqual(
            progress["rewind_receipt"],
            {
                "reason": "missing_completion_sha256",
                "from_next_index": 4,
                "to_next_index": 2,
                "discarded_rows": 2,
                "repaired_at": progress["updated_at"],
            },
        )


class TransientExecRetryTests(unittest.TestCase):
    def test_connection_loss_is_transient(self):
        self.assertTrue(
            CONFIRMATORY.is_transient_remote_failure(
                ["RuntimeError: Connection was lost.\n"]
            )
        )

    def test_local_watchdog_mirrors_colab_timeout_with_grace(self):
        self.assertEqual(
            CONFIRMATORY.local_command_timeout_seconds(
                ["colab", "exec", "--timeout", "120"]
            ),
            150,
        )
        self.assertEqual(
            CONFIRMATORY.local_command_timeout_seconds(
                ["colab", "exec", "--timeout=21600"]
            ),
            21630,
        )
        self.assertIsNone(
            CONFIRMATORY.local_command_timeout_seconds(["colab", "sessions"])
        )

    def test_transient_hub_gateway_failure_retries_in_same_session(self):
        lines: list[str] = []
        log = io.StringIO()

        def fake_run(_command, _log, output):
            if not output:
                output.append("HTTP Error 504 thrown while requesting HEAD\n")
                return 1
            output.append("E1_RESULT {}\n")
            return 0

        with (
            mock.patch.object(CONFIRMATORY, "run_logged", side_effect=fake_run) as run,
            mock.patch.object(CONFIRMATORY.time, "sleep") as sleep,
        ):
            code = CONFIRMATORY.run_logged_with_transient_retries(
                ["colab", "exec"], log, lines, attempts=3, retry_seconds=60
            )

        self.assertEqual(code, 0)
        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once_with(60)
        self.assertIn("preserving session", log.getvalue())

    def test_non_transient_remote_failure_is_not_retried(self):
        lines: list[str] = []
        log = io.StringIO()

        def fake_run(_command, _log, output):
            output.append("RuntimeError: CUDA out of memory\n")
            return 1

        with mock.patch.object(CONFIRMATORY, "run_logged", side_effect=fake_run) as run:
            code = CONFIRMATORY.run_logged_with_transient_retries(
                ["colab", "exec"], log, lines, attempts=3, retry_seconds=0
            )

        self.assertEqual(code, 1)
        self.assertEqual(run.call_count, 1)

    def test_retry_policy_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            CONFIRMATORY.run_logged_with_transient_retries(
                ["colab", "exec"], io.StringIO(), [], attempts=0, retry_seconds=60
            )
        with self.assertRaises(ValueError):
            CONFIRMATORY.run_logged_with_transient_retries(
                ["colab", "exec"], io.StringIO(), [], attempts=1, retry_seconds=-1
            )


class SessionCleanupTests(unittest.TestCase):
    def test_stop_session_retries_cli_stream_failure(self):
        command = ["colab", "--auth=oauth2", "stop", "--session", "e1-test"]
        failed = CONFIRMATORY.subprocess.CompletedProcess(
            command,
            1,
            stdout=(
                "Fatal Python error: init_sys_streams: can't initialize "
                "sys standard streams\nOSError: [Errno 9] Bad file descriptor\n"
            ),
        )
        succeeded = CONFIRMATORY.subprocess.CompletedProcess(
            command, 0, stdout="[colab] Session terminated.\n"
        )
        log = io.StringIO()

        with (
            mock.patch.object(
                CONFIRMATORY.subprocess,
                "run",
                side_effect=[failed, succeeded],
            ) as run,
            mock.patch.object(CONFIRMATORY.time, "sleep") as sleep,
        ):
            released = CONFIRMATORY.stop_session("oauth2", "e1-test", log)

        self.assertTrue(released)
        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once_with(2)
        self.assertIn("attempt=1/3 return_code=1", log.getvalue())
        self.assertIn("attempt=2/3 return_code=0", log.getvalue())

    def test_stop_session_warns_after_retry_exhaustion(self):
        failed = CONFIRMATORY.subprocess.CompletedProcess(
            ["colab", "stop"], 1, stdout="transport failed\n"
        )
        log = io.StringIO()

        with (
            mock.patch.object(
                CONFIRMATORY.subprocess, "run", return_value=failed
            ) as run,
            mock.patch.object(CONFIRMATORY.time, "sleep"),
        ):
            released = CONFIRMATORY.stop_session(
                "oauth2", "e1-test", log, attempts=2, retry_seconds=0
            )

        self.assertFalse(released)
        self.assertEqual(run.call_count, 2)
        self.assertIn("failed to release session e1-test", log.getvalue())


if __name__ == "__main__":
    unittest.main()
