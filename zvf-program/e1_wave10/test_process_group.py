"""E1 wave10 targeted regressions: scoped process-group finalization (v6).

Rewritten from the lost suite recorded in
outputs/PES_Phase2_Review_2026-09-12/finish/e9_completion/e1_v6_rpc_review01/targeted_tests.log
against the sealed contract in
outputs/PES_Phase2_Review_2026-09-12/finish/e1_recovery_v6/REVIEW_PACKET.md.

Four tests (GroupFinalizationTests):
  1. an exit-0 worker that leaves a live same-PGID child is finalized (group
     SIGKILL, worker reap, verified original-group absence, durable receipt)
     BEFORE a successful result is returned -- reproducing the sealed
     original_counterexample fixture (worker exit 0, child.pid 4178 in PGID
     4177, group_finalization.json verified before result.json was read);
  2. an exit-7 worker with the same live child is finalized the same way, but
     the outcome stays UNKNOWN/no-replay (fixture_exit_7: group absence
     verified, unknown.json retained);
  3. unresolved group verification (the independent killpg(pgid,0)->ESRCH wait
     cannot confirm absence within budget) blocks success for an exit-0 worker
     and preserves UNKNOWN;
  4. the same unresolved verification on a failing worker still preserves
     UNKNOWN.

Real subprocesses only: workers spawn SIGTERM-ignoring same-PGID sleepers, so
nothing but a genuine process-group SIGKILL can clear the group inside the
test budget.  os.killpg is never mocked; the sole injected fault is the
unresolved-verification wait seam explained in _e1_procgroup_support.
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _e1_procgroup_support as support


class GroupFinalizationTests(unittest.TestCase):
    """The four group-finalization regressions from targeted_tests.log."""

    MAX_RUN_SECONDS = 45.0  # child sleeps 60s, so success inside 45s proves a group kill
    WORKER_TIMEOUT = 8.0

    def setUp(self):
        support.require_module(self)
        self.workdir = Path(
            os.environ.get(
                "E1_WAVE10_TMP",
                os.path.join(tempfile_dir(), f"e1_wave10_group_{self.id().rsplit('.', 1)[-1][:40]}_{os.getpid()}"),
            )
        )
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(self._remove_workdir)

    def _remove_workdir(self):
        try:
            os.chmod(self.workdir, 0o700)
        except OSError:
            pass
        for entry in list(self.workdir.iterdir()):
            try:
                entry.unlink()
            except OSError:
                pass
        try:
            self.workdir.rmdir()
        except OSError:
            pass

    # -- shared driver -----------------------------------------------------

    def _run_exit_worker(self, exit_code):
        command = support.worker_exit_argv(self.workdir, exit_code)
        started = time.monotonic()
        outcome = support.call_run_child(self, command, self.workdir / "result.json", self.workdir, self.WORKER_TIMEOUT)
        elapsed = time.monotonic() - started
        return outcome, elapsed

    def _assert_group_leader_fixture_shape(self, process_info, before):
        self.assertEqual(
            process_info.get("pgid"),
            process_info.get("pid"),
            "worker must lead its own process group (fixture process.json pid==pgid)",
        )
        self.assertEqual(before["worker_pid"], process_info["pid"])
        self.assertEqual(before["worker_pgid"], process_info["pgid"])
        self.assertEqual(
            before["child_pgid"],
            before["worker_pgid"],
            "the sleeper must be in the worker's process group",
        )
        self.assertEqual(before["child_alive"], True)

    def _register_group_cleanup(self, pgid):
        self.addCleanup(lambda: support.kill_group_hard(pgid))

    def _assert_independent_absence(self, pgid, child_pid, elapsed):
        # Independent observation (mirrors independent_observation.json):
        # after the controller returns, the original group and the child must
        # be gone even though the child was designed to sleep for 60s.
        self.assertTrue(
            support.group_absent(pgid, timeout=5.0),
            "original owned PGID {} still observable after return".format(pgid),
        )
        self.assertTrue(
            support.pid_dead(child_pid, timeout=5.0),
            "same-group child {} still alive/reaped-late after return".format(child_pid),
        )
        self.assertLess(
            elapsed,
            self.MAX_RUN_SECONDS,
            "controller returned in {:.1f}s; a 60s sleeper inside the group can only "
            "have been removed by a real group SIGKILL during finalization".format(elapsed),
        )

    # -- test 1 ------------------------------------------------------------

    def test_worker_exit_zero_with_live_same_group_child_is_finalized_before_success(self):
        outcome, elapsed = self._run_exit_worker(0)

        self.assertTrue(
            support.is_success(outcome),
            "exit-0 worker with verified group finalization must succeed, got {!r}".format(outcome),
        )
        self.assertEqual(
            support.outcome_result(outcome),
            {"returned": True},
            "result.json must be read only after verified group absence",
        )

        process_info = support.load_json(self, self.workdir / "process.json")
        before = support.load_json(self, self.workdir / "t_before.json")
        self._assert_group_leader_fixture_shape(process_info, before)
        pgid = process_info["pgid"]
        child_pid = before["child_pid"]
        self._register_group_cleanup(pgid)

        receipt = support.load_json(self, self.workdir / "group_finalization.json")
        support.assert_finalization_receipt(
            self, receipt, expected_returncode=0
        )
        self.assertEqual(receipt["pgid"], pgid)
        self.assertTrue(
            receipt["group_sigkill_sent"],
            "a live same-group child existed, so finalization must have sent the "
            "group SIGKILL before success was established",
        )
        self.assertTrue(receipt["original_group_absent"])
        self.assertTrue(receipt["verified"])

        self.assertFalse(
            (self.workdir / "unknown.json").exists(),
            "verified success path must not retain an unknown receipt",
        )
        self._assert_independent_absence(pgid, child_pid, elapsed)

    # -- test 2 ------------------------------------------------------------

    def test_worker_failure_with_live_same_group_child_is_finalized_before_unknown(self):
        outcome, elapsed = self._run_exit_worker(7)

        self.assertTrue(
            support.is_unknown(outcome),
            "nonzero worker must stay UNKNOWN even with verified group cleanup, got {!r}".format(outcome),
        )
        self.assertIsNone(
            support.outcome_result(outcome),
            "no successful result may be returned for a failing worker",
        )

        process_info = support.load_json(self, self.workdir / "process.json")
        before = support.load_json(self, self.workdir / "t_before.json")
        self._assert_group_leader_fixture_shape(process_info, before)
        pgid = process_info["pgid"]
        child_pid = before["child_pid"]
        self._register_group_cleanup(pgid)

        receipt = support.load_json(self, self.workdir / "group_finalization.json")
        support.assert_finalization_receipt(
            self, receipt, expected_returncode=7
        )
        self.assertEqual(receipt["pgid"], pgid)
        self.assertTrue(receipt["group_sigkill_sent"])
        self.assertTrue(receipt["original_group_absent"])
        self.assertTrue(receipt["verified"])

        unknown = support.load_json(self, self.workdir / "unknown.json")
        self.assertEqual(unknown.get("returncode"), 7)
        self.assertTrue(unknown.get("reaped"), "unknown receipt must record the reaped worker")
        self.assertTrue(
            unknown.get("replay_forbidden"),
            "UNKNOWN must be immutable/no-replay (fixture_exit_7 unknown.json)",
        )
        self._assert_independent_absence(pgid, child_pid, elapsed)

    # -- tests 3 & 4: unresolved verification -------------------------------

    def _run_unresolved(self, exit_code):
        with support.UnresolvedVerificationFault(self) as fault:
            if not support.deterministic_fault_applied(fault):
                self.skipTest(
                    "no deterministic unresolved-verification seam on bounded_procgroup "
                    "(tried {}, inject_fault/set_fault, env {}); cannot force the "
                    "'group remains present' branch offline".format(support.WAIT_SEAM_CANDIDATES, tuple(support.FAULT_ENV))
                )
            outcome, elapsed = self._run_exit_worker(exit_code)
        return outcome, elapsed

    def test_unresolved_group_verification_blocks_success_and_preserves_unknown(self):
        outcome, elapsed = self._run_unresolved(0)

        self.assertFalse(
            support.is_success(outcome),
            "exit-0 worker whose group absence cannot be verified must NOT produce a "
            "successful result (UNKNOWN/no-replay), got {!r}".format(outcome),
        )
        self.assertTrue(
            support.is_unknown(outcome),
            "unresolved verification must preserve UNKNOWN, got {!r}".format(outcome),
        )
        self.assertIsNone(support.outcome_result(outcome))

        process_info = support.load_json(self, self.workdir / "process.json")
        self._register_group_cleanup(process_info["pgid"])

        unknown = support.load_json(self, self.workdir / "unknown.json")
        self.assertTrue(
            unknown.get("replay_forbidden"),
            "unresolved verification must retain unknown.json with no-replay",
        )
        receipt = support.load_json(self, self.workdir / "group_finalization.json")
        self.assertIn("verified", receipt)
        self.assertFalse(
            bool(receipt["verified"]),
            "receipt must record that group verification did not resolve: {}".format(receipt),
        )
        # Elapsed time still bounded by the controller deadline + cleanup budget.
        self.assertLess(elapsed, self.MAX_RUN_SECONDS)

    def test_unresolved_failure_also_preserves_unknown(self):
        outcome, elapsed = self._run_unresolved(7)

        self.assertTrue(
            support.is_unknown(outcome),
            "unresolved verification on a failing worker must stay UNKNOWN, got {!r}".format(outcome),
        )
        self.assertIsNone(support.outcome_result(outcome))

        process_info = support.load_json(self, self.workdir / "process.json")
        self._register_group_cleanup(process_info["pgid"])

        unknown = support.load_json(self, self.workdir / "unknown.json")
        self.assertEqual(unknown.get("returncode"), 7)
        self.assertTrue(unknown.get("replay_forbidden"))
        receipt = support.load_json(self, self.workdir / "group_finalization.json")
        self.assertFalse(
            bool(receipt["verified"]),
            "receipt must record unresolved verification: {}".format(receipt),
        )
        self.assertLess(elapsed, self.MAX_RUN_SECONDS)


def tempfile_dir():
    import tempfile

    return tempfile.gettempdir()


if __name__ == "__main__":
    unittest.main()
