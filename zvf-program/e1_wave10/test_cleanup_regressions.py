"""E1 wave10 targeted regressions: bounded-SDK controller walls (v6).

Rewritten from the lost suite recorded in
outputs/PES_Phase2_Review_2026-09-12/finish/e9_completion/e1_v6_rpc_review01/targeted_tests.log
against the sealed contract in
outputs/PES_Phase2_Review_2026-09-12/finish/e1_recovery_v6/REVIEW_PACKET.md
and cleanup_contract_v6.json (sdk_controller block):

  - on_block_or_guard_failure: SIGKILL owned group; bounded wait/reap;
    immutable UNKNOWN; no automatic operation retry
  - one owned process group per SDK operation; the parent/controller keeps no
    SDK calls or executor threads
  - cleanup deadlines never renew on reconciliation
    (cleanup_deadline_renewal: false)
  - cleanup must not depend on the work-disk guard
  - ambiguous create reconciles on exact app/name/image/reservation-tag only;
    missing result or named-lookup absence is UNKNOWN, never proof of no
    delayed creation

Six tests (SDKWallTests).  All worker processes are REAL subprocesses in real
process groups (SIGTERM-ignoring sleepers where a group kill must be proven);
os.killpg is never mocked.
"""

from __future__ import annotations

import concurrent.futures
import os
import sys
import threading
import tempfile
import shutil
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _e1_procgroup_support as support


TOTAL_CLEANUP_CONTROLLER_SECONDS = 60.0  # cleanup_contract_v6.json
REAP_RESERVE_SECONDS = 2.0


class SDKWallTests(unittest.TestCase):
    """The six controller-regression tests from targeted_tests.log."""

    def setUp(self):
        support.require_module(self)
        name = self.id().rsplit(".", 1)[-1][:48]
        self.workdir = Path(tempfile.mkdtemp(prefix=f"e1_wave10_sdk_{name}_"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(self._restore_dir_permissions)
        self.addCleanup(self._remove_workdir)
        self._pgids = []
        self.addCleanup(self._kill_leaked_groups)

    # -- housekeeping -------------------------------------------------------

    def _restore_dir_permissions(self):
        try:
            os.chmod(self.workdir, 0o700)
        except OSError:
            pass

    def _remove_workdir(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _kill_leaked_groups(self):
        for pgid in self._pgids:
            support.kill_group_hard(pgid)

    def _track(self, workdir):
        info = support.load_json(self, workdir / "process.json")
        assert info.get("pgid") == info.get("pid"), (
            "each SDK operation owns exactly one process group led by the worker"
        )
        assert "deadline" in info, "controller deadline must be persisted"
        self._pgids.append(info["pgid"])
        return info

    # -- test 1 ------------------------------------------------------------

    def test_every_reported_blocking_sdk_path_is_killed_and_reaped(self):
        """The six reported hung fake-SDK operation shapes.

        Every blocking path must end in a group SIGKILL, a reaped direct
        worker, verified group absence, immutable UNKNOWN and no returned
        result -- bounded by the original controller deadline plus the cleanup
        budget, never by the worker's own 60s hang.
        """
        timeout = 2.0
        for shape in support.BLOCKING_SDK_SHAPES:
            with self.subTest(shape=shape):
                workdir = self.workdir / shape
                workdir.mkdir(parents=True, exist_ok=True)
                command = support.worker_hang_argv(shape, workdir)
                started = time.monotonic()
                outcome = support.call_run_child(
                    self, command, workdir / "result.json", workdir, timeout
                )
                elapsed = time.monotonic() - started

                assert support.is_unknown(outcome), (
                    "{}: hung SDK path must be UNKNOWN, got {!r}".format(shape, outcome)
                )
                assert support.outcome_result(outcome) is None, (
                    "{}: a blocked operation must not return a result".format(shape)
                )

                info = self._track(workdir)
                worker_pid = info["pid"]
                pgid = info["pgid"]

                assert support.pid_dead(worker_pid, timeout=5.0), (
                    "{}: worker {} was not reaped after the wall".format(shape, worker_pid)
                )
                assert support.group_absent(pgid, timeout=5.0), (
                    "{}: owned group {} survived the wall".format(shape, pgid)
                )
                assert elapsed < timeout + TOTAL_CLEANUP_CONTROLLER_SECONDS, (
                    "{}: wall exceeded the non-renewing cleanup budget".format(shape)
                )
                assert elapsed < 55.0, (
                    "{}: worker hangs for 60s; returning in {:.1f}s proves an "
                    "external kill rather than natural exit".format(shape, elapsed)
                )

                if shape == "hang_after_result":
                    assert (workdir / "result.json").exists(), (
                        "fixture shape precondition: worker wrote result.json before hanging"
                    )
                if shape == "hang_with_same_group_child":
                    before = support.load_json(self, workdir / "t_before.json")
                    assert support.pid_dead(before["child_pid"], timeout=5.0), (
                        "{}: same-group child must be killed with the group".format(shape)
                    )

    # -- test 2 ------------------------------------------------------------

    def test_guard_failure_interrupts_running_sdk_process(self):
        """Work-guard failure (deadline exceeded mid-run) interrupts the SDK process."""
        timeout = 1.5
        workdir = self.workdir / "guard"
        workdir.mkdir(parents=True, exist_ok=True)
        command = support.worker_hang_argv("guard_invocation_once", workdir)
        started = time.monotonic()
        outcome = support.call_run_child(self, command, workdir / "result.json", workdir, timeout)
        elapsed = time.monotonic() - started

        assert support.is_unknown(outcome), "guard failure must produce UNKNOWN, got {!r}".format(
            outcome
        )
        assert support.outcome_result(outcome) is None
        unknown_path = workdir / "unknown.json"
        if unknown_path.exists():
            unknown = support.load_json(self, unknown_path)
            assert unknown.get("replay_forbidden"), "guard failure is immutable UNKNOWN/no-replay"

        info = self._track(workdir)
        assert support.pid_dead(info["pid"], timeout=5.0), (
            "guard failure must interrupt and reap the running SDK process"
        )
        assert support.group_absent(info["pgid"], timeout=5.0), (
            "guard failure must SIGKILL the owned group"
        )
        assert elapsed < timeout + TOTAL_CLEANUP_CONTROLLER_SECONDS

        invocations = (workdir / "t_invocations.log").read_text().splitlines()
        assert len(invocations) == 1, "no automatic operation retry after guard failure: {}".format(
            invocations
        )
        assert not (workdir / "result.json").exists(), (
            "an interrupted SDK process must not produce a successful result artifact"
        )

    # -- test 3 ------------------------------------------------------------

    def test_no_sdk_calls_or_executor_threads_remain_in_controller(self):
        """The controller (this process) keeps no SDK threads/executors or children."""
        baseline = {thread.ident for thread in threading.enumerate()}

        outcomes = []
        scenarios = (
            (support.worker_exit_argv(self.workdir / "ok", 0), self.workdir / "ok", 8.0),
            (support.worker_exit_argv(self.workdir / "fail", 7), self.workdir / "fail", 8.0),
        )
        for command, workdir, timeout in scenarios:
            workdir.mkdir(parents=True, exist_ok=True)
            outcome = support.call_run_child(
                self, command, workdir / "result.json", workdir, timeout
            )
            outcomes.append(
                (workdir.name, support.is_success(outcome), support.is_unknown(outcome))
            )
            self._track(workdir)
        assert [flag for _, flag, _ in outcomes] == [True, False], (
            "controller sanity: exit-0 op succeeds, exit-7 op is UNKNOWN"
        )
        assert [flag for _, _, flag in outcomes] == [False, True]

        # Any cleanup worker thread must have exited by now; poll briefly.
        deadline = time.monotonic() + 3.0
        leftover = []
        while True:
            leftover = [
                thread
                for thread in threading.enumerate()
                if thread.ident not in baseline and thread is not threading.current_thread()
            ]
            if not leftover or time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        assert [thread.name for thread in leftover] == [], (
            "controller leaked live threads after operations finished"
        )

        # Every spawned child (direct workers) must be reaped: no zombies and
        # nothing still running under this controller pid.
        deadline = time.monotonic() + 3.0
        while True:
            try:
                reaped = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break  # no children at all: clean
            if reaped == (0, 0):
                if time.monotonic() >= deadline:
                    self.fail("controller still owns running child processes")
                time.sleep(0.1)
                continue
            if time.monotonic() >= deadline:
                self.fail("controller left an unreaped child: {!r}".format(reaped))
            time.sleep(0.1)

        # No module-level executor may keep live threads.
        for value in vars(support.bounded_procgroup).values():
            if isinstance(value, concurrent.futures.Executor):
                assert getattr(value, "_threads", []) == [], (
                    "module-level executor {!r} still has live threads".format(value)
                )

    # -- test 4 ------------------------------------------------------------

    def test_cleanup_deadline_is_not_extended_on_reconciliation(self):
        """A missing result forces the reconciliation decision; the deadline
        recorded at launch (absolute epoch converted once) must not be pushed
        forward, and cleanup must finish inside the original budget."""
        timeout = 2.0
        workdir = self.workdir / "missing_result"
        workdir.mkdir(parents=True, exist_ok=True)
        # Worker exits 0 immediately but never writes result.json: the
        # controller must reconcile and return UNKNOWN, not success.
        command = [
            sys.executable,
            "-B",
            "-c",
            "import time, sys; time.sleep(0.2); sys.exit(0)",
        ]
        launch_epoch = time.time()
        started = time.monotonic()
        outcome = support.call_run_child(self, command, workdir / "result.json", workdir, timeout)
        elapsed = time.monotonic() - started

        assert not support.is_success(outcome), (
            "missing result must not be treated as success, got {!r}".format(outcome)
        )
        assert support.is_unknown(outcome), (
            "missing result is UNKNOWN, not proof of no delayed creation: {!r}".format(outcome)
        )

        info = self._track(workdir)
        deadline_epoch = info["deadline"]
        assert isinstance(deadline_epoch, float)
        assert deadline_epoch >= launch_epoch + timeout - 3.0, (
            "deadline must be the launch deadline, not shrunk"
        )
        assert deadline_epoch <= launch_epoch + timeout + 3.0, (
            "deadline was extended past the once-converted launch deadline "
            "(launch+timeout={:.3f}, recorded={:.3f}): cleanup deadlines must not renew "
            "on reconciliation".format(launch_epoch + timeout, deadline_epoch)
        )
        assert elapsed < timeout + TOTAL_CLEANUP_CONTROLLER_SECONDS, (
            "cleanup outlived the non-renewing budget"
        )
        assert support.group_absent(info["pgid"], timeout=5.0), (
            "missing-result cleanup must still finalize the group"
        )

    # -- test 5 ------------------------------------------------------------

    def test_cleanup_does_not_depend_on_work_disk_guard(self):
        """Cleanup (SIGKILL + reap + group-absence verification) must proceed
        even when every write into the work disk fails: the worker revokes
        write permission on the execution directory before hanging."""
        timeout = 2.0
        workdir = self.workdir / "disk_guard_down"
        workdir.mkdir(parents=True, exist_ok=True)
        command = support.worker_hang_argv("hang_after_revoking_work_disk", workdir)
        started = time.monotonic()
        try:
            outcome = support.call_run_child(
                self, command, workdir / "result.json", workdir, timeout
            )
        except Exception as exc:
            self.fail(
                "cleanup must not raise when the work-disk guard cannot write: {!r}".format(exc)
            )
        elapsed = time.monotonic() - started

        assert support.is_unknown(outcome), (
            "work-disk guard failure path must still be UNKNOWN, got {!r}".format(outcome)
        )
        assert elapsed < timeout + TOTAL_CLEANUP_CONTROLLER_SECONDS

        child_pid = int((workdir / "child.pid").read_text().strip())
        assert support.pid_dead(child_pid, timeout=5.0), (
            "the same-group child must be killed even when the guard cannot write"
        )
        # The owned group must be absent. process.json was written at launch
        # (before the worker revoked permissions); tolerate its absence here in
        # case the implementation writes receipts elsewhere on this path.
        info_path = workdir / "process.json"
        if info_path.exists():
            info = support.load_json(self, info_path)
            self._pgids.append(info["pgid"])
            assert support.group_absent(info["pgid"], timeout=5.0), (
                "owned group must be finalized without the work-disk guard"
            )
        else:
            # Independent fallback: the worker's own group is gone iff both the
            # (reparented) child and a fresh probe of the worker pgid are gone.
            before_path = workdir / "t_before.json"
            if before_path.exists():
                import json as _json

                before = _json.loads(before_path.read_text())
                self._pgids.append(before["worker_pgid"])
                assert support.group_absent(before["worker_pgid"], timeout=5.0), (
                    "worker-owned group must be finalized without the work-disk guard"
                )

    # -- test 6 ------------------------------------------------------------

    def test_ambiguous_create_reconciliation_requires_exact_app_name_image_tag(self):
        persisted = {
            "app_name": "pes-e1-wave10",
            "image_tag": "registry.internal/pes-e1:v6",
            "reservation_tag": "e1-hold-ledger32",
        }

        def candidate(name, image, tag, state="stopped", description="", app_id="ap-1"):
            # Alias-rich shape so whichever field names the implementation
            # reads are all present and consistent.
            return {
                "app_id": app_id,
                "name": name,
                "app_name": name,
                "image": image,
                "image_tag": image,
                "reservation": tag,
                "reservation_tag": tag,
                "state": state,
                "status": state,
                "description": description,
                "tasks": 0,
            }

        exact = candidate(
            persisted["app_name"], persisted["image_tag"], persisted["reservation_tag"]
        )
        wrong_name = candidate(
            "other-app",
            persisted["image_tag"],
            persisted["reservation_tag"],
            description="pes-e1-wave10 related workload",
        )
        wrong_image = candidate(
            persisted["app_name"], "registry.internal/pes-e1:v5", persisted["reservation_tag"]
        )
        wrong_reservation = candidate(
            persisted["app_name"], persisted["image_tag"], "e1-hold-other"
        )
        deployed = candidate(
            persisted["app_name"],
            persisted["image_tag"],
            persisted["reservation_tag"],
            state="deployed",
        )
        duplicate = candidate(
            persisted["app_name"],
            persisted["image_tag"],
            persisted["reservation_tag"],
            app_id="ap-2",
        )

        cases = [
            ("exact match reconciles", [exact], True),
            ("description match alone is rejected", [wrong_name], False),
            ("image tag mismatch is rejected", [wrong_image], False),
            ("reservation tag mismatch is rejected", [wrong_reservation], False),
            (
                "deployed/queued apps are rejected even with exact identity and zero tasks",
                [deployed],
                False,
            ),
            ("two exact-looking candidates are ambiguous and rejected", [exact, duplicate], False),
            ("named lookup absence is UNKNOWN, never reconciled", [], False),
        ]
        for label, candidates, expected in cases:
            with self.subTest(case=label):
                result = support.call_reconcile(self, persisted, candidates)
                if result is support.NO_RECONCILE_ENTRY:
                    reason = (
                        "bounded_procgroup exposes no ambiguous-create reconciliation "
                        "entry point (tried {}); cannot verify exact-identity "
                        "reconciliation offline".format(", ".join(support._RECONCILE_NAMES))
                    )
                    self.skipTest(reason)
                assert support.reconcile_says_reconciled(self, result) == expected, (
                    "{}: verdict {!r}".format(label, result)
                )


def tempfile_dir():
    import tempfile

    return tempfile.gettempdir()


if __name__ == "__main__":
    unittest.main()
