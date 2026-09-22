#!/usr/bin/env python3
"""Offline tests for the E13 hosted cancellation supervisor.

All tests run without network, provider SDK, or paid compute. They cover the
amendment's load-bearing properties:

1. lease expiry triggers cancellation intent (dry-run receipt with the exact
   resource, nonce and immutable deadline)
2. requester death does not renew the deadline
3. the clock is never rebased (immutable D; rollback fails closed)
4. execution is dry-run only (the Modal call stub never executes)

Plus immutability, replay/stale/reflection rejection, exactly-once intent,
and the CLEANUP_VERIFIED / CLEANUP_UNVERIFIED / CLEANUP_LATE statuses.

Run: python3 -B -m unittest zvf-program/e13_balrog/test_hosted_supervisor.py
  or: python3 -B -m unittest discover -s zvf-program/e13_balrog -p 'test_hosted_supervisor.py' -v
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parent / "hosted_supervisor.py"
spec = importlib.util.spec_from_file_location("hosted_supervisor", MODULE)
hs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hs)

RESOURCE = {"provider": "modal", "operation": "FunctionCallCancel",
            "object_id": "fc-01M29V3NA248SYQHM5D4AWTMTJ",
            "nonce": "a" * 64, "account_digest": "b" * 64, "credential_digest": "c" * 64}


class FakeClock:
    """Deterministic wall+monotonic clock; rollback is injectable."""

    def __init__(self, start=1_000_000.0):
        self.wall = start
        self.mono = 0.0

    def advance(self, seconds):
        self.wall += seconds
        self.mono += seconds

    def rollback(self, seconds):
        self.wall -= seconds

    def __call__(self):
        return self.wall, self.mono


class SupervisorHarness:
    def __init__(self, total_lease_seconds=3600, lease_seconds=5, cleanup_reserve=30):
        self.dir = tempfile.TemporaryDirectory(prefix="e13-supervisor-test-")
        self.volume = Path(self.dir.name)
        self.requester_key = hs.write_key(self.volume / "requester.key")
        self.supervisor_key = hs.write_key(self.volume / "supervisor.key")
        self.clock = FakeClock()
        self.sup = hs.HostedSupervisor(self.volume, self.requester_key, self.supervisor_key,
                                       clock=self.clock)
        self.total = total_lease_seconds
        self.lease_seconds = lease_seconds
        self.cleanup_reserve = cleanup_reserve

    def arm(self, requester_boot="boot-1"):
        return self.sup.arm(RESOURCE, {"boot_id": requester_boot},
                            self.lease_seconds, self.cleanup_reserve, self.total)

    def renew(self, seq, at=None, boot="boot-1"):
        issued = self.clock.wall if at is None else at
        record = self.sup.build_renewal(seq, boot, issued)
        return self.sup.submit_renewal(record)

    def cleanup(self):
        self.dir.cleanup()


class AmendmentTests(unittest.TestCase):
    def setUp(self):
        self.h = SupervisorHarness()
        self.addCleanup(self.h.cleanup)

    # ---- immutability ----
    def test_arm_creates_immutable_lease_with_fixed_deadline(self):
        lease = self.h.arm()
        self.assertEqual(lease["deadline_epoch"], lease["t0_epoch"] + 3600)
        self.assertEqual(lease["schema"], hs.LEASE_SCHEMA)
        with self.assertRaises(hs.SupervisorError):
            self.h.arm(requester_boot="boot-2")  # different content must collide
        reread = self.h.sup.load_lease()
        self.assertEqual(reread, lease)

    # ---- 1. expiry triggers cancellation intent ----
    def test_lease_expiry_triggers_dry_run_cancellation_intent(self):
        lease = self.h.arm()
        outcome = self.h.sup.run_once()
        self.assertEqual(outcome["status"], "LIVE")
        self.h.clock.advance(6)  # past the 5 s short lease, requester silent
        outcome = self.h.sup.run_once()
        self.assertEqual(outcome["status"], "CANCELLATION_INTENT_ISSUED")
        intent = self.h.sup.intent_receipt()
        self.assertEqual(intent["reason"], "LEASE_EXPIRED")
        self.assertEqual(intent["execution"], {"mode": "dry-run", "executed": False,
                                               "executor": "ModalCancellationStub",
                                               "live_execution": "NOT_IMPLEMENTED_IN_THIS_ARTIFACT"})
        self.assertEqual(intent["resource"], RESOURCE)
        self.assertEqual(intent["api_call"]["object_id"], RESOURCE["object_id"])
        self.assertEqual(intent["api_call"]["operation"], "FunctionCallCancel")
        self.assertEqual(intent["api_call"]["nonce"], RESOURCE["nonce"])
        self.assertEqual(intent["deadline_epoch"], lease["deadline_epoch"])
        self.assertFalse(intent["clock_rebased"])

    # ---- 2. requester death does not renew ----
    def test_requester_death_does_not_renew(self):
        lease = self.h.arm()
        self.h.renew(1)                       # alive: valid renewals
        self.h.clock.advance(4)
        self.h.renew(2)
        self.h.clock.advance(4)
        self.assertEqual(self.h.sup.run_once()["status"], "LIVE")
        self.h.clock.advance(4)               # requester dies: renewals stop
        outcome = self.h.sup.run_once()
        self.assertEqual(outcome["status"], "CANCELLATION_INTENT_ISSUED")
        self.assertEqual(outcome["deadline_epoch"], lease["deadline_epoch"])  # D unchanged

    def test_foreign_or_unauthenticated_renewal_cannot_hold_lease(self):
        self.h.arm()
        record = self.h.sup.build_renewal(1, "boot-1", self.h.clock.wall)
        forged = dict(record, hmac="0" * 64)
        with self.assertRaises(hs.SupervisorError):
            self.h.sup.submit_renewal(forged)
        self.h.clock.advance(6)
        self.assertEqual(self.h.sup.run_once()["status"], "CANCELLATION_INTENT_ISSUED")

    # ---- 3. clock never rebased ----
    def test_renewals_cannot_rebase_or_extend_deadline(self):
        lease = self.h.arm()
        # a renewal carrying a doctored expiry is rejected
        record = self.h.sup.build_renewal(1, "boot-1", self.h.clock.wall)
        doctored = dict(record, expires_epoch=record["expires_epoch"] + 3600)
        with self.assertRaises(hs.SupervisorError):
            self.h.sup.submit_renewal(doctored)
        # hold the short lease open with one last valid renewal inside the cutoff
        cutoff = lease["deadline_epoch"] - self.h.cleanup_reserve
        self.h.clock.wall = cutoff - 4          # last acceptable renewal moment
        self.h.renew(1)                          # expires at cutoff+1
        self.h.clock.wall = cutoff               # inside the 30 s reserve, lease still valid
        outcome = self.h.sup.run_once()
        self.assertEqual(outcome["status"], "CANCELLATION_INTENT_ISSUED")
        intent = self.h.sup.intent_receipt()
        self.assertEqual(intent["reason"], "EARLY_CLEANUP_CUTOFF")
        self.assertEqual(intent["deadline_epoch"], lease["t0_epoch"] + 3600)
        self.assertLess(self.h.clock.wall, lease["deadline_epoch"])

    def test_wall_clock_rollback_fails_closed(self):
        lease = self.h.arm()
        self.h.renew(1)
        self.h.clock.rollback(120)  # supervisor host clock jumps backward
        outcome = self.h.sup.run_once()
        self.assertEqual(outcome["status"], "CANCELLATION_INTENT_ISSUED")
        self.assertEqual(self.h.sup.intent_receipt()["reason"], "ROLLBACK_DETECTED")
        kinds = [r["kind"] for r in self.h.sup.receipts()]
        self.assertIn(hs.KIND_ROLLBACK, kinds)
        # the lease file itself was never rewritten
        self.assertEqual(self.h.sup.load_lease(), lease)

    # ---- 4. dry-run only ----
    def test_live_execution_refused_in_run_once(self):
        self.h.arm()
        self.h.clock.advance(6)
        with self.assertRaises(hs.NeverExecutedError):
            self.h.sup.run_once(dry_run=False)
        # nothing was recorded as executed
        self.assertIsNone(self.h.sup.intent_receipt())

    def test_executor_stub_never_executes(self):
        request = self.h.sup.executor.build_request(
            {"resource": RESOURCE, "deadline_epoch": 42.0})
        with self.assertRaises(hs.NeverExecutedError):
            self.h.sup.executor.execute(request)
        self.assertEqual(request["service"], "modal")
        self.assertEqual(request["operation"], "FunctionCallCancel")

    def test_cli_refuses_run_without_dry_run(self):
        self.h.arm()
        proc = subprocess.run(
            [sys.executable, "-B", str(MODULE), "run", "--volume", str(self.h.volume), "--once"],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("dry-run", proc.stderr)

    def test_cli_dry_run_end_to_end_offline(self):
        armed = subprocess.run(
            [sys.executable, "-B", str(MODULE), "arm", "--volume", str(self.h.volume),
             "--object-id", RESOURCE["object_id"], "--nonce", RESOURCE["nonce"]],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(armed.returncode, 0, armed.stderr)
        run = subprocess.run(
            [sys.executable, "-B", str(MODULE), "run", "--volume", str(self.h.volume),
             "--dry-run", "--once"],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("LIVE", run.stdout)

    # ---- replay / stale / reflection ----
    def test_replay_stale_and_reflection_rejected(self):
        self.h.arm()
        self.h.renew(1)
        record1 = json.loads((self.h.volume / "renewals" / "ren-0000000001.json").read_text())
        with self.assertRaises(hs.SupervisorError):
            self.h.sup.submit_renewal(record1)            # replay of seq 1
        record2 = self.h.sup.build_renewal(2, "boot-1", self.h.clock.wall)
        self.h.sup.submit_renewal(record2)
        with self.assertRaises(hs.SupervisorError):
            self.h.sup.submit_renewal(record1)            # stale after seq 2
        # reflection: a supervisor receipt submitted as a renewal
        receipt = self.h.sup.receipts()[0]
        with self.assertRaises(hs.SupervisorError):
            self.h.sup.submit_renewal(dict(receipt, seq=3, schema=hs.RENEWAL_SCHEMA))
        # wrong requester boot is rejected
        with self.assertRaises(hs.SupervisorError):
            self.h.sup.submit_renewal(self.h.sup.build_renewal(3, "boot-evil", self.h.clock.wall))

    # ---- exactly once ----
    def test_cancellation_intent_exactly_once(self):
        self.h.arm()
        self.h.clock.advance(6)
        for _ in range(5):
            outcome = self.h.sup.run_once()
        self.assertEqual(outcome["status"], "CANCELLATION_INTENT_ISSUED")
        intents = [r for r in self.h.sup.receipts() if r["kind"] == hs.KIND_INTENT]
        self.assertEqual(len(intents), 1)

    # ---- amendment reporting statuses ----
    def test_cleanup_verified_when_absence_and_billing_observed_before_deadline(self):
        lease = self.h.arm()
        self.h.clock.advance(6)
        self.h.sup.run_once()
        self.h.sup.record_completion_observation(
            lease["deadline_epoch"] - 10, True, True, {"source": "test"})
        terminal = self.h.sup.terminal_receipt()
        self.assertEqual(terminal["kind"], hs.KIND_CLEANUP_VERIFIED)

    def test_cleanup_late_when_completion_observed_after_deadline(self):
        lease = self.h.arm()
        self.h.clock.advance(6)
        self.h.sup.run_once()
        self.h.sup.record_completion_observation(
            lease["deadline_epoch"] + 60, True, True, {"source": "test"})
        terminal = self.h.sup.terminal_receipt()
        self.assertEqual(terminal["kind"], hs.KIND_CLEANUP_LATE)
        self.assertTrue(terminal["possible_continuing_charges"])

    def test_cleanup_unverified_when_window_elapses_without_observation(self):
        lease = self.h.arm()
        self.h.clock.advance(6)
        self.h.sup.run_once()
        # supervision window is anchored at max(D, intent) + observation_seconds
        self.h.clock.advance((lease["deadline_epoch"] - self.h.clock.wall)
                             + hs.DEFAULT_OBSERVATION_SECONDS + 1)
        outcome = self.h.sup.run_once()
        self.assertEqual(outcome["status"], hs.KIND_CLEANUP_UNVERIFIED)
        terminal = self.h.sup.terminal_receipt()
        self.assertFalse(terminal["provider_absence_observed"])
        self.assertFalse(terminal["billing_closure_observed"])

    def test_partial_observation_is_not_terminal(self):
        self.h.arm()
        self.h.clock.advance(6)
        self.h.sup.run_once()
        result = self.h.sup.record_completion_observation(
            self.h.clock.wall, True, False, {"source": "test"})
        self.assertEqual(result["status"], "OBSERVED_INCOMPLETE")
        self.assertIsNone(self.h.sup.terminal_receipt())

    # ---- static safety: no network surface ----
    def test_module_has_no_network_imports(self):
        source = MODULE.read_text()
        for banned in ("import socket", "import requests", "import urllib",
                       "import http.client", "from modal", "import modal"):
            self.assertNotIn(banned, source)
        leaked = {name for name in vars(hs)
                  if name.split(".")[0] in ("socket", "requests", "urllib", "urllib3",
                                            "modal", "http")}
        self.assertEqual(leaked, set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
