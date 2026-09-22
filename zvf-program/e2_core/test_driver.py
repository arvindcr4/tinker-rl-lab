#!/usr/bin/env python3
"""Offline tests for the E2 orchestration driver (no provider, no spend).

All provider behavior runs through FakeComputeProvider; time runs through a
manually advanced fake clock (time.sleep patched to advance it). Nothing here
may import a cloud SDK or open a socket.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import driver


H = 2_000_000
A = H + 100  # satisfies A <= H+210


class FakeClock:
    """Reads are free; only patched sleep advances time (one step per call)."""

    def __init__(self, start, step=60.0):
        self.now_value = float(start)
        self.step = float(step)

    def __call__(self):
        return self.now_value


class ScriptRunner:
    def __init__(self, capsules):
        self._capsules = list(capsules)
        self.ran = []

    def capsules(self):
        return list(self._capsules)

    def __call__(self, capsule_id):
        self.ran.append(capsule_id)
        return {"ok": True}


class AutoDeleteFake(driver.FakeComputeProvider):
    def delete_instance(self, name: str) -> dict:
        result = super().delete_instance(name)
        self.effect_delete(name)
        return result


def make_tmp(test):
    tmp = Path(tempfile.mkdtemp(prefix="e2drv_"))
    test.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
    return tmp


class TestSchedule(unittest.TestCase):
    def test_offsets_and_tail_constraint(self):
        sched = driver.compute_schedule(H, A)
        self.assertEqual(sched["target_cutoff"], A + 1590)
        self.assertEqual(sched["target_budget_end"], A + 1800)
        self.assertEqual(sched["helper_delete_at"], H + 2610)
        self.assertEqual(sched["helper_budget_end"], H + 2700)

    def test_boundary_accepts_equal(self):
        sched = driver.compute_schedule(H, H + 210)
        self.assertEqual(sched["target_arm_epoch_A"], H + 210)

    def test_tail_violation_rejected(self):
        with self.assertRaises(driver.ScheduleError):
            driver.compute_schedule(H, H + 211)

    def test_admission_strict_cutoff(self):
        sched = driver.compute_schedule(H, A)
        self.assertTrue(driver.admission_open(sched, A + 1590 - 0.001))
        self.assertFalse(driver.admission_open(sched, A + 1590))
        self.assertFalse(driver.admission_open(sched, A + 1590 + 100))


class TestWatchdog(unittest.TestCase):
    def test_exact_offsets_and_once_each(self):
        sched = driver.compute_schedule(H, A)
        tmp = make_tmp(self)
        provider = driver.FakeComputeProvider()
        watch = driver.Watchdog(sched, "tgt", "hlp", tmp)
        self.assertEqual(watch.due(A + 1589), [])
        self.assertEqual(watch.due(A + 1590), ["target"])
        watch.fire(provider, "target", A + 1590)
        self.assertEqual(watch.due(A + 1591), [])
        self.assertEqual(watch.due(H + 2609), [])
        self.assertEqual(watch.due(H + 2610), ["helper"])
        watch.fire(provider, "helper", H + 2610)
        self.assertEqual(watch.due(H + 9999), [])
        self.assertTrue((tmp / "delete_intent_target.json").is_file())
        self.assertTrue((tmp / "delete_result_helper.json").is_file())
        kinds = [c[0] for c in provider.calls]
        self.assertEqual(kinds, ["delete_instance", "delete_instance"])


class MissionBase(unittest.TestCase):
    def run_mission(self, provider, clock, runner=None, disks=None):
        tmp = make_tmp(self)
        sched = driver.compute_schedule(H, A)
        # Advance the fake clock through real sleeps: patch sleep to step it.
        def advance(_s):
            clock.now_value += clock.step
        with mock.patch.object(driver.time, "sleep", side_effect=advance):
            terminal = driver.run_mission(
                provider, sched, clock, tmp, "tgt", "hlp",
                target_disks=(disks or []), helper_disks=[],
                capsule_runner=runner, poll_s=0)
        return terminal, tmp, sched


class TestMissionClean(MissionBase):
    def test_clean_path_with_capsules(self):
        provider = AutoDeleteFake()
        clock = FakeClock(A - 500)
        runner = ScriptRunner(["cap-1", "cap-2"])
        terminal, tmp, sched = self.run_mission(provider, clock, runner)
        self.assertEqual(terminal["status"], "COMPLETE_CLEAN")
        self.assertEqual(terminal["verdict"], "CLEAN")
        self.assertEqual(runner.ran, ["cap-1", "cap-2"])
        self.assertTrue((tmp / "mission_terminal.json").is_file())
        self.assertTrue((tmp / "absence_observations.json").is_file())
        # Schedule epochs immutable: no clock rebasing through the run.
        self.assertEqual(sched["target_arm_epoch_A"], A)
        self.assertEqual(sched["helper_epoch_H"], H)


class TestMissionCutoff(MissionBase):
    def test_no_admission_past_cutoff(self):
        provider = AutoDeleteFake()
        clock = FakeClock(A + 1590 + 10)
        runner = ScriptRunner(["cap-1"])
        terminal, tmp, sched = self.run_mission(provider, clock, runner)
        self.assertEqual(runner.ran, [])
        events = [e["event"] for e in terminal["events"]]
        self.assertIn("admission_refused_cutoff", events)
        self.assertEqual(terminal["status"], "COMPLETE_CLEAN")


class TestMissionUnverified(MissionBase):
    def test_unverified_when_absence_unobserved(self):
        provider = driver.FakeComputeProvider()  # deletes accepted, never effective
        clock = FakeClock(A - 100)
        terminal, tmp, sched = self.run_mission(provider, clock)
        self.assertEqual(terminal["verdict"], "CLEANUP_UNVERIFIED")
        self.assertEqual(terminal["status"], "COMPLETE_CLEANUP_UNVERIFIED")
        events = [e["event"] for e in terminal["events"]]
        self.assertIn("root_notified", events)
        obs = json.loads((tmp / "absence_observations.json").read_text())
        self.assertTrue(obs["observations"]["target_vm"]["present"])


class TestMissionUncertain(MissionBase):
    def test_uncertain_mutation_stops_without_replay(self):
        provider = driver.FakeComputeProvider()
        provider.uncertain_ops.add(("create_instance", "hlp"))
        clock = FakeClock(A - 100)
        terminal, tmp, sched = self.run_mission(provider, clock)
        self.assertEqual(terminal["status"], "STOP_UNCERTAIN_NO_REPLAY")
        kinds = [c[0] for c in provider.calls]
        self.assertNotIn("delete_instance", kinds)  # no further mutations
        self.assertEqual(kinds.count("create_instance"), 2)  # target ok, helper raised


class TestIamGate(unittest.TestCase):
    def write_binding(self, tmp, **over):
        binding = {"permissions": list(driver.REQUIRED_IAM_PERMISSIONS),
                   "project": driver.ENVELOPE["project"],
                   "valid_until_epoch": 9_999_999_999}
        binding.update(over)
        path = tmp / "binding.json"
        path.write_text(json.dumps(binding))
        return path

    def test_gate_refusals(self):
        tmp = make_tmp(self)
        with self.assertRaises(driver.IamGateError):
            driver.check_iam_binding(tmp / "missing.json")
        bad = self.write_binding(tmp, permissions=["compute.instances.get"])
        with self.assertRaises(driver.IamGateError):
            driver.check_iam_binding(bad)
        expired = self.write_binding(tmp, valid_until_epoch=1)
        with self.assertRaises(driver.IamGateError):
            driver.check_iam_binding(expired)
        wrong_proj = self.write_binding(tmp, project="other")
        with self.assertRaises(driver.IamGateError):
            driver.check_iam_binding(wrong_proj)

    def test_gate_accepts_before_any_sdk_import(self):
        tmp = make_tmp(self)
        good = self.write_binding(tmp)
        binding = driver.check_iam_binding(good)
        self.assertEqual(binding["project"], driver.ENVELOPE["project"])
        # Live provider refuses before importing the (absent) SDK.
        with self.assertRaises(driver.IamGateError):
            driver.GcpComputeProvider(tmp / "missing.json")

    def test_validate_offline(self):
        report = driver.validate_offline()
        self.assertTrue(report["ok"], report["findings"])


if __name__ == "__main__":
    unittest.main()
