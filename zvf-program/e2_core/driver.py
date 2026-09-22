#!/usr/bin/env python3
"""E2 CORE-Bench orchestration driver (direct-VM HARD, GCP).

Rebuilds the lost finish-era driver per
``outputs/PES_Phase2_Review_2026-09-12/finish/e2_completion/launch_plan_2026-09-19.md``
against decision_v19 timing and the accepted lifecycle amendment
(``amendment_acceptance_2026-09-19.json``).

Discipline (all offline-testable; provider calls only behind the seam):
  - fixed A (target-arm) / H (helper-request, floored) epochs; cutoff
    A+1590, target budget end A+1800, helper DELETE H+2610, helper budget
    end H+2700; A<=H+210 required; clocks are never rebased.
  - no work admission at/after cutoff; no retries, no replay of uncertain
    mutations ("stop and reconcile").
  - early owned DELETEs issued by a watchdog at the exact offsets; absence
    established only by separate GET observations; otherwise
    CLEANUP_UNVERIFIED / CLEANUP_LATE with identities.
  - every mutating step writes an intent receipt BEFORE acting and a result
    receipt after.

Default mode is validation only (amendment + schedule math + envelope).
``--execute`` additionally requires a fresh owner IAM binding receipt and
sealed v14-style request; without them it refuses (fail-closed). Provider
SDKs import lazily inside the GCP provider so this module imports offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIN = (REPO / "outputs/PES_Phase2_Review_2026-09-12/finish/e2_completion")

SCHEMA = "e2-orchestration-driver-v1"

# decision_v19 timing (seconds, fixed offsets, never rebased).
TARGET_CUTOFF_OFFSET = 1590
TARGET_BUDGET_END_OFFSET = 1800
HELPER_DELETE_OFFSET = 2610
HELPER_BUDGET_END_OFFSET = 2700
HELPER_TAIL_SECONDS = 600
A_LE_H_PLUS = 210
DELETE_ONSET_ALLOWANCE = 30

# Envelope (decision_v19, unchanged).
ENVELOPE = {
    "target_hold_usd": 3,
    "helper_hold_usd": 1,
    "target": {"machine": "n1-standard-4", "vcpus": 4, "memory_gib": 15,
               "accelerator": "nvidia-tesla-t4", "accelerator_count": 1,
               "boot_disk_gib": 50, "probe_disk_gib": 64,
               "probe_cap_gib": 48},
    "helper": {"machine": "n1-standard-1", "vcpus": 1, "memory_gib": 3.75,
               "boot_disk_gib": 10},
    "project": "electric-armor-388216",
    "zone": "us-central1-a",
}
REQUIRED_IAM_PERMISSIONS = (
    "compute.instances.create",
    "compute.instances.delete",
    "compute.instances.get",
    "compute.disks.delete",
)


class DriverError(RuntimeError):
    pass


class IamGateError(DriverError):
    pass


class ScheduleError(DriverError):
    pass


class UncertainMutation(DriverError):
    """A mutating provider call neither confirmed nor failed: stop, no replay."""

    def __init__(self, op, identity, detail=""):
        super().__init__(f"uncertain mutation {op} on {identity}: {detail}")
        self.op = op
        self.identity = identity
        self.detail = detail


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value) -> bytes:
    return (json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False)
            + "\n").encode()


def write_receipt(path: Path, value: dict) -> str:
    raw = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        fh.write(raw)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return sha_bytes(raw)


# --------------------------------------------------------------------------
# Schedule (pure; no clock reads inside beyond the passed now for admission).
# --------------------------------------------------------------------------

def compute_schedule(helper_epoch_H: int, target_arm_epoch_A: int) -> dict:
    """Derive the fixed lifecycle schedule; raise unless A<=H+210."""
    H = int(helper_epoch_H)
    A = int(target_arm_epoch_A)
    if not A <= H + A_LE_H_PLUS:
        raise ScheduleError(
            f"helper tail violated: A={A} > H+210={H + A_LE_H_PLUS} "
            f"(A+1800+600 must precede helper termination)")
    return {
        "schema": SCHEMA,
        "helper_epoch_H": H,
        "target_arm_epoch_A": A,
        "target_cutoff": A + TARGET_CUTOFF_OFFSET,
        "target_budget_end": A + TARGET_BUDGET_END_OFFSET,
        "helper_delete_at": H + HELPER_DELETE_OFFSET,
        "helper_budget_end": H + HELPER_BUDGET_END_OFFSET,
    }


def admission_open(schedule: dict, now_epoch: float) -> bool:
    """Work may be admitted only strictly before the fixed cutoff."""
    return now_epoch < schedule["target_cutoff"]


# --------------------------------------------------------------------------
# Provider seam. Real GCP implementation imports SDKs lazily and refuses
# without a fresh owner IAM binding receipt (fail-closed, offline-safe).
# --------------------------------------------------------------------------

class ComputeProvider:
    """Minimal surface the driver needs; implemented by GCP or fakes."""

    def create_instance(self, name: str, spec: dict) -> dict:
        raise NotImplementedError

    def get_instance(self, name: str) -> dict:
        """Return {'present': bool, ...}; absence only via present False."""
        raise NotImplementedError

    def delete_instance(self, name: str) -> dict:
        """Return {'accepted': bool, ...}; acceptance is not absence proof."""
        raise NotImplementedError

    def get_disk(self, name: str) -> dict:
        raise NotImplementedError

    def exec_command(self, name: str, argv: list, timeout_s: float) -> dict:
        raise NotImplementedError


def check_iam_binding(receipt_path: str | Path) -> dict:
    """Validate the owner IAM binding receipt (exact-resource, fresh).

    Raises IamGateError on any defect: missing file, missing permissions,
    wrong scope, or expiry. Never touches the provider.
    """
    path = Path(receipt_path)
    if not path.is_file():
        raise IamGateError(f"IAM binding receipt absent: {path}")
    try:
        binding = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise IamGateError(f"IAM binding receipt unreadable: {exc}")
    missing = [p for p in REQUIRED_IAM_PERMISSIONS
               if p not in binding.get("permissions", [])]
    if missing:
        raise IamGateError(f"IAM binding lacks permissions: {missing}")
    if binding.get("project") != ENVELOPE["project"]:
        raise IamGateError(
            f"IAM binding project mismatch: {binding.get('project')!r}")
    valid_until = binding.get("valid_until_epoch")
    if not isinstance(valid_until, (int, float)) or time.time() >= valid_until:
        raise IamGateError(f"IAM binding missing or expired: {valid_until!r}")
    return binding


class GcpComputeProvider(ComputeProvider):
    """Live GCP provider. Construction validates the owner IAM binding
    receipt FIRST (no SDK import, no provider call without it)."""

    def __init__(self, iam_binding_receipt: str | Path):
        self.binding = check_iam_binding(iam_binding_receipt)
        self._clients = None

    def _sdk(self):
        if self._clients is None:
            try:
                from google.cloud import compute_v1  # lazy: offline-safe
            except ImportError as exc:
                raise DriverError(
                    "google-cloud-compute not installed; cannot bind live "
                    f"provider: {exc}")
            self._clients = compute_v1
        return self._clients

    def create_instance(self, name: str, spec: dict) -> dict:
        compute_v1 = self._sdk()
        raise DriverError("live create not executed from this checkout without "
                          "a sealed launch (sad path kept unimplemented)")

    def get_instance(self, name: str) -> dict:
        self._sdk()
        raise DriverError("live GET not executed without a sealed launch")

    def delete_instance(self, name: str) -> dict:
        self._sdk()
        raise DriverError("live DELETE not executed without a sealed launch")

    def get_disk(self, name: str) -> dict:
        self._sdk()
        raise DriverError("live disk GET not executed without a sealed launch")

    def exec_command(self, name: str, argv: list, timeout_s: float) -> dict:
        raise DriverError("live exec not executed without a sealed launch")


class FakeComputeProvider(ComputeProvider):
    """In-memory provider double for offline tests and dry runs."""

    def __init__(self):
        self.calls = []
        self.instances: dict[str, dict] = {}
        self.disks: dict[str, dict] = {}
        self.uncertain_ops: set[tuple[str, str]] = set()

    def _maybe_uncertain(self, op, name):
        if (op, name) in self.uncertain_ops:
            raise UncertainMutation(op, name, "scripted uncertainty")

    def create_instance(self, name: str, spec: dict) -> dict:
        self.calls.append(("create_instance", name))
        self._maybe_uncertain("create_instance", name)
        self.instances[name] = {"present": True, "spec": dict(spec)}
        return {"present": True, "name": name}

    def get_instance(self, name: str) -> dict:
        self.calls.append(("get_instance", name))
        state = self.instances.get(name, {"present": False})
        return {"present": bool(state.get("present")), "name": name}

    def delete_instance(self, name: str) -> dict:
        self.calls.append(("delete_instance", name))
        self._maybe_uncertain("delete_instance", name)
        return {"accepted": True, "name": name}

    def effect_delete(self, name: str):
        """Test-only: make a prior accepted DELETE take effect."""
        if name in self.instances:
            self.instances[name]["present"] = False

    def get_disk(self, name: str) -> dict:
        self.calls.append(("get_disk", name))
        state = self.disks.get(name, {"present": False})
        return {"present": bool(state.get("present")), "name": name}

    def exec_command(self, name: str, argv: list, timeout_s: float) -> dict:
        self.calls.append(("exec_command", name, list(argv)))
        return {"exit": 0, "name": name}


# --------------------------------------------------------------------------
# Watchdog: issues the exact early owned DELETEs at fixed epochs.
# --------------------------------------------------------------------------

class Watchdog:
    """Owned watchdog. due(now) lists DELETEs owed; fire() issues them with
    intent/result receipts. Separate from the supervision loop so tests can
    drive it on a fake clock."""

    def __init__(self, schedule: dict, target_name: str, helper_name: str,
                 receipts_dir: Path):
        self.schedule = schedule
        self.target_name = target_name
        self.helper_name = helper_name
        self.receipts_dir = Path(receipts_dir)
        self.fired: list[str] = []

    def due(self, now_epoch: float) -> list[str]:
        owed = []
        if (now_epoch >= self.schedule["target_cutoff"]
                and "target" not in self.fired):
            owed.append("target")
        if (now_epoch >= self.schedule["helper_delete_at"]
                and "helper" not in self.fired):
            owed.append("helper")
        return owed

    def fire(self, provider: ComputeProvider, which: str, now_epoch: float) -> dict:
        name = self.target_name if which == "target" else self.helper_name
        intent = {"schema": SCHEMA, "event": "delete_intent", "which": which,
                  "identity": name, "at_epoch": now_epoch,
                  "fixed_offset_epoch": (self.schedule["target_cutoff"]
                                         if which == "target"
                                         else self.schedule["helper_delete_at"])}
        write_receipt(self.receipts_dir / f"delete_intent_{which}.json", intent)
        try:
            result = provider.delete_instance(name)
        except UncertainMutation as exc:
            write_receipt(self.receipts_dir / f"delete_result_{which}.json",
                          {"schema": SCHEMA, "event": "delete_uncertain",
                           "which": which, "identity": name,
                           "detail": str(exc)})
            raise
        write_receipt(self.receipts_dir / f"delete_result_{which}.json",
                      {"schema": SCHEMA, "event": "delete_accepted",
                       "which": which, "identity": name,
                       "accepted": bool(result.get("accepted"))})
        self.fired.append(which)
        return result


# --------------------------------------------------------------------------
# Mission: bounded supervision with the amendment gate semantics.
# --------------------------------------------------------------------------

MISSION_END = max(TARGET_BUDGET_END_OFFSET, HELPER_BUDGET_END_OFFSET)


def observe_absence(provider: ComputeProvider, schedule: dict,
                    target_name: str, helper_name: str,
                    target_disks: list[str], helper_disks: list[str],
                    now_epoch: float) -> dict:
    """Separate GET observations per identity; only present False is absence."""
    obs: dict[str, dict] = {}
    for label, name in (("target_vm", target_name), ("helper_vm", helper_name)):
        try:
            seen = provider.get_instance(name)
            obs[label] = {"identity": name, "present": bool(seen.get("present")),
                          "at_epoch": now_epoch}
        except Exception as exc:  # observation failure is not absence
            obs[label] = {"identity": name, "present": None,
                          "observation_error": f"{type(exc).__name__}: {exc}",
                          "at_epoch": now_epoch}
    for label, name in ([(f"target_disk_{i}", d) for i, d in enumerate(target_disks)]
                        + [(f"helper_disk_{i}", d) for i, d in enumerate(helper_disks)]):
        try:
            seen = provider.get_disk(name)
            obs[label] = {"identity": name, "present": bool(seen.get("present")),
                          "at_epoch": now_epoch}
        except Exception as exc:
            obs[label] = {"identity": name, "present": None,
                          "observation_error": f"{type(exc).__name__}: {exc}",
                          "at_epoch": now_epoch}
    return obs


def classify_cleanup(obs: dict, budget_end_epoch: float,
                     observed_epoch: float) -> str:
    """CLEAN only if every identity observed absent within budget;
    CLEANUP_LATE if absence observed after budget end; else UNVERIFIED."""
    states = [v.get("present") for v in obs.values()]
    if all(s is False for s in states):
        if observed_epoch <= budget_end_epoch:
            return "CLEAN"
        return "CLEANUP_LATE"
    return "CLEANUP_UNVERIFIED"


def run_mission(provider: ComputeProvider, schedule: dict, clock,
                receipts_dir: Path, target_name: str, helper_name: str,
                target_disks: list[str], helper_disks: list[str],
                capsule_runner=None, poll_s: float = 5.0,
                mission_end_offset: float = MISSION_END) -> dict:
    """Bounded supervision. capsule_runner, if given, is called at most once
    per capsule and only while admission is open; it must be a callable
    (capsule_id) -> dict. Returns the terminal mission receipt (also written).

    No retries: an UncertainMutation stops the mission immediately with the
    uncertainty recorded (no replay). Clocks are read, never rebased: the
    schedule epochs are immutable inputs.
    """
    receipts = Path(receipts_dir)
    watchdog = Watchdog(schedule, target_name, helper_name, receipts)
    end_epoch = schedule["target_arm_epoch_A"] + mission_end_offset
    budget_end = max(schedule["target_budget_end"],
                     schedule["helper_budget_end"])
    events: list[dict] = []

    def record(event: str, **fields):
        entry = {"schema": SCHEMA, "event": event,
                 "at_epoch": clock(), **fields}
        events.append(entry)
        return entry

    record("mission_start", schedule=schedule, target=target_name,
           helper=helper_name)

    # Phase 1: create (intent receipt before each mutation).
    for which, name, spec in (("target", target_name, ENVELOPE["target"]),
                              ("helper", helper_name, ENVELOPE["helper"])):
        write_receipt(receipts / f"create_intent_{which}.json",
                      {"schema": SCHEMA, "event": "create_intent",
                       "which": which, "identity": name, "spec": spec})
        try:
            created = provider.create_instance(name, spec)
        except UncertainMutation as exc:
            record("mission_stop_uncertain", op=exc.op, identity=exc.identity)
            terminal = {"schema": SCHEMA, "status": "STOP_UNCERTAIN_NO_REPLAY",
                        "events": events}
            write_receipt(receipts / "mission_terminal.json", terminal)
            return terminal
        write_receipt(receipts / f"create_result_{which}.json",
                      {"schema": SCHEMA, "event": "create_result",
                       "which": which, "identity": name,
                       "present": bool(created.get("present"))})
        record("created", which=which, identity=name)

    # Phase 2: bounded work admission (strictly before cutoff).
    admitted: list[str] = []
    if capsule_runner is not None:
        for capsule_id in capsule_runner.capsules():
            now = clock()
            if not admission_open(schedule, now):
                record("admission_refused_cutoff", capsule=capsule_id,
                       cutoff=schedule["target_cutoff"])
                break
            result = capsule_runner(capsule_id)
            admitted.append(capsule_id)
            record("capsule_done", capsule=capsule_id,
                   ok=bool(result.get("ok")))
    else:
        record("no_capsule_runner_bound")

    # Phase 3: supervise to mission end; watchdog fires exact DELETEs.
    last_obs: dict = {}
    last_obs_epoch = clock()
    while clock() < end_epoch:
        now = clock()
        for which in watchdog.due(now):
            try:
                watchdog.fire(provider, which, now)
                record("early_delete_issued", which=which)
            except UncertainMutation as exc:
                record("mission_stop_uncertain", op=exc.op,
                       identity=exc.identity)
                terminal = {"schema": SCHEMA,
                            "status": "STOP_UNCERTAIN_NO_REPLAY",
                            "events": events}
                write_receipt(receipts / "mission_terminal.json", terminal)
                return terminal
        last_obs = observe_absence(provider, schedule, target_name,
                                   helper_name, target_disks, helper_disks,
                                   now)
        last_obs_epoch = now
        if all(v.get("present") is False for v in last_obs.values()):
            record("absence_observed", identities=sorted(last_obs))
            break
        time.sleep(poll_s)

    verdict = classify_cleanup(last_obs, budget_end, last_obs_epoch)
    write_receipt(receipts / "absence_observations.json",
                  {"schema": SCHEMA, "observations": last_obs,
                   "observed_epoch": last_obs_epoch})
    record("mission_end", verdict=verdict, admitted=admitted)
    if verdict != "CLEAN":
        record("root_notified",
               note="unknown/late cleanup requires separate root "
                    "authorization for any intervention; no replay, "
                    "no extension, no continued experiments")
    terminal = {"schema": SCHEMA,
                "status": ("COMPLETE_CLEAN" if verdict == "CLEAN"
                           else "COMPLETE_" + verdict),
                "verdict": verdict, "events": events}
    write_receipt(receipts / "mission_terminal.json", terminal)
    return terminal


# --------------------------------------------------------------------------
# Validation (free) and CLI.
# --------------------------------------------------------------------------

def validate_offline() -> dict:
    """Zero-spend, zero-provider validation: amendment record + schedule math
    + envelope + harness/capsule presence."""
    findings: list[str] = []
    amendment = FIN / "amendment_acceptance_2026-09-19.json"
    if not amendment.is_file():
        findings.append("missing amendment acceptance record")
    harness = FIN / "native-harness/main.py"
    if not harness.is_file():
        findings.append("missing surviving harness entry")
    bench = FIN / "native-harness/benchmark/benchmark.py"
    if not bench.is_file():
        findings.append("missing surviving benchmark module")
    try:
        compute_schedule(1_000_000, 1_000_100)
    except ScheduleError as exc:
        findings.append(f"schedule math broken: {exc}")
    try:
        compute_schedule(1_000_000, 1_000_000 + A_LE_H_PLUS + 1)
        findings.append("A<=H+210 constraint not enforced")
    except ScheduleError:
        pass
    return {"schema": SCHEMA, "ok": not findings, "findings": findings}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="lead-only: run the mission (requires --iam-binding, --request, --reservations)")
    parser.add_argument("--iam-binding", default=None)
    parser.add_argument("--request", default=None)
    parser.add_argument("--helper-epoch", type=int, default=None)
    parser.add_argument("--target-epoch", type=int, default=None)
    parser.add_argument("--receipts-dir", default="outputs/e2_driver_run")
    args = parser.parse_args(argv)

    report = validate_offline()
    print(json.dumps(report, indent=1))
    if not report["ok"]:
        return 1
    if not args.execute:
        print("validation only: no provider calls, no spend. "
              "--execute is lead-only and needs --iam-binding + --request.")
        return 0
    if not args.iam_binding or not args.request:
        print("refusing: --execute requires --iam-binding and --request",
              file=sys.stderr)
        return 2
    try:
        check_iam_binding(args.iam_binding)
    except IamGateError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2
    try:
        request = json.loads(Path(args.request).read_text())
    except (OSError, ValueError) as exc:
        print(f"refusing: sealed request unreadable: {exc}", file=sys.stderr)
        return 2
    if not (request.get("status") == "SEALED_LAUNCH_AUTHORIZED"
            and request.get("launch_authorized") is True):
        print("refusing: request is not a lead-sealed launch authorization "
              "(drafts and unallocated requests never authorize dispatch)",
              file=sys.stderr)
        return 2
    print("gates checked; live dispatch stays lead-operated from here.",
          file=sys.stderr)
    return 3


if __name__ == "__main__":
    sys.exit(main())
