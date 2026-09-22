"""Shared support for the E1 wave10 process-group regression tests.

Rewritten from the lost targeted suite recorded in
outputs/PES_Phase2_Review_2026-09-12/finish/e9_completion/e1_v6_rpc_review01/targeted_tests.log
(ten tests, all passing pre-loss) against the contract recorded in
outputs/PES_Phase2_Review_2026-09-12/finish/e1_recovery_v6/REVIEW_PACKET.md
and cleanup_contract_v6.json.

Expected module under test: bounded_procgroup.py in this directory (sibling
member e1-driver builds it in parallel).  The import below is tolerant: if the
module is absent, every test that needs it reports unittest.SkipTest with the
import error instead of erroring.

Contract surface expected from bounded_procgroup (behavior, not exact names;
the adapter below maps common spellings):

  run_child(command, result_dir, timeout) -> outcome
      - spawns the worker argv as a NEW process group (worker is its own group
        leader: pgid == pid, as recorded in the fixture process.json files)
      - result_dir receives the durable controller artifacts
        process.json / worker.log / group_finalization.json (+ unknown.json on
        any UNKNOWN outcome); the worker itself owns result.json / child.pid
      - timeout is the controller work-phase deadline in seconds (absolute
        epoch converted once; includes the unchanged 2s reap reserve)
      - on success AND failure it unconditionally finalizes the original owned
        PGID: probe (killpg(pgid,0)), SIGKILL if present, reap the direct
        worker, independently wait for killpg(pgid,0) -> ESRCH, then persist
        group_finalization.json
      - success requires worker exit0 + worker reaped + verified original-group
        absence + remaining controller deadline; result.json is read only after
        group absence and a durable finalization receipt
      - nonzero worker, missing result, failed observation, permission error,
        group still present, or finalization over deadline -> UNKNOWN /
        no-replay (unknown.json retained); no automatic retry

  reconcile_ambiguous_create(...)  [SDKWallTests only]
      - exact app/name/image/reservation-tag reconciliation only; description
        matches, deployed/queued apps, ambiguity and named-lookup absence are
        UNKNOWN, never proof of no delayed creation

Everything in this suite drives REAL subprocesses with REAL process-group
semantics; os.killpg itself is never mocked.  The only fault injection is the
unresolved-verification seam (see force_unresolved_verification), which
replaces the module's own post-SIGKILL ESRCH *wait helper* so the bounded wait
reports the group still present -- SIGKILL, reaping and killpg probes remain
real syscalls.  A user process cannot survive SIGKILL, so no fully-real
offline construction can exercise the "group remains present" branch.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
import time
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:  # tolerant import of the sibling module under test
    bounded_procgroup = importlib.import_module("bounded_procgroup")
    IMPORT_ERROR = None
except BaseException as exc:  # noqa: BLE001 - record any import failure
    bounded_procgroup = None
    IMPORT_ERROR = exc


def require_module(test: unittest.TestCase):
    """Skip *test* when bounded_procgroup is absent (skip-if-import-fails)."""
    if bounded_procgroup is None:
        test.skipTest(
            "bounded_procgroup not importable from {}: {!r}".format(HERE, IMPORT_ERROR)
        )
    return bounded_procgroup


# --------------------------------------------------------------------------
# Adapter for run_child(): primary binding targets the real bounded_procgroup
# signature (argv, *, result_path, receipt_dir, deadline_epoch, ...); a
# generic fallback maps other spellings.  Anything unmappable fails loudly.
# --------------------------------------------------------------------------

_COMMAND_KEYS = ("command", "cmd", "argv", "args")
_DIR_KEYS = ("result_dir", "work_dir", "exec_dir", "execution_dir", "cwd", "dir")
_TIMEOUT_KEYS = ("timeout", "timeout_seconds", "deadline_seconds", "work_seconds", "seconds")


def call_run_child(test, command, result_path, receipt_dir, timeout):
    """Invoke bounded_procgroup.run_child.

    result_path: file the worker is expected to write (its result.json);
    receipt_dir: directory receiving the controller's durable receipts
    (group_finalization.json / unknown.json).  timeout: work+cleanup budget in
    seconds, converted here to the absolute deadline_epoch the contract
    requires ("absolute epoch converted once").
    """
    require_module(test)
    fn = getattr(bounded_procgroup, "run_child", None)
    if not callable(fn):
        test.fail(
            "bounded_procgroup is importable but exposes no callable run_child "
            "(found: {}); contract requires a run_child entry point".format(_public_names())
        )
    command = list(command)
    deadline_epoch = time.time() + float(timeout)
    try:  # primary: the implemented v6 signature
        return fn(
            command,
            result_path=Path(result_path),
            receipt_dir=Path(receipt_dir),
            deadline_epoch=deadline_epoch,
        )
    except TypeError:
        pass
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        sig = None
    if sig is not None:
        names = set(sig.parameters)
        if "result_path" in names or "receipt_dir" in names or "deadline_epoch" in names:
            test.fail(
                "run_child declares v6-style parameters but rejected the v6-style "
                "call; signature {}".format(sig)
            )
        kwargs, unmapped = _map_kwargs(sig.parameters, command, str(result_path), timeout)
        if unmapped:
            test.fail(
                "cannot map run_child signature {}; unmapped: {}".format(sig, ", ".join(unmapped))
            )
        return fn(**kwargs)
    return fn(command, str(result_path), timeout)


def _map_kwargs(sig_params, command, result_dir, timeout):
    """Return (kwargs, unmapped) for a legacy run_child-style signature."""
    names = list(sig_params)
    kwargs = {}
    unmapped = []

    def pick(keys, value, label):
        for key in keys:
            if key in names:
                kwargs[key] = value
                return
        positional = [
            p
            for p in sig_params.values()
            if p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        index = len(kwargs)
        if index < len(positional) and positional[index].default is inspect.Parameter.empty:
            kwargs[positional[index].name] = value
            return
        unmapped.append(label)

    pick(_COMMAND_KEYS, command, "command")
    pick(_DIR_KEYS, str(result_dir), "result_dir")
    pick(_TIMEOUT_KEYS, timeout, "timeout")
    return kwargs, unmapped


def _public_names():
    if bounded_procgroup is None:
        return "<absent>"
    return sorted(n for n in dir(bounded_procgroup) if not n.startswith("__"))


# --------------------------------------------------------------------------
# Outcome normalization: accept dict- or attribute-shaped results.
# --------------------------------------------------------------------------

def _dig(outcome, names, default=None):
    for name in names:
        if isinstance(outcome, dict):
            if name in outcome:
                return outcome[name]
        else:
            value = getattr(outcome, name, None)
            if value is not None:
                return value
    return default


def outcome_status(outcome) -> str:
    raw = _dig(outcome, ("status", "outcome", "state", "disposition", "result_status"))
    if isinstance(raw, str):
        return raw.strip().lower()
    if _dig(outcome, ("ok", "success")) is True:
        return "success"
    if _dig(outcome, ("unknown", "replay_forbidden")) is True:
        return "unknown"
    return ""


def is_success(outcome) -> bool:
    return outcome_status(outcome) in ("success", "ok", "succeeded", "complete", "completed")


def is_unknown(outcome) -> bool:
    status = outcome_status(outcome)
    return ("unknown" in status) or ("uncertain" in status)


def outcome_result(outcome):
    return _dig(outcome, ("result", "value", "payload", "result_value", "returned"))


def outcome_finalization(outcome):
    return _dig(outcome, ("group_finalization", "finalization", "group_finalization_receipt"))


def outcome_reaped(outcome) -> bool:
    value = _dig(outcome, ("reaped", "worker_reaped"))
    return bool(value)


def outcome_returncode(outcome):
    value = _dig(
        outcome,
        ("worker_returncode", "returncode", "worker_exit", "exit_code", "worker_rc"),
    )
    return value if isinstance(value, int) else None


# --------------------------------------------------------------------------
# Receipts on disk (durable artifacts the contract requires).
# --------------------------------------------------------------------------

def load_json(test, path: Path):
    if not path.exists():
        test.fail("required durable artifact missing: {}".format(path))
    return json.loads(path.read_text())


FINALIZATION_CORE_KEYS = (
    "pgid",
    "worker_reaped",
    "worker_returncode",
    "group_sigkill_sent",
    "original_group_absent",
    "verified",
    "errors",
    "observed_at",
)


def assert_finalization_receipt(test, receipt: dict, *, expected_returncode):
    for key in FINALIZATION_CORE_KEYS:
        test.assertIn(key, receipt, "group_finalization receipt lacks key {!r}: {}".format(key, receipt))
    test.assertEqual(
        receipt["worker_returncode"],
        expected_returncode,
        "receipt must record the direct worker returncode",
    )
    test.assertTrue(receipt["worker_reaped"], "direct worker must be reaped: {}".format(receipt))
    test.assertEqual(receipt["errors"], [], "no finalization errors expected on this path")
    provider_verified = receipt.get("provider_cleanup_verified")
    if provider_verified is not None:
        test.assertFalse(
            provider_verified,
            "offline run must not claim provider cleanup verification",
        )


# --------------------------------------------------------------------------
# Real process-group observation helpers (no mocks of kill/killpg anywhere).
# --------------------------------------------------------------------------

def group_absent(pgid: int, timeout: float = 5.0) -> bool:
    """Poll killpg(pgid, 0) until it raises ESRCH (group truly absent)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass  # group exists (zombie members included) but is not signalable
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def pid_dead(pid: int, timeout: float = 5.0) -> bool:
    """Poll kill(pid, 0) until ESRCH (process exited AND was reaped)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def kill_group_hard(pgid: int):
    """Safety net used via addCleanup so a failing assertion cannot leak a group."""
    try:
        os.killpg(pgid, 9)
    except (ProcessLookupError, PermissionError):
        pass
    group_absent(pgid, timeout=3.0)


# --------------------------------------------------------------------------
# Real worker programs.  Each mirrors the sealed fixtures: a worker that
# spawns a SIGTERM-ignoring same-PGID sleeper and then exits 0/7, or hangs in
# one of the reported blocking SDK shapes.  The child stays in the worker's
# process group (no setsid/setpgid in the child), so only a real group kill
# reaps it.
# --------------------------------------------------------------------------

SLEEPING_SAME_GROUP_CHILD = (
    "import signal, time; "
    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "time.sleep(60)"
)

WORKER_EXIT_TEMPLATE = r"""
import json, os, subprocess, sys
from pathlib import Path

d = Path(sys.argv[1])
exit_code = int(sys.argv[2])
p = subprocess.Popen(
    [sys.executable, "-c", sys.argv[3]],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
(d / "child.pid").write_text(str(p.pid))
os.kill(p.pid, 0)  # liveness proof: child alive, in our group, right now
before = {
    "worker_pid": os.getpid(),
    "worker_pgid": os.getpgid(0),
    "child_pid": p.pid,
    "child_pgid": os.getpgid(p.pid),
    "child_alive": True,
    "group_present": True,
}
(d / "t_before.json").write_text(json.dumps(before))
(d / "result.json").write_text(json.dumps({"returned": True}))
sys.exit(exit_code)
"""


def worker_exit_argv(result_dir: Path, exit_code: int):
    """Exit-0/exit-7 worker with a live same-PGID child (fixtures exit_0/exit_7)."""
    return [
        sys.executable,
        "-B",
        "-c",
        WORKER_EXIT_TEMPLATE,
        str(result_dir),
        str(exit_code),
        SLEEPING_SAME_GROUP_CHILD,
    ]


def worker_hang_argv(shape: str, result_dir: Path):
    """Hung fake-SDK workers for the six reported blocking paths."""
    d = str(result_dir)
    if shape == "hang_immediately":
        code = "import time; time.sleep(60)"
        return [sys.executable, "-B", "-c", code, d]
    if shape == "hang_after_result":
        code = (
            "import json, sys, time; from pathlib import Path; "
            "Path(sys.argv[1], 'result.json').write_text(json.dumps({'returned': True})); "
            "time.sleep(60)"
        )
        return [sys.executable, "-B", "-c", code, d]
    if shape == "hang_with_same_group_child":
        code = (
            "import json, os, subprocess, sys, time; from pathlib import Path; "
            "d = Path(sys.argv[1]); "
            "p = subprocess.Popen([sys.executable, '-c', "
            + repr(SLEEPING_SAME_GROUP_CHILD) +
            "], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
            "(d / 'child.pid').write_text(str(p.pid)); "
            "os.kill(p.pid, 0); "
            "(d / 't_before.json').write_text(json.dumps({'worker_pid': os.getpid(), "
            "'worker_pgid': os.getpgid(0), 'child_pid': p.pid, 'child_pgid': os.getpgid(p.pid), "
            "'child_alive': True, 'group_present': True})); "
            "time.sleep(60)"
        )
        return [sys.executable, "-B", "-c", code, d]
    if shape == "hang_ignoring_sigterm":
        code = (
            "import signal, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "time.sleep(60)"
        )
        return [sys.executable, "-B", "-c", code, d]
    if shape == "hang_blocked_on_fifo":
        code = (
            "import os, sys; from pathlib import Path; "
            "fifo = Path(sys.argv[1], 't_block.fifo'); "
            "fifo.exists() or os.mkfifo(fifo); "
            "fd = os.open(fifo, os.O_RDONLY); os.close(fd)"
        )
        return [sys.executable, "-B", "-c", code, d]
    if shape == "hang_busy_loop":
        code = "import sys\nwhile True:\n    pass"
        return [sys.executable, "-B", "-c", code, d]
    if shape == "guard_invocation_once":
        code = (
            "import sys, time; from pathlib import Path; "
            "d = Path(sys.argv[1]); "
            "(d / 't_invocations.log').open('a').write('start %d\\n' % __import__('os').getpid()); "
            "time.sleep(60)"
        )
        return [sys.executable, "-B", "-c", code, d]
    if shape == "hang_after_revoking_work_disk":
        code = (
            "import os, subprocess, sys, time; from pathlib import Path; "
            "d = Path(sys.argv[1]); "
            "p = subprocess.Popen([sys.executable, '-c', "
            + repr(SLEEPING_SAME_GROUP_CHILD) +
            "], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
            "(d / 'child.pid').write_text(str(p.pid)); "
            "os.kill(p.pid, 0); "
            "os.chmod(d, 0o500); "  # any work-disk guard write now fails
            "time.sleep(60)"
        )
        return [sys.executable, "-B", "-c", code, d]
    raise ValueError("unknown hang shape: {}".format(shape))


BLOCKING_SDK_SHAPES = (
    "hang_immediately",
    "hang_after_result",
    "hang_with_same_group_child",
    "hang_ignoring_sigterm",
    "hang_blocked_on_fifo",
    "hang_busy_loop",
)


# --------------------------------------------------------------------------
# Unresolved-verification fault: the ONLY injected fault in this suite.
# It replaces bounded_procgroup's own post-SIGKILL "wait for killpg(pgid,0)
# ESRCH" helper so the bounded wait reports the group as still present.
# probe/SIGKILL/reap stay real; os.killpg itself is never replaced.
# --------------------------------------------------------------------------

_WAIT_SEAM_CANDIDATES = (
    "wait_group_absent",
    "wait_for_group_absence",
    "wait_for_group_esrch",
    "wait_group_esrch",
    "await_group_absence",
    "verify_group_absent",
    "group_absent",
)

WAIT_SEAM_CANDIDATES = _WAIT_SEAM_CANDIDATES  # public, for skip diagnostics

_FAULT_ENV = {
    "BOUNDED_PROCGROUP_FAULT": "group_absent_unresolved",
    "E1_PROCGROUP_FAULT": "unresolved_group_verification",
}

FAULT_ENV = _FAULT_ENV  # public, for skip diagnostics


class UnresolvedVerificationFault:
    """Context manager applying every applicable fault tier; .applied says what."""

    def __init__(self, test):
        self.test = test
        self.applied = []
        self._undo = []

    def __enter__(self):
        require_module(self.test)
        module = bounded_procgroup
        for name in _WAIT_SEAM_CANDIDATES:
            original = getattr(module, name, None)
            if callable(original) and not isinstance(original, type):
                def stuck_present(*_args, **_kwargs):
                    return False  # group still present within budget

                setattr(module, name, stuck_present)
                self._undo.append((name, original))
                self.applied.append("patched:{}".format(name))
                break
        injector = getattr(module, "inject_fault", None) or getattr(module, "set_fault", None)
        if callable(injector):
            try:
                injector("group_absent_unresolved")
                self.applied.append("inject_fault")
            except Exception:  # noqa: BLE001 - injector is best effort
                pass
        saved_env = {k: os.environ.get(k) for k in _FAULT_ENV}
        os.environ.update(_FAULT_ENV)
        self._undo.append(("__env__", saved_env))
        if not any(a.startswith("patched:") or a == "inject_fault" for a in self.applied):
            # No deterministic seam available; env tiers alone are hopeful at
            # best (they only work if the module reads them per call).
            self.applied.append("env-only")
        return self

    def __exit__(self, *_exc):
        for key, value in reversed(self._undo):
            if key == "__env__":
                for env_key, old in value.items():
                    if old is None:
                        os.environ.pop(env_key, None)
                    else:
                        os.environ[env_key] = old
            else:
                setattr(bounded_procgroup, key, value)
        return False


def deterministic_fault_applied(fault: UnresolvedVerificationFault) -> bool:
    return any(a.startswith("patched:") or a == "inject_fault" for a in fault.applied)


# --------------------------------------------------------------------------
# Ambiguous-create reconciliation helper (SDKWallTests).
# --------------------------------------------------------------------------

_RECONCILE_NAMES = ("reconcile_ambiguous_create", "reconcile_create", "reconcile_ambiguous")
_PERSISTED_KEYS = ("persisted", "persisted_identity", "identity", "expected", "recorded")
_CANDIDATE_KEYS = ("candidates", "observed", "apps", "observed_apps", "provider_apps")


NO_RECONCILE_ENTRY = object()


def reconcile_entry():
    """Return the reconciliation entry point, or NO_RECONCILE_ENTRY."""
    if bounded_procgroup is None:
        return NO_RECONCILE_ENTRY
    for name in _RECONCILE_NAMES:
        fn = getattr(bounded_procgroup, name, None)
        if callable(fn):
            return fn
    return NO_RECONCILE_ENTRY


def call_reconcile(test, persisted: dict, candidates: list):
    """Call the reconciliation entry point tolerantly; NO_RECONCILE_ENTRY if absent."""
    require_module(test)
    fn = reconcile_entry()
    if fn is NO_RECONCILE_ENTRY:
        return NO_RECONCILE_ENTRY
    try:
        sig = inspect.signature(fn)
        names = set(sig.parameters)
        persisted_key = next((k for k in _PERSISTED_KEYS if k in names), None)
        candidates_key = next((k for k in _CANDIDATE_KEYS if k in names), None)
        if persisted_key and candidates_key:
            return fn(**{persisted_key: persisted, candidates_key: candidates})
        return fn(persisted, candidates)
    except TypeError:
        return fn(persisted, candidates)


def reconcile_says_reconciled(test, result) -> bool:
    """Normalize a reconciliation verdict; fail loudly on alien shapes."""
    if isinstance(result, bool):
        return result
    if isinstance(result, str):
        return result.strip().lower() in ("reconciled", "matched", "exact", "ok", "true")
    if isinstance(result, dict):
        for key in ("reconciled", "matched", "exact_match", "ok", "is_exact_match"):
            if key in result:
                return bool(result[key])
        status = result.get("status")
        if isinstance(status, str):
            return status.strip().lower() in ("reconciled", "matched", "exact")
    if result is None:
        return False
    test.fail("unrecognized reconciliation verdict shape: {!r}".format(result))
    return False
