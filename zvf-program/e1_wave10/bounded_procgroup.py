"""Bounded process-group child execution implementing the E1 v6 contract.

Spec sources (offline, read before editing):
  outputs/PES_Phase2_Review_2026-09-12/finish/e1_recovery_v6/REVIEW_PACKET.md
  outputs/PES_Phase2_Review_2026-09-12/finish/e1_recovery_v6/cleanup_contract_v6.json
  outputs/PES_Phase2_Review_2026-09-12/finish/e1_recovery_v6/fixture_exit_0/group_finalization.json

Contract summary. ``run_child`` spawns a worker in its own process group
(``start_new_session``). After the worker phase ends -- success OR failure --
it unconditionally runs ``finalize_group``: probe the original owned PGID,
SIGKILL it if present, reap the direct worker, then independently wait until
``killpg(pgid, 0)`` raises ``ProcessLookupError`` (ESRCH). The worker's result
JSON is read ONLY after the group is absent and the finalization receipt is
durable on disk. Success requires worker exit 0, direct-worker reaping,
verified original-group absence, and remaining controller deadline.
Permission/observation failures, a surviving group, or a deadline overrun
produce UNKNOWN with no replay and no returned result. Nonzero workers stay
UNKNOWN even when their group is successfully cleaned. Both success and
failure paths write ``group_finalization.json``; every UNKNOWN path also
writes ``unknown.json``.

Scope boundary (unchanged from v6): finalization establishes absence of the
original owned process group only. It does not claim containment of
descendants that deliberately create new sessions, remote RPC cancellation,
or provider resource deletion.
"""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

REAP_RESERVE_SECONDS = 2.0
FINALIZATION_POLL_SECONDS = 0.05
# Post-deadline finalization grace, fixed once per call and never renewed
# (cleanup_deadline_renewal: false): cleanup_contract_v6.json
# sdk_controller.total_cleanup_controller_seconds.
CLEANUP_GRACE_SECONDS = 60.0
GROUP_FINALIZATION_NAME = "group_finalization.json"
UNKNOWN_NAME = "unknown.json"
PROCESS_NAME = "process.json"
WORKER_LOG_NAME = "worker.log"
SCOPE_NOTE = ("original owned PGID only; descendants creating new sessions "
              "are not claimed contained")


class UnknownOutcome(Exception):
    """Raised internally; every raise site also produces durable receipts."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def digest(raw: bytes) -> str:
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n"


def write_durable(path: Path, value: Any) -> bytes:
    """Write JSON durably: temp file, fsync, atomic rename, fsync directory."""
    raw = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return raw


def probe_group(pgid: int) -> str:
    """Classify the original owned PGID without signals: absent/present/error."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return "absent"
    except PermissionError:
        return "permission_denied"
    except OSError as exc:
        return f"error:{exc.errno or errno.EIO}"
    return "present"


def _exitcode_from_status(status: int) -> int:
    code = os.waitstatus_to_exitcode(status)
    # waitstatus_to_exitcode returns negative signal numbers; normalize to the
    # observable nonzero convention so signal deaths can never masquerade as 0.
    return code if code >= 0 else 128 - code


def reap_worker(pid: int, *, deadline_mono: float,
                poll_seconds: float) -> tuple[int | None, str | None]:
    """Reap the direct worker with os.waitpid; return (exitcode, error)."""
    while True:
        try:
            waited_pid, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            # The child was reaped by someone else or never existed: this
            # controller cannot establish direct-worker reaping.
            return None, "waitpid ECHILD: direct worker not reaped by this controller"
        except OSError as exc:
            return None, f"waitpid error: {exc}"
        if waited_pid == pid:
            return _exitcode_from_status(status), None
        if time.monotonic() >= deadline_mono:
            return None, "reap deadline exceeded before direct worker exited"
        time.sleep(poll_seconds)


def finalize_group(pgid: int | None, worker_pid: int | None, *,
                   deadline_mono: float,
                   poll_seconds: float = FINALIZATION_POLL_SECONDS,
                   pre_reaped: tuple[bool, int | None] = (False, None)) -> dict:
    """Unconditional group finalization; returns the receipt dict.

    Order fixed by the v6 spec: probe original owned PGID, SIGKILL if present,
    reap the direct worker, independently wait for killpg(pgid, 0) ESRCH. The
    deadline is never renewed (cleanup_deadline_renewal: false).

    ``pre_reaped`` carries reaping already established by the caller's worker
    phase (this controller's own waitpid), so the reap step is performed
    exactly once per child; a second waitpid would ECHILD and falsify the
    receipt on every clean success.
    """
    errors: list[str] = []
    group_sigkill_sent = False
    worker_reaped = False
    worker_returncode: int | None = None
    original_group_absent = False

    if pgid is None:
        errors.append("no owned process group: worker spawn never produced a PGID")
    else:
        state = probe_group(pgid)
        if state == "present":
            try:
                os.killpg(pgid, signal.SIGKILL)
                group_sigkill_sent = True
            except ProcessLookupError:
                # Raced out between probe and kill; absence check below decides.
                pass
            except PermissionError:
                errors.append("SIGKILL permission denied on owned process group")
            except OSError as exc:
                errors.append(f"SIGKILL failed: {exc}")
        elif state == "absent":
            pass
        elif state == "permission_denied":
            errors.append("group probe permission denied; presence unobservable")
        else:
            errors.append(f"group probe failed: {state}")

    already_reaped, pre_code = pre_reaped
    if already_reaped:
        worker_reaped = True
        worker_returncode = pre_code
    elif worker_pid is not None:
        code, reap_error = reap_worker(worker_pid, deadline_mono=deadline_mono,
                                       poll_seconds=poll_seconds)
        if reap_error is None:
            worker_reaped = True
            worker_returncode = code
        else:
            errors.append(reap_error)

    if pgid is not None:
        # Independent absence verification, distinct from the kill path above.
        # A False return is classified by one fresh real probe so the receipt
        # records the end state (a patched offline seam reports False while
        # the real group is already gone: still unverified, never success).
        if wait_group_absent(pgid, deadline_mono, poll_seconds):
            original_group_absent = True
        else:
            end_state = probe_group(pgid)
            if end_state == "absent":
                errors.append("group absence verification unresolved within budget")
            elif end_state == "present":
                errors.append("original owned process group still present at "
                              "finalization deadline")
            else:
                errors.append(f"group absence verification failed: {end_state}")

    verified = (not errors and worker_reaped and original_group_absent)
    return {
        "pgid": pgid,
        "worker_reaped": worker_reaped,
        "worker_returncode": worker_returncode,
        "group_sigkill_sent": group_sigkill_sent,
        "original_group_absent": original_group_absent,
        "verified": verified,
        "errors": errors,
        "observed_at": time.time(),
        "scope": SCOPE_NOTE,
        "provider_cleanup_verified": False,
    }


def wait_group_absent(pgid: int, deadline_mono: float,
                      poll_seconds: float = FINALIZATION_POLL_SECONDS
                      ) -> bool:
    """Post-SIGKILL ESRCH wait seam.

    Poll ``killpg(pgid, 0)`` until ESRCH (truly absent) or the non-renewing
    deadline; returns True only on verified absence. A standalone bool
    function so offline tests can force the unresolved-verification branch
    (patched to report present) without touching real syscalls
    (probe/kill/reap stay real); a user process cannot survive SIGKILL, so
    no fully-real construction can exercise that branch.
    """
    while True:
        state = probe_group(pgid)
        if state == "absent":
            return True
        if time.monotonic() >= deadline_mono:
            return False
        if state != "present":
            return False
        time.sleep(poll_seconds)


def reconcile_ambiguous_create(persisted: dict, candidates: list) -> bool:
    """Exact-identity ambiguous-create reconciliation.

    Implements the ``ambiguous_create`` block of cleanup_contract_v6.json:
    exactly one candidate whose app/name, image/image-tag and
    reservation/reservation-tag equal the persisted identity, and whose
    lifecycle state is not deployed/queued, reconciles. Description matches,
    ambiguity (two exact-looking candidates) and named-lookup absence never
    reconcile: a missing result stays UNKNOWN, never proof of no delayed
    creation.
    """
    if not isinstance(candidates, list) or len(candidates) != 1:
        return False
    candidate = candidates[0]
    if not isinstance(candidate, dict) or not isinstance(persisted, dict):
        return False
    name = candidate.get("app_name", candidate.get("name"))
    image = candidate.get("image_tag", candidate.get("image"))
    tag = candidate.get("reservation_tag", candidate.get("reservation"))
    if (name != persisted.get("app_name")
            or image != persisted.get("image_tag")
            or tag != persisted.get("reservation_tag")):
        return False
    state = str(candidate.get("state", candidate.get("status", ""))).strip().lower()
    if state in ("deployed", "queued"):
        return False
    return True


def _persist_finalization_receipt(receipt_path: Path, receipt: dict) -> str | None:
    """Write group_finalization.json; never raise.

    On the revoked-disk path the work disk refuses writes: record that fact
    in the receipt (which forces UNKNOWN downstream) and return None for the
    sha. Cleanup must not raise.
    """
    try:
        return digest(write_durable(receipt_path, receipt))
    except OSError as exc:
        receipt["errors"] = receipt["errors"] + [
            f"group finalization receipt not durable: {exc}"]
        return None


def _write_unknown(receipt_dir: Path, reason: str, receipt: dict,
                   receipt_sha256: str | None, worker_pid: int | None,
                   worker_returncode: int | None, argv: list[str]) -> None:
    """Persist unknown.json in the sealed fixture shape (fixture_exit_7).

    Best effort: cleanup must never raise, even when the work disk refuses
    writes (the revoked-disk regression path keeps UNKNOWN without receipts).
    """
    try:
        write_durable(receipt_dir / UNKNOWN_NAME, {
            "error": reason,
            "pid": worker_pid,
            "reaped": bool(receipt.get("worker_reaped")),
            "returncode": worker_returncode,
            "group_finalization": receipt,
            "group_finalization_ref": {
                "path": str(receipt_dir / GROUP_FINALIZATION_NAME),
                "sha256": receipt_sha256,
            },
            "argv": argv,
            "at": time.time(),
            "replay_forbidden": True,
            "provider_cleanup_verified": False,
        })
    except OSError:
        pass


def run_child(argv: list[str], *, result_path: Path, receipt_dir: Path,
              deadline_epoch: float, cwd: Path | None = None,
              env: dict[str, str] | None = None,
              reap_reserve_seconds: float = REAP_RESERVE_SECONDS) -> dict:
    """Run argv in its own process group under the v6 finalization contract.

    ``deadline_epoch`` is the absolute controller deadline (epoch seconds),
    converted once to a monotonic deadline. Per the sealed
    ``exec_controller_bound`` the caller (worker) phase runs until the
    deadline itself -- the timeout already includes the two-second reap
    reserve. After the work deadline, ``finalize_group`` runs unconditionally
    under a fixed, once-computed post-deadline grace (``CLEANUP_GRACE_SECONDS``,
    never renewed) so SIGKILL + reap + ESRCH verification always complete.

    ``reap_reserve_seconds`` is retained for API compatibility: per the sealed
    ``exec_controller_bound`` the two-second reserve is already included in
    the caller timeout, not subtracted from it.

    Returns ``{"status": "SUCCESS", "result": <result JSON>, ...}`` only when
    the worker exited 0, the direct worker was reaped by this controller, the
    original owned PGID is verified absent, the finalization receipt is
    durable, and controller deadline remains. Any other outcome returns
    ``{"status": "UNKNOWN", "result": None, ...}`` with ``unknown_reason``;
    a successful result is never returned on the UNKNOWN path.
    """
    deadline_mono = time.monotonic() + (deadline_epoch - time.time())
    # Caller (worker) phase runs until the deadline itself: per the sealed
    # exec_controller_bound the timeout already includes the reap reserve.
    worker_deadline_mono = deadline_mono
    # Post-deadline finalization grace, computed once here and never renewed.
    final_deadline_mono = deadline_mono + CLEANUP_GRACE_SECONDS
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / GROUP_FINALIZATION_NAME

    if deadline_epoch <= time.time():
        receipt = finalize_group(None, None, deadline_mono=deadline_mono)
        receipt["errors"] = receipt["errors"] + ["controller deadline already elapsed"]
        receipt_sha256 = _persist_finalization_receipt(receipt_path, receipt)
        _write_unknown(receipt_dir, "controller deadline already elapsed at run_child "
                      "entry; no worker spawned", receipt, receipt_sha256, None, None, argv)
        return {"status": "UNKNOWN", "result": None, "worker_returncode": None,
                "group_finalization": receipt,
                "unknown_reason": "controller deadline already elapsed"}

    proc: subprocess.Popen | None = None
    pgid: int | None = None
    worker_pid: int | None = None
    worker_returncode: int | None = None
    pending: str | None = None
    log_stream = None
    try:
        try:
            try:
                log_stream = open(receipt_dir / WORKER_LOG_NAME, "wb")
            except OSError:
                log_stream = None
            # start_new_session makes the worker a session and group leader,
            # so the original owned PGID equals the worker PID by construction
            # (setpgid equivalent on POSIX; no unsafe preexec_fn is used).
            proc = subprocess.Popen(argv, cwd=None if cwd is None else str(cwd),
                                    env=env, start_new_session=True,
                                    stdin=subprocess.DEVNULL,
                                    stdout=log_stream if log_stream is not None
                                    else subprocess.DEVNULL,
                                    stderr=subprocess.STDOUT)
            pgid = proc.pid
            worker_pid = proc.pid
            try:
                write_durable(receipt_dir / PROCESS_NAME, {
                    "pid": proc.pid,
                    "pgid": proc.pid,
                    "deadline": deadline_epoch,
                    "command": list(argv),
                })
            except OSError as exc:
                pending = f"launch receipt not durable: {exc}"
        except OSError as exc:
            pending = pending or f"worker spawn failed: {exc}"
    except Exception as exc:  # defensive: finalize before any escape
        pending = pending or f"worker phase raised: {exc!r}"
    finally:
        if log_stream is not None:
            try:
                log_stream.close()
            except OSError:
                pass

    if proc is not None:
        # Worker phase ends on exit OR on the worker-phase deadline; either
        # way finalize_group below runs unconditionally.
        code, reap_error = reap_worker(proc.pid, deadline_mono=worker_deadline_mono,
                                       poll_seconds=FINALIZATION_POLL_SECONDS)
        if reap_error is None:
            worker_returncode = code
            proc.returncode = code  # keep Popen destructor state consistent
        else:
            pending = pending or reap_error

    receipt = finalize_group(pgid, proc.pid if proc is not None else None,
                             deadline_mono=final_deadline_mono,
                             pre_reaped=(worker_returncode is not None,
                                         worker_returncode))
    receipt["worker_returncode_observed_in_worker_phase"] = worker_returncode
    receipt_sha256 = _persist_finalization_receipt(receipt_path, receipt)

    try:
        if pending:
            raise UnknownOutcome(pending)
        if receipt["errors"]:
            raise UnknownOutcome("finalization errors: " + "; ".join(receipt["errors"]))
        if not receipt["worker_reaped"]:
            raise UnknownOutcome("direct worker not reaped")
        if not receipt["original_group_absent"]:
            raise UnknownOutcome("original owned process group survived")
        if worker_returncode != 0:
            # Nonzero (or unobserved) workers stay UNKNOWN even when the group
            # was successfully cleaned; the result is deliberately not read.
            raise UnknownOutcome(
                f"worker exit {worker_returncode}; SDK operation UNKNOWN; no replay; "
                "group finalization verified=True")
        if time.monotonic() > deadline_mono:
            raise UnknownOutcome("controller deadline overrun during finalization")
        if not result_path.is_file():
            raise UnknownOutcome("worker result artifact absent")
        try:
            result = json.loads(result_path.read_text())
        except (OSError, ValueError) as exc:
            raise UnknownOutcome(f"worker result unreadable: {exc}") from exc
    except UnknownOutcome as exc:
        _write_unknown(receipt_dir, exc.reason, receipt, receipt_sha256,
                       worker_pid, worker_returncode, argv)
        return {"status": "UNKNOWN", "result": None,
                "worker_returncode": worker_returncode,
                "group_finalization": receipt, "unknown_reason": exc.reason}

    # Result JSON is read only after group absence and durable finalization.
    return {"status": "SUCCESS", "result": result,
            "worker_returncode": worker_returncode,
            "group_finalization": receipt, "unknown_reason": None}


if __name__ == "__main__":  # pragma: no cover - manual fixture reproduction
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--argv-json", required=True,
                        help="JSON list: worker argv (use a fixture that spawns a "
                             "same-PGID child to reproduce the v6 regression)")
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    outcome = run_child(json.loads(args.argv_json), result_path=args.result_path,
                        receipt_dir=args.receipt_dir,
                        deadline_epoch=args.deadline_epoch)
    print(json.dumps(outcome, indent=2, sort_keys=True))
    sys.exit(0 if outcome["status"] == "SUCCESS" else 3)
