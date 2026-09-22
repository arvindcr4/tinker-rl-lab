#!/usr/bin/env python3
"""E13 hosted cancellation supervisor — design artifact, written not deployed.

Implements the separately-hosted cancellation supervisor required by the
accepted E13 lifecycle amendment
(outputs/PES_Phase2_Review_2026-09-12/finish/e13_continuation/amendment_acceptance_2026-09-19.json):

* The supervisor owns its own clock (wall + monotonic) and an immutable
  lease. The absolute deadline D = t0 + 3600 s, where t0 precedes the first
  billable provider operation (build/prepare/queue/startup), NEVER moves on
  process start, reconnect or lease renewal. Clocks are never rebased.
* It polls a lease file on a volume both hosts can see. The requester holds
  a short renewable lease (default 5 s) via HMAC-authenticated, strictly
  sequenced, replay-rejecting renewal records. Renewals extend only the
  short lease, never D. A 30 s early-cleanup reserve stops honored renewals
  at D-30. Requester death (renewals stop, or arrive unsigned/foreign)
  therefore cannot renew the deadline: expiry follows and cancellation
  intent fires.
* On lease expiry, the early-cleanup cutoff, or a detected clock rollback,
  it issues the authenticated exact-resource cancellation for the exact
  resource named in the lease. The Modal API call is STUBBED behind
  --dry-run and is NEVER executed by this artifact; live execution is
  deliberately unimplemented.
* It writes append-only, write-once supervisor receipts (ARMED, RENEWAL
  decisions, ROLLBACK, CANCELLATION_INTENT_DRY_RUN, OBSERVATION, and the
  amendment's terminal CLEANUP_VERIFIED / CLEANUP_UNVERIFIED / CLEANUP_LATE
  statuses with identities, last observations and continuing-charge notes).

No network, provider SDK, or billing call exists in this module. Keys are
32+ byte secret files with 0600 permissions; renewal and receipt HMACs use
direction-separated domains so a supervisor receipt replayed as a renewal
is rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
from pathlib import Path

LEASE_SCHEMA = "e13-hosted-supervisor-lease-v1"
RENEWAL_SCHEMA = "e13-hosted-supervisor-renewal-v1"
RECEIPT_SCHEMA = "e13-hosted-supervisor-receipt-v1"
DOMAIN_RENEWAL = b"e13-supervisor-domain/renewal-v1"
DOMAIN_RECEIPT = b"e13-supervisor-domain/receipt-v1"

STRICT_TOTAL_LEASE_SECONDS = 3600  # E13 strict first-allocation lifecycle
DEFAULT_LEASE_SECONDS = 5
DEFAULT_CLEANUP_RESERVE_SECONDS = 30
DEFAULT_MAX_SKEW_SECONDS = 2.0
DEFAULT_OBSERVATION_SECONDS = 300

KIND_ARMED = "ARMED"
KIND_INTENT = "CANCELLATION_INTENT_DRY_RUN"
KIND_ROLLBACK = "ROLLBACK_DETECTED"
KIND_OBSERVATION = "OBSERVATION"
KIND_CLEANUP_VERIFIED = "CLEANUP_VERIFIED"
KIND_CLEANUP_UNVERIFIED = "CLEANUP_UNVERIFIED"
KIND_CLEANUP_LATE = "CLEANUP_LATE"
TERMINAL_KINDS = (KIND_CLEANUP_VERIFIED, KIND_CLEANUP_UNVERIFIED, KIND_CLEANUP_LATE)


class SupervisorError(RuntimeError):
    pass


class NeverExecutedError(SupervisorError):
    """Raised by the cancellation stub if live execution is ever attempted."""


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint(value) -> str:
    return digest(canonical(value))


def read(path: Path):
    return json.loads(Path(path).read_text())


def _fsync_best_effort(path: Path):
    try:
        with open(path, "rb") as stream:
            os.fsync(stream.fileno())
    except OSError:
        pass
    try:
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def write_once(path: Path, value) -> None:
    path = Path(path)
    data = canonical(value)
    if path.exists():
        if path.read_bytes() != data:
            raise SupervisorError(f"immutable collision: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    _fsync_best_effort(path)


def real_clock():
    return time.time(), time.monotonic()


def sign(domain: bytes, key: bytes, payload: dict) -> str:
    body = {k: v for k, v in payload.items() if k != "hmac"}
    return hmac.new(key, domain + b"\n" + canonical(body), hashlib.sha256).hexdigest()


def verify(domain: bytes, key: bytes, payload: dict) -> bool:
    expected = sign(domain, key, payload)
    provided = payload.get("hmac")
    return isinstance(provided, str) and hmac.compare_digest(expected, provided)


def write_key(path: Path) -> bytes:
    path = Path(path)
    if path.exists():
        key = path.read_bytes().strip()
        if len(key) < 32:
            raise SupervisorError(f"existing key too short: {path}")
        return key
    key = secrets.token_hex(32).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, key + b"\n")
        os.fsync(fd)
    finally:
        os.close(fd)
    return key


def load_key(path: Path) -> bytes:
    key = Path(path).read_bytes().strip()
    if len(key) < 32:
        raise SupervisorError(f"key too short: {path}")
    return key


class ModalCancellationStub:
    """Authenticated exact-resource cancellation for Modal; NEVER EXECUTED.

    build_request() produces the exact call the supervisor would issue:
    FunctionCallCancel on the lease's exact object id, bound to the lease
    nonce, deadline, account and credential digests. execute() is
    deliberately unimplemented: this artifact only records dry-run intent
    receipts. Any attempt to execute raises NeverExecutedError.
    """

    service = "modal"

    def build_request(self, lease: dict) -> dict:
        resource = lease["resource"]
        return {
            "service": self.service,
            "operation": "FunctionCallCancel",
            "object_id": resource["object_id"],
            "nonce": resource["nonce"],
            "deadline_epoch": lease["deadline_epoch"],
            "account_digest": resource.get("account_digest"),
            "credential_digest": resource.get("credential_digest"),
            "authentication": "supervisor-held credentials, verified at arm; not read here",
        }

    def execute(self, request: dict):
        raise NeverExecutedError(
            "live cancellation is not implemented in this artifact; "
            "only --dry-run intent receipts exist (E13 amendment acceptance 2026-09-19)")


class HostedSupervisor:
    """Polls the lease volume; fires cancellation intent exactly once (dry-run only)."""

    def __init__(self, volume, requester_key: bytes, supervisor_key: bytes,
                 clock=real_clock, observation_seconds: int = DEFAULT_OBSERVATION_SECONDS,
                 max_skew_seconds: float = DEFAULT_MAX_SKEW_SECONDS):
        self.volume = Path(volume)
        self.requester_key = requester_key
        self.supervisor_key = supervisor_key
        self.clock = clock
        self.observation_seconds = observation_seconds
        self.max_skew_seconds = max_skew_seconds
        self.executor = ModalCancellationStub()
        self._last_wall = None

    # ---------- layout ----------
    @property
    def lease_path(self) -> Path:
        return self.volume / "lease.json"

    @property
    def renewals_dir(self) -> Path:
        return self.volume / "renewals"

    @property
    def receipts_dir(self) -> Path:
        return self.volume / "receipts"

    @property
    def state_path(self) -> Path:
        return self.volume / "supervisor-state.json"

    # ---------- arm ----------
    def arm(self, resource: dict, requester: dict | None = None,
            lease_seconds: int = DEFAULT_LEASE_SECONDS,
            cleanup_reserve_seconds: int = DEFAULT_CLEANUP_RESERVE_SECONDS,
            total_lease_seconds: int = STRICT_TOTAL_LEASE_SECONDS) -> dict:
        wall, mono = self.clock()
        lease = {
            "schema": LEASE_SCHEMA,
            "lease_id": digest(secrets.token_bytes(32)),
            "t0_epoch": wall,
            "deadline_epoch": wall + total_lease_seconds,
            "total_lease_seconds": total_lease_seconds,
            "lease_seconds": lease_seconds,
            "cleanup_reserve_seconds": cleanup_reserve_seconds,
            "max_clock_skew_seconds": self.max_skew_seconds,
            "armed_monotonic": mono,
            "resource": dict(resource),
            "requester": dict(requester or {}),
            "clock_rule": "D = t0 + total_lease_seconds never moves; renewals extend only the short lease",
        }
        require_resource(resource)
        write_once(self.lease_path, lease)
        self._last_wall = wall
        self._persist_state(wall)
        self._receipt(KIND_ARMED, {
            "lease_id": lease["lease_id"], "t0_epoch": lease["t0_epoch"],
            "deadline_epoch": lease["deadline_epoch"],
            "lease_seconds": lease_seconds,
            "cleanup_reserve_seconds": cleanup_reserve_seconds,
            "resource": lease["resource"]})
        return lease

    # ---------- renewals (requester side helper writes; supervisor verifies) ----------
    def build_renewal(self, seq: int, requester_boot: str, issued_epoch: float) -> dict:
        """Requester-side: construct and sign one renewal record."""
        lease = self.load_lease()
        record = {
            "schema": RENEWAL_SCHEMA,
            "lease_id": lease["lease_id"],
            "seq": seq,
            "issued_epoch": issued_epoch,
            "expires_epoch": issued_epoch + lease["lease_seconds"],
            "requester_boot": requester_boot,
        }
        record["hmac"] = sign(DOMAIN_RENEWAL, self.requester_key, record)
        return record

    def submit_renewal(self, record: dict) -> dict:
        """Validate and durably append one renewal record (write-once per seq)."""
        lease = self.load_lease()
        wall, _ = self.clock()
        latest = self.latest_verified_renewal(lease)
        expected_seq = (latest["seq"] + 1) if latest else 1
        self._check_renewal(record, lease, wall, expected_seq)
        write_once(self.renewals_dir / f"ren-{record['seq']:010d}.json", record)
        return {"accepted": True, "seq": record["seq"],
                "deadline_epoch": lease["deadline_epoch"], "note": "deadline unchanged by renewal"}

    def _verify_renewal_authentic(self, record: dict, lease: dict) -> None:
        """Authenticity checks valid both at submission and on historical re-read."""
        if not isinstance(record, dict) or record.get("schema") != RENEWAL_SCHEMA:
            raise SupervisorError("renewal: wrong schema")
        if record.get("lease_id") != lease["lease_id"]:
            raise SupervisorError("renewal: foreign lease")
        if not verify(DOMAIN_RENEWAL, self.requester_key, record):
            raise SupervisorError("renewal: unauthenticated (bad hmac or reflected receipt)")
        issued = record.get("issued_epoch")
        if not isinstance(issued, (int, float)):
            raise SupervisorError("renewal: issued_epoch missing")
        if record.get("expires_epoch") != issued + lease["lease_seconds"]:
            raise SupervisorError("renewal: expiry must be issued_epoch + lease_seconds")
        requester = lease.get("requester") or {}
        if requester.get("boot_id") and record.get("requester_boot") != requester["boot_id"]:
            raise SupervisorError("renewal: requester identity mismatch")

    def _check_renewal(self, record: dict, lease: dict, wall: float, expected_seq: int) -> None:
        """Full submission-time checks: authenticity + ordering + freshness + cutoff."""
        self._verify_renewal_authentic(record, lease)
        if not isinstance(record.get("seq"), int) or record["seq"] != expected_seq:
            raise SupervisorError(f"renewal: stale/replayed/out-of-order seq (want {expected_seq})")
        issued = record["issued_epoch"]
        if abs(wall - issued) > self.max_skew_seconds:
            raise SupervisorError("renewal: issued_epoch outside clock-skew bound")
        if issued > lease["deadline_epoch"] - lease["cleanup_reserve_seconds"] + self.max_skew_seconds:
            raise SupervisorError("renewal: past early-cleanup cutoff; deadline cannot be held open")

    def latest_verified_renewal(self, lease: dict) -> dict | None:
        best = None
        if not self.renewals_dir.is_dir():
            return None
        for path in sorted(self.renewals_dir.glob("ren-*.json")):
            try:
                record = read(path)
                # Re-read verifies authenticity only. Wall-freshness and the
                # early-cleanup cutoff were enforced at submission time; applying
                # them to historical records would invalidate the whole chain on
                # any clock advance and break seq continuity (fail-open skip).
                self._verify_renewal_authentic(record, lease)
                if not isinstance(record.get("seq"), int):
                    raise SupervisorError("renewal: seq missing")
            except (SupervisorError, ValueError, OSError, KeyError):
                continue  # tampered/invalid record cannot extend the lease
            if best is None or record["seq"] > best["seq"]:
                best = record
        return best

    # ---------- evaluation ----------
    def load_lease(self) -> dict:
        lease = read(self.lease_path)
        if lease.get("schema") != LEASE_SCHEMA:
            raise SupervisorError("lease: wrong schema")
        if lease["deadline_epoch"] != lease["t0_epoch"] + lease["total_lease_seconds"]:
            raise SupervisorError("lease: deadline is not t0 + total (clock rebasing forbidden)")
        return lease

    def _persist_state(self, wall: float) -> None:
        state = self.state_path
        if state.exists():
            previous = read(state).get("wall_hwm_epoch", float("-inf"))
            if wall < previous:
                raise SupervisorError("refusing to lower persisted wall high-water mark")
        tmp = state.with_suffix(".tmp")
        tmp.write_text(canonical({"wall_hwm_epoch": wall}).decode())
        os.replace(tmp, state)
        _fsync_best_effort(state)

    def _rollback_check(self, wall: float) -> bool:
        """True if the supervisor's own wall clock moved backward beyond skew."""
        rolled = False
        if self._last_wall is not None and wall < self._last_wall - self.max_skew_seconds:
            rolled = True
        if self.state_path.exists():
            hwm = read(self.state_path).get("wall_hwm_epoch")
            if isinstance(hwm, (int, float)) and wall < hwm - self.max_skew_seconds:
                rolled = True
        return rolled

    def _receipt(self, kind: str, body: dict) -> Path:
        self.receipts_dir.mkdir(parents=True, exist_ok=True)
        seq = len(list(self.receipts_dir.glob("*.json"))) + 1
        wall, _ = self.clock()
        payload = {"schema": RECEIPT_SCHEMA, "kind": kind, "seq": seq,
                   "recorded_epoch": wall, **body}
        payload["hmac"] = sign(DOMAIN_RECEIPT, self.supervisor_key, payload)
        path = self.receipts_dir / f"{seq:06d}-{kind}.json"
        write_once(path, payload)
        return path

    def receipts(self) -> list:
        if not self.receipts_dir.is_dir():
            return []
        out = []
        for path in sorted(self.receipts_dir.glob("*.json")):
            record = read(path)
            if not verify(DOMAIN_RECEIPT, self.supervisor_key, record):
                raise SupervisorError(f"receipt tampering detected: {path}")
            out.append(record)
        return out

    def intent_receipt(self) -> dict | None:
        for record in self.receipts():
            if record["kind"] == KIND_INTENT:
                return record
        return None

    def terminal_receipt(self) -> dict | None:
        for record in self.receipts():
            if record["kind"] in TERMINAL_KINDS:
                return record
        return None

    def fire_cancellation(self, lease: dict, reason: str, wall: float) -> dict:
        """Record cancellation intent exactly once; dry-run only, never executed."""
        if self.intent_receipt() is not None:
            return self.intent_receipt()
        request = self.executor.build_request(lease)
        payload = {
            "lease_id": lease["lease_id"],
            "reason": reason,
            "resource": lease["resource"],
            "api_call": request,
            "execution": {"mode": "dry-run", "executed": False,
                          "executor": "ModalCancellationStub",
                          "live_execution": "NOT_IMPLEMENTED_IN_THIS_ARTIFACT"},
            "t0_epoch": lease["t0_epoch"],
            "deadline_epoch": lease["deadline_epoch"],
            "clock_rebased": False,
            "last_observation_epoch": wall,
            "continuing_charges_possible": True,
        }
        path = self._receipt(KIND_INTENT, payload)
        return read(path)

    def _unverified_if_window_elapsed(self, lease: dict, wall: float) -> None:
        if self.terminal_receipt() is not None:
            return
        if self.intent_receipt() is None:
            return
        window_end = max(lease["deadline_epoch"],
                         self.intent_receipt()["recorded_epoch"]) + self.observation_seconds
        if wall >= window_end:
            self._receipt(KIND_CLEANUP_UNVERIFIED, {
                "lease_id": lease["lease_id"],
                "resource": lease["resource"],
                "observation_window_seconds": self.observation_seconds,
                "last_observation_epoch": wall,
                "provider_absence_observed": False,
                "billing_closure_observed": False,
                "possible_continuing_charges": True,
                "note": "no verified provider absence/billing closure inside the supervision window; unknown, not success"})

    def run_once(self, dry_run: bool = True) -> dict:
        """One poll of the lease volume; may fire intent and terminal receipts."""
        lease = self.load_lease()
        wall, mono = self.clock()
        if self._rollback_check(wall):
            self._receipt(KIND_ROLLBACK, {
                "lease_id": lease["lease_id"], "detected_epoch": wall,
                "last_wall": self._last_wall, "max_skew_seconds": self.max_skew_seconds,
                "action": "fail-closed: cancellation intent issued; clocks are never rebased"})
            # the persisted high-water mark is never lowered; history is not rewritten
            if not dry_run:
                raise NeverExecutedError("live cancellation execution is not implemented")
            self.fire_cancellation(lease, "ROLLBACK_DETECTED", wall)
            # An intent was issued: report the operator-visible terminal status.
            # The rollback evidence itself lives in the KIND_ROLLBACK receipt
            # and the intent reason stays ROLLBACK_DETECTED.
            status = "CANCELLATION_INTENT_ISSUED"
        else:
            if self._last_wall is None or wall > self._last_wall:
                self._last_wall = wall
            self._persist_state(self._last_wall)
            latest = self.latest_verified_renewal(lease)
            expiry = latest["expires_epoch"] if latest else lease["t0_epoch"] + lease["lease_seconds"]
            cutoff = lease["deadline_epoch"] - lease["cleanup_reserve_seconds"]
            if self.intent_receipt() is not None:
                status = "CANCELLATION_INTENT_ISSUED"
            elif wall >= cutoff:
                if not dry_run:
                    raise NeverExecutedError("live cancellation execution is not implemented")
                self.fire_cancellation(lease, "EARLY_CLEANUP_CUTOFF", wall)
                status = "CANCELLATION_INTENT_ISSUED"
            elif wall > expiry:
                if not dry_run:
                    raise NeverExecutedError("live cancellation execution is not implemented")
                self.fire_cancellation(lease, "LEASE_EXPIRED", wall)
                status = "CANCELLATION_INTENT_ISSUED"
            else:
                status = "LIVE"
        self._unverified_if_window_elapsed(lease, wall)
        terminal = self.terminal_receipt()
        if terminal is not None:
            status = terminal["kind"]
        return {"status": status, "now_epoch": wall, "monotonic": mono,
                "deadline_epoch": lease["deadline_epoch"],
                "intent": self.intent_receipt(), "terminal": terminal}

    # ---------- observation / amendment reporting ----------
    def record_completion_observation(self, observed_epoch: float, provider_absent: bool,
                                      billing_closed: bool, evidence: dict) -> dict:
        """Independently recorded provider/billing observation -> amendment statuses."""
        lease = self.load_lease()
        if self.terminal_receipt() is not None:
            return self.terminal_receipt()
        self._receipt(KIND_OBSERVATION, {
            "lease_id": lease["lease_id"], "observed_epoch": observed_epoch,
            "provider_absent": bool(provider_absent),
            "billing_closed": bool(billing_closed), "evidence": evidence})
        deadline = lease["deadline_epoch"]
        if provider_absent and billing_closed:
            kind = KIND_CLEANUP_VERIFIED if observed_epoch <= deadline else KIND_CLEANUP_LATE
            self._receipt(kind, {
                "lease_id": lease["lease_id"], "observed_epoch": observed_epoch,
                "deadline_epoch": deadline,
                "late": kind == KIND_CLEANUP_LATE,
                "provider_absent": True, "billing_closed": True,
                "possible_continuing_charges": kind == KIND_CLEANUP_LATE})
            return self.terminal_receipt()
        return {"status": "OBSERVED_INCOMPLETE", "provider_absent": provider_absent,
                "billing_closed": billing_closed,
                "note": "absence and billing closure need distinct evidence; neither is inferred"}


def require_resource(resource: dict) -> None:
    if not isinstance(resource, dict):
        raise SupervisorError("resource must be a dict")
    for field in ("provider", "operation", "object_id", "nonce"):
        if not isinstance(resource.get(field), str) or not resource[field]:
            raise SupervisorError(f"resource missing exact field: {field}")


def _open_supervisor(args) -> HostedSupervisor:
    return HostedSupervisor(
        args.volume,
        load_key(Path(args.volume) / "requester.key"),
        load_key(Path(args.volume) / "supervisor.key"),
        observation_seconds=args.observation_seconds)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["keygen", "arm", "renew", "run", "observe", "status"])
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--object-id")
    parser.add_argument("--nonce")
    parser.add_argument("--account-digest")
    parser.add_argument("--credential-digest")
    parser.add_argument("--requester-boot")
    parser.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    parser.add_argument("--cleanup-reserve", type=int, default=DEFAULT_CLEANUP_RESERVE_SECONDS)
    parser.add_argument("--total-lease-seconds", type=int, default=STRICT_TOTAL_LEASE_SECONDS)
    parser.add_argument("--observation-seconds", type=int, default=DEFAULT_OBSERVATION_SECONDS)
    parser.add_argument("--seq", type=int)
    parser.add_argument("--provider-absent", action="store_true")
    parser.add_argument("--billing-closed", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="mandatory safety mode; the cancellation API call is stubbed and never executed")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    volume = args.volume

    if args.command == "keygen":
        write_key(volume / "requester.key")
        write_key(volume / "supervisor.key")
        print(json.dumps({"status": "KEYS_READY", "volume": str(volume)}))
        return 0

    supervisor = _open_supervisor(args)

    if args.command == "arm":
        wall, _ = real_clock()
        resource = {"provider": "modal", "operation": "FunctionCallCancel",
                    "object_id": args.object_id or "fc-UNASSIGNED-AT-ARM",
                    "nonce": args.nonce or digest(secrets.token_bytes(32)),
                    "account_digest": args.account_digest,
                    "credential_digest": args.credential_digest}
        requester = {"boot_id": args.requester_boot} if args.requester_boot else {}
        lease = supervisor.arm(resource, requester, args.lease_seconds,
                               args.cleanup_reserve, args.total_lease_seconds)
        print(json.dumps({"status": "ARMED", "lease_id": lease["lease_id"],
                          "t0_epoch": lease["t0_epoch"], "deadline_epoch": lease["deadline_epoch"],
                          "sha256": fingerprint(lease)}, indent=2))
        return 0

    if args.command == "renew":
        if args.seq is None:
            parser.error("--seq is required for renew")
        wall, _ = real_clock()
        lease = supervisor.load_lease()
        boot = (lease.get("requester") or {}).get("boot_id") or "unbound"
        record = supervisor.build_renewal(args.seq, boot, wall)
        result = supervisor.submit_renewal(record)
        print(json.dumps(result))
        return 0

    if args.command == "run":
        if not args.dry_run:
            print("refusing: live cancellation execution is not implemented in this artifact; "
                  "pass --dry-run to record cancellation intent only", file=sys.stderr)
            return 2
        while True:
            outcome = supervisor.run_once(dry_run=True)
            if args.once or outcome["status"] in TERMINAL_KINDS:
                print(json.dumps({"status": outcome["status"],
                                  "deadline_epoch": outcome["deadline_epoch"],
                                  "intent_issued": outcome["intent"] is not None}, indent=2))
                return 0
            time.sleep(args.poll_interval)

    if args.command == "observe":
        wall, _ = real_clock()
        result = supervisor.record_completion_observation(
            wall, args.provider_absent, args.billing_closed, {"source": "manual CLI observation"})
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "status":
        lease = supervisor.load_lease()
        outcome = supervisor.run_once(dry_run=True)
        print(json.dumps({"lease_id": lease["lease_id"], "status": outcome["status"],
                          "receipt_count": len(supervisor.receipts())}, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
