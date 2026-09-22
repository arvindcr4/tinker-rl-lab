#!/usr/bin/env python3
"""E6/E9 AWS Service Quotas gate watcher. READ-ONLY. GO/NO-GO per lane.

Lanes and gates (as of the 2026-09-19 close-out campaign):

  E6  us-east-2  Standard On-Demand vCPUs (L-1216C47A) needs >= 16
      (two m6a.2xlarge WebArena hosts). Pre-existing increase request
      c67d89b0f3774bd7a0485ca1852ab04bMhhldFKV (desired 16) was filed
      2026-09-12 and was still open at the last check; the
      one-open-request rule blocks filing a new one while it is open.

  E9  us-east-1  Standard On-Demand vCPUs (L-1216C47A) needs >= 8
      (runtime-image builder; >= 4 vCPU is the documented minimum from
      the rejected m6i.xlarge attempt). Increase request
      2eebd37d2a4447ceb81d0a45d1dffbe5fbVtQSM7 (desired 8) was filed
      2026-09-19 and is PENDING.

This script issues ONLY read-only Service Quotas calls:
    get-service-quota
    get-requested-service-quota-change
    list-requested-service-quota-change-history
It never requests increases, never creates resources, and never writes to
AWS. A quota GO is a necessary precondition, not launch authorization:
it does not prove AZ capacity, funding, or the ledger/readiness gates
documented in the E6/E9 close-out records.

Exit codes: 0 = all lanes GO, 1 = at least one NO-GO, 2 = tool/credential
error (nothing evaluated or partial evaluation reported to stderr).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys

QUOTA_CODE = "L-1216C47A"
SERVICE_CODE = "ec2"
ACCOUNT_ID = "464365622480"

# Read-only subcommands this watcher is allowed to issue. Deliberately an
# allowlist: anything mutating (request-service-quota-increase, put*, etc.)
# is out of scope by construction.
READ_ONLY_SUBCOMMANDS = (
    "get-service-quota",
    "get-requested-service-quota-change",
    "list-requested-service-quota-change-history",
)

OPEN_REQUEST_STATUSES = {"PENDING", "CASE_OPENED"}

DEFAULT_LANES = {
    "E6": {
        "region": "us-east-2",
        "required_vcpu": 16.0,
        "tracked_request_id": "c67d89b0f3774bd7a0485ca1852ab04bMhhldFKV",
        "purpose": "two m6a.2xlarge WebArena hosts (8 vCPU each, 16 total)",
        "watcher": "zvf-program/e6e9/check_quota_status.py",
    },
    "E9": {
        "region": "us-east-1",
        "required_vcpu": 8.0,
        "tracked_request_id": "2eebd37d2a4447ceb81d0a45d1dffbe5fbVtQSM7",
        "purpose": (
            "E9 MLDevBench runtime-image builder "
            "(>=4 vCPU documented minimum; 8 requested)"
        ),
        "watcher": "zvf-program/e6e9/check_quota_status.py",
    },
}


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _aws(args: list[str]) -> dict:
    """Run one read-only aws service-quotas call and return parsed JSON."""
    sub = args[0]
    if sub not in READ_ONLY_SUBCOMMANDS:
        raise RuntimeError(f"refusing non-read-only subcommand: {sub}")
    cmd = [
        "aws",
        "service-quotas",
        *args,
        "--output",
        "json",
        "--no-cli-pager",
        "--cli-connect-timeout",
        "10",
        "--cli-read-timeout",
        "20",
    ]
    proc = subprocess.run(  # noqa: S603 (fixed argv, no shell)
        cmd, capture_output=True, text=True, timeout=60
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"aws {' '.join(args)} failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()[:500]}"
        )
    return json.loads(proc.stdout)


def _get_quota(region: str) -> dict:
    data = _aws(
        [
            "get-service-quota",
            "--service-code",
            SERVICE_CODE,
            "--quota-code",
            QUOTA_CODE,
            "--region",
            region,
        ]
    )
    q = data.get("Quota", {})
    return {
        "value": q.get("Value"),
        "name": q.get("QuotaName"),
        "adjustable": q.get("Adjustable"),
        "applied_at_level": q.get("QuotaAppliedAtLevel"),
    }


def _request_summary(rq: dict) -> dict:
    return {
        "id": rq.get("Id"),
        "status": rq.get("Status"),
        "desired_value": rq.get("DesiredValue"),
        "created": rq.get("Created"),
        "last_updated": rq.get("LastUpdated"),
        "case_id": rq.get("CaseId"),
    }


def _get_tracked_request(region: str, request_id: str) -> dict | None:
    try:
        data = _aws(
            [
                "get-requested-service-quota-change",
                "--request-id",
                request_id,
                "--region",
                region,
            ]
        )
    except RuntimeError as exc:
        # A tracked id that no longer resolves is itself signal; surface it.
        return {"id": request_id, "error": str(exc)}
    return _request_summary(data.get("RequestedQuota", {}))


def _history(region: str) -> list[dict]:
    data = _aws(
        [
            "list-requested-service-quota-change-history",
            "--service-code",
            SERVICE_CODE,
            "--region",
            region,
            "--max-results",
            "50",
        ]
    )
    entries = [
        _request_summary(rq)
        for rq in data.get("RequestedQuotas", [])
        if rq.get("QuotaCode") == QUOTA_CODE
    ]
    return sorted(entries, key=lambda e: str(e.get("created") or ""))


def check_lane(lane: str, cfg: dict) -> dict:
    region = cfg["region"]
    required = float(cfg["required_vcpu"])
    quota = _get_quota(region)
    value = quota["value"]
    tracked = _get_tracked_request(region, cfg["tracked_request_id"])
    history = _history(region)

    tracked_status = None
    if tracked and tracked.get("status"):
        tracked_status = tracked["status"]
    open_requests = [
        e for e in history if e.get("status") in OPEN_REQUEST_STATUSES
    ]
    go = isinstance(value, (int, float)) and float(value) >= required

    notes = []
    if not go:
        notes.append(
            f"effective quota {value} vCPU < required {required:g} vCPU"
        )
    else:
        notes.append(f"effective quota {value} vCPU >= required {required:g} vCPU")
    if tracked_status in OPEN_REQUEST_STATUSES:
        notes.append(
            "tracked request still open; one-open-request rule blocks a new request"
        )
    elif tracked_status is not None:
        notes.append(f"tracked request is {tracked_status} (not open)")
    other_open = [
        e for e in open_requests if e.get("id") != cfg["tracked_request_id"]
    ]
    if other_open:
        notes.append(
            "additional open L-1216C47A request(s) present: "
            + ", ".join(str(e.get("id")) for e in other_open)
        )
    if not open_requests and not go:
        notes.append(
            "no open request for this quota; a new increase request may be filed"
        )
    notes.append(
        "quota GO does not prove AZ capacity, funding, or E6/E9 ledger gates"
    )

    return {
        "lane": lane,
        "region": region,
        "quota_code": QUOTA_CODE,
        "quota_name": quota["name"],
        "effective_vcpu": value,
        "required_vcpu": required,
        "purpose": cfg["purpose"],
        "tracked_request": tracked,
        "tracked_request_open": tracked_status in OPEN_REQUEST_STATUSES,
        "open_requests_for_quota": open_requests,
        "history_for_quota": history,
        "decision": "GO" if go else "NO-GO",
        "notes": notes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only E6/E9 AWS Service Quotas gate check. "
            "Prints GO/NO-GO per lane. Never mutates AWS state."
        )
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON only"
    )
    parser.add_argument(
        "--e6-required",
        type=float,
        default=DEFAULT_LANES["E6"]["required_vcpu"],
        help="E6 required vCPUs (default: %(default)s)",
    )
    parser.add_argument(
        "--e9-required",
        type=float,
        default=DEFAULT_LANES["E9"]["required_vcpu"],
        help="E9 required vCPUs (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    lanes = {
        "E6": {**DEFAULT_LANES["E6"], "required_vcpu": args.e6_required},
        "E9": {**DEFAULT_LANES["E9"], "required_vcpu": args.e9_required},
    }

    results = {}
    errors = {}
    for lane, cfg in lanes.items():
        try:
            results[lane] = check_lane(lane, cfg)
        except Exception as exc:  # noqa: BLE001 (report and exit 2)
            errors[lane] = f"{type(exc).__name__}: {exc}"

    report = {
        "schema": "e6e9-quota-watch-v1",
        "observed_at": _now_iso(),
        "account_id": ACCOUNT_ID,
        "service_code": SERVICE_CODE,
        "quota_code": QUOTA_CODE,
        "read_only": True,
        "results": results,
        "errors": errors,
        "overall": (
            "GO"
            if results and not errors and all(
                r["decision"] == "GO" for r in results.values()
            )
            else "NO-GO"
        ),
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=False))
    else:
        print(f"E6/E9 quota watch  {report['observed_at']}  (read-only)")
        print(f"account {ACCOUNT_ID}  quota {QUOTA_CODE}")
        for lane in ("E6", "E9"):
            if lane in errors:
                print(f"{lane}: ERROR  {errors[lane]}")
                continue
            r = results[lane]
            req = r["tracked_request"] or {}
            req_id = req.get("id") or "?"
            req_status = req.get("status") or "?"
            print(
                f"{lane} [{r['region']}]: {r['decision']}  "
                f"quota {r['effective_vcpu']} vCPU / need {r['required_vcpu']:g}"
            )
            print(
                f"    request {req_id}: {req_status}"
                f" (desired {req.get('desired_value')})"
            )
            for note in r["notes"]:
                print(f"    - {note}")
        if errors:
            print(f"overall: ERROR ({len(errors)} lane(s) failed to evaluate)")
        else:
            print(f"overall: {report['overall']}")

    if errors:
        return 2
    return 0 if report["overall"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
