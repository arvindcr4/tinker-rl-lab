#!/usr/bin/env python3
"""Read-only Tinker history inspector for the NeurIPS 36320 evidence audit.

The API key is read once from stdin, placed only in this process environment,
and never printed or written. The script performs list/get operations only.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
import sys
from typing import Any


TARGET_MODEL = "Qwen/Qwen3-8B"
MATCHED_START = datetime(2026, 7, 24, 8, 30, tzinfo=timezone.utc)
MATCHED_END = datetime(2026, 7, 24, 11, 30, tzinfo=timezone.utc)
WIDE_START = datetime(2026, 7, 23, 0, 0, tzinfo=timezone.utc)
WIDE_END = datetime(2026, 7, 25, 0, 0, tzinfo=timezone.utc)


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def safe_run(run: dict[str, Any], checkpoint_counts: Counter[str]) -> dict[str, Any]:
    run_id = str(run.get("training_run_id", ""))
    return {
        "training_run_id": run_id,
        "base_model": run.get("base_model"),
        "model_owner": run.get("model_owner"),
        "is_lora": run.get("is_lora"),
        "lora_rank": run.get("lora_rank"),
        "corrupted": run.get("corrupted"),
        "last_request_time": run.get("last_request_time"),
        "user_metadata": run.get("user_metadata"),
        "checkpoint_count": checkpoint_counts[run_id],
        "last_sampler_checkpoint": run.get("last_sampler_checkpoint"),
    }


def checkpoint_span(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: str(row.get("time", "")))
    if not ordered:
        return {"count": 0, "first": None, "last": None}
    return {
        "count": len(ordered),
        "first": ordered[0],
        "last": ordered[-1],
    }


def paginate(method: Any, field: str, **kwargs: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        response = method(limit=100, offset=offset, **kwargs).result(timeout=120)
        page = []
        for item in getattr(response, field):
            if hasattr(item, "model_dump"):
                page.append(item.model_dump(mode="json", exclude_none=True))
            elif isinstance(item, dict):
                page.append(item)
            else:
                page.append({"value": str(item)})
        rows.extend(page)
        total = int(response.cursor.total_count)
        if not page or len(rows) >= total:
            return rows
        offset += len(page)


def list_sessions_once(rest: Any) -> list[dict[str, Any]]:
    response = rest.list_sessions(
        limit=100,
        offset=0,
        access_scope="owned",
    ).result(timeout=120)
    rows: list[dict[str, Any]] = []
    for item in response.sessions:
        if hasattr(item, "model_dump"):
            rows.append(item.model_dump(mode="json", exclude_none=True))
        elif isinstance(item, dict):
            rows.append(item)
        else:
            rows.append({"value": str(item)})
    return rows


def main() -> int:
    key = sys.stdin.readline().strip()
    if not key:
        print(json.dumps({"error": "empty Tinker API key"}))
        return 2
    os.environ["TINKER_API_KEY"] = key

    try:
        import tinker

        rest = tinker.ServiceClient().create_rest_client()
        runs = paginate(
            rest.list_training_runs,
            "training_runs",
            access_scope="owned",
        )
        checkpoints = paginate(rest.list_user_checkpoints, "checkpoints")
        sessions = list_sessions_once(rest)

        checkpoint_counts: Counter[str] = Counter()
        checkpoint_rows: dict[str, list[dict[str, Any]]] = {}
        for checkpoint in checkpoints:
            path = str(checkpoint.get("tinker_path", ""))
            if not path.startswith("tinker://") or "/" not in path:
                continue
            run_id = path[len("tinker://") :].split("/", 1)[0]
            checkpoint_counts[run_id] += 1
            checkpoint_rows.setdefault(run_id, []).append(
                {
                    "checkpoint_id": checkpoint.get("checkpoint_id"),
                    "checkpoint_type": checkpoint.get("checkpoint_type"),
                    "time": checkpoint.get("time"),
                    "tinker_path": checkpoint.get("tinker_path"),
                    "size_bytes": checkpoint.get("size_bytes"),
                }
            )

        matched: list[dict[str, Any]] = []
        wide: list[dict[str, Any]] = []
        for run in runs:
            dt = parse_time(run.get("last_request_time"))
            if run.get("base_model") != TARGET_MODEL or dt is None:
                continue
            compact = safe_run(run, checkpoint_counts)
            if MATCHED_START <= dt <= MATCHED_END:
                compact["checkpoint_span"] = checkpoint_span(
                    checkpoint_rows.get(str(run.get("training_run_id", "")), [])
                )
                matched.append(compact)
            if WIDE_START <= dt <= WIDE_END:
                wide.append(compact)

        model_counts = Counter(str(run.get("base_model")) for run in runs)
        day_counts = Counter(
            parse_time(run.get("last_request_time")).date().isoformat()
            for run in runs
            if parse_time(run.get("last_request_time")) is not None
        )
        newest = sorted(
            runs,
            key=lambda row: parse_time(row.get("last_request_time"))
            or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )[:10]

        session_window: list[dict[str, Any]] = []
        for session in sessions:
            serialized = json.dumps(session, sort_keys=True, default=str)
            if "2026-07-24" in serialized or any(
                str(row["training_run_id"]) in serialized for row in matched
            ):
                session_window.append(session)

        wide_groups: dict[str, list[dict[str, Any]]] = {}
        for row in wide:
            base_uuid = str(row["training_run_id"]).split(":train:", 1)[0]
            wide_groups.setdefault(base_uuid, []).append(row)

        result = {
            "sdk_version": getattr(tinker, "__version__", None),
            "run_count": len(runs),
            "checkpoint_count": len(checkpoints),
            "session_count": len(sessions),
            "run_keys": sorted(runs[0]) if runs else [],
            "session_keys": sorted(sessions[0]) if sessions else [],
            "target_model_count": model_counts[TARGET_MODEL],
            "model_counts_top10": model_counts.most_common(10),
            "run_counts_2026_07_23_to_25": {
                day: day_counts[day]
                for day in ("2026-07-23", "2026-07-24", "2026-07-25")
            },
            "matched_window": [
                safe
                for safe in sorted(
                    matched,
                    key=lambda row: str(row.get("last_request_time", "")),
                )
            ],
            "wide_window_count": len(wide),
            "wide_window": [
                safe
                for safe in sorted(
                    wide,
                    key=lambda row: str(row.get("last_request_time", "")),
                )
            ],
            "wide_groups": [
                {
                    "base_uuid": base_uuid,
                    "members": sorted(
                        members,
                        key=lambda row: str(row.get("training_run_id", "")),
                    ),
                }
                for base_uuid, members in sorted(
                    wide_groups.items(),
                    key=lambda item: min(
                        str(row.get("last_request_time", "")) for row in item[1]
                    ),
                )
            ],
            "session_window": session_window,
            "newest_runs": [safe_run(row, checkpoint_counts) for row in newest],
        }
        print(json.dumps(result, sort_keys=True, default=str))
        return 0
    except Exception as exc:
        message = str(exc).replace(key, "<redacted>")
        print(json.dumps({"error": type(exc).__name__, "message": message}))
        return 1
    finally:
        os.environ.pop("TINKER_API_KEY", None)


if __name__ == "__main__":
    raise SystemExit(main())
