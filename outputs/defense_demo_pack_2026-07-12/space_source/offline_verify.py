#!/usr/bin/env python3
"""Verify the defense snapshot using only Python's standard library."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "evidence_snapshot.json"
IMAGES = (ROOT / "wandb_claim2.png", ROOT / "wandb_run_hygiene.png")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify() -> dict:
    data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    runs = data["claim_2_matched_budget"]
    checks = {
        "snapshot_date_present": bool(data.get("snapshot_date")),
        "account_object_count_is_983": data["audit"]["account_object_count"] == 983,
        "curated_telemetry_is_70_plus": (
            data["audit"]["curated_telemetry_runs_minimum"] == 70
            and data["audit"]["curated_telemetry_label"] == "70+"
        ),
        "gold_rows_are_19": data["audit"]["gold_rows"] == 19,
        "claim_2_has_G2_and_G16": {row["group_size"] for row in runs} == {2, 16},
        "run_ids_are_unique": len({row["run_id"] for row in runs}) == len(runs),
        "all_run_links_are_https": all(row["url"].startswith("https://") for row in runs),
        "bundled_images_present": all(path.is_file() for path in IMAGES),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "sha256": {
            SNAPSHOT.name: sha256(SNAPSHOT),
            **{path.name: sha256(path) for path in IMAGES if path.is_file()},
        },
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
