#!/usr/bin/env python3
"""Verify evidence against an independently supplied reviewed manifest digest.

No digest is minted from the evidence being verified. The caller must obtain the
manifest SHA-256 from a trusted release/signature, separately from this bundle.
"""

from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "evidence_snapshot.json"
IMAGES = (ROOT / "wandb_claim2.png", ROOT / "wandb_run_hygiene.png")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(manifest_path=None, expected_manifest_sha256=None):
    manifest_path = manifest_path or os.environ.get("EVIDENCE_MANIFEST_PATH")
    anchor = expected_manifest_sha256 or os.environ.get("EVIDENCE_MANIFEST_SHA256")
    checks = {"trusted_manifest_present": bool(manifest_path and anchor)}
    hashes = {}
    try:
        if not checks["trusted_manifest_present"]:
            return {
                "status": "FAIL",
                "checks": checks,
                "sha256": hashes,
                "reason": "An independently trusted manifest digest is required",
            }
        checks["manifest_integrity"] = sha256(manifest_path) == anchor
        manifest = json.loads(Path(manifest_path).read_text())
        paths = [SNAPSHOT, *IMAGES]
        hashes = {p.name: sha256(p) for p in paths}
        checks["artifact_integrity"] = (
            set(manifest["sha256"]) == set(hashes) and manifest["sha256"] == hashes
        )
        data = json.loads(SNAPSHOT.read_text())
        datetime.date.fromisoformat(data["snapshot_date"])
        checks["valid_date"] = True
        runs = data["claim_2_matched_budget"]
        checks["exact_run_identity"] = sorted(r["run_id"] for r in runs) == sorted(
            manifest["run_ids"]
        )
        checks["unique_runs"] = len(set(r["run_id"] for r in runs)) == len(runs)
        checks["png_format"] = all(p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for p in IMAGES)
        checks["https_links"] = all(
            isinstance(r["url"], str) and r["url"].startswith("https://") for r in runs
        )
    except (OSError, ValueError, KeyError, TypeError):
        checks["readable_valid_schema"] = False
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "sha256": hashes,
        "scope": "release artifact integrity; audit counts require a separate reconciled inventory",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args(argv)
    result = verify(args.manifest, args.manifest_sha256)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
