from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

from .protocol import REPO_ROOT, build_screening_plan, canonical_fingerprint, load_protocol


DEFAULT_OUTPUT = REPO_ROOT / "zvf-program/flagship/pilot/plans-v2-corpus-resume-r4-2"


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_manifest() -> dict[str, object]:
    protocol = load_protocol()
    plans = [build_screening_plan(protocol, unit) for unit in protocol.screening_units()]
    manifest: dict[str, object] = {
        "schema_version": "flagship-pilot-screening-manifest-v1",
        "status": "dry_run_only",
        "protocol_sha256": protocol.sha256,
        "unit_count": len(plans),
        "allocation_allowed": False,
        "units": plans,
    }
    manifest["fingerprint"] = canonical_fingerprint(manifest)
    return manifest


def write_manifest(output_dir: Path = DEFAULT_OUTPUT) -> Path:
    manifest = build_manifest()
    for plan in manifest["units"]:
        unit_id = plan["unit"]["id"]
        atomic_json(output_dir / "units" / f"{unit_id}.json", plan)
    path = output_dir / "screening_manifest.json"
    atomic_json(path, manifest)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the fail-closed 24-unit flagship pilot screening plan."
    )
    parser.add_argument("--write", action="store_true", help="atomically write plan JSON files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.write:
        path = write_manifest(args.output_dir)
        print(path)
    else:
        print(json.dumps(build_manifest(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
