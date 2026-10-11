"""Verify the review package against its exact file inventory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> int:
    folder = Path(__file__).resolve().parent
    root = folder.parents[1]
    manifest = json.loads((folder / "REVIEW_MANIFEST.json").read_text())
    failures = []
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.resolve().is_relative_to(root) or not path.is_file():
            failures.append({"path": row["path"], "error": "missing or invalid path"})
            continue
        data = path.read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            failures.append({"path": row["path"], "error": "content mismatch"})
    print(
        json.dumps(
            {
                "status": "FAIL" if failures else "PASS",
                "files": len(manifest["files"]),
                "failures": failures,
            },
            indent=2,
        )
    )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
