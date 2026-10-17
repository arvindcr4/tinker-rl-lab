"""Fail-closed provenance inventory contract v1, followed by existing verifiers."""

import json
from pathlib import Path
import subprocess
import sys


def verify(directory):
    manifest = json.loads((directory / "required_records.json").read_text())
    rows = manifest["records"]
    names = [row["path"] for row in rows]
    if manifest["schema_version"] != 1 or not names or len(set(names)) != len(names):
        raise ValueError("invalid required provenance inventory")
    if any(Path(n).name != n or not n.endswith(".provenance.json") for n in names):
        raise ValueError("invalid provenance record path")
    if set(names) != {p.name for p in directory.glob("*.provenance.json")}:
        raise ValueError("missing or unexpected provenance record")
    expected_signatures = {row["path"] + ".sig" for row in rows if row["signature_required"]}
    if expected_signatures != {p.name for p in directory.glob("*.provenance.json.sig")}:
        raise ValueError("missing or unexpected signature")
    for row in rows:
        record = directory / row["path"]
        subprocess.run(
            [sys.executable, str(directory / "minreport.py"), "verify", str(record), "--strict"],
            check=True,
        )
        if row["signature_required"]:
            subprocess.run(
                [sys.executable, str(directory / "sign.py"), "verify", str(record)], check=True
            )


if __name__ == "__main__":
    verify(Path(__file__).resolve().parent)
