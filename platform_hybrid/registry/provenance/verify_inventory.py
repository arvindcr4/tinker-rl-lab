"""Inventory integrity only; historical rows never imply publication acceptance."""

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

try:
    from .sign import strict_json_loads
except ImportError:
    from sign import strict_json_loads


def verify(directory, trusted_key=None):
    manifest = strict_json_loads((directory / "required_records.json").read_text())
    rows = manifest["records"]
    names = [row["path"] for row in rows]
    if manifest["schema_version"] != 1 or not names or len(set(names)) != len(names):
        raise ValueError("invalid required provenance inventory")
    if any(Path(n).name != n or not n.endswith(".provenance.json") for n in names):
        raise ValueError("invalid provenance record path")
    if set(names) != {p.name for p in directory.glob("*.provenance.json")}:
        raise ValueError("missing or unexpected provenance record")
    if any(type(row["signature_required"]) is not bool for row in rows):
        raise ValueError("signature_required must be boolean")
    expected_signatures = {row["path"] + ".sig" for row in rows if row["signature_required"]}
    if expected_signatures != {p.name for p in directory.glob("*.provenance.json.sig")}:
        raise ValueError("missing or unexpected signature")
    for row in rows:
        record = directory / row["path"]
        strict_json_loads(record.read_text())
        mode = row.get("verification_mode", "publication")
        if mode == "historical_inventory":
            # Pin bytes for inventory drift detection, not authenticity or scientific validity.
            for artifact, key in [(record, "sha256")] + (
                [(Path(str(record) + ".sig"), "signature_sha256")]
                if row["signature_required"]
                else []
            ):
                strict_json_loads(artifact.read_text())
                if hashlib.sha256(artifact.read_bytes()).hexdigest() != row[key]:
                    raise ValueError("historical inventory digest mismatch")
        elif mode == "publication":
            if trusted_key is None:
                raise ValueError(
                    "publication inventory requires an independently trusted public key"
                )
            subprocess.run(
                [
                    sys.executable,
                    str(directory / "minreport.py"),
                    "verify",
                    str(record),
                    "--strict",
                    "--trusted-key",
                    str(trusted_key),
                ],
                check=True,
            )
            if row["signature_required"]:
                subprocess.run(
                    [
                        sys.executable,
                        str(directory / "sign.py"),
                        "verify",
                        str(record),
                        "--trusted-key",
                        str(trusted_key),
                    ],
                    check=True,
                )
        else:
            raise ValueError("unknown inventory verification mode")
    return {
        "inventory_valid": True,
        "publication_accepted": all(
            row.get("verification_mode", "publication") == "publication" for row in rows
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trusted-key", type=Path)
    args = parser.parse_args()
    print(verify(Path(__file__).resolve().parent, args.trusted_key))
