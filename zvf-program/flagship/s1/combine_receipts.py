"""Combine two exact stack receipts into the fail-closed S1 freeze manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combine(trl_path: Path, verl_path: Path) -> dict[str, Any]:
    trl, verl = _load(trl_path), _load(verl_path)
    errors: list[str] = []
    if trl.get("stack") != "trl" or verl.get("stack") != "verl":
        errors.append("receipt stack labels are invalid")
    if trl.get("status") != "PASS" or verl.get("status") != "PASS":
        errors.append("both intended stack receipts must pass")
    for field in ("tolerances", "fixture_digest", "controller_matrix", "controller_action_ontology"):
        if trl.get(field) != verl.get(field):
            errors.append(f"cross-stack field differs: {field}")
    for stack, receipt in (("trl", trl), ("verl", verl)):
        if not receipt.get("intended_cases"):
            errors.append(f"{stack} has no intended cases")
        if any(case.get("verdict") != "PASS" for case in receipt.get("intended_cases", [])):
            errors.append(f"{stack} contains a non-passing intended case")
        if not any(
            case.get("verdict") in {"MATERIAL_DIFFERENCE", "NOT_TESTED"}
            for case in receipt.get("native_cases", [])
        ):
            errors.append(f"{stack} native audit contains no preserved discrepancy")
    combiner_path = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "status": "S1_PASS" if not errors else "S1_FAIL",
        "errors": errors,
        "trl_receipt": {"path": str(trl_path), "sha256": _sha256(trl_path)},
        "verl_receipt": {"path": str(verl_path), "sha256": _sha256(verl_path)},
        "tolerances": trl.get("tolerances"),
        "fixture_digest": trl.get("fixture_digest"),
        "source_hashes": {"trl": trl.get("source_hashes"), "verl": verl.get("source_hashes")},
        "combiner_source": {"path": str(combiner_path), "sha256": _sha256(combiner_path)},
        "native_verdicts": {
            "trl": [case.get("verdict") for case in trl.get("native_cases", [])],
            "verl": [case.get("verdict") for case in verl.get("native_cases", [])],
        },
        "intended_case_count": {
            "trl": len(trl.get("intended_cases", [])),
            "verl": len(verl.get("intended_cases", [])),
        },
        "controller_case_count": len(trl.get("controller_matrix", [])),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trl", type=Path, required=True)
    parser.add_argument("--verl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = combine(args.trl, args.verl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0 if manifest["status"] == "S1_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
