"""Combine two exact stack receipts into the fail-closed S1 freeze manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .receipt import INTENDED_CASES, _fixture_digest, validate_controller_matrix
from .reference import RTOL, ATOL


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combine(trl_path: Path, verl_path: Path) -> dict[str, Any]:
    trl_bytes, verl_bytes = trl_path.read_bytes(), verl_path.read_bytes()
    trl, verl = json.loads(trl_bytes), json.loads(verl_bytes)
    errors: list[str] = []
    if trl.get("stack") != "trl" or verl.get("stack") != "verl":
        errors.append("receipt stack labels are invalid")
    if trl.get("status") != "PASS" or verl.get("status") != "PASS":
        errors.append("both intended stack receipts must pass")
    for field in (
        "tolerances",
        "fixture_digest",
        "controller_matrix",
        "controller_action_ontology",
    ):
        if trl.get(field) != verl.get(field):
            errors.append(f"cross-stack field differs: {field}")
    for stack, receipt in (("trl", trl), ("verl", verl)):
        root = Path(__file__).resolve().parent
        if receipt.get("schema_version") != 2:
            errors.append(f"{stack}: receipt schema v2 required")
        if (
            receipt.get("tolerances") != {"rtol": RTOL, "atol": ATOL, "dtype": "float64"}
            or receipt.get("fixture_digest") != _fixture_digest()
        ):
            errors.append(f"{stack}: frozen fixture identity mismatch")
        expected_cases = [(arm, fixture.name) for arm, fixture in INTENDED_CASES] + [
            ("grpo", "base_selected_0_1_3")
        ]
        actual_cases = [
            (row.get("arm"), row.get("fixture")) for row in receipt.get("intended_cases", [])
        ]
        if sorted(actual_cases, key=str) != sorted(expected_cases) or len(actual_cases) != len(
            set(actual_cases)
        ):
            errors.append(f"{stack}: exact intended case roster required")
        required_sources = [
            "reference.py",
            "fixtures.py",
            "S1_AMENDMENT.md",
            "receipt.py",
            "test_reference.py",
            "test_receipts.py",
            f"{stack}_adapter.py",
            f"test_{stack}_adapter.py",
        ]
        if any(
            receipt.get("source_hashes", {}).get(name) != _sha256(root / name)
            for name in required_sources
        ):
            errors.append(f"{stack}: required source hashes mismatch")
        for case in receipt.get("intended_cases", []):
            fields = case.get("fields", [])
            required_fields = {"advantages", "ratios", "mask", "surrogate", "loss", "gradient"}
            if (
                case.get("conforms") is not True
                or {field.get("field") for field in fields} != required_fields
                or len(fields) != len(required_fields)
                or any(field.get("agrees") is not True for field in fields)
            ):
                errors.append(f"{stack}: complete numerical objective evidence required")
            for name in ("actual_loss", "expected_loss"):
                value = case.get(name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    errors.append(f"{stack}: finite objective loss required")
        provenance = receipt.get("provenance")
        if not isinstance(provenance, dict) or not provenance:
            errors.append(f"{stack}: imported source provenance required")
        else:
            from . import trl_adapter, verl_adapter

            pinned = (
                {
                    "trainer_source_sha256": trl_adapter.GRPO_TRAINER_SHA256,
                    "config_source_sha256": trl_adapter.GRPO_CONFIG_SHA256,
                    "locked_wheel_sha256": trl_adapter.TRL_WHEEL_SHA256,
                    "trl_version": trl_adapter.TRL_VERSION,
                    "transformers_version": trl_adapter.TRANSFORMERS_VERSION,
                }
                if stack == "trl"
                else {
                    "core_algos_sha256": verl_adapter.CORE_ALGOS_SHA256,
                    "torch_functional_sha256": verl_adapter.TORCH_FUNCTIONAL_SHA256,
                    "metadata_sha256": verl_adapter.METADATA_SHA256,
                    "verl_version": verl_adapter.VERL_VERSION,
                    "transformers_version": verl_adapter.TRANSFORMERS_VERSION,
                }
            )
            if any(provenance.get(k) != v for k, v in pinned.items()):
                errors.append(f"{stack}: imported runtime differs from frozen source identities")
            source_keys = (
                ("trainer_source", "config_source")
                if stack == "trl"
                else ("core_algos_source", "torch_functional_source")
            )
            for key in source_keys:
                try:
                    hash_key = (
                        key + "_sha256" if stack == "trl" else key.replace("_source", "_sha256")
                    )
                    if _sha256(Path(provenance[key])) != provenance[hash_key]:
                        errors.append(f"{stack}: imported source hash mismatch: {key}")
                except (KeyError, OSError, TypeError):
                    errors.append(f"{stack}: imported source unavailable: {key}")
        if not validate_controller_matrix(receipt.get("controller_matrix")):
            errors.append(f"{stack}: controller decisions differ from frozen expectations")
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
        "status": "KERNEL_FIXTURES_PASS" if not errors else "S1_FAIL",
        "training_integration_status": "NOT_TESTED",
        "scientific_claim_eligible": False,
        "errors": errors,
        "trl_receipt": {"path": str(trl_path), "sha256": hashlib.sha256(trl_bytes).hexdigest()},
        "verl_receipt": {"path": str(verl_path), "sha256": hashlib.sha256(verl_bytes).hexdigest()},
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
    return 0 if manifest["status"] == "KERNEL_FIXTURES_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
