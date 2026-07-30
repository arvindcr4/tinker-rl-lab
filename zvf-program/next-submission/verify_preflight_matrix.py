#!/usr/bin/env python3
"""Fail-closed gate between non-evidence preflights and confirmatory execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = HERE / "preregistration.json"
DEFAULT_RESULTS = HERE / "results/preflight/results"
SOURCE_BINDINGS = {
    "contrast_sampler.py": "contrast_sampler_sha256",
    "remote_preflight.py": "remote_preflight_sha256",
    "trl_sampler_adapter.py": "trl_sampler_adapter_sha256",
}


class PreflightGateError(ValueError):
    """Raised when a receipt matrix cannot safely gate confirmatory execution."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightGateError(message)


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_object)
    require(isinstance(payload, dict), f"{path} must contain a JSON object")
    return payload


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_cells(protocol: dict[str, Any]) -> set[tuple[str, str]]:
    matrix = protocol.get("matrix")
    require(isinstance(matrix, list) and matrix, "protocol matrix is missing")
    cells: set[tuple[str, str]] = set()
    for row in matrix:
        require(isinstance(row, dict), "protocol matrix row must be an object")
        task = row.get("task_id")
        arm = row.get("arm_id")
        require(isinstance(task, str) and isinstance(arm, str), "protocol cell identity missing")
        require((task, arm) not in cells, f"duplicate protocol cell: {task}/{arm}")
        cells.add((task, arm))
    return cells


def _validate_audit(audit: Any, task: str, arm: str, seed: int) -> dict[str, Any]:
    require(isinstance(audit, dict), "preflight audit record missing")
    require(
        audit.get("task_id") == task and audit.get("arm_id") == arm and audit.get("seed") == seed,
        "preflight audit identity drift",
    )
    require(audit.get("status") == "complete", "preflight audit is not complete")
    require(
        audit.get("evidence_tier") == "preflight_not_scientific_evidence",
        "preflight audit was promoted to scientific evidence",
    )
    for field in (
        "all_correct_fraction",
        "all_wrong_fraction",
        "completion_clipped_fraction",
        "mixed_fraction",
    ):
        value = audit.get(field)
        require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= 1,
            f"invalid {field}",
        )
    for field in ("generated_rollouts", "rollout_groups", "updated_groups"):
        value = audit.get(field)
        require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"invalid {field}",
        )
    require(
        audit["completion_clipped_fraction"] < 1,
        "all-clipped preflight cannot clear the infrastructure gate",
    )
    require(
        audit["updated_groups"] <= audit["rollout_groups"],
        "updated groups exceed rollout groups",
    )
    require(
        math.isclose(
            audit["all_correct_fraction"] + audit["all_wrong_fraction"] + audit["mixed_fraction"],
            1.0,
            abs_tol=1e-9,
        ),
        "group fractions do not sum to one",
    )
    require(
        math.isclose(
            audit["mixed_fraction"] * audit["rollout_groups"],
            audit["updated_groups"],
            abs_tol=1e-9,
        ),
        "mixed-group fraction and updated-group count disagree",
    )
    return audit


def validate_receipt(
    receipt: dict[str, Any],
    protocol: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    require(
        receipt.get("schema_version") == "aiml-next-preflight-request-v1",
        f"{path.name}: receipt schema drift",
    )
    task = receipt.get("task")
    arm = receipt.get("arm")
    seed = receipt.get("seed")
    require(
        isinstance(task, str)
        and isinstance(arm, str)
        and isinstance(seed, int)
        and not isinstance(seed, bool),
        f"{path.name}: receipt identity missing",
    )
    require(receipt.get("status") == "completed", f"{path.name}: receipt is not completed")
    require(receipt.get("return_code") == 0, f"{path.name}: remote process did not succeed")
    require(receipt.get("failed_step") is None, f"{path.name}: failed step is still recorded")
    require(receipt.get("allocation_started") is True, f"{path.name}: allocation was not recorded")

    cleanup = receipt.get("cleanup")
    require(isinstance(cleanup, dict), f"{path.name}: cleanup receipt missing")
    require(
        cleanup.get("session_absent_verified") is True,
        f"{path.name}: provider session absence was not verified",
    )

    payload = receipt.get("payload")
    manifest = receipt.get("manifest")
    require(isinstance(payload, dict), f"{path.name}: result payload missing")
    require(isinstance(manifest, dict), f"{path.name}: remote manifest missing")
    require(
        payload.get("schema_version") == "aiml-next-preflight-result-v1",
        f"{path.name}: result payload schema drift",
    )
    require(
        manifest.get("schema_version") == "aiml-next-preflight-run-v1",
        f"{path.name}: remote manifest schema drift",
    )
    require(payload.get("status") == "completed", f"{path.name}: result payload is incomplete")
    require(
        payload.get("evidence_class") == "preflight-not-evidence"
        and manifest.get("evidence_class") == "preflight-not-evidence",
        f"{path.name}: preflight was promoted to scientific evidence",
    )
    require(manifest.get("status") == "complete", f"{path.name}: remote manifest is incomplete")

    run_config = payload.get("run_config")
    manifest_config = manifest.get("run_config")
    require(isinstance(run_config, dict), f"{path.name}: run config missing")
    require(run_config == manifest_config, f"{path.name}: payload/manifest run config drift")
    require(
        run_config.get("task") == task
        and run_config.get("arm") == arm
        and run_config.get("seed") == seed,
        f"{path.name}: run config identity drift",
    )
    require(
        run_config.get("unit_fingerprint") == receipt.get("fingerprint"),
        f"{path.name}: unit fingerprint drift",
    )
    require(
        isinstance(receipt.get("fingerprint"), str)
        and re.fullmatch(r"[0-9a-f]{64}", receipt["fingerprint"]) is not None,
        f"{path.name}: invalid unit fingerprint",
    )
    require(
        run_config.get("stack_fingerprint") == receipt.get("stack_fingerprint"),
        f"{path.name}: stack fingerprint drift",
    )
    require(
        isinstance(receipt.get("stack_fingerprint"), str)
        and re.fullmatch(r"[0-9a-f]{64}", receipt["stack_fingerprint"]) is not None,
        f"{path.name}: invalid stack fingerprint",
    )
    require(
        run_config.get("source_commit") == receipt.get("source_commit"),
        f"{path.name}: source commit drift",
    )
    require(
        run_config.get("protocol_sha256") == receipt.get("protocol_sha256"),
        f"{path.name}: protocol hash drift within receipt",
    )
    require(
        run_config.get("decoder") == protocol.get("treatment", {}).get("decoder"),
        f"{path.name}: decoder differs from the frozen protocol",
    )

    audit = _validate_audit(payload.get("audit_record"), task, arm, seed)
    require(audit == manifest.get("audit_record"), f"{path.name}: payload/manifest audit drift")

    source_files = manifest.get("source_files")
    bindings = protocol.get("bindings")
    require(
        isinstance(source_files, dict) and isinstance(bindings, dict),
        f"{path.name}: source provenance missing",
    )
    for filename, binding_key in SOURCE_BINDINGS.items():
        require(
            source_files.get(filename) == bindings.get(binding_key),
            f"{path.name}: {filename} differs from the frozen source binding",
        )

    remote = payload.get("remote")
    verification = receipt.get("remote_verification")
    require(
        isinstance(remote, dict) and isinstance(verification, dict),
        f"{path.name}: remote reconciliation missing",
    )
    require(verification.get("hf_private") is True, f"{path.name}: HF repository is not private")
    require(
        verification.get("hf_repo") == remote.get("hf_repo") == receipt.get("hf_repo"),
        f"{path.name}: HF repository identity drift",
    )
    require(
        verification.get("hf_commit") == remote.get("hf_commit"),
        f"{path.name}: HF commit identity drift",
    )
    require(
        isinstance(remote.get("hf_commit"), str)
        and re.fullmatch(r"[0-9a-f]{40}", remote["hf_commit"]) is not None,
        f"{path.name}: invalid HF commit",
    )
    require(
        {"run_manifest.json", "final/adapter_model.safetensors"}.issubset(
            set(verification.get("hf_files") or [])
        ),
        f"{path.name}: required HF artifacts are missing",
    )
    wandb = verification.get("wandb")
    require(isinstance(wandb, dict), f"{path.name}: W&B reconciliation missing")
    require(wandb.get("state") == "finished", f"{path.name}: W&B run is not finished")
    require(
        wandb.get("run_id") == remote.get("wandb_run_id"),
        f"{path.name}: W&B run identity drift",
    )
    require(
        wandb.get("run_url") == remote.get("wandb_run_url"),
        f"{path.name}: W&B run URL drift",
    )

    return {
        "task_id": task,
        "arm_id": arm,
        "seed": seed,
        "fingerprint": receipt["fingerprint"],
        "stack_fingerprint": receipt["stack_fingerprint"],
        "receipt_path": path.name,
        "receipt_sha256": sha256(path),
        "completion_clipped_fraction": audit["completion_clipped_fraction"],
        "homogeneous_stop_observed": (
            arm == "contrast_early_stop_g2_to_g8"
            and audit["all_correct_fraction"] + audit["all_wrong_fraction"] > 0
        ),
        "mixed_update_observed": audit["mixed_fraction"] > 0 and audit["updated_groups"] > 0,
        "updated_groups": audit["updated_groups"],
    }


def evaluate_matrix(
    protocol: dict[str, Any],
    receipt_paths: list[Path],
) -> dict[str, Any]:
    cells = expected_cells(protocol)
    require(receipt_paths, "no preflight receipts found")

    records: list[dict[str, Any]] = []
    identities: set[tuple[str, str, int]] = set()
    for path in sorted(receipt_paths):
        record = validate_receipt(load_json(path), protocol, path)
        cell = (record["task_id"], record["arm_id"])
        require(cell in cells, f"unexpected preflight cell: {cell[0]}/{cell[1]}")
        identity = (*cell, record["seed"])
        require(identity not in identities, f"duplicate preflight identity: {identity}")
        identities.add(identity)
        records.append(record)

    observed_cells = {(record["task_id"], record["arm_id"]) for record in records}
    missing_cells = sorted(cells - observed_cells)
    require(not missing_cells, f"preflight matrix is incomplete: {missing_cells}")
    stack_fingerprints = {record["stack_fingerprint"] for record in records}
    require(
        len(stack_fingerprints) == 1,
        "preflight receipts do not share one frozen scientific stack",
    )

    cell_reports: list[dict[str, Any]] = []
    missing_scientific_seams: list[str] = []
    for task, arm in sorted(cells):
        cell_records = [
            record for record in records if record["task_id"] == task and record["arm_id"] == arm
        ]
        mixed_update = any(record["mixed_update_observed"] for record in cell_records)
        homogeneous_stop = any(record["homogeneous_stop_observed"] for record in cell_records)
        if not mixed_update:
            missing_scientific_seams.append(f"{task}/{arm}:mixed_reward_optimizer_update")
        if arm == "contrast_early_stop_g2_to_g8" and not homogeneous_stop:
            missing_scientific_seams.append(f"{task}/{arm}:homogeneous_early_stop")
        cell_reports.append(
            {
                "task_id": task,
                "arm_id": arm,
                "seeds": sorted(record["seed"] for record in cell_records),
                "receipt_count": len(cell_records),
                "infrastructure_verified": True,
                "mixed_reward_optimizer_update_observed": mixed_update,
                "homogeneous_early_stop_observed": homogeneous_stop,
                "receipts": [
                    {
                        "path": record["receipt_path"],
                        "sha256": record["receipt_sha256"],
                        "fingerprint": record["fingerprint"],
                    }
                    for record in cell_records
                ],
            }
        )

    ready = not missing_scientific_seams
    return {
        "schema_version": "aiml-next-preflight-gate-v1",
        "evidence_class": "preflight-gate-not-scientific-evidence",
        "infrastructure_matrix_verified": True,
        "scientific_seams_verified": ready,
        "confirmatory_execution_gate": "pass" if ready else "blocked",
        "stack_fingerprint": next(iter(stack_fingerprints)),
        "expected_cell_count": len(cells),
        "receipt_count": len(records),
        "missing_scientific_seams": missing_scientific_seams,
        "cells": cell_reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_json(args.protocol)
    report = evaluate_matrix(protocol, list(args.results_dir.glob("*.json")))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["confirmatory_execution_gate"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
