"""Emit one stack-local S1 conformance receipt without launching training."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .fixtures import (
    AERO_POSTERIOR_FIXTURE,
    ALL_CORRECT_FIXTURE,
    ALL_WRONG_FIXTURE,
    BASE_FIXTURE,
    DAPO_CLIP_FIXTURE,
    FIXTURES,
    GRADED_FIXTURE,
    LOW_CLIP_FIXTURE,
    TRANSLATED_FIXTURE,
    ZERO_MASK_FIXTURE,
)
from .reference import ATOL, RTOL, decide_policy, decide_policy_observation

ARMS = ("grpo", "dapo", "gspo", "drgrpo", "aero")
POLICIES = (
    "static_g8",
    "static_g16",
    "symmetric_zvf",
    "failure_only",
    "boundary_aware",
    "full_triage",
)

INTENDED_CASES = (
    ("grpo", BASE_FIXTURE),
    ("grpo", ALL_WRONG_FIXTURE),
    ("grpo", ALL_CORRECT_FIXTURE),
    ("grpo", GRADED_FIXTURE),
    ("grpo", TRANSLATED_FIXTURE),
    ("grpo", LOW_CLIP_FIXTURE),
    ("dapo", DAPO_CLIP_FIXTURE),
    ("dapo", LOW_CLIP_FIXTURE),
    ("gspo", BASE_FIXTURE),
    ("gspo", ZERO_MASK_FIXTURE),
    ("drgrpo", BASE_FIXTURE),
    ("drgrpo", TRANSLATED_FIXTURE),
    ("aero", AERO_POSTERIOR_FIXTURE),
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _fixture_digest() -> str:
    payload = {
        name: asdict(fixture)
        for name, fixture in sorted(FIXTURES.items())
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _compact(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = result.summary()
    provenance = summary.pop("provenance")
    return summary, provenance


def _controller_matrix() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    observed = {
        "all_wrong": [0.0, 0.0],
        "all_correct": [1.0, 1.0],
        "mixed": [0.0, 1.0],
    }
    for policy in POLICIES:
        for case, rewards in observed.items():
            decision = decide_policy(policy, rewards)
            rows.append({
                "policy": policy,
                "case": case,
                "status": "observed",
                "decision": asdict(decision),
                "charged_rollouts": decision.group_size,
            })
        for status in ("noisy", "missing", "delayed"):
            decision = decide_policy_observation(policy, [None, None], status=status)
            rows.append({
                "policy": policy,
                "case": status,
                "status": status,
                "decision": asdict(decision),
                "charged_rollouts": decision.group_size,
            })
    return rows


def build_receipt(stack: str) -> dict[str, Any]:
    if stack == "trl":
        from .trl_adapter import (
            TRLUnsupportedObjective,
            evaluate_fixture,
            evaluate_intended_fixture,
            load_pinned_runtime,
        )

        _, provenance_object = load_pinned_runtime()
        provenance = asdict(provenance_object)

        def native(arm: str, fixture: Any) -> dict[str, Any]:
            try:
                summary, _ = _compact(evaluate_fixture(fixture, arm))
                return summary
            except TRLUnsupportedObjective as error:
                return {
                    "fixture": fixture.name,
                    "arm": arm,
                    "verdict": "NOT_TESTED",
                    "not_tested_reason": str(error),
                }

        intended_evaluator = evaluate_intended_fixture
        adapter_path = Path(__file__).with_name("trl_adapter.py")
        adapter_test_path = Path(__file__).with_name("test_trl_adapter.py")
    elif stack == "verl":
        from .verl_adapter import evaluate_fixture, evaluate_intended_fixture, load_pinned_runtime

        _, _, provenance_object = load_pinned_runtime()
        provenance = asdict(provenance_object)

        def native(arm: str, fixture: Any) -> dict[str, Any]:
            summary, _ = _compact(evaluate_fixture(fixture, arm))
            return summary

        intended_evaluator = evaluate_intended_fixture
        adapter_path = Path(__file__).with_name("verl_adapter.py")
        adapter_test_path = Path(__file__).with_name("test_verl_adapter.py")
    else:
        raise ValueError(f"unknown stack: {stack}")

    native_cases = [
        native("grpo", BASE_FIXTURE),
        native("dapo", DAPO_CLIP_FIXTURE),
        native("gspo", BASE_FIXTURE),
        native("drgrpo", BASE_FIXTURE),
        native("aero", AERO_POSTERIOR_FIXTURE),
    ]
    intended_cases = []
    for arm, fixture in INTENDED_CASES:
        summary, _ = _compact(intended_evaluator(fixture, arm))
        intended_cases.append(summary)
    selected_summary, _ = _compact(
        intended_evaluator(BASE_FIXTURE, "grpo", selected_indices=(0, 1, 3))
    )
    selected_summary["fixture"] = "base_selected_0_1_3"
    intended_cases.append(selected_summary)

    root = Path(__file__).resolve().parent
    source_hashes = {
        path.name: _sha256_file(path)
        for path in (
            root / "reference.py",
            root / "fixtures.py",
            root / "S1_AMENDMENT.md",
            root / "receipt.py",
            root / "test_reference.py",
            root / "test_receipts.py",
            adapter_path,
            adapter_test_path,
        )
    }
    controllers = _controller_matrix()
    action_set = sorted({row["decision"]["action"] for row in controllers})
    intended_pass = all(case["verdict"] == "PASS" for case in intended_cases)
    controller_pass = action_set == ["drop", "escalate", "keep", "recheck"]
    return {
        "schema_version": 1,
        "stack": stack,
        "status": "PASS" if intended_pass and controller_pass else "FAIL",
        "tolerances": {"rtol": RTOL, "atol": ATOL, "dtype": "float64"},
        "fixture_digest": _fixture_digest(),
        "source_hashes": source_hashes,
        "provenance": provenance,
        "native_cases": native_cases,
        "intended_cases": intended_cases,
        "controller_matrix": controllers,
        "controller_action_ontology": {
            "retry": "escalate",
            "keep": "keep",
            "recheck": "recheck",
            "stop_collecting": "drop",
        },
        "controller_actions_observed": action_set,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", required=True, choices=("trl", "verl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = build_receipt(args.stack)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
