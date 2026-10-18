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
from .reference import ATOL, RTOL, decide_policy_observation

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
    payload = {name: asdict(fixture) for name, fixture in sorted(FIXTURES.items())}
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _compact(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = result.summary()
    provenance = summary.pop("provenance")
    return summary, provenance


def controller_fixtures():
    """Frozen expected actions independent of the controller under test."""
    rows = []
    for policy in POLICIES:
        expected = {
            "static_g8": [("keep", 8)] * 5,
            "static_g16": [("keep", 16)] * 5,
            "symmetric_zvf": [
                ("escalate", 16),
                ("escalate", 16),
                ("keep", 8),
                ("escalate", 16),
                ("escalate", 16),
            ],
            "failure_only": [
                ("escalate", 16),
                ("keep", 8),
                ("keep", 8),
                ("escalate", 16),
                ("keep", 8),
            ],
            "boundary_aware": [
                ("escalate", 16),
                ("drop", 0),
                ("keep", 8),
                ("escalate", 16),
                ("drop", 0),
            ],
            "full_triage": [("escalate", 16), ("keep", 8), ("keep", 8), ("drop", 0), ("drop", 0)],
        }[policy]
        cases = [
            ("all_wrong", [0.0, 0.0], 0, 0),
            ("all_correct", [1.0, 1.0], 0, 0),
            ("mixed", [0.0, 1.0], 0, 0),
            ("failure_confidence", [0.0, 0.0], 0, 1000),
            ("mastery_confidence", [1.0, 1.0], 1000, 1000),
        ]
        for (case, rewards, successes, total), (action, size) in zip(cases, expected):
            rows.append(
                {
                    "policy": policy,
                    "case": case,
                    "status": "observed",
                    "rewards": rewards,
                    "history_successes": successes,
                    "history_total": total,
                    "expected_action": action,
                    "expected_group_size": size,
                }
            )
        for status in ("noisy", "missing", "delayed"):
            rows.append(
                {
                    "policy": policy,
                    "case": status,
                    "status": status,
                    "rewards": [None, None],
                    "history_successes": 0,
                    "history_total": 0,
                    "expected_action": "recheck",
                    "expected_group_size": 0,
                }
            )
    return rows


def _controller_matrix() -> list[dict[str, Any]]:
    rows = []
    for fixture in controller_fixtures():
        decision = decide_policy_observation(
            fixture["policy"],
            fixture["rewards"],
            status=fixture["status"],
            history_successes=fixture["history_successes"],
            history_total=fixture["history_total"],
        )
        rows.append(
            {
                **fixture,
                "decision": asdict(decision),
                "charged_rollouts": decision.group_size,
                "selected_ids": list(range(decision.group_size)),
            }
        )
    return rows


def validate_controller_matrix(rows) -> bool:
    expected = controller_fixtures()
    if not isinstance(rows, list) or len(rows) != len(expected):
        return False
    for actual, fixture in zip(rows, expected):
        if not isinstance(actual, dict) or any(actual.get(k) != v for k, v in fixture.items()):
            return False
        size = fixture["expected_group_size"]
        if (
            actual.get("decision", {}).get("action") != fixture["expected_action"]
            or actual.get("decision", {}).get("group_size") != size
            or actual.get("charged_rollouts") != size
            or actual.get("selected_ids") != list(range(size))
        ):
            return False
    return True


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
    controller_pass = validate_controller_matrix(controllers)
    return {
        "schema_version": 2,
        "evidence_class": "KERNEL_FIXTURES_PASS"
        if intended_pass and controller_pass
        else "KERNEL_FIXTURES_FAIL",
        "training_integration_status": "NOT_TESTED",
        "scientific_claim_eligible": False,
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
