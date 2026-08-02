#!/usr/bin/env python3
"""Offline verifier for the anonymous TMLR supplement."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest() -> dict[str, Any]:
    entries = []
    for line in (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = ROOT / name
        assert path.is_file(), name
        assert sha256(path) == digest, name
        entries.append(name)
    return {"status": "PASS", "files": len(entries)}


def homogeneous_probability(p: float, group_size: int) -> float:
    assert 0.0 <= p <= 1.0 and group_size >= 2
    return p**group_size + (1.0 - p) ** group_size


def informative_probability(p: float, group_size: int) -> float:
    edge = min(p, 1.0 - p)
    if edge == 0.0:
        return 0.0
    return -math.expm1(group_size * math.log1p(-edge)) - edge**group_size


def verify_math() -> dict[str, Any]:
    for group_size in (2, 4, 8, 16, 32):
        for p in (1e-8, 0.001, 0.1, 0.5, 0.9, 1.0 - 1e-8):
            q = informative_probability(p, group_size)
            assert math.isclose(q, 1.0 - homogeneous_probability(p, group_size), rel_tol=1e-9, abs_tol=2e-15)
    epsilon = 1e-8
    for group_size in (2, 4, 8, 16, 32):
        q = informative_probability(epsilon, group_size)
        assert abs(epsilon * group_size / q - 1.0) < 1e-6
        assert group_size / q >= 1.0 / epsilon - 1e-3
    retry_cost = 8 * 0.02
    recoverable = 1.0 * (1.0 - 0.9**8) - retry_cost
    regret = retry_cost * recoverable / (retry_cost + recoverable)
    assert recoverable > 0.0
    assert math.isclose(regret, 0.115050876, abs_tol=1e-12)
    return {"status": "PASS", "decision_reversal_regret": regret}


def verify_s1() -> dict[str, Any]:
    expected_native = {
        "trl": {"MATERIAL_DIFFERENCE": 4, "NOT_TESTED": 1},
        "verl": {"MATERIAL_DIFFERENCE": 1, "NOT_TESTED": 4},
    }
    summaries: dict[str, Any] = {}
    fixture_digests = set()
    for stack in ("trl", "verl"):
        receipt = load_json(ROOT / f"evidence/s1/{stack}_receipt.anonymous.json")
        assert receipt["stack"] == stack and receipt["status"] == "PASS"
        assert receipt["tolerances"] == {"atol": 1e-8, "dtype": "float64", "rtol": 1e-6}
        assert len(receipt["original_receipt_sha256"]) == 64
        assert len(receipt["intended_cases"]) == 14
        assert all(item["conforms"] and item["verdict"] == "PASS" for item in receipt["intended_cases"])
        assert len(receipt["controller_matrix"]) == 36
        assert {item["policy"] for item in receipt["controller_matrix"]} == {
            "static_g8", "static_g16", "symmetric_zvf", "failure_only",
            "boundary_aware", "full_triage",
        }
        assert {item["case"] for item in receipt["controller_matrix"]} == {
            "all_wrong", "all_correct", "mixed", "noisy", "missing", "delayed",
        }
        native = dict(Counter(item["verdict"] for item in receipt["native_cases"]))
        assert native == expected_native[stack]
        for name, digest in receipt["source_hashes"].items():
            assert sha256(ROOT / "evidence/s1/source" / name) == digest
        fixture_digests.add(receipt["fixture_digest"])
        summaries[stack] = {
            "intended_cases": 14,
            "controller_cases": 36,
            "native_verdicts": native,
        }
    assert len(fixture_digests) == 1
    return {"status": "PASS", "stacks": summaries}


def equivalent_steps(receipts: list[dict[str, Any]]) -> tuple[int, Counter[str], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    nonzero = []
    accepted = 0
    for item in receipts:
        relation = item["gradient_relation"]
        counts[relation] += 1
        if relation == "joint_zero":
            assert item["intended_gradient_norm"] == item["native_gradient_norm"] == 0.0
            assert item["gradient_cosine"] is None and item["gradient_relative_l2"] is None
            accepted += 1
        else:
            assert relation == "nonzero"
            assert item["intended_gradient_norm"] > 0.0 and item["native_gradient_norm"] > 0.0
            assert -1.0 <= item["gradient_cosine"] <= 1.0
            assert item["gradient_relative_l2"] >= 0.0
            nonzero.append(item)
            if item["gradient_cosine"] >= 0.999 and item["gradient_relative_l2"] <= 0.01:
                accepted += 1
    return accepted, counts, nonzero


def verify_r4_2() -> dict[str, Any]:
    design = load_json(ROOT / "evidence/r4_2/design_and_disposition.json")
    assert design["job_count"] == 31
    assert dict(Counter(design["job_statuses"].values())) == {
        "accepted": 10,
        "descoped_contract_infeasible": 14,
        "failed_infrastructure": 2,
        "failed_validation": 1,
        "pending_quota_reset": 4,
    }
    assert design["mechanism_gate"]["required_equivalent_steps"] == 95
    source = ROOT / "evidence/r4_2/source/r4-2-objective.py"
    assert sha256(source) == design["executed_objective_sha256"]

    expected = {
        "fpilot__epsilon_only__balanced_equal_length__s11": (20, 17, 62, 0.999858, 0.016844, 396672),
        "fpilot__reduction_only__balanced_equal_length__s11": (20, 19, 62, 0.999844, 0.017681, 396672),
        "fpilot__epsilon_only__balanced_equal_length__s23": (20, 17, 65, 0.999894, 0.014576, 400448),
        "fpilot__intended_full__balanced_equal_length__s23": (20, 20, 65, 0.999881, 0.015440, 400448),
        "fpilot__native_trl__balanced_equal_length__s23": (20, 17, 65, 0.999894, 0.014576, 400448),
        "fpilot__reduction_only__balanced_equal_length__s23": (20, 20, 65, 0.999881, 0.015440, 400448),
    }
    summaries = []
    for path in sorted((ROOT / "evidence/r4_2/units").glob("*.json")):
        unit = load_json(path)
        unit_id = unit["unit"]["id"]
        assert unit_id == path.stem and unit_id in expected
        assert unit["status"] == "accepted" and len(unit["original_acceptance_sha256"]) == 64
        receipts = unit["gradient_receipts"]
        assert len(receipts) == 100
        registered, counts, nonzero = equivalent_steps(receipts)
        evaluations = unit["evaluations"]
        assert [item["step"] for item in evaluations] == [0, 20, 40, 60, 80, 100]
        assert all(item["heldout_n"] == 128 and item["accuracy"] == item["correct"] / 128 for item in evaluations)
        ledger = unit["token_flop_ledger"]
        start, final, zeros, min_cos, max_l2, tokens = expected[unit_id]
        assert evaluations[0]["correct"] == start and evaluations[-1]["correct"] == final
        assert counts == Counter({"joint_zero": zeros, "nonzero": 100 - zeros})
        assert math.isclose(min(item["gradient_cosine"] for item in nonzero), min_cos, abs_tol=5e-7)
        assert math.isclose(max(item["gradient_relative_l2"] for item in nonzero), max_l2, abs_tol=5e-7)
        assert ledger["charged_generated_tokens"] == tokens
        summaries.append({"id": unit_id, "registered_equivalent_steps": registered})
    assert {item["id"] for item in summaries} == set(expected)
    intended = next(item for item in summaries if "intended_full" in item["id"])
    assert intended["registered_equivalent_steps"] == 69 < 95
    return {"status": "PASS", "units": len(summaries), "gradient_relations": 600, "intended_full_gate": "69/100 < 95/100"}


def verify_anchors() -> dict[str, Any]:
    anchors = load_json(ROOT / "evidence/provenance_anchors.json")
    for key, value in anchors.items():
        if key.endswith("sha256"):
            assert isinstance(value, str) and len(value) == 64
    return {"status": "PASS", "note": "format checked; unredacted source objects are withheld during double-blind review"}


def main() -> None:
    report = {
        "manifest": verify_manifest(),
        "math": verify_math(),
        "s1": verify_s1(),
        "r4_2": verify_r4_2(),
        "provenance_anchors": verify_anchors(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
