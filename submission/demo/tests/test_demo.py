from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_DIR))

import run_demo
import pytest


class GroupMathTests(unittest.TestCase):
    def test_mixed_binary_group_has_unit_advantages(self) -> None:
        result = run_demo.compute_group([0.0, 1.0, 0.0, 1.0])
        assert result["mean_reward"] == 0.5
        assert result["std_reward"] == 0.5
        assert not result["zero_variance"]
        assert result["advantages"] == [-1.0, 1.0, -1.0, 1.0]

    def test_equal_reward_group_has_zero_advantages(self) -> None:
        result = run_demo.compute_group([1.0, 1.0, 1.0, 1.0])
        assert result["zero_variance"]
        assert result["advantages"] == [0.0, 0.0, 0.0, 0.0]

    def test_fixture_contract(self) -> None:
        fixture = run_demo.load_json(run_demo.DEFAULT_FIXTURE)
        result = run_demo.analyze_fixture(fixture)
        assert result["status"] == "PASS"
        assert result["group_count"] == 4
        assert result["effective_groups"] == 2
        assert math.isclose(result["zvf"], 0.5)


class ArtifactTests(unittest.TestCase):
    def test_recorded_artifact_contract(self) -> None:
        fixture = run_demo.load_json(run_demo.DEFAULT_FIXTURE)
        contract = fixture["artifact_contract"]
        artifact = run_demo.REPO_ROOT / contract["path_from_repo_root"]
        result = run_demo.audit_recorded_artifact(artifact, contract)
        assert result["status"] == "PASS"
        assert result["reward_count"] == 80
        assert math.isclose(result["overall_accuracy"], 0.6875)
        assert math.isclose(result["overall_zvf"], 0.3)

    def test_hash_mismatch_is_rejected(self) -> None:
        fixture = run_demo.load_json(run_demo.DEFAULT_FIXTURE)
        contract = dict(fixture["artifact_contract"])
        contract["sha256"] = "0" * 64
        artifact = run_demo.REPO_ROOT / contract["path_from_repo_root"]
        with pytest.raises(run_demo.DemoError):
            run_demo.audit_recorded_artifact(artifact, contract)


class OutputTests(unittest.TestCase):
    def test_json_and_html_are_deterministic(self) -> None:
        fixture = run_demo.load_json(run_demo.DEFAULT_FIXTURE)
        mechanism = run_demo.analyze_fixture(fixture)
        contract = fixture["artifact_contract"]
        artifact = run_demo.audit_recorded_artifact(
            run_demo.REPO_ROOT / contract["path_from_repo_root"], contract
        )
        report = {
            "schema_version": 1,
            "demo_status": "PASS",
            "mode": "offline",
            "mechanism": mechanism,
            "artifact_audit": artifact,
            "claim_boundary": "test",
        }
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_paths = run_demo.write_report(report, Path(first))
            second_paths = run_demo.write_report(report, Path(second))
            for first_path, second_path in zip(first_paths, second_paths, strict=False):
                assert first_path.read_bytes() == second_path.read_bytes()
            parsed = json.loads(first_paths[0].read_text(encoding="utf-8"))
            assert parsed["demo_status"] == "PASS"

    def test_live_json_extraction(self) -> None:
        assert run_demo._extract_json_object('prefix {"answer": 42} suffix')["answer"] == 42


if __name__ == "__main__":
    unittest.main()
