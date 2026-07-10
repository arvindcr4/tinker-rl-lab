from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_DIR))

import run_demo  # noqa: E402


class GroupMathTests(unittest.TestCase):
    def test_mixed_binary_group_has_unit_advantages(self) -> None:
        result = run_demo.compute_group([0.0, 1.0, 0.0, 1.0])
        self.assertEqual(result["mean_reward"], 0.5)
        self.assertEqual(result["std_reward"], 0.5)
        self.assertFalse(result["zero_variance"])
        self.assertEqual(result["advantages"], [-1.0, 1.0, -1.0, 1.0])

    def test_equal_reward_group_has_zero_advantages(self) -> None:
        result = run_demo.compute_group([1.0, 1.0, 1.0, 1.0])
        self.assertTrue(result["zero_variance"])
        self.assertEqual(result["advantages"], [0.0, 0.0, 0.0, 0.0])

    def test_fixture_contract(self) -> None:
        fixture = run_demo.load_json(run_demo.DEFAULT_FIXTURE)
        result = run_demo.analyze_fixture(fixture)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["group_count"], 4)
        self.assertEqual(result["effective_groups"], 2)
        self.assertTrue(math.isclose(result["zvf"], 0.5))


class ArtifactTests(unittest.TestCase):
    def test_recorded_artifact_contract(self) -> None:
        fixture = run_demo.load_json(run_demo.DEFAULT_FIXTURE)
        contract = fixture["artifact_contract"]
        artifact = run_demo.REPO_ROOT / contract["path_from_repo_root"]
        result = run_demo.audit_recorded_artifact(artifact, contract)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["reward_count"], 80)
        self.assertTrue(math.isclose(result["overall_accuracy"], 0.6875))
        self.assertTrue(math.isclose(result["overall_zvf"], 0.3))

    def test_hash_mismatch_is_rejected(self) -> None:
        fixture = run_demo.load_json(run_demo.DEFAULT_FIXTURE)
        contract = dict(fixture["artifact_contract"])
        contract["sha256"] = "0" * 64
        artifact = run_demo.REPO_ROOT / contract["path_from_repo_root"]
        with self.assertRaises(run_demo.DemoError):
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
            for first_path, second_path in zip(first_paths, second_paths):
                self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            parsed = json.loads(first_paths[0].read_text(encoding="utf-8"))
            self.assertEqual(parsed["demo_status"], "PASS")

    def test_live_json_extraction(self) -> None:
        self.assertEqual(run_demo._extract_json_object('prefix {"answer": 42} suffix')["answer"], 42)


if __name__ == "__main__":
    unittest.main()

