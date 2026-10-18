import json
import re
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


class FlagshipPreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((HERE / "preregistration.json").read_text())

    def test_protocol_is_frozen_before_screening(self):
        assert self.protocol["schema_version"] == "zvf-flagship-v1"
        assert self.protocol["status"] == "frozen-screening-not-started"
        assert self.protocol["stages"]["S0_evidence_sync"]["status"] == "complete"
        for stage in ("S1_objective_differential", "S2_screening", "S3_confirmatory"):
            assert self.protocol["stages"][stage]["status"] == "not_started"

    def test_training_seed_sets_are_disjoint_and_complete(self):
        screening = self.protocol["seeds"]["screening"]
        confirmatory = self.protocol["seeds"]["confirmatory"]
        assert len(screening) == 3
        assert len(confirmatory) == 5
        assert not set(screening) & set(confirmatory)

    def test_public_sources_are_exact_revisions(self):
        revisions = {
            **self.protocol["frozen_sources"]["models"],
            **self.protocol["frozen_sources"]["datasets"],
        }
        assert len(revisions) == 5
        for name, revision in revisions.items():
            with self.subTest(name=name):
                assert re.search(re.compile(r"^[0-9a-f]{40}$"), revision)

    def test_scope_contains_required_external_validity_axes(self):
        assert set(self.protocol["tasks"]) == {"gsm8k_binary", "math500_sparse", "mbpp_graded"}
        assert len(self.protocol["training"]["confirmatory_models"]) == 2
        assert {stack["name"] for stack in self.protocol["frozen_sources"]["stacks"].values()} == {
            "trl",
            "verl",
        }

    def test_compute_expansion_is_fail_closed(self):
        screening_gate = self.protocol["stages"]["S2_screening"]["expansion_gate"]
        assert "Advance only if" in screening_gate
        assert any("Do not start S3" in condition for condition in self.protocol["stop_conditions"])
        assert any("Do not purchase" in condition for condition in self.protocol["stop_conditions"])

    def test_confirmatory_claim_requires_effect_and_non_inferiority(self):
        acceptance = self.protocol["stages"]["S3_confirmatory"]["acceptance"]
        assert "excludes zero" in acceptance
        assert "non-inferior" in acceptance
        assert "FLOPs-to-target" in acceptance


if __name__ == "__main__":
    unittest.main()
