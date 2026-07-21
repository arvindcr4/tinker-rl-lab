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
        self.assertEqual(self.protocol["schema_version"], "zvf-flagship-v1")
        self.assertEqual(self.protocol["status"], "frozen-screening-not-started")
        self.assertEqual(self.protocol["stages"]["S0_evidence_sync"]["status"], "complete")
        for stage in ("S1_objective_differential", "S2_screening", "S3_confirmatory"):
            self.assertEqual(self.protocol["stages"][stage]["status"], "not_started")

    def test_training_seed_sets_are_disjoint_and_complete(self):
        screening = self.protocol["seeds"]["screening"]
        confirmatory = self.protocol["seeds"]["confirmatory"]
        self.assertEqual(len(screening), 3)
        self.assertEqual(len(confirmatory), 5)
        self.assertFalse(set(screening) & set(confirmatory))

    def test_public_sources_are_exact_revisions(self):
        revisions = {
            **self.protocol["frozen_sources"]["models"],
            **self.protocol["frozen_sources"]["datasets"],
        }
        self.assertEqual(len(revisions), 5)
        for name, revision in revisions.items():
            with self.subTest(name=name):
                self.assertRegex(revision, re.compile(r"^[0-9a-f]{40}$"))

    def test_scope_contains_required_external_validity_axes(self):
        self.assertEqual(
            set(self.protocol["tasks"]), {"gsm8k_binary", "math500_sparse", "mbpp_graded"}
        )
        self.assertEqual(len(self.protocol["training"]["confirmatory_models"]), 2)
        self.assertEqual(
            {stack["name"] for stack in self.protocol["frozen_sources"]["stacks"].values()},
            {"trl", "verl"},
        )

    def test_compute_expansion_is_fail_closed(self):
        screening_gate = self.protocol["stages"]["S2_screening"]["expansion_gate"]
        self.assertIn("Advance only if", screening_gate)
        self.assertTrue(
            any("Do not start S3" in condition for condition in self.protocol["stop_conditions"])
        )
        self.assertTrue(
            any("Do not purchase" in condition for condition in self.protocol["stop_conditions"])
        )

    def test_confirmatory_claim_requires_effect_and_non_inferiority(self):
        acceptance = self.protocol["stages"]["S3_confirmatory"]["acceptance"]
        self.assertIn("excludes zero", acceptance)
        self.assertIn("non-inferior", acceptance)
        self.assertIn("FLOPs-to-target", acceptance)


if __name__ == "__main__":
    unittest.main()
