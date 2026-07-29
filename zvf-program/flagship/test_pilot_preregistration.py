from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "zvf-program/flagship/pilot_preregistration.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PilotPreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(PROTOCOL_PATH.read_text())

    def test_parent_and_gate_evidence_hashes_match(self) -> None:
        for key in ("parent_protocol", "s1_freeze"):
            record = self.protocol[key]
            self.assertEqual(sha256(ROOT / record["path"]), record["sha256"])
        theory = self.protocol["theory_gate"]
        self.assertEqual(sha256(ROOT / theory["audit_path"]), theory["audit_sha256"])
        freeze = json.loads((ROOT / self.protocol["s1_freeze"]["path"]).read_text())
        self.assertEqual(freeze["status"], self.protocol["s1_freeze"]["required_status"])

    def test_versioned_zero_relation_amendment_and_corpus_reuse_are_explicit(self) -> None:
        self.assertEqual(self.protocol["protocol_version"], 2)
        self.assertEqual(self.protocol["implementation_revision"], 7)
        amendment = self.protocol["amendment"]
        self.assertEqual(amendment["id"], "A1-corpus-intermediate-persistence")
        self.assertTrue(amendment["authorized_by_user"])
        self.assertRegex(amendment["previous_protocol_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(len(amendment["prior_attempts"]), 3)
        self.assertIn("never pooled", amendment["evidence_treatment"])
        corrections = amendment["control_plane_corrections"]
        r4 = corrections[-3]
        self.assertEqual(
            r4["id"],
            "A1-R4-explicit-degenerate-gradient-relations-with-corpus-reuse",
        )
        self.assertTrue(r4["authorized_by_user"])
        self.assertIn("joint-zero", r4["observation"])
        self.assertIn("No replay group is dropped", r4["correction"])
        self.assertEqual(
            corrections[-2]["id"],
            "A1-R4.1-package-frozen-archives-for-remote-validation",
        )
        self.assertEqual(corrections[-2]["authorized_under"], r4["id"])
        self.assertIn("No W&B run", corrections[-2]["observation"])
        self.assertEqual(
            corrections[-1]["id"],
            "A1-R4.2-exact-identical-gradient-diagnostics",
        )
        self.assertEqual(corrections[-1]["authorized_under"], r4["id"])
        self.assertIn("1.000000000002599", corrections[-1]["observation"])
        binding = self.protocol["corpus_reuse_binding"]
        self.assertEqual(sha256(ROOT / binding["path"]), binding["sha256"])
        self.assertEqual(
            sha256(ROOT / binding["frozen_source_archive_path"]),
            binding["frozen_source_archive_sha256"],
        )

    def test_screening_matrix_is_exactly_twenty_four_units(self) -> None:
        count = self.protocol["unit_count"]
        self.assertEqual(count["conditions"] * count["regimes"] * count["seeds"], 24)
        self.assertEqual(count["total"], 24)
        self.assertEqual(len(self.protocol["conditions"]), 4)
        self.assertEqual(len(self.protocol["regimes"]), 2)
        self.assertEqual(len(self.protocol["runtime"]["screening_seeds"]), 3)

    def test_screening_and_confirmatory_seeds_are_disjoint(self) -> None:
        screening = set(self.protocol["runtime"]["screening_seeds"])
        confirmatory = set(self.protocol["runtime"]["confirmatory_seeds"])
        self.assertEqual(len(confirmatory), 5)
        self.assertTrue(screening.isdisjoint(confirmatory))

    def test_gate_requires_direction_not_only_scalar_loss_change(self) -> None:
        question = self.protocol["central_question"]
        mechanism = self.protocol["screening_gate"]["mechanism"]
        primary = self.protocol["measurements"]["primary_mechanism"]
        self.assertIn("gradient direction", question)
        self.assertIn("cosine", primary)
        self.assertIn("relative L2", primary)
        self.assertIn("frozen nonzero gradient-discrepancy thresholds", mechanism)
        self.assertIn("reduction_only", self.protocol["screening_gate"]["causal_attribution"])
        self.assertIn("epsilon_only", self.protocol["screening_gate"]["causal_attribution"])

    def test_compute_matching_and_provenance_fail_closed(self) -> None:
        matching = self.protocol["matched_compute"]
        self.assertIn("masked", matching["charged_tokens"])
        self.assertIn("FLOPs", matching["flops"])
        self.assertGreaterEqual(len(matching["frozen_fields"]), 8)
        self.assertIn("not a scientific observation", matching["fail_closed"])
        self.assertIn("Local, W&B, and Hugging Face", self.protocol["provenance"]["acceptance"])

    def test_power_and_equivalence_are_explicit(self) -> None:
        power = self.protocol["power"]
        self.assertIn("100000 Monte Carlo", power["confirmatory"])
        self.assertIn("0.80 power", power["confirmatory"])
        self.assertIn("TOST", power["confirmatory"])
        self.assertIn("INCONCLUSIVE", power["equivalence"])

    def test_gpu_authorization_is_smoke_scoped_and_gated(self) -> None:
        self.assertEqual(self.protocol["status"], "ready_to_run")
        self.assertTrue(self.protocol["authorization"]["gpu"])
        self.assertIn(
            "A fresh implementation-revision-7 non-scientific A100 smoke",
            self.protocol["authorization"]["scope"],
        )
        self.assertIn("no confirmatory jobs", self.protocol["authorization"]["scope"])

    def test_execution_contract_resolves_fixed_step_and_token_matching(self) -> None:
        contract = self.protocol["runtime"]["execution_contract"]
        self.assertEqual(contract["design"], "shared_frozen_offpolicy_replay")
        self.assertEqual(contract["accepted_groups_per_corpus"], 100)
        self.assertEqual(contract["charged_generated_token_ceiling"], 819200)
        self.assertIn("all 100 groups", contract["matched_budget_horizon_rule"])
        self.assertIn("charged identically", contract["matched_budget_horizon_rule"])
        self.assertEqual(contract["heldout_n"], 128)
        self.assertEqual(contract["replay_gradient_steps"]["count"], 100)
        self.assertEqual(contract["generation_batch_size"]["filtered_variable_length"], 16)
        self.assertIn("non-scientific A100", contract["preflight"])
        self.assertIn("blocks all six corpus jobs", contract["preflight"])
        resume = contract["corpus_checkpoint_resume_contract"]
        self.assertEqual(resume["checkpoint_groups"], [20, 40, 60, 80])
        self.assertEqual(resume["storage_prefix"], "resume/")
        self.assertEqual(resume["attempt_limit"], 3)
        self.assertEqual(resume["max_parallel_corpus_sessions"], 1)
        self.assertIn("exact protocol", resume["resume_validation"])
        self.assertIn("does not change any generated group", resume["determinism"])

    def test_dataset_revisions_and_train_orders_are_frozen(self) -> None:
        for regime in self.protocol["regimes"].values():
            self.assertRegex(regime["dataset_revision"], r"^[0-9a-f]{40}$")
            for digest in regime["source_ordered_row_sha256"].values():
                self.assertRegex(digest, r"^[0-9a-f]{64}$")
        hashes = self.protocol["runtime"]["execution_contract"]["train_order_hash"]
        for regime in ("balanced_equal_length", "filtered_variable_length"):
            self.assertEqual(set(hashes[regime]), {"11", "23", "37"})
            for digest in hashes[regime].values():
                self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_remote_stack_is_fully_pinned_to_the_resolved_set(self) -> None:
        self.assertEqual(self.protocol["runtime"]["python"], ">=3.11,<3.13")
        pins = set(self.protocol["runtime"]["package_pins"])
        self.assertEqual(
            pins,
            {
                "trl==1.2.0",
                "transformers==5.5.4",
                "torch==2.7.1",
                "datasets==4.8.4",
                "peft==0.19.1",
                "huggingface-hub==1.11.0",
                "accelerate==1.13.0",
                "wandb==0.21.0",
                "numpy==2.2.6",
            },
        )

    def test_tracking_namespaces_are_frozen_and_private(self) -> None:
        tracking = self.protocol["runtime"]["execution_contract"]["tracking"]
        self.assertEqual(tracking["wandb_project"], "tinker-rl-lab")
        self.assertEqual(tracking["wandb_unit_group"], "flagship-s1-conformance-screening")
        self.assertTrue(tracking["hugging_face_private"])


if __name__ == "__main__":
    unittest.main()
