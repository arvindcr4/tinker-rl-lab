from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from flagship.build_pavlovs_campaign_manifest import build_manifest
from flagship.pavlovs_domain_contract import load_contract


class PavlovsCampaignManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load_contract()

    def test_preview_is_blocked_and_never_launches_jobs(self) -> None:
        manifest = build_manifest(self.contract)
        self.assertEqual(manifest["status"], "BLOCKED")
        self.assertFalse(manifest["launches_any_job"])
        self.assertGreater(len(manifest["blockers"]), 0)
        self.assertTrue(manifest["budget_guard"]["status_reconciled"])
        self.assertFalse(
            any(
                "conflicts with paid_jobs_may_launch=true" in blocker
                for blocker in manifest["blockers"]
            )
        )
        self.assertEqual(manifest["budget_guard"]["maximum_usd"], 18.0)

    def test_every_company_inherits_a_primary_evaluation(self) -> None:
        manifest = build_manifest(self.contract)
        self.assertEqual(len(manifest["company_eval_coverage"]), 53)
        self.assertTrue(all(manifest["company_eval_coverage"].values()))
        self.assertEqual(len(manifest["company_train_coverage"]), 53)
        for coverage in manifest["company_domain_coverage"].values():
            self.assertTrue(all(coverage["training"].values()))
            self.assertTrue(all(coverage["primary_eval"].values()))

    def test_company_domain_coverage_is_not_satisfied_by_one_intersection(self) -> None:
        candidate = copy.deepcopy(self.contract)
        candidate["companies"][0]["domains"] = ["multi_domain", "not_declared"]
        with patch(
            "flagship.build_pavlovs_campaign_manifest.validate_contract",
            return_value=[],
        ):
            with self.assertRaisesRegex(
                ValueError, "required domains without training coverage"
            ):
                build_manifest(candidate)

    def test_gsm8k_is_not_a_training_or_primary_evaluation_suite(self) -> None:
        manifest = build_manifest(self.contract)
        self.assertEqual(manifest["gsm8k_role"], "calibration_only")
        self.assertFalse(
            any("gsm8k" in suite_id for suite_id in manifest["training_suite_ids"])
        )
        self.assertFalse(
            any(
                "gsm8k" in suite_id
                for suite_id in manifest["primary_evaluation_suite_ids"]
            )
        )

    def test_manifest_covers_all_training_and_primary_eval_suites_and_domains(self) -> None:
        manifest = build_manifest(self.contract)
        expected_training = sorted(
            suite_id
            for suite_id, suite in self.contract["suite_registry"].items()
            if suite["role"] == "train"
        )
        expected_primary_eval = sorted(
            suite_id
            for suite_id, suite in self.contract["suite_registry"].items()
            if suite["role"] == "primary_eval"
        )
        expected_structural_held_out = sorted(
            suite_id
            for suite_id, suite in self.contract["suite_registry"].items()
            if suite["role"] == "primary_eval"
            and any(
                marker in suite["split"].lower() for marker in ("held-out", "private")
            )
        )
        self.assertEqual(len(expected_training), 12)
        self.assertEqual(len(expected_primary_eval), 14)
        self.assertEqual(manifest["training_suite_ids"], expected_training)
        self.assertEqual(len(expected_structural_held_out), 6)
        self.assertEqual(
            manifest["primary_eval_suite_ids"], expected_primary_eval
        )
        self.assertEqual(
            manifest["held_out_suite_ids"], expected_structural_held_out
        )
        self.assertEqual(len(manifest["pending_held_out_suite_ids"]), 8)
        self.assertEqual(
            set(manifest["domain_training_suite_ids"]),
            set(self.contract["domains"]),
        )
        self.assertEqual(
            set(manifest["domain_primary_eval_suite_ids"]),
            set(self.contract["domains"]),
        )
        self.assertTrue(
            all(manifest["training_suite_domains"].values())
            and all(manifest["held_out_suite_domains"].values())
        )

    def test_three_lr_arms_are_fail_closed_without_receipts(self) -> None:
        manifest = build_manifest(self.contract)
        self.assertEqual(
            [arm["arm_id"] for arm in manifest["arms"]],
            ["lr-1e-5", "lr-2e-5", "lr-4e-5"],
        )
        self.assertEqual(
            [arm["learning_rate"] for arm in manifest["arms"]],
            [1e-5, 2e-5, 4e-5],
        )
        required = set(manifest["receipt_policy"]["required"])
        for arm in manifest["arms"]:
            self.assertFalse(arm["launchable"])
            self.assertEqual(set(arm["missing_receipts"]), required)
            self.assertIn("wandb_online", arm["missing_receipts"])
            self.assertIn("hf_publication", arm["missing_receipts"])
            self.assertIn("verifier", arm["missing_receipts"])

    def test_every_train_and_primary_eval_suite_has_an_independent_provenance_gate(self) -> None:
        manifest = build_manifest(self.contract)
        expected = {
            suite_id
            for suite_id, suite in self.contract["suite_registry"].items()
            if suite["role"] in {"train", "primary_eval"}
        }
        self.assertEqual(set(manifest["suite_receipt_status"]), expected)
        self.assertEqual(set(manifest["pending_suite_receipts"]), expected)
        self.assertEqual(
            set(manifest["pending_training_receipts"])
            | set(manifest["pending_primary_eval_receipts"]),
            expected,
        )

    def test_complete_receipts_are_the_only_way_to_clear_arm_gate(self) -> None:
        candidate = copy.deepcopy(self.contract)
        for model in candidate["model_candidates"]:
            model["revision"] = "a" * 40
        checkpoint_revisions = ["1" * 40, "2" * 40, "3" * 40]
        candidate["receipts"] = {
            "dataset_revision": "b" * 40,
            "license": {"receipt_id": "4" * 40, "approved": True},
            "split_manifest_hash": "c" * 64,
            "container_digest": "sha256:" + "d" * 64,
            "decontamination": {
                "status": "verified",
                "receipt_id": "5" * 40,
            },
            "budget": {
                "authorization_id": "6" * 40,
                "maximum_usd": 18.0,
            },
            "wandb_online": {
                "online": True,
                "run_id": "run-1",
                "run_url": "https://wandb.ai/entity/project/runs/run-1",
            },
            "hf_publication": {
                "checkpoints": [
                    {
                        "stage": stage,
                        "repo_url": "https://huggingface.co/org/pavlov",
                        "revision": revision,
                        "url": f"https://huggingface.co/org/pavlov/commit/{revision}",
                        "visibility": "private" if stage == "periodic" else "public",
                        "safe_public_artifact": True,
                    }
                    for stage, revision in zip(
                        ("initial", "periodic", "final"), checkpoint_revisions
                    )
                ]
            },
            "verifier": "e" * 64,
            "model_revision": {
                "primary": "f" * 40,
                "replication": "0" * 40,
            },
        }
        candidate["status"] = "authorized"
        candidate["budget_gate"]["status"] = "AUTHORIZED_TINKER_ONLY"
        per_suite = {
            suite_id: {
                "dataset_revision": "1" * 40,
                "license": {"receipt_id": "2" * 40, "approved": True},
                "split_manifest_hash": "3" * 64,
                "container_digest": "sha256:" + "4" * 64,
                "decontamination": {
                    "status": "verified",
                    "receipt_id": "5" * 40,
                },
            }
            for suite_id in candidate["suite_registry"]
            if candidate["suite_registry"][suite_id]["role"] in {"train", "primary_eval"}
        }
        candidate["receipts"]["suite_receipts"] = per_suite
        manifest = build_manifest(candidate)
        self.assertEqual(manifest["status"], "READY")
        self.assertTrue(manifest["launchable"])
        self.assertTrue(all(arm["launchable"] for arm in manifest["arms"]))
        self.assertEqual(manifest["missing_receipts"], [])

    def test_status_only_wandb_and_hf_values_do_not_authorize(self) -> None:
        candidate = copy.deepcopy(self.contract)
        candidate["receipts"] = {
            "wandb_online": {"status": "complete", "online": True},
            "hf_publication": {
                "status": "complete",
                "published": True,
                "repo": "org/pavlov",
            },
        }
        manifest = build_manifest(candidate)
        self.assertEqual(manifest["status"], "BLOCKED")
        self.assertIn("wandb_online", manifest["missing_receipts"])
        self.assertIn("hf_publication", manifest["missing_receipts"])

    def test_hf_policy_requires_safe_per_checkpoint_visibility_and_unique_urls(self) -> None:
        manifest = build_manifest(self.contract)
        policy = manifest["hf_publication_policy"]
        self.assertNotEqual(policy["visibility"], "private")
        self.assertEqual(set(policy["allowed_visibility"]), {"public", "private"})
        self.assertTrue(policy["every_checkpoint"])
        self.assertTrue(policy["periodic_and_final"])
        self.assertTrue(policy["unique_repo_revision_url_required"])
        self.assertTrue(policy["safe_public_artifact_rule"]["required"])

    def test_selection_uses_sealed_slice_until_held_out_receipt_is_proven(self) -> None:
        manifest = build_manifest(self.contract)
        halving = manifest["successive_halving"]
        self.assertIn("sealed selection slice", halving["selection_metric"])
        self.assertNotIn("frozen held-out", halving["selection_metric"])
        self.assertTrue(halving["held_out_separation"]["held_out_label_requires_independent_receipt"])
        self.assertFalse(manifest["claim_policy"]["held_out_suite_claim_allowed"])

    def test_successive_halving_separates_selection_and_final_evaluation(self) -> None:
        manifest = build_manifest(self.contract)
        halving = manifest["successive_halving"]
        self.assertEqual(halving["method"], "successive_halving")
        self.assertEqual(len(halving["screening_arm_ids"]), 3)
        self.assertEqual(
            halving["tie_breakers"], ["strict mean reward", "lower estimated cost"]
        )
        separation = halving["held_out_separation"]
        self.assertTrue(separation["must_be_disjoint"])
        self.assertTrue(separation["selection_split"]["consulted_during_selection"])
        self.assertFalse(separation["final_eval_split"]["consulted_during_selection"])
        self.assertIn("extend only the winning arm", halving["winner_extension"])

    def test_xlam_only_scope_cannot_launch_or_claim_company_usefulness(self) -> None:
        candidate = copy.deepcopy(self.contract)
        candidate["xlam_only"] = True
        manifest = build_manifest(candidate)
        self.assertFalse(manifest["claim_policy"]["xlam_only_launch_allowed"])
        self.assertFalse(manifest["claim_policy"]["xlam_only_claim_allowed"])
        self.assertTrue(manifest["claim_policy"]["xlam_only_scope_detected"])
        self.assertTrue(
            any("xLAM-only" in blocker for blocker in manifest["blockers"])
        )

    def test_xlam_observation_is_not_frozen_portfolio_evidence(self) -> None:
        manifest = build_manifest(self.contract)
        self.assertFalse(manifest["claim_policy"]["xlam_observation_claim_allowed"])
        self.assertIn("seed-809 slice only", manifest["claim_policy"]["xlam_observation_status"])
        self.assertTrue(
            manifest["claim_policy"]["xlam_requires_immutable_revisions_and_split_hashes"]
        )

    def test_invalid_contract_cannot_generate_a_manifest(self) -> None:
        candidate = copy.deepcopy(self.contract)
        candidate["sampling_contract"]["maximum_math_fraction"] = 1.0
        with self.assertRaisesRegex(ValueError, "math may occupy at most 5%"):
            build_manifest(candidate)


if __name__ == "__main__":
    unittest.main()
