from __future__ import annotations

import hashlib
import json
import re
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
            assert sha256(ROOT / record["path"]) == record["sha256"]
        theory = self.protocol["theory_gate"]
        assert sha256(ROOT / theory["audit_path"]) == theory["audit_sha256"]
        freeze = json.loads((ROOT / self.protocol["s1_freeze"]["path"]).read_text())
        assert freeze["status"] == self.protocol["s1_freeze"]["required_status"]

    def test_versioned_zero_relation_amendment_and_corpus_reuse_are_explicit(self) -> None:
        assert self.protocol["protocol_version"] == 2
        assert self.protocol["implementation_revision"] == 7
        amendment = self.protocol["amendment"]
        assert amendment["id"] == "A1-corpus-intermediate-persistence"
        assert amendment["authorized_by_user"]
        assert re.search(r"^[0-9a-f]{64}$", amendment["previous_protocol_sha256"])
        assert len(amendment["prior_attempts"]) == 3
        assert "never pooled" in amendment["evidence_treatment"]
        corrections = amendment["control_plane_corrections"]
        r4 = corrections[-3]
        assert r4["id"] == "A1-R4-explicit-degenerate-gradient-relations-with-corpus-reuse"
        assert r4["authorized_by_user"]
        assert "joint-zero" in r4["observation"]
        assert "No replay group is dropped" in r4["correction"]
        assert corrections[-2]["id"] == "A1-R4.1-package-frozen-archives-for-remote-validation"
        assert corrections[-2]["authorized_under"] == r4["id"]
        assert "No W&B run" in corrections[-2]["observation"]
        assert corrections[-1]["id"] == "A1-R4.2-exact-identical-gradient-diagnostics"
        assert corrections[-1]["authorized_under"] == r4["id"]
        assert "1.000000000002599" in corrections[-1]["observation"]
        binding = self.protocol["corpus_reuse_binding"]
        assert sha256(ROOT / binding["path"]) == binding["sha256"]
        assert (
            sha256(ROOT / binding["frozen_source_archive_path"])
            == binding["frozen_source_archive_sha256"]
        )

    def test_screening_matrix_is_exactly_twenty_four_units(self) -> None:
        count = self.protocol["unit_count"]
        assert count["conditions"] * count["regimes"] * count["seeds"] == 24
        assert count["total"] == 24
        assert len(self.protocol["conditions"]) == 4
        assert len(self.protocol["regimes"]) == 2
        assert len(self.protocol["runtime"]["screening_seeds"]) == 3

    def test_screening_and_confirmatory_seeds_are_disjoint(self) -> None:
        screening = set(self.protocol["runtime"]["screening_seeds"])
        confirmatory = set(self.protocol["runtime"]["confirmatory_seeds"])
        assert len(confirmatory) == 5
        assert screening.isdisjoint(confirmatory)

    def test_gate_requires_direction_not_only_scalar_loss_change(self) -> None:
        question = self.protocol["central_question"]
        mechanism = self.protocol["screening_gate"]["mechanism"]
        primary = self.protocol["measurements"]["primary_mechanism"]
        assert "gradient direction" in question
        assert "cosine" in primary
        assert "relative L2" in primary
        assert "frozen nonzero gradient-discrepancy thresholds" in mechanism
        assert "reduction_only" in self.protocol["screening_gate"]["causal_attribution"]
        assert "epsilon_only" in self.protocol["screening_gate"]["causal_attribution"]

    def test_compute_matching_and_provenance_fail_closed(self) -> None:
        matching = self.protocol["matched_compute"]
        assert "masked" in matching["charged_tokens"]
        assert "FLOPs" in matching["flops"]
        assert len(matching["frozen_fields"]) >= 8
        assert "not a scientific observation" in matching["fail_closed"]
        assert "Local, W&B, and Hugging Face" in self.protocol["provenance"]["acceptance"]

    def test_power_and_equivalence_are_explicit(self) -> None:
        power = self.protocol["power"]
        assert "100000 Monte Carlo" in power["confirmatory"]
        assert "0.80 power" in power["confirmatory"]
        assert "TOST" in power["confirmatory"]
        assert "INCONCLUSIVE" in power["equivalence"]

    def test_gpu_authorization_is_smoke_scoped_and_gated(self) -> None:
        assert self.protocol["status"] == "ready_to_run"
        assert self.protocol["authorization"]["gpu"]
        assert (
            "A fresh implementation-revision-7 non-scientific A100 smoke"
            in self.protocol["authorization"]["scope"]
        )
        assert "no confirmatory jobs" in self.protocol["authorization"]["scope"]

    def test_execution_contract_resolves_fixed_step_and_token_matching(self) -> None:
        contract = self.protocol["runtime"]["execution_contract"]
        assert contract["design"] == "shared_frozen_offpolicy_replay"
        assert contract["accepted_groups_per_corpus"] == 100
        assert contract["charged_generated_token_ceiling"] == 819200
        assert "all 100 groups" in contract["matched_budget_horizon_rule"]
        assert "charged identically" in contract["matched_budget_horizon_rule"]
        assert contract["heldout_n"] == 128
        assert contract["replay_gradient_steps"]["count"] == 100
        assert contract["generation_batch_size"]["filtered_variable_length"] == 16
        assert "non-scientific A100" in contract["preflight"]
        assert "blocks all six corpus jobs" in contract["preflight"]
        resume = contract["corpus_checkpoint_resume_contract"]
        assert resume["checkpoint_groups"] == [20, 40, 60, 80]
        assert resume["storage_prefix"] == "resume/"
        assert resume["attempt_limit"] == 3
        assert resume["max_parallel_corpus_sessions"] == 1
        assert "exact protocol" in resume["resume_validation"]
        assert "does not change any generated group" in resume["determinism"]

    def test_dataset_revisions_and_train_orders_are_frozen(self) -> None:
        for regime in self.protocol["regimes"].values():
            assert re.search(r"^[0-9a-f]{40}$", regime["dataset_revision"])
            for digest in regime["source_ordered_row_sha256"].values():
                assert re.search(r"^[0-9a-f]{64}$", digest)
        hashes = self.protocol["runtime"]["execution_contract"]["train_order_hash"]
        for regime in ("balanced_equal_length", "filtered_variable_length"):
            assert set(hashes[regime]) == {"11", "23", "37"}
            for digest in hashes[regime].values():
                assert re.search(r"^[0-9a-f]{64}$", digest)

    def test_remote_stack_is_fully_pinned_to_the_resolved_set(self) -> None:
        assert self.protocol["runtime"]["python"] == ">=3.11,<3.13"
        pins = set(self.protocol["runtime"]["package_pins"])
        assert pins == {
            "trl==1.2.0",
            "transformers==5.5.4",
            "torch==2.7.1",
            "datasets==4.8.4",
            "peft==0.19.1",
            "huggingface-hub==1.11.0",
            "accelerate==1.13.0",
            "wandb==0.21.0",
            "numpy==2.2.6",
        }

    def test_tracking_namespaces_are_frozen_and_private(self) -> None:
        tracking = self.protocol["runtime"]["execution_contract"]["tracking"]
        assert tracking["wandb_project"] == "tinker-rl-lab"
        assert tracking["wandb_unit_group"] == "flagship-s1-conformance-screening"
        assert tracking["hugging_face_private"]


if __name__ == "__main__":
    unittest.main()
