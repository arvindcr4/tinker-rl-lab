from __future__ import annotations

import copy
import unittest

from pilot.artifacts import (
    CHECKPOINT_STEPS,
    CORPUS_CHECKPOINT_GROUPS,
    EVALUATION_STEPS,
    REQUIRED_CHECKPOINT_FILES,
    ArtifactValidationError,
    greatest_compatible_checkpoint,
    validate_corpus_checkpoint_manifest,
    validate_checkpoint_manifest,
    validate_corpus_manifest,
    validate_full_record,
    with_fingerprint,
)
from pilot.protocol import build_screening_plan, load_protocol


HASH = "a" * 64


class ArtifactValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_protocol()
        unit = next(self.protocol.screening_units())
        self.plan = build_screening_plan(self.protocol, unit)
        self.corpus_binding = self.protocol.corpus_binding(unit.regime, unit.seed)
        contract = self.protocol.payload["runtime"]["execution_contract"]
        regime = unit.regime
        groups = [
            {
                "index": index,
                "source_row_index": index,
                "fingerprint": f"{index:064x}",
                "active_rows": 8,
                "selected_length_cv": 0.0,
                "charged_generated_tokens": 64,
                "artifact_path": f"groups/group-{index:03d}.pt",
            }
            for index in range(100)
        ]
        self.corpus = with_fingerprint(
            {
                "schema_version": "flagship-pilot-corpus-v2",
                "status": "complete",
                "protocol_sha256": self.corpus_binding["protocol_sha256"],
                "regime": regime,
                "seed": unit.seed,
                "model": self.protocol.payload["runtime"]["model"],
                "dataset": self.protocol.payload["regimes"][regime]["dataset"],
                "dataset_revision": self.protocol.payload["regimes"][regime]["dataset_revision"],
                "train_order_hash": contract["train_order_hash"][regime][str(unit.seed)],
                "groups": groups,
                "rejected_generated_tokens": 0,
                "charged_generated_tokens": 6400,
                "artifact_files": {"replay.jsonl": HASH, "ledger.json": HASH},
                "source_manifest": dict(self.corpus_binding["source_manifest"]),
                "wandb": {
                    "run_id": "corpus-run",
                    "run_url": "https://wandb.ai/entity/tinker-rl-lab/runs/corpus-run",
                    "entity": "entity",
                    "project": "tinker-rl-lab",
                },
                "corpus_resume": {
                    "schema_version": "flagship-pilot-corpus-resume-v1",
                    "enabled": True,
                    "checkpoint_groups": list(CORPUS_CHECKPOINT_GROUPS),
                    "resume_count": 0,
                    "attempts": [
                        {
                            "run_id": "corpus-run",
                            "run_url": "https://wandb.ai/entity/tinker-rl-lab/runs/corpus-run",
                            "start_group": 0,
                            "completed_through": 100,
                        }
                    ],
                    "latest_checkpoint": {
                        "completed_groups": 80,
                        "fingerprint": HASH,
                        "hf_commit": "b" * 40,
                    },
                },
            }
        )
        self.corpus = validate_corpus_manifest(
            self.corpus,
            protocol=self.protocol,
            regime=regime,
            seed=unit.seed,
        )

    def corpus_checkpoint(self, completed: int = 20) -> dict[str, object]:
        unit = self.plan["unit"]
        contract = self.protocol.payload["runtime"]["execution_contract"]
        regime = unit["regime"]
        groups = copy.deepcopy(self.corpus["groups"][:completed])
        files = {"source_manifest.json": HASH}
        files.update({f"groups/group-{index:03d}.pt": HASH for index in range(completed)})
        return with_fingerprint(
            {
                "schema_version": "flagship-pilot-corpus-checkpoint-v1",
                "status": "partial",
                "protocol_sha256": self.corpus_binding["protocol_sha256"],
                "regime": regime,
                "seed": unit["seed"],
                "model": self.protocol.payload["runtime"]["model"],
                "dataset": self.protocol.payload["regimes"][regime]["dataset"],
                "dataset_revision": self.protocol.payload["regimes"][regime]["dataset_revision"],
                "train_order_hash": contract["train_order_hash"][regime][str(unit["seed"])],
                "completed_groups": completed,
                "groups": groups,
                "charged_generated_tokens": 64 * completed,
                "flop_ledger": {
                    "profiled_steps": [
                        step
                        for step in contract["flop_counter"]["profiled_steps"]
                        if step <= completed
                    ],
                    "profiled_generated_tokens": 64,
                    "profiled_generation_flops": 1.0,
                },
                "runtime_versions": {"torch": "2.7.1"},
                "accelerator": "NVIDIA A100-SXM4-40GB",
                "source_manifest": dict(self.corpus_binding["source_manifest"]),
                "artifact_files": files,
                "resume_count": 0,
                "attempts": [
                    {
                        "run_id": "corpus-run",
                        "run_url": "https://wandb.ai/entity/project/runs/corpus-run",
                        "start_group": 0,
                        "completed_through": completed,
                    }
                ],
                "wall_clock_seconds": 1.0,
            }
        )

    def checkpoint(self, step: int) -> dict[str, object]:
        unit = self.plan["unit"]
        return with_fingerprint(
            {
                "schema_version": "flagship-pilot-checkpoint-v1",
                "step": step,
                "unit_fingerprint": self.plan["fingerprint"],
                "protocol_sha256": self.plan["protocol"]["sha256"],
                "corpus_fingerprint": self.corpus["fingerprint"],
                "condition": unit["condition"],
                "regime": unit["regime"],
                "seed": unit["seed"],
                "replay_cursor": step,
                "gradient_receipt_count": step,
                "evaluation_steps": [0, step],
                "token_flop_ledger": {
                    "charged_generated_tokens": self.corpus["charged_generated_tokens"],
                    "policy_forward_flops": 1.0,
                    "diagnostic_backward_flops": 1.0,
                    "optimizer_backward_flops": 1.0,
                },
                "files": {
                    **{path: HASH for path in REQUIRED_CHECKPOINT_FILES},
                    "evaluations/step-000.jsonl": HASH,
                    f"evaluations/step-{step:03d}.jsonl": HASH,
                },
            }
        )

    def full_record(self) -> dict[str, object]:
        unit = self.plan["unit"]
        return with_fingerprint(
            {
                "schema_version": "flagship-pilot-unit-v1",
                "status": "completed",
                "unit_fingerprint": self.plan["fingerprint"],
                "corpus_fingerprint": self.corpus["fingerprint"],
                "condition": unit["condition"],
                "regime": unit["regime"],
                "seed": unit["seed"],
                "training_steps": 100,
                "gradient_receipt_count": 100,
                "checkpoint_steps": list(CHECKPOINT_STEPS),
                "evaluations": [
                    {"step": step, "heldout_n": 128, "evidence_sha256": HASH}
                    for step in EVALUATION_STEPS
                ],
                "token_flop_ledger": {
                    "charged_generated_tokens": self.corpus["charged_generated_tokens"],
                    "replay_generation_flops": 1.0,
                    "policy_forward_flops": 1.0,
                    "diagnostic_backward_flops": 1.0,
                    "optimizer_backward_flops": 1.0,
                },
                "manifest": {
                    "schema_version": "flagship-pilot-run-manifest-v1",
                    "corpus_fingerprint": self.corpus["fingerprint"],
                    "gradient_receipts": [
                        {
                            "step": step,
                            "condition": unit["condition"],
                            "group_fingerprint": self.corpus["groups"][step - 1]["fingerprint"],
                            "selected_loss": 0.1,
                            "intended_loss": 0.1,
                            "native_loss": 0.2,
                            "gradient_relation": "nonzero",
                            "gradient_cosine": 0.9,
                            "gradient_relative_l2": 0.1,
                            "intended_gradient_norm": 1.0,
                            "native_gradient_norm": 1.0,
                            "selected_gradient_norm": 1.0,
                            "selected_vs_intended_relation": "nonzero",
                            "selected_vs_intended_cosine": 1.0,
                            "selected_vs_intended_relative_l2": 0.0,
                            "optimizer_update": "applied",
                            "active_rows": 8,
                            "active_tokens": 64,
                            "optimizer_learning_rate": 1e-5,
                        }
                        for step in range(1, 101)
                    ],
                },
                "wandb": {
                    "state": "finished",
                    "run_id": "run-123",
                    "run_url": "https://wandb.ai/entity/tinker-rl-lab/runs/run-123",
                    "entity": "entity",
                    "project": "tinker-rl-lab",
                },
                "hugging_face": {
                    "private": True,
                    "repo": self.plan["identity"]["hf_repo"],
                    "commit": "b" * 40,
                    "checkpoint_steps": list(CHECKPOINT_STEPS),
                    "final_adapter_sha256": HASH,
                    "manifest_sha256": HASH,
                },
            }
        )

    def test_valid_corpus_checkpoint_and_full_record_pass(self) -> None:
        corpus_checkpoint = validate_corpus_checkpoint_manifest(
            self.corpus_checkpoint(),
            protocol=self.protocol,
            regime=self.plan["unit"]["regime"],
            seed=self.plan["unit"]["seed"],
        )
        self.assertEqual(corpus_checkpoint["completed_groups"], 20)
        checkpoint = validate_checkpoint_manifest(
            self.checkpoint(20), plan=self.plan, corpus=self.corpus
        )
        self.assertEqual(checkpoint["step"], 20)
        record = validate_full_record(self.full_record(), plan=self.plan, corpus=self.corpus)
        self.assertEqual(record["status"], "completed")

    def test_corpus_checkpoint_rejects_tampering_and_non_cadence_prefix(self) -> None:
        wrong_order = self.corpus_checkpoint()
        wrong_order["train_order_hash"] = HASH
        wrong_order = with_fingerprint(wrong_order)
        with self.assertRaisesRegex(ArtifactValidationError, "train-order hash mismatch"):
            validate_corpus_checkpoint_manifest(
                wrong_order,
                protocol=self.protocol,
                regime=self.plan["unit"]["regime"],
                seed=self.plan["unit"]["seed"],
            )
        non_cadence = self.corpus_checkpoint()
        non_cadence["completed_groups"] = 19
        non_cadence["groups"] = non_cadence["groups"][:19]
        non_cadence = with_fingerprint(non_cadence)
        with self.assertRaisesRegex(ArtifactValidationError, "amended cadence"):
            validate_corpus_checkpoint_manifest(
                non_cadence,
                protocol=self.protocol,
                regime=self.plan["unit"]["regime"],
                seed=self.plan["unit"]["seed"],
            )

    def test_corpus_rejects_stale_protocol_and_bad_ledger(self) -> None:
        stale = copy.deepcopy(self.corpus)
        stale["protocol_sha256"] = HASH
        stale = with_fingerprint(stale)
        with self.assertRaisesRegex(ArtifactValidationError, "protocol hash mismatch"):
            validate_corpus_manifest(
                stale,
                protocol=self.protocol,
                regime=self.plan["unit"]["regime"],
                seed=self.plan["unit"]["seed"],
            )
        bad_ledger = copy.deepcopy(self.corpus)
        bad_ledger["charged_generated_tokens"] += 1
        bad_ledger = with_fingerprint(bad_ledger)
        with self.assertRaisesRegex(ArtifactValidationError, "does not sum"):
            validate_corpus_manifest(
                bad_ledger,
                protocol=self.protocol,
                regime=self.plan["unit"]["regime"],
                seed=self.plan["unit"]["seed"],
            )

    def test_checkpoint_rejects_wrong_corpus_and_incomplete_files(self) -> None:
        wrong = self.checkpoint(20)
        wrong["corpus_fingerprint"] = HASH
        wrong = with_fingerprint(wrong)
        with self.assertRaisesRegex(ArtifactValidationError, "corpus hash mismatch"):
            validate_checkpoint_manifest(wrong, plan=self.plan, corpus=self.corpus)
        incomplete = self.checkpoint(20)
        del incomplete["files"]["rng_state.pt"]
        incomplete = with_fingerprint(incomplete)
        with self.assertRaisesRegex(ArtifactValidationError, "file set"):
            validate_checkpoint_manifest(incomplete, plan=self.plan, corpus=self.corpus)

    def test_resume_selects_greatest_valid_checkpoint_and_reports_invalid(self) -> None:
        invalid = self.checkpoint(100)
        invalid["replay_cursor"] = 99
        invalid = with_fingerprint(invalid)
        selected, errors = greatest_compatible_checkpoint(
            [self.checkpoint(20), self.checkpoint(60), invalid],
            plan=self.plan,
            corpus=self.corpus,
        )
        self.assertEqual(selected["step"], 60)
        self.assertEqual(len(errors), 1)
        self.assertIn("replay cursor", errors[0])

    def test_full_record_rejects_unfinished_wandb_or_missing_evaluation(self) -> None:
        unfinished = self.full_record()
        unfinished["wandb"]["state"] = "running"
        unfinished = with_fingerprint(unfinished)
        with self.assertRaisesRegex(ArtifactValidationError, "not finished"):
            validate_full_record(unfinished, plan=self.plan, corpus=self.corpus)
        missing = self.full_record()
        missing["evaluations"].pop()
        missing = with_fingerprint(missing)
        with self.assertRaisesRegex(ArtifactValidationError, "evaluation set"):
            validate_full_record(missing, plan=self.plan, corpus=self.corpus)

    def test_full_record_rejects_invalid_gradient_diagnostics(self) -> None:
        mutations = (
            ("gradient_cosine", 1.0001, "outside"),
            ("selected_vs_intended_cosine", -1.0001, "outside"),
            ("gradient_relative_l2", -0.01, "negative"),
            ("selected_vs_intended_relative_l2", -0.01, "negative"),
            ("selected_gradient_norm", 0.0, "inconsistent"),
        )
        for field, value, error in mutations:
            with self.subTest(field=field, value=value):
                invalid = self.full_record()
                invalid["manifest"]["gradient_receipts"][0][field] = value
                invalid = with_fingerprint(invalid)
                with self.assertRaisesRegex(ArtifactValidationError, error):
                    validate_full_record(invalid, plan=self.plan, corpus=self.corpus)

    def test_full_record_accepts_explicit_joint_zero_receipt(self) -> None:
        record = self.full_record()
        receipt = record["manifest"]["gradient_receipts"][0]
        receipt.update(
            {
                "gradient_relation": "joint_zero",
                "gradient_cosine": None,
                "gradient_relative_l2": None,
                "intended_gradient_norm": 0.0,
                "native_gradient_norm": 0.0,
                "selected_gradient_norm": 0.0,
                "selected_vs_intended_relation": "joint_zero",
                "selected_vs_intended_cosine": None,
                "selected_vs_intended_relative_l2": None,
                "optimizer_update": "no_op_zero_gradient",
            }
        )
        validated = validate_full_record(
            with_fingerprint(record), plan=self.plan, corpus=self.corpus
        )
        self.assertEqual(
            validated["manifest"]["gradient_receipts"][0]["gradient_relation"],
            "joint_zero",
        )

    def test_full_record_rejects_fabricated_zero_vector_cosine(self) -> None:
        record = self.full_record()
        receipt = record["manifest"]["gradient_receipts"][0]
        receipt.update(
            {
                "gradient_relation": "joint_zero",
                "gradient_cosine": 1.0,
                "gradient_relative_l2": 0.0,
                "intended_gradient_norm": 0.0,
                "native_gradient_norm": 0.0,
            }
        )
        with self.assertRaisesRegex(ArtifactValidationError, "null diagnostics"):
            validate_full_record(with_fingerprint(record), plan=self.plan, corpus=self.corpus)

    def test_full_record_accepts_explicit_one_sided_zero_receipt(self) -> None:
        record = self.full_record()
        receipt = record["manifest"]["gradient_receipts"][0]
        receipt.update(
            {
                "gradient_relation": "intended_zero",
                "gradient_cosine": None,
                "gradient_relative_l2": None,
                "intended_gradient_norm": 0.0,
                "native_gradient_norm": 1.0,
                "selected_gradient_norm": 0.0,
                "selected_vs_intended_relation": "joint_zero",
                "selected_vs_intended_cosine": None,
                "selected_vs_intended_relative_l2": None,
                "optimizer_update": "no_op_zero_gradient",
            }
        )
        validated = validate_full_record(
            with_fingerprint(record), plan=self.plan, corpus=self.corpus
        )
        self.assertEqual(
            validated["manifest"]["gradient_receipts"][0]["gradient_relation"],
            "intended_zero",
        )


if __name__ == "__main__":
    unittest.main()
