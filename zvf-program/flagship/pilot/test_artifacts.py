from __future__ import annotations

import copy
import unittest

from pilot.artifacts import (
    CHECKPOINT_STEPS,
    EVALUATION_STEPS,
    REQUIRED_CHECKPOINT_FILES,
    ArtifactValidationError,
    greatest_compatible_checkpoint,
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
        contract = self.protocol.payload["runtime"]["execution_contract"]
        regime = unit.regime
        groups = [
            {
                "index": index,
                "fingerprint": f"{index:064x}",
                "active_rows": 8,
                "selected_length_cv": 0.0,
                "charged_generated_tokens": 64,
            }
            for index in range(100)
        ]
        self.corpus = with_fingerprint(
            {
                "schema_version": "flagship-pilot-corpus-v1",
                "status": "complete",
                "protocol_sha256": self.protocol.sha256,
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
                "wandb": {
                    "run_id": "corpus-run",
                    "run_url": "https://wandb.ai/entity/tinker-rl-lab/runs/corpus-run",
                    "entity": "entity",
                    "project": "tinker-rl-lab",
                },
            }
        )
        self.corpus = validate_corpus_manifest(
            self.corpus,
            protocol=self.protocol,
            regime=regime,
            seed=unit.seed,
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
        checkpoint = validate_checkpoint_manifest(
            self.checkpoint(20), plan=self.plan, corpus=self.corpus
        )
        self.assertEqual(checkpoint["step"], 20)
        record = validate_full_record(self.full_record(), plan=self.plan, corpus=self.corpus)
        self.assertEqual(record["status"], "completed")

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


if __name__ == "__main__":
    unittest.main()
