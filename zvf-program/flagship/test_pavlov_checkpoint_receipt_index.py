from __future__ import annotations

import copy
import unittest

from flagship.pavlov_checkpoint_receipt_index import (
    REQUIRED_CHECKPOINT_KINDS,
    compute_checkpoint_content_digest,
    index_checkpoint_receipts,
    validate_checkpoint_receipts,
)


def _checkpoint(
    kind: str,
    step: int,
    source: str,
    revision: str,
    *,
    visibility: str = "public",
) -> dict[str, object]:
    repo_url = "https://huggingface.co/org/pavlov"
    checkpoint: dict[str, object] = {
        "repo": repo_url,
        "revision": revision,
        "kind": kind,
        "step": step,
        "source": source,
        "visibility": visibility,
        "safe_public_artifact": visibility == "public",
        "url": f"{repo_url}/commit/{revision}",
    }
    checkpoint["content_digest"] = compute_checkpoint_content_digest(checkpoint)
    return checkpoint


def _record(
    source: str,
    run_id: str = "run-checkpoint-index-001",
    *,
    state: str = "finished",
    public_first: bool = True,
) -> dict[str, object]:
    checkpoints = [
        _checkpoint("initial", 0, source, "a" * 40, visibility="public" if public_first else "private"),
        _checkpoint("periodic", 5, source, "b" * 40, visibility="private"),
        _checkpoint("final", 10, source, "c" * 40, visibility="public"),
    ]
    return {
        "source": source,
        "wandb_run_identity": {
            "entity": "pavlov-lab",
            "project": "tracking",
            "group": "checkpoint-receipts",
            "run_id": run_id,
            "run_url": f"https://wandb.ai/pavlov-lab/tracking/runs/{run_id}",
            "online": True,
            "acknowledged": True,
            "state": state,
        },
        "tinker_run_identity": {
            "run_id": run_id,
            "state": state,
        },
        "hf_checkpoints": checkpoints,
    }


class PavlovCheckpointReceiptIndexTests(unittest.TestCase):
    def test_valid_records_build_deterministic_index_and_digest(self) -> None:
        first = index_checkpoint_receipts(
            [_record("train"), _record("eval"), _record("train-extra")]
        )
        second = index_checkpoint_receipts(
            [_record("train-extra"), _record("eval"), _record("train")]
        )
        self.assertEqual(first["index_digest"], second["index_digest"])
        self.assertEqual(first["checkpoint_count"], 9)
        self.assertEqual(first["kind_coverage"], list(REQUIRED_CHECKPOINT_KINDS))
        self.assertEqual(
            first["run_ids"]["wandb"],
            "run-checkpoint-index-001",
        )
        self.assertNotEqual(first["index_digest"], "deadbeef" * 8)

    def test_digest_is_sensitive_to_semantic_checkpoint_content(self) -> None:
        base = _record("train")
        indexed = index_checkpoint_receipts(base)
        mutated = _record("train-alt")
        self.assertNotEqual(indexed["index_digest"], index_checkpoint_receipts(mutated)["index_digest"])

    def test_completeness_check_requires_initial_periodic_and_final(self) -> None:
        record = _record("train")
        assert isinstance(record["hf_checkpoints"], list)
        record["hf_checkpoints"] = [
            copy.deepcopy(record["hf_checkpoints"][0]),
            copy.deepcopy(record["hf_checkpoints"][2]),
        ]
        errors = validate_checkpoint_receipts(record)
        self.assertTrue(any("missing kinds" in error for error in errors))

    def test_duplicate_step_kind_source_repo_commit_identity_is_rejected(self) -> None:
        first = _record("shared-source")
        second = _record("shared-source")
        # Intentionally repeat one identity across records.
        first["hf_checkpoints"][0] = copy.deepcopy(second["hf_checkpoints"][0])
        errors = validate_checkpoint_receipts([first, second])
        self.assertTrue(any("duplicate checkpoint identity" in error for error in errors))

    def test_branch_only_urls_are_rejected(self) -> None:
        record = _record("train")
        assert isinstance(record["hf_checkpoints"], list)
        first = record["hf_checkpoints"][0]
        assert isinstance(first, dict)
        first["url"] = "https://huggingface.co/org/pavlov/tree/main"
        first["content_digest"] = compute_checkpoint_content_digest(first)
        errors = validate_checkpoint_receipts(record)
        self.assertTrue(any("branch-only URL" in error for error in errors))

    def test_non_immutable_commit_ids_are_rejected(self) -> None:
        record = _record("train")
        assert isinstance(record["hf_checkpoints"], list)
        first = record["hf_checkpoints"][0]
        assert isinstance(first, dict)
        first["revision"] = "main"
        first["url"] = "https://huggingface.co/org/pavlov/commit/main"
        first["content_digest"] = compute_checkpoint_content_digest(first)
        errors = validate_checkpoint_receipts(record)
        self.assertTrue(any("immutable 40-hex commit" in error for error in errors))

    def test_run_id_drift_is_rejected(self) -> None:
        first = _record("train", run_id="run-a")
        second = _record("train", run_id="run-b")
        errors = validate_checkpoint_receipts([first, second])
        self.assertTrue(any("run_id drifts" in error for error in errors))

    def test_run_id_mismatch_between_wandb_and_tinker_is_rejected(self) -> None:
        record = _record("train", run_id="run-wandb")
        tinker = record["tinker_run_identity"]
        assert isinstance(tinker, dict)
        tinker["run_id"] = "run-tinker"
        errors = validate_checkpoint_receipts(record)
        self.assertTrue(any("run-id drift between W&B and Tinker receipts" in error for error in errors))

    def test_wandb_run_url_shape_and_terminal_state_is_strict(self) -> None:
        record = _record("train", state="running")
        wandb = record["wandb_run_identity"]
        assert isinstance(wandb, dict)
        wandb["run_url"] = "https://api.wandb.ai/pavlov-lab/tracking/runs/run-checkpoint-index-001"
        errors = validate_checkpoint_receipts(record)
        self.assertTrue(any("invalid wandb host/path shape" in error for error in errors))
        self.assertTrue(any("state must be one of" in error for error in errors))

    def test_public_private_visibility_policy_is_enforced(self) -> None:
        record = _record("train")
        assert isinstance(record["hf_checkpoints"], list)
        public = record["hf_checkpoints"][0]
        private = record["hf_checkpoints"][1]
        assert isinstance(public, dict)
        assert isinstance(private, dict)
        public["safe_public_artifact"] = False
        private["safe_public_artifact"] = True
        public["content_digest"] = compute_checkpoint_content_digest(public)
        private["content_digest"] = compute_checkpoint_content_digest(private)
        errors = validate_checkpoint_receipts(record)
        self.assertTrue(any("must be true when visibility is public" in error for error in errors))
        self.assertTrue(any("must be false when visibility is private" in error for error in errors))

    def test_content_digest_is_required_and_bound(self) -> None:
        record = _record("train")
        assert isinstance(record["hf_checkpoints"], list)
        first = record["hf_checkpoints"][0]
        assert isinstance(first, dict)
        first.pop("content_digest")
        missing = validate_checkpoint_receipts(record)
        self.assertTrue(any("content_digest is required" in error for error in missing))

        record = _record("train")
        changed = copy.deepcopy(record["hf_checkpoints"][0])
        assert isinstance(changed, dict)
        changed["revision"] = "b" * 40
        changed["url"] = f"https://huggingface.co/org/pavlov/commit/{'b' * 40}"
        record["hf_checkpoints"][0] = changed
        mismatch = validate_checkpoint_receipts(record)
        self.assertTrue(any("must match canonical content digest binding" in error for error in mismatch))

    def test_fabricated_booleans_and_status_strings_are_rejected(self) -> None:
        record = _record("train")
        wandb = record["wandb_run_identity"]
        assert isinstance(wandb, dict)
        wandb["online"] = "true"
        wandb["acknowledged"] = 1
        tinker = record["tinker_run_identity"]
        assert isinstance(tinker, dict)
        tinker["state"] = "in_progress"
        checkpoint = record["hf_checkpoints"][0]
        assert isinstance(checkpoint, dict)
        checkpoint["safe_public_artifact"] = "yes"
        checkpoint["content_digest"] = compute_checkpoint_content_digest(checkpoint)
        errors = validate_checkpoint_receipts(record)
        self.assertTrue(any("online must be boolean true" in error for error in errors))
        self.assertTrue(any("acknowledged must be boolean true" in error for error in errors))
        self.assertTrue(any("safe_public_artifact must be a boolean" in error for error in errors))
        self.assertTrue(any("tinker_run_identity.state/status must be one of" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
