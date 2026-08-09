from __future__ import annotations

import unittest

from flagship.pavlov_tracking_attestation import (
    compute_checkpoint_content_digest,
    validate_tracking_records,
    validate_tracking_attestation,
)


def _checkpoint(kind: str, step: int, visibility: str, revision: str) -> dict[str, object]:
    repo_url = "https://huggingface.co/org/pavlov"
    checkpoint_url = f"{repo_url}/commit/{revision}"
    checkpoint = {
        "repo_url": repo_url,
        "commit": revision,
        "kind": kind,
        "step": step,
        "visibility": visibility,
        "safe_public_artifact": visibility == "public",
        "url": checkpoint_url,
    }
    checkpoint["content_digest"] = compute_checkpoint_content_digest(checkpoint)
    return checkpoint


def _valid_record() -> dict[str, object]:
    return {
        "wandb_run_identity": {
            "entity": "acme-lab",
            "project": "tinker-rl-lab",
            "group": "contrast-early-stop",
            "run_id": "run-8f9d2a",
            "run_url": "https://wandb.ai/acme-lab/tinker-rl-lab/runs/run-8f9d2a",
            "online": True,
            "acknowledged": True,
            "state": "finished",
        },
        "tinker_run_identity": {"run_id": "tinker-run-2026-a"},
        "hf_checkpoints": [
            _checkpoint("initial", 0, "public", "a" * 40),
            _checkpoint("periodic", 5, "private", "b" * 40),
            _checkpoint("final", 10, "public", "c" * 40),
        ],
    }


class PavlovTrackingAttestationTests(unittest.TestCase):
    def test_valid_record_is_accepted(self) -> None:
        self.assertEqual(validate_tracking_attestation(_valid_record()), [])

    def test_wandb_online_must_be_real_boolean_true(self) -> None:
        record = _valid_record()
        record["wandb_run_identity"]["online"] = "true"
        errors = validate_tracking_attestation(record)
        self.assertTrue(any("online must be the boolean true" in error for error in errors))

    def test_wandb_run_must_have_terminal_state(self) -> None:
        record = _valid_record()
        record["wandb_run_identity"]["state"] = "running"
        errors = validate_tracking_attestation(record)
        self.assertTrue(any("terminal" in error for error in errors))

    def test_wandb_run_url_shape_is_strict(self) -> None:
        record = _valid_record()
        record["wandb_run_identity"]["run_url"] = "https://api.wandb.ai/acme-lab/tinker-rl-lab/runs/run-8f9d2a"
        errors = validate_tracking_attestation(record)
        self.assertTrue(any("run_url has invalid wandb host/path shape" in error for error in errors))

    def test_tinker_run_id_placeholder_fails(self) -> None:
        record = _valid_record()
        record["tinker_run_identity"]["run_id"] = "unknown"
        errors = validate_tracking_attestation(record)
        self.assertTrue(any("tinker_run_identity.run_id" in error for error in errors))

    def test_tinker_run_id_must_be_a_string(self) -> None:
        record = _valid_record()
        record["tinker_run_identity"]["run_id"] = 12345
        errors = validate_tracking_attestation(record)
        self.assertTrue(any("tinker_run_identity.run_id must be a non-placeholder string" in error for error in errors))

    def test_hf_commit_and_url_shape_are_strict(self) -> None:
        record = _valid_record()
        first = record["hf_checkpoints"][0]
        assert isinstance(first, dict)
        first["commit"] = "not-a-commit"
        first["url"] = "https://huggingface.co/org/pavlov/tree/main"
        first["content_digest"] = compute_checkpoint_content_digest(first)
        errors = validate_tracking_attestation(record)
        self.assertTrue(any("revision must be an immutable 40-hex commit" in error for error in errors))

    def test_checkpoints_require_initial_periodic_final_coverage(self) -> None:
        record = _valid_record()
        record["hf_checkpoints"] = [
            _checkpoint("initial", 0, "public", "a" * 40),
            _checkpoint("initial", 5, "private", "b" * 40),
            _checkpoint("initial", 10, "public", "c" * 40),
        ]
        errors = validate_tracking_attestation(record)
        self.assertTrue(any("missing: ['final', 'periodic']" in error for error in errors))

    def test_checkpoint_kind_step_uniqueness(self) -> None:
        record = _valid_record()
        duplicate = _checkpoint("periodic", 5, "private", "b" * 40)
        record["hf_checkpoints"].append(duplicate)
        # Keep the duplicate distinct by digest and index.
        record["hf_checkpoints"] = [
            _checkpoint("initial", 0, "public", "a" * 40),
            _checkpoint("periodic", 5, "private", "b" * 40),
            _checkpoint("periodic", 5, "public", "c" * 40),
            _checkpoint("final", 10, "private", "d" * 40),
        ]
        errors = validate_tracking_attestation(record)
        self.assertTrue(any("duplicate checkpoint (kind,step) pair" in error for error in errors))

    def test_public_visibility_requires_safe_public_artifact_true(self) -> None:
        record = _valid_record()
        checkpoint = record["hf_checkpoints"][0]
        assert isinstance(checkpoint, dict)
        checkpoint["safe_public_artifact"] = False
        checkpoint["content_digest"] = compute_checkpoint_content_digest(checkpoint)
        errors = validate_tracking_attestation(record)
        self.assertTrue(any("must be true when visibility is public" in error for error in errors))

    def test_private_visibility_requires_safe_public_artifact_false(self) -> None:
        record = _valid_record()
        checkpoint = record["hf_checkpoints"][1]
        assert isinstance(checkpoint, dict)
        checkpoint["safe_public_artifact"] = True
        checkpoint["content_digest"] = compute_checkpoint_content_digest(checkpoint)
        errors = validate_tracking_attestation(record)
        self.assertTrue(any("must be false when visibility is private" in error for error in errors))

    def test_content_digest_is_a_hard_canonical_binding(self) -> None:
        record = _valid_record()
        record["hf_checkpoints"][0]["repo_url"] = "https://huggingface.co/org/pavlov-v2"
        errors = validate_tracking_attestation(record)
        self.assertTrue(
            any(
                "content_digest must match canonical content digest binding" in error
                or "repo name does not match repo_url" in error
                for error in errors
            )
        )

    def test_fabricated_boolean_and_status_strings_fail(self) -> None:
        record = _valid_record()
        record["wandb_run_identity"]["acknowledged"] = False
        record["wandb_run_identity"]["state"] = "doneish"
        record["hf_checkpoints"][0]["safe_public_artifact"] = "yes"
        record["hf_checkpoints"][0]["content_digest"] = compute_checkpoint_content_digest(
            record["hf_checkpoints"][0]
        )
        errors = validate_tracking_attestation(record)
        self.assertGreaterEqual(len(errors), 2)
        self.assertTrue(any("acknowledged must be the boolean true" in error for error in errors))
        self.assertTrue(any("state must be terminal" in error for error in errors))
        self.assertTrue(any("safe_public_artifact must be a boolean" in error for error in errors))

    def test_validate_tracking_records_adds_record_prefixes(self) -> None:
        records = [_valid_record(), _valid_record()]
        records[1]["wandb_run_identity"]["state"] = "running"
        errors = validate_tracking_records(records)
        self.assertTrue(any("record[1]" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
