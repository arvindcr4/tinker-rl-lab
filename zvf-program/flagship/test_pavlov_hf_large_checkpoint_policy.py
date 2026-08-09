from __future__ import annotations

import copy
import unittest

from flagship.pavlov_hf_large_checkpoint_policy import (
    REQUIRED_CHECKPOINT_KINDS,
    build_large_checkpoint_policy_index,
    compute_large_checkpoint_content_digest,
    validate_large_checkpoint_policy,
)


def _checkpoint(kind: str, step: int, revision: str, repo: str, *, visibility: str = "public") -> dict[str, object]:
    checkpoint = {
        "hf_repo": repo,
        "hf_commit": revision,
        "kind": kind,
        "step": step,
        "visibility": visibility,
        "safe_public_artifact": visibility == "public",
        "url": f"{repo}/commit/{revision}",
        "precreated_revision": "c" * 40,
    }
    checkpoint["content_digest"] = compute_large_checkpoint_content_digest(checkpoint)
    return checkpoint


def _record(
    run_id: str,
    *,
    status: str = "completed",
    repo_suffix: str = "primary",
    run_prefix: str = "run-large",
    parent_run_id: str | None = None,
) -> dict[str, object]:
    repo_url = f"https://huggingface.co/org/{run_prefix}-{repo_suffix}"
    return {
        "run_id": run_id,
        "status": status,
        "precreated_revision": "c" * 40,
        "retry_parent_run_id": parent_run_id,
        "wandb_run_identity": {
            "entity": "acme-lab",
            "project": "pavlov",
            "run_id": run_id,
            "run_url": f"https://wandb.ai/acme-lab/pavlov/runs/{run_id}",
            "online": True,
            "acknowledged": True,
            "state": "finished",
        },
        "tinker_run_identity": {
            "run_id": run_id,
            "state": "completed",
        },
        "hf_large_checkpoints": [
            _checkpoint("initial", 0, "a" * 40, repo_url, visibility="public"),
            _checkpoint("periodic", 5, "d" * 40, repo_url, visibility="private"),
            _checkpoint("final", 10, "c" * 40, repo_url, visibility="public"),
        ],
    }


class PavlovLargeCheckpointPolicyTests(unittest.TestCase):
    def test_valid_records_build_deterministic_index(self) -> None:
        first = build_large_checkpoint_policy_index(
            [_record("run-alpha", repo_suffix="alpha"), _record("run-beta", repo_suffix="beta")]
        )
        second = build_large_checkpoint_policy_index(
            [_record("run-beta", repo_suffix="beta"), _record("run-alpha", repo_suffix="alpha")]
        )
        self.assertEqual(first["policy_digest"], second["policy_digest"])
        self.assertEqual(first["required_kinds"], list(REQUIRED_CHECKPOINT_KINDS))
        self.assertEqual(first["run_count"], 2)
        self.assertEqual(first["checkpoint_count"], 6)

    def test_precreated_revision_is_required_and_immutable(self) -> None:
        record = _record("run-alpha", repo_suffix="alpha")
        record["precreated_revision"] = "latest"
        errors = validate_large_checkpoint_policy(record)
        self.assertTrue(any("precreated_revision must be immutable 40-hex" in error for error in errors))

        checkpoint = _checkpoint("initial", 0, "a" * 40, "https://huggingface.co/org/alpha", visibility="public")
        checkpoint["precreated_revision"] = "b" * 40
        record = _record("run-alpha", repo_suffix="alpha")
        record["hf_large_checkpoints"] = [
            checkpoint,
            _checkpoint("periodic", 5, "d" * 40, "https://huggingface.co/org/alpha", visibility="private"),
            _checkpoint("final", 10, "c" * 40, "https://huggingface.co/org/alpha", visibility="public"),
        ]
        errors = validate_large_checkpoint_policy(record)
        self.assertTrue(any("must match record precreated_revision" in error for error in errors))

    def test_unique_repo_is_enforced(self) -> None:
        base = _record("run-alpha", repo_suffix="shared")
        other = _record("run-beta", repo_suffix="shared")
        errors = validate_large_checkpoint_policy([base, other])
        self.assertTrue(any("reuses repo" in error for error in errors))

    def test_lifecycle_coverage_is_required(self) -> None:
        record = _record("run-alpha", repo_suffix="alpha")
        assert isinstance(record["hf_large_checkpoints"], list)
        record["hf_large_checkpoints"] = [
            copy.deepcopy(record["hf_large_checkpoints"][0]),
            copy.deepcopy(record["hf_large_checkpoints"][2]),
        ]
        errors = validate_large_checkpoint_policy(record)
        self.assertTrue(any("missing kinds" in error for error in errors))

    def test_checkpoint_kind_step_uniqueness(self) -> None:
        record = _record("run-alpha", repo_suffix="alpha")
        assert isinstance(record["hf_large_checkpoints"], list)
        duplicate = copy.deepcopy(record["hf_large_checkpoints"][1])
        duplicate["content_digest"] = compute_large_checkpoint_content_digest(duplicate)
        record["hf_large_checkpoints"].append(duplicate)
        errors = validate_large_checkpoint_policy(record)
        self.assertTrue(any("duplicate checkpoint kind/step pair" in error for error in errors))

    def test_branch_only_evidence_is_rejected(self) -> None:
        record = _record("run-alpha", repo_suffix="alpha")
        assert isinstance(record["hf_large_checkpoints"], list)
        first = record["hf_large_checkpoints"][0]
        assert isinstance(first, dict)
        first["url"] = "https://huggingface.co/org/alpha/tree/main"
        first["content_digest"] = compute_large_checkpoint_content_digest(first)
        errors = validate_large_checkpoint_policy(record)
        self.assertTrue(any("branch-only evidence" in error for error in errors))

    def test_run_id_drift_between_record_and_identities_is_rejected(self) -> None:
        record = _record("run-alpha", repo_suffix="alpha")
        record["wandb_run_identity"]["run_id"] = "run-wandb"
        errors = validate_large_checkpoint_policy(record)
        self.assertTrue(any("run_id drift" in error for error in errors))

    def test_wandb_identity_requires_terminal_state_and_real_booleans(self) -> None:
        record = _record("run-alpha", repo_suffix="alpha")
        record["wandb_run_identity"]["online"] = "true"
        record["wandb_run_identity"]["acknowledged"] = 1
        record["wandb_run_identity"]["state"] = "running"
        errors = validate_large_checkpoint_policy(record)
        self.assertTrue(any("must be the boolean true" in error for error in errors))
        self.assertTrue(any(".state must be terminal" in error for error in errors))

    def test_wandb_host_and_run_url_shape_is_strict(self) -> None:
        record = _record("run-alpha", repo_suffix="alpha")
        record["wandb_run_identity"]["run_url"] = "https://api.wandb.ai/acme-lab/pavlov/runs/run-alpha"
        errors = validate_large_checkpoint_policy(record)
        self.assertTrue(any("exact wandb host/path shape" in error for error in errors))

    def test_tinker_identity_run_id_and_state_are_required(self) -> None:
        record = _record("run-alpha", repo_suffix="alpha")
        record["tinker_run_identity"]["run_id"] = "unknown"
        errors = validate_large_checkpoint_policy(record)
        self.assertTrue(any("tinker_run_identity.run_id must be a non-placeholder string" in error for error in errors))

        record = _record("run-alpha", repo_suffix="alpha")
        record["tinker_run_identity"]["state"] = "in_progress"
        errors = validate_large_checkpoint_policy(record)
        self.assertTrue(any("tinker_run_identity.state/status" in error for error in errors))

    def test_content_digest_required_and_canonical(self) -> None:
        record = _record("run-alpha", repo_suffix="alpha")
        assert isinstance(record["hf_large_checkpoints"], list)
        first = record["hf_large_checkpoints"][0]
        assert isinstance(first, dict)
        first.pop("content_digest")
        errors = validate_large_checkpoint_policy(record)
        self.assertTrue(any("content_digest must be a lowercase" in error for error in errors))

        mutated = _record("run-alpha", repo_suffix="alpha")
        assert isinstance(mutated["hf_large_checkpoints"], list)
        changed = copy.deepcopy(mutated["hf_large_checkpoints"][0])
        assert isinstance(changed, dict)
        changed["step"] = 1
        mutated["hf_large_checkpoints"][0] = changed
        errors = validate_large_checkpoint_policy(mutated)
        self.assertTrue(any("content_digest must match canonical content digest binding" in error for error in errors))

    def test_public_private_safety_policy_is_enforced(self) -> None:
        record = _record("run-alpha", repo_suffix="alpha")
        public = copy.deepcopy(record["hf_large_checkpoints"][0])
        private = copy.deepcopy(record["hf_large_checkpoints"][1])
        assert isinstance(public, dict)
        assert isinstance(private, dict)
        public["safe_public_artifact"] = False
        private["safe_public_artifact"] = True
        public["content_digest"] = compute_large_checkpoint_content_digest(public)
        private["content_digest"] = compute_large_checkpoint_content_digest(private)
        record["hf_large_checkpoints"][0] = public
        record["hf_large_checkpoints"][1] = private
        errors = validate_large_checkpoint_policy(record)
        self.assertTrue(any("safe_public_artifact must be true for public visibility" in error for error in errors))
        self.assertTrue(any("safe_public_artifact must be false for private visibility" in error for error in errors))

    def test_retry_parent_lineage_and_final_revision_match_is_enforced(self) -> None:
        parent = _record("run-parent", repo_suffix="parent")
        child = _record("run-child", repo_suffix="child", parent_run_id="run-parent")
        parent["hf_large_checkpoints"][2]["revision"] = "f" * 40
        parent["hf_large_checkpoints"][2]["url"] = "https://huggingface.co/org/run-large-parent/commit/" + ("f" * 40)
        parent["hf_large_checkpoints"][2]["content_digest"] = compute_large_checkpoint_content_digest(parent["hf_large_checkpoints"][2])
        parent["hf_large_checkpoints"][2]["hf_commit"] = "f" * 40
        child["precreated_revision"] = "e" * 40
        child["hf_large_checkpoints"][0]["precreated_revision"] = "e" * 40
        child["hf_large_checkpoints"][1]["precreated_revision"] = "e" * 40
        child["hf_large_checkpoints"][2]["precreated_revision"] = "e" * 40

        errors = validate_large_checkpoint_policy([parent, child])
        self.assertTrue(any("precreated_revision must match parent final revision" in error for error in errors))

    def test_retry_cycles_are_rejected(self) -> None:
        first = _record("run-a", repo_suffix="a")
        second = _record("run-b", repo_suffix="b")
        first["retry_parent_run_id"] = "run-b"
        second["retry_parent_run_id"] = "run-a"
        errors = validate_large_checkpoint_policy([first, second])
        self.assertTrue(any("cycle" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
