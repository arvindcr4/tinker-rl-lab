#!/usr/bin/env python3
"""Unit tests for the E13 OpenReward games local runner.

Every manifest in this file is SYNTHETIC. Nothing here is derived from a model
rollout and nothing here may ever be read as a benchmark score. The fixtures
mimic the shape of the public ``EnvCommons`` game environments (train seeds in
``[0, N)``, eval seeds in ``[10000, 10000+N)``) so the separation logic is
exercised against the real convention, but the values are fabricated.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from e13_openreward_games_local_runner import (
    SYNTHETIC_FIXTURE_MARKER,
    EpisodeRecord,
    GameManifestError,
    GameTaskSpec,
    ProgrammaticRewardVerifier,
    ReceiptIntegrityError,
    build_receipt,
    emit_receipt,
    main,
    parse_split_manifest,
    parse_task_spec,
    prove_seed_separation,
    verify_episodes,
)
import pytest

SYNTHETIC_ENV = "SYNTHETIC-FIXTURE/DoNotScore"
SYNTHETIC_REV = "0" * 40
VARIANTS = ("DoNotScore-v0", "DoNotScore-v0-hardcore")
PER_VARIANT = 5


def synthetic_manifest(
    split: str,
    *,
    offset: int,
    environment: str = SYNTHETIC_ENV,
    revision: str = SYNTHETIC_REV,
    variants=VARIANTS,
) -> dict:
    """Build a synthetic split manifest mirroring the upstream seed convention."""
    tasks = []
    for variant in variants:
        for idx in range(PER_VARIANT):
            seed = idx + offset
            tasks.append(
                {"id": f"{variant}_seed{seed}", "env_id": variant, "seed": seed, "variant": variant}
            )
    return {
        "environment": environment,
        "split": split,
        "source_revision": revision,
        "synthetic": True,
        "tasks": tasks,
    }


class TaskSpecSchemaTests(unittest.TestCase):
    def test_parses_a_well_formed_task(self):
        spec = parse_task_spec({"id": "A_seed3", "env_id": "A", "seed": 3, "variant": "A"})
        assert spec == GameTaskSpec(id="A_seed3", env_id="A", seed=3, variant="A")

    def test_variant_defaults_to_env_id(self):
        assert parse_task_spec({"id": "A_seed0", "env_id": "A", "seed": 0}).variant == "A"

    def test_rejects_unknown_keys(self):
        with pytest.raises(GameManifestError):
            parse_task_spec({"id": "A", "env_id": "A", "seed": 0, "answer": "leaked"})

    def test_rejects_boolean_seed(self):
        with pytest.raises(GameManifestError):
            parse_task_spec({"id": "A", "env_id": "A", "seed": True})

    def test_rejects_negative_seed(self):
        with pytest.raises(GameManifestError):
            parse_task_spec({"id": "A", "env_id": "A", "seed": -1})

    def test_rejects_empty_id(self):
        with pytest.raises(GameManifestError):
            parse_task_spec({"id": "  ", "env_id": "A", "seed": 0})


class SplitManifestSchemaTests(unittest.TestCase):
    def test_parses_synthetic_manifest(self):
        manifest = parse_split_manifest(synthetic_manifest("train", offset=0))
        assert len(manifest.tasks) == PER_VARIANT * len(VARIANTS)
        assert manifest.variants == frozenset(VARIANTS)
        assert manifest.synthetic

    def test_rejects_bad_environment_name(self):
        payload = synthetic_manifest("train", offset=0, environment="NoSlash")
        with pytest.raises(GameManifestError):
            parse_split_manifest(payload)

    def test_rejects_non_hex_revision(self):
        payload = synthetic_manifest("train", offset=0, revision="not-a-commit")
        with pytest.raises(GameManifestError):
            parse_split_manifest(payload)

    def test_rejects_empty_task_list(self):
        payload = synthetic_manifest("train", offset=0)
        payload["tasks"] = []
        with pytest.raises(GameManifestError):
            parse_split_manifest(payload)

    def test_same_seed_across_variants_is_allowed(self):
        """Upstream emits seed 0 for every variant; that is not a duplicate."""
        manifest = parse_split_manifest(synthetic_manifest("train", offset=0))
        seeds_per_variant = {}
        for task in manifest.tasks:
            seeds_per_variant.setdefault(task.seed, set()).add(task.variant)
        assert seeds_per_variant[0] == set(VARIANTS)
        assert len(manifest.instance_keys) == PER_VARIANT * len(VARIANTS)

    def test_rejects_duplicate_instance_keys(self):
        payload = synthetic_manifest("train", offset=0)
        payload["tasks"][1] = dict(payload["tasks"][0], id="different-id")
        with pytest.raises(GameManifestError):
            parse_split_manifest(payload)

    def test_digest_is_order_independent_and_content_sensitive(self):
        base = synthetic_manifest("train", offset=0)
        shuffled = dict(base, tasks=list(reversed(base["tasks"])))
        assert parse_split_manifest(base).digest() == parse_split_manifest(shuffled).digest()
        mutated = json.loads(json.dumps(base))
        mutated["tasks"][0]["seed"] = 9999
        mutated["tasks"][0]["id"] = "DoNotScore-v0_seed9999"
        assert parse_split_manifest(base).digest() != parse_split_manifest(mutated).digest()


class SeedSeparationTests(unittest.TestCase):
    def test_disjoint_splits_prove_separation(self):
        train = parse_split_manifest(synthetic_manifest("train", offset=0))
        evaluation = parse_split_manifest(synthetic_manifest("test", offset=10000))
        proof = prove_seed_separation(train, evaluation)
        assert proof.holds
        assert proof.shared_instances == ()
        assert proof.shared_seeds == ()
        assert proof.shared_task_ids == ()
        assert proof.variant_coverage_matches
        assert proof.eval_instance_count == PER_VARIANT * len(VARIANTS)

    def test_overlapping_instances_break_the_proof(self):
        train = parse_split_manifest(synthetic_manifest("train", offset=0))
        evaluation = parse_split_manifest(synthetic_manifest("test", offset=PER_VARIANT - 1))
        proof = prove_seed_separation(train, evaluation)
        assert not proof.holds
        assert proof.shared_instances
        assert any("instance" in v for v in proof.violations)

    def test_revision_mismatch_breaks_the_proof(self):
        train = parse_split_manifest(synthetic_manifest("train", offset=0))
        evaluation = parse_split_manifest(
            synthetic_manifest("test", offset=10000, revision="1" * 40)
        )
        proof = prove_seed_separation(train, evaluation)
        assert not proof.holds
        assert any("source_revision" in v for v in proof.violations)

    def test_environment_mismatch_breaks_the_proof(self):
        train = parse_split_manifest(synthetic_manifest("train", offset=0))
        evaluation = parse_split_manifest(
            synthetic_manifest("test", offset=10000, environment="SYNTHETIC-FIXTURE/Other")
        )
        assert not prove_seed_separation(train, evaluation).holds

    def test_variant_coverage_mismatch_breaks_the_proof(self):
        train = parse_split_manifest(synthetic_manifest("train", offset=0))
        evaluation = parse_split_manifest(
            synthetic_manifest("test", offset=10000, variants=(VARIANTS[0],))
        )
        proof = prove_seed_separation(train, evaluation)
        assert not proof.holds
        assert not proof.variant_coverage_matches

    def test_same_split_name_breaks_the_proof(self):
        train = parse_split_manifest(synthetic_manifest("train", offset=0))
        evaluation = parse_split_manifest(synthetic_manifest("train", offset=10000))
        assert not prove_seed_separation(train, evaluation).holds


TASK = GameTaskSpec(
    id="DoNotScore-v0_seed10000", env_id="DoNotScore-v0", seed=10000, variant="DoNotScore-v0"
)


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.verifier = ProgrammaticRewardVerifier()

    def test_accepts_a_terminal_in_band_reward(self):
        outcome = self.verifier.verify(
            EpisodeRecord(task=TASK, steps=4, finished=True, terminal_reward=1.0)
        )
        assert outcome.accepted
        assert outcome.reward == 1.0

    def test_rejects_unfinished_episode(self):
        outcome = self.verifier.verify(
            EpisodeRecord(task=TASK, steps=6, finished=False, terminal_reward=0.5)
        )
        assert not outcome.accepted
        assert outcome.reward is None

    def test_rejects_missing_reward(self):
        outcome = self.verifier.verify(
            EpisodeRecord(task=TASK, steps=6, finished=True, terminal_reward=None)
        )
        assert not outcome.accepted

    def test_rejects_out_of_band_reward(self):
        outcome = self.verifier.verify(
            EpisodeRecord(task=TASK, steps=1, finished=True, terminal_reward=7.5)
        )
        assert not outcome.accepted
        assert any("outside declared band" in r for r in outcome.reasons)

    def test_rejects_nan_reward(self):
        outcome = self.verifier.verify(
            EpisodeRecord(task=TASK, steps=1, finished=True, terminal_reward=float("nan"))
        )
        assert not outcome.accepted

    def test_rejects_zero_step_episode(self):
        outcome = self.verifier.verify(
            EpisodeRecord(task=TASK, steps=0, finished=True, terminal_reward=1.0)
        )
        assert not outcome.accepted

    def test_verify_episodes_maps_over_the_batch(self):
        episodes = [
            EpisodeRecord(task=TASK, steps=2, finished=True, terminal_reward=1.0),
            EpisodeRecord(task=TASK, steps=2, finished=False, terminal_reward=1.0),
        ]
        outcomes = verify_episodes(self.verifier, episodes)
        assert [o.accepted for o in outcomes] == [True, False]


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.train = parse_split_manifest(synthetic_manifest("train", offset=0))
        self.evaluation = parse_split_manifest(synthetic_manifest("test", offset=10000))
        self.proof = prove_seed_separation(self.train, self.evaluation)
        self.accepted = (
            ProgrammaticRewardVerifier().verify(
                EpisodeRecord(task=TASK, steps=3, finished=True, terminal_reward=1.0)
            ),
        )

    def test_harness_validation_never_carries_a_score(self):
        receipt = build_receipt(
            lane="E13",
            suite="openreward_games_eval",
            status="PARTIAL",
            separation=self.proof,
            outcomes=self.accepted,
            run_kind="harness_validation",
            is_model_score=False,
            synthetic=True,
        )
        assert receipt["score"] is None
        assert not receipt["is_model_score"]
        assert receipt["synthetic_fixture"] == SYNTHETIC_FIXTURE_MARKER
        assert receipt["score_withheld_because"]

    def test_blocked_status_withholds_score(self):
        receipt = build_receipt(
            lane="E13",
            suite="openreward_games_eval",
            status="BLOCKED",
            separation=self.proof,
            outcomes=self.accepted,
            run_kind="model_rollout",
            is_model_score=True,
            synthetic=False,
        )
        assert receipt["score"] is None
        assert "status is BLOCKED" in receipt["score_withheld_because"]

    def test_missing_separation_proof_withholds_score(self):
        receipt = build_receipt(
            lane="E13",
            suite="openreward_games_eval",
            status="RUNNING",
            separation=None,
            outcomes=self.accepted,
            run_kind="model_rollout",
            is_model_score=True,
        )
        assert receipt["score"] is None
        assert "no seed-separation proof supplied" in receipt["score_withheld_because"]

    def test_broken_separation_proof_withholds_score(self):
        bad = prove_seed_separation(
            self.train, parse_split_manifest(synthetic_manifest("test", offset=0))
        )
        receipt = build_receipt(
            lane="E13",
            suite="openreward_games_eval",
            status="RUNNING",
            separation=bad,
            outcomes=self.accepted,
            run_kind="model_rollout",
            is_model_score=True,
        )
        assert receipt["score"] is None

    def test_rejected_episode_withholds_score(self):
        rejected = ProgrammaticRewardVerifier().verify(
            EpisodeRecord(task=TASK, steps=3, finished=False, terminal_reward=1.0)
        )
        receipt = build_receipt(
            lane="E13",
            suite="openreward_games_eval",
            status="RUNNING",
            separation=self.proof,
            outcomes=(rejected,),
            run_kind="model_rollout",
            is_model_score=True,
        )
        assert receipt["score"] is None

    def test_all_local_gates_passing_remains_blocked_and_unscored(self):
        receipt = build_receipt(
            lane="E13",
            suite="openreward_games_eval",
            status="RUNNING",
            separation=self.proof,
            outcomes=self.accepted,
            run_kind="model_rollout",
            is_model_score=True,
            synthetic=False,
        )
        assert receipt["score"] is None
        assert (
            "local verifier outcomes are not a provider-signed native OpenReward result"
            in receipt["score_withheld_because"]
        )

    def test_is_model_score_requires_model_rollout(self):
        with pytest.raises(ReceiptIntegrityError):
            build_receipt(
                lane="E13",
                suite="openreward_games_eval",
                status="RUNNING",
                separation=self.proof,
                outcomes=self.accepted,
                run_kind="harness_validation",
                is_model_score=True,
            )

    def test_synthetic_run_cannot_be_a_model_score(self):
        with pytest.raises(ReceiptIntegrityError):
            build_receipt(
                lane="E13",
                suite="openreward_games_eval",
                status="RUNNING",
                separation=self.proof,
                outcomes=self.accepted,
                run_kind="model_rollout",
                is_model_score=True,
                synthetic=True,
            )

    def test_invalid_status_is_rejected(self):
        with pytest.raises(ReceiptIntegrityError):
            build_receipt(
                lane="E13",
                suite="openreward_games_eval",
                status="DONE",
                separation=self.proof,
                outcomes=(),
            )

    def test_emit_refuses_a_tampered_receipt(self):
        receipt = build_receipt(
            lane="E13",
            suite="openreward_games_eval",
            status="PARTIAL",
            separation=self.proof,
            outcomes=self.accepted,
            run_kind="harness_validation",
            is_model_score=False,
            synthetic=True,
        )
        receipt["score"] = 0.99  # hand-edited after the fact
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ReceiptIntegrityError):
                emit_receipt(Path(tmp) / "receipt.json", receipt)

    def test_emit_refuses_caller_supplied_local_model_score(self):
        receipt = build_receipt(
            lane="E13",
            suite="openreward_games_eval",
            status="RUNNING",
            separation=self.proof,
            outcomes=self.accepted,
            run_kind="model_rollout",
            is_model_score=True,
            synthetic=False,
        )
        receipt["score"] = 1.0
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ReceiptIntegrityError):
                emit_receipt(Path(tmp) / "receipt.json", receipt)

    def test_emit_writes_a_clean_receipt(self):
        receipt = build_receipt(
            lane="E13",
            suite="openreward_games_eval",
            status="BLOCKED",
            separation=self.proof,
            outcomes=(),
            synthetic=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = emit_receipt(Path(tmp) / "nested" / "receipt.json", receipt)
            written = json.loads(path.read_text())
        assert written["score"] is None
        assert written["synthetic_fixture"] == SYNTHETIC_FIXTURE_MARKER


class CliTests(unittest.TestCase):
    def _write(self, directory: Path, name: str, payload: dict) -> str:
        path = directory / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def test_cli_returns_zero_on_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            train = self._write(d, "train.json", synthetic_manifest("train", offset=0))
            evaluation = self._write(d, "test.json", synthetic_manifest("test", offset=10000))
            out = str(d / "receipt.json")
            assert (
                main(["--train-manifest", train, "--eval-manifest", evaluation, "--out", out]) == 0
            )
            receipt = json.loads(Path(out).read_text())
        assert receipt["score"] is None
        assert receipt["seed_separation"]["holds"]

    def test_cli_returns_nonzero_on_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            train = self._write(d, "train.json", synthetic_manifest("train", offset=0))
            evaluation = self._write(d, "test.json", synthetic_manifest("test", offset=0))
            out = str(d / "receipt.json")
            assert (
                main(["--train-manifest", train, "--eval-manifest", evaluation, "--out", out]) == 1
            )
            receipt = json.loads(Path(out).read_text())
        assert receipt["score"] is None
        assert receipt["status"] == "BLOCKED"


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
