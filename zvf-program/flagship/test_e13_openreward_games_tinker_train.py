#!/usr/bin/env python3
"""Tests for the E13 -> Tinker LoRA RL training driver.

No test here makes a network call, constructs a Tinker client, or spends
anything. The W&B and sampling surfaces are exercised through fakes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import base64
import tempfile
import unittest
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

try:  # package import
    from flagship.e13_openreward_games_local_runner import (
        EpisodeRecord,
        GameTaskSpec,
        build_e13_provider_execution_request,
        collect_e13_provider_signed_result,
        main as e13_runner_main,
        parse_split_manifest,
        validate_provider_suite_grant,
    )
    from flagship.e13_openreward_games_tinker_train import (
        HOLDOUT_SEED_BASE,
        BudgetError,
        E13TrainConfig,
        LicenseRecordError,
        NativeGameReward,
        SplitFirewall,
        SplitLeakError,
        assert_within_cap,
        build_checkpoint_record,
        license_record,
        load_budget,
        normalize_group_rewards,
        plan,
        project_cost,
        project_episode_tokens,
        start_wandb_online,
    )
    from flagship.tinker_model_policy import ModelPolicyError
except ImportError:  # direct import
    from e13_openreward_games_local_runner import (
        EpisodeRecord,
        GameTaskSpec,
        build_e13_provider_execution_request,
        collect_e13_provider_signed_result,
        main as e13_runner_main,
        parse_split_manifest,
        validate_provider_suite_grant,
    )
    from e13_openreward_games_tinker_train import (
        HOLDOUT_SEED_BASE,
        BudgetError,
        E13TrainConfig,
        LicenseRecordError,
        NativeGameReward,
        SplitFirewall,
        SplitLeakError,
        assert_within_cap,
        build_checkpoint_record,
        license_record,
        load_budget,
        normalize_group_rewards,
        plan,
        project_cost,
        project_episode_tokens,
        start_wandb_online,
    )
    from tinker_model_policy import ModelPolicyError

VARIANTS = ("Wordle-v0", "Wordle-v0-hardcore")
TEST_E13_ROOT = {
    "schema_version": "provider-lane-trust-root-v1",
    "lane": "E13",
    "suite_id": "openreward_games_eval",
    "provider": "OpenReward",
    "key_id": "openreward-test-e13",
    "public_key_hex": "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
}
PER_VARIANT = 5


def manifest(split: str, *, offset: int, variants=VARIANTS) -> dict:
    tasks = [
        {"id": f"{v}_seed{i + offset}", "env_id": v, "seed": i + offset, "variant": v}
        for v in variants
        for i in range(PER_VARIANT)
    ]
    return {
        "environment": "GeneralReasoning/Wordle",
        "split": split,
        "source_revision": "92bea32efa102e86275dedd2e0367e86d3754754",
        "synthetic": False,
        "tasks": tasks,
    }


def firewall() -> SplitFirewall:
    return SplitFirewall(
        parse_split_manifest(manifest("train", offset=0)),
        parse_split_manifest(manifest("test", offset=HOLDOUT_SEED_BASE)),
    )


class SplitFirewallTests(unittest.TestCase):
    def test_provider_grant_rejects_local_wordle_selection(self):
        fw = firewall()
        grant = {
            "schema_version": "e13-openreward-provider-grant-v1",
            "suite_id": "openreward_games_eval",
            "provider": "OpenReward",
            "issued_at": "2026-08-30T00:00:00Z",
            "expires_at": "2030-08-30T00:00:00Z",
            "grant_id": "g1",
            "license": {"approved": True, "sha256": "a" * 64},
            "deployment": {
                "approved": True,
                "sha256": "b" * 64,
                "revision": fw.evaluation.source_revision,
            },
            "runtime": {
                "approved": True,
                "sha256": "d" * 64,
                "revision": fw.evaluation.source_revision,
            },
            "grader": {"approved": True, "sha256": "c" * 64},
            "train_manifest_sha256": fw.train.digest(),
            "heldout_manifest_sha256": fw.evaluation.digest(),
            "train_task_count": len(fw.train.tasks),
            "heldout_task_count": len(fw.evaluation.tasks),
        }
        grant["signature_key_id"] = TEST_E13_ROOT["key_id"]
        grant["signature"] = base64.b64encode(
            Ed25519PrivateKey.from_private_bytes(
                bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
            ).sign(json.dumps(grant, sort_keys=True, separators=(",", ":")).encode())
        ).decode()
        with self.assertRaises(ValueError):
            validate_provider_suite_grant(grant, fw.train, fw.evaluation)
        self.assertEqual(
            validate_provider_suite_grant(grant, fw.train, fw.evaluation, trust_root=TEST_E13_ROOT)[
                "grant_id"
            ],
            "g1",
        )
        grant["heldout_manifest_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_provider_suite_grant(grant, fw.train, fw.evaluation, trust_root=TEST_E13_ROOT)

    def test_builds_over_separated_manifests(self):
        fw = firewall()
        self.assertTrue(fw.proof.holds)

    def test_refuses_to_build_over_overlapping_manifests(self):
        with self.assertRaises(SplitLeakError):
            SplitFirewall(
                parse_split_manifest(manifest("train", offset=0)),
                parse_split_manifest(manifest("test", offset=0)),
            )

    def test_admits_a_train_seed(self):
        fw = firewall()
        task = GameTaskSpec(id="Wordle-v0_seed0", env_id="Wordle-v0", seed=0, variant="Wordle-v0")
        self.assertIs(fw.assert_train_seed(task), task)
        self.assertIn(("Wordle-v0", 0), fw.admitted_train)

    def test_rejects_holdout_seed_in_training(self):
        fw = firewall()
        task = GameTaskSpec(
            id="Wordle-v0_seed10000",
            env_id="Wordle-v0",
            seed=HOLDOUT_SEED_BASE,
            variant="Wordle-v0",
        )
        with self.assertRaises(SplitLeakError) as ctx:
            fw.assert_train_seed(task)
        self.assertIn("held-out range", str(ctx.exception))

    def test_rejects_train_seed_in_evaluation(self):
        fw = firewall()
        task = GameTaskSpec(id="Wordle-v0_seed0", env_id="Wordle-v0", seed=0, variant="Wordle-v0")
        with self.assertRaises(SplitLeakError) as ctx:
            fw.assert_eval_seed(task)
        self.assertIn("train range", str(ctx.exception))

    def test_rejects_seed_absent_from_manifest(self):
        fw = firewall()
        ghost = GameTaskSpec(
            id="Wordle-v0_seed777", env_id="Wordle-v0", seed=777, variant="Wordle-v0"
        )
        with self.assertRaises(SplitLeakError):
            fw.assert_train_seed(ghost)

    def test_rejects_unknown_variant(self):
        fw = firewall()
        odd = GameTaskSpec(id="Wordle-v9_seed0", env_id="Wordle-v9", seed=0, variant="Wordle-v9")
        with self.assertRaises(SplitLeakError):
            fw.assert_train_seed(odd)

    def test_assert_no_leak_passes_when_disjoint(self):
        fw = firewall()
        fw.assert_train_seed(GameTaskSpec(id="a", env_id="Wordle-v0", seed=1, variant="Wordle-v0"))
        fw.assert_eval_seed(
            GameTaskSpec(
                id="b", env_id="Wordle-v0", seed=HOLDOUT_SEED_BASE + 1, variant="Wordle-v0"
            )
        )
        fw.assert_no_leak()

    def test_assert_no_leak_catches_a_manufactured_overlap(self):
        fw = firewall()
        fw.admitted_train.add(("Wordle-v0", 3))
        fw.admitted_eval.add(("Wordle-v0", 3))
        with self.assertRaises(SplitLeakError):
            fw.assert_no_leak()


class E13SignedProviderResultTests(unittest.TestCase):
    _PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
    )
    _MODEL_REVISION = hashlib.sha1(b"e13-provider-bound-model").hexdigest()

    @classmethod
    def _sign(cls, payload: dict) -> dict:
        unsigned = copy.deepcopy(payload)
        unsigned["signature_key_id"] = TEST_E13_ROOT["key_id"]
        unsigned.pop("signature", None)
        unsigned["signature"] = base64.b64encode(
            cls._PRIVATE_KEY.sign(
                json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
            )
        ).decode()
        return unsigned

    def _grant(self) -> dict:
        fw = firewall()
        return self._sign(
            {
                "schema_version": "e13-openreward-provider-grant-v1",
                "suite_id": "openreward_games_eval",
                "provider": "OpenReward",
                "issued_at": "2026-08-30T00:00:00Z",
                "expires_at": "2030-08-30T00:00:00Z",
                "grant_id": "openreward-e13-test-grant",
                "license": {"approved": True, "sha256": hashlib.sha256(b"license").hexdigest()},
                "deployment": {
                    "approved": True,
                    "sha256": hashlib.sha256(b"deployment").hexdigest(),
                    "revision": fw.evaluation.source_revision,
                },
                "runtime": {
                    "approved": True,
                    "sha256": hashlib.sha256(b"runtime").hexdigest(),
                    "revision": fw.evaluation.source_revision,
                },
                "grader": {"approved": True, "sha256": hashlib.sha256(b"grader").hexdigest()},
                "train_manifest_sha256": fw.train.digest(),
                "heldout_manifest_sha256": fw.evaluation.digest(),
                "train_task_count": len(fw.train.tasks),
                "heldout_task_count": len(fw.evaluation.tasks),
            }
        )

    def _valid_triplet(self) -> tuple[SplitFirewall, dict, dict, dict]:
        fw = firewall()
        grant = self._grant()
        request = build_e13_provider_execution_request(
            grant,
            fw.train,
            fw.evaluation,
            model_revision=self._MODEL_REVISION,
            trust_root=TEST_E13_ROOT,
        )
        result = self._sign(
            {
                "schema_version": "e13-provider-signed-result-v1",
                "suite_id": "openreward_games_eval",
                "provider": "OpenReward",
                "issued_at": grant["issued_at"],
                "expires_at": grant["expires_at"],
                "grant_fingerprint": hashlib.sha256(
                    json.dumps(grant, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "request_fingerprint": hashlib.sha256(
                    json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "train_manifest_sha256": fw.train.digest(),
                "heldout_manifest_sha256": fw.evaluation.digest(),
                "train_task_count": len(fw.train.tasks),
                "heldout_task_count": len(fw.evaluation.tasks),
                "license_sha256": grant["license"]["sha256"],
                "deployment_sha256": grant["deployment"]["sha256"],
                "deployment_revision": grant["deployment"]["revision"],
                "runtime_sha256": grant["runtime"]["sha256"],
                "runtime_revision": grant["runtime"]["revision"],
                "grader_sha256": grant["grader"]["sha256"],
                "model_revision": self._MODEL_REVISION,
                "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
                "artifact_sha256": hashlib.sha256(b"artifact").hexdigest(),
                "wandb_before_tinker": True,
                "receipts": {
                    "wandb": {
                        "project": "openreward-e13",
                        "entity": "openreward",
                        "run_id": "a1B2c3D4",
                        "run_url": "https://wandb.ai/openreward/openreward-e13/runs/a1B2c3D4",
                    },
                    "tinker": {
                        "job_id": "123e4567-e89b-12d3-a456-426614174000",
                        "receipt_sha256": hashlib.sha256(b"tinker").hexdigest(),
                    },
                    "hugging_face": {
                        "repo_id": "openreward/e13-model",
                        "commit": self._MODEL_REVISION,
                        "receipt_sha256": hashlib.sha256(b"huggingface").hexdigest(),
                    },
                },
                "metric": "openreward_games_score",
                "score": 0.75,
                "completed_at": "2026-08-30T00:00:00Z",
            }
        )
        return fw, grant, request, result

    def test_positive_signed_completion_is_the_only_scored_path(self):
        fw, grant, request, result = self._valid_triplet()
        collected = collect_e13_provider_signed_result(
            grant, request, result, fw.train, fw.evaluation, trust_root=TEST_E13_ROOT
        )
        self.assertEqual(collected["status"], "COMPLETE")
        self.assertEqual(collected["score"], 0.75)
        self.assertTrue(collected["provider_signed"])
        self.assertIsNone(request["score"])
        self.assertEqual(request["spent_usd"], 0.0)

    def test_rejects_tampered_request_manifest_and_result_signature(self):
        fw, grant, request, result = self._valid_triplet()
        bad_request = copy.deepcopy(request)
        bad_request["spent_usd"] = 1.0
        with self.assertRaisesRegex(ValueError, "exact validated"):
            collect_e13_provider_signed_result(
                grant, bad_request, result, fw.train, fw.evaluation, trust_root=TEST_E13_ROOT
            )

        bad_manifest = copy.deepcopy(result)
        bad_manifest["heldout_manifest_sha256"] = hashlib.sha256(b"other manifest").hexdigest()
        bad_manifest = self._sign(bad_manifest)
        with self.assertRaisesRegex(ValueError, "bindings"):
            collect_e13_provider_signed_result(
                grant, request, bad_manifest, fw.train, fw.evaluation, trust_root=TEST_E13_ROOT
            )

        bad_signature = copy.deepcopy(result)
        bad_signature["score"] = 0.9
        with self.assertRaisesRegex(ValueError, "signature"):
            collect_e13_provider_signed_result(
                grant, request, bad_signature, fw.train, fw.evaluation, trust_root=TEST_E13_ROOT
            )

    def test_rejects_expired_signature_and_missing_receipts(self):
        fw, grant, request, result = self._valid_triplet()
        expired = copy.deepcopy(result)
        expired["expires_at"] = "2020-08-30T00:00:00Z"
        expired = self._sign(expired)
        with self.assertRaisesRegex(ValueError, "signature"):
            collect_e13_provider_signed_result(
                grant, request, expired, fw.train, fw.evaluation, trust_root=TEST_E13_ROOT
            )
        for receipt_name in ("wandb", "tinker", "hugging_face"):
            with self.subTest(receipt_name=receipt_name):
                missing = copy.deepcopy(result)
                del missing["receipts"][receipt_name]
                missing = self._sign(missing)
                with self.assertRaisesRegex(ValueError, "result.receipts"):
                    collect_e13_provider_signed_result(
                        grant, request, missing, fw.train, fw.evaluation, trust_root=TEST_E13_ROOT
                    )

    def test_rejects_license_deployment_and_grader_mismatches(self):
        fw, grant, request, result = self._valid_triplet()
        for field in ("license_sha256", "deployment_sha256", "runtime_sha256", "grader_sha256"):
            with self.subTest(field=field):
                mismatch = copy.deepcopy(result)
                mismatch[field] = hashlib.sha256(field.encode()).hexdigest()
                mismatch = self._sign(mismatch)
                with self.assertRaisesRegex(ValueError, "bindings"):
                    collect_e13_provider_signed_result(
                        grant, request, mismatch, fw.train, fw.evaluation, trust_root=TEST_E13_ROOT
                    )

    def test_collect_cli_writes_only_verified_provider_completion(self):
        _, grant, request, result = self._valid_triplet()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            paths = {
                "train": directory / "train.json",
                "evaluation": directory / "evaluation.json",
                "grant": directory / "grant.json",
                "root": directory / "root.json",
                "request": directory / "request.json",
                "result": directory / "result.json",
                "out": directory / "collected.json",
            }
            payloads = {
                "train": manifest("train", offset=0),
                "evaluation": manifest("test", offset=HOLDOUT_SEED_BASE),
                "grant": grant,
                "root": TEST_E13_ROOT,
                "request": request,
                "result": result,
            }
            for key, payload in payloads.items():
                paths[key].write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                e13_runner_main(
                    [
                        "--mode",
                        "collect",
                        "--train-manifest",
                        str(paths["train"]),
                        "--eval-manifest",
                        str(paths["evaluation"]),
                        "--provider-grant",
                        str(paths["grant"]),
                        "--trust-root",
                        str(paths["root"]),
                        "--request",
                        str(paths["request"]),
                        "--result",
                        str(paths["result"]),
                        "--out",
                        str(paths["out"]),
                    ]
                ),
                0,
            )
            self.assertEqual(json.loads(paths["out"].read_text())["status"], "COMPLETE")


class ModelPolicyTests(unittest.TestCase):
    def test_authorized_models_pass(self):
        for model in ("Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.5-9B"):
            E13TrainConfig(model=model).validate()

    def test_served_but_unauthorized_model_is_refused(self):
        with self.assertRaises(ModelPolicyError):
            E13TrainConfig(model="Qwen/Qwen3-8B").validate()

    def test_unserved_model_is_refused(self):
        with self.assertRaises(ModelPolicyError):
            E13TrainConfig(model="gpt-4o").validate()

    def test_wandb_project_is_contractual(self):
        with self.assertRaises(ValueError):
            E13TrainConfig(wandb_project="some-other-project").validate()


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.task = GameTaskSpec(
            id="Wordle-v0_seed0", env_id="Wordle-v0", seed=0, variant="Wordle-v0"
        )
        self.reward = NativeGameReward()

    def test_accepts_native_terminal_reward(self):
        r, outcome = self.reward.score(
            EpisodeRecord(task=self.task, steps=3, finished=True, terminal_reward=1.0)
        )
        self.assertEqual(r, 1.0)
        self.assertTrue(outcome.accepted)

    def test_unfinished_episode_yields_no_reward_not_a_guess(self):
        r, outcome = self.reward.score(
            EpisodeRecord(task=self.task, steps=6, finished=False, terminal_reward=None)
        )
        self.assertIsNone(r)
        self.assertFalse(outcome.accepted)
        self.assertEqual(self.reward.rejected, 1)

    def test_out_of_band_reward_is_rejected(self):
        r, _ = self.reward.score(
            EpisodeRecord(task=self.task, steps=1, finished=True, terminal_reward=42.0)
        )
        self.assertIsNone(r)

    def test_group_normalization_centres_rewards(self):
        advs = normalize_group_rewards([1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(sum(advs), 0.0, places=6)
        self.assertGreater(advs[0], 0)

    def test_group_normalization_handles_all_equal(self):
        self.assertEqual(normalize_group_rewards([0.5, 0.5, 0.5]), [0.0, 0.0, 0.0])

    def test_group_normalization_handles_all_rejected(self):
        self.assertEqual(normalize_group_rewards([None, None]), [0.0, 0.0])


class LicenseTests(unittest.TestCase):
    def test_reports_absence_never_an_spdx(self):
        rec = license_record()
        self.assertEqual(rec["observed_state"], "absent_at_pinned_revision")
        self.assertIsNone(rec["claimed_spdx"])
        self.assertIn("LICENSE_RISK_ACCEPTANCE", rec["proceeding_under"])
        self.assertNotIn("MIT", json.dumps(rec))

    def test_fails_closed_when_the_record_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(LicenseRecordError):
                license_record(Path(tmp) / "nope.md")

    def test_fails_closed_when_the_record_does_not_cover_envcommons(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "other.md"
            p.write_text("covers something else entirely", encoding="utf-8")
            with self.assertRaises(LicenseRecordError):
                license_record(p)


class CostTests(unittest.TestCase):
    def setUp(self):
        self.cfg = E13TrainConfig()
        self.budget = load_budget()

    def test_episode_token_profile_is_positive(self):
        per = project_episode_tokens(self.cfg)
        for key in ("prefill_tokens", "sample_tokens", "train_tokens"):
            self.assertGreater(per[key], 0)

    def test_cost_scales_with_episode_count(self):
        one = project_cost(self.cfg, episodes_sampled=1, episodes_trained=1, budget=self.budget)
        ten = project_cost(self.cfg, episodes_sampled=10, episodes_trained=10, budget=self.budget)
        self.assertAlmostEqual(ten["usd"]["total"], one["usd"]["total"] * 10, places=3)

    def test_pinned_prices_are_used(self):
        self.assertEqual(self.budget["usd_per_million_tokens"]["prefill"], 0.54)
        self.assertEqual(self.budget["usd_per_million_tokens"]["sample"], 1.335)
        self.assertEqual(self.budget["usd_per_million_tokens"]["train"], 1.177)

    def test_cap_check_passes_for_a_small_run(self):
        proj = project_cost(self.cfg, episodes_sampled=10, episodes_trained=10, budget=self.budget)
        assert_within_cap(proj, budget=self.budget)

    def test_cap_check_rejects_an_oversized_run(self):
        proj = project_cost(
            self.cfg, episodes_sampled=10_000_000, episodes_trained=10_000_000, budget=self.budget
        )
        with self.assertRaises(BudgetError):
            assert_within_cap(proj, budget=self.budget)

    def test_reserve_must_stay_unspent(self):
        """A projection that fits the cap but eats the reserve is refused."""
        spendable = self.budget["operational_cap_usd"] - self.budget["safety_reserve_usd"]
        proj = {"usd": {"total": spendable + 0.01}}
        with self.assertRaises(BudgetError):
            assert_within_cap(proj, budget=self.budget)


class _FakeRun:
    def __init__(self, **kw):
        self.id = kw.get("id", "run123")
        self.mode = kw.get("mode", "online")
        self.disabled = kw.get("disabled", False)
        self.offline = kw.get("offline", False)
        self.logged: list[dict] = []

    def log(self, payload):
        self.logged.append(payload)
        return True


class WandbGuardTests(unittest.TestCase):
    def test_refuses_non_online_wandb_mode(self):
        import os

        old = os.environ.get("WANDB_MODE")
        os.environ["WANDB_MODE"] = "offline"
        try:
            with self.assertRaises(RuntimeError):
                start_wandb_online(E13TrainConfig())
        finally:
            os.environ.pop("WANDB_MODE", None)
            if old is not None:
                os.environ["WANDB_MODE"] = old

    def test_refuses_when_wandb_disabled(self):
        import os

        os.environ["WANDB_DISABLED"] = "1"
        try:
            with self.assertRaises(RuntimeError):
                start_wandb_online(E13TrainConfig())
        finally:
            os.environ.pop("WANDB_DISABLED", None)

    def test_fake_run_logs_the_contract_metrics(self):
        try:
            from flagship.e13_openreward_games_tinker_train import wandb_log
        except ImportError:
            from e13_openreward_games_tinker_train import wandb_log
        run = _FakeRun()
        for metric in ("train/reward", "train/loss", "train/step", "eval/reward"):
            wandb_log(run, {metric: 1.0})
        keys = {k for entry in run.logged for k in entry}
        self.assertEqual(keys, {"train/reward", "train/loss", "train/step", "eval/reward"})

    def test_rejected_log_is_fatal(self):
        try:
            from flagship.e13_openreward_games_tinker_train import wandb_log
        except ImportError:
            from e13_openreward_games_tinker_train import wandb_log

        class Rejecting(_FakeRun):
            def log(self, payload):
                return False

        with self.assertRaises(RuntimeError):
            wandb_log(Rejecting(), {"train/reward": 1.0})


class CheckpointExportTests(unittest.TestCase):
    def test_record_reports_probe_blocked_without_hf_fields(self):
        fw = firewall()
        rec = build_checkpoint_record(
            cfg=E13TrainConfig(), sampler_path="tinker://sampler/abc", step=10, firewall=fw
        )
        self.assertFalse(rec["e11_transfer_probe"]["ready"])
        self.assertIn("HF_TOKEN", rec["e11_transfer_probe"]["blocker"])
        self.assertIsNone(rec["license"]["claimed_spdx"])

    def test_record_is_e11_ready_when_hf_fields_present(self):
        fw = firewall()
        rec = build_checkpoint_record(
            cfg=E13TrainConfig(),
            sampler_path="tinker://sampler/abc",
            step=10,
            firewall=fw,
            hf_repo="org/repo",
            hf_revision="ckpt-10",
            hf_commit="a" * 40,
        )
        self.assertTrue(rec["e11_transfer_probe"]["ready"])
        cmd = rec["e11_transfer_probe"]["command"]
        for flag in ("--sampler-path", "--hf-repo", "--hf-revision", "--hf-commit"):
            self.assertIn(flag, cmd)

    def test_rejects_non_hex_commit(self):
        with self.assertRaises(ValueError):
            build_checkpoint_record(
                cfg=E13TrainConfig(),
                sampler_path="p",
                step=1,
                firewall=firewall(),
                hf_repo="o/r",
                hf_revision="rev",
                hf_commit="not-hex",
            )

    def test_export_blocked_by_a_leak(self):
        fw = firewall()
        fw.admitted_train.add(("Wordle-v0", 2))
        fw.admitted_eval.add(("Wordle-v0", 2))
        with self.assertRaises(SplitLeakError):
            build_checkpoint_record(cfg=E13TrainConfig(), sampler_path="p", step=1, firewall=fw)


class PlanTests(unittest.TestCase):
    def test_plan_spends_nothing_and_projects_three_numbers(self):
        result = plan(E13TrainConfig(), firewall())
        self.assertEqual(result["spent_usd"], 0.0)
        for key in ("one_smoke_episode", "short_pilot", "full_pass_200_train_tasks"):
            self.assertIn(key, result["projections"])
            self.assertGreater(result["projections"][key]["usd"]["total"], 0)

    def test_plan_never_claims_a_license(self):
        self.assertNotIn('"MIT"', json.dumps(plan(E13TrainConfig(), firewall())))

    def test_plan_reports_separation_holding(self):
        result = plan(E13TrainConfig(), firewall())
        self.assertTrue(result["split_firewall"]["separation_proof"]["holds"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
