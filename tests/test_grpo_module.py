"""Tests for the consolidated ``tinkerrl.grpo`` module."""

import unittest
import json
import sys
import subprocess
import tempfile
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

from platform_tinker.tinkerrl import grpo
from platform_tinker.tinkerrl.grpo import (
    GRPOConfig,
    GRPORunResult,
    InMemoryDataset,
    MathReward,
    ExactMathReward,
    PAVLOV_DECLARED_DOMAINS,
    PAVLOV_DOMAIN_TAGS,
    PAVLOV_HELDOUT_SUITE_IDS,
    PAVLOV_PRIMARY_EVALUATION_DOMAIN_UNION,
    PAVLOV_PRIMARY_EVALUATION_SUITE_IDS,
    PAVLOV_TRAINING_DOMAIN_UNION,
    PAVLOV_TRAINING_SUITE_IDS,
    ToolCallReward,
    StrictToolCallReward,
    TrainingExample,
    make_grpo_loss_fn,
    make_synthetic_math_dataset,
    make_synthetic_tool_use_dataset,
    make_xlam_dataset,
    normalize_rewards,
    run_grpo,
)


class TestNormalizeRewards(unittest.TestCase):
    def test_basic(self):
        advs = normalize_rewards([1.0, 2.0, 3.0, 4.0, 5.0])
        mean = sum(advs) / len(advs)
        self.assertAlmostEqual(mean, 0.0, places=7)
        std = (sum((a - mean) ** 2 for a in advs) / len(advs)) ** 0.5
        self.assertAlmostEqual(std, 1.0, places=5)

    def test_identical(self):
        advs = normalize_rewards([3.0, 3.0, 3.0])
        for a in advs:
            self.assertAlmostEqual(a, 0.0, places=7)

    def test_empty(self):
        self.assertEqual(normalize_rewards([]), [])

    def test_single(self):
        advs = normalize_rewards([42.0])
        self.assertAlmostEqual(advs[0], 0.0, places=7)

    def test_monotonic(self):
        advs = normalize_rewards([1.0, 2.0, 3.0, 4.0, 5.0])
        for i in range(len(advs) - 1):
            self.assertLess(advs[i], advs[i + 1])


class TestMakeGrpoLossFn(unittest.TestCase):
    def test_positive_advantage(self):
        import torch

        loss_fn = make_grpo_loss_fn([2.0])
        logprobs = [torch.tensor([-0.5, -0.2, -0.1], requires_grad=True)]
        loss, metrics = loss_fn(None, logprobs)
        expected = -(2.0) * (-0.8)
        self.assertAlmostEqual(loss.item(), expected, places=5)
        self.assertEqual(metrics["grpo_loss"], loss.item())

    def test_negative_advantage(self):
        import torch

        loss_fn = make_grpo_loss_fn([-1.0])
        logprobs = [torch.tensor([-0.5, -0.2, -0.1], requires_grad=True)]
        loss, _ = loss_fn(None, logprobs)
        expected = -(-1.0) * (-0.8)
        self.assertAlmostEqual(loss.item(), expected, places=5)

    def test_zero_advantage(self):
        import torch

        loss_fn = make_grpo_loss_fn([0.0])
        logprobs = [torch.tensor([-0.5, -0.2], requires_grad=True)]
        loss, _ = loss_fn(None, logprobs)
        self.assertEqual(loss.item(), 0.0)

    def test_batch(self):
        import torch

        loss_fn = make_grpo_loss_fn([1.0, -1.0, 0.0])
        logprobs = [
            torch.tensor([-1.0]),
            torch.tensor([-2.0]),
            torch.tensor([-3.0]),
        ]
        loss, _ = loss_fn(None, logprobs)
        expected = (1.0 - 2.0 + 0.0) / 3.0
        self.assertAlmostEqual(loss.item(), expected, places=5)

    def test_empty(self):
        loss_fn = make_grpo_loss_fn([])
        loss, metrics = loss_fn(None, [])
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(metrics["grpo_loss"], 0.0)


class TestGRPOConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = GRPOConfig(name="test")
        self.assertEqual(cfg.model, "Qwen/Qwen3-8B")
        self.assertEqual(cfg.lora_rank, 32)
        self.assertEqual(cfg.steps, 200)
        self.assertEqual(cfg.group_size, 8)
        self.assertEqual(cfg.batch_size, 4)
        self.assertEqual(cfg.lr, 3e-5)
        self.assertEqual(cfg.temperature, 0.8)
        self.assertEqual(cfg.top_p, 0.95)
        self.assertEqual(cfg.max_prompt_tokens, 1024)
        self.assertEqual(cfg.max_response_tokens, 512)
        self.assertIsNone(cfg.save_every)
        self.assertEqual(cfg.seed, 42)
        self.assertEqual(cfg.num_seeds, 1)
        self.assertFalse(cfg.evaluate_heldout)

    def test_effective_save_every_explicit(self):
        cfg = GRPOConfig(name="t", save_every=10)
        self.assertEqual(cfg.effective_save_every(), 10)

    def test_effective_save_every_computed(self):
        cfg = GRPOConfig(name="t", steps=200)
        self.assertEqual(cfg.effective_save_every(), 50)

    def test_effective_save_every_minimum(self):
        cfg = GRPOConfig(name="t", steps=10)
        self.assertEqual(cfg.effective_save_every(), 10)

    def test_tracking_is_mandatory(self):
        with self.assertRaisesRegex(ValueError, "W&B tracking is mandatory"):
            GRPOConfig(name="t", wandb_project=None).validate_tracking()
        with self.assertRaisesRegex(ValueError, "Hugging Face checkpoint tracking"):
            GRPOConfig(name="t", hf_enabled=False).validate_tracking()

    def test_contradictory_campaign_budget_is_fail_closed(self):
        config = GRPOConfig(
            name="blocked-campaign",
            campaign_status="draft-awaiting-budget-cap",
            budget_status="AUTHORIZED_TINKER_ONLY",
            paid_jobs_may_launch=True,
            authorized_budget_usd=18.0,
            maximum_usd=18.0,
        )
        with self.assertRaisesRegex(ValueError, "not launchable"):
            config.validate_campaign_gate()

    def test_campaign_budget_launch_flag_must_be_boolean(self):
        config = GRPOConfig(
            name="malformed-campaign",
            campaign_status="authorized",
            budget_status="AUTHORIZED_TINKER_ONLY",
            paid_jobs_may_launch="yes",  # type: ignore[arg-type]
            authorized_budget_usd=18.0,
            maximum_usd=18.0,
        )
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            config.validate_campaign_gate()

    def test_campaign_metadata_is_detached_and_immutable(self):
        training = ["train-a", "train-b"]
        primary_eval = ["eval-a"]
        domains = ["code", "browser"]
        declared = ["code", "browser", "math"]
        config = GRPOConfig(
            name="metadata",
            training_suite_ids=training,
            primary_evaluation_suite_ids=primary_eval,
            domain_tags=domains,
            declared_domains=declared,
            training_domain_union=declared,
            primary_evaluation_domain_union=declared,
        )
        training.append("secret-train")
        primary_eval.append("secret-eval")
        domains.append("secret-domain")
        declared.append("secret-domain")

        snapshot = grpo._immutable_config(config, config.seed)
        snapshot["training_suite_ids"].append("mutated")
        snapshot["domain_tags"].append("mutated")

        self.assertEqual(config.training_suite_ids, ("train-a", "train-b"))
        self.assertEqual(config.primary_evaluation_suite_ids, ("eval-a",))
        self.assertEqual(config.domain_tags, ("code", "browser"))
        self.assertEqual(config.declared_domains, ("code", "browser", "math"))
        self.assertEqual(config.training_domain_union, ("code", "browser", "math"))
        self.assertNotIn("secret", json.dumps(snapshot))

    def test_campaign_constants_cover_full_pavlov_metadata(self):
        self.assertEqual(len(PAVLOV_TRAINING_SUITE_IDS), 12)
        self.assertEqual(len(PAVLOV_PRIMARY_EVALUATION_SUITE_IDS), 14)
        self.assertEqual(len(PAVLOV_DECLARED_DOMAINS), 16)
        self.assertEqual(len(PAVLOV_TRAINING_DOMAIN_UNION), 16)
        self.assertEqual(len(PAVLOV_PRIMARY_EVALUATION_DOMAIN_UNION), 16)
        self.assertEqual(len(PAVLOV_DOMAIN_TAGS), 14)
        self.assertEqual(len(PAVLOV_HELDOUT_SUITE_IDS), 6)
        self.assertTrue(set(PAVLOV_HELDOUT_SUITE_IDS).issubset(PAVLOV_PRIMARY_EVALUATION_SUITE_IDS))

    def test_campaign_metadata_is_logged_without_credentials(self):
        run = SimpleNamespace(log=Mock(), summary={})
        config = GRPOConfig(
            name="metadata",
            training_suite_ids=("train-a",),
            primary_evaluation_suite_ids=("eval-a",),
            domain_tags=("code", "browser"),
            declared_domains=("code", "browser", "math"),
            training_domain_union=("code", "browser", "math"),
            primary_evaluation_domain_union=("code", "browser", "math"),
        )

        grpo._log_campaign_metadata(run, config)

        payload = run.log.call_args.args[0]
        self.assertEqual(payload["campaign/training_suite_ids"], ["train-a"])
        self.assertEqual(payload["campaign/primary_evaluation_suite_ids"], ["eval-a"])
        self.assertEqual(payload["campaign/domain_tags"], ["code", "browser"])
        self.assertEqual(payload["campaign/declared_domains"], ["code", "browser", "math"])
        self.assertEqual(payload["campaign/training_domain_union"], ["code", "browser", "math"])
        self.assertNotIn("TOKEN", repr(payload).upper())
        self.assertEqual(run.summary["domain_tags"], ["code", "browser"])


class TestTrackingFailClosed(unittest.TestCase):
    def _dataset(self):
        return InMemoryDataset(train=[TrainingExample(prompt="q", target="1")])

    def _config(self):
        return GRPOConfig(name="fail-closed", steps=1, save_every=1)

    def _fake_runtime(self):
        events = []
        holder = {}

        class Future:
            def __init__(self, value):
                self.value = value

            def result(self):
                return self.value

        class Response:
            tokens = [1, 2]

        class Responses:
            sequences = [Response(), Response()]

        class SamplingClient:
            def sample(self, *_args, **_kwargs):
                events.append("sample")
                return Future(Responses())

        class TrainingClient:
            model_id = "tinker-run-1"

            def __init__(self):
                self.save_count = 0
                self.forward_count = 0
                self.optim_count = 0

            def save_weights_for_sampler(self, name):
                self.save_count += 1
                events.append(("save_sampler", name))
                return Future(SimpleNamespace(path=f"tinker://run/sampler/{self.save_count}"))

            def create_sampling_client(self, model_path):
                events.append(("sampling_client", model_path))
                return SamplingClient()

            def forward_backward_custom(self, **_kwargs):
                self.forward_count += 1
                events.append("forward_backward")
                return Future(SimpleNamespace(metrics={"grpo_loss": 0.25}))

            def optim_step(self, _params):
                self.optim_count += 1
                events.append("optim_step")
                return Future(None)

            def save_state(self, name, overwrite):
                events.append(("save_state", name, overwrite))
                return Future(SimpleNamespace(path=f"tinker://run/state/{name}"))

        class ServiceClient:
            def __init__(self, **_kwargs):
                events.append("service_client")
                self.training_client = TrainingClient()
                holder["training_client"] = self.training_client

            def create_lora_training_client(self, **_kwargs):
                events.append("create_training_client")
                return self.training_client

        class HfApi:
            info_count = 0

            def __init__(self, **_kwargs):
                pass

            def whoami(self, **_kwargs):
                return {"name": "owner"}

            def model_info(self, repo_id, revision):
                type(self).info_count += 1
                return SimpleNamespace(sha=f"{type(self).info_count:040x}")

            def create_repo(self, **_kwargs):
                return None

            def create_branch(self, **_kwargs):
                return None

        class Tokenizer:
            def encode(self, _prompt, add_special_tokens=False):
                return [1]

            def decode(self, _tokens, skip_special_tokens=True):
                return "1"

        wandb_run = SimpleNamespace(
            id="wandb-run-1",
            mode="online",
            summary={},
            logs=[],
            finish=Mock(),
        )

        def log(payload):
            wandb_run.logs.append(payload)

        wandb_run.log = log
        wandb = types.ModuleType("wandb")
        wandb.init = Mock(return_value=wandb_run)
        hf = types.ModuleType("huggingface_hub")
        hf.HfApi = HfApi
        tinker = types.ModuleType("tinker")
        tinker.ServiceClient = ServiceClient
        tinker_types = types.ModuleType("tinker.types")
        tinker_types.ModelInput = SimpleNamespace(from_ints=lambda values: values)
        tinker_types.TensorData = lambda **kwargs: SimpleNamespace(**kwargs)
        tinker_types.Datum = lambda **kwargs: SimpleNamespace(**kwargs)
        tinker_types.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
        tinker_types.AdamParams = lambda **kwargs: SimpleNamespace(**kwargs)
        tinker.types = tinker_types
        return {
            "events": events,
            "holder": holder,
            "wandb": wandb,
            "hf": hf,
            "tinker": tinker,
            "tinker_types": tinker_types,
            "tokenizer": Tokenizer(),
            "wandb_run": wandb_run,
        }

    def test_wandb_failure_precedes_service_client(self):
        service_client = Mock()
        wandb = types.ModuleType("wandb")
        wandb.init = Mock(side_effect=RuntimeError("online unavailable"))
        tinker = types.ModuleType("tinker")
        tinker.ServiceClient = service_client

        with patch.dict(sys.modules, {"wandb": wandb, "tinker": tinker}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "W&B online initialization failed"):
                grpo._run_one_seed(self._config(), self._dataset(), ExactMathReward(), tokenizer=None)

        service_client.assert_not_called()

    def test_wandb_without_live_run_id_precedes_service_client(self):
        run = SimpleNamespace(
            id="",
            mode="online",
            summary={},
            log=Mock(),
            finish=Mock(),
        )
        wandb = types.ModuleType("wandb")
        wandb.init = Mock(return_value=run)
        service_client = Mock()
        tinker = types.ModuleType("tinker")
        tinker.ServiceClient = service_client

        with patch.dict(sys.modules, {"wandb": wandb, "tinker": tinker}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "no live run ID"):
                grpo._run_one_seed(
                    self._config(), self._dataset(), ExactMathReward(), tokenizer=None
                )

        service_client.assert_not_called()
        self.assertEqual(run.summary["status"], "failed")
        run.finish.assert_called_once_with(exit_code=1)

    def test_huggingface_auth_failure_precedes_service_client_and_fails_wandb(self):
        run = SimpleNamespace(
            id="wandb-auth-preflight",
            log=Mock(),
            summary={},
            finish=Mock(),
            mode="online",
        )
        wandb = types.ModuleType("wandb")
        wandb.init = Mock(return_value=run)

        class FailingHfApi:
            def __init__(self, **_kwargs):
                pass

            def whoami(self, **_kwargs):
                raise RuntimeError("authentication unavailable")

        hf = types.ModuleType("huggingface_hub")
        hf.HfApi = FailingHfApi
        service_client = Mock()
        tinker = types.ModuleType("tinker")
        tinker.ServiceClient = service_client

        with patch.dict(
            sys.modules,
            {"wandb": wandb, "huggingface_hub": hf, "tinker": tinker},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "Hugging Face authentication preflight failed"):
                grpo._run_one_seed(self._config(), self._dataset(), ExactMathReward(), tokenizer=None)

        service_client.assert_not_called()
        self.assertEqual(run.summary["status"], "failed")
        run.finish.assert_called()

    def test_campaign_gate_precedes_hf_and_service_client(self):
        run = SimpleNamespace(
            id="wandb-campaign-gate",
            log=Mock(),
            summary={},
            finish=Mock(),
            mode="online",
        )
        wandb = types.ModuleType("wandb")
        wandb.init = Mock(return_value=run)
        hf_api = Mock()
        hf = types.ModuleType("huggingface_hub")
        hf.HfApi = hf_api
        service_client = Mock()
        tinker = types.ModuleType("tinker")
        tinker.ServiceClient = service_client
        config = GRPOConfig(
            name="blocked-campaign",
            campaign_status="draft-awaiting-budget-cap",
            paid_jobs_may_launch=True,
            authorized_budget_usd=18.0,
            maximum_usd=18.0,
        )

        with patch.dict(
            sys.modules,
            {"wandb": wandb, "huggingface_hub": hf, "tinker": tinker},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "not launchable"):
                grpo._run_one_seed(config, self._dataset(), ExactMathReward(), tokenizer=None)

        hf_api.assert_not_called()
        service_client.assert_not_called()
        self.assertEqual(run.summary["status"], "failed")
        run.finish.assert_called()

    def test_huggingface_export_failure_propagates(self):
        config = GRPOConfig(name="export", hf_public=True)
        with patch.object(grpo, "_preflight_hf", return_value="owner"):
            with patch.object(grpo, "_prepare_hf_revision"):
                with patch.object(
                    grpo.subprocess,
                    "run",
                    side_effect=subprocess.CalledProcessError(9, ["python"]),
                ):
                    with self.assertRaisesRegex(RuntimeError, "checkpoint export failed"):
                        grpo._publish_checkpoint(
                            config, 1, "tinker://source", lambda _msg: None, step=0
                        )

    def test_hf_export_uses_official_argv_and_records_revision_receipt(self):
        config = GRPOConfig(name="export", hf_public=True)
        api = Mock()
        api.model_info.return_value = SimpleNamespace(sha="a" * 40)
        hf = types.ModuleType("huggingface_hub")
        hf.HfApi = lambda **_kwargs: api
        completed = SimpleNamespace(returncode=0)

        with patch.dict(sys.modules, {"huggingface_hub": hf}, clear=False):
            with patch.object(grpo, "_preflight_hf", return_value="owner"):
                with patch.object(grpo.subprocess, "run", return_value=completed) as run:
                    receipt = grpo._publish_checkpoint(
                        config,
                        1,
                        "tinker://run/sampler/step-0",
                        lambda _msg: None,
                        step=0,
                        return_receipt=True,
                    )

        command = run.call_args.args[0]
        repo_id = command[7]
        revision = command[9]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(
            command,
            [
                sys.executable,
                "-m",
                "tinker.cli",
                "checkpoint",
                "push-hf",
                "tinker://run/sampler/step-0",
                "--repo",
                repo_id,
                "--revision",
                revision,
                "--public",
            ],
        )
        api.model_info.assert_called_once_with(repo_id, revision=revision)
        api.create_repo.assert_called_once_with(
            repo_id=repo_id, private=False, exist_ok=True
        )
        api.create_branch.assert_called_once_with(
            repo_id=repo_id, branch=revision, exist_ok=True
        )
        self.assertEqual(receipt["commit_sha"], "a" * 40)
        self.assertEqual(receipt["hf_commit_sha"], "a" * 40)
        self.assertEqual(receipt["revision"], revision)
        self.assertEqual(receipt["revision_url"], f"https://huggingface.co/{repo_id}/tree/{revision}")
        self.assertEqual(receipt["commit_url"], f"https://huggingface.co/{repo_id}/commit/{'a' * 40}")
        self.assertNotIn("TOKEN", repr(command).upper())

    def test_hf_revision_verification_failure_propagates(self):
        config = GRPOConfig(name="verify-failure")
        api = Mock()
        api.model_info.side_effect = RuntimeError("revision not found")
        hf = types.ModuleType("huggingface_hub")
        hf.HfApi = lambda **_kwargs: api
        with patch.dict(sys.modules, {"huggingface_hub": hf}, clear=False):
            with patch.object(grpo, "_preflight_hf", return_value="owner"):
                with patch.object(grpo.subprocess, "run", return_value=SimpleNamespace()):
                    with self.assertRaisesRegex(RuntimeError, "checkpoint verification failed"):
                        grpo._publish_checkpoint(
                            config,
                            1,
                            "tinker://run/sampler/step-0",
                            lambda _msg: None,
                            step=0,
                            return_receipt=True,
                        )

    def test_checkpoint_receipt_rejects_mismatched_or_noncanonical_urls(self):
        receipt = {
            "step": 0,
            "repo_id": "owner/checkpoint",
            "revision": "checkpoint-step-0",
            "commit_sha": "d" * 40,
            "repo_url": "https://huggingface.co/owner/checkpoint",
            "revision_url": "https://example.invalid/revision",
            "commit_url": "https://huggingface.co/owner/checkpoint/commit/" + "d" * 40,
            "source_path": "tinker://source/0",
        }
        with self.assertRaisesRegex(RuntimeError, "invalid revision URL"):
            grpo._require_checkpoint_receipt(receipt, step=0)
        receipt["revision_url"] = "https://huggingface.co/owner/checkpoint/tree/checkpoint-step-0"
        with self.assertRaisesRegex(RuntimeError, "step mismatch"):
            grpo._require_checkpoint_receipt(receipt, step=1)

    def test_initial_periodic_and_final_checkpoints_have_verified_receipts(self):
        runtime = self._fake_runtime()
        commands = []

        def fake_subprocess(command, **kwargs):
            commands.append((command, kwargs))
            return SimpleNamespace(returncode=0)

        with patch.dict(
            sys.modules,
            {
                "wandb": runtime["wandb"],
                "huggingface_hub": runtime["hf"],
                "tinker": runtime["tinker"],
                "tinker.types": runtime["tinker_types"],
            },
            clear=False,
        ):
            with patch.object(grpo.subprocess, "run", side_effect=fake_subprocess):
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    config = GRPOConfig(
                        name="receipt-run",
                        steps=2,
                        save_every=1,
                        group_size=2,
                        batch_size=1,
                        checkpoint_dir=checkpoint_dir,
                        hf_public=True,
                    )
                    result = grpo._run_one_seed(
                        config,
                        self._dataset(),
                        ExactMathReward(),
                        runtime["tokenizer"],
                        logger=lambda _message: None,
                    )

        self.assertEqual(result.run_id, "tinker-run-1")
        self.assertEqual(len(commands), 4)  # step 0, periodic steps 1/2, final
        self.assertEqual(len(result.checkpoint_receipts), 4)
        self.assertEqual(len(result.checkpoint_urls), 4)
        self.assertEqual(len(result.checkpoint_commit_shas), 4)
        self.assertEqual(
            len({receipt["repo_id"] for receipt in result.checkpoint_receipts}), 4
        )
        self.assertEqual(
            len({receipt["revision"] for receipt in result.checkpoint_receipts}), 4
        )
        for command, kwargs in commands:
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(command[1:5], ["-m", "tinker.cli", "checkpoint", "push-hf"])
            self.assertIn("--repo", command)
            self.assertIn("--revision", command)
            self.assertIn("--public", command)
            self.assertTrue(kwargs["check"])
            self.assertTrue(kwargs["capture_output"])
            self.assertTrue(kwargs["text"])
        for receipt in result.checkpoint_receipts:
            self.assertTrue(receipt["commit_sha"])
            self.assertIn(f"/tree/{receipt['revision']}", receipt["revision_url"])
            self.assertIn(f"/commit/{receipt['commit_sha']}", receipt["commit_url"])
        self.assertEqual(
            runtime["wandb_run"].summary["checkpoint_commit_shas"],
            result.checkpoint_commit_shas,
        )
        self.assertEqual(runtime["wandb_run"].summary["status"], "success")
        self.assertEqual(runtime["wandb_run"].summary["tinker_run_id"], result.run_id)
        runtime["wandb_run"].finish.assert_called_once_with(exit_code=0)

        # W&B owns a detached copy of every receipt, so mutating the returned
        # result cannot rewrite the recorded immutable revision metadata.
        original_revision = runtime["wandb_run"].summary["checkpoint_receipts"][0]["revision"]
        result.checkpoint_receipts[0]["revision"] = "mutated"
        self.assertEqual(
            runtime["wandb_run"].summary["checkpoint_receipts"][0]["revision"],
            original_revision,
        )

    def test_receipt_failure_stops_future_training_and_optimizer_calls(self):
        runtime = self._fake_runtime()
        publish_calls = []

        def fake_receipt(_config, _seed, folder_path, _logger, *, step, **_kwargs):
            publish_calls.append(step)
            if len(publish_calls) == 2:
                raise RuntimeError("receipt verification failed")
            revision = f"checkpoint-step-{step}"
            sha = f"{len(publish_calls):040x}"
            return {
                "step": step,
                "repo_id": "owner/receipt-run",
                "revision": revision,
                "commit_sha": sha,
                "hf_commit_sha": sha,
                "repo_url": "https://huggingface.co/owner/receipt-run",
                "revision_url": f"https://huggingface.co/owner/receipt-run/tree/{revision}",
                "commit_url": f"https://huggingface.co/owner/receipt-run/commit/{sha}",
                "source_path": str(folder_path),
            }

        with patch.dict(
            sys.modules,
            {
                "wandb": runtime["wandb"],
                "huggingface_hub": runtime["hf"],
                "tinker": runtime["tinker"],
                "tinker.types": runtime["tinker_types"],
            },
            clear=False,
        ):
            with patch.object(grpo, "_publish_checkpoint", side_effect=fake_receipt):
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    config = GRPOConfig(
                        name="receipt-failure",
                        steps=2,
                        save_every=1,
                        group_size=2,
                        batch_size=1,
                        checkpoint_dir=checkpoint_dir,
                    )
                    with self.assertRaisesRegex(RuntimeError, "receipt verification failed"):
                        grpo._run_one_seed(
                            config,
                            self._dataset(),
                            ExactMathReward(),
                            runtime["tokenizer"],
                            logger=lambda _message: None,
                        )

        training_client = runtime["holder"]["training_client"]
        self.assertEqual(publish_calls, [0, 1])
        self.assertEqual(training_client.forward_count, 1)
        self.assertEqual(training_client.optim_count, 1)
        self.assertEqual(
            runtime["events"][-1], ("save_sampler", "step_seed42_1")
        )
        self.assertEqual(runtime["wandb_run"].summary["status"], "failed")
        runtime["wandb_run"].finish.assert_called_once_with(exit_code=1)

    def test_initial_receipt_failure_performs_no_paid_training_work(self):
        runtime = self._fake_runtime()
        with patch.dict(
            sys.modules,
            {
                "wandb": runtime["wandb"],
                "huggingface_hub": runtime["hf"],
                "tinker": runtime["tinker"],
                "tinker.types": runtime["tinker_types"],
            },
            clear=False,
        ):
            with patch.object(
                grpo,
                "_publish_checkpoint",
                side_effect=RuntimeError("initial receipt failed"),
            ):
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    config = GRPOConfig(
                        name="initial-failure",
                        steps=1,
                        save_every=1,
                        checkpoint_dir=checkpoint_dir,
                    )
                    with self.assertRaisesRegex(RuntimeError, "initial receipt failed"):
                        grpo._run_one_seed(
                            config,
                            self._dataset(),
                            ExactMathReward(),
                            runtime["tokenizer"],
                            logger=lambda _message: None,
                        )

        training_client = runtime["holder"]["training_client"]
        self.assertEqual(training_client.forward_count, 0)
        self.assertEqual(training_client.optim_count, 0)
        self.assertNotIn("sample", runtime["events"])
        self.assertEqual(runtime["wandb_run"].summary["status"], "failed")
        runtime["wandb_run"].finish.assert_called_once_with(exit_code=1)

    def test_missing_tinker_run_id_stops_before_first_sampler_or_training_call(self):
        runtime = self._fake_runtime()
        original_service = runtime["tinker"].ServiceClient

        def service_without_run_id(**kwargs):
            client = original_service(**kwargs)
            client.training_client.model_id = "   "
            return client

        runtime["tinker"].ServiceClient = service_without_run_id
        with patch.dict(
            sys.modules,
            {
                "wandb": runtime["wandb"],
                "huggingface_hub": runtime["hf"],
                "tinker": runtime["tinker"],
                "tinker.types": runtime["tinker_types"],
            },
            clear=False,
        ):
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                config = GRPOConfig(
                    name="missing-run-id",
                    steps=1,
                    save_every=1,
                    checkpoint_dir=checkpoint_dir,
                )
                with self.assertRaisesRegex(RuntimeError, "no nonempty run ID"):
                    grpo._run_one_seed(
                        config,
                        self._dataset(),
                        ExactMathReward(),
                        runtime["tokenizer"],
                        logger=lambda _message: None,
                    )

        training_client = runtime["holder"]["training_client"]
        self.assertEqual(training_client.save_count, 0)
        self.assertEqual(training_client.forward_count, 0)
        self.assertEqual(training_client.optim_count, 0)
        self.assertEqual(runtime["wandb_run"].summary["status"], "failed")
        runtime["wandb_run"].finish.assert_called_once_with(exit_code=1)

    def test_wandb_checkpoint_log_failure_is_not_swallowed(self):
        runtime = self._fake_runtime()
        receipt = {
            "step": 0,
            "repo_id": "owner/log-failure",
            "revision": "checkpoint-step-0",
            "commit_sha": "b" * 40,
            "hf_commit_sha": "b" * 40,
            "repo_url": "https://huggingface.co/owner/log-failure",
            "revision_url": "https://huggingface.co/owner/log-failure/tree/checkpoint-step-0",
            "commit_url": f"https://huggingface.co/owner/log-failure/commit/{'b' * 40}",
            "source_path": "tinker://source/0",
        }
        with patch.dict(
            sys.modules,
            {
                "wandb": runtime["wandb"],
                "huggingface_hub": runtime["hf"],
                "tinker": runtime["tinker"],
                "tinker.types": runtime["tinker_types"],
            },
            clear=False,
        ):
            with patch.object(grpo, "_publish_checkpoint", return_value=receipt):
                with patch.object(
                    grpo,
                    "_log_wandb_checkpoint",
                    side_effect=RuntimeError("W&B checkpoint log failed"),
                ):
                    with tempfile.TemporaryDirectory() as checkpoint_dir:
                        config = GRPOConfig(
                            name="wandb-log-failure",
                            steps=1,
                            save_every=1,
                            checkpoint_dir=checkpoint_dir,
                        )
                        with self.assertRaisesRegex(RuntimeError, "W&B checkpoint log failed"):
                            grpo._run_one_seed(
                                config,
                                self._dataset(),
                                ExactMathReward(),
                                runtime["tokenizer"],
                                logger=lambda _message: None,
                            )

        training_client = runtime["holder"]["training_client"]
        self.assertEqual(training_client.forward_count, 0)
        self.assertEqual(training_client.optim_count, 0)
        self.assertEqual(runtime["wandb_run"].summary["status"], "failed")
        runtime["wandb_run"].finish.assert_called_once_with(exit_code=1)

    def test_wandb_finish_failure_makes_result_inadmissible(self):
        runtime = self._fake_runtime()
        runtime["wandb_run"].finish.side_effect = RuntimeError("finish unavailable")
        with patch.dict(
            sys.modules,
            {
                "wandb": runtime["wandb"],
                "huggingface_hub": runtime["hf"],
                "tinker": runtime["tinker"],
                "tinker.types": runtime["tinker_types"],
            },
            clear=False,
        ):
            with patch.object(
                grpo,
                "_publish_checkpoint",
                side_effect=lambda _config, _seed, folder_path, _logger, *, step, **_kwargs: {
                    "step": step,
                    "repo_id": "owner/finish-failure",
                    "revision": f"checkpoint-{step}",
                    "commit_sha": "c" * 40,
                    "hf_commit_sha": "c" * 40,
                    "repo_url": "https://huggingface.co/owner/finish-failure",
                    "revision_url": f"https://huggingface.co/owner/finish-failure/tree/checkpoint-{step}",
                    "commit_url": f"https://huggingface.co/owner/finish-failure/commit/{'c' * 40}",
                    "source_path": str(folder_path),
                },
            ):
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    config = GRPOConfig(
                        name="wandb-finish-failure",
                        steps=0,
                        save_every=1,
                        checkpoint_dir=checkpoint_dir,
                    )
                    with self.assertRaisesRegex(RuntimeError, "W&B failure receipt"):
                        grpo._run_one_seed(
                            config,
                            self._dataset(),
                            ExactMathReward(),
                            runtime["tokenizer"],
                            logger=lambda _message: None,
                        )


class TestTrainingExample(unittest.TestCase):
    def test_defaults(self):
        ex = TrainingExample(prompt="hello")
        self.assertEqual(ex.prompt, "hello")
        self.assertIsNone(ex.target)
        self.assertEqual(ex.metadata, {})

    def test_with_target(self):
        ex = TrainingExample(prompt="q", target={"tool": "calc"})
        self.assertEqual(ex.target, {"tool": "calc"})


class TestInMemoryDataset(unittest.TestCase):
    def test_train_and_test(self):
        train = [TrainingExample(prompt="a"), TrainingExample(prompt="b")]
        test = [TrainingExample(prompt="c")]
        ds = InMemoryDataset(train=train, test=test)
        self.assertEqual(ds.train_examples(), train)
        self.assertEqual(ds.test_examples(), test)

    def test_empty_test(self):
        ds = InMemoryDataset(train=[TrainingExample(prompt="x")])
        self.assertEqual(ds.test_examples(), ())


class TestSyntheticToolUseDataset(unittest.TestCase):
    def test_non_empty(self):
        ds = make_synthetic_tool_use_dataset()
        self.assertGreater(len(ds.train_examples()), 0)

    def test_target_structure(self):
        ds = make_synthetic_tool_use_dataset()
        ex = ds.train_examples()[0]
        self.assertIn("tool", ex.target)
        self.assertIn("arguments", ex.target)

    def test_prompt_format(self):
        ds = make_synthetic_tool_use_dataset()
        ex = ds.train_examples()[0]
        self.assertIn("<|im_start|>", ex.prompt)
        self.assertIn("Available tools:", ex.prompt)


class TestXLAMDataset(unittest.TestCase):
    def test_pinned_revision_is_passed_to_load_dataset(self):
        calls = []
        datasets = types.ModuleType("datasets")

        def fake_load_dataset(name, **kwargs):
            calls.append((name, kwargs))
            return [
                {
                    "tools": "[]",
                    "answers": '[{"name":"search","arguments":{"query":"q"}}]',
                    "query": "q",
                }
            ]

        datasets.load_dataset = fake_load_dataset
        revision = "26d14ebfe18b1f7b524bd39b404b50af5dc97866"
        with patch.dict(sys.modules, {"datasets": datasets}, clear=False):
            dataset = make_xlam_dataset(seed=809, revision=revision)

        self.assertEqual(
            calls,
            [
                (
                    "Salesforce/xlam-function-calling-60k",
                    {"split": "train", "revision": revision},
                )
            ],
        )
        self.assertEqual(dataset.train_examples()[0].target["tool"], "search")

    def test_unpinned_revision_is_omitted_for_generic_runs(self):
        calls = []
        datasets = types.ModuleType("datasets")

        def fake_load_dataset(name, **kwargs):
            calls.append((name, kwargs))
            return []

        datasets.load_dataset = fake_load_dataset
        with patch.dict(sys.modules, {"datasets": datasets}, clear=False):
            make_xlam_dataset()

        self.assertEqual(calls, [("Salesforce/xlam-function-calling-60k", {"split": "train"})])


class TestSyntheticMathDataset(unittest.TestCase):
    def test_non_empty(self):
        ds = make_synthetic_math_dataset()
        self.assertGreater(len(ds.train_examples()), 0)

    def test_target_is_string(self):
        ds = make_synthetic_math_dataset()
        ex = ds.train_examples()[0]
        self.assertIsInstance(ex.target, str)

    def test_prompt_format(self):
        ds = make_synthetic_math_dataset()
        ex = ds.train_examples()[0]
        self.assertIn("<|im_start|>", ex.prompt)
        self.assertIn("\\boxed{}", ex.prompt)


class TestToolCallReward(unittest.TestCase):
    def _ex(self, tool, arguments):
        return TrainingExample(prompt="q", target={"tool": tool, "arguments": arguments})

    def test_perfect_json(self):
        r = ToolCallReward()
        resp = '{"tool": "calculator", "arguments": {"expression": "1+1"}}'
        score = r.score(resp, self._ex("calculator", {"expression": "1+1"}))
        self.assertAlmostEqual(score, 1.0)

    def test_correct_tool_wrong_args(self):
        r = ToolCallReward()
        resp = '{"tool": "calculator", "arguments": {"wrong": "x"}}'
        score = r.score(resp, self._ex("calculator", {"expression": "1+1"}))
        self.assertAlmostEqual(score, 0.7)

    def test_valid_json_wrong_tool(self):
        r = ToolCallReward()
        resp = '{"tool": "other_tool", "arguments": {}}'
        score = r.score(resp, self._ex("calculator", {"expression": "1+1"}))
        self.assertAlmostEqual(score, 0.3)

    def test_no_json(self):
        r = ToolCallReward()
        self.assertAlmostEqual(r.score("no json here", self._ex("calc", {})), 0.0)

    def test_invalid_json(self):
        r = ToolCallReward()
        # "{bad json" has no closing brace, so regex finds no JSON object => 0.0
        self.assertAlmostEqual(r.score("{bad json", self._ex("calc", {})), 0.0)

    def test_malformed_json_with_brace(self):
        r = ToolCallReward()
        # "{bad json}" has braces but json.loads fails => 0.1
        self.assertAlmostEqual(r.score("{bad json}", self._ex("calc", {})), 0.1)


class TestStrictToolCallReward(unittest.TestCase):
    def _ex(self, tool, arguments):
        return TrainingExample(prompt="q", target={"tool": tool, "arguments": arguments})

    def test_exact_call_receives_full_credit(self):
        reward = StrictToolCallReward()
        response = '{"tool":"search","arguments":{"query":"rl environments"}}'
        self.assertEqual(
            reward.score(response, self._ex("search", {"query": "rl environments"})),
            1.0,
        )

    def test_legacy_name_and_parameters_aliases_remain_supported(self):
        reward = StrictToolCallReward()
        example = TrainingExample(
            prompt="q", target={"name": "search", "parameters": {"query": "right"}}
        )
        response = '{"name":"search","parameters":{"query":"right"}}'
        self.assertEqual(reward.score(response, example), 1.0)

    def test_correct_keys_with_wrong_values_cannot_receive_full_credit(self):
        reward = StrictToolCallReward()
        response = '{"tool":"search","arguments":{"query":"wrong"}}'
        score = reward.score(response, self._ex("search", {"query": "right"}))
        self.assertAlmostEqual(score, 0.7)

    def test_extra_argument_is_penalized(self):
        reward = StrictToolCallReward()
        response = '{"tool":"search","arguments":{"query":"right","extra":1}}'
        score = reward.score(response, self._ex("search", {"query": "right"}))
        self.assertLess(score, 1.0)

    def test_leading_prose_is_rejected(self):
        reward = StrictToolCallReward()
        response = '<think>plan</think>\n{"tool":"search","arguments":{"query":"right"}}'
        self.assertEqual(reward.score(response, self._ex("search", {"query": "right"})), 0.0)

    def test_trailing_prose_is_rejected(self):
        reward = StrictToolCallReward()
        response = '{"tool":"search","arguments":{"query":"right"}}\nDone.'
        self.assertEqual(reward.score(response, self._ex("search", {"query": "right"})), 0.0)

    def test_multiple_json_objects_are_rejected(self):
        reward = StrictToolCallReward()
        response = (
            '{"tool":"search","arguments":{"query":"right"}}'
            '{"tool":"search","arguments":{"query":"right"}}'
        )
        self.assertEqual(reward.score(response, self._ex("search", {"query": "right"})), 0.0)

    def test_conflicting_tool_aliases_are_rejected(self):
        reward = StrictToolCallReward()
        response = (
            '{"tool":"search","name":"lookup",'
            '"arguments":{"query":"right"}}'
        )
        self.assertEqual(reward.score(response, self._ex("search", {"query": "right"})), 0.0)

    def test_conflicting_argument_aliases_are_rejected(self):
        reward = StrictToolCallReward()
        response = (
            '{"tool":"search","arguments":{"query":"right"},'
            '"parameters":{"query":"other"}}'
        )
        self.assertEqual(reward.score(response, self._ex("search", {"query": "right"})), 0.0)

    def test_duplicate_keys_are_rejected(self):
        reward = StrictToolCallReward()
        response = '{"tool":"search","tool":"search","arguments":{}}'
        self.assertEqual(reward.score(response, self._ex("search", {})), 0.0)

    def test_nonstandard_json_constants_are_rejected(self):
        reward = StrictToolCallReward()
        response = '{"tool":"search","arguments":{"query":NaN}}'
        self.assertEqual(reward.score(response, self._ex("search", {"query": float("nan")})), 0.0)

    def test_tool_name_and_string_arguments_remain_case_sensitive(self):
        reward = StrictToolCallReward()
        response = '{"tool":"Search","arguments":{"query":"Right"}}'
        score = reward.score(response, self._ex("search", {"query": "right"}))
        self.assertLess(score, 1.0)

    def test_string_argument_whitespace_is_not_normalized(self):
        reward = StrictToolCallReward()
        response = '{"tool":"search","arguments":{"query":"right "}}'
        score = reward.score(response, self._ex("search", {"query": "right"}))
        self.assertAlmostEqual(score, 0.7)

    def test_conflicting_target_aliases_are_rejected(self):
        reward = StrictToolCallReward()
        example = TrainingExample(
            prompt="q",
            target={
                "tool": "search",
                "name": "lookup",
                "arguments": {"query": "right"},
            },
        )
        response = '{"tool":"search","arguments":{"query":"right"}}'
        self.assertEqual(reward.score(response, example), 0.0)

    def test_unknown_top_level_fields_are_rejected(self):
        reward = StrictToolCallReward()
        response = '{"tool":"search","arguments":{},"comment":"ok"}'
        self.assertEqual(reward.score(response, self._ex("search", {})), 0.0)


class TestMathReward(unittest.TestCase):
    def _ex(self, answer):
        return TrainingExample(prompt="q", target=answer)

    def test_boxed_exact(self):
        r = MathReward()
        score = r.score("the answer is \\boxed{42}", self._ex("42"))
        self.assertAlmostEqual(score, 1.0)

    def test_boxed_float_match(self):
        r = MathReward()
        score = r.score("\\boxed{3.14}", self._ex("3.14"))
        self.assertAlmostEqual(score, 1.0)

    def test_boxed_wrong(self):
        r = MathReward()
        score = r.score("\\boxed{99}", self._ex("42"))
        self.assertAlmostEqual(score, 0.3)

    def test_standalone_number_match(self):
        r = MathReward()
        # Standalone number match (not last-number fallback) => 0.5
        score = r.score("I think the answer is 42", self._ex("42"))
        self.assertAlmostEqual(score, 0.5)

    def test_last_number_fallback(self):
        r = MathReward()
        # "42" appears standalone at end; the \b match fires first => 0.5
        score = r.score("the answer is 42", self._ex("42"))
        self.assertAlmostEqual(score, 0.5)

    def test_no_match(self):
        r = MathReward()
        score = r.score("I don't know", self._ex("42"))
        self.assertAlmostEqual(score, 0.0)

    def test_partial_math_chars(self):
        r = MathReward()
        score = r.score("1 + 2 = ?", self._ex("42"))
        self.assertAlmostEqual(score, 0.1)


class TestExactMathReward(unittest.TestCase):
    def test_binary_reward_has_no_partial_credit(self):
        reward = ExactMathReward()
        example = TrainingExample(prompt="q", target="42")

        self.assertEqual(reward.score("\\boxed{42}", example), 1.0)
        self.assertEqual(reward.score("The answer might be \\boxed{41}", example), 0.0)


class TestRunGrpo(unittest.TestCase):
    def test_multiple_seeds_copy_slotted_config_safely(self):
        config = GRPOConfig(name="seed-test", seed=10, num_seeds=3)
        dataset = InMemoryDataset(train=[TrainingExample(prompt="q", target="1")])

        def fake_run(cfg, *_args, **_kwargs):
            return GRPORunResult(seed=cfg.seed)

        with patch("platform_tinker.tinkerrl.grpo._run_one_seed", side_effect=fake_run):
            results = run_grpo(config, dataset, ExactMathReward())

        self.assertEqual([result.seed for result in results], [10, 11, 12])
        self.assertEqual(config.seed, 10)


class TestGRPORunResult(unittest.TestCase):
    def test_defaults(self):
        r = GRPORunResult(seed=0)
        self.assertEqual(r.seed, 0)
        self.assertIsNone(r.run_id)
        self.assertIsNone(r.sampler_path)
        self.assertEqual(r.reward_trace, [])
        self.assertEqual(r.avg_first5, 0.0)
        self.assertEqual(r.avg_last10, 0.0)
        self.assertEqual(r.peak_reward, 0.0)
        self.assertEqual(r.zero_loss_steps, 0)
        self.assertEqual(r.zero_reward_steps, 0)
        self.assertIsNone(r.heldout_reward)


if __name__ == "__main__":
    unittest.main()
