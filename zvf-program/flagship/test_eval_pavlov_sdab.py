from __future__ import annotations

import argparse
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from flagship import eval_pavlov_sdab as harness


class _FakeRuntime:
    def list_tasks(self, *, split: str, limit: int, seed: int) -> list[dict[str, str]]:
        del split, seed
        return [{"task_id": "sdab-task-1"}] * limit

    def evaluate_task(
        self,
        task: dict[str, str],
        sampler: object,
        *,
        max_actions: int,
        seed: int,
    ) -> dict[str, object]:
        del sampler, max_actions, seed
        return {
            "task_id": task["task_id"],
            "score": 1.0,
            "prompt_tokens": 10,
            "sample_tokens": 5,
            "tool_calls": 1,
        }

    def verify_result(self, result: dict[str, object]) -> dict[str, object]:
        del result
        return {"verified": True, "verifier": "fake-native-verifier"}


class _FakeRun:
    id = "wandb-run-1"
    url = "https://wandb.invalid/wandb-run-1"


class PavlovSDABServiceLifetimeTests(unittest.TestCase):
    def test_completed_receipt_retains_tinker_service_run_id(self) -> None:
        service = types.SimpleNamespace(run_id="tinker-run-1")
        bundle = harness.RuntimeBundle(
            manifest={
                "benchmark_revision": "benchmark-revision-1",
                "task_id_sha256": "a" * 64,
                "train_task_id_sha256": "b" * 64,
                "verifier_sha256": "c" * 64,
                "container_digest": "sha256:" + "d" * 64,
                "environment_digest": "sha256:" + "e" * 64,
                "license_id": "provider-license",
                "disjointness_receipt": "provider-disjointness-receipt",
            },
            runtime=_FakeRuntime(),
            entrypoint="fake_runtime:create",
        )
        report = {
            "status": "READY",
            "gates": [
                {
                    "name": "authorized_model_revision",
                    "source_kind": "base_model",
                    "model_id": harness.MODEL_ID,
                    "base_model_revision": "1" * 40,
                    "adapter_revision": None,
                    "evaluated_hf_commit": "1" * 40,
                },
                {
                    "name": "tinker_cost_envelope",
                    "projected_max_cost_usd": 0.01,
                    "requested_cap_usd": 0.50,
                },
                {
                    "name": "exact_native_sdab_runtime",
                    "status": "PASS",
                    "manifest": "runtime.json",
                    "benchmark_revision": "benchmark-revision-1",
                    "task_id_sha256": "a" * 64,
                    "train_task_id_sha256": "b" * 64,
                    "container_digest": "sha256:" + "d" * 64,
                    "environment_digest": "sha256:" + "e" * 64,
                    "verifier_identity": "fake-native-verifier",
                    "verifier_sha256": "c" * 64,
                },
            ],
            "blockers": [],
            "required_external_access": [],
        }
        args = argparse.Namespace(
            runtime_manifest=Path("runtime.json"),
            limit=1,
            seed=809,
            max_actions=8,
            max_prompt_tokens=4096,
            max_response_tokens=256,
            wandb_name="pavlov-sdab-e3-eval",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "receipt.json"
            with (
                patch.dict(
                    sys.modules,
                    {
                        "wandb": types.ModuleType("wandb"),
                        "tinker": types.ModuleType("tinker"),
                    },
                ),
                patch.object(harness, "load_native_runtime", return_value=bundle),
                patch.object(harness, "_start_wandb", return_value=_FakeRun()),
                patch.object(harness, "_finish_wandb"),
                patch.object(harness, "_create_tinker_sampler", return_value=(service, object())),
            ):
                receipt = harness.run_evaluation(report=report, args=args, out=out)

        self.assertEqual(receipt["status"], "COMPLETED")
        self.assertEqual(receipt["tinker"], {"calls": 1, "run_id": "tinker-run-1"})


if __name__ == "__main__":
    unittest.main()
