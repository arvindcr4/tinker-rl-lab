from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from flagship import frontier_swe_eval as harness


class FrontierSWEPreflightTests(unittest.TestCase):
    def test_exact_roster_and_smallest_task_are_frozen(self) -> None:
        self.assertEqual(len(harness.OFFICIAL_TASK_IDS), 17)
        self.assertEqual(harness.SMALLEST_TASK_ID, "revideo-perf-opt")
        self.assertIn("frontier_swe_eval", "frontier_swe_eval")

    def test_task_ids_must_be_disjoint(self) -> None:
        with self.assertRaisesRegex(harness.GateError, "not disjoint"):
            harness.validate_task_disjointness(["revideo-perf-opt"], ["revideo-perf-opt"])

    def test_blocked_receipt_has_precise_license_blocker_and_no_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            receipt = {
                "status": "BLOCKED",
                "score": None,
                "benchmark": {
                    "repository": harness.OFFICIAL_REPO_URL,
                    "revision": harness.PINNED_BENCHMARK_REVISION,
                    "task_ids": [harness.SMALLEST_TASK_ID],
                    "license": {"status": "missing"},
                },
                "blockers": [
                    {
                        "code": "benchmark_license_missing",
                        "required_receipt": "maintainer authorization",
                    }
                ],
            }
            harness.write_receipt(path, receipt)
            loaded = json.loads(path.read_text())
            self.assertEqual(loaded["status"], "BLOCKED")
            self.assertIsNone(loaded["score"])
            self.assertEqual(loaded["blockers"][0]["code"], "benchmark_license_missing")

    def test_model_revision_is_immutable_and_model_is_fixed(self) -> None:
        payload = {"sha": "a" * 40}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                return iter(())

        Response.read = lambda self: json.dumps(payload).encode()

        def opener(_request, timeout=0):
            del timeout
            return Response()

        with patch.object(harness, "MODEL_API_URL", "https://example.invalid/model"):
            self.assertEqual(harness.resolve_model_revision(opener=opener), "a" * 40)


class WandbOrderingTests(unittest.TestCase):
    def test_online_wandb_precedes_tinker_factory(self) -> None:
        events: list[str] = []

        class Run:
            id = "run-1"
            mode = "online"

            def finish(self, *, exit_code: int) -> None:
                events.append(f"finish:{exit_code}")

        class Wandb:
            def init(self, **kwargs):
                self.kwargs = kwargs
                events.append("wandb.init")
                return Run()

        def tinker_factory():
            events.append("tinker")
            return types.SimpleNamespace()

        _run, _client = harness.run_tinker_guarded(
            wandb_module=Wandb(),
            wandb_config={"suite_id": "frontier_swe_eval"},
            tinker_factory=tinker_factory,
        )
        self.assertEqual(events[:2], ["wandb.init", "tinker"])

    def test_offline_wandb_blocks_tinker_factory(self) -> None:
        events: list[str] = []

        class Run:
            id = "run-1"
            mode = "offline"

            def finish(self, *, exit_code: int) -> None:
                events.append(f"finish:{exit_code}")

        class Wandb:
            def init(self, **kwargs):
                events.append("wandb.init")
                return Run()

        with self.assertRaisesRegex(RuntimeError, "not verifiably online"):
            harness.run_tinker_guarded(
                wandb_module=Wandb(),
                wandb_config={},
                tinker_factory=lambda: events.append("tinker"),
            )
        self.assertNotIn("tinker", events)


if __name__ == "__main__":
    unittest.main()
