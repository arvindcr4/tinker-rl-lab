from __future__ import annotations

import os
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from . import run_harbor_via_tinker_bridge as bridge


class HarborBridgeLaunchGateTests(unittest.TestCase):
    def health(self, maximum: str = "1.00") -> dict[str, object]:
        return {
            "status": "READY",
            "model": bridge.MODEL_ALIAS,
            "hf_commit": bridge.HF_COMMIT,
            "evidence_class": "infrastructure_not_model_score",
            "wandb_url": "https://wandb.ai/entity/project/runs/run-id",
            "budget": {
                "maximum_usd": maximum,
                "charged_usd": 0.0,
                "reserved_usd": 0.0,
            },
        }

    def test_health_requires_exact_checkpoint_and_online_wandb(self) -> None:
        bridge.validate_health(self.health())
        bad = self.health()
        bad["hf_commit"] = "drift"
        with self.assertRaisesRegex(bridge.LaunchGateError, "commit drifted"):
            bridge.validate_health(bad)

    def test_execute_total_must_match_remote_cap(self) -> None:
        bridge.validate_execute_budget(self.health(), Decimal("1.00"))
        with self.assertRaisesRegex(bridge.LaunchGateError, "exactly match"):
            bridge.validate_execute_budget(self.health(), Decimal("0.50"))

    def test_zero_budget_cannot_execute(self) -> None:
        with self.assertRaisesRegex(bridge.LaunchGateError, "positive"):
            bridge.validate_execute_budget(self.health("0.00"), Decimal("0.00"))

    def test_dotenv_does_not_override_process_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("A=file\nB='quoted'\n", encoding="utf-8")
            env = {"A": "process"}
            bridge.load_dotenv(path, env)
        self.assertEqual(env, {"A": "process", "B": "quoted"})

    def test_harbor_environment_maps_bridge_key_without_mutating_process(self) -> None:
        lane = bridge.LANES["E4"]
        before = os.environ.get("OPENAI_API_KEY")
        before_gemini = os.environ.get("GEMINI_API_KEY")
        env = bridge.build_harbor_env("secret-value", lane)
        self.assertEqual(env["OPENAI_API_KEY"], "secret-value")
        self.assertEqual(env["OPENAI_BASE_URL"], bridge.BRIDGE_API_BASE)
        self.assertIn(str(lane.checkout), env["PYTHONPATH"].split(os.pathsep))
        self.assertIn(str(bridge.HARBOR_EXT_ROOT), env["PYTHONPATH"].split(os.pathsep))
        if before_gemini is None:
            self.assertEqual(env["GEMINI_API_KEY"], "modal-secret-injected")
        else:
            self.assertEqual(env["GEMINI_API_KEY"], before_gemini)
        self.assertEqual(os.environ.get("OPENAI_API_KEY"), before)
        self.assertEqual(os.environ.get("GEMINI_API_KEY"), before_gemini)

    def test_bridge_credential_is_distinct_from_provider_credential(self) -> None:
        env = {
            "TINKER_BRIDGE_API_KEY": "bridge-only",
            "TINKER_API_KEY": "provider-only",
        }
        self.assertEqual(bridge.resolve_bridge_api_key(env), "bridge-only")

    def test_all_lanes_use_the_current_harbor_runtime(self) -> None:
        self.assertEqual(bridge.LANES["E2"].harbor_command, ("harbor",))
        self.assertEqual(bridge.LANES["E4"].harbor_command, ("harbor",))


if __name__ == "__main__":
    unittest.main()
