from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pilot.flops import REQUIRED_TRAINING_PHASES
from pilot.protocol import load_protocol, sha256_file
from pilot.remote_core import expected_runtime_versions
from pilot.verifier import (
    SMOKE_PREFIX,
    VerificationError,
    _verify_files,
    _verify_wandb_run,
    verify_preflight_log,
)


class FakeWandbApi:
    def __init__(self, state: str) -> None:
        self.state = state
        self.requested = None

    def run(self, path: str):
        self.requested = path
        return SimpleNamespace(state=self.state, id="run-id", config={})


class VerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_protocol()

    def test_wandb_verifier_requires_complete_identity_and_finished_state(self) -> None:
        api = FakeWandbApi("finished")
        run = _verify_wandb_run(
            api,
            {"entity": "entity", "project": "project", "run_id": "run-id"},
            label="unit",
        )
        self.assertEqual(run.id, "run-id")
        self.assertEqual(api.requested, "entity/project/run-id")
        with self.assertRaisesRegex(VerificationError, "not finished"):
            _verify_wandb_run(
                FakeWandbApi("running"),
                {"entity": "entity", "project": "project", "run_id": "run-id"},
                label="unit",
            )
        with self.assertRaisesRegex(VerificationError, "incomplete"):
            _verify_wandb_run(api, {"run_id": "run-id"}, label="unit")

    def test_file_verifier_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "artifact"
            path.write_bytes(b"correct")
            digest = sha256_file(path)
            _verify_files(root, {"artifact": digest}, label="unit")
            path.write_bytes(b"wrong")
            with self.assertRaisesRegex(VerificationError, "hash mismatch"):
                _verify_files(root, {"artifact": digest}, label="unit")

    def test_preflight_verifier_accepts_only_complete_exact_receipt(self) -> None:
        smoke = {
            "status": "smoke_pass",
            "runtime_versions": {
                "python": "3.12.1",
                **expected_runtime_versions(self.protocol),
            },
            "accelerator": "NVIDIA A100-SXM4-40GB",
            "group_fingerprint": "a" * 64,
            "charged_generated_tokens": 64,
            "phase_flops": {phase: 1.0 for phase in REQUIRED_TRAINING_PHASES},
            "receipt": {
                "step": 1,
                "condition": "intended_full",
                "selected_loss": 0.0,
                "intended_loss": 0.0,
                "native_loss": 0.1,
                "gradient_cosine": 0.9,
                "gradient_relative_l2": 0.1,
                "intended_gradient_norm": 1.0,
                "native_gradient_norm": 1.0,
                "selected_gradient_norm": 1.0,
                "selected_vs_intended_cosine": 1.0,
                "selected_vs_intended_relative_l2": 0.0,
                "active_rows": 8,
                "active_tokens": 64,
                "optimizer_learning_rate": 1e-5,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "smoke.log"
            acceptance = Path(temporary) / "acceptance.json"
            log.write_text(SMOKE_PREFIX + __import__("json").dumps(smoke) + "\n")
            receipt = verify_preflight_log(
                protocol=self.protocol, log_path=log, output_path=acceptance
            )
            self.assertEqual(receipt["status"], "accepted")
            self.assertTrue(acceptance.is_file())
            smoke["accelerator"] = "NVIDIA L4"
            log.write_text(SMOKE_PREFIX + __import__("json").dumps(smoke) + "\n")
            with self.assertRaisesRegex(VerificationError, "A100"):
                verify_preflight_log(protocol=self.protocol, log_path=log)


if __name__ == "__main__":
    unittest.main()
