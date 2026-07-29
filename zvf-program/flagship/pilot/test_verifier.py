from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pilot.artifacts import with_fingerprint
from pilot.flops import REQUIRED_TRAINING_PHASES
from pilot.protocol import load_protocol, sha256_file
from pilot.remote_core import expected_runtime_versions
from pilot.verifier import (
    SMOKE_PREFIX,
    VerificationError,
    _verify_corpus_checkpoint_remote,
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

    def test_remote_corpus_checkpoint_verifier_hashes_entire_prefix(self) -> None:
        regime = "balanced_equal_length"
        seed = 11
        contract = self.protocol.payload["runtime"]["execution_contract"]
        regime_contract = self.protocol.payload["regimes"][regime]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source_manifest.json"
            corpus_binding = self.protocol.corpus_binding(regime, seed)
            sources = dict(corpus_binding["source_manifest"])
            source_path.write_text(json.dumps(sources, sort_keys=True) + "\n")
            files = {"source_manifest.json": sha256_file(source_path)}
            groups = []
            for index in range(20):
                path = root / "groups" / f"group-{index:03d}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"group-{index}".encode())
                relative = f"groups/group-{index:03d}.pt"
                files[relative] = sha256_file(path)
                groups.append(
                    {
                        "index": index,
                        "source_row_index": index,
                        "fingerprint": f"{index + 1:064x}",
                        "active_rows": 8,
                        "selected_length_cv": 0.0,
                        "charged_generated_tokens": 64,
                        "artifact_path": relative,
                    }
                )
            manifest = with_fingerprint(
                {
                    "schema_version": "flagship-pilot-corpus-checkpoint-v1",
                    "status": "partial",
                    "protocol_sha256": corpus_binding["protocol_sha256"],
                    "regime": regime,
                    "seed": seed,
                    "model": self.protocol.payload["runtime"]["model"],
                    "dataset": regime_contract["dataset"],
                    "dataset_revision": regime_contract["dataset_revision"],
                    "train_order_hash": contract["train_order_hash"][regime][str(seed)],
                    "completed_groups": 20,
                    "groups": groups,
                    "charged_generated_tokens": 1280,
                    "flop_ledger": {
                        "profiled_steps": [1, 20],
                        "profiled_generated_tokens": 64,
                        "profiled_generation_flops": 1.0,
                    },
                    "runtime_versions": {"torch": "2.7.1"},
                    "accelerator": "NVIDIA A100-SXM4-40GB",
                    "source_manifest": sources,
                    "artifact_files": files,
                    "resume_count": 0,
                    "attempts": [
                        {
                            "run_id": "run-id",
                            "run_url": "https://wandb.ai/e/p/runs/run-id",
                            "start_group": 0,
                            "completed_through": 20,
                        }
                    ],
                    "wall_clock_seconds": 1.0,
                }
            )
            manifest_path = root / "corpus_checkpoint_manifest.json"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

            def download(**kwargs: object) -> str:
                filename = str(kwargs["filename"])
                return str(root / filename.removeprefix("resume/"))

            with mock.patch("huggingface_hub.hf_hub_download", side_effect=download):
                verified = _verify_corpus_checkpoint_remote(
                    repo="private/repo",
                    revision="b" * 40,
                    protocol=self.protocol,
                    regime=regime,
                    seed=seed,
                    token="token",
                )
                self.assertEqual(verified["completed_groups"], 20)
                (root / "groups/group-019.pt").write_bytes(b"tampered")
                with self.assertRaisesRegex(VerificationError, "remote hash mismatch"):
                    _verify_corpus_checkpoint_remote(
                        repo="private/repo",
                        revision="b" * 40,
                        protocol=self.protocol,
                        regime=regime,
                        seed=seed,
                        token="token",
                    )

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
                "gradient_relation": "nonzero",
                "gradient_cosine": 0.9,
                "gradient_relative_l2": 0.1,
                "intended_gradient_norm": 1.0,
                "native_gradient_norm": 1.0,
                "selected_gradient_norm": 1.0,
                "selected_vs_intended_relation": "nonzero",
                "selected_vs_intended_cosine": 1.0,
                "selected_vs_intended_relative_l2": 0.0,
                "optimizer_update": "applied",
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
            smoke["accelerator"] = "NVIDIA A100-SXM4-40GB"
            for field, value in (
                ("gradient_cosine", 1.0001),
                ("gradient_cosine", -1.0001),
                ("selected_vs_intended_cosine", 1.0001),
                ("selected_vs_intended_cosine", -1.0001),
            ):
                with self.subTest(field=field, value=value):
                    smoke["receipt"][field] = value
                    log.write_text(SMOKE_PREFIX + __import__("json").dumps(smoke) + "\n")
                    with self.assertRaisesRegex(VerificationError, "outside"):
                        verify_preflight_log(protocol=self.protocol, log_path=log)
                    smoke["receipt"][field] = 0.9 if field == "gradient_cosine" else 1.0


if __name__ == "__main__":
    unittest.main()
