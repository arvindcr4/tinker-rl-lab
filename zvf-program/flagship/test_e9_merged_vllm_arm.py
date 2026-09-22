from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    from . import e9_merged_vllm_arm as arm
except ImportError:
    import e9_merged_vllm_arm as arm


class E9MergedVllmArmTests(unittest.TestCase):
    @staticmethod
    def _attestation_keypair() -> tuple[str, str]:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        private_key = Ed25519PrivateKey.generate()
        private_seed = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return (
            base64.b64encode(private_seed).decode("ascii"),
            base64.b64encode(public_key).decode("ascii"),
        )

    @staticmethod
    def _allocation_checkpoint() -> dict[str, object]:
        return {
            "verified_shard_count": 26,
            "merge_receipt_sha256": "a" * 64,
            "base_commit": arm.EXPECTED_BASE_COMMIT,
            "adapter_commit": arm.EXPECTED_ADAPTER_COMMIT,
            "weight_shard_sha256": {
                f"model-{number:05d}-of-00026.safetensors": "b" * 64 for number in range(1, 27)
            },
        }

    def _allocation_gate(self, checkpoint: dict[str, object] | None = None) -> dict[str, object]:
        return arm.build_gpu_allocation_gate(
            wandb_receipt={
                "mode": "online",
                "run_id": "preflight-abc",
                "url": "https://wandb.ai/example/project/runs/preflight-abc",
                "initialized_at": "2026-08-29T12:00:00+00:00",
                "server_confirmed": True,
                "server_run_path": "example/project/preflight-abc",
            },
            checkpoint_verification=checkpoint or self._allocation_checkpoint(),
            preflight_completed_at="2026-08-29T12:00:01+00:00",
        )

    def _allocation_attestation(
        self,
        *,
        private_key: str,
        competition_id: str = "hotel-id-2021-fgvc8",
        reservation_id: str = "a" * 32,
        launch_nonce: str = "b" * 64,
        prompt: str = "exact v7 prompt",
        issued_at: str = "2026-08-30T04:00:00+00:00",
        expires_at: str = "2026-08-30T04:05:00+00:00",
    ) -> dict[str, object]:
        return arm.build_allocation_attestation_v2(
            competition_id=competition_id,
            reservation_id=reservation_id,
            launch_nonce=launch_nonce,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            prompt_builder_sha256="c" * 64,
            deployment_revision="deployment-revision-1",
            source_bundle_sha256="d" * 64,
            wandb_receipt={
                "mode": "online",
                "run_id": "attested-run",
                "url": "https://wandb.ai/example/project/runs/attested-run",
                "initialized_at": "2026-08-30T03:59:59+00:00",
                "server_confirmed": True,
                "server_run_path": "example/project/attested-run",
            },
            checkpoint_verification=self._allocation_checkpoint(),
            reservation=arm.reserve_pilot_budget(gpu_usd="0.5", cpu_usd="0.2"),
            issued_at_utc=issued_at,
            expires_at_utc=expires_at,
            private_key_seed_base64=private_key,
        )

    def _checkpoint(self, root: Path) -> tuple[dict[str, object], Path, int]:
        hashes: dict[str, str] = {}
        for number in range(1, 27):
            name = f"model-{number:05d}-of-00026.safetensors"
            payload = f"shard-{number}".encode()
            (root / name).write_bytes(payload)
            hashes[name] = hashlib.sha256(payload).hexdigest()
        receipt: dict[str, object] = {
            "schema_version": arm.MERGE_SCHEMA,
            "status": "READY",
            "all_adapter_tensors_consumed": True,
            "base_commit": arm.EXPECTED_BASE_COMMIT,
            "adapter_commit": arm.EXPECTED_ADAPTER_COMMIT,
            "merge_method": arm.EXPECTED_MERGE_METHOD,
            "merged_path": str(root),
            "weight_bytes": sum((root / name).stat().st_size for name in hashes),
            "weight_file_count": 26,
            "weight_shard_sha256": hashes,
        }
        receipt["receipt_sha256"] = hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return receipt, root, int(receipt["weight_bytes"])

    @staticmethod
    def _expected_hash(receipt: dict[str, object]) -> str:
        return str(receipt["receipt_sha256"])

    def test_checkpoint_validation_hashes_every_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt, root, total_bytes = self._checkpoint(Path(directory))
            verified = arm.validate_merge_receipt_and_shards(
                receipt,
                merged_root=root,
                expected_total_bytes=total_bytes,
                expected_receipt_sha256=self._expected_hash(receipt),
            )
            self.assertEqual(verified["verified_shard_count"], 26)
            self.assertEqual(verified["verified_weight_bytes"], total_bytes)
            (root / "model-00026-of-00026.safetensors").write_text("drift")
            with self.assertRaisesRegex(arm.LaunchGateError, "shard hash drifted"):
                arm.validate_merge_receipt_and_shards(
                    receipt,
                    merged_root=root,
                    expected_total_bytes=total_bytes,
                    expected_receipt_sha256=self._expected_hash(receipt),
                )

    def test_checkpoint_validation_rejects_exact_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt, root, total_bytes = self._checkpoint(Path(directory))
            receipt["base_commit"] = "wrong"
            receipt["receipt_sha256"] = hashlib.sha256(
                json.dumps(
                    {key: value for key, value in receipt.items() if key != "receipt_sha256"},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            with self.assertRaisesRegex(arm.LaunchGateError, "base_commit drifted"):
                arm.validate_merge_receipt_and_shards(
                    receipt,
                    merged_root=root,
                    expected_total_bytes=total_bytes,
                    expected_receipt_sha256=self._expected_hash(receipt),
                )

    def test_checkpoint_validation_rejects_shard_total_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt, root, total_bytes = self._checkpoint(Path(directory))
            receipt["weight_bytes"] = total_bytes + 1
            receipt["receipt_sha256"] = hashlib.sha256(
                json.dumps(
                    {key: value for key, value in receipt.items() if key != "receipt_sha256"},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            with self.assertRaisesRegex(arm.LaunchGateError, "byte total drifted"):
                arm.validate_merge_receipt_and_shards(
                    receipt,
                    merged_root=root,
                    expected_total_bytes=total_bytes + 1,
                    expected_receipt_sha256=self._expected_hash(receipt),
                )

    def test_reservation_is_combined_and_hard_capped(self) -> None:
        reservation = arm.reserve_pilot_budget(gpu_usd="1.5", cpu_usd="0.5")
        self.assertEqual(reservation["projected_combined_usd"], 2.0)
        with self.assertRaisesRegex(arm.LaunchGateError, "combined"):
            arm.reserve_pilot_budget(gpu_usd="1.9", cpu_usd="0.2")

    def test_allocation_gate_is_self_hashed_and_requires_server_confirmation(self) -> None:
        gate = self._allocation_gate()
        self.assertEqual(
            arm.validate_gpu_allocation_gate(gate)["schema_version"],
            arm.ALLOCATION_GATE_SCHEMA,
        )
        tampered = dict(gate)
        tampered["preflight_completed_at"] = "2026-08-29T12:00:02+00:00"
        with self.assertRaisesRegex(arm.LaunchGateError, "self-hash"):
            arm.validate_gpu_allocation_gate(tampered)
        unconfirmed = self._allocation_gate()
        unconfirmed["wandb_receipt"]["server_confirmed"] = False
        unconfirmed.pop("allocation_gate_sha256")
        unconfirmed["allocation_gate_sha256"] = arm._canonical_sha256(unconfirmed)
        with self.assertRaisesRegex(arm.LaunchGateError, "server-confirmed"):
            arm.validate_gpu_allocation_gate(unconfirmed)
        wrong_schema = self._allocation_gate()
        wrong_schema["schema_version"] = "wrong-schema"
        wrong_schema.pop("allocation_gate_sha256")
        wrong_schema["allocation_gate_sha256"] = arm._canonical_sha256(wrong_schema)
        with self.assertRaisesRegex(arm.LaunchGateError, "schema"):
            arm.validate_gpu_allocation_gate(wrong_schema)

    def test_v3_attestation_is_signed_fresh_and_launch_bound(self) -> None:
        private_key, public_key = self._attestation_keypair()
        prompt = "exact v7 prompt"
        attestation = self._allocation_attestation(
            private_key=private_key,
            prompt=prompt,
        )
        validated = arm.validate_allocation_attestation_v2(
            attestation,
            public_key_base64=public_key,
            expected_competition_id="hotel-id-2021-fgvc8",
            expected_reservation_id="a" * 32,
            expected_launch_nonce="b" * 64,
            expected_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            expected_prompt_builder_sha256="c" * 64,
            expected_deployment_revision="deployment-revision-1",
            expected_source_bundle_sha256="d" * 64,
            now_utc="2026-08-30T04:02:00+00:00",
        )
        self.assertEqual(validated["schema_version"], arm.ALLOCATION_ATTESTATION_SCHEMA_V2)

        tampered = json.loads(json.dumps(attestation))
        tampered["payload"]["competition_id"] = "imet-2020-fgvc7"
        tampered.pop("attestation_sha256")
        tampered["attestation_sha256"] = arm._canonical_sha256(tampered)
        with self.assertRaisesRegex(arm.LaunchGateError, "signature"):
            arm.validate_allocation_attestation_v2(
                tampered,
                public_key_base64=public_key,
                expected_competition_id="imet-2020-fgvc7",
                expected_reservation_id="a" * 32,
                expected_launch_nonce="b" * 64,
                expected_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                expected_prompt_builder_sha256="c" * 64,
                expected_deployment_revision="deployment-revision-1",
                expected_source_bundle_sha256="d" * 64,
                now_utc="2026-08-30T04:02:00+00:00",
            )

        with self.assertRaisesRegex(arm.LaunchGateError, "expired"):
            arm.validate_allocation_attestation_v2(
                attestation,
                public_key_base64=public_key,
                expected_competition_id="hotel-id-2021-fgvc8",
                expected_reservation_id="a" * 32,
                expected_launch_nonce="b" * 64,
                expected_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                expected_prompt_builder_sha256="c" * 64,
                expected_deployment_revision="deployment-revision-1",
                expected_source_bundle_sha256="d" * 64,
                now_utc="2026-08-30T04:05:00+00:00",
            )

        with self.assertRaisesRegex(arm.LaunchGateError, "competition_id"):
            arm.validate_allocation_attestation_v2(
                attestation,
                public_key_base64=public_key,
                expected_competition_id="imet-2020-fgvc7",
                expected_reservation_id="a" * 32,
                expected_launch_nonce="b" * 64,
                expected_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                expected_prompt_builder_sha256="c" * 64,
                expected_deployment_revision="deployment-revision-1",
                expected_source_bundle_sha256="d" * 64,
                now_utc="2026-08-30T04:02:00+00:00",
            )

        # Collection may happen well after expiry: the important fact is that
        # the paid generation itself began while the signed interval was open.
        arm.validate_allocation_attestation_v2(
            attestation,
            public_key_base64=public_key,
            expected_competition_id="hotel-id-2021-fgvc8",
            expected_reservation_id="a" * 32,
            expected_launch_nonce="b" * 64,
            expected_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            expected_prompt_builder_sha256="c" * 64,
            expected_deployment_revision="deployment-revision-1",
            expected_source_bundle_sha256="d" * 64,
            generation_started_at="2026-08-30T04:02:00+00:00",
        )
        with self.assertRaisesRegex(arm.LaunchGateError, "does not cover"):
            arm.validate_allocation_attestation_v2(
                attestation,
                public_key_base64=public_key,
                expected_competition_id="hotel-id-2021-fgvc8",
                expected_reservation_id="a" * 32,
                expected_launch_nonce="b" * 64,
                expected_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                expected_prompt_builder_sha256="c" * 64,
                expected_deployment_revision="deployment-revision-1",
                expected_source_bundle_sha256="d" * 64,
                generation_started_at="2026-08-30T04:05:00+00:00",
            )

    def test_v3_generation_binds_prompt_response_program_and_attestation(self) -> None:
        private_key, _ = self._attestation_keypair()
        prompt = "exact v7 prompt"
        attestation = self._allocation_attestation(
            private_key=private_key,
            prompt=prompt,
        )
        response = "import pandas as pd\nprint('ok')"
        program = response
        generation = {
            "generation_mode": "fresh_merged_vllm_generation",
            "competition_id": "hotel-id-2021-fgvc8",
            "prompt_template_version": arm.PROMPT_TEMPLATE_VERSION_V7,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "response_text": response,
            "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
            "program": program,
            "program_sha256": hashlib.sha256((program + "\n").encode()).hexdigest(),
            "allocation_attestation_sha256": attestation["attestation_sha256"],
        }
        arm.validate_v3_generation_binding(
            generation,
            competition_id="hotel-id-2021-fgvc8",
            prompt=prompt,
            allocation_attestation=attestation,
        )
        for field, replacement, message in (
            ("competition_id", "imet-2020-fgvc7", "competition"),
            ("prompt_template_version", "v6", "v7"),
            ("prompt_sha256", "0" * 64, "prompt hash"),
            ("response_sha256", "0" * 64, "response hash"),
            ("program_sha256", "0" * 64, "program hash"),
            ("allocation_attestation_sha256", "0" * 64, "attestation"),
        ):
            mutated = dict(generation)
            mutated[field] = replacement
            with self.subTest(field=field), self.assertRaisesRegex(arm.LaunchGateError, message):
                arm.validate_v3_generation_binding(
                    mutated,
                    competition_id="hotel-id-2021-fgvc8",
                    prompt=prompt,
                    allocation_attestation=attestation,
                )

        substituted = dict(generation)
        substituted["program"] = "import os\nprint('substituted')"
        substituted["program_sha256"] = hashlib.sha256(
            (substituted["program"] + "\n").encode()
        ).hexdigest()
        with self.assertRaisesRegex(arm.LaunchGateError, "sampled response"):
            arm.validate_v3_generation_binding(
                substituted,
                competition_id="hotel-id-2021-fgvc8",
                prompt=prompt,
                allocation_attestation=attestation,
            )

    def test_cpu_preparation_failure_receipt_is_unscored_and_cpu_only(self) -> None:
        reservation = arm.reserve_pilot_budget(gpu_usd="0.5", cpu_usd="0.2")
        receipt = arm.build_cpu_preparation_failure_receipt_v1(
            competition_id="iwildcam-2019-fgvc6",
            reservation_id="a" * 32,
            launch_nonce="b" * 64,
            reservation=reservation,
            prepare_elapsed_seconds=10,
            estimated_cpu_usd="0.01",
            error_type="TimeoutError",
            error_message="native preparation exceeded its deadline",
            recorded_at_utc="2026-08-30T04:00:00+00:00",
        )
        self.assertEqual(receipt["status"], "CPU_PREPARATION_FAILED_BEFORE_GPU_GENERATION")
        self.assertFalse(receipt["gpu_generation_started"])
        self.assertEqual(receipt["merged_vllm_budget"]["estimated_modal_gpu_usd"], 0.0)
        self.assertIsNone(receipt["score"])
        self.assertEqual(receipt["legacy_tinker_coverage_increment"], 0)
        sensitive = arm.build_cpu_preparation_failure_receipt_v1(
            competition_id="iwildcam-2019-fgvc6",
            reservation_id="a" * 32,
            launch_nonce="b" * 64,
            reservation=reservation,
            prepare_elapsed_seconds=10,
            estimated_cpu_usd="0.01",
            error_type="RuntimeError",
            error_message=(
                "Authorization: Bearer SECRET_TOKEN_123 "
                'api_key=PRIVATE_KEY_456 "access_token": "JSON_SECRET_789" '
                "/Users/arvind/private/data.csv /tmp/private/path /tmp"
            ),
            recorded_at_utc="2026-08-30T04:00:00+00:00",
        )
        failure = sensitive["failure"]
        self.assertTrue(failure["redaction_applied"])
        self.assertNotIn("SECRET_TOKEN_123", failure["error_message_tail"])
        self.assertNotIn("PRIVATE_KEY_456", failure["error_message_tail"])
        self.assertNotIn("JSON_SECRET_789", failure["error_message_tail"])
        self.assertNotIn("/Users/arvind", failure["error_message_tail"])
        self.assertNotIn("/tmp/private", failure["error_message_tail"])
        self.assertNotIn("/tmp", failure["error_message_tail"])
        self.assertIn("[REDACTED]", failure["error_message_tail"])
        self.assertIn("[REDACTED_PATH]", failure["error_message_tail"])
        sanitized_again, changed_again = arm._sanitize_failure_message(
            failure["error_message_tail"]
        )
        self.assertEqual(sanitized_again, failure["error_message_tail"])
        self.assertFalse(changed_again)
        with self.assertRaisesRegex(arm.LaunchGateError, "exceeds"):
            arm.build_cpu_preparation_failure_receipt_v1(
                competition_id="iwildcam-2019-fgvc6",
                reservation_id="a" * 32,
                launch_nonce="b" * 64,
                reservation=reservation,
                prepare_elapsed_seconds=10,
                estimated_cpu_usd="0.21",
                error_type="TimeoutError",
                error_message="native preparation exceeded its deadline",
                recorded_at_utc="2026-08-30T04:00:00+00:00",
            )

    def test_cpu_preflight_is_ordered_before_parameterized_gpu_instantiation(self) -> None:
        flagship = Path(__file__).parent
        modal_source = (flagship / "modal_e9_mle_bench_streaming.py").read_text(encoding="utf-8")
        control_source = (flagship / "e9_merged_vllm_pilot_control.py").read_text(encoding="utf-8")
        self.assertLess(
            modal_source.index("def preflight_merged_vllm_allocation"),
            modal_source.index("class E9MergedVllmGenerator"),
        )
        self.assertIn("allocation_gate_json: str = modal.parameter()", modal_source)
        self.assertLess(
            control_source.index('"preflight_merged_vllm_allocation"'),
            control_source.index('"run_one"'),
        )
        self.assertIn("merged_vllm_allocation_gate", modal_source)
        self.assertIn("checkpoint evidence changed after allocation", modal_source)

    def test_generation_extracts_fresh_python_only_after_wandb_gate(self) -> None:
        seen: list[str] = []
        generator = arm.E9MergedVllmProgramGenerator(
            lambda prompt: seen.append(prompt) or "```python\nimport os\nprint('ok')\n```",
            wandb_receipt={
                "mode": "online",
                "run_id": "abc",
                "url": "https://wandb.ai/example/project/runs/abc",
                "initialized_at": "2026-08-29T12:00:00+00:00",
            },
            gpu_load_started_at="2026-08-29T12:00:01+00:00",
        )
        generated = generator.generate_program(
            "E9 competition prompt", generation_started_at="2026-08-29T12:00:02+00:00"
        )
        self.assertEqual(seen, ["E9 competition prompt"])
        self.assertEqual(generated["program"], "import os\nprint('ok')")
        with self.assertRaises(arm.LaunchGateError):
            arm.E9MergedVllmProgramGenerator(
                lambda _: "import os",
                wandb_receipt={
                    "mode": "offline",
                    "run_id": "abc",
                    "url": "https://wandb.ai/example/project/runs/abc",
                    "initialized_at": "2026-08-29T12:00:03+00:00",
                },
                gpu_load_started_at="2026-08-29T12:00:02+00:00",
            )
        with self.assertRaisesRegex(arm.LaunchGateError, "strictly precede"):
            arm.E9MergedVllmProgramGenerator(
                lambda _: "import os",
                wandb_receipt={
                    "mode": "online",
                    "run_id": "abc",
                    "url": "https://wandb.ai/example/project/runs/abc",
                    "initialized_at": "2026-08-29T12:00:03+00:00",
                },
                gpu_load_started_at="2026-08-29T12:00:02+00:00",
            )
        with self.assertRaisesRegex(arm.LaunchGateError, "strictly precede"):
            arm.E9MergedVllmProgramGenerator(
                lambda _: "import os",
                wandb_receipt={
                    "mode": "online",
                    "run_id": "abc",
                    "url": "https://wandb.ai/example/project/runs/abc",
                    "initialized_at": "2026-08-29T12:00:02+00:00",
                },
                gpu_load_started_at="2026-08-29T12:00:02+00:00",
            )

    def test_receipt_is_separate_arm_and_never_scores_or_unions_legacy(self) -> None:
        checkpoint = self._allocation_checkpoint()
        reservation = arm.reserve_pilot_budget(gpu_usd="0.5", cpu_usd="0.2")
        generation = {
            "generation_mode": "fresh_merged_vllm_generation",
            "wandb_receipt": {
                "mode": "online",
                "run_id": "abc",
                "url": "https://wandb.ai/example/project/runs/abc",
                "initialized_at": "2026-08-29T12:00:00+00:00",
            },
            "generation_started_at": "2026-08-29T12:00:02+00:00",
            "program_sha256": "b" * 64,
        }
        receipt = arm.build_task_receipt(
            competition_id="hotel-id-2021-fgvc8",
            native_grade={
                "competition_id": "hotel-id-2021-fgvc8",
                "score": 0.25,
                "valid_submission": True,
            },
            generation=generation,
            checkpoint_verification=checkpoint,
            reservation=reservation,
            allocation_gate=self._allocation_gate(checkpoint),
        )
        self.assertEqual(receipt["schema_version"], arm.ARM_SCHEMA_V2)
        self.assertEqual(receipt["arm_id"], arm.ARM_ID)
        self.assertIsNone(receipt["score"])
        self.assertFalse(receipt["is_full_suite_score"])
        self.assertEqual(receipt["legacy_tinker_coverage_increment"], 0)
        unhashed = dict(receipt)
        stored = unhashed.pop("receipt_sha256")
        self.assertEqual(
            stored,
            hashlib.sha256(
                json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        )

    def test_receipt_requires_explicitly_valid_native_submission(self) -> None:
        reservation = arm.reserve_pilot_budget(gpu_usd="0.5", cpu_usd="0.2")
        generation = {
            "generation_mode": "fresh_merged_vllm_generation",
            "wandb_receipt": {
                "mode": "online",
                "run_id": "abc",
                "url": "https://wandb.ai/example/project/runs/abc",
                "initialized_at": "2026-08-29T12:00:00+00:00",
            },
            "generation_started_at": "2026-08-29T12:00:02+00:00",
        }
        checkpoint = self._allocation_checkpoint()
        receipt = arm.build_task_receipt(
            competition_id="hotel-id-2021-fgvc8",
            native_grade={"competition_id": "hotel-id-2021-fgvc8", "score": 0.25},
            generation=generation,
            checkpoint_verification=checkpoint,
            reservation=reservation,
            allocation_gate=self._allocation_gate(checkpoint),
        )
        self.assertEqual(receipt["status"], "NATIVE_SINGLE_COMPETITION_INVALID")
        self.assertIsNone(receipt["competition_score"])

    def test_receipt_rejects_native_grade_for_another_competition(self) -> None:
        reservation = arm.reserve_pilot_budget(gpu_usd="0.5", cpu_usd="0.2")
        generation = {
            "generation_mode": "fresh_merged_vllm_generation",
            "wandb_receipt": {
                "mode": "online",
                "run_id": "abc",
                "url": "https://wandb.ai/example/project/runs/abc",
                "initialized_at": "2026-08-29T12:00:00+00:00",
            },
            "generation_started_at": "2026-08-29T12:00:02+00:00",
        }
        checkpoint = self._allocation_checkpoint()
        with self.assertRaisesRegex(arm.LaunchGateError, "competition_id"):
            arm.build_task_receipt(
                competition_id="hotel-id-2021-fgvc8",
                native_grade={
                    "competition_id": "wrong-competition",
                    "score": 0.25,
                    "valid_submission": True,
                },
                generation=generation,
                checkpoint_verification=checkpoint,
                reservation=reservation,
                allocation_gate=self._allocation_gate(checkpoint),
            )


if __name__ == "__main__":
    unittest.main()
