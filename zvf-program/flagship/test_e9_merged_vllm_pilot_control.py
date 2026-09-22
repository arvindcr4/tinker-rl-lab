from __future__ import annotations

import copy
import base64
import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path
from datetime import datetime, timedelta, timezone


sys.path.insert(0, str(Path(__file__).resolve().parent))

import e9_merged_vllm_pilot_control as control
import e9_merged_vllm_arm as arm


def _result(launch: dict) -> dict:
    solution = "import pandas as pd"
    competition_id = launch["competition_id"]
    grade = {
        "competition_id": competition_id,
        "score": 0.1,
        "valid_submission": True,
    }
    receipt = {
        "schema_version": control.ARM_SCHEMA_V2,
        "arm_id": control.ARM_ID,
        "competition_id": competition_id,
        "run_id": f"{competition_id}-abc123",
        "status": "NATIVE_SINGLE_COMPETITION_GRADED",
        "competition_score": 0.1,
        "native_grade": grade,
        "score": None,
        "is_full_suite_score": False,
        "legacy_tinker_coverage_increment": 0,
        "sample_reused": False,
        "combined_gpu_cpu_reservation": launch["combined_gpu_cpu_reservation"],
        "generation": {
            "generation_mode": "fresh_merged_vllm_generation",
            "prompt_template_version": launch.get("prompt_template_version"),
        },
        "checkpoint_verification": {"verified_shard_count": 26},
        "merged_vllm_budget": {"estimated_combined_usd": 0.5},
        "artifacts": {
            "solution_sha256": hashlib.sha256((solution + "\n").encode()).hexdigest(),
            "native_grade_sha256": hashlib.sha256(
                json.dumps(grade, sort_keys=True).encode()
            ).hexdigest(),
            "submission_sha256": None,
        },
    }
    if launch.get("schema_version") == control.LAUNCH_SCHEMA_V2:
        receipt["allocation_gate"] = launch["allocation_gate"]
    receipt["receipt_sha256"] = control._canonical_sha256(receipt)
    return {
        "receipt": receipt,
        "solution_code": solution,
        "native_grade_json": grade,
        "submission_csv": None,
    }


class PilotControlTests(unittest.TestCase):
    def setUp(self) -> None:
        checkpoint = {
            "verified_shard_count": 26,
            "merge_receipt_sha256": "a" * 64,
            "base_commit": arm.EXPECTED_BASE_COMMIT,
            "adapter_commit": arm.EXPECTED_ADAPTER_COMMIT,
            "weight_shard_sha256": {
                f"model-{number:05d}-of-00026.safetensors": "b" * 64 for number in range(1, 27)
            },
        }
        self.launch = {
            "schema_version": control.LAUNCH_SCHEMA_V2,
            "competition_id": control.DEFAULT_COMPETITION_ID,
            "combined_gpu_cpu_reservation": control.reserve_pilot_budget(
                gpu_usd=control.GPU_RESERVATION_USD,
                cpu_usd=control.CPU_RESERVATION_USD,
            ),
            "allocation_gate": arm.build_gpu_allocation_gate(
                wandb_receipt={
                    "mode": "online",
                    "run_id": "preflight-test",
                    "url": "https://wandb.ai/example/project/runs/preflight-test",
                    "initialized_at": "2026-08-29T12:00:00+00:00",
                    "server_confirmed": True,
                    "server_run_path": "example/project/preflight-test",
                },
                checkpoint_verification=checkpoint,
                preflight_completed_at="2026-08-29T12:00:01+00:00",
            ),
        }

    def _v3_launch_and_result(self) -> tuple[dict, dict]:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        now = datetime.now(timezone.utc)
        timestamp = lambda value: value.isoformat(timespec="seconds")
        private_key = Ed25519PrivateKey.generate()
        private_seed = base64.b64encode(
            private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
        ).decode("ascii")
        public_key = base64.b64encode(
            private_key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        ).decode("ascii")
        checkpoint = {
            "verified_shard_count": control.EXPECTED_SHARD_COUNT,
            "merge_receipt_sha256": control.EXPECTED_MERGE_RECEIPT_SHA256,
            "base_commit": control.EXPECTED_BASE_COMMIT,
            "adapter_commit": control.EXPECTED_ADAPTER_COMMIT,
            "verified_weight_bytes": control.EXPECTED_WEIGHT_BYTES,
            "weight_shard_sha256": {
                f"model-{number:05d}-of-00026.safetensors": "b" * 64
                for number in range(1, 27)
            },
        }
        online = {
            "mode": "online",
            "run_id": "v3-preflight",
            "url": "https://wandb.ai/example/project/runs/v3-preflight",
            "initialized_at": timestamp(now - timedelta(minutes=2)),
            "server_confirmed": True,
            "server_run_path": "example/project/v3-preflight",
        }
        gate = arm.build_gpu_allocation_gate(
            wandb_receipt=online,
            checkpoint_verification=checkpoint,
            preflight_completed_at=timestamp(now - timedelta(minutes=1)),
        )
        reservation = control.reserve_pilot_budget(
            gpu_usd=control.GPU_RESERVATION_USD,
            cpu_usd=control.CPU_RESERVATION_USD,
        )
        reservation_id = "a" * 32
        launch_nonce = "b" * 64
        prompt = "import pandas as pd\nprint('v3')"
        launch = {
            "schema_version": control.LAUNCH_SCHEMA_V3,
            "arm_id": control.ARM_ID,
            "competition_id": control.DEFAULT_COMPETITION_ID,
            "spend_reservation_id": reservation_id,
            "launch_nonce": launch_nonce,
            "combined_gpu_cpu_reservation": reservation,
            "allocation_gate": gate,
            "v3_prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_template_version": control.PROMPT_TEMPLATE_VERSION,
            "prompt_builder_sha256": "c" * 64,
            "deployment_revision": "v3-test-deployment",
            "source_bundle_sha256": "d" * 64,
            "allocation_attestation_public_key_base64": public_key,
        }
        attestation = arm.build_allocation_attestation_v2(
            competition_id=launch["competition_id"],
            reservation_id=reservation_id,
            launch_nonce=launch_nonce,
            prompt_sha256=launch["prompt_sha256"],
            prompt_builder_sha256=launch["prompt_builder_sha256"],
            deployment_revision=launch["deployment_revision"],
            source_bundle_sha256=launch["source_bundle_sha256"],
            wandb_receipt=online,
            checkpoint_verification=checkpoint,
            reservation=reservation,
            issued_at_utc=timestamp(now - timedelta(seconds=20)),
            expires_at_utc=timestamp(now + timedelta(minutes=4)),
            private_key_seed_base64=private_seed,
        )
        launch.update(
            {
                "generation_backend": "merged_vllm",
                "allocation_attestation": attestation,
                "allocation_attestation_sha256": attestation["attestation_sha256"],
            }
        )
        solution = "import pandas as pd"
        generation_wandb = {
            "mode": "online",
            "run_id": "v3-generation",
            "url": "https://wandb.ai/example/project/runs/v3-generation",
            "initialized_at": timestamp(now - timedelta(seconds=10)),
            "server_confirmed": True,
        }
        load_wandb = {**generation_wandb, "run_id": "v3-load"}
        response = solution
        generation = {
            "generation_mode": "fresh_merged_vllm_generation",
            "competition_id": launch["competition_id"],
            "prompt_template_version": control.PROMPT_TEMPLATE_VERSION,
            "prompt_sha256": launch["prompt_sha256"],
            "response_text": response,
            "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
            "program": solution,
            "program_sha256": hashlib.sha256((solution + "\n").encode()).hexdigest(),
            "allocation_attestation_sha256": attestation["attestation_sha256"],
            "wandb_receipt": generation_wandb,
            "generation_started_at": timestamp(now - timedelta(seconds=5)),
            "load_receipt": {
                "wandb_receipt": load_wandb,
                "paid_load_started_at": timestamp(now - timedelta(seconds=8)),
                "checkpoint_verification": checkpoint,
            },
        }
        grade = {
            "competition_id": launch["competition_id"],
            "score": 0.1,
            "valid_submission": True,
        }
        receipt = {
            "schema_version": control.ARM_SCHEMA_V3,
            "arm_id": control.ARM_ID,
            "competition_id": launch["competition_id"],
            "run_id": f"{launch['competition_id']}-v3abc",
            "status": "NATIVE_SINGLE_COMPETITION_GRADED",
            "competition_score": 0.1,
            "native_grade": grade,
            "score": None,
            "is_full_suite_score": False,
            "legacy_tinker_coverage_increment": 0,
            "sample_reused": False,
            "combined_gpu_cpu_reservation": reservation,
            "reservation_id": reservation_id,
            "launch_nonce": launch_nonce,
            "allocation_gate": gate,
            "allocation_attestation": attestation,
            "generation": generation,
            "checkpoint_verification": checkpoint,
            "merged_vllm_budget": {
                "mode": "fresh_merged_vllm_generation_and_native_grade",
                "estimated_modal_gpu_usd": 0.3,
                "estimated_modal_cpu_usd": 0.2,
                "estimated_combined_usd": 0.5,
                "reservation": reservation,
            },
            "artifacts": {
                "solution_sha256": hashlib.sha256((solution + "\n").encode()).hexdigest(),
                "native_grade_sha256": hashlib.sha256(
                    json.dumps(grade, sort_keys=True).encode()
                ).hexdigest(),
                "submission_sha256": None,
            },
        }
        receipt["receipt_sha256"] = control._canonical_sha256(receipt)
        return launch, {
            "receipt": receipt,
            "solution_code": solution,
            "native_grade_json": grade,
            "submission_csv": None,
        }

    def test_valid_result_preserves_scientific_boundary(self) -> None:
        receipt = control.validate_returned_result(_result(self.launch), self.launch)
        self.assertIsNone(receipt["score"])
        self.assertEqual(receipt["legacy_tinker_coverage_increment"], 0)

    def test_paid_spawn_is_frozen_until_v3_security_review_passes(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "paid launches are frozen"):
            control.spawn_pilot(competition_id=control.DEFAULT_COMPETITION_ID)

    def test_v3_returned_success_is_authenticated_and_launch_bound(self) -> None:
        launch, result = self._v3_launch_and_result()
        receipt = control.validate_returned_result(result, launch)
        self.assertEqual(receipt["schema_version"], control.ARM_SCHEMA_V3)
        self.assertIsNone(receipt["score"])
        self.assertEqual(receipt["legacy_tinker_coverage_increment"], 0)

        tampered_launch = copy.deepcopy(launch)
        tampered_launch["source_bundle_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "authenticated provenance"):
            control.validate_returned_result(result, tampered_launch)

        tampered_binding = copy.deepcopy(result)
        tampered_binding["receipt"]["reservation_id"] = "f" * 32
        tampered_binding["receipt"].pop("receipt_sha256")
        tampered_binding["receipt"]["receipt_sha256"] = control._canonical_sha256(
            tampered_binding["receipt"]
        )
        with self.assertRaisesRegex(ValueError, "reservation or nonce"):
            control.validate_returned_result(tampered_binding, launch)

        tampered_attestation = copy.deepcopy(result)
        tampered_attestation["receipt"]["allocation_attestation"]["payload"][
            "competition_id"
        ] = "imet-2020-fgvc7"
        tampered_attestation["receipt"]["allocation_attestation"].pop("attestation_sha256")
        tampered_attestation["receipt"]["allocation_attestation"][
            "attestation_sha256"
        ] = arm._canonical_sha256(tampered_attestation["receipt"]["allocation_attestation"])
        tampered_attestation["receipt"].pop("receipt_sha256")
        tampered_attestation["receipt"]["receipt_sha256"] = control._canonical_sha256(
            tampered_attestation["receipt"]
        )
        with self.assertRaisesRegex(ValueError, "authenticated provenance"):
            control.validate_returned_result(tampered_attestation, launch)

    def test_v3_collection_is_idempotent_after_authenticated_success(self) -> None:
        launch, result = self._v3_launch_and_result()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "arm"
            ledger_path = root / "spend.json"
            ledger_path.write_text(
                json.dumps(
                    {
                        "authorized_incremental_spend_usd": 50.0,
                        "total_counted_incremental_spend_usd": 1.0,
                        "remaining_authorized_incremental_spend_usd": 49.0,
                        "within_cap": True,
                    }
                ),
                encoding="utf-8",
            )
            launch.update(
                {
                    "status": "REMOTE_CALL_SPAWNED_UNCOLLECTED",
                    "function_call_id": "fc-v3-success",
                    "output_dir": str(output_dir),
                    "spend_ledger": str(ledger_path),
                }
            )
            control._rehash_launch(launch)
            launch_path = root / "launch.json"
            control._write_json_atomic(launch_path, launch)
            reservations_path = output_dir / "_pending/spend_reservations.json"
            reservations_path.parent.mkdir(parents=True)
            control._write_json_atomic(
                reservations_path,
                {
                    "schema_version": control.RESERVATION_SCHEMA,
                    "reservations": [
                        {
                            "reservation_id": launch["spend_reservation_id"],
                            "reserved_usd": launch["combined_gpu_cpu_reservation"][
                                "projected_combined_usd"
                            ],
                            "reconciled_to_spend_ledger": False,
                        }
                    ],
                },
            )

            class ResultCall:
                def get(self, *, timeout: float) -> dict:
                    self_timeout = timeout
                    return result

            modal = types.ModuleType("modal")
            modal.FunctionCall = types.SimpleNamespace(from_id=lambda _: ResultCall())
            original_legacy = control.LEGACY_OUTPUT_DIR
            control.LEGACY_OUTPUT_DIR = root / "legacy"
            try:
                with mock.patch.dict(sys.modules, {"modal": modal}):
                    destination = control.collect_pilot(launch_path)
                    self.assertEqual(control.collect_pilot(launch_path), destination)
            finally:
                control.LEGACY_OUTPUT_DIR = original_legacy
            collection = json.loads((destination / "collection_receipt.json").read_text())
            self.assertIsNone(collection["score"])
            self.assertEqual(collection["legacy_tinker_coverage_increment"], 0)
            ledger = json.loads(ledger_path.read_text())
            self.assertEqual(len(ledger["modal_e9_merged_vllm_pilots"]), 1)

    def test_spawn_persists_call_id_before_launch_and_recovery_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "arm"
            ledger_path = root / "spend.json"
            ledger_path.write_text("{}", encoding="utf-8")
            calls: list[tuple] = []

            class Remote:
                def remote(self) -> dict:
                    return self_gate

            class Runner:
                def spawn(self, *args: object) -> object:
                    calls.append(args)
                    return types.SimpleNamespace(object_id="fc-recoverable")

            self_gate = self.launch["allocation_gate"]
            modal = types.ModuleType("modal")
            modal.Function = types.SimpleNamespace(
                from_name=lambda _, name: Remote() if name.startswith("preflight") else Runner()
            )
            original_write = control._write_json_atomic
            reservations_path = output_dir / "_pending/spend_reservations.json"

            def crash_before_launch(path: Path, payload: dict) -> None:
                if path.name.endswith("fc-recoverable.json"):
                    raise OSError("simulated crash after call-id persistence")
                original_write(path, payload)

            with mock.patch.dict(os.environ, {
                "E9_ALLOCATION_ATTESTATION_PUBLIC_KEY_BASE64": "test-public-key",
                "E9_V3_PROMPT_BUILDER_SHA256": "c" * 64,
                "E9_V3_DEPLOYMENT_REVISION": "test-revision",
                "E9_V3_SOURCE_BUNDLE_SHA256": "d" * 64,
            }), mock.patch.object(control, "PAID_LAUNCH_ENABLED", True), mock.patch.object(
                control,
                "_validate_campaign_gate",
                return_value=({"gate": "test"}, self.launch["combined_gpu_cpu_reservation"]),
            ), mock.patch.object(control, "_write_json_atomic", side_effect=crash_before_launch), mock.patch.dict(
                sys.modules, {"modal": modal}
            ):
                with self.assertRaisesRegex(OSError, "simulated crash"):
                    control.spawn_pilot(
                        competition_id=control.DEFAULT_COMPETITION_ID,
                        output_dir=output_dir,
                        spend_ledger_path=ledger_path,
                    )
            self.assertEqual(len(calls), 1)
            recovered = control.recover_orphaned_spawns(output_dir=output_dir)
            self.assertEqual(len(recovered), 1)
            self.assertEqual(control.recover_orphaned_spawns(output_dir=output_dir), [])
            reservations = json.loads(reservations_path.read_text())
            self.assertEqual(reservations["reservations"][0]["function_call_id"], "fc-recoverable")
            self.assertEqual(len(calls), 1)

    def test_unknown_spawn_outcome_is_an_orphan_and_never_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "arm"
            ledger_path = root / "spend.json"
            ledger_path.write_text("{}", encoding="utf-8")
            calls: list[tuple] = []

            class Remote:
                def remote(self) -> dict:
                    return self_gate

            class Runner:
                def spawn(self, *args: object) -> object:
                    calls.append(args)
                    return types.SimpleNamespace(object_id="fc-unknown")

            self_gate = self.launch["allocation_gate"]
            modal = types.ModuleType("modal")
            modal.Function = types.SimpleNamespace(
                from_name=lambda _, name: Remote() if name.startswith("preflight") else Runner()
            )
            original_write = control._write_json_atomic
            reservations_path = output_dir / "_pending/spend_reservations.json"

            def crash_before_call_id(path: Path, payload: dict) -> None:
                if path == reservations_path and payload["reservations"][0].get(
                    "function_call_id"
                ) == "fc-unknown":
                    raise OSError("simulated crash before call-id persistence")
                original_write(path, payload)

            with mock.patch.dict(os.environ, {
                "E9_ALLOCATION_ATTESTATION_PUBLIC_KEY_BASE64": "test-public-key",
                "E9_V3_PROMPT_BUILDER_SHA256": "c" * 64,
                "E9_V3_DEPLOYMENT_REVISION": "test-revision",
                "E9_V3_SOURCE_BUNDLE_SHA256": "d" * 64,
            }), mock.patch.object(control, "PAID_LAUNCH_ENABLED", True), mock.patch.object(
                control,
                "_validate_campaign_gate",
                return_value=({"gate": "test"}, self.launch["combined_gpu_cpu_reservation"]),
            ), mock.patch.object(control, "_write_json_atomic", side_effect=crash_before_call_id), mock.patch.dict(
                sys.modules, {"modal": modal}
            ):
                with self.assertRaisesRegex(OSError, "simulated crash"):
                    control.spawn_pilot(
                        competition_id=control.DEFAULT_COMPETITION_ID,
                        output_dir=output_dir,
                        spend_ledger_path=ledger_path,
                    )
            self.assertEqual(control.recover_orphaned_spawns(output_dir=output_dir), [])
            reservation = json.loads(reservations_path.read_text())["reservations"][0]
            self.assertEqual(reservation["status"], control.SPAWN_ORPHAN_STATUS)
            self.assertEqual(len(calls), 1)

    def test_cpu_preparation_failure_result_is_v3_bound_and_unscored(self) -> None:
        reservation_id = "a" * 32
        launch_nonce = "b" * 64
        launch = {
            "schema_version": control.LAUNCH_SCHEMA_V3,
            "competition_id": "iwildcam-2019-fgvc6",
            "spend_reservation_id": reservation_id,
            "launch_nonce": launch_nonce,
            "combined_gpu_cpu_reservation": self.launch["combined_gpu_cpu_reservation"],
        }
        receipt = arm.build_cpu_preparation_failure_receipt_v1(
            competition_id=launch["competition_id"],
            reservation_id=reservation_id,
            launch_nonce=launch_nonce,
            reservation=launch["combined_gpu_cpu_reservation"],
            prepare_elapsed_seconds=10,
            estimated_cpu_usd=0.01,
            error_type="TimeoutError",
            error_message="native preparation exceeded its deadline",
            recorded_at_utc="2026-08-30T04:00:00+00:00",
        )
        validated = control.validate_cpu_preparation_failure_result({"receipt": receipt}, launch)
        self.assertIsNone(validated["score"])
        self.assertFalse(validated["gpu_generation_started"])
        self.assertEqual(validated["legacy_tinker_coverage_increment"], 0)
        sanitized_receipt = arm.build_cpu_preparation_failure_receipt_v1(
            competition_id=launch["competition_id"],
            reservation_id=reservation_id,
            launch_nonce=launch_nonce,
            reservation=launch["combined_gpu_cpu_reservation"],
            prepare_elapsed_seconds=10,
            estimated_cpu_usd=0.01,
            error_type="RuntimeError",
            error_message='api_key=SECRET "access_token": "JSON_SECRET" /tmp',
            recorded_at_utc="2026-08-30T04:00:00+00:00",
        )
        self.assertTrue(sanitized_receipt["failure"]["redaction_applied"])
        control.validate_cpu_preparation_failure_result({"receipt": sanitized_receipt}, launch)

        tampered = copy.deepcopy(receipt)
        tampered["competition_id"] = control.DEFAULT_COMPETITION_ID
        tampered.pop("receipt_sha256")
        tampered["receipt_sha256"] = control._canonical_sha256(tampered)
        with self.assertRaisesRegex(ValueError, "competition_id"):
            control.validate_cpu_preparation_failure_result({"receipt": tampered}, launch)
        with self.assertRaisesRegex(ValueError, "only its receipt"):
            control.validate_cpu_preparation_failure_result(
                {"receipt": receipt, "solution_code": "import os"}, launch
            )
        extra = copy.deepcopy(receipt)
        extra["program"] = "import os"
        extra.pop("receipt_sha256")
        extra["receipt_sha256"] = control._canonical_sha256(extra)
        with self.assertRaisesRegex(ValueError, "receipt fields"):
            control.validate_cpu_preparation_failure_result({"receipt": extra}, launch)
        unsanitized = copy.deepcopy(receipt)
        unsanitized["failure"]["error_message_tail"] = (
            'Authorization: Bearer SECRET_TOKEN "api_key": "JSON_SECRET" /Users/arvind/private /tmp'
        )
        unsanitized["failure"]["redaction_applied"] = False
        unsanitized.pop("receipt_sha256")
        unsanitized["receipt_sha256"] = control._canonical_sha256(unsanitized)
        with self.assertRaisesRegex(ValueError, "not sanitized"):
            control.validate_cpu_preparation_failure_result({"receipt": unsanitized}, launch)
        invalid_metadata = {
            "phase": lambda item: item.__setitem__("phase", "api_key=SECRET"),
            "recorded_at_utc": lambda item: item.__setitem__(
                "recorded_at_utc", "/Users/arvind/private"
            ),
            "prepare_elapsed_seconds": lambda item: item.__setitem__(
                "prepare_elapsed_seconds", "token=SECRET"
            ),
            "claim_boundary": lambda item: item.__setitem__(
                "claim_boundary", "Authorization: Bearer SECRET"
            ),
            "error_type": lambda item: item["failure"].__setitem__("error_type", "api_key=SECRET"),
            "error_message_sha256": lambda item: item["failure"].__setitem__(
                "error_message_sha256", "/tmp/private"
            ),
            "budget_mode": lambda item: item["merged_vllm_budget"].__setitem__(
                "mode", "api_key=SECRET"
            ),
        }
        for label, mutate in invalid_metadata.items():
            with self.subTest(field=label):
                injected = copy.deepcopy(receipt)
                mutate(injected)
                injected.pop("receipt_sha256")
                injected["receipt_sha256"] = control._canonical_sha256(injected)
                with self.assertRaisesRegex(ValueError, "invalid"):
                    control.validate_cpu_preparation_failure_result({"receipt": injected}, launch)
        unsafe_launch = {**launch, "competition_id": "../escaped"}
        unsafe_receipt = {**receipt, "competition_id": "../escaped"}
        with self.assertRaisesRegex(ValueError, "safe basename"):
            control._cpu_failure_attempt_id(unsafe_launch, unsafe_receipt)

    def test_cpu_preparation_failure_collection_is_receipt_only_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "arm"
            reservation_id = "a" * 32
            launch_nonce = "b" * 64
            ledger_path = root / "spend.json"
            ledger_path.write_text(
                json.dumps(
                    {
                        "authorized_incremental_spend_usd": 50.0,
                        "total_counted_incremental_spend_usd": 1.0,
                        "remaining_authorized_incremental_spend_usd": 49.0,
                        "within_cap": True,
                    }
                ),
                encoding="utf-8",
            )
            reservations_path = output_dir / "_pending/spend_reservations.json"
            reservations_path.parent.mkdir(parents=True)
            reservations_path.write_text(
                json.dumps(
                    {
                        "schema_version": control.RESERVATION_SCHEMA,
                        "reservations": [
                            {
                                "reservation_id": reservation_id,
                                "reserved_usd": self.launch["combined_gpu_cpu_reservation"][
                                    "projected_combined_usd"
                                ],
                                "reconciled_to_spend_ledger": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            launch = {
                "schema_version": control.LAUNCH_SCHEMA_V3,
                "status": "REMOTE_CALL_SPAWNED_UNCOLLECTED",
                "function_call_id": "fc-cpu-prepare-failure",
                "competition_id": "iwildcam-2019-fgvc6",
                "spend_reservation_id": reservation_id,
                "launch_nonce": launch_nonce,
                "combined_gpu_cpu_reservation": self.launch["combined_gpu_cpu_reservation"],
                "output_dir": str(output_dir),
                "spend_ledger": str(ledger_path),
            }
            launch["launch_sha256"] = control._canonical_sha256(launch)
            launch_path = root / "launch.json"
            control._write_json_atomic(launch_path, launch)
            receipt = arm.build_cpu_preparation_failure_receipt_v1(
                competition_id=launch["competition_id"],
                reservation_id=reservation_id,
                launch_nonce=launch_nonce,
                reservation=launch["combined_gpu_cpu_reservation"],
                prepare_elapsed_seconds=10,
                estimated_cpu_usd=0.01,
                error_type="TimeoutError",
                error_message="native preparation exceeded its deadline",
                recorded_at_utc="2026-08-30T04:00:00+00:00",
            )

            malicious_destination = (
                output_dir / f"{launch['competition_id']}-cpu-prepare-{launch_nonce[:12]}"
            )
            malicious_receipts = []
            for field, value in (
                ("error_message_tail", "Authorization: Bearer SECRET_TOKEN /tmp"),
                ("phase", "api_key=SECRET"),
                ("recorded_at_utc", "/Users/arvind/private"),
                ("prepare_elapsed_seconds", "token=SECRET"),
                ("claim_boundary", "Authorization: Bearer SECRET"),
                ("error_type", "api_key=SECRET"),
                ("error_message_sha256", "/tmp/private"),
                ("budget_mode", "api_key=SECRET"),
            ):
                malicious_receipt = copy.deepcopy(receipt)
                if field == "error_message_tail":
                    malicious_receipt["failure"][field] = value
                    malicious_receipt["failure"]["redaction_applied"] = False
                elif field in {"error_type", "error_message_sha256"}:
                    malicious_receipt["failure"][field] = value
                elif field == "budget_mode":
                    malicious_receipt["merged_vllm_budget"]["mode"] = value
                else:
                    malicious_receipt[field] = value
                malicious_receipt.pop("receipt_sha256")
                malicious_receipt["receipt_sha256"] = control._canonical_sha256(malicious_receipt)
                malicious_receipts.append((field, malicious_receipt))

            for field, malicious_receipt in malicious_receipts:

                class MaliciousResultCall:
                    def get(self, timeout: float) -> dict[str, object]:
                        self.timeout = timeout
                        return {"receipt": malicious_receipt}

                malicious_modal_module = types.ModuleType("modal")
                malicious_modal_module.FunctionCall = types.SimpleNamespace(
                    from_id=mock.Mock(return_value=MaliciousResultCall())
                )
                with self.subTest(collection_field=field):
                    with mock.patch.dict(sys.modules, {"modal": malicious_modal_module}):
                        with self.assertRaises(ValueError):
                            control.collect_pilot(launch_path)
                    self.assertFalse(malicious_destination.exists())

            class ResultCall:
                def get(self, timeout: float) -> dict[str, object]:
                    self.timeout = timeout
                    return {"receipt": receipt}

            modal_module = types.ModuleType("modal")
            modal_module.FunctionCall = types.SimpleNamespace(
                from_id=mock.Mock(return_value=ResultCall())
            )
            write_json = control._write_json_atomic

            def fail_reservation_write(path: Path, payload: dict) -> None:
                if path == reservations_path:
                    raise OSError("simulated CPU failure crash after ledger write")
                write_json(path, payload)

            with (
                mock.patch.dict(sys.modules, {"modal": modal_module}),
                mock.patch.object(
                    control,
                    "_write_json_atomic",
                    side_effect=fail_reservation_write,
                ),
            ):
                with self.assertRaisesRegex(OSError, "after ledger write"):
                    control.collect_pilot(launch_path)
            after_crash = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.assertEqual(after_crash["total_counted_incremental_spend_usd"], 1.01)
            self.assertFalse(
                json.loads(reservations_path.read_text(encoding="utf-8"))["reservations"][0][
                    "reconciled_to_spend_ledger"
                ]
            )

            with (
                mock.patch.dict(sys.modules, {"modal": modal_module}),
                mock.patch.object(
                    control,
                    "write_legacy_terminal_override",
                    side_effect=AssertionError("CPU failure must not write a terminal override"),
                ),
            ):
                destination = control.collect_pilot(launch_path)

            self.assertEqual({path.name for path in destination.iterdir()}, {"receipt.json"})
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.assertEqual(ledger["total_counted_incremental_spend_usd"], 1.01)
            self.assertEqual(len(ledger["modal_e9_merged_vllm_cpu_preparation_failures"]), 1)
            reservation = json.loads(reservations_path.read_text(encoding="utf-8"))["reservations"][
                0
            ]
            self.assertEqual(
                reservation["status"],
                "CPU_PREPARATION_FAILURE_RECONCILED_TO_SPEND_LEDGER",
            )

            with mock.patch.dict(sys.modules, {"modal": types.ModuleType("modal")}):
                self.assertEqual(control.collect_pilot(launch_path), destination)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.assertEqual(ledger["total_counted_incremental_spend_usd"], 1.01)
            self.assertEqual(len(ledger["modal_e9_merged_vllm_cpu_preparation_failures"]), 1)

            (destination / "solution.py").write_text("raise RuntimeError\n", encoding="utf-8")
            with mock.patch.dict(sys.modules, {"modal": types.ModuleType("modal")}):
                with self.assertRaisesRegex(ValueError, "model evidence"):
                    control.collect_pilot(launch_path)

    def test_second_prospective_competition_binds_to_its_launch(self) -> None:
        launch = {
            **self.launch,
            "competition_id": "imet-2020-fgvc7",
            "prompt_template_version": control.PROMPT_TEMPLATE_VERSION,
        }
        receipt = control.validate_returned_result(_result(launch), launch)
        self.assertEqual(receipt["competition_id"], "imet-2020-fgvc7")

    def test_recursive_tabular_competition_binds_to_its_launch(self) -> None:
        launch = {
            **self.launch,
            "competition_id": "predict-volcanic-eruptions-ingv-oe",
            "prompt_template_version": control.PROMPT_TEMPLATE_VERSION,
        }
        receipt = control.validate_returned_result(_result(launch), launch)
        self.assertEqual(receipt["competition_id"], "predict-volcanic-eruptions-ingv-oe")

    def test_directory_label_competition_binds_to_its_launch(self) -> None:
        launch = {
            **self.launch,
            "competition_id": "alaska2-image-steganalysis",
            "prompt_template_version": control.PROMPT_TEMPLATE_VERSION,
        }
        receipt = control.validate_returned_result(_result(launch), launch)
        self.assertEqual(receipt["competition_id"], "alaska2-image-steganalysis")

    def test_structured_submission_competition_binds_to_its_launch(self) -> None:
        launch = {
            **self.launch,
            "competition_id": "hubmap-kidney-segmentation",
            "prompt_template_version": control.PROMPT_TEMPLATE_VERSION,
        }
        receipt = control.validate_returned_result(_result(launch), launch)
        self.assertEqual(receipt["competition_id"], "hubmap-kidney-segmentation")

    def test_large_structured_submission_competition_binds_to_its_launch(self) -> None:
        launch = {
            **self.launch,
            "competition_id": "vesuvius-challenge-ink-detection",
            "prompt_template_version": control.PROMPT_TEMPLATE_VERSION,
        }
        receipt = control.validate_returned_result(_result(launch), launch)
        self.assertEqual(receipt["competition_id"], "vesuvius-challenge-ink-detection")

    def test_recommendation_competition_binds_to_its_launch(self) -> None:
        launch = {
            **self.launch,
            "competition_id": "h-and-m-personalized-fashion-recommendations",
            "prompt_template_version": control.PROMPT_TEMPLATE_VERSION,
        }
        receipt = control.validate_returned_result(_result(launch), launch)
        self.assertEqual(
            receipt["competition_id"],
            "h-and-m-personalized-fashion-recommendations",
        )

    def test_rejects_self_hash_drift(self) -> None:
        result = _result(self.launch)
        result["receipt"]["competition_score"] = 0.2
        with self.assertRaisesRegex(ValueError, "self-hash"):
            control.validate_returned_result(result, self.launch)

    def test_rejects_legacy_coverage_increment(self) -> None:
        result = _result(self.launch)
        result["receipt"]["legacy_tinker_coverage_increment"] = 1
        result["receipt"].pop("receipt_sha256")
        result["receipt"]["receipt_sha256"] = control._canonical_sha256(result["receipt"])
        with self.assertRaisesRegex(ValueError, "scientific boundary"):
            control.validate_returned_result(result, self.launch)

    def test_rejects_native_grade_payload_drift(self) -> None:
        result = _result(self.launch)
        result["native_grade_json"] = {
            **result["native_grade_json"],
            "score": 0.2,
        }
        with self.assertRaisesRegex(ValueError, "native grade.*receipt payload"):
            control.validate_returned_result(result, self.launch)

    def test_rejects_native_grade_status_and_score_inconsistency(self) -> None:
        result = _result(self.launch)
        grade = result["native_grade_json"]
        grade["valid_submission"] = False
        result["receipt"]["native_grade"] = grade
        result["receipt"]["artifacts"]["native_grade_sha256"] = hashlib.sha256(
            json.dumps(grade, sort_keys=True).encode()
        ).hexdigest()
        result["receipt"].pop("receipt_sha256")
        result["receipt"]["receipt_sha256"] = control._canonical_sha256(result["receipt"])
        with self.assertRaisesRegex(ValueError, "status does not match"):
            control.validate_returned_result(result, self.launch)

    def test_rejects_native_grade_competition_score_inconsistency(self) -> None:
        result = _result(self.launch)
        result["receipt"]["competition_score"] = 0.2
        result["receipt"].pop("receipt_sha256")
        result["receipt"]["receipt_sha256"] = control._canonical_sha256(result["receipt"])
        with self.assertRaisesRegex(ValueError, "competition score does not match"):
            control.validate_returned_result(result, self.launch)

    def test_v2_result_rejects_tampered_or_missing_allocation_gate(self) -> None:
        result = _result(self.launch)
        result["receipt"]["allocation_gate"]["status"] = "tampered"
        result["receipt"].pop("receipt_sha256")
        result["receipt"]["receipt_sha256"] = control._canonical_sha256(result["receipt"])
        with self.assertRaisesRegex(Exception, "self-hash"):
            control.validate_returned_result(result, self.launch)

    def test_v1_launch_and_receipt_remain_supported_without_a_gate(self) -> None:
        launch = {**self.launch, "schema_version": control.LAUNCH_SCHEMA_V1}
        launch.pop("allocation_gate")
        result = _result(launch)
        result["receipt"].pop("allocation_gate", None)
        result["receipt"]["schema_version"] = control.ARM_SCHEMA_V1
        result["receipt"].pop("receipt_sha256")
        result["receipt"]["receipt_sha256"] = control._canonical_sha256(result["receipt"])
        receipt = control.validate_returned_result(result, launch)
        self.assertEqual(receipt["schema_version"], control.ARM_SCHEMA_V1)

    def test_materialization_is_idempotent_and_fail_closed(self) -> None:
        result = _result(self.launch)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = control._materialize(result, root)
            self.assertEqual(control._materialize(result, root), first)
            changed = copy.deepcopy(result)
            changed["solution_code"] = "raise RuntimeError"
            with self.assertRaisesRegex(ValueError, "differs"):
                control._materialize(changed, root)

    def test_spend_reconciliation_is_idempotent(self) -> None:
        result = _result(self.launch)
        receipt = result["receipt"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "arm"
            destination = control._materialize(result, output_dir)
            ledger_path = root / "spend.json"
            ledger_path.write_text(
                json.dumps(
                    {
                        "authorized_incremental_spend_usd": 50.0,
                        "total_counted_incremental_spend_usd": 1.0,
                        "remaining_authorized_incremental_spend_usd": 49.0,
                        "within_cap": True,
                    }
                ),
                encoding="utf-8",
            )
            reservation_id = "reservation-test"
            reservations_path = output_dir / "_pending/spend_reservations.json"
            reservations_path.parent.mkdir(parents=True)
            reservations_path.write_text(
                json.dumps(
                    {
                        "schema_version": control.RESERVATION_SCHEMA,
                        "reservations": [
                            {
                                "reservation_id": reservation_id,
                                "reserved_usd": 1.75,
                                "reconciled_to_spend_ledger": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            launch = {
                **self.launch,
                "output_dir": str(output_dir),
                "spend_ledger": str(ledger_path),
                "spend_reservation_id": reservation_id,
            }
            first = control._reconcile_spend(
                launch=launch, receipt=receipt, destination=destination
            )
            second = control._reconcile_spend(
                launch=launch, receipt=receipt, destination=destination
            )
            self.assertEqual(first["total_counted_incremental_spend_usd"], 1.5)
            self.assertEqual(second["total_counted_incremental_spend_usd"], 1.5)

    def test_spend_reconciliation_recovers_after_ledger_write_crash(self) -> None:
        result = _result(self.launch)
        receipt = result["receipt"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "arm"
            destination = control._materialize(result, output_dir)
            ledger_path = root / "spend.json"
            ledger_path.write_text(
                json.dumps(
                    {
                        "authorized_incremental_spend_usd": 50.0,
                        "total_counted_incremental_spend_usd": 1.0,
                        "remaining_authorized_incremental_spend_usd": 49.0,
                        "within_cap": True,
                    }
                ),
                encoding="utf-8",
            )
            reservation_id = "reservation-crash-test"
            reservations_path = output_dir / "_pending/spend_reservations.json"
            reservations_path.parent.mkdir(parents=True)
            reservations_path.write_text(
                json.dumps(
                    {
                        "schema_version": control.RESERVATION_SCHEMA,
                        "reservations": [
                            {
                                "reservation_id": reservation_id,
                                "reserved_usd": 1.75,
                                "reconciled_to_spend_ledger": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            launch = {
                **self.launch,
                "output_dir": str(output_dir),
                "spend_ledger": str(ledger_path),
                "spend_reservation_id": reservation_id,
            }
            write_json = control._write_json_atomic

            def fail_reservation_write(path: Path, payload: dict) -> None:
                if path == reservations_path:
                    raise OSError("simulated crash after ledger write")
                write_json(path, payload)

            with mock.patch.object(
                control, "_write_json_atomic", side_effect=fail_reservation_write
            ):
                with self.assertRaisesRegex(OSError, "simulated crash"):
                    control._reconcile_spend(
                        launch=launch, receipt=receipt, destination=destination
                    )
            after_crash = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.assertEqual(after_crash["total_counted_incremental_spend_usd"], 1.5)
            recovered = control._reconcile_spend(
                launch=launch, receipt=receipt, destination=destination
            )
            self.assertEqual(recovered["total_counted_incremental_spend_usd"], 1.5)
            self.assertEqual(len(recovered["modal_e9_merged_vllm_pilots"]), 1)

    def _materialized_launch(
        self,
        root: Path,
        *,
        invalid_native_grade: bool = False,
        terminal_status: str | None = None,
    ) -> tuple[Path, Path]:
        result = _result(
            {**self.launch, "prompt_template_version": control.PROMPT_TEMPLATE_VERSION}
        )
        receipt = result["receipt"]
        if invalid_native_grade:
            grade = result["native_grade_json"]
            grade["score"] = None
            grade["valid_submission"] = False
            receipt["status"] = "NATIVE_SINGLE_COMPETITION_INVALID"
            receipt["competition_score"] = None
            receipt["artifacts"]["native_grade_sha256"] = hashlib.sha256(
                json.dumps(grade, sort_keys=True).encode()
            ).hexdigest()
            receipt.pop("receipt_sha256")
            receipt["receipt_sha256"] = control._canonical_sha256(receipt)
        output_dir = root / "arm"
        destination = control._materialize(result, output_dir)
        ledger_path = root / "spend.json"
        ledger_path.write_text(
            json.dumps(
                {
                    "authorized_incremental_spend_usd": 50.0,
                    "total_counted_incremental_spend_usd": 1.0,
                    "remaining_authorized_incremental_spend_usd": 49.0,
                    "within_cap": True,
                }
            ),
            encoding="utf-8",
        )
        reservation_id = "reservation-materialized-test"
        reservations_path = output_dir / "_pending/spend_reservations.json"
        reservations_path.parent.mkdir(parents=True)
        reservations_path.write_text(
            json.dumps(
                {
                    "schema_version": control.RESERVATION_SCHEMA,
                    "reservations": [
                        {
                            "reservation_id": reservation_id,
                            "reserved_usd": 1.75,
                            "reconciled_to_spend_ledger": False,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        launch = {
            **self.launch,
            "schema_version": control.LAUNCH_SCHEMA_V2,
            "status": "REMOTE_CALL_SPAWNED_UNCOLLECTED",
            "function_call_id": "fc-materialized-test",
            "output_dir": str(output_dir),
            "spend_ledger": str(ledger_path),
            "spend_reservation_id": reservation_id,
            "prompt_template_version": control.PROMPT_TEMPLATE_VERSION,
        }
        original_output = control.LEGACY_OUTPUT_DIR
        control.LEGACY_OUTPUT_DIR = root / "legacy"
        try:
            terminal_override = control.write_legacy_terminal_override(destination / "receipt.json")
        finally:
            control.LEGACY_OUTPUT_DIR = original_output
        if terminal_status is not None:
            terminal = json.loads(terminal_override.read_text(encoding="utf-8"))
            terminal["status"] = terminal_status
            terminal.pop("receipt_sha256")
            terminal["receipt_sha256"] = control._canonical_sha256(terminal)
            control._write_json_atomic(terminal_override, terminal)
        ledger = control._reconcile_spend(launch=launch, receipt=receipt, destination=destination)
        collected_at_epoch = 123.0
        collection = {
            "schema_version": control.COLLECTION_SCHEMA,
            "status": "REMOTE_RESULT_MATERIALIZED",
            "function_call_id": launch["function_call_id"],
            "competition_id": receipt["competition_id"],
            "arm_id": control.ARM_ID,
            "output_dir": str(destination),
            "remote_receipt_sha256": control._file_sha256(destination / "receipt.json"),
            "legacy_no_resample_override": str(terminal_override),
            "competition_score": receipt["competition_score"],
            "score": None,
            "is_full_suite_score": False,
            "legacy_tinker_coverage_increment": 0,
            "counted_incremental_spend_usd": receipt["merged_vllm_budget"][
                "estimated_combined_usd"
            ],
            "remaining_authorized_incremental_spend_usd": ledger[
                "remaining_authorized_incremental_spend_usd"
            ],
            "collected_at_epoch": collected_at_epoch,
        }
        control._write_json_atomic(destination / "collection_receipt.json", collection)
        launch.update(
            {
                "status": "REMOTE_RESULT_MATERIALIZED",
                "scientific_status": receipt["status"],
                "competition_score": receipt["competition_score"],
                "materialized_output_dir": str(destination),
                "remote_receipt_sha256": collection["remote_receipt_sha256"],
                "collected_at_epoch": collected_at_epoch,
            }
        )
        launch["launch_sha256"] = control._canonical_sha256(launch)
        launch_path = root / "launch.json"
        control._write_json_atomic(launch_path, launch)
        return launch_path, destination

    def test_already_materialized_collection_revalidates_all_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launch_path, destination = self._materialized_launch(Path(temporary))
            with mock.patch.dict(sys.modules, {"modal": types.ModuleType("modal")}):
                self.assertEqual(control.collect_pilot(launch_path), destination)

    def test_already_materialized_collection_accepts_historical_invalid_override_status(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launch_path, destination = self._materialized_launch(
                Path(temporary),
                invalid_native_grade=True,
                terminal_status="MERGED_VLLM_SEPARATE_ARM_TERMINAL_INVALID",
            )
            with mock.patch.dict(sys.modules, {"modal": types.ModuleType("modal")}):
                self.assertEqual(control.collect_pilot(launch_path), destination)

    def test_already_materialized_collection_rejects_tampered_ledger_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launch_path, _ = self._materialized_launch(Path(temporary))
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            ledger_path = Path(launch["spend_ledger"])
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["modal_e9_merged_vllm_pilots"][0]["estimated_actual_compute_usd"] = 0.6
            control._write_json_atomic(ledger_path, ledger)
            with mock.patch.dict(sys.modules, {"modal": types.ModuleType("modal")}):
                with self.assertRaisesRegex(ValueError, "spend ledger"):
                    control.collect_pilot(launch_path)

    def test_already_materialized_collection_accepts_exact_historical_ledger_entry(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launch_path, destination = self._materialized_launch(Path(temporary))
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            ledger_path = Path(launch["spend_ledger"])
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["modal_e9_merged_vllm_pilots"][0].pop("reconciliation_id")
            control._write_json_atomic(ledger_path, ledger)
            with mock.patch.dict(sys.modules, {"modal": types.ModuleType("modal")}):
                self.assertEqual(control.collect_pilot(launch_path), destination)

    def test_already_materialized_collection_rejects_mismatched_historical_ledger_entry(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launch_path, _ = self._materialized_launch(Path(temporary))
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            ledger_path = Path(launch["spend_ledger"])
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            entry = ledger["modal_e9_merged_vllm_pilots"][0]
            entry.pop("reconciliation_id")
            entry["estimated_actual_compute_usd"] = 0.6
            control._write_json_atomic(ledger_path, ledger)
            with mock.patch.dict(sys.modules, {"modal": types.ModuleType("modal")}):
                with self.assertRaisesRegex(ValueError, "spend ledger"):
                    control.collect_pilot(launch_path)

    def test_already_materialized_collection_rejects_ambiguous_historical_ledger_entries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launch_path, _ = self._materialized_launch(Path(temporary))
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            ledger_path = Path(launch["spend_ledger"])
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            entry = ledger["modal_e9_merged_vllm_pilots"][0]
            entry.pop("reconciliation_id")
            ledger["modal_e9_merged_vllm_pilots"].append(copy.deepcopy(entry))
            control._write_json_atomic(ledger_path, ledger)
            with mock.patch.dict(sys.modules, {"modal": types.ModuleType("modal")}):
                with self.assertRaisesRegex(ValueError, "spend ledger"):
                    control.collect_pilot(launch_path)

    def test_already_materialized_collection_rejects_invalid_spend_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launch_path, destination = self._materialized_launch(Path(temporary))
            collection_path = destination / "collection_receipt.json"
            collection = json.loads(collection_path.read_text(encoding="utf-8"))
            collection["remaining_authorized_incremental_spend_usd"] = -0.01
            control._write_json_atomic(collection_path, collection)
            with mock.patch.dict(sys.modules, {"modal": types.ModuleType("modal")}):
                with self.assertRaisesRegex(ValueError, "spend snapshot"):
                    control.collect_pilot(launch_path)

    def test_already_materialized_collection_rejects_tampered_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            launch_path, destination = self._materialized_launch(Path(temporary))
            collection = json.loads(
                (destination / "collection_receipt.json").read_text(encoding="utf-8")
            )
            terminal_path = Path(collection["legacy_no_resample_override"])
            terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
            terminal["score"] = 0.1
            terminal.pop("receipt_sha256")
            terminal["receipt_sha256"] = control._canonical_sha256(terminal)
            control._write_json_atomic(terminal_path, terminal)
            with mock.patch.dict(sys.modules, {"modal": types.ModuleType("modal")}):
                with self.assertRaisesRegex(ValueError, "terminal override"):
                    control.collect_pilot(launch_path)

    def test_terminal_override_is_self_hashed_and_unscored(self) -> None:
        result = _result(self.launch)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_path = control._materialize(result, root / "arm") / "receipt.json"
            original_output = control.LEGACY_OUTPUT_DIR
            control.LEGACY_OUTPUT_DIR = root / "legacy"
            try:
                terminal_path = control.write_legacy_terminal_override(receipt_path)
                terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
                stored_hash = terminal.pop("receipt_sha256")
                self.assertEqual(stored_hash, control._canonical_sha256(terminal))
                self.assertIsNone(terminal["score"])
                self.assertFalse(terminal["relaunch_allowed"])
            finally:
                control.LEGACY_OUTPUT_DIR = original_output


if __name__ == "__main__":
    unittest.main()
