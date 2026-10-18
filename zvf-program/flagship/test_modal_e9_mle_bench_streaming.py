from __future__ import annotations

import ast
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import e9_merged_vllm_arm as merged_arm
    import e9_mle_bench_streaming as core
    import modal_e9_mle_bench_streaming as runner
except ModuleNotFoundError as exc:
    if exc.name != "modal":
        raise
    core = None
    runner = None


@unittest.skipIf(runner is None, "Modal SDK is installed in its uv tool environment")
class ModalE9RecoverabilityTests(unittest.TestCase):
    @staticmethod
    def _rehash_receipt(receipt: dict[str, object]) -> None:
        receipt.pop("receipt_sha256", None)
        receipt["receipt_sha256"] = hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _valid_contract(self, root: Path) -> tuple[dict[str, object], dict[str, object]]:
        competition_id = "plant-pathology-2021-fgvc8"
        source_solution = "print('source')"
        source_solution_sha256 = hashlib.sha256((source_solution + "\n").encode()).hexdigest()
        source_receipt: dict[str, object] = {
            "competition_id": competition_id,
            "run_id": f"{competition_id}-source",
            "status": "AGENT_EXECUTION_FAILED",
            "hf_commit": core.HF_COMMIT,
            "wandb_url": "https://wandb.ai/example/project/runs/source",
            "artifacts": {"solution_sha256": source_solution_sha256},
        }
        source_receipt["receipt_sha256"] = hashlib.sha256(
            json.dumps(source_receipt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        source_receipt_path = root / "receipt.json"
        source_receipt_path.write_text(
            json.dumps(source_receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        returned_solution = source_solution
        native_grade = {"score": None, "valid_submission": False}
        provenance = core.validate_replay_provenance(
            competition_id=competition_id,
            solution_sha256=source_solution_sha256,
            source_receipt=source_receipt,
        )
        receipt: dict[str, object] = {
            "competition_id": competition_id,
            "run_id": f"{competition_id}-returned",
            "status": "AGENT_EXECUTION_FAILED",
            "score": None,
            "sample_reused": True,
            "bridge_budget": {"charged_this_run_usd": 0.0},
            "replay_provenance": provenance,
            "artifacts": {
                "solution_sha256": hashlib.sha256((returned_solution + "\n").encode()).hexdigest(),
                "native_grade_sha256": hashlib.sha256(
                    json.dumps(native_grade, sort_keys=True).encode()
                ).hexdigest(),
                "submission_sha256": None,
            },
        }
        receipt["execution_binding"] = core.bind_effective_execution(
            competition_id=competition_id,
            original=(source_solution + "\n").encode(),
            effective=(returned_solution + "\n").encode(),
            submission=None,
            native_grade=json.dumps(native_grade, sort_keys=True).encode(),
            grader_input=None,
            repairs=[],
        )
        receipt["receipt_sha256"] = hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        result: dict[str, object] = {
            "receipt": receipt,
            "solution_code": returned_solution,
            "original_solution_code": source_solution,
            "submission_csv": None,
            "native_grade_json": native_grade,
        }
        launch: dict[str, object] = {
            "competition_id": competition_id,
            "source_receipt": str(source_receipt_path),
            "source_receipt_sha256": hashlib.sha256(source_receipt_path.read_bytes()).hexdigest(),
            "source_solution_sha256": source_solution_sha256,
        }
        return result, launch

    def _graded_contract(
        self, root: Path, source_status: str = "AGENT_EXECUTION_FAILED"
    ) -> tuple[dict[str, object], dict[str, object]]:
        result, launch = self._valid_contract(root)
        source_receipt_path = Path(str(launch["source_receipt"]))
        source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
        source_receipt["status"] = source_status
        self._rehash_receipt(source_receipt)
        source_receipt_path.write_text(
            json.dumps(source_receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        launch["source_receipt_sha256"] = hashlib.sha256(
            source_receipt_path.read_bytes()
        ).hexdigest()
        receipt = result["receipt"]
        native_grade = {"score": 0.42, "valid_submission": True}
        receipt.update(
            {
                "status": "NATIVE_SINGLE_COMPETITION_GRADED",
                "competition_score": 0.42,
                "score": None,
                "replay_provenance": core.validate_replay_provenance(
                    competition_id=str(launch["competition_id"]),
                    solution_sha256=str(launch["source_solution_sha256"]),
                    source_receipt=source_receipt,
                ),
            }
        )
        receipt["artifacts"]["native_grade_sha256"] = hashlib.sha256(
            json.dumps(native_grade, sort_keys=True).encode()
        ).hexdigest()
        result["submission_csv"] = "id,prediction\r\na,0.42\r\n"
        receipt["artifacts"]["submission_sha256"] = hashlib.sha256(
            result["submission_csv"].encode()
        ).hexdigest()
        receipt["execution_binding"] = core.bind_effective_execution(
            competition_id=str(launch["competition_id"]),
            original=(result["original_solution_code"] + "\n").encode(),
            effective=(result["solution_code"] + "\n").encode(),
            submission=result["submission_csv"].encode(),
            native_grade=json.dumps(native_grade, sort_keys=True).encode(),
            grader_input={
                "competition_id": str(launch["competition_id"]),
                "submission_sha256": receipt["artifacts"]["submission_sha256"],
                "grader_revision": "synthetic-fixture",
            },
            repairs=[],
        )
        self._rehash_receipt(receipt)
        result["native_grade_json"] = native_grade
        return result, launch

    def test_returned_result_is_bound_to_launch_and_retry_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, launch = self._valid_contract(root)
            runner._validate_returned_replay_result(result, launch)
            first = runner._materialize_run_result(result, str(root / "out"))
            second = runner._materialize_run_result(result, str(root / "out"))
            assert first == second

    def test_rehashed_unregistered_program_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, launch = self._valid_contract(Path(directory))
            result["solution_code"] = "print('different treatment')"
            digest = hashlib.sha256((result["solution_code"] + "\n").encode()).hexdigest()
            result["receipt"]["artifacts"]["solution_sha256"] = digest
            result["receipt"]["execution_binding"]["effective_program_sha256"] = digest
            self._rehash_receipt(result["receipt"])
            with pytest.raises(core.LaunchGateError, match="without a registered repair"):
                runner._validate_returned_replay_result(result, launch)

    def test_returned_result_rejects_competition_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, launch = self._valid_contract(Path(directory))
            launch["competition_id"] = "wrong-competition"
            with pytest.raises(ValueError, match="competition"):
                runner._validate_returned_replay_result(result, launch)

    def test_collector_records_python_exception_but_not_poll_timeout(self) -> None:
        class FailedCall:
            def get(self, timeout: float) -> object:
                raise ValueError(f"remote failed after {timeout}")

        class RunningCall:
            def get(self, timeout: float) -> object:
                raise TimeoutError(f"still running after {timeout}")

        class TerminalTimeoutCall:
            def get(self, timeout: float) -> object:
                raise runner.modal.exception.FunctionTimeoutError(
                    f"remote terminal timeout after {timeout}"
                )

        launch = {
            "schema_version": "e9-modal-spawn-replay-v1",
            "status": "REMOTE_CALL_SPAWNED_UNCOLLECTED",
            "function_call_id": "fc-test-contract",
            "competition_id": "plant-pathology-2021-fgvc8",
            "output_dir": "unused",
        }
        raw_collect = runner.collect_replay.info.raw_f
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "launch.json"
            path.write_text(json.dumps(launch), encoding="utf-8")
            with mock.patch.object(runner.modal.FunctionCall, "from_id", return_value=FailedCall()):
                with pytest.raises(RuntimeError):
                    raw_collect(str(path), 1.0)
            assert json.loads(path.read_text())["status"] == "REMOTE_CALL_FAILED_UNMATERIALIZED"
            path.write_text(json.dumps(launch), encoding="utf-8")
            with mock.patch.object(
                runner.modal.FunctionCall, "from_id", return_value=RunningCall()
            ):
                with pytest.raises(TimeoutError):
                    raw_collect(str(path), 1.0)
            assert json.loads(path.read_text())["status"] == "REMOTE_CALL_SPAWNED_UNCOLLECTED"
            path.write_text(json.dumps(launch), encoding="utf-8")
            with mock.patch.object(
                runner.modal.FunctionCall,
                "from_id",
                return_value=TerminalTimeoutCall(),
            ):
                with pytest.raises(RuntimeError):
                    raw_collect(str(path), 1.0)
            assert json.loads(path.read_text())["status"] == "REMOTE_CALL_FAILED_UNMATERIALIZED"

    def test_collector_revalidates_local_result_without_remote_refetch(self) -> None:
        class ResultCall:
            def __init__(self, result: dict[str, object]) -> None:
                self.result = result

            def get(self, timeout: float) -> dict[str, object]:
                return self.result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, contract = self._valid_contract(root)
            launch = {
                **contract,
                "schema_version": "e9-modal-spawn-replay-v1",
                "status": "REMOTE_CALL_SPAWNED_UNCOLLECTED",
                "function_call_id": "fc-idempotent-contract",
                "output_dir": str(root / "out"),
            }
            path = root / "launch.json"
            path.write_text(json.dumps(launch), encoding="utf-8")
            raw_collect = runner.collect_replay.info.raw_f
            with mock.patch.object(
                runner.modal.FunctionCall,
                "from_id",
                return_value=ResultCall(result),
            ):
                raw_collect(str(path), 1.0)
            assert json.loads(path.read_text())["status"] == "REMOTE_RESULT_MATERIALIZED"
            with mock.patch.object(
                runner.modal.FunctionCall,
                "from_id",
                side_effect=AssertionError("remote result should not be fetched twice"),
            ):
                raw_collect(str(path), 1.0)

    def test_collector_promotes_new_native_grade_control_metadata_once(self) -> None:
        class ResultCall:
            def __init__(self, result: dict[str, object]) -> None:
                self.result = result

            def get(self, timeout: float) -> dict[str, object]:
                return self.result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, contract = self._graded_contract(root)
            launch = {
                **contract,
                "schema_version": "e9-modal-spawn-replay-v1",
                "status": "REMOTE_CALL_SPAWNED_UNCOLLECTED",
                "scientific_status": "NON_SCORE_EXECUTION_CONTROL_EVIDENCE",
                "score": None,
                "coverage_increment": 0,
                "function_call_id": "fc-new-native-grade",
                "output_dir": str(root / "out"),
            }
            path = root / "launch.json"
            path.write_text(json.dumps(launch), encoding="utf-8")
            raw_collect = runner.collect_replay.info.raw_f
            with mock.patch.object(
                runner.modal.FunctionCall,
                "from_id",
                return_value=ResultCall(result),
            ):
                raw_collect(str(path), 1.0)

            materialized = json.loads(path.read_text(encoding="utf-8"))
            collection = json.loads(
                (
                    Path(str(materialized["materialized_output_dir"])) / "collection_receipt.json"
                ).read_text(encoding="utf-8")
            )
            for payload in (materialized, collection):
                assert payload["scientific_status"] == "NATIVE_SINGLE_COMPETITION_GRADED"
                assert payload["competition_score"] == 0.42
                assert payload["score"] is None
                assert payload["coverage_increment"] == 1

            with mock.patch.object(
                runner.modal.FunctionCall,
                "from_id",
                side_effect=AssertionError("remote result should not be fetched twice"),
            ):
                raw_collect(str(path), 1.0)

    def test_collector_does_not_increment_existing_native_grade(self) -> None:
        class ResultCall:
            def __init__(self, result: dict[str, object]) -> None:
                self.result = result

            def get(self, timeout: float) -> dict[str, object]:
                return self.result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, contract = self._graded_contract(
                root, source_status="NATIVE_SINGLE_COMPETITION_GRADED"
            )
            launch = {
                **contract,
                "schema_version": "e9-modal-spawn-replay-v1",
                "status": "REMOTE_CALL_SPAWNED_UNCOLLECTED",
                "function_call_id": "fc-duplicate-native-grade",
                "output_dir": str(root / "out"),
            }
            path = root / "launch.json"
            path.write_text(json.dumps(launch), encoding="utf-8")
            raw_collect = runner.collect_replay.info.raw_f
            with mock.patch.object(
                runner.modal.FunctionCall,
                "from_id",
                return_value=ResultCall(result),
            ):
                raw_collect(str(path), 1.0)

            materialized = json.loads(path.read_text(encoding="utf-8"))
            collection = json.loads(
                (
                    Path(str(materialized["materialized_output_dir"])) / "collection_receipt.json"
                ).read_text(encoding="utf-8")
            )
            assert materialized["coverage_increment"] == 0
            assert collection["coverage_increment"] == 0

    def test_collector_reconciles_stale_materialized_control_metadata_idempotently(self) -> None:
        class ResultCall:
            def __init__(self, result: dict[str, object]) -> None:
                self.result = result

            def get(self, timeout: float) -> dict[str, object]:
                return self.result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, contract = self._graded_contract(root)
            launch = {
                **contract,
                "schema_version": "e9-modal-spawn-replay-v1",
                "status": "REMOTE_CALL_SPAWNED_UNCOLLECTED",
                "function_call_id": "fc-stale-control-metadata",
                "output_dir": str(root / "out"),
            }
            path = root / "launch.json"
            path.write_text(json.dumps(launch), encoding="utf-8")
            raw_collect = runner.collect_replay.info.raw_f
            with mock.patch.object(
                runner.modal.FunctionCall,
                "from_id",
                return_value=ResultCall(result),
            ):
                raw_collect(str(path), 1.0)

            materialized = json.loads(path.read_text(encoding="utf-8"))
            destination = Path(str(materialized["materialized_output_dir"]))
            collection_path = destination / "collection_receipt.json"
            collection = json.loads(collection_path.read_text(encoding="utf-8"))
            stale = {
                "scientific_status": "STALE_CONTROL_VALUE",
                "competition_score": -1.0,
                "score": -1.0,
                "coverage_increment": 0,
            }
            materialized.update(stale)
            collection.update(stale)
            path.write_text(json.dumps(materialized, sort_keys=True), encoding="utf-8")
            collection_path.write_text(json.dumps(collection, sort_keys=True), encoding="utf-8")
            remote_receipt_before = (destination / "receipt.json").read_bytes()
            launch_bindings = {
                field: materialized[field]
                for field in (
                    "status",
                    "function_call_id",
                    "materialized_output_dir",
                    "remote_receipt_sha256",
                    "collected_at_epoch",
                )
            }
            collection_bindings = {
                field: collection[field]
                for field in (
                    "status",
                    "function_call_id",
                    "remote_receipt_sha256",
                    "collected_at_epoch",
                    "output_dir",
                )
            }

            with mock.patch.object(
                runner.modal.FunctionCall,
                "from_id",
                side_effect=AssertionError("materialized result should not be refetched"),
            ):
                raw_collect(str(path), 1.0)
            repaired_launch = json.loads(path.read_text(encoding="utf-8"))
            repaired_collection = json.loads(collection_path.read_text(encoding="utf-8"))
            for payload in (repaired_launch, repaired_collection):
                assert payload["scientific_status"] == "NATIVE_SINGLE_COMPETITION_GRADED"
                assert payload["competition_score"] == 0.42
                assert payload["score"] is None
                assert payload["coverage_increment"] == 1
            assert {field: repaired_launch[field] for field in launch_bindings} == launch_bindings
            assert {
                field: repaired_collection[field] for field in collection_bindings
            } == collection_bindings
            assert (destination / "receipt.json").read_bytes() == remote_receipt_before

            repaired_launch_bytes = path.read_bytes()
            repaired_collection_bytes = collection_path.read_bytes()
            with mock.patch.object(
                runner.modal.FunctionCall,
                "from_id",
                side_effect=AssertionError("materialized result should not be refetched"),
            ):
                raw_collect(str(path), 1.0)
            assert path.read_bytes() == repaired_launch_bytes
            assert collection_path.read_bytes() == repaired_collection_bytes

    def test_spawn_gate_blocks_an_unresolved_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = root / "ledger.json"
            ledger.write_text(
                json.dumps(
                    {
                        "within_cap": True,
                        "remaining_authorized_incremental_spend_usd": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            pending = root / "out/_pending"
            pending.mkdir(parents=True)
            (pending / "active.json").write_text(
                json.dumps({"status": "REMOTE_CALL_SPAWNED_UNCOLLECTED"}),
                encoding="utf-8",
            )
            with pytest.raises(RuntimeError, match="unresolved"):
                runner._validate_spawn_budget_and_queue(
                    output_dir=str(root / "out"), spend_ledger_path=str(ledger)
                )

    def test_v7_prompt_forbids_whole_slide_pixel_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            public = Path(directory)
            (public / "description.md").write_text("structured image task", encoding="utf-8")
            (public / "sample_submission.csv").write_text(
                "id,predicted\nexample,\n", encoding="utf-8"
            )
            prompt = runner._agent_prompt("vesuvius-challenge-ink-detection", public)
            normalized_prompt = " ".join(prompt.split())
            assert "Never train a per-pixel classifier" in normalized_prompt
            assert "must always write the schema-valid CSV" in normalized_prompt
            assert "never iterate in Python over every pixel" in normalized_prompt
            assert runner.MERGED_VLLM_PROMPT_TEMPLATE_VERSION == "e9_hard_bounded_collections_v7"

    def test_cpu_prepare_failure_helper_is_receipt_only_unscored_and_zero_gpu(self) -> None:
        reservation = merged_arm.reserve_pilot_budget(gpu_usd=1.0, cpu_usd=0.25)
        result = runner._cpu_prepare_failure_result(
            competition_id="iwildcam-2019-fgvc6",
            reservation_id="a" * 32,
            launch_nonce="b" * 64,
            reservation=reservation,
            elapsed_seconds=10.0,
            error=TimeoutError("prepare timed out before private data existed"),
            recorded_at_utc="2026-08-30T04:00:00+00:00",
        )

        assert set(result) == {"receipt"}
        receipt = result["receipt"]
        assert receipt["schema_version"] == merged_arm.CPU_PREPARATION_FAILURE_SCHEMA_V1
        assert receipt["status"] == "CPU_PREPARATION_FAILED_BEFORE_GPU_GENERATION"
        assert not receipt["gpu_generation_started"]
        assert receipt["competition_score"] is None
        assert receipt["score"] is None
        assert not receipt["is_full_suite_score"]
        assert receipt["legacy_tinker_coverage_increment"] == 0
        assert receipt["merged_vllm_budget"]["estimated_modal_gpu_usd"] == 0.0
        assert (
            receipt["merged_vllm_budget"]["estimated_modal_cpu_usd"]
            <= reservation["projected_modal_cpu_usd"]
        )
        assert (
            receipt["receipt_sha256"]
            == hashlib.sha256(
                json.dumps(
                    {key: value for key, value in receipt.items() if key != "receipt_sha256"},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        )

    def test_prepare_failure_returns_before_gpu_generator_construction(self) -> None:
        checkpoint = {
            "verified_shard_count": 26,
            "merge_receipt_sha256": "a" * 64,
            "base_commit": merged_arm.EXPECTED_BASE_COMMIT,
            "adapter_commit": merged_arm.EXPECTED_ADAPTER_COMMIT,
            "weight_shard_sha256": {
                f"model-{number:05d}-of-00026.safetensors": "b" * 64 for number in range(1, 27)
            },
        }
        allocation_gate = merged_arm.build_gpu_allocation_gate(
            wandb_receipt={
                "mode": "online",
                "run_id": "cpu-prepare-test",
                "url": "https://wandb.ai/example/project/runs/cpu-prepare-test",
                "initialized_at": "2026-08-30T04:00:00+00:00",
                "server_confirmed": True,
                "server_run_path": "example/project/cpu-prepare-test",
            },
            checkpoint_verification=checkpoint,
            preflight_completed_at="2026-08-30T04:00:01+00:00",
        )
        reservation = merged_arm.reserve_pilot_budget(gpu_usd=1.0, cpu_usd=0.25)
        # Execute the production function body with only its Modal decorator
        # removed, avoiding SDK-version-specific private unwrapping methods.
        node = next(
            node
            for node in ast.parse(Path(runner.__file__).read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "run_one"
        )
        node.decorator_list = []
        namespace = dict(vars(runner))
        exec(compile(ast.Module(body=[node], type_ignores=[]), runner.__file__, "exec"), namespace)
        raw_run_one = namespace["run_one"]
        outcomes = {
            "prepare timeout": TimeoutError("prepare timed out"),
            "missing prepared directories": mock.DEFAULT,
        }
        for label, outcome in outcomes.items():
            with self.subTest(label=label):
                run_mock = mock.Mock()
                if outcome is mock.DEFAULT:
                    run_mock.return_value = mock.Mock(stdout="", stderr="")
                else:
                    run_mock.side_effect = outcome
                with (
                    mock.patch.object(runner, "MERGED_VLLM_GPU_GENERATION_ENABLED", True),
                    mock.patch.object(runner, "_validate_v3_signing_keypair"),
                    mock.patch.object(runner, "_run", run_mock),
                    mock.patch.object(
                        runner,
                        "E9MergedVllmGenerator",
                        side_effect=AssertionError("GPU generator was constructed"),
                    ) as generator,
                    mock.patch.object(
                        runner,
                        "_agent_prompt",
                        side_effect=AssertionError("prompt was built after prepare failed"),
                    ) as prompt,
                ):
                    namespace.update(vars(runner))
                    namespace["_require_enforced_budget"] = lambda: (
                        None
                    )  # local fake execution only
                    result = raw_run_one(
                        competition_id="alaska2-image-steganalysis",
                        generation_backend="merged_vllm",
                        merged_arm_reservation=reservation,
                        merged_vllm_allocation_gate=allocation_gate,
                        merged_vllm_reservation_id="a" * 32,
                        merged_vllm_launch_nonce="b" * 64,
                        merged_vllm_v3_contract={},
                    )
                run_mock.assert_called_once()
                assert "mlebench" in run_mock.call_args.args[0]
                assert "prepare" in run_mock.call_args.args[0]
                generator.assert_not_called()
                prompt.assert_not_called()
                receipt = result["receipt"]
                assert receipt["status"] == "CPU_PREPARATION_FAILED_BEFORE_GPU_GENERATION"
                assert not receipt["gpu_generation_started"]
                assert receipt["merged_vllm_budget"]["estimated_modal_gpu_usd"] == 0.0

        def prepare_success(command: list[str], **_: object) -> mock.Mock:
            data_dir = Path(command[command.index("--data-dir") + 1])
            prepared = data_dir / "alaska2-image-steganalysis" / "prepared"
            (prepared / "public").mkdir(parents=True)
            (prepared / "private").mkdir(parents=True)
            return mock.Mock(stdout="", stderr="")

        frozen_run = mock.Mock(side_effect=prepare_success)
        with (
            mock.patch.object(runner, "_run", frozen_run),
            mock.patch.object(runner, "_agent_prompt", return_value="exact v7 prompt"),
            mock.patch.object(
                runner,
                "E9MergedVllmGenerator",
                side_effect=AssertionError("GPU generator was constructed while frozen"),
            ) as generator,
        ):
            namespace.update(vars(runner))
            namespace["_require_enforced_budget"] = lambda: None
            with pytest.raises(core.LaunchGateError, match="remote boundary"):
                raw_run_one(
                    competition_id="alaska2-image-steganalysis",
                    generation_backend="merged_vllm",
                    merged_arm_reservation=reservation,
                    merged_vllm_allocation_gate=allocation_gate,
                    merged_vllm_reservation_id="a" * 32,
                    merged_vllm_launch_nonce="b" * 64,
                )
        generator.assert_not_called()
        # Production freeze is checked before any expensive preparation work.
        frozen_run.assert_not_called()

    def test_spawn_gate_subtracts_unreconciled_reservations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = root / "ledger.json"
            ledger.write_text(
                json.dumps(
                    {
                        "within_cap": True,
                        "remaining_authorized_incremental_spend_usd": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            pending = root / "out/_pending"
            pending.mkdir(parents=True)
            (pending / "spend_reservations.json").write_text(
                json.dumps(
                    {
                        "schema_version": "e9-modal-spend-reservations-v1",
                        "reservations": [
                            {
                                "reserved_usd": 0.75,
                                "reconciled_to_spend_ledger": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with pytest.raises(RuntimeError, match="cannot cover"):
                runner._validate_spawn_budget_and_queue(
                    output_dir=str(root / "out"), spend_ledger_path=str(ledger)
                )


if __name__ == "__main__":
    unittest.main()
