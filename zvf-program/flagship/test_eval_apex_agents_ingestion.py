#!/usr/bin/env python3
"""Unit tests for the APEX-Agents dataset ingestion path in eval_apex_agents.py.

EVERY FIXTURE IN THIS FILE IS SYNTHETIC.  None of it is `mercor/apex-agents`
content: the dataset is gated and was not downloaded.  The fixtures are shaped
to match the field contract that Archipelago's own loader
(`examples/hugging_face_task/main.py` @ 1c3dcd4694b313020cd626699c9c7cc1c0a2fc58)
imposes, so that when the real dataset is unblocked the ingestion path is
already exercised.  Nothing here produces or implies a benchmark score.
"""

from __future__ import annotations

import argparse
import json
import os
import unittest
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest

try:  # `python -m unittest flagship.test_...` from zvf-program/
    from flagship import eval_apex_agents as runner
except ImportError:  # `python -m unittest test_...` from flagship/
    import eval_apex_agents as runner  # type: ignore[no-redef]


SYNTHETIC_MARKER = "SYNTHETIC FIXTURE - NOT mercor/apex-agents CONTENT"


def synthetic_task(
    suffix: str,
    *,
    world_id: str = "world_synthetic_001",
    rubric_size: int = 2,
    with_input_files: bool = False,
) -> dict:
    """Build one synthetic task record with a schema-valid task_id."""
    task_id = f"task_{suffix * 32}"[:37]
    return {
        "task_id": task_id,
        "world_id": world_id,
        "task_name": f"{SYNTHETIC_MARKER} task {suffix}",
        "domain": "synthetic_domain",
        "prompt": f"{SYNTHETIC_MARKER}: do the synthetic thing {suffix}.",
        "task_input_files": ["input.xlsx"] if with_input_files else [],
        "rubric": [
            {
                "verifier_id": f"ver_{suffix}{index}",
                "criteria": f"synthetic criterion {index}",
            }
            for index in range(rubric_size)
        ],
    }


def synthetic_world(world_id: str = "world_synthetic_001") -> dict:
    return {
        "world_id": world_id,
        "world_name": f"{SYNTHETIC_MARKER} world",
        "world_description": "synthetic world used only for schema tests",
    }


class TaskRecordSchemaTests(unittest.TestCase):
    def test_valid_records_produce_no_errors(self):
        tasks = [synthetic_task("a"), synthetic_task("b")]
        assert runner.validate_task_records(tasks) == []

    def test_non_array_is_rejected(self):
        errors = runner.validate_task_records({"task_id": "task_" + "a" * 32})
        assert len(errors) == 1
        assert "JSON array" in errors[0]

    def test_empty_array_is_rejected(self):
        assert "zero task records" in runner.validate_task_records([])[0]

    def test_each_required_field_is_enforced(self):
        for field in runner.APEX_TASK_REQUIRED_FIELDS:
            with self.subTest(field=field):
                task = synthetic_task("a")
                del task[field]
                errors = runner.validate_task_records([task])
                assert any(f"tasks[0].{field}" in error for error in errors), (
                    f"missing {field} was not reported: {errors}"
                )

    def test_blank_required_field_is_rejected(self):
        task = synthetic_task("a")
        task["prompt"] = "   "
        errors = runner.validate_task_records([task])
        assert any("tasks[0].prompt" in error for error in errors)

    def test_malformed_task_id_is_rejected(self):
        task = synthetic_task("a")
        task["task_id"] = "not-a-task-id"
        errors = runner.validate_task_records([task])
        assert any("does not match" in error for error in errors)

    def test_uppercase_hex_task_id_is_rejected(self):
        task = synthetic_task("a")
        task["task_id"] = "task_" + "A" * 32
        errors = runner.validate_task_records([task])
        assert any("does not match" in error for error in errors)

    def test_duplicate_task_ids_are_reported(self):
        task = synthetic_task("a")
        errors = runner.validate_task_records([task, dict(task)])
        assert any("duplicated" in error for error in errors)

    def test_missing_rubric_is_rejected(self):
        task = synthetic_task("a")
        del task["rubric"]
        errors = runner.validate_task_records([task])
        assert any("rubric is missing" in error for error in errors)

    def test_empty_rubric_is_rejected(self):
        task = synthetic_task("a", rubric_size=0)
        errors = runner.validate_task_records([task])
        assert any("non-empty JSON array" in error for error in errors)

    def test_rubric_criterion_fields_are_enforced(self):
        for field in runner.APEX_RUBRIC_REQUIRED_FIELDS:
            with self.subTest(field=field):
                task = synthetic_task("a", rubric_size=1)
                del task["rubric"][0][field]
                errors = runner.validate_task_records([task])
                assert any(f"rubric[0].{field}" in error for error in errors), (
                    f"missing rubric {field} was not reported: {errors}"
                )

    def test_duplicate_verifier_ids_are_reported(self):
        task = synthetic_task("a", rubric_size=1)
        task["rubric"].append(dict(task["rubric"][0]))
        errors = runner.validate_task_records([task])
        assert any("verifier_id" in e and "duplicated" in e for e in errors)


class WorldRecordSchemaTests(unittest.TestCase):
    def test_valid_world_records(self):
        assert runner.validate_world_records([synthetic_world()]) == []

    def test_missing_world_name_is_rejected(self):
        world = synthetic_world()
        del world["world_name"]
        errors = runner.validate_world_records([world])
        assert any("worlds[0].world_name" in error for error in errors)

    def test_duplicate_world_ids_are_reported(self):
        world = synthetic_world()
        errors = runner.validate_world_records([world, dict(world)])
        assert any("duplicated" in error for error in errors)


class ReferentialIntegrityTests(unittest.TestCase):
    def test_resolvable_world_reference(self):
        tasks = [synthetic_task("a")]
        assert runner.validate_dataset_references(tasks, [synthetic_world()]) == []

    def test_dangling_world_reference_is_reported(self):
        tasks = [synthetic_task("a", world_id="world_missing")]
        errors = runner.validate_dataset_references(tasks, [synthetic_world()])
        assert any("has no entry" in error for error in errors)


class IngestionReportTests(unittest.TestCase):
    def test_valid_dataset_reports_valid_with_stable_hashes(self):
        tasks = [synthetic_task("a"), synthetic_task("b")]
        worlds = [synthetic_world()]
        report = runner.dataset_ingestion_report(tasks, worlds)
        assert report["valid"]
        assert report["errors"] == []
        assert report["task_count"] == 2
        assert report["world_count"] == 1
        assert report["dataset_revision"] == runner.DATASET_REVISION
        # Hash is order-independent: task order in the file must not change it.
        reordered = runner.dataset_ingestion_report(list(reversed(tasks)), worlds)
        assert report["task_id_sha256"] == reordered["task_id_sha256"]

    def test_task_id_hash_changes_when_the_set_changes(self):
        base = runner.dataset_ingestion_report([synthetic_task("a")], [synthetic_world()])
        grown = runner.dataset_ingestion_report(
            [synthetic_task("a"), synthetic_task("b")], [synthetic_world()]
        )
        assert base["task_id_sha256"] != grown["task_id_sha256"]

    def test_count_mismatch_warns_but_does_not_invalidate(self):
        report = runner.dataset_ingestion_report([synthetic_task("a")], [synthetic_world()])
        assert report["valid"]
        assert report["count_warnings"]
        assert "480" in " ".join(report["count_warnings"])

    def test_schema_error_marks_report_invalid(self):
        task = synthetic_task("a")
        del task["prompt"]
        report = runner.dataset_ingestion_report([task], [synthetic_world()])
        assert not report["valid"]
        assert report["errors"]


class RequiredAssetTests(unittest.TestCase):
    def test_task_without_input_files_needs_three_paths(self):
        paths = runner.required_task_assets(synthetic_task("a"))
        assert paths == [
            "tasks_and_rubrics.json",
            "world_descriptions.json",
            "world_files_zipped/world_synthetic_001.zip",
        ]

    def test_task_with_input_files_adds_the_task_files_prefix(self):
        task = synthetic_task("a", with_input_files=True)
        paths = runner.required_task_assets(task)
        assert f"task_files/{task['task_id']}" in paths

    def test_official_runtime_downloads_are_pinned(self):
        source = (
            'HF_DATASET = "mercor/apex-agents"\n'
            "    zip_path = hf_hub_download(\n"
            '        HF_DATASET, f"world_files_zipped/{world_id}.zip", repo_type="dataset"\n'
            "    )\n"
            "        snapshot_dir = snapshot_download(\n"
            '            HF_DATASET, repo_type="dataset", allow_patterns=[f"{task_prefix}/**"]\n'
            "        )\n"
        )
        adapted = runner._pin_official_hf_downloads(source)
        assert "APEX_HF_DATASET_REVISION" in adapted
        assert adapted.count("revision=HF_DATASET_REVISION") == 2

    def test_official_runtime_download_pin_fails_closed_on_source_drift(self):
        with pytest.raises(RuntimeError, match="seam drifted"):
            runner._pin_official_hf_downloads('HF_DATASET = "mercor/apex-agents"\n')


class DatasetSchemaGateTests(unittest.TestCase):
    def _write(self, directory: Path, tasks: list, worlds: list) -> dict:
        tasks_path = directory / "tasks_and_rubrics.json"
        worlds_path = directory / "world_descriptions.json"
        tasks_path.write_text(json.dumps(tasks), encoding="utf-8")
        worlds_path.write_text(json.dumps(worlds), encoding="utf-8")
        return {"tasks": tasks_path, "worlds": worlds_path}

    def test_gate_blocks_when_nothing_was_downloaded(self):
        gate = runner._dataset_schema_gate(None)
        assert gate.status == "BLOCKED"
        assert not gate.details["validated"]
        assert gate.required_receipt is not None

    def test_gate_passes_on_a_schema_valid_synthetic_dataset(self):
        with TemporaryDirectory() as tmp:
            downloaded = self._write(Path(tmp), [synthetic_task("a")], [synthetic_world()])
            gate = runner._dataset_schema_gate(downloaded)
        assert gate.status == "PASS"
        assert gate.details["valid"]

    def test_gate_blocks_on_a_dangling_world_reference(self):
        with TemporaryDirectory() as tmp:
            downloaded = self._write(
                Path(tmp),
                [synthetic_task("a", world_id="world_missing")],
                [synthetic_world()],
            )
            gate = runner._dataset_schema_gate(downloaded)
        assert gate.status == "BLOCKED"
        assert gate.required_receipt["kind"] == "dataset_schema_mismatch"

    def test_gate_is_required_for_launch(self):
        assert not runner._all_launch_gates_pass([runner._dataset_schema_gate(None)])

    def test_runtime_wandb_handshake_is_the_only_deferred_launch_gate(self):
        required = (
            "benchmark_metadata",
            "benchmark_access",
            "dataset_schema",
            "task_split",
            "native_verifier",
            "isolated_runtime",
            "model_identity",
            "budget",
            "tinker_access",
            "native_grader_credentials",
        )
        gates = [runner.Gate(name, "PASS", {}) for name in required]
        gates.append(
            runner.Gate(
                "wandb_online_before_tinker",
                "PENDING_RUNTIME_ONLINE_HANDSHAKE",
                {},
            )
        )
        assert runner._all_launch_gates_pass(gates)
        gates[-1] = runner.Gate("wandb_online_before_tinker", "BLOCKED", {})
        assert not runner._all_launch_gates_pass(gates)


class JudgeCredentialGateTests(unittest.TestCase):
    def test_judge_key_must_match_selected_provider(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "present"}, clear=True):
            assert runner._judge_key_gate("openai/gpt-4.1-mini").status == "PASS"
            assert runner._judge_key_gate("vertex_ai/gemini-2.5-flash").status == "BLOCKED"

    def test_tinker_same_model_judge_uses_bridge_credential_boundary(self):
        gate = runner._judge_key_gate(f"openai/{runner.TINKER_BRIDGE_MODEL_ALIAS}")
        assert gate.status == "PASS"
        assert gate.details["same_model_judge"]
        assert gate.details["provider_key_present"] == "TINKER_BRIDGE_API_KEY_OR_KEYCHAIN"


class BridgeCredentialGateTests(unittest.TestCase):
    def test_non_macos_preflight_fails_closed_without_raising(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(runner.shutil, "which", return_value=None),
        ):
            assert runner._resolve_bridge_credential() == (None, None)
            assert runner._tinker_gate().status == "BLOCKED"


class LocalDatasetGateTests(unittest.TestCase):
    """`--dataset-dir` must prove the revision, not just the filenames."""

    def _materialize(self, root: Path, *, commit: str | None) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        payloads = {
            "tasks_and_rubrics.json": json.dumps([synthetic_task("a")]),
            "world_descriptions.json": json.dumps([synthetic_world()]),
            "eval.yaml": "name: APEX-Agents\n",
        }
        cache = root / ".cache" / "huggingface" / "download"
        cache.mkdir(parents=True, exist_ok=True)
        for name, body in payloads.items():
            (root / name).write_text(body, encoding="utf-8")
            if commit is not None:
                (cache / f"{name}.metadata").write_text(
                    f"{commit}\netag\n123.0\n", encoding="utf-8"
                )
        # This expected manifest is fixed before any adversarial mutation and
        # represents an independent trust source only within this fixture.
        self.enterContext(
            mock.patch.object(
                runner,
                "TRUSTED_LOCAL_DATASET_SHA256",
                {name: runner.sha256_bytes(body.encode()) for name, body in payloads.items()},
            )
        )
        return root

    def test_changed_bytes_do_not_inherit_cache_revision_trust(self):
        with TemporaryDirectory() as tmp:
            root = self._materialize(Path(tmp) / "ds", commit=runner.DATASET_REVISION)
            original_metadata = (
                root / ".cache/huggingface/download/tasks_and_rubrics.json.metadata"
            ).read_bytes()
            (root / "tasks_and_rubrics.json").write_text(json.dumps([synthetic_task("b")]))
            gate, resolved = runner._dataset_access_gate(
                token=None, cache_dir=Path(tmp) / "cache", dataset_dir=root
            )
            assert (
                root / ".cache/huggingface/download/tasks_and_rubrics.json.metadata"
            ).read_bytes() == original_metadata
            assert gate.status == "BLOCKED"
            assert resolved is None
            assert not gate.details["content_verified"]

    def test_missing_independent_expected_manifest_is_blocked(self):
        with TemporaryDirectory() as tmp:
            root = self._materialize(Path(tmp) / "ds", commit=runner.DATASET_REVISION)
            with mock.patch.object(runner, "TRUSTED_LOCAL_DATASET_SHA256", {}):
                gate, resolved = runner._dataset_access_gate(
                    token=None, cache_dir=Path(tmp) / "cache", dataset_dir=root
                )
            assert gate.status == "BLOCKED"
            assert resolved is None

    def test_pinned_revision_passes_and_records_digests(self):
        with TemporaryDirectory() as tmp:
            root = self._materialize(Path(tmp) / "ds", commit=runner.DATASET_REVISION)
            gate, resolved = runner._dataset_access_gate(
                token=None, cache_dir=Path(tmp) / "cache", dataset_dir=root
            )
        assert gate.status == "PASS"
        assert gate.details["source"] == "local_dataset_dir"
        assert set(resolved) == {"tasks", "worlds", "eval"}
        for record in gate.details["files"].values():
            assert record["recorded_commit"] == runner.DATASET_REVISION
            assert len(record["sha256"]) == 64

    def test_copy_from_another_revision_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = self._materialize(Path(tmp) / "ds", commit="0" * 40)
            gate, resolved = runner._dataset_access_gate(
                token=None, cache_dir=Path(tmp) / "cache", dataset_dir=root
            )
        assert gate.status == "BLOCKED"
        assert resolved is None
        assert gate.details["wrong_revision"]

    def test_directory_without_cache_metadata_is_rejected(self):
        """A hand-assembled directory cannot masquerade as the pinned revision."""
        with TemporaryDirectory() as tmp:
            root = self._materialize(Path(tmp) / "ds", commit=None)
            gate, resolved = runner._dataset_access_gate(
                token=None, cache_dir=Path(tmp) / "cache", dataset_dir=root
            )
        assert gate.status == "BLOCKED"
        assert resolved is None
        assert [entry["recorded_commit"] for entry in gate.details["wrong_revision"]] == [
            None,
            None,
            None,
        ]

    def test_missing_file_is_reported(self):
        with TemporaryDirectory() as tmp:
            root = self._materialize(Path(tmp) / "ds", commit=runner.DATASET_REVISION)
            (root / "eval.yaml").unlink()
            gate, _ = runner._dataset_access_gate(
                token=None, cache_dir=Path(tmp) / "cache", dataset_dir=root
            )
        assert gate.status == "BLOCKED"
        assert gate.details["missing"] == ["eval.yaml"]


class ReceiptEmissionTests(unittest.TestCase):
    """The receipt must stay honest when ingestion cannot happen."""

    def _args(self, tmp: Path) -> argparse.Namespace:
        return argparse.Namespace(
            launch=False,
            dataset_revision=runner.DATASET_REVISION,
            cache_dir=tmp / "cache",
            archipelago_dir=tmp / "archipelago",
            training_task_ids=None,
        )

    def test_blocked_receipt_carries_schema_gate_and_no_score(self):
        with TemporaryDirectory() as tmp:
            args = self._args(Path(tmp))
            gates = [runner._dataset_schema_gate(None)]
            receipt = runner._build_receipt(
                args=args,
                gates=gates,
                config={"suite_id": runner.SUITE_ID},
                status="BLOCKED",
                selected_tasks=[],
            )
        assert receipt["status"] == "BLOCKED"
        assert receipt["launch"]["score"] is None
        assert receipt["launch"]["tinker_calls"] == 0
        assert receipt["no_score_claim"]
        names = [gate["name"] for gate in receipt["gates"]]
        assert "dataset_schema" in names
        assert receipt["task_selection"]["count"] == 0

    def test_required_external_receipts_exclude_passing_gates(self):
        with TemporaryDirectory() as tmp:
            args = self._args(Path(tmp))
            downloaded_dir = Path(tmp)
            (downloaded_dir / "tasks_and_rubrics.json").write_text(
                json.dumps([synthetic_task("a")]), encoding="utf-8"
            )
            (downloaded_dir / "world_descriptions.json").write_text(
                json.dumps([synthetic_world()]), encoding="utf-8"
            )
            gate = runner._dataset_schema_gate(
                {
                    "tasks": downloaded_dir / "tasks_and_rubrics.json",
                    "worlds": downloaded_dir / "world_descriptions.json",
                }
            )
            receipt = runner._build_receipt(
                args=args,
                gates=[gate],
                config={"suite_id": runner.SUITE_ID},
                status="BLOCKED",
                selected_tasks=[],
            )
        assert receipt["required_external_receipts"] == []


if __name__ == "__main__":
    unittest.main()
