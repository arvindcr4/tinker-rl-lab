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
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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
        self.assertEqual(runner.validate_task_records(tasks), [])

    def test_non_array_is_rejected(self):
        errors = runner.validate_task_records({"task_id": "task_" + "a" * 32})
        self.assertEqual(len(errors), 1)
        self.assertIn("JSON array", errors[0])

    def test_empty_array_is_rejected(self):
        self.assertIn("zero task records", runner.validate_task_records([])[0])

    def test_each_required_field_is_enforced(self):
        for field in runner.APEX_TASK_REQUIRED_FIELDS:
            with self.subTest(field=field):
                task = synthetic_task("a")
                del task[field]
                errors = runner.validate_task_records([task])
                self.assertTrue(
                    any(f"tasks[0].{field}" in error for error in errors),
                    f"missing {field} was not reported: {errors}",
                )

    def test_blank_required_field_is_rejected(self):
        task = synthetic_task("a")
        task["prompt"] = "   "
        errors = runner.validate_task_records([task])
        self.assertTrue(any("tasks[0].prompt" in error for error in errors))

    def test_malformed_task_id_is_rejected(self):
        task = synthetic_task("a")
        task["task_id"] = "not-a-task-id"
        errors = runner.validate_task_records([task])
        self.assertTrue(any("does not match" in error for error in errors))

    def test_uppercase_hex_task_id_is_rejected(self):
        task = synthetic_task("a")
        task["task_id"] = "task_" + "A" * 32
        errors = runner.validate_task_records([task])
        self.assertTrue(any("does not match" in error for error in errors))

    def test_duplicate_task_ids_are_reported(self):
        task = synthetic_task("a")
        errors = runner.validate_task_records([task, dict(task)])
        self.assertTrue(any("duplicated" in error for error in errors))

    def test_missing_rubric_is_rejected(self):
        task = synthetic_task("a")
        del task["rubric"]
        errors = runner.validate_task_records([task])
        self.assertTrue(any("rubric is missing" in error for error in errors))

    def test_empty_rubric_is_rejected(self):
        task = synthetic_task("a", rubric_size=0)
        errors = runner.validate_task_records([task])
        self.assertTrue(any("non-empty JSON array" in error for error in errors))

    def test_rubric_criterion_fields_are_enforced(self):
        for field in runner.APEX_RUBRIC_REQUIRED_FIELDS:
            with self.subTest(field=field):
                task = synthetic_task("a", rubric_size=1)
                del task["rubric"][0][field]
                errors = runner.validate_task_records([task])
                self.assertTrue(
                    any(f"rubric[0].{field}" in error for error in errors),
                    f"missing rubric {field} was not reported: {errors}",
                )

    def test_duplicate_verifier_ids_are_reported(self):
        task = synthetic_task("a", rubric_size=1)
        task["rubric"].append(dict(task["rubric"][0]))
        errors = runner.validate_task_records([task])
        self.assertTrue(any("verifier_id" in e and "duplicated" in e for e in errors))


class WorldRecordSchemaTests(unittest.TestCase):
    def test_valid_world_records(self):
        self.assertEqual(runner.validate_world_records([synthetic_world()]), [])

    def test_missing_world_name_is_rejected(self):
        world = synthetic_world()
        del world["world_name"]
        errors = runner.validate_world_records([world])
        self.assertTrue(any("worlds[0].world_name" in error for error in errors))

    def test_duplicate_world_ids_are_reported(self):
        world = synthetic_world()
        errors = runner.validate_world_records([world, dict(world)])
        self.assertTrue(any("duplicated" in error for error in errors))


class ReferentialIntegrityTests(unittest.TestCase):
    def test_resolvable_world_reference(self):
        tasks = [synthetic_task("a")]
        self.assertEqual(
            runner.validate_dataset_references(tasks, [synthetic_world()]), []
        )

    def test_dangling_world_reference_is_reported(self):
        tasks = [synthetic_task("a", world_id="world_missing")]
        errors = runner.validate_dataset_references(tasks, [synthetic_world()])
        self.assertTrue(any("has no entry" in error for error in errors))


class IngestionReportTests(unittest.TestCase):
    def test_valid_dataset_reports_valid_with_stable_hashes(self):
        tasks = [synthetic_task("a"), synthetic_task("b")]
        worlds = [synthetic_world()]
        report = runner.dataset_ingestion_report(tasks, worlds)
        self.assertTrue(report["valid"])
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["task_count"], 2)
        self.assertEqual(report["world_count"], 1)
        self.assertEqual(report["dataset_revision"], runner.DATASET_REVISION)
        # Hash is order-independent: task order in the file must not change it.
        reordered = runner.dataset_ingestion_report(list(reversed(tasks)), worlds)
        self.assertEqual(report["task_id_sha256"], reordered["task_id_sha256"])

    def test_task_id_hash_changes_when_the_set_changes(self):
        base = runner.dataset_ingestion_report(
            [synthetic_task("a")], [synthetic_world()]
        )
        grown = runner.dataset_ingestion_report(
            [synthetic_task("a"), synthetic_task("b")], [synthetic_world()]
        )
        self.assertNotEqual(base["task_id_sha256"], grown["task_id_sha256"])

    def test_count_mismatch_warns_but_does_not_invalidate(self):
        report = runner.dataset_ingestion_report(
            [synthetic_task("a")], [synthetic_world()]
        )
        self.assertTrue(report["valid"])
        self.assertTrue(report["count_warnings"])
        self.assertIn("480", " ".join(report["count_warnings"]))

    def test_schema_error_marks_report_invalid(self):
        task = synthetic_task("a")
        del task["prompt"]
        report = runner.dataset_ingestion_report([task], [synthetic_world()])
        self.assertFalse(report["valid"])
        self.assertTrue(report["errors"])


class RequiredAssetTests(unittest.TestCase):
    def test_task_without_input_files_needs_three_paths(self):
        paths = runner.required_task_assets(synthetic_task("a"))
        self.assertEqual(
            paths,
            [
                "tasks_and_rubrics.json",
                "world_descriptions.json",
                "world_files_zipped/world_synthetic_001.zip",
            ],
        )

    def test_task_with_input_files_adds_the_task_files_prefix(self):
        task = synthetic_task("a", with_input_files=True)
        paths = runner.required_task_assets(task)
        self.assertIn(f"task_files/{task['task_id']}", paths)


class DatasetSchemaGateTests(unittest.TestCase):
    def _write(self, directory: Path, tasks: list, worlds: list) -> dict:
        tasks_path = directory / "tasks_and_rubrics.json"
        worlds_path = directory / "world_descriptions.json"
        tasks_path.write_text(json.dumps(tasks), encoding="utf-8")
        worlds_path.write_text(json.dumps(worlds), encoding="utf-8")
        return {"tasks": tasks_path, "worlds": worlds_path}

    def test_gate_blocks_when_nothing_was_downloaded(self):
        gate = runner._dataset_schema_gate(None)
        self.assertEqual(gate.status, "BLOCKED")
        self.assertFalse(gate.details["validated"])
        self.assertIsNotNone(gate.required_receipt)

    def test_gate_passes_on_a_schema_valid_synthetic_dataset(self):
        with TemporaryDirectory() as tmp:
            downloaded = self._write(
                Path(tmp), [synthetic_task("a")], [synthetic_world()]
            )
            gate = runner._dataset_schema_gate(downloaded)
        self.assertEqual(gate.status, "PASS")
        self.assertTrue(gate.details["valid"])

    def test_gate_blocks_on_a_dangling_world_reference(self):
        with TemporaryDirectory() as tmp:
            downloaded = self._write(
                Path(tmp),
                [synthetic_task("a", world_id="world_missing")],
                [synthetic_world()],
            )
            gate = runner._dataset_schema_gate(downloaded)
        self.assertEqual(gate.status, "BLOCKED")
        self.assertEqual(gate.required_receipt["kind"], "dataset_schema_mismatch")

    def test_gate_is_required_for_launch(self):
        self.assertFalse(
            runner._all_launch_gates_pass([runner._dataset_schema_gate(None)])
        )


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
        self.assertEqual(receipt["status"], "BLOCKED")
        self.assertIsNone(receipt["launch"]["score"])
        self.assertEqual(receipt["launch"]["tinker_calls"], 0)
        self.assertTrue(receipt["no_score_claim"])
        names = [gate["name"] for gate in receipt["gates"]]
        self.assertIn("dataset_schema", names)
        self.assertEqual(receipt["task_selection"]["count"], 0)

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
        self.assertEqual(receipt["required_external_receipts"], [])


if __name__ == "__main__":
    unittest.main()
