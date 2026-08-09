#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
import sys
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from . import pavlov_verilog_eval_split_manifest as validator  # noqa: F401
else:
    try:
        from . import pavlov_verilog_eval_split_manifest as validator
    except ImportError:
        _TEST_DIR = Path(__file__).resolve().parent
        if str(_TEST_DIR) not in sys.path:
            sys.path.insert(0, str(_TEST_DIR))
        validator = import_module("pavlov_verilog_eval_split_manifest")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _task_ids() -> list[str]:
    return sorted(hashlib.sha256(f"verilog-eval-task-{index}".encode("utf-8")).hexdigest() for index in range(3))


def _aggregate(task_hashes: list[str]) -> str:
    return _sha256("\n".join(task_hashes))


def _split_manifest_hash(aggregate: str) -> str:
    payload = _canonical_json({"primary_eval": aggregate})
    return "sha256:" + _sha256(payload)


def _valid_manifest() -> dict[str, Any]:
    task_hashes = _task_ids()
    aggregate = _aggregate(task_hashes)
    return {
        "suite_id": "verilog_eval",
        "source": "https://github.com/NVlabs/verilog-eval",
        "category": "code",
        "role": "primary_eval",
        "dataset": {
            "revision": "a" * 40,
            "license": "apache-2.0",
            "source": "https://github.com/NVlabs/verilog-eval",
        },
        "task_id_hashes": task_hashes,
        "split": {
            "primary_eval": list(task_hashes),
            "hash": "sha256:" + aggregate,
        },
        "split_hashes": {
            "primary_eval": "sha256:" + aggregate,
        },
        "split_manifest_hash": _split_manifest_hash(aggregate),
        "split_manifest_receipt_ref": "sha256:" + ("a" * 64),
        "decontamination": {
            "status": "verified",
            "receipt_id": "b" * 40,
            "visibility": "private",
            "safe_public_artifact": True,
        },
    }


class VerilogEvalSplitManifestBoundaryTests(unittest.TestCase):
    def test_valid_manifest_builds_ready_and_disables_launch(self) -> None:
        report = validator.build_split_manifest_record(_valid_manifest())
        self.assertEqual(report["status"], "READY")
        self.assertTrue(report["split_ready"])
        self.assertFalse(report["paid_launch_allowed"])
        self.assertFalse(report["launch"]["allowed"])
        self.assertIn("launch is intentionally disabled", report["launch"]["reasons"])
        self.assertEqual(report["split"]["primary_eval"]["count"], 3)
        self.assertEqual(len(report["split"]["primary_eval"]["task_id_hashes"]), 3)

    def test_wrong_suite_is_blocked(self) -> None:
        manifest = _valid_manifest()
        manifest["suite_id"] = "openreward_games_eval"
        blocked = validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertIn("suite_id must be 'verilog_eval'", " ".join(blocked["blockers"]))

    def test_authoritative_source_is_required(self) -> None:
        manifest = _valid_manifest()
        manifest["source"] = "https://github.com/example/fake"
        blocked = validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(
            any("source must match authoritative verilog_eval source" in item for item in blocked["blockers"])
        )

    def test_rejects_xlam_and_glaive_substitution_markers(self) -> None:
        manifest = _valid_manifest()
        manifest["dataset"]["source"] = "https://github.com/Salesforce/xlam-function-calling-60k"
        blocked = validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("blocked source" in item or "authoritative" in item for item in blocked["blockers"]))

        manifest = _valid_manifest()
        manifest["source"] = "https://github.com/glaiveai/glaive-function-calling-v2"
        blocked = validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("source must match authoritative verilog_eval source" in item for item in blocked["blockers"]))

    def test_task_and_split_hashes_must_be_deterministic(self) -> None:
        manifest = _valid_manifest()
        manifest["task_id_hashes"][0], manifest["task_id_hashes"][1] = manifest["task_id_hashes"][1], manifest["task_id_hashes"][0]
        blocked = validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("must be deterministically sorted" in item for item in blocked["blockers"]))

        manifest = _valid_manifest()
        manifest["split"]["hash"] = "sha256:" + ("d" * 64)
        blocked = validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("hash does not match aggregate" in item for item in blocked["blockers"]))

        manifest = _valid_manifest()
        manifest["split_manifest_hash"] = "sha256:" + ("e" * 64)
        blocked = validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(
            any(
                "split_manifest_hash does not match observed split aggregate" in item
                for item in blocked["blockers"]
            )
        )

    def test_decontamination_status_is_constrained(self) -> None:
        manifest = _valid_manifest()
        manifest["decontamination"]["status"] = "pending"
        blocked = validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("decontamination.status is invalid" in item for item in blocked["blockers"]))

    def test_scope_with_e_suite_reference_is_blocked(self) -> None:
        manifest = _valid_manifest()
        manifest["scope"] = {"e_suite_ids": ["swe_bench_pro_eval"]}
        blocked = validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("E-suite" in item or "E-suite entries" in item for item in blocked["blockers"]))

    def test_no_network_or_credentials(self) -> None:
        manifest = _valid_manifest()
        manifest["requires_network"] = True
        blocked = validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("network is disallowed" in item for item in blocked["blockers"]))

        manifest = _valid_manifest()
        manifest["api_key"] = "secret-token"
        blocked = validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("api_key is not allowed" in item for item in blocked["blockers"]))

    def test_dataset_and_receipt_digests_are_deterministic(self) -> None:
        first = validator.build_split_manifest_record(_valid_manifest())
        second = validator.build_split_manifest_record(copy.deepcopy(_valid_manifest()))
        self.assertEqual(first["status"], "READY")
        self.assertEqual(second["status"], "READY")
        self.assertEqual(first["task_id_digest"], second["task_id_digest"])
        self.assertEqual(
            first["split_manifest_payload_digest"],
            second["split_manifest_payload_digest"],
        )

    def test_validate_record_helper_matches_status(self) -> None:
        ready = validator.build_split_manifest_record(_valid_manifest())
        self.assertEqual(validator.validate_split_manifest_record(ready), [])

        blocked = validator.build_split_manifest_record({"suite_id": "x"})
        self.assertNotEqual(validator.validate_split_manifest_record(blocked), [])

    def test_main_reports_status(self) -> None:
        temporary = Path(__file__).resolve().parent / "_tmp_verilog_eval_split_manifest.json"
        try:
            manifest = _valid_manifest()
            temporary.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(validator.main(["--manifest", str(temporary)]), 0)

            manifest["suite_id"] = "not-verilog"
            temporary.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(validator.main(["--manifest", str(temporary)]), 1)
        finally:
            if temporary.exists():
                temporary.unlink()


_PINNED_CHECKOUT = (
    Path(__file__).resolve().parents[2]
    / "outputs/e11_verilog_eval/nvlabs_verilog_eval_c498220d"
)


def _write_synthetic_checkout(root: Path, *, problems: dict[str, list[str]]) -> Path:
    """Create a miniature checkout with the same file contract as upstream."""

    for dataset_name, problem_ids in problems.items():
        dataset_dir = root / f"dataset_{dataset_name}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "problems.txt").write_text(
            "\n".join(problem_ids) + "\n", encoding="utf-8"
        )
        for problem_id in problem_ids:
            (dataset_dir / f"{problem_id}_prompt.txt").write_text(
                f"prompt for {dataset_name}/{problem_id}\n", encoding="utf-8"
            )
            (dataset_dir / f"{problem_id}_ref.sv").write_text(
                "module RefModule (\n  output zero\n);\n  assign zero = 1'b0;\nendmodule\n",
                encoding="utf-8",
            )
            (dataset_dir / f"{problem_id}_test.sv").write_text(
                "module tb;\nendmodule\n", encoding="utf-8"
            )
            if dataset_name == "code-complete-iccad2023":
                (dataset_dir / f"{problem_id}_ifc.txt").write_text(
                    "module TopModule (\n  output zero\n);\n", encoding="utf-8"
                )
    return root


class VerilogEvalSplitManifestBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.addCleanup(self._temp.cleanup)
        self.problems = {
            "code-complete-iccad2023": ["Prob001_zero", "Prob002_m2014_q4i"],
            "spec-to-rtl": ["Prob001_zero", "Prob002_m2014_q4i"],
        }
        _write_synthetic_checkout(self.root, problems=self.problems)

    def test_canonical_task_id_disambiguates_the_two_task_framings(self) -> None:
        code_complete = validator.canonical_task_id("code-complete-iccad2023", "Prob001_zero")
        spec_to_rtl = validator.canonical_task_id("spec-to-rtl", "Prob001_zero")
        self.assertEqual(code_complete, "verilog_eval/code-complete-iccad2023/Prob001_zero")
        self.assertNotEqual(code_complete, spec_to_rtl)
        self.assertNotEqual(
            validator.task_id_hash("code-complete-iccad2023", "Prob001_zero"),
            validator.task_id_hash("spec-to-rtl", "Prob001_zero"),
        )

    def test_task_id_hash_is_sha256_of_canonical_id(self) -> None:
        self.assertEqual(
            validator.task_id_hash("spec-to-rtl", "Prob001_zero"),
            _sha256("verilog_eval/spec-to-rtl/Prob001_zero"),
        )

    def test_manifest_hashes_are_internally_consistent(self) -> None:
        manifest = validator.build_manifest_from_checkout(self.root)
        task_hashes = manifest["task_id_hashes"]

        self.assertEqual(len(task_hashes), 4)
        self.assertEqual(task_hashes, sorted(task_hashes))
        self.assertEqual(len(set(task_hashes)), 4)

        aggregate = _aggregate(task_hashes)
        self.assertEqual(manifest["split"]["hash"], "sha256:" + aggregate)
        self.assertEqual(manifest["split_hashes"]["primary_eval"], "sha256:" + aggregate)
        self.assertEqual(manifest["split_manifest_hash"], _split_manifest_hash(aggregate))
        self.assertEqual(manifest["split"]["primary_eval"], task_hashes)

    def test_per_task_content_hashes_are_present_and_deterministic(self) -> None:
        first = validator.build_manifest_from_checkout(self.root)
        second = validator.build_manifest_from_checkout(self.root)
        self.assertEqual(first["tasks"], second["tasks"])

        by_id = {task["canonical_task_id"]: task for task in first["tasks"]}
        code_complete = by_id["verilog_eval/code-complete-iccad2023/Prob001_zero"]
        spec_to_rtl = by_id["verilog_eval/spec-to-rtl/Prob001_zero"]

        self.assertEqual(
            sorted(code_complete["artifact_sha256"]), ["ifc.txt", "prompt.txt", "ref.sv", "test.sv"]
        )
        # spec-to-rtl ships no interface file, by upstream design.
        self.assertEqual(sorted(spec_to_rtl["artifact_sha256"]), ["prompt.txt", "ref.sv", "test.sv"])
        self.assertEqual(
            code_complete["content_digest"],
            _sha256(_canonical_json(code_complete["artifact_sha256"])),
        )

    def test_artifact_present_but_unlisted_is_rejected(self) -> None:
        dataset_dir = self.root / "dataset_spec-to-rtl"
        (dataset_dir / "Prob999_smuggled_ref.sv").write_text("module RefModule ();\n", encoding="utf-8")
        with self.assertRaises(validator.VerilogEvalSplitManifestError) as caught:
            validator.build_manifest_from_checkout(self.root)
        self.assertIn("absent from problems.txt", str(caught.exception))

    def test_listed_problem_without_reference_is_rejected(self) -> None:
        dataset_dir = self.root / "dataset_spec-to-rtl"
        (dataset_dir / "problems.txt").write_text(
            "Prob001_zero\nProb002_m2014_q4i\nProb003_ghost\n", encoding="utf-8"
        )
        with self.assertRaises(validator.VerilogEvalSplitManifestError) as caught:
            validator.build_manifest_from_checkout(self.root)
        self.assertIn("no reference", str(caught.exception))

    def test_missing_required_artifact_is_rejected(self) -> None:
        (self.root / "dataset_code-complete-iccad2023" / "Prob001_zero_ifc.txt").unlink()
        with self.assertRaises(validator.VerilogEvalSplitManifestError) as caught:
            validator.build_manifest_from_checkout(self.root)
        self.assertIn("missing required artifact", str(caught.exception))

    def test_duplicate_problem_list_entry_is_rejected(self) -> None:
        (self.root / "dataset_spec-to-rtl" / "problems.txt").write_text(
            "Prob001_zero\nProb001_zero\nProb002_m2014_q4i\n", encoding="utf-8"
        )
        with self.assertRaises(validator.VerilogEvalSplitManifestError) as caught:
            validator.build_manifest_from_checkout(self.root)
        self.assertIn("duplicates", str(caught.exception))

    def test_receipt_blocks_on_missing_decontamination_only(self) -> None:
        receipt = validator.build_split_manifest_receipt(self.root)
        self.assertEqual(receipt["status"], "BLOCKED")
        self.assertEqual(receipt["task_count"], 4)
        self.assertFalse(receipt["is_model_score"])
        self.assertIsNone(receipt["score"])
        self.assertFalse(receipt["launch"]["paid_work_launched"])
        self.assertEqual(receipt["validation"]["blockers"], ["decontamination must be an object"])

    def test_receipt_is_ready_once_a_decontamination_receipt_exists(self) -> None:
        receipt = validator.build_split_manifest_receipt(
            self.root,
            decontamination={
                "status": "verified",
                "receipt_id": "b" * 40,
                "visibility": "private",
                "safe_public_artifact": True,
            },
        )
        self.assertEqual(receipt["status"], "READY")
        self.assertEqual(receipt["validation"]["split"]["primary_eval"]["count"], 4)
        self.assertFalse(receipt["validation"]["paid_launch_allowed"])

    @unittest.skipUnless(_PINNED_CHECKOUT.is_dir(), "pinned NVlabs checkout not present")
    def test_pinned_checkout_yields_312_tasks(self) -> None:
        manifest = validator.build_manifest_from_checkout(_PINNED_CHECKOUT)
        self.assertEqual(len(manifest["task_id_hashes"]), 312)
        self.assertEqual(manifest["dataset"]["revision"], validator.PINNED_REVISION)
        self.assertEqual(manifest["dataset"]["license"], "MIT")
        self.assertEqual(
            manifest["datasets"]["code-complete-iccad2023"]["problem_count"], 156
        )
        self.assertEqual(manifest["datasets"]["spec-to-rtl"]["problem_count"], 156)
        self.assertEqual(
            manifest["split_hashes"]["primary_eval"],
            "sha256:" + _aggregate(manifest["task_id_hashes"]),
        )

    @unittest.skipUnless(_PINNED_CHECKOUT.is_dir(), "pinned NVlabs checkout not present")
    def test_pinned_checkout_manifest_is_reproducible(self) -> None:
        first = validator.build_manifest_from_checkout(_PINNED_CHECKOUT)
        second = validator.build_manifest_from_checkout(_PINNED_CHECKOUT)
        self.assertEqual(first["split_manifest_hash"], second["split_manifest_hash"])
        self.assertEqual(
            first["split_manifest_receipt_ref"], second["split_manifest_receipt_ref"]
        )

    def test_main_checkout_mode_writes_a_receipt(self) -> None:
        output = self.root / "receipt.json"
        exit_code = validator.main(
            ["--checkout", str(self.root), "--output", str(output)]
        )
        self.assertEqual(exit_code, 1)  # blocked on decontamination
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["task_count"], 4)
        self.assertEqual(payload["manifest"]["suite_id"], "verilog_eval")


if __name__ == "__main__":
    unittest.main()
