from __future__ import annotations

import hashlib
import json
import unittest
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING
import sys

if TYPE_CHECKING:
    from . import pavlov_bfcl_split_manifest as split_validator  # noqa: F401
else:
    try:
        from . import pavlov_bfcl_split_manifest as split_validator
    except ImportError:
        _TEST_DIR = Path(__file__).resolve().parent
        if str(_TEST_DIR) not in sys.path:
            sys.path.insert(0, str(_TEST_DIR))
        split_validator = import_module("pavlov_bfcl_split_manifest")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sorted_hex_hashes(*hashes: str) -> list[str]:
    return sorted(hashes)


def _split_aggregate(task_hashes: list[str]) -> str:
    return _sha256("\n".join(task_hashes))


def _valid_manifest() -> dict:
    train = _sorted_hex_hashes("1" * 64, "3" * 64)
    primary_eval = _sorted_hex_hashes("0" * 64, "2" * 64)
    train_aggregate = _split_aggregate(train)
    primary_aggregate = _split_aggregate(primary_eval)
    split_manifest_hash = _sha256(
        _canonical_json({"primary_eval": primary_aggregate, "train": train_aggregate})
    )

    return {
        "suite_id": "bfcl_train",
        "role": "train",
        "category": "tool_use",
        "dataset": {
            "revision": "a" * 40,
            "license": "cc-by-4.0",
            "source": "https://gorilla.cs.berkeley.edu/leaderboard.html",
        },
        "split": {
            "train": train,
            "primary_eval": primary_eval,
        },
        "split_hashes": {
            "train": f"sha256:{train_aggregate}",
            "primary_eval": f"sha256:{primary_aggregate}",
        },
        "split_manifest_hash": f"sha256:{split_manifest_hash}",
        "split_manifest_receipt_ref": f"sha256:{'a' * 64}",
        "decontamination": {
            "status": "verified",
            "receipt_id": "b" * 40,
            "visibility": "private",
            "safe_public_artifact": True,
        },
    }


class BFCLSplitManifestBoundaryTests(unittest.TestCase):
    def test_valid_manifest_builds_ready_and_disables_launch(self) -> None:
        report = split_validator.build_split_manifest_record(_valid_manifest())

        self.assertEqual(report["status"], "READY")
        self.assertTrue(report["split_ready"])
        self.assertFalse(report["paid_launch_allowed"])
        self.assertFalse(report["launch"]["allowed"])
        self.assertIn("launch is intentionally disabled", report["launch"]["reasons"])
        self.assertEqual(len(report["split"]["train"]["task_hashes"]), 2)
        self.assertEqual(report["split"]["train"]["count"], 2)
        self.assertEqual(report["split"]["train"]["aggregate_sha256"], report["split_hashes"]["train"].removeprefix("sha256:"))
        self.assertEqual(len(report["split"]["primary_eval"]["task_hashes"]), 2)

    def test_non_bfcl_suite_is_blocked(self) -> None:
        manifest = _valid_manifest()
        manifest["suite_id"] = "openreward_train"
        report = split_validator.build_split_manifest_record(manifest)

        self.assertEqual(report["status"], "BLOCKED")
        self.assertFalse(report["split_ready"])
        self.assertIn("suite_id must be 'bfcl_train'", " ".join(report["blockers"]))

    def test_e_suite_is_blocked(self) -> None:
        manifest = _valid_manifest()
        manifest["suite_id"] = "swe_bench_pro_eval"
        report = split_validator.build_split_manifest_record(manifest)

        self.assertEqual(report["status"], "BLOCKED")
        self.assertFalse(report["split_ready"])
        self.assertTrue(any("not valid for T4 split validation" in item for item in report["blockers"]))

    def test_rejects_scope_e_suite_reference(self) -> None:
        manifest = _valid_manifest()
        manifest["scope"] = {"e_suite_ids": ["swe_bench_pro_eval", "other"]}
        report = split_validator.build_split_manifest_record(manifest)

        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("manifest scope includes E-suite entries" in item for item in report["blockers"]))

    def test_rejects_banned_dataset_markers(self) -> None:
        manifest = _valid_manifest()
        manifest["dataset"]["source"] = "Glaive synthetic function calling corpus"
        report = split_validator.build_split_manifest_record(manifest)

        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("dataset/source references synthetic simulator or Glaive evidence" in item for item in report["blockers"]))

        manifest = _valid_manifest()
        manifest["notes"] = "Uses bfclv4_tool_use.py simulator"
        report = split_validator.build_split_manifest_record(manifest)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("manifest notes references synthetic/Glaive sources" in item for item in report["blockers"]))

    def test_split_hash_checks(self) -> None:
        manifest = _valid_manifest()
        manifest["split_hashes"]["train"] = f"sha256:{'c' * 64}"
        report = split_validator.build_split_manifest_record(manifest)

        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("split_hashes.train does not match observed train aggregate" in item for item in report["blockers"]))

        manifest = _valid_manifest()
        manifest["split_hashes"] = {}
        report = split_validator.build_split_manifest_record(manifest)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("split_hashes.train is required" in item for item in report["blockers"]))
        self.assertTrue(any("split_hashes.primary_eval is required" in item for item in report["blockers"]))

    def test_split_manifest_hash_validation(self) -> None:
        manifest = _valid_manifest()
        manifest["split_manifest_hash"] = "sha256:" + ("e" * 64)
        report = split_validator.build_split_manifest_record(manifest)

        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("split_manifest_hash does not match observed split aggregates", report["blockers"])

    def test_decontamination_requires_verified_or_clean_status(self) -> None:
        manifest = _valid_manifest()
        manifest["decontamination"]["status"] = "reviewed"
        report = split_validator.build_split_manifest_record(manifest)

        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("decontamination.status is invalid", report["blockers"])

    def test_overlap_and_nondeterministic_split_are_rejected(self) -> None:
        manifest = _valid_manifest()
        manifest["split"]["train"] = ["3" * 64, "1" * 64]
        report = split_validator.build_split_manifest_record(manifest)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("split.train must be sorted deterministically" in item for item in report["blockers"]))

        manifest = _valid_manifest()
        manifest["split"]["primary_eval"][1] = "1" * 64
        report = split_validator.build_split_manifest_record(manifest)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("train and primary_eval task IDs overlap" in item for item in report["blockers"]))

    def test_no_network_and_no_credentials(self) -> None:
        manifest = _valid_manifest()
        manifest["requires_network"] = True
        blocked = split_validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertIn("dataset download/network is disallowed in this boundary", " ".join(blocked["blockers"]))

        manifest = _valid_manifest()
        manifest["api_key"] = "secret-token"
        blocked = split_validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("api_key is not allowed in offline split validation" in item for item in blocked["blockers"]))

    def test_validate_record_helper_matches_status(self) -> None:
        ready = split_validator.build_split_manifest_record(_valid_manifest())
        self.assertEqual(
            split_validator.validate_split_manifest_record(ready),
            [],
        )

        blocked = split_validator.build_split_manifest_record({"suite_id": "openreward_train"})
        self.assertNotEqual(blocked["status"], "READY")
        self.assertEqual(
            split_validator.validate_split_manifest_record(blocked),
            blocked["blockers"],
        )

    def test_main_reports_status_exit_code(self) -> None:
        temporary = Path(__file__).resolve().parent / "_tmp_bfcl_split_manifest.json"
        try:
            manifest = _valid_manifest()
            temporary.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(split_validator.main(["--manifest", str(temporary)]), 0)

            manifest["suite_id"] = "wrong_suite"
            temporary.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(split_validator.main(["--manifest", str(temporary)]), 1)
        finally:
            if temporary.exists():
                temporary.unlink()


if __name__ == "__main__":
    raise SystemExit(unittest.main())
