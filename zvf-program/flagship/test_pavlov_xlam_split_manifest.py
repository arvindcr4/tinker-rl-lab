from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from flagship.pavlov_xlam_split_manifest import (
    DATASET_ID,
    DEFAULT_SEED,
    SplitManifestError,
    build_manifest,
    main,
    task_hash,
    validate_portfolio_domain_coverage,
    verify_manifest,
)


@dataclass
class FakeExample:
    prompt: str
    target: dict[str, object]


@dataclass
class FakeDataset:
    train: list[FakeExample]
    test: list[FakeExample]


REVISION = "a" * 40


def fake_dataset() -> FakeDataset:
    return FakeDataset(
        train=[
            FakeExample("train zero", {"tool": "search", "arguments": {"q": "zero"}}),
            FakeExample("train one", {"tool": "search", "arguments": {"q": "one"}}),
        ],
        test=[
            FakeExample("test zero", {"tool": "search", "arguments": {"q": "heldout"}}),
        ],
    )


class PavlovXlamSplitManifestTests(unittest.TestCase):
    def test_manifest_is_deterministic_and_content_free(self) -> None:
        first = build_manifest(REVISION, dataset=fake_dataset())
        second = build_manifest(REVISION, dataset=fake_dataset())

        self.assertEqual(first, second)
        self.assertEqual(first["dataset_id"], DATASET_ID)
        self.assertEqual(first["seed"], DEFAULT_SEED)
        self.assertEqual(first["counts"], {"train": 2, "test": 1})
        self.assertEqual(first["split_roles"], {"train": "train", "test": "primary_eval"})
        self.assertEqual(first["domain_tags"], ["tool_use"])
        self.assertEqual(first["status"], "BLOCKED")
        self.assertFalse(first["launch_authorized"])
        self.assertIn("not portfolio-wide evidence", first["evidence_scope"])
        rendered = json.dumps(first)
        self.assertNotIn("train zero", rendered)
        self.assertNotIn("heldout", rendered)
        self.assertNotIn('"prompt"', rendered)
        self.assertNotIn('"target"', rendered)
        self.assertEqual(
            first["train_aggregate_sha256"],
            first["aggregate_hashes"]["train"],
        )
        self.assertFalse(first["launches_any_job"])

    def test_unpinned_revision_is_rejected(self) -> None:
        with self.assertRaisesRegex(SplitManifestError, "immutable 40-character"):
            build_manifest("main", dataset=fake_dataset())
        with self.assertRaisesRegex(SplitManifestError, "immutable 40-character"):
            build_manifest("A" * 40, dataset=fake_dataset())

    def test_factory_receives_pinned_revision_and_frozen_seed(self) -> None:
        calls: list[tuple[int, str]] = []

        def factory(*, seed: int, revision: str) -> FakeDataset:
            calls.append((seed, revision))
            return fake_dataset()

        build_manifest(REVISION, dataset_factory=factory)
        self.assertEqual(calls, [(DEFAULT_SEED, REVISION)])

    def test_train_test_overlap_is_hard_failure(self) -> None:
        duplicate = FakeExample("same", {"tool": "search", "arguments": {"q": "same"}})
        with self.assertRaisesRegex(SplitManifestError, "train/test contamination"):
            build_manifest(
                REVISION,
                dataset=FakeDataset(train=[duplicate], test=[duplicate]),
            )

    def test_verify_rejects_revision_drift_and_task_drift(self) -> None:
        manifest = build_manifest(REVISION, dataset=fake_dataset())
        self.assertTrue(verify_manifest(manifest, expected_revision=REVISION))
        with self.assertRaisesRegex(SplitManifestError, "revision drift"):
            verify_manifest(manifest, expected_revision="b" * 40)

        changed = fake_dataset()
        changed.test[0] = FakeExample("changed", {"tool": "search", "arguments": {"q": "changed"}})
        with self.assertRaisesRegex(SplitManifestError, "test task hashes drift"):
            verify_manifest(manifest, dataset=changed)

    def test_cross_suite_overlap_is_hard_failure(self) -> None:
        manifest = build_manifest(REVISION, dataset=fake_dataset())
        other = copy.deepcopy(manifest)
        other["suite_id"] = "other_suite"
        with self.assertRaisesRegex(SplitManifestError, "cross-suite task-hash overlap"):
            build_manifest(
                REVISION,
                dataset=fake_dataset(),
                cross_suite_manifests=[other],
            )

    def test_portfolio_receipts_and_roles_are_composable(self) -> None:
        manifest = build_manifest(
            REVISION,
            dataset=fake_dataset(),
            suite_id="xlam_tool_use",
            domain_tags=["tool_use", "function_calling", "tool_use"],
            receipt_refs={
                "revision": "receipt://revision",
                "license": "receipt://license",
                "container": "receipt://container",
                "decontamination": "receipt://decontamination",
            },
        )
        self.assertEqual(manifest["suite_id"], "xlam_tool_use")
        self.assertEqual(manifest["domain_tags"], ["function_calling", "tool_use"])
        self.assertEqual(manifest["receipt_refs"]["license"], "receipt://license")
        self.assertTrue(manifest["receipt_refs"]["split_manifest"].startswith("sha256:"))
        self.assertEqual(
            manifest["provenance_receipts"]["decontamination_receipt_ref"],
            "receipt://decontamination",
        )
        self.assertEqual(manifest["portfolio_contract"]["expected_training_suite_count"], 12)
        self.assertEqual(manifest["portfolio_contract"]["expected_primary_eval_suite_count"], 14)

    def test_full_portfolio_domain_check_requires_both_role_unions(self) -> None:
        domains = [f"domain_{index}" for index in range(16)]
        manifests = []
        for index in range(12):
            dataset = FakeDataset(
                train=[FakeExample(f"train-{index}", {"index": index})],
                test=[FakeExample(f"train-test-{index}", {"index": index})],
            )
            manifest = build_manifest(
                REVISION,
                dataset=dataset,
                suite_id=f"train_suite_{index}",
                domain_tags=[domains[index % len(domains)]],
                receipt_refs={
                    "revision": f"receipt://revision/{index}",
                    "license": f"receipt://license/{index}",
                    "container": f"receipt://container/{index}",
                    "decontamination": f"receipt://decontamination/{index}",
                },
            )
            manifest["suite_role"] = "train"
            manifests.append(manifest)
        for index in range(14):
            dataset = FakeDataset(
                train=[FakeExample(f"eval-train-{index}", {"index": index})],
                test=[FakeExample(f"eval-{index}", {"index": index})],
            )
            manifest = build_manifest(
                REVISION,
                dataset=dataset,
                suite_id=f"eval_suite_{index}",
                domain_tags=[domains[(index + 2) % len(domains)]],
                receipt_refs={
                    "revision": f"receipt://eval-revision/{index}",
                    "license": f"receipt://eval-license/{index}",
                    "container": f"receipt://eval-container/{index}",
                    "decontamination": f"receipt://eval-decontamination/{index}",
                },
            )
            manifest["suite_role"] = "primary_eval"
            manifests.append(manifest)

        errors = validate_portfolio_domain_coverage(
            manifests,
            declared_domains=domains,
            company_required_domains={"company": domains[:2]},
        )
        self.assertTrue(any("train domain union missing" in error for error in errors))
        self.assertTrue(any("primary_eval domain union missing" in error for error in errors))

    def test_cli_verifies_manifest_and_detects_revision_drift(self) -> None:
        manifest = build_manifest(REVISION, dataset=fake_dataset())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(
                main(["verify", "--manifest", str(path), "--revision", REVISION]),
                0,
            )
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main(["verify", "--manifest", str(path), "--revision", "b" * 40]),
                    2,
                )

    def test_task_hash_is_stable_for_mapping_and_dataclass_examples(self) -> None:
        example = fake_dataset().train[0]
        as_mapping = {"prompt": example.prompt, "target": example.target}
        self.assertEqual(task_hash(example), task_hash(as_mapping))
        self.assertEqual(len(task_hash(example)), 64)


if __name__ == "__main__":
    unittest.main()
