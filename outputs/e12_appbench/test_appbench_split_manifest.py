"""Unit tests for the E12 AppBench split manifest.

Run:
    python3 -m unittest discover -s outputs/e12_appbench -p 'test_*.py' -v
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sm = _load("appbench_split_manifest", _HERE / "appbench_split_manifest.py")
adapter = _load(
    "pavlov_appbench_openreward_games_adapter",
    _REPO_ROOT / "zvf-program" / "flagship" / "pavlov_appbench_openreward_games_adapter.py",
)

CSV_PATH = _HERE / "hf_dataset" / "AppBench vExternal.csv"

EXPECTED_APP_NAMES = [
    "Financial Dashboard",
    "Hospital Dashboard",
    "Legal Assistant",
    "Pharmacy System",
    "Drawing Game",
    "Rental Booking",
]
EXPECTED_RUBRIC_COUNTS = [24, 33, 22, 25, 23, 24]


def _rewrite_csv(mutate) -> Path:
    """Copy the source CSV to a temp file, applying ``mutate(rows, fieldnames)``."""

    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    mutate(rows, fieldnames)
    tmp = Path(tempfile.mkdtemp()) / "AppBench vExternal.csv"
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return tmp


class TestSourceCsv(unittest.TestCase):
    def test_csv_present_and_pinned(self):
        self.assertTrue(CSV_PATH.is_file(), f"missing pinned CSV at {CSV_PATH}")
        raw = CSV_PATH.read_bytes()
        import hashlib

        self.assertEqual(hashlib.sha256(raw).hexdigest(), sm.SOURCE_SHA256)
        self.assertEqual(
            hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest(),
            sm.SOURCE_BLOB_SHA1,
            "local CSV no longer matches the Hugging Face blob oid at the pinned revision",
        )

    def test_task_shape(self):
        tasks = sm.load_tasks(CSV_PATH)
        self.assertEqual(len(tasks), 6)
        self.assertEqual([t["app_name"] for t in tasks], EXPECTED_APP_NAMES)
        self.assertEqual([t["ordinal"] for t in tasks], [1, 2, 3, 4, 5, 6])
        self.assertEqual(
            [len(t["rubric_items"]) for t in tasks], EXPECTED_RUBRIC_COUNTS
        )

    def test_rubric_items_pair_with_prompt_requirements(self):
        for task in sm.load_tasks(CSV_PATH):
            self.assertEqual(
                len(task["rubric_items"]),
                len(task["requirement_items"]),
                f"{task['app_name']}: rubric/requirement counts diverge",
            )

    def test_every_task_declares_the_same_runtime_environment(self):
        additions = {t["cli_addition"] for t in sm.load_tasks(CSV_PATH)}
        self.assertEqual(len(additions), 1)
        only = additions.pop()
        self.assertIn("Next.js", only)
        self.assertIn("Docker container", only)
        self.assertIn("Supabase", only)


class TestTaskIds(unittest.TestCase):
    def setUp(self):
        self.tasks = sm.load_tasks(CSV_PATH)

    def test_ids_are_64_hex_and_unique(self):
        ids = [sm.task_id(t) for t in self.tasks]
        self.assertEqual(len(set(ids)), 6)
        for value in ids:
            self.assertRegex(value, r"^[0-9a-f]{64}$")

    def test_ids_are_deterministic_across_rebuilds(self):
        first = [sm.task_id(t) for t in sm.load_tasks(CSV_PATH)]
        second = [sm.task_id(t) for t in sm.load_tasks(CSV_PATH)]
        self.assertEqual(first, second)

    def test_id_changes_when_prompt_changes(self):
        base = sm.task_id(self.tasks[0])
        mutated = dict(self.tasks[0])
        mutated["prompt"] = mutated["prompt"] + " x"
        self.assertNotEqual(base, sm.task_id(mutated))

    def test_id_changes_when_rubric_changes(self):
        base = sm.task_id(self.tasks[2])
        mutated = dict(self.tasks[2])
        mutated["rubric"] = mutated["rubric"].replace("Application", "App", 1)
        self.assertNotEqual(base, sm.task_id(mutated))

    def test_id_is_revision_bound_but_content_hash_is_not(self):
        task = self.tasks[1]
        other_rev = "0" * 39 + "1"
        self.assertNotEqual(sm.task_id(task), sm.task_id(task, revision=other_rev))
        self.assertEqual(sm.content_sha256(task), sm.content_sha256(dict(task)))

    def test_line_ending_transport_does_not_change_ids(self):
        lf_bytes = CSV_PATH.read_bytes().replace(b"\r\n", b"\n")
        tmp = Path(tempfile.mkdtemp()) / "lf.csv"
        tmp.write_bytes(lf_bytes)
        self.assertNotEqual(lf_bytes, CSV_PATH.read_bytes())
        self.assertEqual(
            [sm.task_id(t) for t in sm.load_tasks(tmp)],
            [sm.task_id(t) for t in self.tasks],
        )

    def test_result_columns_do_not_participate_in_identity(self):
        def fill_scores(rows, _fieldnames):
            for row in rows:
                for column in sm.RESULT_COLUMNS:
                    row[column] = "1.0"

        tmp = _rewrite_csv(fill_scores)
        self.assertEqual(
            [sm.task_id(t) for t in sm.load_tasks(tmp)],
            [sm.task_id(t) for t in self.tasks],
        )


class TestSplitHash(unittest.TestCase):
    def setUp(self):
        self.ids = [sm.task_id(t) for t in sm.load_tasks(CSV_PATH)]

    def test_order_independent(self):
        shuffled = list(self.ids)
        random.Random(809).shuffle(shuffled)
        self.assertNotEqual(shuffled, self.ids)
        self.assertEqual(sm.split_hash(shuffled), sm.split_hash(self.ids))

    def test_matches_adapter_algorithm(self):
        self.assertEqual(
            sm.split_hash(self.ids),
            adapter._deterministic_split_hash(list(self.ids)),
            "split hash must be byte-compatible with the E12 contract adapter",
        )

    def test_changes_when_a_task_is_dropped(self):
        self.assertNotEqual(sm.split_hash(self.ids[:-1]), sm.split_hash(self.ids))


class TestDisjointness(unittest.TestCase):
    def setUp(self):
        self.ids = [sm.task_id(t) for t in sm.load_tasks(CSV_PATH)]

    def test_eval_split_is_internally_unique(self):
        proof = sm.assert_disjoint({"e12_eval": self.ids})
        self.assertTrue(proof["disjoint"])
        self.assertEqual(proof["splits"]["e12_eval"], 6)

    def test_disjoint_from_a_foreign_split(self):
        import hashlib

        foreign = [hashlib.sha256(f"foreign-{i}".encode()).hexdigest() for i in range(4)]
        proof = sm.assert_disjoint({"e12_eval": self.ids, "e13_openreward": foreign})
        self.assertTrue(proof["disjoint"])
        self.assertEqual(len(proof["pairs_checked"]), 1)

    def test_overlap_is_rejected(self):
        with self.assertRaises(sm.AppBenchSplitManifestError):
            sm.assert_disjoint({"a": self.ids, "b": [self.ids[0]]})

    def test_intra_split_duplicate_is_rejected(self):
        with self.assertRaises(sm.AppBenchSplitManifestError):
            sm.assert_disjoint({"a": self.ids + [self.ids[0]]})


class TestForeignIdScan(unittest.TestCase):
    def setUp(self):
        import hashlib

        self.root = Path(tempfile.mkdtemp())
        self.foreign = hashlib.sha256(b"foreign-lane-task").hexdigest()
        (self.root / "lane_x").mkdir()
        (self.root / "lane_x" / "receipt.json").write_text(
            json.dumps({"task_ids": [self.foreign], "note": "not an id"}), encoding="utf-8"
        )
        # A vendored tree that must be pruned, carrying a decoy ID.
        vendored = self.root / "lane_y" / "venv" / "site-packages"
        vendored.mkdir(parents=True)
        self.decoy = hashlib.sha256(b"vendored-decoy").hexdigest()
        (vendored / "data.json").write_text(
            json.dumps({"task_id": self.decoy}), encoding="utf-8"
        )
        # The lane's own directory must be excluded from its own scan.
        (self.root / "e12_appbench").mkdir()
        self.own = hashlib.sha256(b"own-id").hexdigest()
        (self.root / "e12_appbench" / "split_manifest.json").write_text(
            json.dumps({"tasks": [{"task_id": self.own}]}), encoding="utf-8"
        )

    def test_scan_finds_foreign_ids_and_prunes_vendored_and_own(self):
        found = sm.collect_foreign_task_ids(self.root, self.root / "e12_appbench")
        all_ids = {i for ids in found.values() for i in ids}
        self.assertIn(self.foreign, all_ids)
        self.assertNotIn(self.decoy, all_ids, "vendored tree was not pruned")
        self.assertNotIn(self.own, all_ids, "the lane's own directory was not excluded")

    def test_emitted_disjointness_proof_is_clean(self):
        path = _HERE / "disjointness_proof.json"
        if not path.is_file():
            self.skipTest("disjointness_proof.json not built yet")
        proof = json.loads(path.read_text(encoding="utf-8"))
        manifest = sm.build_manifest(CSV_PATH)
        self.assertTrue(proof["disjoint"])
        self.assertEqual(proof["intersection_with_eval"], [])
        self.assertEqual(proof["eval_split_hash"], manifest["split"]["split_hash"])
        self.assertGreater(proof["foreign_unique_ids"], 0)


class TestManifest(unittest.TestCase):
    def setUp(self):
        self.manifest = sm.build_manifest(CSV_PATH)

    def test_manifest_is_reproducible(self):
        self.assertEqual(
            sm.canonical_json(self.manifest),
            sm.canonical_json(sm.build_manifest(CSV_PATH)),
        )

    def test_manifest_core_fields(self):
        self.assertEqual(self.manifest["boundary_id"], "E12")
        self.assertEqual(self.manifest["suite_id"], "appbench_eval")
        self.assertEqual(self.manifest["split"]["task_count"], 6)
        self.assertEqual(self.manifest["split"]["unique_task_count"], 6)
        self.assertEqual(
            self.manifest["source"]["file_sha256"],
            self.manifest["source"]["expected_file_sha256"],
        )
        self.assertRegex(self.manifest["aggregate_sha256"], r"^[0-9a-f]{64}$")

    def test_manifest_does_not_claim_training_holdout(self):
        claim = self.manifest["heldout_claim"]
        self.assertIsNone(claim["held_out_from_model_training"])
        self.assertTrue(claim["immutable_task_ids"])
        self.assertIn("vExternal", claim["held_out_reason"])

    def test_verify_accepts_a_clean_manifest(self):
        self.assertEqual(sm.verify_manifest(self.manifest, CSV_PATH)["verified"], True)

    def test_verify_rejects_a_tampered_task_id(self):
        tampered = copy.deepcopy(self.manifest)
        tampered["tasks"][0]["task_id"] = "0" * 63 + "1"
        with self.assertRaises(sm.AppBenchSplitManifestError):
            sm.verify_manifest(tampered, CSV_PATH)

    def test_verify_rejects_a_tampered_split_hash(self):
        tampered = copy.deepcopy(self.manifest)
        tampered["split"]["split_hash"] = "a" * 63 + "b"
        with self.assertRaises(sm.AppBenchSplitManifestError):
            sm.verify_manifest(tampered, CSV_PATH)

    def test_aggregate_hash_covers_every_field(self):
        tampered = copy.deepcopy(self.manifest)
        tampered["tasks"][3]["rubric_item_count"] = 999
        self.assertNotEqual(sm.aggregate_sha256(tampered), tampered["aggregate_sha256"])

    def test_emitted_manifest_on_disk_matches_the_builder(self):
        path = _HERE / "split_manifest.json"
        if not path.is_file():
            self.skipTest("split_manifest.json not built yet")
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(sm.verify_manifest(on_disk, CSV_PATH)["verified"], True)
        self.assertEqual(on_disk["aggregate_sha256"], self.manifest["aggregate_sha256"])


class TestFailClosedAgainstAdapter(unittest.TestCase):
    """The real E12 boundary must still be rejected: no license, no native receipts."""

    def _real_e12_boundary(self) -> dict:
        manifest = sm.build_manifest(CSV_PATH)
        ids = [t["task_id"] for t in manifest["tasks"]]
        return {
            "name": "appbench_eval",
            "authoritative_source": "https://huggingface.co/datasets/AfterQuery/App-Bench",
            "revision": manifest["source"]["revision"],
            "evaluation_role": "receipt_proven_heldout",
            "task_ids": ids,
            "split_hash": manifest["split"]["split_hash"],
            "license": None,
            "native_contract": None,
        }

    def test_adapter_rejects_the_real_boundary_for_missing_license(self):
        payload = {
            "paid_launch_allowed": False,
            "stateful_trajectory": True,
            "boundaries": {"E12": self._real_e12_boundary(), "E13": {}},
        }
        with self.assertRaises(adapter.PavlovAppbenchOpenrewardGamesAdapterError) as ctx:
            adapter.validate_appbench_openreward_games_contract(payload)
        self.assertIn("license", str(ctx.exception).lower())

    def test_real_task_ids_and_split_hash_satisfy_the_adapter_rules(self):
        """The parts this lane can close do pass the adapter's own checks."""

        boundary = self._real_e12_boundary()
        ids = adapter._validate_task_ids(boundary["task_ids"], "boundaries[E12]")
        self.assertEqual(len(ids), 6)
        self.assertEqual(
            adapter._validate_split_hash(boundary["split_hash"], ids, "boundaries[E12]"),
            boundary["split_hash"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
