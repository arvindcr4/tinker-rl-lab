"""Synthetic admission fixtures are not provider evidence or benchmark results."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import validate_v1 as v

REPO = Path(__file__).resolve().parents[2]


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        if shutil.disk_usage(REPO).free < v.FLOOR:
            self.skipTest("Disk floor: no fixture writes")
        self.temp = tempfile.TemporaryDirectory(dir=REPO / ".codex-run/finish_20260912/e12_admission")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.m = {"schema_version": "e12-offline-admission-v1", "suite_id": "appbench_eval",
                  "dataset_revision": v.REVISION, "scope": "original_heldout",
                  "model": {"id": "SYNTHETIC_TEST_MODEL", "revision": "a" * 40}, "assets": {}}
        for name, filename in (("csv", "hf_dataset/AppBench vExternal.csv"), ("split", "split_manifest.json"), ("disjointness", "disjointness_proof.json")):
            self.m["assets"][name] = self.put(name, (REPO / "outputs/e12_appbench" / filename).read_bytes())
        self.ids = [t["task_id"] for t in json.loads((self.root / "split").read_text())["tasks"]]
        self.binding = {"suite_id": "appbench_eval", "dataset_revision": v.REVISION,
                        "split_sha256": v.PINS["split"], "model": self.m["model"]}
        self.m["attempt"] = {"task_id": self.ids[0], "ordinal": 1, "run_id": "SYNTHETIC_ONLY"}
        self.docs = {
            "permission": {"issuer": "AfterQuery", "evaluation_allowed": True, "aggregate_publication_allowed": True},
            "holdout": {"issuer": "AfterQuery", "held_out_from_model_training": True, "task_ids": self.ids},
            "environment": {"issuer": "AfterQuery", "official_exact_environment": True, "image_digest": "sha256:" + "b" * 64},
            "grading": {"issuer": "AfterQuery", "graders": [{"id": "fixture1", "qualified_full_stack": True}, {"id": "fixture2", "qualified_full_stack": True}],
                        "independent_then_consensus": True, "binary_per_item": True, "attempts_per_task": 3,
                        "aggregation": "sum_best_of_three_per_task_over_151", "rubric_counts": [24, 33, 22, 25, 23, 24],
                        "website_23_vs_csv_24_resolved": True},
            "attempt_ledger": {"authoritative_complete": True, "attempts": [{"task_id": t, "ordinal": n, "state": "not_started"} for t in self.ids for n in (1, 2, 3)]},
        }
        for n in ("template", "runtime_reset", "deployment", "artifact_verification", "side_effect_verification", "credentials_policy"):
            self.docs["environment"][n] = self.put(n, b"SYNTHETIC ENVIRONMENT EVIDENCE - NOT AUTHENTIC")
        self.seal()

    def put(self, name, value):
        raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode()
        (self.root / name).write_bytes(raw)
        return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}

    def seal(self):
        for name, doc in self.docs.items():
            doc["binding"] = copy.deepcopy(self.binding)
            self.m["assets"][name] = self.put(name, doc)
        self.m["assets"]["parent_review"] = self.put("parent_review", {
            "binding": self.binding, "attempt": self.m["attempt"], "reviewer": "SYNTHETIC_REVIEW",
            "authenticity_and_history_verified": True,
            "evidence_sha256": {n: self.m["assets"][n]["sha256"] for n in v.EVIDENCE}})

    def result(self):
        return v.validate(self.m, self.root)

    def test_consistent_fixture_never_authorizes_launch(self):
        r = self.result()
        self.assertEqual(r["blockers"], [])
        self.assertFalse(r["launch_authorized"])
        self.assertIsNone(r["score"])

    def test_every_asset_required(self):
        for name in self.m["assets"]:
            m = copy.deepcopy(self.m)
            del m["assets"][name]
            with self.subTest(name=name):
                self.assertEqual(v.validate(m, self.root)["status"], "BLOCKED")

    def test_tampered_csv_and_environment(self):
        for name in ("csv", "deployment"):
            p = self.root / name
            original = p.read_bytes()
            p.write_bytes(original + b"tamper")
            self.assertEqual(self.result()["status"], "BLOCKED")
            p.write_bytes(original)

    def test_public_substitution_and_model_drift(self):
        for key, value in (("scope", "public"), ("dataset_revision", "c" * 40), ("model", {"id": "other", "revision": "b" * 40})):
            m = copy.deepcopy(self.m)
            m[key] = value
            self.assertEqual(v.validate(m, self.root)["status"], "BLOCKED")

    def test_permission_holdout_and_grading_fail_closed(self):
        for doc, key, bad in (("permission", "evaluation_allowed", False), ("holdout", "held_out_from_model_training", None),
                              ("environment", "official_exact_environment", False), ("grading", "website_23_vs_csv_24_resolved", False),
                              ("grading", "graders", [])):
            old = self.docs[doc][key]
            self.docs[doc][key] = bad
            self.seal()
            self.assertEqual(self.result()["status"], "BLOCKED")
            self.docs[doc][key] = old

    def test_replay_active_failed_completed_unknown(self):
        for state in ("active", "failed", "completed", "unknown"):
            self.docs["attempt_ledger"]["attempts"][0]["state"] = state
            self.seal()
            self.assertEqual(self.result()["status"], "BLOCKED")

    def test_missing_ledger_slot(self):
        self.docs["attempt_ledger"]["attempts"].pop()
        self.seal()
        self.assertIn("COMPLETE_18_SLOT_LEDGER", self.result()["blockers"])

    def test_reused_run_id(self):
        self.docs["attempt_ledger"]["attempts"][1]["run_id"] = "SYNTHETIC_ONLY"
        self.seal()
        self.assertIn("RUN_ID_ALREADY_USED", self.result()["blockers"])

    def test_disk_floor(self):
        with patch.object(v.shutil, "disk_usage", return_value=shutil._ntuple_diskusage(10, 9, 1)):
            self.assertEqual(self.result()["status"], "BLOCKED")

    def test_malformed_and_escape(self):
        for data in (None, [], {"assets": []}, {"assets": {"csv": {"path": "../escape", "sha256": "a" * 64}}}):
            self.assertEqual(v.validate(data, self.root)["status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
