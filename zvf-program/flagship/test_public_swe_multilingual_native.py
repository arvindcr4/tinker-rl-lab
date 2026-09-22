"""Offline contracts; temporary synthetic logs are never experiment results."""
import base64
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from flagship import public_swe_multilingual_native as m


def actor_identity():
    return {"model_id": "Qwen/Qwen3.6-35B-A3B", "model_revision": "a" * 40,
            "hf_repo": "unit-test/adapter", "hf_commit": "b" * 40, "served_model_id": "unit-fixture"}


def image_row(row):
    raw = m.canonical({"schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json", "layers": []})
    sha = m.digest(raw)
    return {"instance_id": row["instance_id"], "source_tag": row["image"], "platform": "linux/amd64",
            "image": row["image"].rsplit(":", 1)[0] + "@sha256:" + sha,
            "manifest_sha256": sha, "registry_manifest_base64": base64.b64encode(raw).decode(),
            "resolved_at": "UNIT_TEST_ONLY"}


class ContractTests(unittest.TestCase):
    def test_actor_projection_excludes_gold_test_and_hints(self):
        raw = {"instance_id": "a__b-1", "repo": "a/b", "base_commit": "c" * 40,
               "problem_statement": "Fix it", "patch": "SECRET GOLD", "test_patch": "SECRET TEST",
               "FAIL_TO_PASS": ["hidden"], "hints_text": "HINT"}
        public = m.actor_task(raw)
        self.assertEqual(set(public), set(m.ACTOR_FIELDS))
        request = m.build_actor_request(public, {"main.go": "package main"}, "actor", max_tokens=512, temperature=0, seed=1)
        self.assertNotIn("SECRET", json.dumps(request))
        with self.assertRaises(m.BoundaryError):
            m.build_actor_request(raw, {}, "actor", max_tokens=512, temperature=0, seed=1)

    def test_actor_source_path_and_sampling_bounds(self):
        actor = dict(zip(m.ACTOR_FIELDS, ["a__b-1", "a/b", "c" * 40, "Fix it"]))
        for sources, cap in [({"../../gold.py": "x"}, 512), ({"/etc/file": "x"}, 512), ({}, 0)]:
            with self.assertRaises(m.BoundaryError):
                m.build_actor_request(actor, sources, "actor", max_tokens=cap, temperature=0, seed=1)

    def test_identity_and_duplicate_guards(self):
        identity = actor_identity()
        identity["hf_commit"] = "main"
        with self.assertRaises(m.BoundaryError):
            m.validate_identity(identity)
        with self.assertRaises(m.BoundaryError):
            m.indexed([{"instance_id": "same"}, {"instance_id": "same"}])
        with self.assertRaises(m.BoundaryError):
            m.indexed([{"instance_id": "../escape"}])

    def test_immutable_artifacts_reject_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "receipt.json"
            m.write(p, {"a": 1})
            m.write(p, {"a": 1})
            with self.assertRaises(m.BoundaryError):
                m.write(p, {"a": 2})

    def test_images_require_manifest_bytes_and_matching_native_tag(self):
        task = {"instance_id": "a__b-1", "image": "swebench/eval:latest"}
        row = image_row(task)
        m.validate_images({"images": [row]}, {task["instance_id"]: task})
        for field, value in [("source_tag", "other:latest"), ("platform", "linux/arm64"),
                             ("registry_manifest_base64", "e30="), ("image", "swebench/eval:latest")]:
            changed = {**row, field: value}
            with self.assertRaises(m.BoundaryError):
                m.validate_images({"images": [changed]}, {task["instance_id"]: task})

    def make_attempt(self, directory):
        public = dict(zip(m.ACTOR_FIELDS, ["a__b-1", "a/b", "c" * 40, "Fix it"]))
        task = {**public, "actor_task_sha256": m.object_hash(public)}
        manifest = {"tasks": [task], "task_inventory_sha256": "unit-inventory"}
        identity = actor_identity()
        folder = directory / public["instance_id"]
        folder.mkdir()
        source = {"repo": public["repo"], "base_commit": public["base_commit"], "actor_task": public, "files": {"a.go": "old"}}
        request = m.build_actor_request(public, source["files"], identity["served_model_id"], max_tokens=512, temperature=0, seed=1)
        m.write(folder / "generation_request.json", request)
        m.write(folder / "source_context.json", source)
        patch_text = "diff --git a/a.go b/a.go\n--- a/a.go\n+++ b/a.go\n@@ -1 +1 @@\n-old\n+new\n"
        (folder / "generation_response.txt").write_text(patch_text)
        receipt = {"instance_id": public["instance_id"], "status": "GENERATED", "sample_started": True,
                   "sample_completed": True, "task_inventory_sha256": "unit-inventory",
                   "actor_task_sha256": task["actor_task_sha256"], "model_identity_sha256": m.object_hash(identity),
                   "wandb_run_id": "UNIT_TEST_ONLY", "wandb_mode": "online", "started_at": "UNIT", "finished_at": "UNIT",
                   "response_sha256": m.digest(patch_text.encode()), "source_sha256": m.object_hash(source["files"]),
                   "request_sha256": m.digest((folder / "generation_request.json").read_bytes()),
                   "patch": patch_text, "patch_sha256": m.digest(patch_text.encode())}
        m.write(folder / "generation.json", receipt)
        return manifest, identity, folder, receipt

    def test_all_attempts_required_and_ambiguous_receipt_not_resampled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, identity, folder, receipt = self.make_attempt(root)
            predictions, receipts = m.collect_attempts(manifest, root, identity)
            self.assertEqual(len(predictions), 1)
            self.assertEqual(len(receipts), 1)
            for field, value in [("sample_completed", False), ("status", "GENERATION_ARTIFACT_LOST"),
                                 ("model_identity_sha256", "other"), ("wandb_mode", "offline")]:
                (folder / "generation.json").write_text(json.dumps({**receipt, field: value}))
                with self.assertRaises(m.BoundaryError):
                    m.collect_attempts(manifest, root, identity)
            (folder / "generation.json").unlink()
            with self.assertRaises(m.BoundaryError):
                m.collect_attempts(manifest, root, identity)

    def test_prompt_and_response_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, identity, folder, receipt = self.make_attempt(root)
            request = m.read(folder / "generation_request.json")
            request["messages"][1]["content"] += "\nSECRET EVAL PATCH"
            (folder / "generation_request.json").write_text(json.dumps(request))
            receipt["request_sha256"] = m.digest((folder / "generation_request.json").read_bytes())
            (folder / "generation.json").write_text(json.dumps(receipt))
            with self.assertRaises(m.BoundaryError):
                m.collect_attempts(manifest, root, identity)

    def test_failed_completed_response_is_explicit_native_empty_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, identity, folder, receipt = self.make_attempt(root)
            receipt.update(status="GENERATION_FAILED", patch="", patch_sha256=None)
            (folder / "generation.json").write_text(json.dumps(receipt))
            predictions, _ = m.collect_attempts(manifest, root, identity)
            self.assertEqual(predictions[0]["model_patch"], "")


@unittest.skipUnless(Path(sys.prefix).resolve() == (m.DEFAULT_SETUP / ".venv").resolve()
                     or os.environ.get("SWE_MULTILINGUAL_NATIVE_OFFLINE_TESTS") == "1",
                     "Native source checks require the separately installed pinned runtime")
class NativeOfflineTests(unittest.TestCase):
    def test_actual_300_tasks_native_specs_and_immutable_reprepare(self):
        manifest, raw = m.load_prepared(m.DEFAULT_SETUP)
        utils, _ = m.native(m.DEFAULT_SETUP)
        specs = m.indexed(m.lines(m.DEFAULT_SETUP / "prepared/native_test_specs.jsonl"))
        actors = m.lines(m.DEFAULT_SETUP / "prepared/actor_tasks.jsonl")
        self.assertEqual(len(raw), 300)
        self.assertTrue(all(set(row) == set(m.ACTOR_FIELDS) for row in actors))
        for iid, row in raw.items():
            spec = utils.make_test_spec(row)
            self.assertEqual(m.object_hash({**m.asdict(spec), "eval_script": spec.eval_script}), m.object_hash(specs[iid]))
        result = m.prepare(m.DEFAULT_SETUP)
        self.assertIsNone(result["score"])
        self.assertEqual(result["task_inventory_sha256"], manifest["task_inventory_sha256"])

    def test_source_modification_is_detected_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / "source_manifest.json").write_bytes((m.DEFAULT_SETUP / "source_manifest.json").read_bytes())
            with self.assertRaises(m.BoundaryError):
                m.verify_sources(target)

    def test_native_log_fixture_full_coverage_and_partial_evidence_gate(self):
        """One generated test log +299 empty fixtures; no model or Docker run."""
        manifest, raw = m.load_prepared(m.DEFAULT_SETUP)
        utils, grading = m.native(m.DEFAULT_SETUP)
        from swebench.harness import reporting
        identity = actor_identity()
        chosen = next(row for row in raw.values() if row["log_parser"] == "parse_log_gotest")
        iid = chosen["instance_id"]
        predictions = [{"instance_id": key, "model_patch": "unit patch" if key == iid else "", "model_name_or_path": identity["served_model_id"]} for key in raw]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "native-fixture"
            identity_path = Path(directory) / "identity.json"
            images_path = Path(directory) / "images.json"
            m.write(identity_path, identity)
            images = {"images": [image_row(row) for row in raw.values()]}
            m.write(images_path, images)
            attempt_index = [{"instance_id": row["instance_id"], "status": "GENERATED" if row["model_patch"] else "GENERATION_FAILED",
                              "generation_sha256": "a" * 64, "response_sha256": "b" * 64, "wandb_run_id": "UNIT_TEST_ONLY",
                              "patch_sha256": m.digest(row["model_patch"].encode())} for row in predictions]
            with patch.object(m, "collect_attempts", return_value=(predictions, attempt_index)):
                plan = m.plan(m.DEFAULT_SETUP, out, Path(directory) / "attempts", identity_path, images_path, "unit-fixture")
            self.assertNotIn("--gold", plan["argv"])
            self.assertNotIn("multilingual", plan["argv"])
            native_root = out / "logs/run_evaluation/unit-fixture"
            task_dir = native_root / identity["served_model_id"] / iid
            task_dir.mkdir(parents=True)
            m.write(native_root / "run.json", {"dataset": str((out / "runtime_dataset.jsonl").resolve()), "split": "test", "task_repo": None})
            spec = utils.make_test_spec(chosen)
            test_log = ">>>>> Start Test Output\n" + "\n".join(f"--- PASS: {name} (0.01s)" for name in spec.FAIL_TO_PASS + spec.PASS_TO_PASS) + "\n>>>>> End Test Output\n>>>>> Test Exit Code: 0\n"
            (task_dir / "test_output.txt").write_text(test_log)
            (task_dir / "patch.diff").write_text("unit patch")
            (task_dir / "eval.sh").write_text(spec.eval_script)
            (task_dir / "run_instance.log").write_text("UNIT TEST FIXTURE: no container was started")
            prediction = next(row for row in predictions if row["instance_id"] == iid)
            report = grading.get_eval_report(spec, prediction, str(task_dir / "test_output.txt"), True)
            self.assertTrue(report[iid]["resolved"])
            m.write(task_dir / "report.json", report)
            with patch.object(reporting, "RUN_EVALUATION_LOG_DIR", out / "logs/run_evaluation"), redirect_stdout(io.StringIO()):
                reporting.make_run_report(m.indexed(predictions), list(raw.values()), "unit-fixture", client=None, report_dir=str(out / "reports"))
            receipt = {"plan_sha256": m.digest((out / "native_plan.json").read_bytes()), "returncode": 0,
                       "started_at": "UNIT_TEST_ONLY", "finished_at": "UNIT_TEST_ONLY", "wandb_mode": "online", "wandb_run_id": "UNIT_TEST_ONLY",
                       "native_artifacts": {str(path.relative_to(out)): m.digest(path.read_bytes()) for root in [out / "logs", out / "reports"] for path in root.rglob("*") if path.is_file()},
                       "image_observations": {iid: image_row(chosen)["image"]}}
            execution_path = out / "execution_fixture.json"
            m.write(execution_path, receipt)
            result = m.ingest(m.DEFAULT_SETUP, out, execution_path)
            self.assertEqual(result["score"], 1 / 300)
            self.assertEqual(result["completed_attempts"], 300)
            # An interrupted process cannot be promoted from its native logs.
            changed = copy.deepcopy(receipt)
            changed["returncode"] = 1
            execution_path.write_text(json.dumps(changed))
            self.assertIsNone(m.ingest(m.DEFAULT_SETUP, out, execution_path)["score"])
            # Rehashing a forged native report cannot bypass the actual grader.
            forged = copy.deepcopy(report)
            forged[iid]["resolved"] = False
            (task_dir / "report.json").write_text(json.dumps(forged))
            changed["native_artifacts"][str((task_dir / "report.json").relative_to(out))] = m.digest((task_dir / "report.json").read_bytes())
            execution_path.write_text(json.dumps(changed))
            with self.assertRaises(m.BoundaryError):
                m.ingest(m.DEFAULT_SETUP, out, execution_path)


if __name__ == "__main__":
    unittest.main()
