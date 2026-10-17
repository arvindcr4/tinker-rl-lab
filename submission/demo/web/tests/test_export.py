"""Exercise the publication exporter against immutable history and canonical analysis."""
import gzip
import importlib.util
import pathlib
import json
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_demo_web", HERE / "build_demo_web.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def synthetic_bundle(directory, record, texts):
    """Tiny explicitly fake files exercise the real completion validator; no model runs."""
    from platform_hybrid.experiments.samestack_contract import write_completion
    name = f"{record['arm']}_s{record['seed']}"
    path = directory / f"{name}.json"
    path.write_text(json.dumps(record, allow_nan=False))
    with gzip.open(directory / f"{name}_texts.json.gz", "wt") as stream:
        json.dump(texts, stream)
    adapter = directory / "adapters" / name
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"SYNTHETIC TEST ONLY: NOT MODEL WEIGHTS")
    for filename in ("adapter_config.json", "tokenizer_config.json", "tokenizer.json"):
        (adapter / filename).write_text('{"synthetic":true}')
    for phase, config in record["manifest"]["generation"].items():
        (adapter / f"{phase}_generation_config.json").write_text(json.dumps(config))
    write_completion(path)
    return path


class ExportTests(unittest.TestCase):
    def test_historical_inputs_are_unverified_not_fabricated(self):
        before = {p: export.sha(p) for p in export.V2.glob("*.json")}
        ss, analysis = export.samestack()
        self.assertEqual(analysis["status"], "interim")
        self.assertFalse(analysis["final_eligible"])
        self.assertEqual(analysis["completion"]["selected"], 0)
        self.assertTrue(ss["runs"])
        self.assertTrue(all(r["run_id"] is None for r in ss["runs"]))
        self.assertEqual(list(analysis["primary_contrasts"]), ["g8-ppo", "g8-ppo_8x8", "g8-g2"])
        self.assertTrue(all(c["holm_tost_p"] == 1 for c in analysis["primary_contrasts"].values()))
        self.assertEqual(before, {p: export.sha(p) for p in before})

    def test_verified_manifest_is_preserved_and_normative_selector_is_used(self):
        sys.path.insert(0, str(export.ROOT / "tests"))
        from samestack_fixtures import synthetic_run
        with tempfile.TemporaryDirectory() as tmp, patch.object(export, "V2", pathlib.Path(tmp)):
            records = []
            for arm in export.ARMS:
                for seed in (42, 123):
                    record, texts = synthetic_run(arm, seed, smoke=False)
                    records.append(record)
                    synthetic_bundle(pathlib.Path(tmp), record, texts)
            plan = export.default_plan()
            plan.update({"planned_seeds": [42, 123], "endpoint": {"declared": True, "description": "synthetic fixture"},
                         "shared_stack_hash": export.shared_stack_fingerprint(records[0]),
                         **{key: records[0]["manifest"][key] for key in ("environment", "environment_hash", "protocol_hash", "ordered_test_item_hash")}})
            ss, analysis = export.samestack(plan)
            self.assertEqual(analysis["completion"], {"planned": 8, "selected": 8, "missing": 0})
            self.assertEqual(analysis["status"], "endpoint_complete")
            self.assertFalse(analysis["final_eligible"])
            for row in ss["runs"]:
                self.assertEqual(row["run_id"], row["manifest"]["run_id"])
                self.assertEqual(row["protocol_hash"], row["manifest"]["protocol_hash"])
                self.assertEqual(row["environment_hash"], row["manifest"]["environment_hash"])
                self.assertEqual(row["shared_stack_hash"], plan["shared_stack_hash"])
                self.assertTrue(row["completion_bundle_validated"])
            self.assertEqual(analysis["primary_contrasts"]["g8-ppo"]["diffs"], [0, 0])
            adapter = (HERE / "src/analysis.js").as_uri()
            script = f"import {{analysisModel}} from {json.dumps(adapter)}; import fs from 'node:fs'; const m=analysisModel(JSON.parse(fs.readFileSync(0,'utf8'))); console.log(JSON.stringify({{selected:m.selectedRuns.length,valid:m.selectionValid,endpoint:m.endpointComplete,final:m.complete,n:m.contrasts[0].stats?.n}}));"
            result = subprocess.run(["node", "--input-type=module", "-e", script],
                                    input=json.dumps({"samestack": ss, "analysis": analysis}),
                                    text=True, capture_output=True, check=True)
            self.assertEqual(json.loads(result.stdout), {"selected": 8, "valid": True, "endpoint": True, "final": False, "n": 2})

    def test_missing_and_corrupt_completion_bundles_are_excluded(self):
        sys.path.insert(0, str(export.ROOT / "tests"))
        from samestack_fixtures import synthetic_run
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            first = None
            for arm in export.ARMS:
                for seed in (42, 123):
                    record, texts = synthetic_run(arm, seed, smoke=False)
                    first = first or record
                    synthetic_bundle(directory, record, texts)
            (directory / "grpo_g8_s42.complete.json").unlink()
            (directory / "adapters/ppo_s123/adapter_model.safetensors").write_bytes(b"CORRUPTED SYNTHETIC WEIGHTS")
            plan = export.default_plan()
            plan.update({"planned_seeds": [42, 123], "endpoint": {"declared": True, "description": "synthetic fixture"},
                         "shared_stack_hash": export.shared_stack_fingerprint(first),
                         **{key: first["manifest"][key] for key in ("environment", "environment_hash", "protocol_hash", "ordered_test_item_hash")}})
            ss, analysis = export.samestack(plan, directory)
            self.assertEqual(analysis["completion"], {"planned": 8, "selected": 6, "missing": 2})
            self.assertEqual(sum(r["completion_bundle_validated"] for r in ss["runs"]), 6)
            self.assertEqual(analysis["status"], "interim")
            self.assertEqual(analysis["primary_contrasts"]["g8-ppo"]["n"], 0)
            failures = [r for r in ss["runs"] if r["completion_failure"]]
            self.assertEqual(len(failures), 2)
            self.assertTrue(any("inventory/hash mismatch" in r["completion_failure"] for r in failures))

    def test_no_records_remains_a_complete_export_shape(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(export, "V2", pathlib.Path(tmp)):
            ss, analysis = export.samestack()
            self.assertEqual(ss["runs"], [])
            self.assertIsNone(ss["setup"]["model"])
            self.assertEqual(analysis["completion"]["selected"], 0)
            ex = export.examples()
            self.assertEqual(ex["items"], [])
            self.assertTrue(ex["missing"])

    def test_provenance_chain_names_inputs_and_missing_coverage(self):
        ss, analysis = export.samestack()
        data = {"samestack": ss, "analysis": analysis, "examples": {}, "grader_check": {}}
        manifest = export.provenance(data)
        roles = {s["role"] for s in manifest["sources"]}
        self.assertTrue({"immutable run record", "completion sidecar", "ordered gold questions and answers",
                         "offline analysis implementation", "interactive presentation implementation", "exporter",
                         "registry input entry", "registry comparison implementation", "audit summary"} <= roles)
        self.assertTrue(all(len(s["sha256"]) == 64 for s in manifest["sources"]))
        self.assertTrue(any("unknown run_id" in m for m in manifest["missing"]))
        self.assertEqual(len(manifest["analysis_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
