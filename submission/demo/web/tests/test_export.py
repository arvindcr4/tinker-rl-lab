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
    @staticmethod
    def passing_report():
        return dict(grading_spec_id="boxed-exact-rational-v2", historical_scores_mutated=False,
                    runs=1, runs_checked=1, expected_items_per_phase=2, completions=4,
                    disagreements=0, stored_correct_now_incorrect=0, stored_incorrect_now_correct=0,
                    coverage_complete=True, missing_sidecars=[], invalid_inputs=[],
                    length_mismatches=[], invalid_scores=[])

    def test_grader_check_requires_successful_process_and_valid_report(self):
        report = self.passing_report()
        with patch.object(export.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, json.dumps(report), "")) as run:
            self.assertEqual(export.grader_check()["audit_exit_code"], 0)
            self.assertNotIn("--report-only", run.call_args.args[0])
        for code, stdout in [(1, json.dumps(report)), (2, json.dumps(report)), (0, "not json"), (0, "[]"), (0, "null")]:
            with self.subTest(code=code, stdout=stdout), patch.object(export.subprocess, "run", return_value=subprocess.CompletedProcess([], code, stdout, "failure")):
                with self.assertRaises(RuntimeError):
                    export.grader_check()
        with patch.object(export.subprocess, "run", side_effect=OSError("node unavailable")):
            with self.assertRaisesRegex(RuntimeError, "could not run"):
                export.grader_check()

    def test_grader_schema_rejects_disagreements_even_with_exit_zero(self):
        changes = [dict(disagreements=1), dict(stored_correct_now_incorrect=1),
                   dict(coverage_complete=False), dict(completions=3), dict(runs_checked=0),
                   dict(runs=0, runs_checked=0, completions=0), dict(expected_items_per_phase=0, completions=0),
                   dict(disagreements=False), dict(disagreements="0"), dict(invalid_inputs=["bad"]),
                   dict(historical_scores_mutated=True), dict(grading_spec_id="old")]
        for change in changes:
            report = {**self.passing_report(), **change}
            with self.subTest(change=change), patch.object(export.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, json.dumps(report), "")):
                with self.assertRaises(RuntimeError):
                    export.grader_check()

    def test_failed_audit_never_overwrites_existing_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = pathlib.Path(tmp) / "data.json"
            target.write_text("existing evidence")
            with patch.object(sys, "argv", ["build_demo_web.py", "--output", str(target)]), \
                    patch.object(export, "samestack", return_value=({}, {})), \
                    patch.object(export, "samestack_v1", return_value=None), \
                    patch.object(export, "examples", return_value={}), \
                    patch.object(export.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "disagreement", "")):
                with self.assertRaisesRegex(RuntimeError, "audit failed"):
                    export.main()
            self.assertEqual(target.read_text(), "existing evidence")

    def test_examples_preserve_gold_grouping_for_the_exact_grader(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            gold_path = directory / "gold.json"
            for arm in ("grpo_g8", "ppo"):
                record = dict(pre_correct=[1], post_correct=[1], pre_trunc=[0], post_trunc=[0])
                (directory / f"{arm}_s42.json").write_text(json.dumps(record))
                with gzip.open(directory / f"{arm}_s42_texts.json.gz", "wt") as stream:
                    json.dump(dict(pre=[r"\boxed{42}"], post=[r"\boxed{42}"]), stream)
            for answer in ("1,234", "42.0,000"):
                with self.subTest(answer=answer), patch.object(export, "GOLD", gold_path):
                    gold_path.write_text(json.dumps({"test": [{"question": "synthetic", "answer": "#### " + answer}]}))
                    examples = export.examples(seed=42, results_dir=directory)
                    self.assertEqual(examples["items"][0]["gold"], answer)

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
                    leaf = pathlib.Path(tmp) / f"{arm}_s{seed}"
                    leaf.mkdir()
                    synthetic_bundle(leaf, record, texts)
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
