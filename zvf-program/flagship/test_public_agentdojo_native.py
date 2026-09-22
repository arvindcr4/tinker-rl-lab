"""Offline contract tests; oracle fixture results are never benchmark scores."""
import argparse
import copy
import contextlib
import importlib
import json
import os
import sys
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from flagship import public_agentdojo_native as native


def identity():
    return {"model_id": "actor", "model_revision": "a" * 40, "hf_repo": "owner/adapter",
            "hf_commit": "b" * 40, "served_model_id": "served-actor"}


def task_rows():
    return [{"suite_name": name, "task_id": f"user_task_{index}",
             "evaluation_id": f"{name}/user_task_{index}", "prompt_sha256": f"prompt-{name}-{index}",
             "initial_environment_sha256": "fixture", "native_class": "fixture"}
            for name, count in native.COUNTS.items() for index in range(count)]


def run_manifest():
    rows = task_rows()
    return {"schema_version": native.SCHEMA, "protocol": native.PROTOCOL, "tasks": rows,
            "source_manifest_sha256": native.SOURCE_MANIFEST_SHA256,
            "task_inventory_sha256": native.fingerprint(rows),
            "model_identity": identity(), "decontamination": {"status": "TRAINING_INVENTORY_ABSENT"}}


def final_trace(row):
    return {"suite_name": row["suite_name"], "user_task_id": row["task_id"],
            "pipeline_name": "openai-compatible", "benchmark_version": native.VERSION,
            "attack_type": None, "injection_task_id": None, "injections": {},
            "messages": [{"role": "user", "content": []}, {"role": "assistant", "content": []}],
            "utility": True, "security": True, "duration": 0.001, "error": None}


class ContractTests(unittest.TestCase):
    def test_actor_identity_rejects_mutable_or_missing_bindings(self):
        native.validate_identity(identity())
        for key, value in [("hf_commit", "main"), ("model_revision", "latest"), ("served_model_id", "")]:
            changed = identity()
            changed[key] = value
            with self.assertRaises(native.ContractError):
                native.validate_identity(changed)

    def test_source_inventory_cannot_be_replaced_or_truncated(self):
        with self.assertRaises(native.ContractError):
            native.source_manifest({"truncated": True, "tree": []})
        with self.assertRaises(native.ContractError):
            native.source_manifest({"truncated": False, "tree": []})

    def test_proxy_headers_are_scoped_to_current_episode_and_run(self):
        transport = mock.Mock()
        llm = types.SimpleNamespace(client=transport)
        pipeline = types.SimpleNamespace(elements=[object(), object(), llm, object()])
        for task_id in ["banking/user_task_0", "banking/user_task_1"]:
            native.bind_task_header(pipeline, transport, task_id, "run-hash")
            transport.with_options.assert_called_with(default_headers={
                "X-Public-Task-ID": task_id, "X-Public-Run-ID": "run-hash"})
            self.assertIs(llm.client, transport.with_options.return_value)

    def test_overlap_reports_ids_and_content_even_with_different_names(self):
        manifest = {"tasks": task_rows()}
        training = {"complete": True, "model_identity_sha256": "bound",
                    "tasks": [{"suite_name": "banking", "task_id": "user_task_0"},
                              {"prompt_sha256": "prompt-slack-2", "evaluation_id": "renamed/task"}]}
        report = native.overlap_report(manifest, training, "bound")
        self.assertEqual(report["status"], "OVERLAP_FOUND")
        self.assertEqual(report["matching_evaluation_ids"], ["banking/user_task_0"])
        self.assertEqual(report["matching_prompt_hashes"], ["prompt-slack-2"])
        self.assertFalse(report["heldout_claim"])

    def test_absent_partial_or_wrong_model_training_is_not_a_disjointness_proof(self):
        manifest = {"tasks": task_rows()}
        self.assertEqual(native.overlap_report(manifest)["status"], "TRAINING_INVENTORY_ABSENT")
        training = {"complete": True, "model_identity_sha256": "other", "tasks": []}
        report = native.overlap_report(manifest, training, "this")
        self.assertEqual(report["status"], "NO_MATCH_IN_INCOMPLETE_OR_UNBOUND_MANIFEST")
        self.assertFalse(report["heldout_claim"])

    def test_atomic_receipt_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            native.write_once(path, {"value": 1})
            native.write_once(path, {"value": 1})
            with self.assertRaises(native.ContractError):
                native.write_once(path, {"value": 2})

    def test_partial_coverage_keeps_score_null_and_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            manifest = run_manifest()
            native.write_once(output / "run_manifest.json", manifest)
            row = manifest["tasks"][0]
            path = native.trace_path(output, row)
            native.write_once(path, final_trace(row))
            receipt = native.task_result(output, row, native.fingerprint(manifest))
            native.write_once(output / "receipts" / row["suite_name"] / (row["task_id"] + ".json"), receipt)
            report = native.summarize(output)
            self.assertIsNone(report["score"])
            self.assertEqual(report["completed_episodes"], 1)
            self.assertEqual(len(report["missing_evaluation_ids"]), 96)
            modified = final_trace(row)
            modified["utility"] = False
            path.write_text(json.dumps(modified))
            with self.assertRaises(native.ContractError):
                native.summarize(output)

    def test_duplicate_task_ids_never_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = run_manifest()
            manifest["tasks"][1] = copy.deepcopy(manifest["tasks"][0])
            manifest["task_inventory_sha256"] = native.fingerprint(manifest["tasks"])
            native.write_once(Path(tmp) / "run_manifest.json", manifest)
            with self.assertRaises(native.ContractError):
                native.summarize(tmp)

    def test_incomplete_or_attacked_native_trace_cannot_be_recovered(self):
        row = task_rows()[0]
        mutations = [("utility", None), ("duration", None), ("security", None),
                     ("attack_type", "tool_knowledge"), ("benchmark_version", "v1"),
                     ("pipeline_name", "another-model"), ("messages", [])]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            for key, value in mutations:
                trace = final_trace(row)
                trace[key] = value
                path.write_text(json.dumps(trace))
                with self.subTest(key=key), self.assertRaises(native.ContractError):
                    native.validate_native_trace(path, row)

    def test_offline_wandb_blocks_before_native_pipeline_or_model_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            native.write_once(directory / "identity.json", identity())
            manifest = {"tasks": task_rows(), "task_inventory_sha256": native.fingerprint(task_rows())}
            args = argparse.Namespace(setup=directory, output=directory / "results",
                                      model_identity=directory / "identity.json", training_manifest=None,
                                      wandb_entity="entity", wandb_project="project", wandb_run_id="run",
                                      max_tasks=1)
            tracking = types.SimpleNamespace(settings=types.SimpleNamespace(mode="offline"), id="run",
                                             finish=mock.Mock())
            wandb = types.SimpleNamespace(init=mock.Mock(return_value=tracking))
            with mock.patch.object(native, "prepare", return_value=manifest), \
                 mock.patch.object(native, "load_native") as load, \
                 mock.patch.dict(sys.modules, {"wandb": wandb}), \
                 mock.patch.dict(os.environ, {"OPENAI_COMPATIBLE_BASE_URL": "http://localhost/v1",
                                             "OPENAI_COMPATIBLE_API_KEY": "fixture"}):
                with self.assertRaises(native.ContractError):
                    native.run(args)
                load.assert_not_called()
                tracking.finish.assert_called_once()

    def test_budget_failure_stays_partial_and_resume_never_resamples_ambiguous_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            native.write_once(directory / "identity.json", identity())
            manifest = {"tasks": task_rows(), "task_inventory_sha256": native.fingerprint(task_rows())}
            args = argparse.Namespace(setup=directory, output=directory / "results",
                                      model_identity=directory / "identity.json", training_manifest=None,
                                      wandb_entity="entity", wandb_project="project", wandb_run_id="run",
                                      max_tasks=1)
            tracking = types.SimpleNamespace(settings=types.SimpleNamespace(mode="online"), id="run",
                                             url="https://wandb.example/run", log=mock.Mock(),
                                             summary={}, finish=mock.Mock())
            wandb = types.SimpleNamespace(init=mock.Mock(return_value=tracking))
            client = mock.Mock()
            pipeline = types.SimpleNamespace(elements=[object(), object(), types.SimpleNamespace(client=client), object()])
            pipeline_module = types.SimpleNamespace(
                AgentPipeline=types.SimpleNamespace(from_config=mock.Mock(return_value=pipeline)),
                PipelineConfig=lambda **kwargs: kwargs)
            benchmark = types.SimpleNamespace(run_task_without_injection_tasks=mock.Mock(side_effect=RuntimeError("budget exhausted")))
            logger = types.SimpleNamespace(OutputLogger=lambda path: contextlib.nullcontext())
            modules = {"agentdojo.agent_pipeline.agent_pipeline": pipeline_module,
                       "agentdojo.benchmark": benchmark, "agentdojo.logging": logger}
            with mock.patch.object(native, "prepare", return_value=manifest), \
                 mock.patch.object(native, "load_native", return_value={"banking": mock.Mock()}), \
                 mock.patch.object(native.importlib, "import_module", side_effect=lambda name: modules[name]), \
                 mock.patch.dict(sys.modules, {"wandb": wandb}), \
                 mock.patch.dict(os.environ, {"OPENAI_COMPATIBLE_BASE_URL": "http://localhost/v1",
                                             "OPENAI_COMPATIBLE_API_KEY": "fixture"}):
                with self.assertRaises(RuntimeError):
                    native.run(args)
                report = native.summarize(args.output)
                self.assertIsNone(report["score"])
                self.assertEqual(report["completed_episodes"], 0)
                self.assertTrue((args.output / "intents/banking/user_task_0.json").exists())
                self.assertEqual(len(list((args.output / "failures").glob("*.json"))), 1)
                benchmark.run_task_without_injection_tasks.reset_mock()
                with self.assertRaisesRegex(native.ContractError, "ambiguous task intent"):
                    native.run(args)
                benchmark.run_task_without_injection_tasks.assert_not_called()


@unittest.skipUnless(
    (native.DEFAULT_SETUP / "source_receipt.json").exists()
    and (Path(sys.prefix).resolve() == (native.DEFAULT_SETUP / ".venv").resolve()
         or os.environ.get("AGENTDOJO_NATIVE_OFFLINE_TESTS") == "1"),
    "use isolated AgentDojo runtime or set AGENTDOJO_NATIVE_OFFLINE_TESTS=1")
class NativeOfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suites = native.load_native(native.DEFAULT_SETUP)

    def test_native_version_loads_all97_unique_tasks_and_reproducible_environments(self):
        first = native.inventory(self.suites)
        second = native.inventory(self.suites)
        self.assertEqual(len(first), 97)
        self.assertEqual(native.fingerprint(first), native.fingerprint(second))

    def test_native_oracle_fixture_uses_official_environment_utility_and_trace(self):
        # One oracle fixture verifies plumbing. It is not a model sample or suite result.
        pipeline_mod = importlib.import_module("agentdojo.agent_pipeline.agent_pipeline")
        oracle_cls = importlib.import_module("agentdojo.agent_pipeline.ground_truth_pipeline").GroundTruthPipeline
        benchmark = importlib.import_module("agentdojo.benchmark")
        logger_mod = importlib.import_module("agentdojo.logging")
        suite = self.suites["banking"]
        task = suite.get_user_task_by_id("user_task_0")
        oracle = oracle_cls(task)
        oracle.name = "openai-compatible"
        pipeline = pipeline_mod.AgentPipeline.from_config(pipeline_mod.PipelineConfig(
            llm=oracle, model_id=None, defense=None, system_message=None, system_message_name=None))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with logger_mod.OutputLogger(str(output / "native")):
                utility, _ = benchmark.run_task_without_injection_tasks(
                    suite, pipeline, task, output / "native", False, native.VERSION)
            row = {"suite_name": "banking", "task_id": task.ID}
            trace = native.validate_native_trace(native.trace_path(output, row), row)
            self.assertTrue(utility)
            self.assertIs(trace["utility"], utility)
            # The untouched benchmark can reload the same finalized native trace.
            with logger_mod.OutputLogger(str(output / "native")), \
                 mock.patch.object(pipeline, "query", side_effect=AssertionError("oracle resampled")):
                cached, _ = benchmark.run_task_without_injection_tasks(
                    suite, pipeline, task, output / "native", False, native.VERSION)
            self.assertTrue(cached)


if __name__ == "__main__":
    unittest.main()
