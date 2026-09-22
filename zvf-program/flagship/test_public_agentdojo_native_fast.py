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
import threading
import types
import unittest
from unittest import mock

from flagship import public_agentdojo_native_fast as native


def identity():
    return {"model_id": "actor", "model_revision": "a" * 40, "hf_repo": "owner/adapter",
            "hf_commit": "b" * 40, "served_model_id": "served-actor", "runtime_limits": {"max_context_tokens":32768,"per_request_max_output_tokens":4096}}


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
    def test_loopback_route_restrictions_and_stable_logical_identity(self):
        for url in ['http://127.0.0.1:1234/v1','http://localhost:5678/v1/','http://[::1]:3333/v1']:
            self.assertEqual(native.validate_loopback_endpoint(url),url.rstrip('/'))
        for url in ['https://localhost:123/v1','http://example.com:123/v1','http://127.0.0.1/v1',
                    'http://127.0.0.1:0/v1','http://127.0.0.1:70000/v1','http://@127.0.0.1:123/v1',
                    'http://user:pass@localhost:123/v1','http://localhost:123/v1?x=1','http://localhost:123/v1#x',
                    'http://localhost:123/v2',' http://localhost:123/v1']:
            with self.subTest(url=url),self.assertRaises(native.ContractError):native.validate_loopback_endpoint(url)
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'runtime.py';source.write_text('fixture source v1')
            first=native.logical_endpoint_identity(identity(),source)
            with mock.patch.dict(os.environ,{'OPENAI_COMPATIBLE_BASE_URL':'http://127.0.0.1:9999/v1'}):
                self.assertEqual(first,native.logical_endpoint_identity(identity(),source))
            changed=identity();changed['hf_commit']='c'*40
            self.assertNotEqual(first,native.logical_endpoint_identity(changed,source))
            source.write_text('fixture source v2')
            self.assertNotEqual(first,native.logical_endpoint_identity(identity(),source))

    def test_resume_changes_only_invocation_port_and_skips_proven_completed_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);native.write_once(directory/'identity.json',identity())
            manifest={'tasks':task_rows(),'task_inventory_sha256':native.fingerprint(task_rows())}
            args=argparse.Namespace(setup=directory,output=directory/'results',model_identity=directory/'identity.json',
                training_manifest=None,wandb_entity='entity',wandb_project='project',wandb_run_id='run',max_tasks=1)
            tracking=types.SimpleNamespace(settings=types.SimpleNamespace(mode='online'),id='run',url='http://fixture',summary={},finish=mock.Mock(),log=mock.Mock())
            pipeline=types.SimpleNamespace(elements=[None,None,types.SimpleNamespace(client=mock.Mock()),None])
            suite=types.SimpleNamespace(get_user_task_by_id=lambda name:types.SimpleNamespace(ID=name))
            called=[]
            def benchmark_run(suite,pipeline,task,**kwargs):
                row=next(x for x in task_rows() if x['evaluation_id']=='banking/'+task.ID)
                called.append(row['evaluation_id'])
                native.write_once(native.trace_path(args.output,row),final_trace(row))
            modules={'agentdojo.agent_pipeline.agent_pipeline':types.SimpleNamespace(AgentPipeline=types.SimpleNamespace(from_config=lambda x:pipeline),PipelineConfig=lambda **x:x),
                'agentdojo.benchmark':types.SimpleNamespace(run_task_without_injection_tasks=benchmark_run),
                'agentdojo.logging':types.SimpleNamespace(OutputLogger=lambda p:contextlib.nullcontext())}
            immutable=None
            with mock.patch.object(native,'prepare',return_value=manifest),mock.patch.object(native,'load_native',return_value={'banking':suite}), \
                mock.patch.object(native.importlib,'import_module',side_effect=lambda n:modules[n]), \
                mock.patch.dict(sys.modules,{'wandb':types.SimpleNamespace(init=lambda **kw:tracking)}), \
                mock.patch.object(native,'runtime_episode_boundary',return_value=None):
                for port in [18080,18081]:
                    url=f'http://127.0.0.1:{port}/v1'
                    with mock.patch.dict(os.environ,{'OPENAI_COMPATIBLE_BASE_URL':url,'OPENAI_COMPATIBLE_API_KEY':'fixture',
                        'PUBLIC_RUNTIME_EPISODE_BOUNDARY_URL':url+'/runtime/episode-boundary'}):
                        report=native.run(args)
                        self.assertIsNone(report['score'])
                        now_manifest=(args.output/'run_manifest.json').read_bytes()
                        if immutable is not None:self.assertEqual(immutable,now_manifest)
                        immutable=now_manifest
            self.assertEqual(called,['banking/user_task_0','banking/user_task_1'])
            self.assertEqual(report['completed_episodes'],2)
            invocations=[json.loads(p.read_text()) for p in (args.output/'runtime_invocations').glob('*.json')]
            self.assertEqual(len(invocations),2)
            self.assertEqual({r['actual_endpoint'] for r in invocations},
                             {'http://127.0.0.1:18080/v1','http://127.0.0.1:18081/v1'})
            for invocation in invocations:
                self.assertEqual(invocation['actual_episode_boundary'],
                                 invocation['actual_endpoint']+'/runtime/episode-boundary')
                self.assertEqual(invocation['actual_endpoint_sha256'],
                                 native.digest(invocation['actual_endpoint'].encode()))
            self.assertEqual(len({r['actual_endpoint_sha256'] for r in invocations}),2)
            self.assertEqual(len({r['logical_endpoint_identity_sha256'] for r in invocations}),1)

    def test_episode_boundary_boolean_is_authoritative(self):
        class Response:
            def __init__(self,value):self.value=value
            def __enter__(self):return self
            def __exit__(self,*unused):return False
            def read(self):return json.dumps(self.value).encode()
        with mock.patch.dict(os.environ,{'PUBLIC_RUNTIME_EPISODE_BOUNDARY_URL':'http://127.0.0.1:18080/v1/runtime/episode-boundary','OPENAI_COMPATIBLE_BASE_URL':'http://127.0.0.1:18080/v1','OPENAI_COMPATIBLE_API_KEY':'fixture'}):
            for decision,expected in [({'may_start_episode':False,'reason':None},'RUNTIME_STOP_REQUESTED'),
                                      ({'may_start_episode':False,'reason':'STOP_MARKER_REQUESTED'},'STOP_MARKER_REQUESTED'),
                                      ({'may_start_episode':True,'reason':None},None)]:
                with mock.patch.object(native.urllib.request,'urlopen',return_value=Response(decision)):
                    self.assertEqual(native.runtime_episode_boundary(),expected)
            with mock.patch.object(native.urllib.request,'urlopen',return_value=Response({'may_start_episode':'false'})):
                with self.assertRaises(native.ContractError):native.runtime_episode_boundary()

    def test_stop_or_unavailable_boundary_precedes_new_immutable_intent(self):
        for decision in ['STOP_MARKER_REQUESTED',RuntimeError('control-plane unavailable')]:
            with self.subTest(decision=decision),tempfile.TemporaryDirectory() as tmp:
                directory=Path(tmp);native.write_once(directory/'identity.json',identity())
                manifest={'tasks':task_rows(),'task_inventory_sha256':native.fingerprint(task_rows())}
                args=argparse.Namespace(setup=directory,output=directory/'results',model_identity=directory/'identity.json',
                    training_manifest=None,wandb_entity='entity',wandb_project='project',wandb_run_id='run',max_tasks=1)
                tracking=types.SimpleNamespace(settings=types.SimpleNamespace(mode='online'),id='run',url='http://fixture',summary={},finish=mock.Mock())
                pipeline=types.SimpleNamespace(elements=[None,None,types.SimpleNamespace(client=mock.Mock())])
                benchmark=types.SimpleNamespace(run_task_without_injection_tasks=mock.Mock())
                modules={'agentdojo.agent_pipeline.agent_pipeline':types.SimpleNamespace(AgentPipeline=types.SimpleNamespace(from_config=lambda x:pipeline),PipelineConfig=lambda **x:x),
                    'agentdojo.benchmark':benchmark,'agentdojo.logging':types.SimpleNamespace(OutputLogger=lambda p:contextlib.nullcontext())}
                with mock.patch.object(native,'prepare',return_value=manifest),mock.patch.object(native,'load_native',return_value={}), \
                    mock.patch.object(native.importlib,'import_module',side_effect=lambda n:modules[n]), \
                    mock.patch.dict(sys.modules,{'wandb':types.SimpleNamespace(init=lambda **kw:tracking)}), \
                    mock.patch.dict(os.environ,{'OPENAI_COMPATIBLE_BASE_URL':'http://127.0.0.1:18080/v1','PUBLIC_RUNTIME_EPISODE_BOUNDARY_URL':'http://127.0.0.1:18080/v1/runtime/episode-boundary','OPENAI_COMPATIBLE_API_KEY':'fixture','PUBLIC_RUNTIME_EPISODE_STOP_RECEIPT':str(directory/'stop.json')}), \
                    mock.patch.object(native,'runtime_episode_boundary',side_effect=decision if isinstance(decision,Exception) else None,return_value=decision):
                    if isinstance(decision,Exception):
                        with self.assertRaises(RuntimeError):native.run(args)
                    else:
                        report=native.run(args)
                        self.assertIsNone(report['score'])
                        self.assertFalse(json.loads((directory/'stop.json').read_text())['intent_written'])
                    benchmark.run_task_without_injection_tasks.assert_not_called()
                    self.assertFalse((args.output/'intents/banking/user_task_0.json').exists())

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

    def test_publication_succeeds_when_hardlinks_are_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(native.os,"link",side_effect=PermissionError("hardlinks unsupported")) as link:
            path=Path(tmp)/"run_manifest.json"
            native.write_once(path,{"fixture":True})
            link.assert_not_called()
            self.assertEqual(path.read_bytes(),native.canonical({"fixture":True}))
            claim=native.read_json(path.with_name(".run_manifest.json.publish.claim"))
            self.assertEqual(claim["content_sha256"],native.digest(path.read_bytes()))

    def test_failed_rename_leaves_permanent_claim_and_blocks_automatic_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"receipt.json"
            with mock.patch.object(native.os,"rename",side_effect=OSError("synthetic crash before publication")):
                with self.assertRaises(OSError):native.write_once(path,{"value":1})
            self.assertFalse(path.exists())
            self.assertTrue(path.with_name(".receipt.json.publish.claim").exists())
            self.assertEqual(list(Path(tmp).glob(".publish-*")),[])
            with self.assertRaisesRegex(native.ContractError,"unfinished publication claim"):
                native.write_once(path,{"value":1})

    def test_claim_and_complete_temp_are_fsynced_before_rename(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"receipt.json";events=[]
            fsync=native.os.fsync;rename=native.os.rename
            def sync(fd):events.append("fsync");return fsync(fd)
            def publish(source,target):
                self.assertEqual(events,["fsync","fsync","fsync"])
                self.assertFalse(path.exists())
                self.assertEqual(Path(source).read_bytes(),native.canonical({"value":1}))
                events.append("rename");return rename(source,target)
            with mock.patch.object(native.os,"fsync",side_effect=sync),mock.patch.object(native.os,"rename",side_effect=publish):
                native.write_once(path,{"value":1})
            self.assertEqual(events,["fsync","fsync","fsync","rename","fsync"])

    def test_claim_directory_fsync_failure_prevents_publication_and_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"receipt.json"
            with mock.patch.object(native,"fsync_directory",side_effect=OSError("directory sync failed")), \
                 mock.patch.object(native.os,"rename") as rename:
                with self.assertRaises(OSError):native.write_once(path,{"value":1})
            rename.assert_not_called()
            self.assertFalse(path.exists())
            self.assertTrue(path.with_name(".receipt.json.publish.claim").exists())
            with self.assertRaisesRegex(native.ContractError,"unfinished publication claim"):
                native.write_once(path,{"value":1})

    def test_concurrent_publisher_cannot_replace_claimed_inflight_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"receipt.json";ready=threading.Event();release=threading.Event();errors=[]
            rename=native.os.rename
            def publish(source,target):
                ready.set()
                if not release.wait(timeout=2):raise RuntimeError("fixture timed out")
                return rename(source,target)
            def first():
                try:native.write_once(path,{"value":1})
                except BaseException as exc:errors.append(exc)
            with mock.patch.object(native.os,"rename",side_effect=publish):
                worker=threading.Thread(target=first);worker.start()
                try:
                    self.assertTrue(ready.wait(timeout=2))
                    with self.assertRaisesRegex(native.ContractError,"unfinished publication claim"):
                        native.write_once(path,{"value":2})
                finally:release.set();worker.join(timeout=2)
            self.assertFalse(worker.is_alive());self.assertEqual(errors,[])
            self.assertEqual(native.read_json(path),{"value":1})

    def test_symlink_or_uncoordinated_existing_destination_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"receipt.json";other=Path(tmp)/"other.json";other.write_bytes(b"original")
            path.symlink_to(other)
            with self.assertRaisesRegex(native.ContractError,"symlinked"):native.write_once(path,{"value":1})
            self.assertEqual(other.read_bytes(),b"original")
            path.unlink()
            original=native.tempfile.mkstemp
            def surprise(*args,**kwargs):
                path.write_bytes(b"external bytes")
                return original(*args,**kwargs)
            with mock.patch.object(native.tempfile,"mkstemp",side_effect=surprise),mock.patch.object(native.os,"rename") as rename:
                with self.assertRaisesRegex(native.ContractError,"appeared after exclusive claim"):
                    native.write_once(path,{"value":1})
            rename.assert_not_called();self.assertEqual(path.read_bytes(),b"external bytes")

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
                 mock.patch.object(native, "runtime_episode_boundary", return_value=None), \
                 mock.patch.object(native, "load_native") as load, \
                 mock.patch.dict(sys.modules, {"wandb": wandb}), \
                 mock.patch.dict(os.environ, {"OPENAI_COMPATIBLE_BASE_URL": "http://localhost:18080/v1",
                                             "PUBLIC_RUNTIME_EPISODE_BOUNDARY_URL":"http://localhost:18080/v1/runtime/episode-boundary",
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
                 mock.patch.object(native, "runtime_episode_boundary", return_value=None), \
                 mock.patch.object(native, "load_native", return_value={"banking": mock.Mock()}), \
                 mock.patch.object(native.importlib, "import_module", side_effect=lambda name: modules[name]), \
                 mock.patch.dict(sys.modules, {"wandb": wandb}), \
                 mock.patch.dict(os.environ, {"OPENAI_COMPATIBLE_BASE_URL": "http://localhost:18080/v1",
                                             "PUBLIC_RUNTIME_EPISODE_BOUNDARY_URL":"http://localhost:18080/v1/runtime/episode-boundary",
                                             "OPENAI_COMPATIBLE_API_KEY": "fixture"}):
                with self.assertRaises(RuntimeError):
                    native.run(args)
                report = native.summarize(args.output)
                self.assertIsNone(report["score"])
                self.assertEqual(report["completed_episodes"], 0)
                self.assertTrue((args.output / "intents/banking/user_task_0.json").exists())
                self.assertEqual(len(list((args.output / "failures").glob("*.json"))), 1)
                benchmark.run_task_without_injection_tasks.reset_mock()
                with mock.patch.dict(os.environ, {"OPENAI_COMPATIBLE_BASE_URL":"http://localhost:18081/v1",
                    "PUBLIC_RUNTIME_EPISODE_BOUNDARY_URL":"http://localhost:18081/v1/runtime/episode-boundary"}):
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
