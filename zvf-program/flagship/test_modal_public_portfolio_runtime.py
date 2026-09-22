"""Local fail-closed checks only: never allocate hardware or load model weights."""
import datetime as dt
import ast
import importlib.util
import json
import io
import http.client
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
import sys
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / ".codex-run/public_colab_runtime.py"
spec = importlib.util.spec_from_file_location("public_runtime_under_test", SCRIPT)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def reservation():
    return {"status": "RESERVED", "reservation_id": "fixture-not-real-money",
            "provider": "modal", "unit": "USD", "reserved_units": 2,
            "max_hourly_units": 8, "max_wall_seconds": 900, "max_requests": 2,
            "expires_at": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).isoformat()}


def profile_helpers():
    source = SCRIPT.parents[1] / "zvf-program/flagship/modal_public_portfolio_runtime.py"
    tree = ast.parse(source.read_text())
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name in {"runtime_profile", "validate_profile_reservation"}]
    namespace = {"load_runtime": lambda: runtime,
                 "INCLUSIVE_HOURLY_USD": 3600 * (0.001261 + 4 * 0.0000131 + 128 * 0.00000222)}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


class RuntimeGuards(unittest.TestCase):
    def test_jsonl_preserves_literal_unicode_separators_and_full_lab_count(self):
        row = {"task_id": "unicode-fixture", "payload": {"messages": [{"role": "user", "content": "before\u2028middle\u2029after"}]}}
        text = json.dumps(row, ensure_ascii=False) + "\n"
        self.assertEqual(runtime.parse_request_jsonl(text), [row])
        self.assertGreater(len(text.splitlines()), 1)
        source = SCRIPT.parents[1] / "zvf-program/flagship/modal_public_portfolio_runtime.py"
        tree = ast.parse(source.read_text())
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate_inputs")
        counts = []
        namespace = {"PROFILE": {}, "validate_profile_reservation": lambda profile, funds, count: counts.append(count)}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), namespace)
        early = {"mode": "online", "initialized_before_model_work": True}
        namespace["validate_inputs"](text, {}, early)
        self.assertEqual(counts.pop(), 3)  # one actual row plus two runtime smokes
        batch = SCRIPT.parents[1] / "outputs/public_portfolio_2026-09-05/labbench_run/batch-001.jsonl"
        self.assertTrue(batch.exists(), "immutable prepared full LAB batch required for this regression")
        actual = batch.read_text()
        rows = runtime.parse_request_jsonl(actual)
        self.assertEqual(len(rows),1967)
        with batch.open() as handle:
            self.assertEqual(rows, [json.loads(line) for line in handle if line.strip()])
        namespace["validate_inputs"](actual, {}, early)
        self.assertEqual(counts.pop(),1969)

    def test_clean_deadline_stop_between_waves_preserves_complete_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            args = SimpleNamespace(output_dir=directory, served_model_name=runtime.SERVED_MODEL,
                                   request_seconds=180, max_model_len=2048, max_num_seqs=2, commit_volume=None)
            rows = [{"task_id": f"fixture-{i}", "payload": {"model": runtime.SERVED_MODEL,
                    "messages": [{"role": "user", "content": str(i)}], "max_tokens": 32}} for i in range(5)]
            raw = json.dumps({"model": runtime.SERVED_MODEL, "choices": [{"index": 0, "message": {"content": "fixture"}}]}).encode()
            clock = {"now": 580.0}
            generation_tasks = []
            def fake_http(url, payload, timeout, return_raw=False):
                if url.endswith("/tokenize"):
                    return {"count": 1}
                generation_tasks.append(payload["messages"][0]["content"])
                clock["now"] = 620.0  #380s left: below180s tokenize +180s generate +35s teardown.
                return None, raw, 200
            receipt = {"requests_completed": 2, "score": None}
            with (directory/"responses.jsonl").open("x") as output, (directory/"started_requests.jsonl").open("x") as started:
                with patch.object(runtime.time, "monotonic", side_effect=lambda: clock["now"]), patch.object(runtime, "request_json", side_effect=fake_http):
                    complete = runtime.execute_batch_waves(rows,args,"http://fixture",output,started,threading.Lock(),receipt,SimpleNamespace(log=lambda value:None),1000.0)
                self.assertFalse(complete)
                # These reads occur before closing handles: data must already be durable.
                starts = [json.loads(x) for x in (directory/"started_requests.jsonl").read_text().splitlines()]
                results = [json.loads(x) for x in (directory/"responses.jsonl").read_text().splitlines()]
                transport = [json.loads(x) for x in (directory/"http_responses.jsonl").read_text().splitlines()]
            self.assertEqual(sorted(generation_tasks), ["0", "1"])
            self.assertEqual({r["task_id"] for r in starts}, {"fixture-0", "fixture-1"})
            self.assertEqual({r["task_id"] for r in starts}, {r["task_id"] for r in results})
            self.assertEqual(len(transport), 2)
            self.assertTrue(all(r["status"] == "HTTP_RESPONSE_RECORDED" for r in results))
            self.assertEqual(receipt["status"], "RUNTIME_BATCH_PARTIAL_CLEAN_STOP")
            self.assertEqual(receipt["requests_completed"],4)
            self.assertEqual(receipt["clean_stop"]["next_unstarted_row_index"],2)
            self.assertEqual(receipt["clean_stop"]["unstarted_rows"],3)
            self.assertEqual(receipt["clean_stop"]["active_requests"],0)
            self.assertEqual(receipt["clean_stop"]["required_seconds"],395)
            self.assertEqual(receipt["clean_stop"]["tokenization_allowance_seconds"],180)
            self.assertEqual(receipt["clean_stop"]["generation_allowance_seconds"],180)
            self.assertEqual(json.loads((directory/"runtime_receipt.json").read_text()),receipt)

    def test_deadline_guard_stops_before_any_new_intent(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            args = SimpleNamespace(output_dir=directory, request_seconds=180, max_num_seqs=8, commit_volume=None)
            receipt = {"score": None}
            with (directory/"responses.jsonl").open("x") as output, (directory/"started_requests.jsonl").open("x") as started:
                with patch.object(runtime.time,"monotonic",return_value=605.001), patch.object(runtime,"execute_request") as execute:
                    self.assertFalse(runtime.execute_batch_waves([{}],args,"http://fixture",output,started,threading.Lock(),receipt,SimpleNamespace(log=lambda value:None),1000.0))
                    execute.assert_not_called()
                self.assertEqual(output.tell(),0)
                self.assertEqual(started.tell(),0)
            self.assertEqual(receipt["clean_stop"]["required_seconds"],395)

    def test_both_profiles_bind_budget_wall_time_and_request_limit(self):
        helpers = profile_helpers()
        for name, cap, seconds in (("canary", 2, 900), ("fullbatch", 8, 3600)):
            with self.subTest(profile=name):
                profile = helpers["runtime_profile"](name)
                self.assertEqual(profile["function_seconds"] + profile["startup_seconds"], seconds)
                funds = {**reservation(), "runtime_profile": name, "reserved_units": cap,
                         "max_wall_seconds": seconds, "max_requests": 1969,
                         "expires_at": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=2)).isoformat()}
                helpers["validate_profile_reservation"](profile, funds, 1969)
                for key, value in (("reserved_units", cap - .01), ("reserved_units", cap + .01),
                                   ("max_wall_seconds", seconds - 1), ("max_wall_seconds", seconds + 1),
                                   ("max_requests", 1970), ("max_hourly_units", 5.0),
                                   ("runtime_profile", "different")):
                    with self.subTest(profile=name, invalid=key, value=value):
                        with self.assertRaises(ValueError):
                            helpers["validate_profile_reservation"](profile, {**funds, key: value}, 1969)
                with self.assertRaises(ValueError):
                    helpers["validate_profile_reservation"](profile, funds, 1970)

    def test_fullbatch_cannot_be_selected_by_omitting_explicit_binding(self):
        helpers = profile_helpers()
        full = helpers["runtime_profile"]("fullbatch")
        funds = {**reservation(), "reserved_units": 8, "max_wall_seconds": 3600,
                 "expires_at": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=2)).isoformat()}
        with self.assertRaisesRegex(ValueError, "explicit runtime_profile"):
            helpers["validate_profile_reservation"](full, funds, 2)
        helpers["validate_profile_reservation"](helpers["runtime_profile"]("canary"), reservation(), 2)
        with self.assertRaises(ValueError):
            helpers["runtime_profile"]("automatic")

    def test_flat_remote_source_path_never_indexes_absent_parent(self):
        source = SCRIPT.parents[1] / "zvf-program/flagship/modal_public_portfolio_runtime.py"
        tree = ast.parse(source.read_text())
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "runtime_source_path")
        namespace = {"Path": Path, "REMOTE_RUNTIME": "/root/public_colab_runtime.py"}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), namespace)
        self.assertEqual(namespace["runtime_source_path"]("/root/module.py", False), Path("/root/public_colab_runtime.py"))

    def test_reservation_covers_inclusive_wall_time(self):
        runtime.check_reservation(reservation(), 900, 2)
        for key, value in (("reserved_units", 1.99), ("status", "PENDING"),
                           ("max_requests", 1), ("max_hourly_units", float("nan")),
                           ("expires_at", "2020-01-01T00:00:00+00:00")):
            with self.subTest(key=key):
                bad = {**reservation(), key: value}
                with self.assertRaises(ValueError):
                    runtime.check_reservation(bad, 900, 2)

    def test_twenty_six_shards_are_rehashed_and_drift_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            hashes = {}
            weights = {}
            for i in range(26):
                name = f"model-{i+1:05d}-of-00026.safetensors"
                (path / name).write_bytes(f"NONMODEL FIXTURE {i}".encode())
                hashes[name] = runtime.digest(path / name)
                weights[f"weight{i}"] = name
            (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights}))
            (path / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5MoeForConditionalGeneration"]}))
            receipt = {"base_model": runtime.BASE, "base_commit": runtime.BASE_REVISION,
                       "adapter_repo": runtime.ADAPTER, "adapter_commit": runtime.ADAPTER_REVISION,
                       "merge_method": runtime.MERGE_METHOD, "all_adapter_tensors_consumed": True,
                       "adapter_tensor_count": 862, "adapter_module_count": 431,
                       "weight_file_count": 26, "weight_shard_sha256": hashes}
            runtime.check_merge_identity(receipt, path)
            (path / sorted(hashes)[0]).write_bytes(b"CORRUPTED")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                runtime.check_merge_identity(receipt, path)

    def test_smokes_are_declared_multimodal_and_have_no_suite_grade(self):
        # PIL is optional for local lightweight installations.
        try:
            rows = runtime.smoke_requests()
        except ImportError:
            self.skipTest("Pillow unavailable in this local Python")
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[1]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertNotIn("score", rows[1])

    def test_request_boundaries_preserve_native_metadata_and_raw_bytes(self):
        row = {"task_id": "fixture-task", "batch_id": "fixture-batch", "contract_sha256": "c",
               "request_sha256": "native-record-hash", "payload": {"model": runtime.SERVED_MODEL,
               "messages": [{"role": "user", "content": "fixture"}], "max_tokens": 32}}
        args = SimpleNamespace(served_model_name=runtime.SERVED_MODEL, request_seconds=5,
                               max_model_len=8192, commit_volume=None)
        raw = b'{ "model":"pavlov-public-portfolio-bf16", "choices": [{"index":0,"message":{"content":"fixture answer"}}] }'
        events = []
        def fake_http(url, payload, timeout, return_raw=False):
            events.append(url.rsplit("/", 1)[-1])
            if url.endswith("/tokenize"):
                return {"count": 100}
            self.assertIn("commit", events)
            return json.loads(raw), raw, 200
        with tempfile.TemporaryDirectory() as temp:
            with (Path(temp) / "responses").open("w+") as out, (Path(temp) / "started").open("w+") as started:
                with patch.object(runtime, "request_json", side_effect=fake_http), patch.object(runtime, "sync_volume", side_effect=lambda name: events.append("commit")):
                    result = runtime.execute_request(2, row, args, "http://fixture", out, started, threading.Lock())
                self.assertEqual(result["request_sha256"], "native-record-hash")
                self.assertEqual(runtime.base64.b64decode(result["raw_body_base64"]), raw)
                self.assertEqual(result["prompt_token_count"], 100)
                started.seek(0)
                self.assertEqual(json.loads(started.read())["task_id"], "fixture-task")
        self.assertEqual(events, ["tokenize", "commit", "completions", "commit", "commit"])

    def test_received_http_evidence_survives_decode_envelope_and_smoke_failures(self):
        class Response:
            status = 200
            def __init__(self, raw, error=None): self.raw, self.error = raw, error
            def __enter__(self): return self
            def __exit__(self, *unused): return False
            def read(self):
                if self.error: raise self.error
                return self.raw
        payload = {"model": runtime.SERVED_MODEL, "messages": [], "max_tokens": 32}
        args = SimpleNamespace(served_model_name=runtime.SERVED_MODEL, request_seconds=1,
                               max_model_len=2048, commit_volume=None)
        cases = [(b'{broken-json', "MALFORMED_JSON_RESPONSE"),
                 (b'\xff', "MALFORMED_JSON_RESPONSE"),
                 (b'[]', "INVALID_SERVER_ENVELOPE"),
                 (b'{"model":"pavlov-public-portfolio-bf16","choices":[null]}', "INVALID_SERVER_ENVELOPE"),
                 (b'{"model":"pavlov-public-portfolio-bf16","choices":[{"index":0,"message":{"content":""}}]}', "EMPTY_SMOKE_GENERATION")]
        for raw, expected in cases:
            with self.subTest(expected=expected, raw=raw), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                original_loads = json.loads
                def loads(value, *a, **kw):
                    if value == raw:
                        transport = original_loads((directory/"http_responses.jsonl").read_text())
                        self.assertEqual(runtime.base64.b64decode(transport["raw_body_base64"]), raw)
                        self.assertEqual(transport["http_status"], 200)
                    return original_loads(value, *a, **kw)
                with (directory/"responses").open("w+") as out, (directory/"starts").open("w+") as starts:
                    with patch.object(runtime.urllib.request, "urlopen", side_effect=[Response(b'{"count":1}'), Response(raw)]), patch.object(runtime.json, "loads", side_effect=loads):
                        result = runtime.execute_request(0, payload, args, "http://fixture", out, starts, threading.Lock())
                    self.assertEqual(result["status"], expected)
                    self.assertEqual(result["http_status"], 200)
                    self.assertEqual(runtime.base64.b64decode(result["raw_body_base64"]), raw)
                    out.seek(0)
                    self.assertEqual(json.loads(out.read())["raw_body_base64"], result["raw_body_base64"])

    def test_http_error_and_partial_body_preserve_known_status_and_bytes(self):
        class Response:
            status = 200
            def __init__(self, raw, error=None): self.raw, self.error = raw, error
            def __enter__(self): return self
            def __exit__(self, *unused): return False
            def read(self):
                if self.error: raise self.error
                return self.raw
        raw = b'partial-non-json'
        payload = {"model": runtime.SERVED_MODEL, "messages": [], "max_tokens": 32}
        args = SimpleNamespace(served_model_name=runtime.SERVED_MODEL, request_seconds=1,
                               max_model_len=2048, commit_volume=None)
        cases = [(runtime.urllib.error.HTTPError("http://fixture", 502, "fixture", {}, io.BytesIO(raw)), 502, "HTTP_ERROR_RECORDED"),
                 (Response(b'', http.client.IncompleteRead(raw, 100)), 200, "HTTP_BODY_READ_FAILED")]
        for reply, status, label in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                with (Path(temp)/"responses").open("w+") as out, (Path(temp)/"starts").open("w+") as starts:
                    with patch.object(runtime.urllib.request, "urlopen", side_effect=[Response(b'{"count":1}'), reply]):
                        result = runtime.execute_request(2, payload, args, "http://fixture", out, starts, threading.Lock())
                    self.assertEqual(result["status"], label)
                    self.assertEqual(result["http_status"], status)
                    self.assertEqual(runtime.base64.b64decode(result["raw_body_base64"]), raw)

    def test_overlength_request_never_generates_or_marks_started(self):
        args = SimpleNamespace(served_model_name=runtime.SERVED_MODEL, request_seconds=5,
                               max_model_len=8192, commit_volume=None)
        row = {"model": runtime.SERVED_MODEL, "messages": [], "max_tokens": 32}
        with tempfile.TemporaryDirectory() as temp:
            with (Path(temp) / "responses").open("w+") as out, (Path(temp) / "started").open("w+") as started:
                with patch.object(runtime, "request_json", return_value={"count": 8190}) as http:
                    with self.assertRaisesRegex(ValueError, "no truncation permitted"):
                        runtime.execute_request(2, row, args, "http://fixture", out, started, threading.Lock())
                self.assertEqual(http.call_count, 1)
                self.assertEqual(started.tell(), 0)

    def test_private_driver_proxy_caches_episode_retry_and_enforces_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            driver = path / "fixture_driver.py"
            driver.write_text('''import json,os,urllib.request,urllib.error
url=os.environ["OPENAI_COMPATIBLE_BASE_URL"]+"/chat/completions"
payload={"model":"pavlov-public-portfolio-bf16","messages":[{"role":"user","content":"fixture"}]}
statuses=[]
for task in ["episode1","episode1","episode2"]:
 req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={"Authorization":"Bearer "+os.environ["OPENAI_COMPATIBLE_API_KEY"],"Content-Type":"application/json","X-Public-Task-ID":task})
 try:
  with urllib.request.urlopen(req,timeout=5) as response: statuses.append(response.status)
 except urllib.error.HTTPError as exc: statuses.append(exc.code)
assert statuses==[200,200,403],statuses
''')
            args = SimpleNamespace(output_dir=path, served_model_name=runtime.SERVED_MODEL,
                                   request_seconds=5, max_model_len=8192, commit_volume=None,
                                   enable_tool_calling=True)
            calls = []
            raw = b'{"model":"pavlov-public-portfolio-bf16","choices":[{"index":0,"message":{"content":"fixture"}}]}'
            def fake_http(url, payload, timeout, return_raw=False):
                calls.append(url)
                if url.endswith("/tokenize"):
                    return {"count": 100}
                self.assertEqual(payload["max_tokens"], 64)
                return json.loads(raw), raw, 200
            receipt = {}
            with patch.object(runtime, "request_json", side_effect=fake_http):
                runtime.run_native_driver({"driver_id":"fixture", "max_requests":1,"default_max_tokens":64,
                                           "argv":[sys.executable,str(driver)]},args,"http://fixture",receipt)
            self.assertEqual(len(calls),2)
            self.assertEqual(receipt["native_driver"]["requests"],1)
            self.assertEqual(receipt["native_driver"]["cache_hits"],1)
            self.assertEqual(receipt["native_driver"]["rejected"],1)


if __name__ == "__main__":
    unittest.main()
