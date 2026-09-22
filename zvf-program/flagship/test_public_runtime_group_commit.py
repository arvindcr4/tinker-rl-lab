"""Deterministic local barriers/crash fixtures; no network, model or provider."""
import base64
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("group_runtime_test", ROOT / ".codex-run/public_colab_runtime_group.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
journal = runtime.journal


class Fixture:
    def __init__(self, directory, backend=lambda: None, **options):
        self.paths = {phase: directory / (phase + ".jsonl") for phase in journal.PHASES}
        self.fence = directory / "epochs.jsonl"
        self.coordinator = journal.GroupCommitCoordinator(self.paths, self.fence, backend, **options)

    def starts(self, n=1):
        tickets = self.coordinator.submit_many("STARTED", [(str(i), {"task_id": str(i)}, None) for i in range(n)])
        self.coordinator.wait_all(tickets)
        return tickets

    def snapshot(self):
        return {phase: path.read_bytes() for phase, path in self.paths.items()}


class GroupCommitTests(unittest.TestCase):
    def test_32_intents_share_one_acknowledged_epoch_before_dispatch(self):
        entered, release = threading.Event(), threading.Event()
        snapshots, generated = [], []
        with tempfile.TemporaryDirectory() as temporary:
            def commit():
                snapshots.append(fixture.snapshot())
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("fixture release missing")
            fixture = Fixture(Path(temporary), commit)
            coordinator = fixture.coordinator
            tickets = coordinator.submit_many("STARTED", [(str(i), {"task_id": str(i)}, None) for i in range(32)])
            with ThreadPoolExecutor(max_workers=32) as executor:
                futures = [executor.submit(lambda t=t: (t.wait_committed(), generated.append(t.key))) for t in tickets]
                self.assertTrue(entered.wait(3))
                self.assertEqual(generated, [])
                self.assertEqual(len(snapshots[0]["STARTED"].split(b"\n")) - 1, 32)
                self.assertTrue(all(not ticket.committed for ticket in tickets))
                release.set()
                for future in futures:
                    future.result(3)
            coordinator.flush_and_close()
            self.assertEqual(len(snapshots), 1)
            self.assertEqual(set(generated), {str(i) for i in range(32)})
            self.assertTrue(all(ticket.epoch == 1 for ticket in tickets))

    def test_late_arrivals_wait_for_their_own_epoch_and_prefix_fences_verify(self):
        entered = [threading.Event(), threading.Event()]
        release = [threading.Event(), threading.Event()]
        count = []
        with tempfile.TemporaryDirectory() as temporary:
            def commit():
                i = len(count)
                count.append(i)
                entered[i].set()
                if not release[i].wait(3):
                    raise RuntimeError("fixture release missing")
            fixture = Fixture(Path(temporary), commit, max_delay_ms=0)
            first = fixture.coordinator.submit("STARTED", "a", {"a": 1})
            self.assertTrue(entered[0].wait(3))
            late = fixture.coordinator.submit("STARTED", "b", {"b": 2})
            release[0].set()
            first.wait_committed(3)
            self.assertTrue(entered[1].wait(3))
            self.assertFalse(late.committed)
            release[1].set()
            late.wait_committed(3)
            fixture.coordinator.flush_and_close()
            fences = [json.loads(x) for x in fixture.fence.read_text().split("\n") if x]
            self.assertEqual([first.epoch, late.epoch], [1, 2])
            self.assertTrue(all(journal.verify_prefixes(fixture.paths, fence) for fence in fences))
            self.assertTrue(all("committed" not in fence for fence in fences))
            fixture.paths["STARTED"].write_bytes(b"corrupt")
            with self.assertRaises(journal.JournalFailure):
                journal.verify_prefixes(fixture.paths, fences[-1])

    def test_exact_raw_bytes_committed_before_parse_including_malformed_utf8(self):
        for body in (b"not json", b"\xff\xfe"):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as temporary:
                entered, release = threading.Event(), threading.Event()
                calls, snapshots = [], []
                def commit():
                    snapshots.append(fixture.snapshot())
                    if len(snapshots) == 2:
                        entered.set()
                        if not release.wait(3):
                            raise RuntimeError("fixture release missing")
                fixture = Fixture(Path(temporary), commit, max_delay_ms=0)
                intent = fixture.starts()[0]
                raw = fixture.coordinator.submit("RAW_HTTP", "0", {"http_status": 200,
                    "raw_body_base64": base64.b64encode(body).decode()}, intent)
                def parse():
                    raw.wait_committed(3)
                    calls.append("parse")
                    try:
                        json.loads(body)
                    except (ValueError, UnicodeError):
                        pass
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(parse)
                    self.assertTrue(entered.wait(3))
                    self.assertEqual(calls, [])
                    stored = json.loads(snapshots[-1]["RAW_HTTP"])
                    self.assertEqual(base64.b64decode(stored["raw_body_base64"]), body)
                    release.set()
                    future.result(3)
                parsed = fixture.coordinator.submit("PARSED", "0", {"status": "MALFORMED_JSON_RESPONSE"}, raw)
                parsed.wait_committed(3)
                fixture.coordinator.flush_and_close()
                self.assertEqual(calls, ["parse"])
                self.assertEqual([intent.epoch, raw.epoch, parsed.epoch], [1, 2, 3])

    def test_prerequisite_mutation_duplicate_and_unicode_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), max_delay_ms=0)
            record = {"text": "a\u2028b\u2029c"}
            ticket = fixture.coordinator.submit("STARTED", "a", record)
            record["text"] = "mutated"
            ticket.wait_committed(3)
            with self.assertRaises(ValueError):
                fixture.coordinator.submit("STARTED", "a", {"different": True})
            with self.assertRaises(journal.JournalFailure):
                fixture.coordinator.submit("PARSED", "a", {}, ticket)
            with self.assertRaises(journal.JournalFailure):
                fixture.coordinator.submit("RAW_HTTP", "other", {}, ticket)
            raw = fixture.coordinator.submit("RAW_HTTP", "a", {}, ticket)
            raw.wait_committed(3)
            fixture.coordinator.submit("PARSED", "a", {}, raw).wait_committed(3)
            fixture.coordinator.flush_and_close()
            self.assertEqual(json.loads(fixture.paths["STARTED"].read_bytes()), {"text": "a\u2028b\u2029c"})
            self.assertEqual(len(fixture.paths["STARTED"].read_bytes().split(b"\n")), 2)

    def test_three_full_phase_cohorts_need_exactly_three_commits(self):
        commits = []
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), lambda: commits.append(True))
            starts = fixture.starts(32)
            raws = fixture.coordinator.submit_many("RAW_HTTP", [(str(i), {"raw": i}, starts[i]) for i in range(32)])
            fixture.coordinator.wait_all(raws)
            parsed = fixture.coordinator.submit_many("PARSED", [(str(i), {"parsed": i}, raws[i]) for i in range(32)])
            fixture.coordinator.wait_all(parsed)
            fixture.coordinator.flush_and_close()
            self.assertEqual(len(commits), 3)
            self.assertEqual([e["records"] for e in fixture.coordinator.snapshot()["epochs"]], [32, 32, 32])

    def test_unacknowledged_prerequisite_and_cross_request_mixed_epoch(self):
        entered, release = threading.Event(), threading.Event()
        commits = []
        with tempfile.TemporaryDirectory() as temporary:
            def commit():
                commits.append(1)
                if len(commits) == 2:
                    entered.set()
                    if not release.wait(3):
                        raise RuntimeError("fixture release missing")
            fixture = Fixture(Path(temporary), commit, max_delay_ms=0)
            starts = fixture.starts(2)
            raw_b = fixture.coordinator.submit("RAW_HTTP", "1", {}, starts[1])
            self.assertTrue(entered.wait(3))
            with self.assertRaises(journal.JournalFailure):
                fixture.coordinator.submit("PARSED", "1", {}, raw_b)
            release.set()
            raw_b.wait_committed(3)
            with fixture.coordinator._condition:
                raw_a = fixture.coordinator.submit("RAW_HTTP", "0", {}, starts[0])
                parsed_b = fixture.coordinator.submit("PARSED", "1", {}, raw_b)
            fixture.coordinator.wait_all([raw_a, parsed_b])
            self.assertEqual(raw_a.epoch, parsed_b.epoch)
            fixture.coordinator.flush_and_close()

    def test_oversized_cohort_fails_before_any_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), max_pending_bytes=10)
            with self.assertRaises(journal.JournalFailure):
                fixture.coordinator.submit("STARTED", "a", {"too_large": "payload"})
            fixture.coordinator.flush_and_close()
            self.assertFalse(fixture.paths["STARTED"].read_bytes())

    def test_write_fsync_commit_failures_poison_without_acknowledgement(self):
        for failure in ("write", "fsync", "commit"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                def fail(*unused):
                    raise OSError("fixture injected failure")
                fixture = Fixture(Path(temporary), fail if failure == "commit" else lambda: None,
                                  max_delay_ms=0, **({"fsync": fail} if failure == "fsync" else {}))
                if failure == "write":
                    fixture.coordinator._append = fail
                ticket = fixture.coordinator.submit("STARTED", "a", {})
                with self.assertRaises(journal.JournalFailure):
                    ticket.wait_committed(3)
                self.assertFalse(ticket.committed)
                with self.assertRaises(journal.JournalFailure):
                    fixture.coordinator.submit("STARTED", "b", {})
                with self.assertRaises(journal.JournalFailure):
                    fixture.coordinator.flush_and_close(3)

    def test_remote_success_client_failure_requires_recovery_no_new_dispatch(self):
        saved = []
        with tempfile.TemporaryDirectory() as temporary:
            def commit():
                saved.append(fixture.snapshot())
                raise OSError("lost acknowledgement after remote copy")
            fixture = Fixture(Path(temporary), commit, max_delay_ms=0)
            ticket = fixture.coordinator.submit("STARTED", "a", {"task_id": "a"})
            with self.assertRaises(journal.JournalFailure):
                ticket.wait_committed(3)
            self.assertIn(b'"task_id":"a"', saved[0]["STARTED"])
            self.assertFalse(ticket.committed)
            with self.assertRaises(journal.JournalFailure):
                fixture.coordinator.ensure_healthy()
            with self.assertRaises(journal.JournalFailure):
                fixture.coordinator.flush_and_close(3)

    def test_close_flushes_partial_cohort_and_stuck_commit_never_acknowledges(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            ticket = fixture.coordinator.submit("STARTED", "single", {})
            fixture.coordinator.flush_and_close(3)
            self.assertTrue(ticket.committed)
            self.assertFalse(fixture.coordinator.snapshot()["writer_alive"])
        with tempfile.TemporaryDirectory() as temporary:
            entered, release = threading.Event(), threading.Event()
            def stuck():
                entered.set()
                release.wait(3)
            fixture = Fixture(Path(temporary), stuck, max_delay_ms=0)
            ticket = fixture.coordinator.submit("STARTED", "single", {})
            self.assertTrue(entered.wait(3))
            with self.assertRaises(journal.JournalFailure):
                fixture.coordinator.flush_and_close(0)
            self.assertFalse(ticket.committed)
            release.set()
            fixture.coordinator._thread.join(3)
            self.assertFalse(ticket.committed)


class RuntimeGroupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.args = types.SimpleNamespace(output_dir=self.path, served_model_name=runtime.fast.SERVED_MODEL,
            request_seconds=180, max_model_len=32768, max_num_seqs=32, commit_volume=None)
        self.rows = [{"task_id": str(i), "payload": {"model": runtime.fast.SERVED_MODEL,
            "messages": [{"role": "user", "content": f"text{i}\u2028preserved"}],
            "max_tokens": 32, "temperature": 0, "seed": 809}} for i in range(32)]
        self.raw = json.dumps({"model": runtime.fast.SERVED_MODEL,
            "choices": [{"index": 0, "message": {"content": "fixture"}}]}).encode()

    def tearDown(self):
        coordinator = runtime.COORDINATORS.pop(str(self.path.resolve()), None)
        if coordinator:
            try:
                coordinator.flush_and_close(3)
            except journal.JournalFailure:
                pass
        self.temp.cleanup()

    def execute(self, rows=None, deadline=None, receipt=None):
        return runtime.execute_batch_waves(self.rows if rows is None else rows, self.args, "http://fixture",
            None, None, threading.Lock(), {} if receipt is None else receipt,
            types.SimpleNamespace(log=lambda _: None), runtime.time.monotonic() + 1000 if deadline is None else deadline)

    def test_full_wave_intents_single_commit_raw_evidence_and_exact_payloads(self):
        barrier = threading.Barrier(32, timeout=3)
        sent, snapshots = [], []
        def commit(*unused):
            snapshots.append({p.name: p.read_bytes() for p in self.path.glob("*.jsonl")})
        def http(url, payload, timeout, return_raw=False):
            if url.endswith("/tokenize"):
                return {"count": 10}
            self.assertEqual(len(snapshots[0]["started_requests.jsonl"].split(b"\n")) - 1, 32)
            sent.append(payload)
            barrier.wait()
            return None, self.raw, 200
        with patch.object(runtime, "request_json", side_effect=http), patch.object(runtime.fast, "sync_volume", side_effect=commit):
            self.assertTrue(self.execute())
        self.assertEqual(sorted(map(runtime.stable_hash, sent)), sorted(runtime.stable_hash(r["payload"]) for r in self.rows))
        coordinator = runtime.coordinator_for(self.args)
        epochs = coordinator.snapshot()["epochs"]
        self.assertEqual(epochs[0]["records"], 32)
        self.assertEqual(set(epochs[0]["phases"]), {"STARTED"})
        self.assertEqual(sum(e["records"] for e in epochs), 96)
        results = runtime.fast.parse_request_jsonl((self.path / "responses.jsonl").read_text())
        self.assertEqual({r["task_id"] for r in results}, {str(i) for i in range(32)})
        self.assertTrue(all(base64.b64decode(r["raw_body_base64"]) == self.raw for r in results))
        self.assertTrue(all(r["payload_sha256"] == runtime.stable_hash(self.rows[int(r["task_id"])]["payload"]) for r in results))

    def test_prepare_failure_has_no_intents_or_generations(self):
        calls = []
        def http(url, payload, timeout, return_raw=False):
            calls.append(url)
            return {"count": 32768 if payload["messages"][0]["content"].startswith("text31") else 10}
        with patch.object(runtime, "request_json", side_effect=http), self.assertRaises(ValueError):
            self.execute()
        self.assertEqual(len(calls), 32)
        self.assertTrue(all(url.endswith("/tokenize") for url in calls))
        self.assertFalse((self.path / "started_requests.jsonl").read_bytes())

    def test_clock_cutoff_before_and_after_prepare_leaves_no_intent(self):
        receipt = {}
        with patch.object(runtime.time, "monotonic", return_value=605.001), patch.object(runtime, "request_json") as http:
            self.assertFalse(self.execute(deadline=1000, receipt=receipt))
            http.assert_not_called()
        self.assertEqual(receipt["clean_stop"]["required_seconds"], 395)
        clock = {"now": 580}
        def tokenize(*unused, **kwargs):
            clock["now"] = 786  #214seconds left: generation180 +teardown35 cannot fit
            return {"count": 10}
        with patch.object(runtime.time, "monotonic", side_effect=lambda: clock["now"]), patch.object(runtime, "request_json", side_effect=tokenize):
            self.assertFalse(self.execute(deadline=1000, receipt=receipt))
        self.assertEqual(receipt["clean_stop"]["boundary"], "AFTER_PREPARATION_BEFORE_INTENT")
        self.assertFalse((self.path / "started_requests.jsonl").read_bytes())

    def test_malformed_response_remains_in_raw_and_parsed_journals(self):
        def http(url, payload, timeout, return_raw=False):
            return {"count": 10} if url.endswith("/tokenize") else (None, b"\xffbad", 200)
        with patch.object(runtime, "request_json", side_effect=http), self.assertRaises(RuntimeError):
            self.execute(self.rows[:1])
        for name in ("http_responses.jsonl", "responses.jsonl"):
            row = json.loads((self.path / name).read_bytes())
            self.assertEqual(row["http_status"], 200)
            self.assertEqual(base64.b64decode(row["raw_body_base64"]), b"\xffbad")
        self.assertEqual(row["status"], "MALFORMED_JSON_RESPONSE")

    def test_committed_intent_cannot_authorize_second_generation(self):
        sent = []
        def http(url, payload, timeout, return_raw=False):
            if url.endswith("/tokenize"):
                return {"count": 10}
            sent.append(payload)
            return None, self.raw, 200
        with patch.object(runtime, "request_json", side_effect=http):
            prepared = runtime.prepare_request(2, self.rows[0], self.args, "http://fixture")
            coordinator = runtime.coordinator_for(self.args)
            intent = coordinator.submit("STARTED", prepared["key"], prepared["result"])
            coordinator.wait_all([intent])
            runtime.execute_prepared(prepared, intent, coordinator, self.args, "http://fixture", threading.Lock())
            with self.assertRaisesRegex(journal.JournalFailure, "already claimed"):
                runtime.execute_prepared(prepared, intent, coordinator, self.args, "http://fixture", threading.Lock())
        self.assertEqual(len(sent), 1)

    def test_independent_coordinators_do_not_confuse_reused_numeric_object_ids(self):
        sent = []
        def http(url, payload, timeout, return_raw=False):
            if url.endswith("/tokenize"):
                return {"count": 10}
            sent.append(payload)
            return None, self.raw, 200
        # Force the old numeric-id implementation's collision deterministically;
        # distinct coordinator objects must remain distinct invocation owners.
        with patch.object(runtime, "id", return_value=7, create=True), patch.object(runtime, "request_json", side_effect=http):
            prepared = runtime.prepare_request(2, self.rows[0], self.args, "http://fixture")
            for name in ("first", "second"):
                fixture = Fixture(self.path / name)
                intent = fixture.coordinator.submit("STARTED", prepared["key"], prepared["result"])
                fixture.coordinator.wait_all([intent])
                runtime.execute_prepared(prepared, intent, fixture.coordinator, self.args, "http://fixture", threading.Lock())
                fixture.coordinator.flush_and_close(3)
        self.assertEqual(len(sent), 2)

    def test_raw_commit_failure_drains_peers_without_parse_or_next_wave(self):
        self.args.max_num_seqs = 2
        barrier = threading.Barrier(2, timeout=3)
        count, sent = [], []
        def commit(*unused):
            count.append(1)
            if len(count) > 1:
                raise OSError("fixture raw commit failed")
        def http(url, payload, timeout, return_raw=False):
            if url.endswith("/tokenize"):
                return {"count": 10}
            sent.append(payload)
            barrier.wait()
            return None, b"not parsed", 200
        with patch.object(runtime.fast, "sync_volume", side_effect=commit), patch.object(runtime, "request_json", side_effect=http), self.assertRaises(journal.JournalFailure):
            self.execute(self.rows[:3])
        self.assertEqual(len(sent), 2)
        self.assertFalse((self.path / "responses.jsonl").read_bytes())
        records = runtime.fast.parse_request_jsonl((self.path / "emergency_http_responses.jsonl").read_text())
        self.assertEqual(len(records), 2)
        self.assertTrue(all(base64.b64decode(r["raw_body_base64"]) == b"not parsed" for r in records))

    def test_raw_and_parsed_share_absolute_deadline_preserving_server_teardown(self):
        waits = []
        ticket = types.SimpleNamespace(owner=types.SimpleNamespace(wait_seconds=20),
                                       wait_committed=lambda timeout: waits.append(timeout))
        with patch.object(runtime.time, "monotonic", return_value=965):
            runtime.wait_phase(ticket, 1000)
        with patch.object(runtime.time, "monotonic", return_value=968):
            runtime.wait_phase(ticket, 1000)
        with patch.object(runtime.time, "monotonic", return_value=970):
            runtime.wait_phase(ticket, 1000)
        self.assertEqual(waits, [4, 1, 0])


if __name__ == "__main__":
    unittest.main()
