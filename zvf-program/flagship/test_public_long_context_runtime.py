"""Offline long-context tail tests. No provider, model, or GPU allocation."""

import ast
import copy
import datetime as dt
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import sys
import os
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch
import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".codex-run/public_colab_runtime_group_long_context.py"
spec = importlib.util.spec_from_file_location("test_long_context_group", SOURCE)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
launcher_path = ROOT / "zvf-program/flagship/modal_public_portfolio_long_context_runtime.py"

# The unchanged durable coordinator is also tested through the new module.
s = importlib.util.spec_from_file_location(
    "retained_group_barrier_tests", Path(__file__).with_name("test_public_runtime_group_commit.py")
)
inherited = importlib.util.module_from_spec(s)
s.loader.exec_module(inherited)
inherited.runtime = runtime
inherited.journal = runtime.journal


class RetainedBarrierTests(inherited.GroupCommitTests):
    pass


class LongContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = types.SimpleNamespace(
            output_dir=self.root,
            served_model_name=runtime.fast.SERVED_MODEL,
            request_seconds=180,
            max_model_len=65536,
            max_num_seqs=32,
            commit_volume=None,
            stop_marker=None,
            stop_marker_volume_path=None,
        )
        self.receipt = {"requests_input_sha256": "fixture-only", "requests_completed": 2}
        self.rows = [
            {
                "task_id": f"fixture-{i}",
                "custom_id": f"fixture-{i}",
                "kind": "actor",
                "batch_id": "b" * 64,
                "contract_sha256": "c" * 64,
                "request_sha256": str(i) * 64,
                "api_path": "/v1/chat/completions",
                "payload": {
                    "model": runtime.fast.SERVED_MODEL,
                    "messages": [{"role": "user", "content": str(i)}],
                    "max_tokens": 1024,
                    "temperature": 0,
                    "top_p": 1,
                    "n": 1,
                    "stream": False,
                },
            }
            for i in range(4)
        ]
        self.events = []
        self.lock = threading.Lock()
        self.patches = [
            patch.object(runtime.fast, "sync_volume", return_value=None),
            patch.object(runtime.fast, "stop_requested", return_value=False),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.close)

    def close(self):
        for key, coordinator in list(runtime.COORDINATORS.items()):
            if key == str(self.root.resolve()):
                try:
                    coordinator.flush_and_close(timeout=1)
                except runtime.journal.JournalFailure:
                    pass
                del runtime.COORDINATORS[key]

    def tokenize(self, url, payload, timeout):
        assert url.endswith("/tokenize")
        with self.lock:
            self.events.append(("tokenize", payload["messages"][0]["content"]))
        return {"count": 40000}

    def test_entire_tail_preflight_before_any_native_intent_and_exact_payloads(self):
        payloads = copy.deepcopy([r["payload"] for r in self.rows])

        def generate(prepared, ticket, coordinator, args, base_url, lock, deadline):
            ticket.wait_committed()
            with self.lock:
                assert sum(e[0] == "tokenize" for e in self.events) == len(self.rows)
                self.events.append(("generate", prepared["result"]["task_id"]))
            # Generation is a fixture callback. Real transport is covered by retained barrier tests.
            return {"status": "HTTP_RESPONSE_RECORDED", "elapsed_seconds": 0}

        with (
            patch.object(runtime, "request_json", side_effect=self.tokenize),
            patch.object(runtime, "execute_prepared", side_effect=generate),
        ):
            done = runtime.execute_batch_waves(
                self.rows,
                self.args,
                "http://fixture",
                io.StringIO(),
                io.StringIO(),
                self.lock,
                self.receipt,
                Mock(),
                time.monotonic() + 1200,
            )
        assert done
        assert [r["payload"] for r in self.rows] == payloads
        proof = json.loads((self.root / "tail_tokenization_preflight.json").read_text())
        assert proof["status"] == "ALL_TAIL_CONTEXT_VALIDATED"
        assert proof["max_prompt_tokens"] == 40000
        assert proof["max_prompt_plus_completion_tokens"] == 41024
        assert proof["native_generation_intents"] == 0
        assert proof["tokenized_rows"] == 4
        assert not proof["prompt_truncation_allowed"]

    def test_oversized_tail_still_tokenizes_every_row_and_starts_none(self):
        def tokenize(url, payload, timeout):
            answer = self.tokenize(url, payload, timeout)
            if payload["messages"][0]["content"] == "1":
                answer["count"] = 65000
            return answer

        with (
            patch.object(runtime, "request_json", side_effect=tokenize),
            patch.object(runtime, "execute_prepared") as generate,
        ):
            with pytest.raises(ValueError, match="no native intents"):
                runtime.execute_batch_waves(
                    self.rows,
                    self.args,
                    "http://fixture",
                    None,
                    None,
                    self.lock,
                    self.receipt,
                    Mock(),
                    time.monotonic() + 1200,
                )
        generate.assert_not_called()
        assert len(self.events) == 4
        assert not (self.root / "started_requests.jsonl").exists()
        proof = json.loads((self.root / "tail_tokenization_preflight.json").read_text())
        assert proof["oversized_task_ids"] == ["fixture-1"]
        assert proof["max_prompt_plus_completion_tokens"] == 66024

    def test_tokenizer_failure_is_recorded_and_blocks_all_generation(self):
        def tokenize(url, payload, timeout):
            answer = self.tokenize(url, payload, timeout)
            if payload["messages"][0]["content"] == "2":
                raise TimeoutError("fixture")
            return answer

        with (
            patch.object(runtime, "request_json", side_effect=tokenize),
            patch.object(runtime, "execute_prepared") as generate,
        ):
            with pytest.raises(ValueError):
                runtime.execute_batch_waves(
                    self.rows,
                    self.args,
                    "http://fixture",
                    None,
                    None,
                    self.lock,
                    self.receipt,
                    Mock(),
                    time.monotonic() + 1200,
                )
        generate.assert_not_called()
        assert len(self.events) == 4
        proof = json.loads((self.root / "tail_tokenization_preflight.json").read_text())
        assert proof["processed_rows"] == 4
        assert proof["tokenized_rows"] == 3
        assert proof["errors"][0]["error_type"] == "TimeoutError"
        assert not (self.root / "started_requests.jsonl").exists()

    def test_guard_stops_before_preflight_with_no_native_intent(self):
        with (
            patch.object(runtime, "request_json") as tokenize,
            patch.object(runtime, "execute_prepared") as generate,
        ):
            done = runtime.execute_batch_waves(
                self.rows,
                self.args,
                "http://fixture",
                None,
                None,
                self.lock,
                self.receipt,
                Mock(),
                time.monotonic() + 394,
            )
        assert not done
        tokenize.assert_not_called()
        generate.assert_not_called()
        assert self.receipt["clean_stop"]["next_unstarted_row_index"] == 0
        assert self.receipt["clean_stop"]["active_requests"] == 0

    def test_stop_after_full_preflight_leaves_all_native_rows_unstarted(self):
        sequence = iter([False, True])
        with (
            patch.object(runtime, "request_json", side_effect=self.tokenize),
            patch.object(runtime.fast, "stop_requested", side_effect=lambda _: next(sequence)),
            patch.object(runtime, "execute_prepared") as generate,
        ):
            done = runtime.execute_batch_waves(
                self.rows,
                self.args,
                "http://fixture",
                None,
                None,
                self.lock,
                self.receipt,
                Mock(),
                time.monotonic() + 1200,
            )
        assert not done
        generate.assert_not_called()
        assert self.receipt["clean_stop"]["next_unstarted_row_index"] == 0
        assert self.receipt["tail_tokenization_preflight"]["status"] == "ALL_TAIL_CONTEXT_VALIDATED"

    def test_generation_transport_and_journal_logic_are_exact_original_definitions(self):
        original = ast.parse((ROOT / ".codex-run/public_colab_runtime_group.py").read_text())
        changed = ast.parse(SOURCE.read_text())
        for name in [
            "execute_prepared",
            "phase_timeout",
            "wait_phase",
            "coordinator_for",
            "clean_boundary",
            "execute_request",
        ]:
            old = next(
                n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name
            )
            new = next(n for n in changed.body if isinstance(n, ast.FunctionDef) and n.name == name)
            assert ast.dump(old, include_attributes=False) == ast.dump(
                new, include_attributes=False
            ), name
        a = ast.parse((ROOT / ".codex-run/public_colab_runtime_fast.py").read_text())
        b = ast.parse((ROOT / ".codex-run/public_colab_runtime_long_context.py").read_text())
        for name in [
            "server_command",
            "smoke_requests",
            "request_json",
            "check_merge_identity",
            "check_live_hf",
            "stop_process",
        ]:
            old = next(n for n in a.body if isinstance(n, ast.FunctionDef) and n.name == name)
            new = next(n for n in b.body if isinstance(n, ast.FunctionDef) and n.name == name)
            assert ast.dump(old, include_attributes=False) == ast.dump(
                new, include_attributes=False
            ), name

    def test_new_context_ceiling_and_native_capacity_are_explicit(self):
        text = (ROOT / ".codex-run/public_colab_runtime_long_context.py").read_text()
        assert "args.max_model_len != 65536" in text
        assert "native_context != 262144" in text
        assert "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99" in text
        assert '"prompt_truncation_allowed": False' in text
        assert runtime.fast.GRAPH_CONFIG == inherited.runtime.fast.GRAPH_CONFIG

    def test_exact_context_cli_gate_runs_without_model_or_provider_import(self):
        funds = {
            "status": "RESERVED",
            "reservation_id": "LOCAL_CLI_FIXTURE",
            "provider": "modal",
            "unit": "USD",
            "reserved_units": 4,
            "max_hourly_units": 8,
            "max_wall_seconds": 1800,
            "max_requests": 225,
            "expires_at": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=2)).isoformat(),
        }
        reservation = self.root / "reservation.json"
        reservation.write_text(json.dumps(funds))
        wandb = self.root / "wandb.json"
        wandb.write_text(
            json.dumps(
                {"mode": "online", "initialized_before_model_work": True, "run_id": "fixture"}
            )
        )
        for context, expected in [(65536, 0), (32768, 1), (65537, 1)]:
            output = self.root / f"validate-{context}"
            argv = [
                "fixture",
                "--model-dir",
                str(self.root / "ABSENT_MODEL"),
                "--merge-receipt",
                str(self.root / "ABSENT_MERGE"),
                "--reservation",
                str(reservation),
                "--wandb-receipt",
                str(wandb),
                "--output-dir",
                str(output),
                "--max-model-len",
                str(context),
                "--validate-only",
            ]
            with self.subTest(context=context), patch.object(sys, "argv", argv):
                assert runtime.fast.main() == expected
            assert not (output / "started_requests.jsonl").exists()
            receipt = json.loads((output / "runtime_receipt.json").read_text())
            assert receipt["status"] == (
                "LOCAL_GATES_VALIDATED_NOT_LOADED" if expected == 0 else "RUNTIME_FAILED"
            )

    def test_actual_pinned_config_allows65536_without_rope_override_and_rejects_drift(self):
        actual = ROOT / "outputs/public_portfolio_2026-09-05/merged_model_config_context.json"
        proof = runtime.fast.check_context_capacity(actual, 65536)
        assert proof["native_max_position_embeddings"] == 262144
        assert proof["rope_scaling"] is None
        for context in (32768, 65537):
            with pytest.raises(ValueError):
                runtime.fast.check_context_capacity(actual, context)
        changed = self.root / "changed-config.json"
        data = json.loads(actual.read_text())
        data["text_config"]["max_position_embeddings"] = 32768
        changed.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="hash"):
            runtime.fast.check_context_capacity(changed, 65536)

    def test_client_cleanup_on_uncertain_spawn_interrupt_and_failed_result_write(self):
        tree = ast.parse(launcher_path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        fn.decorator_list = []
        for failure in ("spawn", "interrupt", "result_write", "failure_write", "cancel"):
            directory = self.root / failure
            directory.mkdir()
            reservation = directory / "funds"
            reservation.write_text(json.dumps({"reservation_id": "fixture"}))
            wandb = directory / "wandb"
            wandb.write_text("{}")
            requests = directory / "requests"
            requests.write_text("fixture")
            output = directory / "result"
            call = types.SimpleNamespace(
                get=Mock(return_value={"receipt": {"status": "done"}, "responses": []}),
                cancel=Mock(),
            )
            spawn = Mock(return_value=call)
            stop = Mock()
            if failure == "spawn":
                spawn.side_effect = RuntimeError("uncertain spawn")
            elif failure == "interrupt":
                call.get.side_effect = KeyboardInterrupt()
            elif failure in ("failure_write", "cancel"):
                call.get.side_effect = RuntimeError("RPC failure")
            if failure == "cancel":
                call.cancel.side_effect = ValueError("cancel failure")
            namespace = {
                "Path": Path,
                "json": json,
                "hashlib": hashlib,
                "os": os,
                "sys": sys,
                "__file__": str(launcher_path),
                "validate_inputs": Mock(),
                "source_manifest": lambda: {"fixture": "hash"},
                "PROFILE": {"name": "fixture"},
                "TOTAL_RESERVED_SECONDS": 1800,
                "run_batch": types.SimpleNamespace(spawn=spawn),
                "app": types.SimpleNamespace(app_id="ap-fixture"),
                "subprocess": types.SimpleNamespace(run=stop),
            }
            exec(
                compile(ast.Module(body=[fn], type_ignores=[]), str(launcher_path), "exec"),
                namespace,
            )
            original_open = Path.open

            def failing_open(p, *args, **kwargs):
                if p == output and failure in ("result_write", "failure_write"):
                    raise OSError("disk fixture")
                return original_open(p, *args, **kwargs)

            with self.subTest(failure=failure), patch.object(Path, "open", failing_open):
                with pytest.raises(BaseException):
                    namespace["main"](str(reservation), str(wandb), str(output), str(requests))
            stop.assert_called_once()
            assert stop.call_args.args[0][-4:] == ["app", "stop", "--yes", "ap-fixture"]
            if failure == "spawn":
                call.cancel.assert_not_called()
            else:
                call.cancel.assert_called_once_with(terminate_containers=True)

    def test_launcher_profile_caps_source_mounts_and_context_are_bound(self):
        tree = ast.parse(launcher_path.read_text())
        names = {"runtime_profile", "validate_profile_reservation", "validate_inputs"}
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        namespace = {
            "load_runtime": lambda: runtime.fast,
            "hashlib": hashlib,
            "RESOURCE_LIMITS": {"cpu": [4, 4], "memory_mib": [131072, 131072]},
            "source_manifest": lambda: {"fixture": "source"},
        }
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), str(launcher_path), "exec"),
            namespace,
        )
        profile = namespace["runtime_profile"]()
        namespace["PROFILE"] = profile
        requests = "".join(
            json.dumps({**self.rows[i % 4], "task_id": f"fixture-{i}"}) + "\n" for i in range(223)
        )
        namespace["TAIL_REQUEST_SHA256"] = hashlib.sha256(requests.encode()).hexdigest()
        namespace["TAIL_SELECTION_SHA256"] = "fixture-selection-sha"
        funds = {
            "status": "RESERVED",
            "reservation_id": "LOCAL_FIXTURE_ONLY",
            "provider": "modal",
            "unit": "USD",
            "reserved_units": 4,
            "max_hourly_units": 8,
            "max_wall_seconds": 1800,
            "max_requests": 225,
            "runtime_profile": "group-long-context-tail",
            "explicit_resource_limits": namespace["RESOURCE_LIMITS"],
            "declared_context_window": 65536,
            "tail_selection_receipt_sha256": "fixture-selection-sha",
            "source_dependencies_sha256": {"fixture": "source"},
            "requests_sha256": hashlib.sha256(requests.encode()).hexdigest(),
            "expires_at": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).isoformat(),
        }
        wandb = {"mode": "online", "initialized_before_model_work": True, "run_id": "fixture"}
        namespace["validate_inputs"](requests, funds, wandb)
        for key, value in [
            ("max_requests", 226),
            ("declared_context_window", 32768),
            ("reserved_units", 8),
            ("requests_sha256", "wrong"),
            ("source_dependencies_sha256", {"wrong": "hash"}),
            ("tail_selection_receipt_sha256", "wrong"),
        ]:
            with self.subTest(key=key), pytest.raises(ValueError):
                namespace["validate_inputs"](requests, {**funds, key: value}, wandb)
        with pytest.raises(ValueError):
            namespace["validate_inputs"](requests, funds, wandb, True)
        duplicated = "".join(json.dumps(self.rows[0]) + "\n" for _ in range(223))
        with pytest.raises(ValueError):
            namespace["validate_inputs"](
                duplicated,
                {**funds, "requests_sha256": hashlib.sha256(duplicated.encode()).hexdigest()},
                wandb,
            )
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_batch")
        decorator = fn.decorator_list[0]
        kw = {k.arg: ast.unparse(k.value) for k in decorator.keywords}
        assert kw["cpu"] == "(CPU_CORES, CPU_CORES)"
        assert kw["memory"] == "(MEMORY_GIB * 1024, MEMORY_GIB * 1024)"
        assert kw["retries"] == "0"
        assert kw["single_use_containers"] == "True"
        assert '"--max-model-len", "65536"' in launcher_path.read_text()
        assert "tail_tokenization_preflight" in launcher_path.read_text()


if __name__ == "__main__":
    unittest.main()
