"""New launcher guards evaluated locally without importing/allocating Modal."""
import ast
import datetime as dt
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "zvf-program/flagship/modal_public_portfolio_group_runtime.py"
FAST_SOURCE = SOURCE.with_name("modal_public_portfolio_fast_runtime.py")
spec = importlib.util.spec_from_file_location("group_launcher_actor_fixture", ROOT / ".codex-run/public_colab_runtime_fast.py")
actor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(actor)


def helpers():
    fast_tree = ast.parse(FAST_SOURCE.read_text())
    validator = next(n for n in fast_tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate_profile_reservation")
    namespace = {"load_runtime": lambda: actor, "INCLUSIVE_HOURLY_USD": 5.751216}
    exec(compile(ast.Module(body=[validator], type_ignores=[]), str(FAST_SOURCE), "exec"), namespace)
    frozen = types.SimpleNamespace(validate_profile_reservation=namespace["validate_profile_reservation"])
    tree = ast.parse(SOURCE.read_text())
    names = {"runtime_profile", "validate_profile_reservation", "validate_inputs", "runtime_source_path", "frozen_modal_source_path", "stop_marker_path", "main"}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for function in functions:
        function.decorator_list = []
    namespace = {"frozen": frozen, "load_runtime": lambda: actor, "Path": Path,
                 "REMOTE_RUNTIME": "/root/public_colab_runtime_group.py", "json": json}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def funds(profile):
    return {"status": "RESERVED", "reservation_id": "LOCAL_FIXTURE_NOT_REAL_SPEND",
        "provider": "modal", "unit": "USD", "reserved_units": profile["reserved_usd"],
        "max_hourly_units": 8, "max_wall_seconds": profile["total_seconds"],
        "max_requests": profile["max_requests"], "runtime_profile": profile["name"],
        "expires_at": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=4)).isoformat()}


class GroupLauncherTests(unittest.TestCase):
    def test_separate_profiles_enforce_total_time_requests_and_inclusive_reservation(self):
        namespace = helpers()
        for purpose, seconds, dollars, requests in [("canary", 1800, 4, 34), ("fullbatch", 3600, 8, 4430)]:
            profile = namespace["runtime_profile"](purpose)
            self.assertEqual(profile["name"], "group-" + purpose)
            self.assertEqual(profile["function_seconds"] + profile["startup_seconds"], seconds)
            self.assertEqual(profile["reserved_usd"], dollars)
            self.assertEqual(profile["max_requests"], requests)
            reservation = funds(profile)
            namespace["validate_profile_reservation"](profile, reservation, requests)
            for key, value in [("reserved_units", dollars - .01), ("reserved_units", dollars + .01),
                    ("max_wall_seconds", seconds - 1), ("max_wall_seconds", seconds + 1),
                    ("max_requests", requests + 1), ("runtime_profile", purpose),
                    ("max_hourly_units", 5.0), ("provider", "colab")]:
                with self.subTest(purpose=purpose, key=key, value=value), self.assertRaises(ValueError):
                    namespace["validate_profile_reservation"](profile, {**reservation, key: value}, requests)
            with self.assertRaises(ValueError):
                namespace["validate_profile_reservation"](profile, reservation, requests + 1)

    def test_canary_requires_32_synthetic_requests_and_fullbatch_excludes_them(self):
        namespace = helpers()
        early = {"mode": "online", "initialized_before_model_work": True, "run_id": "fixture"}
        profile = namespace["runtime_profile"]("canary")
        namespace["PROFILE"] = profile
        namespace["validate_inputs"]("", funds(profile), early, True)
        for rows, perf in [("", False), ('{"task_id":"native"}\n', True)]:
            with self.assertRaises(ValueError):
                namespace["validate_inputs"](rows, funds(profile), early, perf)
        with self.assertRaises(ValueError):
            namespace["validate_inputs"]("", funds(profile), {**early, "run_id": ""}, True)
        profile = namespace["runtime_profile"]("fullbatch")
        namespace["PROFILE"] = profile
        with self.assertRaises(ValueError):
            namespace["validate_inputs"]("", funds(profile), early, True)

    def test_native_full_lab_count_and_unicode_payload_unchanged(self):
        namespace = helpers()
        profile = namespace["runtime_profile"]("fullbatch")
        namespace["PROFILE"] = profile
        early = {"mode": "online", "initialized_before_model_work": True, "run_id": "fixture"}
        path = ROOT / "outputs/public_portfolio_2026-09-05/labbench_run/batch-001.jsonl"
        source = path.read_text()
        namespace["validate_inputs"](source, funds(profile), early, False)
        self.assertEqual(len(actor.parse_request_jsonl(source)), 1967)
        row = {"text": "before\u2028middle\u2029after"}
        source = json.dumps(row, ensure_ascii=False) + "\n"
        self.assertEqual(actor.parse_request_jsonl(source), [row])

    def test_remote_flattened_source_path_and_distinct_stop_marker(self):
        namespace = helpers()
        self.assertEqual(namespace["runtime_source_path"]("/root/module.py", False), Path("/root/public_colab_runtime_group.py"))
        self.assertEqual(namespace["frozen_modal_source_path"]("/other/automount/group.py", False),
                         Path("/root/modal_public_portfolio_fast_runtime.py"))
        self.assertEqual(namespace["frozen_modal_source_path"]("/local/source/group.py", True),
                         Path("/local/source/modal_public_portfolio_fast_runtime.py"))
        self.assertTrue(namespace["stop_marker_path"]("fixture").startswith("/public-portfolio-group-stops/"))

    def test_one_gpu_function_no_retries_all_sources_packaged(self):
        source = SOURCE.read_text()
        tree = ast.parse(source)
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and any(
            isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "function" for d in n.decorator_list)]
        self.assertEqual([n.name for n in functions], ["run_batch"])
        decorator = functions[0].decorator_list[0]
        values = {k.arg: ast.literal_eval(k.value) for k in decorator.keywords if isinstance(k.value, ast.Constant)}
        self.assertEqual(values["gpu"], "H200")
        self.assertEqual(values["retries"], 0)
        self.assertEqual(values["max_containers"], 1)
        self.assertTrue(values["single_use_containers"])
        self.assertIn("frozen.image", source)
        image_assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                                and any(isinstance(t, ast.Name) and t.id == "image" for t in n.targets))
        mounts = []
        node = image_assignment.value
        while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_local_file":
            mounts.append((ast.unparse(node.args[0]), ast.unparse(node.args[1]),
                           {k.arg: ast.literal_eval(k.value) for k in node.keywords}))
            node = node.func.value
        self.assertEqual(mounts, [
            ("str(RUNTIME_SCRIPT)", "REMOTE_RUNTIME", {"copy": True}),
            ("str(COORDINATOR_SCRIPT)", "REMOTE_COORDINATOR", {"copy": True}),
            ("str(FROZEN_MODAL_PATH)", "'/root/modal_public_portfolio_fast_runtime.py'", {"copy": True})])
        self.assertEqual(ast.unparse(node), "frozen.image")
        for filename in ("modal_public_portfolio_fast_runtime.py", "public_colab_runtime_group.py", "public_runtime_journal_group_commit.py"):
            self.assertIn(filename, source)
        self.assertIn("source_sha256 = source_manifest()", source)
        self.assertNotIn("def run_agentdojo", source)

    def test_client_timeout_cancels_and_stops_transient_app_noninteractively(self):
        namespace = helpers()
        profile = namespace["runtime_profile"]("canary")
        namespace["PROFILE"] = profile
        call = types.SimpleNamespace(get=Mock(side_effect=TimeoutError()), cancel=Mock())
        subprocess = types.SimpleNamespace(run=Mock())
        namespace.update({"TOTAL_RESERVED_SECONDS": 1800, "source_manifest": lambda: {"fixture": "hash"},
            "run_batch": types.SimpleNamespace(spawn=Mock(return_value=call)), "subprocess": subprocess,
            "sys": types.SimpleNamespace(executable="fixture-python"), "app": types.SimpleNamespace(app_id="fixture-app")})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "reservation").write_text(json.dumps(funds(profile)))
            (path / "wandb").write_text(json.dumps({"mode": "online", "initialized_before_model_work": True, "run_id": "fixture"}))
            with self.assertRaises(TimeoutError):
                namespace["main"](str(path / "reservation"), str(path / "wandb"), str(path / "output"), performance_smoke=True)
            self.assertEqual(json.loads((path / "output").read_text())["score"], None)
        call.get.assert_called_once_with(timeout=1800)
        call.cancel.assert_called_once_with(terminate_containers=True)
        self.assertEqual(subprocess.run.call_args.args[0], ["fixture-python", "-m", "modal", "app", "stop", "--yes", "fixture-app"])


if __name__ == "__main__":
    unittest.main()
