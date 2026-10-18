from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pilot.launcher import (
    LauncherError,
    _retries_for,
    _run,
    build_campaign_manifest,
    main,
    write_source_bundle,
)
from pilot.protocol import build_screening_plan, load_protocol
import pytest


class LauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_protocol()
        self.temporary = tempfile.TemporaryDirectory(dir=self.protocol.path.parent)
        payload = json.loads(self.protocol.path.read_text())
        payload["status"] = "locked_not_authorized"
        payload["authorization"]["gpu"] = False
        self.locked_path = Path(self.temporary.name) / "pilot_preregistration.json"
        self.locked_path.write_text(json.dumps(payload))
        self.locked_protocol = load_protocol(self.locked_path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_authorized_dag_has_smoke_six_corpora_and_twenty_four_units(self) -> None:
        manifest = build_campaign_manifest(self.protocol)
        assert manifest["status"] == "ready_to_run"
        assert manifest["allocation_allowed"]
        assert manifest["job_count"] == 31
        assert manifest["preflight_job_count"] == 1
        assert manifest["corpus_job_count"] == 6
        assert manifest["unit_job_count"] == 24
        assert manifest["max_parallel_corpus_sessions"] == 1
        assert manifest["max_attempts_per_job"] == 3
        jobs = {job["id"]: job for job in manifest["jobs"]}
        for job in jobs.values():
            assert job["execution_plan"] is not None
            if job["kind"] == "unit":
                assert len(job["depends_on"]) == 1
                assert jobs[job["depends_on"][0]]["kind"] == "corpus"
            elif job["kind"] == "corpus":
                assert job["depends_on"] == ["preflight__a100_stack_smoke"]
                assert job["source_scope"] == "frozen_revision_4_corpus_generator"
                assert job["source_bindings_sha256"] == manifest["corpus_source_bindings_sha256"]
                assert job["source_archive_sha256"] == manifest["corpus_source_archive_sha256"]
            else:
                assert job["source_scope"] == "revision_5_unit_training"
        assert manifest["unit_source_bindings_sha256"] != manifest["corpus_source_bindings_sha256"]

    def test_source_bundle_is_deterministic_and_contains_only_bound_files(self) -> None:
        plan = build_screening_plan(self.protocol, next(self.protocol.screening_units()))
        assert "zvf-program/flagship/pilot/runtime_install.py" in plan["source_bindings"]
        assert (
            "zvf-program/flagship/pilot/provenance/r3-corpus-source.tar.gz"
            in plan["source_bindings"]
        )
        assert (
            "zvf-program/flagship/pilot/provenance/r3-control-source.tar.gz"
            in plan["source_bindings"]
        )
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.tar.gz"
            second = Path(temporary) / "second.tar.gz"
            first_sha = write_source_bundle(first, plan["source_bindings"])
            second_sha = write_source_bundle(second, plan["source_bindings"])
            assert first_sha == second_sha
            assert first.read_bytes() == second.read_bytes()
            with tarfile.open(first, "r:gz") as archive:
                names = {member.name for member in archive.getmembers()}
            expected = {f"tinker-rl-lab/{path}" for path in plan["source_bindings"]}
            assert names == expected

    def test_runtime_install_uses_long_timeout_exec_instead_of_cli_install(self) -> None:
        manifest = build_campaign_manifest(self.protocol)
        commands = manifest["jobs"][0]["execution_plan"]
        assert not any("install" in command[:3] for command in commands)
        installer = commands[4]
        assert installer[-2:] == ["--timeout", "1800"]
        assert installer[installer.index("--file") + 1].endswith("runtime_install.py")

    def test_install_is_followed_by_kernel_restart_before_environment_check(self) -> None:
        manifest = build_campaign_manifest(self.protocol)
        job = manifest["jobs"][0]
        commands = job["execution_plan"]
        restart = commands[5]
        assert restart == ["colab", "--auth=oauth2", "restart-kernel", "--session", job["session"]]
        assert commands[6][-1] == "/content/flagship-pilot-request.json"

    def test_execute_refuses_before_colab_when_protocol_is_locked(self) -> None:
        with mock.patch("subprocess.run") as run:
            with pytest.raises(SystemExit, match="allocation is forbidden"):
                main(
                    [
                        "--protocol",
                        str(self.locked_path),
                        "--execute-job",
                        "corpus__balanced_equal_length__s11",
                    ]
                )
        run.assert_not_called()

    def test_retries_only_idempotent_control_plane_steps(self) -> None:
        base = ["colab", "--auth=oauth2"]
        for subcommand in ("upload", "restart-kernel", "stop"):
            assert _retries_for([*base, subcommand, "--session", "s"]) == 2
        for subcommand in ("new", "exec"):
            assert _retries_for([*base, subcommand, "--session", "s"]) == 0

    def test_run_retries_idempotent_step_then_raises(self) -> None:
        command = ["colab", "--auth=oauth2", "restart-kernel", "--session", "s"]
        failures = mock.Mock(returncode=1)
        with (
            mock.patch("subprocess.run", return_value=failures) as run,
            mock.patch("time.sleep") as sleep,
        ):
            with pytest.raises(LauncherError, match="restart-kernel"):
                _run(command, log=io.StringIO(), retries=_retries_for(command))
        assert run.call_count == 3
        assert sleep.call_count == 2

    def test_run_does_not_retry_non_idempotent_step(self) -> None:
        command = ["colab", "--auth=oauth2", "exec", "--session", "s"]
        failures = mock.Mock(returncode=1)
        with (
            mock.patch("subprocess.run", return_value=failures) as run,
            mock.patch("time.sleep") as sleep,
        ):
            with pytest.raises(LauncherError, match="exec"):
                _run(command, log=io.StringIO(), retries=_retries_for(command))
        run.assert_called_once()
        sleep.assert_not_called()

    def test_locked_dag_contains_no_execution_commands(self) -> None:
        manifest = build_campaign_manifest(self.locked_protocol)
        assert not manifest["allocation_allowed"]
        assert all(job["execution_plan"] is None for job in manifest["jobs"])


if __name__ == "__main__":
    unittest.main()
