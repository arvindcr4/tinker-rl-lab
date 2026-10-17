"""Unsupported canonical training must not start jobs or report invented steps."""

from unittest.mock import patch

import pytest

from platform_local.unified.canonical import CanonicalSpec
from platform_local.unified.launcher import TrainingResult, UnifiedLauncher


@pytest.mark.parametrize("framework", ["tinker", "skyrl"])
def test_unsupported_canonical_route_fails_before_any_subprocess(framework):
    launcher = UnifiedLauncher()
    launcher.framework = framework
    launcher.spec = CanonicalSpec()
    with (
        patch.dict("os.environ", {"SKYRL_CHECKOUT": "/synthetic/never-accessed"}),
        patch("subprocess.run") as execute,
        pytest.raises(NotImplementedError, match="canonical"),
    ):
        launcher.dispatch_framework()
    execute.assert_not_called()


def test_unknown_step_count_cannot_print_training_complete(capsys):
    launcher = UnifiedLauncher()
    launcher.framework = "trl"
    launcher.spec = CanonicalSpec()
    result = TrainingResult("trl", "synthetic/model", "grpo", None, [], [])
    with patch.object(launcher, "dispatch_framework", return_value=result):
        launcher.run()
    output = capsys.readouterr().out
    assert "Training Complete!" not in output
    assert "unverified" in output.lower()


@pytest.mark.parametrize("framework", ["tinker", "skyrl"])
def test_unsupported_canonical_route_retains_read_only_dry_run(framework, capsys):
    launcher = UnifiedLauncher()
    launcher.framework = framework
    launcher.spec = CanonicalSpec()
    launcher.dry_run = True
    with patch("subprocess.run") as execute:
        launcher.run()
    execute.assert_not_called()
    assert f"local/{framework}" in capsys.readouterr().out
