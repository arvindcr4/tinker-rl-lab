"""Importing the supported TRL API must not start telemetry or create artifacts."""

import subprocess
import sys
from pathlib import Path


def test_trl_import_has_no_telemetry_side_effects(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = f"""
import sys
sys.path.insert(0, {str(root)!r})

class RejectTelemetryImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "codecarbon" or fullname.startswith("codecarbon."):
            raise AssertionError("TRL import must not initialize CodeCarbon")
        return None

sys.meta_path.insert(0, RejectTelemetryImports())
from platform_local.trl_integrations import TRLTrainer
assert "codecarbon" not in sys.modules
print(TRLTrainer.__name__)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == "TRLTrainer"
    assert list(tmp_path.iterdir()) == []
