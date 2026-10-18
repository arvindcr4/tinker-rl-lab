"""Clean-checkout regressions without import caches or generated repository files."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_isolated(code, tmp_path, *args):
    env = {**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl-cache")}
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, *map(str, args)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_bound_power_loader_resolves_exact_local_contract(tmp_path):
    checkout = tmp_path / "checkout"
    audit = checkout / "zvf-program/audit"
    audit.mkdir(parents=True)
    for name in ("aggregate_audit.py", "audit_evidence_contract.py"):
        shutil.copy2(ROOT / "zvf-program/audit" / name, audit / name)
    run_isolated(
        """
import importlib.util
import pathlib
import sys
import types

verifier_path, checkout = map(pathlib.Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location('isolated_verifier', verifier_path)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
# A cached unrelated module must not replace the exact sibling implementation.
foreign = types.ModuleType('audit_evidence_contract')
sys.modules['audit_evidence_contract'] = foreign
before = sys.path[:]
relative = 'zvf-program/audit/aggregate_audit.py'
power = verifier._load_power_implementation(checkout, {
    'power_implementation_path': relative,
    'power_implementation_sha256': verifier.sha256(checkout / relative),
})
assert sys.path == before
assert sys.modules['audit_evidence_contract'] is foreign
assert power.record_errors({}, {}, heldout_n=0) == ['positive heldout_n required']
assert callable(power.minimum_paired_t_sample_size)
assert callable(power.paired_t_power_one_sided)
""",
        tmp_path,
        ROOT / "zvf-program/next-submission/verify_design.py",
        checkout,
    )
    assert {p.name for p in audit.iterdir()} <= {
        "aggregate_audit.py",
        "audit_evidence_contract.py",
        "__pycache__",
    }


def test_gradient_import_does_not_create_output_directories(tmp_path):
    platform = tmp_path / "checkout/platform_modal"
    scripts = platform / "scripts"
    scripts.mkdir(parents=True)
    module = scripts / "zvf_gradient_coupling.py"
    shutil.copy2(ROOT / "platform_modal/scripts/zvf_gradient_coupling.py", module)
    run_isolated(
        r"""
import importlib.util
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('isolated_gradient', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.tex_escape('a_b%') == r'a\_b\%'
assert not module.PAPER.exists()
assert not module.FIGS.exists()
assert not module.RESULTS.exists()
""",
        tmp_path,
        module,
    )
    assert {p.name for p in platform.iterdir()} == {"scripts"}
