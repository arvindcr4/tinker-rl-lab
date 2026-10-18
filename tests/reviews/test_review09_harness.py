"""Exercise shell launch boundaries without Git side effects or agent services."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def executable(path, source):
    path.write_text(source)
    path.chmod(0o755)


def setup_harness(tmp_path):
    root = tmp_path / "repo with spaces"
    harness = root / "tools/harness"
    shutil.copytree(ROOT / "tools/harness", harness)
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CAPTURE": str(tmp_path / "args.json"),
    }
    return root, harness, bin_dir, env


@pytest.mark.parametrize("tmux_failure", [False, True])
def test_tmux_preserves_argv_reports_child_exit_and_propagates_setup_failure(
    tmp_path, tmux_failure
):
    root, harness, bin_dir, env = setup_harness(tmp_path)
    executable(
        harness / "ralph.sh",
        f"""#!{sys.executable}
import json, os, sys
open(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))
sys.exit(7)
""",
    )
    executable(
        bin_dir / "tmux",
        f"""#!{sys.executable}
import subprocess, sys
if sys.argv[1] == 'has-session': sys.exit(1)
if {tmux_failure!r}: sys.exit(19)
r = subprocess.run(['bash', '-c', sys.argv[-1]], input='\\n', text=True, capture_output=True)
print(r.stdout, end='')
# Creating the tmux session succeeds independently of child completion.
sys.exit(0)
""",
    )
    args = [
        "--verify",
        "make lint-ruff && make test",
        "--workdir",
        str(root),
        "--task-file",
        "name with ; $(false).md",
    ]
    result = subprocess.run(
        ["bash", str(harness / "run-tmux.sh"), "test", *args],
        env=env,
        text=True,
        capture_output=True,
    )
    if tmux_failure:
        assert result.returncode == 19
        assert "started tmux session" not in result.stdout
        assert not Path(env["CAPTURE"]).exists()
    else:
        assert result.returncode == 0
        assert json.loads(Path(env["CAPTURE"]).read_text()) == args
        assert "[exit 7]" in result.stdout


def test_worktree_failure_never_announces_success(tmp_path):
    _, harness, bin_dir, env = setup_harness(tmp_path)
    executable(bin_dir / "git", "#!/bin/sh\necho 'fatal: invalid base' >&2\nexit 23\n")
    result = subprocess.run(
        ["bash", str(harness / "worktree-new.sh"), "run", "invalid"],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 23
    assert "worktree:" not in result.stdout


def test_hanging_verifier_is_bounded_and_run_state_is_worktree_local(tmp_path):
    root, harness, bin_dir, env = setup_harness(tmp_path)
    executable(
        bin_dir / "git",
        f"""#!{sys.executable}
import sys
from pathlib import Path
root = Path(sys.argv[sys.argv.index('-C')+1])
if 'rev-parse' in sys.argv: print(root / '.git')
elif 'status' in sys.argv: print(' M synthetic-change')
""",
    )
    executable(bin_dir / "muse", "#!/bin/sh\nexit 0\n")
    for name in ("first workdir", "second workdir"):
        work = tmp_path / name
        (work / ".git").mkdir(parents=True)
        result = subprocess.run(
            [
                "bash",
                str(harness / "ralph.sh"),
                "--workdir",
                str(work),
                "--max-iters",
                "1",
                "--allow-dirty",
                "--verify-timeout",
                "1",
                "--verify",
                "sleep 30",
            ],
            env=env,
            text=True,
            capture_output=True,
            timeout=5,
        )
        assert "VERIFY FAILED" in result.stdout, result.stdout + result.stderr
        assert "PASS, committed" not in result.stdout
        state = work / ".git/ralph-run/state"
        assert "ITER=1" in state.read_text()
    assert not (harness / ".ralph-state").exists()
    assert not (harness / "logs").exists()
