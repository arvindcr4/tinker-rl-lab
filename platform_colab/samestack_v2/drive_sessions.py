"""Drive one Colab GPU session through a list of same-stack v2 runs, downloading each result as it lands.

Usage (from the repository root):
  python3 platform_colab/samestack_v2/drive_sessions.py SESSION ppo:4096 ppo_8x8:42 ...
Results land in platform_hybrid/experiments/results/samestack_gsm8k_v2s_colab/ (staging, verified before use).
Set GPU (default L4) and PARALLEL (runs trained at once per session, default 1); BATCH sets how many runs go
to the session per exec as one rolling pool (default PARALLEL), downloaded when that exec returns.
Set STAGE_DIR (repo-relative) to stage elsewhere, e.g. for cross-platform controls.
Runs already present in staging are skipped, so the command can be rerun after a dropped session.
"""

import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODAL_SCRIPT = ROOT / "platform_hybrid/experiments/modal/modal_samestack_gsm8k_cot_v2.py"
RUNNER = ROOT / "platform_colab/samestack_v2/colab_run_arms.py"
STAGE = ROOT / os.environ.get("STAGE_DIR", "platform_hybrid/experiments/results/samestack_gsm8k_v2s_colab")


def colab(*args, check=True):
    print("$ colab", " ".join(args), flush=True)
    return subprocess.run(["colab", *args], check=check)


def main():
    session, runs = sys.argv[1], sys.argv[2:]
    STAGE.mkdir(parents=True, exist_ok=True)
    sessions = subprocess.run(["colab", "sessions"], capture_output=True, text=True).stdout
    if session not in sessions:
        gpu = os.environ.get("GPU", "L4")
        colab("new", "--gpu", gpu, *(["--high-mem"] if gpu == "A100" else []), "-s", session)
    colab("upload", "-s", session, str(MODAL_SCRIPT), "/content/modal_samestack_gsm8k_cot_v2.py")
    colab("upload", "-s", session, str(RUNNER), "/content/colab_run_arms.py")
    todo = []
    for run in runs:
        if (STAGE / f"{run.replace(':', '_s')}.json").exists():
            print(f"skip {run}: already staged", flush=True)
        else:
            todo.append(run)
    par = int(os.environ.get("PARALLEL", "1"))
    batch = int(os.environ.get("BATCH", par))
    for i in range(0, len(todo), batch):
        group = todo[i:i + batch]
        waves = -(-len(group) // par)
        # kernel env persists between execs, so every variable is set each time
        colab("exec", "-s", session, "-f", str(RUNNER), "--timeout", str(14400 * waves), "--env", f"RUNS={','.join(group)}",
              "--env", "SMOKE=0", "--env", f"PARALLEL={par}", check=False)
        for run in group:
            name = run.replace(":", "_s")
            for suffix in (".json", "_texts.json.gz"):
                colab("download", "-s", session, f"/content/results/samestack_gsm8k_v2s/{name}{suffix}", str(STAGE / f"{name}{suffix}"), check=False)
            print(f"{'STAGED' if (STAGE / f'{name}.json').exists() else 'MISSING'} {name}", flush=True)
    colab("stop", "-s", session, check=False)


if __name__ == "__main__":
    main()
