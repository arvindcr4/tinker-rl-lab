"""Drive one Colab GPU session through a list of same-stack v2 runs, downloading each result as it lands.

Usage (from the repository root):
  python3 platform_colab/samestack_v2/drive_sessions.py SESSION ppo:4096 ppo_8x8:42 ...
Results land in platform_hybrid/experiments/results/samestack_gsm8k_v2s_colab/ (staging, verified before use).
Runs already present in staging are skipped, so the command can be rerun after a dropped session.
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODAL_SCRIPT = ROOT / "platform_hybrid/experiments/modal/modal_samestack_gsm8k_cot_v2.py"
RUNNER = ROOT / "platform_colab/samestack_v2/colab_run_arms.py"
STAGE = ROOT / "platform_hybrid/experiments/results/samestack_gsm8k_v2s_colab"


def colab(*args, check=True):
    print("$ colab", " ".join(args), flush=True)
    return subprocess.run(["colab", *args], check=check)


def main():
    session, runs = sys.argv[1], sys.argv[2:]
    STAGE.mkdir(parents=True, exist_ok=True)
    sessions = subprocess.run(["colab", "sessions"], capture_output=True, text=True).stdout
    if session not in sessions:
        colab("new", "--gpu", "L4", "-s", session)
    colab("upload", "-s", session, str(MODAL_SCRIPT), "/content/modal_samestack_gsm8k_cot_v2.py")
    colab("upload", "-s", session, str(RUNNER), "/content/colab_run_arms.py")
    for run in runs:
        arm, seed = run.split(":")
        name = f"{arm}_s{seed}"
        if (STAGE / f"{name}.json").exists():
            print(f"skip {name}: already staged", flush=True)
            continue
        colab("exec", "-s", session, "-f", str(RUNNER), "--timeout", "10800", "--env", f"RUNS={run}", "--env", "SMOKE=0")  # kernel env persists between execs
        for suffix in (".json", "_texts.json.gz"):
            colab("download", "-s", session, f"/content/results/samestack_gsm8k_v2s/{name}{suffix}", str(STAGE / f"{name}{suffix}"))
        print(f"STAGED {name}", flush=True)
    colab("stop", "-s", session, check=False)


if __name__ == "__main__":
    main()
