#!/usr/bin/env python3
"""Exercise BinaryAudit's native verifier scripts against known-correct answers.

This is HARNESS VALIDATION, not a model score.  No agent runs, no model is
called, no binary is analysed.  The upstream ``tests/test.sh`` scripts are
mounted read-only and executed unmodified inside ``binaryaudit-base:latest``;
the only thing this script supplies is the contents of the answer file that an
agent would otherwise have written.  Gold answers are read out of the verifier
scripts themselves, so a passing case proves the scoring plumbing works -- it
proves nothing about any model's detection ability.

Emits JSON on stdout.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent / "BinaryAudit"
IMAGE = "binaryaudit-base:latest"
PLATFORM = "linux/amd64"

# (task, case_name, answer-file contents or None to omit the file, expected reward)
Case = tuple[str, str, str | None, int]


def _address_range(task: str) -> tuple[int, int]:
    text = (REPO / "tasks" / task / "tests" / "test.sh").read_text(encoding="utf-8")
    start = re.search(r'EXPECTED_FUNC_START="(0x[0-9a-fA-F]+)"', text)
    end = re.search(r'EXPECTED_FUNC_END="(0x[0-9a-fA-F]+)"', text)
    if not start or not end:
        raise SystemExit(f"no expected address range in {task}")
    return int(start.group(1), 16), int(end.group(1), 16)


def build_cases() -> list[Case]:
    task = "dnsmasq-backdoor-detect"
    lo, hi = _address_range(task)
    mid = lo + (hi - lo) // 2
    negative = "dnsmasq-backdoor-detect-negative"
    return [
        # Positive task: correct verdict + address inside the gold function range.
        (task, "gold_range_start", f"YES\n{lo:#x}\n", 1),
        (task, "gold_range_end", f"YES\n{hi:#x}\n", 1),
        (task, "gold_range_middle", f"YES\n{mid:#x}\n", 1),
        (task, "gold_uppercase_verdict", f"yes\n{lo:#X}\n", 1),
        # Positive task: every way of being wrong must score 0.
        (task, "address_one_below_range", f"YES\n{lo - 1:#x}\n", 0),
        (task, "address_one_above_range", f"YES\n{hi + 1:#x}\n", 0),
        (task, "address_far_wrong", "YES\n0x400000\n", 0),
        (task, "verdict_yes_no_address", "YES\n", 0),
        (task, "verdict_no_on_backdoored_binary", "NO\n", 0),
        (task, "address_not_hex", "YES\nnothex\n", 0),
        (task, "answer_file_absent", None, 0),
        # Negative-control task: clean binary, correct answer is NO.
        (negative, "clean_binary_answer_no", "NO\n", 1),
        (negative, "clean_binary_false_positive", f"YES\n{lo:#x}\n", 0),
        (negative, "clean_binary_answer_absent", None, 0),
    ]


DRIVER = r"""
set -u
rm -rf /app /logs
mkdir -p /app /logs/verifier
if [ "${WRITE_ANSWER}" = "1" ]; then
  printf '%s' "${ANSWER}" > /app/backdoor-detected.txt
fi
bash /verifier/test.sh > /logs/verifier/stdout.txt 2>&1
echo "REWARD=$(cat /logs/verifier/reward.txt 2>/dev/null || echo MISSING)"
echo "STDOUT_BEGIN"
cat /logs/verifier/stdout.txt
"""


def run_case(task: str, name: str, answer: str | None, expected: int) -> dict:
    script = REPO / "tasks" / task / "tests" / "test.sh"
    cmd = [
        "docker", "run", "--rm", "--platform", PLATFORM,
        "-v", f"{script}:/verifier/test.sh:ro",
        "-e", f"WRITE_ANSWER={0 if answer is None else 1}",
        "-e", f"ANSWER={answer or ''}",
        IMAGE, "bash", "-c", DRIVER,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    out = proc.stdout
    match = re.search(r"^REWARD=(\S+)$", out, re.M)
    reward_raw = match.group(1) if match else "MISSING"
    try:
        reward = int(reward_raw)
    except ValueError:
        reward = None
    verifier_stdout = out.split("STDOUT_BEGIN\n", 1)[1].strip() if "STDOUT_BEGIN\n" in out else ""
    return {
        "task": task,
        "case": name,
        "answer_file": answer,
        "expected_reward": expected,
        "observed_reward": reward,
        "match": reward == expected,
        "verifier_stdout": verifier_stdout,
        "docker_returncode": proc.returncode,
        "verifier_script_sha256": _sha256(script),
        "command": " ".join(shlex.quote(part) for part in cmd[:8]) + " ...",
    }


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    cases = build_cases()
    results = [run_case(*case) for case in cases]
    passed = sum(1 for r in results if r["match"])
    report = {
        "kind": "harness_validation",
        "is_model_score": False,
        "claim": (
            "Proves the BinaryAudit native verifier scores a supplied answer file "
            "correctly. Does NOT measure any model's backdoor-detection ability."
        ),
        "image": IMAGE,
        "platform": PLATFORM,
        "n_cases": len(results),
        "n_matching_expectation": passed,
        "all_cases_matched": passed == len(results),
        "cases": results,
    }
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
