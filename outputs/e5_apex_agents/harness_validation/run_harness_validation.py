#!/usr/bin/env python3
"""Run the Archipelago native grading runner against a SYNTHETIC fixture.

This is `harness_validation`, NOT a benchmark run.

  is_model_score: false
  suite score:    null

What it proves: the pinned Archipelago grading verifier
(Mercor-Intelligence/archipelago @ 1c3dcd4694b313020cd626699c9c7cc1c0a2fc58)
installs and executes end to end on this machine -- snapshot diff -> helper
execution -> programmatic verifier -> scoring method -> grades.json -- and that
it discriminates (one verifier is constructed to fail).

What it does NOT prove: anything about APEX-Agents task difficulty, any model's
capability, or any score on the `mercor/apex-agents` suite.  The gated dataset
was never downloaded.  No LLM judge runs: every verifier used is registered
`eval_types=[EvalType.PROGRAMMATIC]`, so no paid API call is made.

Usage:
    python3 run_harness_validation.py
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
E5 = HERE.parent
ARCHIPELAGO = E5 / "sprint.8cEFaN" / "archipelago"
GRADING = ARCHIPELAGO / "grading"
VENV_PYTHON = E5 / "venv-grading" / "bin" / "python"
FIXTURES = HERE / "fixtures"
OUT = HERE / "out"

ARCHIPELAGO_REVISION = "1c3dcd4694b313020cd626699c9c7cc1c0a2fc58"
GRADING_RUN_ID = "gr_harness_validation_001"
TRAJECTORY_ID = "harness_validation_synthetic"

EXPECTED_VERIFIER_SCORES = {
    "ver_pass_at_least": 1.0,
    "ver_fail_at_least": 0.0,
    "ver_pass_exact": 1.0,
}
EXPECTED_FINAL_SCORE = 2.0 / 3.0
EXPECTED_STATUS = "completed"


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(ARCHIPELAGO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def grading_tree_sha256() -> str:
    """Order-stable digest of every tracked file under grading/runner."""
    digest = hashlib.sha256()
    root = GRADING / "runner"
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main() -> int:
    if not VENV_PYTHON.is_file():
        fail(
            f"grading venv missing at {VENV_PYTHON}; create it with:\n"
            f"  cd {GRADING} && UV_PROJECT_ENVIRONMENT={E5/'venv-grading'} "
            "uv sync --locked --python 3.13"
        )

    revision = git_revision()
    if revision != ARCHIPELAGO_REVISION:
        fail(
            f"archipelago checkout is at {revision}, expected the pinned "
            f"{ARCHIPELAGO_REVISION}"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    print("[1/3] building synthetic fixture ...")
    subprocess.run(
        [str(VENV_PYTHON), str(HERE / "build_fixture.py")], check=True, stdout=subprocess.DEVNULL
    )

    grades_path = OUT / "grades.json"
    log_path = OUT / "grading_run.log"
    print("[2/3] running the native grading runner ...")
    # Strip provider keys so an accidental LLM call cannot succeed or bill.
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "VERTEX_PROJECT",
        }
    }
    env["PYTHONPATH"] = str(GRADING)
    completed = subprocess.run(
        [
            str(VENV_PYTHON),
            "-m",
            "runner.main",
            "--grading-run-id",
            GRADING_RUN_ID,
            "--trajectory-id",
            TRAJECTORY_ID,
            "--initial-snapshot",
            str(FIXTURES / "initial_snapshot.zip"),
            "--final-snapshot",
            str(FIXTURES / "final_snapshot.zip"),
            "--trajectory",
            str(FIXTURES / "trajectory.json"),
            "--grading-settings",
            str(FIXTURES / "grading_settings.json"),
            "--verifiers",
            str(FIXTURES / "verifiers.json"),
            "--eval-configs",
            str(FIXTURES / "eval_configs.json"),
            "--scoring-config",
            str(FIXTURES / "scoring_config.json"),
            "--output",
            str(grades_path),
        ],
        cwd=GRADING,
        env=env,
        capture_output=True,
        text=True,
    )
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        fail(f"grading runner exited {completed.returncode}; see {log_path}")

    print("[3/3] asserting expected outcome ...")
    grades = json.loads(grades_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if grades.get("grading_run_status") != EXPECTED_STATUS:
        problems.append(
            f"grading_run_status={grades.get('grading_run_status')!r}, "
            f"expected {EXPECTED_STATUS!r}"
        )
    observed = {v["verifier_id"]: v["score"] for v in grades.get("verifier_results", [])}
    for verifier_id, expected in EXPECTED_VERIFIER_SCORES.items():
        if observed.get(verifier_id) != expected:
            problems.append(
                f"{verifier_id}: score={observed.get(verifier_id)!r}, expected {expected}"
            )
    final_score = grades.get("scoring_results", {}).get("final_score")
    if final_score is None or abs(final_score - EXPECTED_FINAL_SCORE) > 1e-9:
        problems.append(f"final_score={final_score!r}, expected {EXPECTED_FINAL_SCORE}")
    for v in grades.get("verifier_results", []):
        if v.get("status") != "ok":
            problems.append(f"{v['verifier_id']}: status={v.get('status')!r}")

    receipt = {
        "schema_version": "pavlov-e5-apex-agents-harness-validation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite_id": "apex_agents_eval",
        "kind": "harness_validation",
        "is_model_score": False,
        "suite_score": None,
        "status": "PASS" if not problems else "FAIL",
        "claim": (
            "The pinned Archipelago native grading verifier installs and runs "
            "end to end on this host against a synthetic fixture. This is NOT "
            "an APEX-Agents result and implies nothing about any model."
        ),
        "not_claimed": [
            "any APEX-Agents suite score",
            "any model capability measurement",
            "that mercor/apex-agents data was obtained (it was not)",
        ],
        "native_verifier": {
            "repository": "https://github.com/Mercor-Intelligence/archipelago",
            "revision": revision,
            "grading_runner_tree_sha256": grading_tree_sha256(),
            "python": "3.13",
            "venv": str(VENV_PYTHON.parent.parent),
            "install_command": "uv sync --locked --python 3.13",
        },
        "fixture": json.loads(
            (FIXTURES / "fixture_manifest.json").read_text(encoding="utf-8")
        ),
        "evals_exercised": [
            {
                "eval_defn_id": "content_length_check",
                "eval_types": ["PROGRAMMATIC"],
                "llm_judge_used": False,
            }
        ],
        "scoring_method": "task_score_unweighted_and_universal_penalty",
        "helpers_exercised": ["snapshot_diff", "final_answer"],
        "observed": {
            "grading_run_status": grades.get("grading_run_status"),
            "verifier_scores": observed,
            "final_score": final_score,
        },
        "expected": {
            "grading_run_status": EXPECTED_STATUS,
            "verifier_scores": EXPECTED_VERIFIER_SCORES,
            "final_score": EXPECTED_FINAL_SCORE,
        },
        "problems": problems,
        "paid_api_calls": 0,
        "tinker_calls": 0,
        "evidence": {
            "grades_json": str(grades_path),
            "grading_run_log": str(log_path),
            "fixture_dir": str(FIXTURES),
        },
    }
    receipt_path = OUT / "harness_validation_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    if problems:
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        fail(f"harness validation did not match expectations; see {receipt_path}")

    print(f"PASS: harness_validation (is_model_score=false, suite score=null)")
    print(f"  verifier scores: {observed}")
    print(f"  final_score:     {final_score}")
    print(f"  receipt:         {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
