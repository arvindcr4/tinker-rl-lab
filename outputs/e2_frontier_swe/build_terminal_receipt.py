#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


LANE = Path(__file__).resolve().parent
REPO = LANE / "frontier-swe"
SOURCE_TRIAL = (
    REPO
    / "tasks/revideo-perf-opt/jobs/revideo-perf-opt-pavlov-tinker-pass1-v10"
    / "revideo-perf-opt__LPQjQkC"
)
REPLAY_JOB = (
    REPO
    / "tasks/revideo-perf-opt/jobs/revideo-perf-opt-frozen-artifact-replay-20260822-v3"
)
REPLAY_TRIAL = REPLAY_JOB / "revideo-perf-opt__7rYxH3L"
OUTPUT = LANE / "e2_terminal_attempt_receipt_2026-08-22.json"
DELTA_REL = "node_modules/@types/dom-webcodecs/webcodecs.generated.d.ts"
IMAGE_DIGEST = "sha256:675d298493278f891a50e41ed31ffdb71590d4583dfc1987385a48d872f25103"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(LANE.parents[1].resolve()).as_posix()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args], text=True
    ).strip()


def main() -> int:
    source_result = json.loads((SOURCE_TRIAL / "result.json").read_text())
    trajectory = json.loads((SOURCE_TRIAL / "agent/trajectory.json").read_text())
    replay_result = json.loads((REPLAY_TRIAL / "result.json").read_text())
    replay_log = json.loads((REPLAY_TRIAL / "agent/artifact-replay.json").read_text())
    reward = json.loads((REPLAY_TRIAL / "verifier/reward.json").read_text())
    info = json.loads((REPLAY_TRIAL / "verifier/info.json").read_text())

    mutations = []
    for step in trajectory.get("steps", []):
        for call in step.get("tool_calls") or []:
            if call.get("function_name") in {"edit", "write", "apply_patch"}:
                mutations.append(
                    {
                        "step_id": step.get("step_id"),
                        "function_name": call.get("function_name"),
                        "file_path": (call.get("arguments") or {}).get("filePath"),
                    }
                )

    expected_mutations = [
        {"step_id": 19, "function_name": "edit", "file_path": f"/app/revideo/{DELTA_REL}"},
        {"step_id": 26, "function_name": "edit", "file_path": f"/app/revideo/{DELTA_REL}"},
    ]
    assert mutations == expected_mutations
    assert replay_result["config"]["agent"]["model_name"] is None
    assert replay_result["agent_info"]["model_info"] is None
    assert replay_result["exception_info"] is None
    assert replay_log["model_calls"] == 0
    assert reward == {"reward": 0.725611}
    assert info["correctness_ok"] is True
    assert info["num_hidden_scenes"] == info["num_speedups_computed"] == 8
    assert info["hard_fail_reasons"] == []
    assert all(item["correct"] for item in info["correctness_details"])

    delta_path = SOURCE_TRIAL / "artifacts/app/revideo" / DELTA_REL
    delta_hash = sha256(delta_path)
    assert delta_hash == replay_log["restored"][0]["sha256"]

    normalized_score = round(0.5 + 0.5 * info["geometric_mean_speedup"], 4)
    assert normalized_score == 0.8628

    receipt = {
        "schema_version": "e2-frozen-artifact-verifier-replay-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE",
        "lane": "E2",
        "scope": {
            "level": "attempt",
            "suite": "FrontierSWE",
            "task_id": "revideo-perf-opt",
            "official_suite_task_count": 17,
            "tasks_executed": 1,
            "suite_score": None,
            "suite_score_note": (
                "null: this receipt closes one frozen task attempt only; it is not a "
                "17-task FrontierSWE suite run"
            ),
        },
        "result": {
            "harbor_reward": reward["reward"],
            "native_task_reward": info["reward"],
            "geometric_mean_speedup": info["geometric_mean_speedup"],
            "correctness_ok": info["correctness_ok"],
            "hidden_scenes": info["num_hidden_scenes"],
            "speedups_computed": info["num_speedups_computed"],
            "hard_fail_reasons": info["hard_fail_reasons"],
            "leaderboard_normalized_task_score": normalized_score,
            "leaderboard_command": (
                "python3 scripts/score_from_reward.py --task revideo-perf-opt "
                "tasks/revideo-perf-opt/jobs/"
                "revideo-perf-opt-frozen-artifact-replay-20260822-v3/"
                "revideo-perf-opt__7rYxH3L/verifier/info.json"
            ),
            "leaderboard_command_output": {
                "category": "performance",
                "correctness": 1.0,
                "speedup": 0.7256,
                "score": 0.8628,
            },
            "timing_variance_note": (
                "The original verifier artifacts measured reward 0.753697; the fresh "
                "ABBA replay measured 0.725611. Both independently rendered and passed "
                "8/8 hidden scenes. Performance timing is remeasured, not copied."
            ),
        },
        "frozen_source_attempt": {
            "trial_name": source_result["trial_name"],
            "trial_id": source_result["id"],
            "task_checksum": source_result["task_checksum"],
            "started_at": source_result["started_at"],
            "finished_at": source_result["finished_at"],
            "result_path": rel(SOURCE_TRIAL / "result.json"),
            "result_sha256": sha256(SOURCE_TRIAL / "result.json"),
            "trajectory_path": rel(SOURCE_TRIAL / "agent/trajectory.json"),
            "trajectory_sha256": sha256(SOURCE_TRIAL / "agent/trajectory.json"),
            "original_failure_boundary": (
                "native verifier completed, but Harbor rejected rich non-numeric "
                "diagnostic values in reward.json"
            ),
            "mutating_tool_calls": mutations,
        },
        "candidate_reconstruction": {
            "mode": "fresh official image plus frozen trajectory file delta",
            "byte_identical_full_snapshot": False,
            "why_delta_is_sufficient": (
                "The audited trajectory contains exactly two mutating agent tool calls, "
                "both edits of the same file. The native verifier rebuilds the candidate "
                "packages before measurement; post-verifier hidden-scene files in the "
                "downloaded full snapshot were deliberately not replayed."
            ),
            "official_image": "ghcr.io/proximal-labs/frontier-swe/revideo-perf-opt:v4",
            "official_image_digest": IMAGE_DIGEST,
            "restored_files": replay_log["restored"],
            "delta_file_sha256": delta_hash,
        },
        "verifier_replay": {
            "job_name": "revideo-perf-opt-frozen-artifact-replay-20260822-v3",
            "trial_name": replay_result["trial_name"],
            "trial_id": replay_result["id"],
            "task_checksum": replay_result["task_checksum"],
            "started_at": replay_result["started_at"],
            "finished_at": replay_result["finished_at"],
            "result_path": rel(REPLAY_TRIAL / "result.json"),
            "exception": replay_result["exception_info"],
            "model_name": replay_result["config"]["agent"]["model_name"],
            "model_info": replay_result["agent_info"]["model_info"],
            "model_calls": replay_log["model_calls"],
            "tinker_calls": 0,
            "tinker_cost_usd": 0.0,
            "agent_execution_seconds": 1.803501,
            "verifier_execution_seconds": 272.115049,
            "schema_repair": {
                "behavior": (
                    "reward.json contains only Harbor's numeric reward map; info.json "
                    "preserves the unchanged rich native diagnostics"
                ),
                "compute_reward_diff": "9 insertions, 0 deletions, serialization only",
                "reward_json": reward,
            },
        },
        "provenance": {
            "frontier_swe_checkout_commit": git("rev-parse", "HEAD"),
            "frontier_swe_checkout_tree": git("rev-parse", "HEAD^{tree}"),
            "files": {
                rel(REPLAY_TRIAL / "result.json"): sha256(REPLAY_TRIAL / "result.json"),
                rel(REPLAY_TRIAL / "agent/artifact-replay.json"): sha256(
                    REPLAY_TRIAL / "agent/artifact-replay.json"
                ),
                rel(REPLAY_TRIAL / "verifier/reward.json"): sha256(
                    REPLAY_TRIAL / "verifier/reward.json"
                ),
                rel(REPLAY_TRIAL / "verifier/info.json"): sha256(
                    REPLAY_TRIAL / "verifier/info.json"
                ),
                rel(REPLAY_TRIAL / "verifier/correctness_results.json"): sha256(
                    REPLAY_TRIAL / "verifier/correctness_results.json"
                ),
                rel(REPO / "tasks/revideo-perf-opt/tests/test.sh"): sha256(
                    REPO / "tasks/revideo-perf-opt/tests/test.sh"
                ),
                rel(REPO / "tasks/revideo-perf-opt/tests/compute_reward.py"): sha256(
                    REPO / "tasks/revideo-perf-opt/tests/compute_reward.py"
                ),
                rel(REPO / "tasks/revideo-perf-opt/tests/prep_build.py"): sha256(
                    REPO / "tasks/revideo-perf-opt/tests/prep_build.py"
                ),
                rel(REPO / "tasks/revideo-perf-opt/tests/hidden-scenes.tar.gz"): sha256(
                    REPO / "tasks/revideo-perf-opt/tests/hidden-scenes.tar.gz"
                ),
                rel(REPO / "scripts/score_from_reward.py"): sha256(
                    REPO / "scripts/score_from_reward.py"
                ),
                rel(REPO / "harbor_ext/artifact_replay.py"): sha256(
                    REPO / "harbor_ext/artifact_replay.py"
                ),
                rel(REPO / "tasks/revideo-perf-opt/job-artifact-replay.yaml"): sha256(
                    REPO / "tasks/revideo-perf-opt/job-artifact-replay.yaml"
                ),
            },
        },
        "failed_pre_execution_launches": [
            {
                "job_name": "revideo-perf-opt-frozen-artifact-replay-20260822",
                "reason": "custom agent import missing from PYTHONPATH",
                "environment_started": False,
                "agent_started": False,
                "verifier_started": False,
                "model_calls": 0,
            },
            {
                "job_name": "revideo-perf-opt-frozen-artifact-replay-20260822-v2",
                "reason": "Harbor output-directory mkdir/chmod ordering bug",
                "environment_started": False,
                "agent_started": False,
                "verifier_started": False,
                "model_calls": 0,
            },
        ],
        "validation": {
            "tests_command": (
                "/Users/arvind/.local/share/uv/tools/harbor/bin/python -m unittest -v "
                "harbor_ext.test_artifact_replay "
                "tasks.revideo-perf-opt.tests.test_compute_reward_schema"
            ),
            "tests_passed": 3,
            "tests_failed": 0,
            "harbor_command": (
                "PYTHONPATH=. harbor run --config "
                "tasks/revideo-perf-opt/job-artifact-replay.yaml --job-name "
                "revideo-perf-opt-frozen-artifact-replay-20260822-v3 --yes"
            ),
            "harbor_exception_count": 0,
        },
        "evidence_boundary": (
            "Complete, verified E2 attempt for revideo-perf-opt. Not a FrontierSWE "
            "suite result and not evidence for the other 16 official tasks."
        ),
    }

    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUTPUT}")
    print(f"receipt_sha256={sha256(OUTPUT)}")
    print(f"harbor_reward={receipt['result']['harbor_reward']}")
    print(
        "leaderboard_normalized_task_score="
        f"{receipt['result']['leaderboard_normalized_task_score']}"
    )
    print(f"suite_score={receipt['scope']['suite_score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
