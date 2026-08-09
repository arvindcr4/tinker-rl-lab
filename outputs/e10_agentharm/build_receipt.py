#!/usr/bin/env python3
"""Build the E10 AgentHarm receipt from real local state.

Binds the immutable dataset revision, per-task ID hashes over the public split,
the split manifest, and the verifier identity, then runs the fail-closed emitter.
The emitter is what decides whether a score may exist; this script never writes a
score itself.

Usage:
    outputs/_setup/venvs/e10/bin/python outputs/e10_agentharm/build_receipt.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "zvf-program"))

from flagship import pavlov_agentharm_frontiermath_adapter as adapter  # noqa: E402

LANE_DIR = REPO / "outputs" / "e10_agentharm"
EVIDENCE = LANE_DIR / "evidence"
DATASET_ROOT = LANE_DIR / "hf_dataset"
VENV = REPO / "outputs" / "_setup" / "venvs" / "e10"
PACKAGE_ROOT = VENV / "lib" / "python3.12" / "site-packages" / "inspect_evals" / "agentharm"
INSPECT_CACHE = (
    Path.home() / "Library" / "Caches" / "inspect_evals" / "agentharm_dataset" / "AgentHarm"
)


def _thin(manifest: dict) -> dict:
    """Manifest without the full task-hash list (kept in its own artifact)."""
    thin = {k: v for k, v in manifest.items() if k != "task_id_hashes"}
    thin["task_id_hashes_artifact"] = "evidence/public_split_task_id_hashes.json"
    thin["files"] = {
        name: {k: v for k, v in meta.items() if k != "behavior_ids"}
        for name, meta in manifest["files"].items()
    }
    return thin


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)

    public = adapter.build_agentharm_split_manifest(DATASET_ROOT, "test_public")
    validation = adapter.build_agentharm_split_manifest(DATASET_ROOT, "val")
    private = adapter.build_agentharm_split_manifest(DATASET_ROOT, "test_private")
    verifier = adapter.agentharm_verifier_identity(PACKAGE_ROOT)
    heldout = adapter.check_heldout_split_available(DATASET_ROOT)
    heldout_cache = adapter.check_heldout_split_available(INSPECT_CACHE)

    (EVIDENCE / "public_split_task_id_hashes.json").write_text(
        json.dumps(
            {
                "dataset_id": public["dataset_id"],
                "dataset_revision": public["dataset_revision"],
                "split": public["split"],
                "split_task_id_hash": public["split_task_id_hash"],
                "task_count": public["task_count"],
                "behavior_ids_by_file": {
                    name: meta["behavior_ids"] for name, meta in public["files"].items()
                },
                "task_id_hashes": public["task_id_hashes"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    harness = json.loads((EVIDENCE / "harness_validation_result.json").read_text())
    log_probe = json.loads((EVIDENCE / "harness_validation_log_probe.json").read_text())
    categories = sorted(
        {
            key.split("/")[1].rsplit("_avg_", 1)[0]
            for key in harness["configs"][0]["metrics"]
            if "_avg_scores" in key
        }
    )

    # The score gate. Deliberately fed the real (blocked) local state.
    emitted = adapter.emit_agentharm_score(
        split_manifest=private,
        verifier_identity=verifier,
        heldout_availability=heldout,
        run={
            "label": "agentharm_heldout",
            "is_model_score": False,
            "raw_score": None,
            "model_id": adapter.MODEL_ID,
            "model_revision": adapter.MODEL_REVISION,
            "policy_grader_artifact": None,
        },
    )
    # Same gate applied to the harness-validation run, to show it is also refused.
    emitted_from_harness = adapter.emit_agentharm_score(
        split_manifest=public,
        verifier_identity=verifier,
        heldout_availability=heldout,
        run={
            "label": "harness_validation",
            "is_model_score": False,
            "raw_score": harness["configs"][0]["metrics"]["combined_scorer/avg_score"],
            "model_id": "mockllm/model",
            "model_revision": None,
            "policy_grader_artifact": None,
        },
    )

    receipt = {
        "suite_id": "agentharm_eval",
        "lane": "E10",
        "date": "2026-08-09",
        "status": "PARTIAL",
        "score": None,
        "is_model_score": False,
        "score_gate": emitted,
        "score_gate_applied_to_harness_run": emitted_from_harness,
        "why_no_score": (
            "The AgentHarm score is defined on the held-out (test_private) split. "
            "All three held-out files are absent from the public dataset revision and "
            "are gated by the UK AI Safety Institute. No paid model artifact exists "
            "either. The emitter refuses to produce a score; see score_gate.blockers."
        ),
        "harness_validation": {
            "label": "harness_validation",
            "is_model_score": False,
            "score": None,
            "note": harness["note"],
            "split": "test_public",
            "dataset": "harmful",
            "samples": 176,
            "configs": [
                {
                    "config": c["config"],
                    "agent": c["agent"],
                    "eval_status": c["eval_status"],
                    "samples_completed": c["samples_completed"],
                    "eval_model": c["eval_model"],
                    "refusal_judge_model": c["refusal_judge_model"],
                    "semantic_judge_model": c["semantic_judge_model"],
                    "metrics": c["metrics"],
                    "log": c["log_location"],
                }
                for c in harness["configs"]
            ],
            "plumbing_proven": {
                "dataset_loaded": True,
                "samples_completed_per_config": 176,
                "distinct_target_tools_instantiated": log_probe["refusal_agent"][
                    "distinct_target_tools"
                ],
                "target_tool_slots": log_probe["refusal_agent"]["total_target_tool_slots"],
                "samples_with_grading_breakdown": log_probe["refusal_agent"][
                    "samples_with_grading_breakdown"
                ],
                "refusal_judge_exercised": True,
                "categories_scored": categories,
            },
            "paid_model_calls": 0,
            "estimated_cost_usd": 0.0,
        },
        "binding": {
            "dataset_id": adapter.AGENTHARM_DATASET_ID,
            "dataset_revision": adapter.AGENTHARM_DATASET_REVISION,
            "dataset_revision_immutable": True,
            "dataset_license": "hf_dataset/LICENSE (AISI AgentHarm license, present)",
            "harness_repo": adapter.AGENTHARM_HARNESS_REPO,
            "harness_revision": adapter.AGENTHARM_HARNESS_REVISION,
            "inspect_ai_version": "0.3.254",
            "task_registry_name": "inspect_evals/agentharm",
            "verifier_identity": {
                k: v for k, v in verifier.items() if k != "package_root"
            },
            "split_manifests": {
                "test_public": _thin(public),
                "val": _thin(validation),
                "test_private": _thin(private),
            },
            "heldout_availability": {
                "lane_dataset_copy": heldout,
                "inspect_evals_cache": heldout_cache,
            },
        },
        "blockers": [
            {
                "id": "E10-HELDOUT-SPLIT",
                "summary": (
                    "AgentHarm held-out split files are absent from the public dataset "
                    "revision and gated by the UK AI Safety Institute."
                ),
                "required_files": list(adapter.AGENTHARM_HELDOUT_FILES),
                "loader_flag": 'split="test_private"',
                "external_receipt_needed": (
                    "Written AISI grant of held-out AgentHarm access, plus the file "
                    "revision/checksums AISI publishes with the grant."
                ),
                "request_artifact": "outputs/e10_agentharm/AISI_ACCESS_REQUEST.md",
            },
            {
                "id": "E10-POLICY-GRADER-ARTIFACT",
                "summary": (
                    "The contract marks agentharm_eval artifact_required; the approved "
                    "policy-grader artifact is not present locally."
                ),
                "external_receipt_needed": (
                    "AISI-approved policy grader / refusal-and-semantic judge "
                    "specification with a pinned identity hash."
                ),
                "request_artifact": "outputs/e10_agentharm/AISI_ACCESS_REQUEST.md",
            },
            {
                "id": "E10-JUDGE-MODEL-KEY",
                "summary": (
                    "The official grader defaults to openai/gpt-4o-2024-08-06 for both the "
                    "refusal judge and the semantic judge. No paid model key is authorized "
                    "for benchmark use in this lane."
                ),
                "external_receipt_needed": (
                    "Budget authorization plus an OpenAI key (or the AISI-approved judge "
                    "substitute) before any scored run."
                ),
            },
            {
                "id": "E10-MODEL-ARTIFACT",
                "summary": (
                    f"No agent artifact exists for {adapter.MODEL_ID}@"
                    f"{adapter.MODEL_REVISION[:12]}; the scored run needs an agent model, "
                    "which is a paid call this lane may not make."
                ),
                "external_receipt_needed": "Tinker/W&B run identity and budget receipt.",
            },
        ],
        "evidence": {
            "inspect_eval_help": "outputs/e10_agentharm/evidence/inspect_eval_help.txt",
            "task_registry_entry": "outputs/e10_agentharm/evidence/agentharm_registry_entry.json",
            "harness_validation_result": "outputs/e10_agentharm/evidence/harness_validation_result.json",
            "harness_validation_log_probe": "outputs/e10_agentharm/evidence/harness_validation_log_probe.json",
            "eval_logs": "outputs/e10_agentharm/logs/harness_validation/",
            "public_split_task_id_hashes": "outputs/e10_agentharm/evidence/public_split_task_id_hashes.json",
            "adapter_unit_tests": "outputs/e10_agentharm/evidence/adapter_unit_tests.log",
            "runner": "outputs/e10_agentharm/harness_validation_run.py",
            "receipt_builder": "outputs/e10_agentharm/build_receipt.py",
            "adapter": "zvf-program/flagship/pavlov_agentharm_frontiermath_adapter.py",
            "adapter_tests": "zvf-program/flagship/test_pavlov_agentharm_frontiermath_adapter.py",
            "access_request": "outputs/e10_agentharm/AISI_ACCESS_REQUEST.md",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "venv": str(VENV),
            "network_used_during_validation": False,
            "hf_hub_offline": True,
        },
    }

    out = LANE_DIR / "receipt_2026-08-09.json"
    body = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    out.write_text(body)
    print(f"wrote {out} ({len(body)} bytes)")
    print("receipt_sha256=" + hashlib.sha256(body.encode()).hexdigest())
    print("score=" + json.dumps(receipt["score"]))
    print("score_gate.status=" + receipt["score_gate"]["status"])
    for blocker in receipt["score_gate"]["blockers"]:
        print("  blocker: " + blocker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
