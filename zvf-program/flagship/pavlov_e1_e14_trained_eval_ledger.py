"""Bind the completed Pavlov sampler to the live E1--E14 evidence ledger.

This command does not turn preflights or harness checks into model scores.  It
records exactly which suites executed the trained sampler and which stopped at
an upstream access/runtime gate, with one online W&B audit run for the ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITES = (
    ("E1", "swe_bench_pro", "outputs/e1_swe_bench_pro/lane_receipt_2026-08-09.json"),
    ("E2", "frontier_swe", "outputs/e2_frontier_swe/e2_frontier_swe_lane_receipt_2026-08-09.json"),
    ("E3", "sdab", "outputs/e3_sdab/receipt_2026-08-09.json"),
    ("E4", "banker_toolbench", "outputs/e4_banker_toolbench/e4_harbor_rerun_receipt.json"),
    ("E5", "apex_agents", "outputs/e5_apex_agents/harness_validation/out/harness_validation_receipt.json"),
    ("E6", "webbench", "outputs/e6_webbench/e6_lane_receipt_2026-08-09.json"),
    ("E7", "binaryaudit", "outputs/e7_binaryaudit/e7_binaryaudit_receipt_2026-08-09.json"),
    ("E8", "lifescibench", "outputs/e8_lifescibench/lane_receipt_2026-08-09.json"),
    ("E9", "mle_bench", "outputs/e9_mle_bench/e9_mle_bench_receipt_2026-08-09.json"),
    ("E10", "agentharm", "outputs/e10_agentharm/receipt_2026-08-09.json"),
    ("E11", "verilog_eval", "outputs/e11_verilog_eval/e11_trained_step40_receipt.json"),
    ("E12", "appbench", "outputs/e12_appbench/receipt_2026-08-09.json"),
    ("E13", "openreward_games", "outputs/e13_openreward_games/receipt_2026-08-09.json"),
    ("E14", "frontiermath", "outputs/e14_frontiermath/receipt_2026-08-09.json"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _list_strings(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def _blockers(receipt: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("blockers", "outstanding_blockers", "missing", "errors"):
        values.extend(_list_strings(receipt.get(key)))
    if not values and receipt.get("status") not in {"COMPLETE", "SCORED"}:
        values.append("source receipt contains no completed trained-model score")
    return values


def build_ledger(training_receipt_path: Path) -> dict[str, Any]:
    training = json.loads(training_receipt_path.read_text(encoding="utf-8"))
    if training.get("status") != "completed" or training.get("final_status") != "success":
        raise ValueError("training receipt is not completed successfully")
    result = training["result"]
    final_checkpoint = result["checkpoint_receipts"][-1]
    if final_checkpoint.get("step") != "final":
        raise ValueError("training receipt has no final checkpoint")

    rows: list[dict[str, Any]] = []
    for lane, suite_id, relative in SUITES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing {lane} receipt: {path}")
        source = json.loads(path.read_text(encoding="utf-8"))
        model_executed = lane == "E11" and source.get("is_model_score") is True
        score = source.get("score") if model_executed else None
        if score is None and model_executed:
            score = (source.get("pass_at_1") or {}).get("corrected", {}).get("pass_at_1")
        rows.append(
            {
                "lane": lane,
                "suite_id": suite_id,
                "status": "SCORED" if model_executed and score is not None else "BLOCKED_BEFORE_MODEL_SCORE",
                "model_executed": model_executed,
                "is_model_score": model_executed and score is not None,
                "score": score,
                "source_status": source.get("status") or source.get("final_status"),
                "source_evidence_class": source.get("evidence_class"),
                "blockers": [] if model_executed and score is not None else _blockers(source),
                "receipt_path": relative,
                "receipt_sha256": _sha256(path),
                "evaluated_sampler": (
                    source.get("model", {}).get("sampler_path") if model_executed else None
                ),
            }
        )

    return {
        "schema_version": "pavlov-e1-e14-trained-ledger-v1",
        "status": "PARTIAL",
        "claim_boundary": (
            "Only rows with is_model_score=true are benchmark results; blocked and harness-only "
            "rows are not model evidence."
        ),
        "training": {
            "tinker_run_id": result["run_id"],
            "sampler_path": result["sampler_path"],
            "wandb_url": "https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-pavlov/runs/bsv8vx04",
            "final_hf_repo": final_checkpoint["repo_url"],
            "final_hf_revision": final_checkpoint["revision"],
            "final_hf_commit": final_checkpoint["commit_sha"],
        },
        "summary": {
            "suite_count": len(rows),
            "model_executed": sum(bool(row["model_executed"]) for row in rows),
            "model_scored": sum(bool(row["is_model_score"]) for row in rows),
            "blocked_before_model_score": sum(not bool(row["is_model_score"]) for row in rows),
        },
        "suites": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ledger = build_ledger(args.training_receipt)

    os.environ["WANDB_MODE"] = "online"
    import wandb

    run = wandb.init(
        entity="arvindcr4-pes-university",
        project="tinker-rl-lab-pavlov",
        group="pavlov-e1-e14-eval-20260809",
        job_type="evaluation-ledger",
        name="pavlov_e1_e14_trained_checkpoint_ledger",
        tags=["pavlov", "e1-e14", "evaluation-ledger", "claim-boundary"],
        config={"training": ledger["training"], "claim_boundary": ledger["claim_boundary"]},
    )
    for index, row in enumerate(ledger["suites"]):
        run.log(
            {
                "suite/index": index,
                "suite/model_executed": int(row["model_executed"]),
                "suite/is_model_score": int(row["is_model_score"]),
                "suite/score": row["score"] if row["score"] is not None else 0.0,
                f"suite/{row['lane']}/model_executed": int(row["model_executed"]),
                f"suite/{row['lane']}/is_model_score": int(row["is_model_score"]),
            },
            step=index,
        )
    ledger["wandb"] = {"run_id": run.id, "url": run.url, "project": run.project}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run.summary.update(ledger["summary"])
    run.finish(exit_code=0)
    print(json.dumps({"status": ledger["status"], **ledger["summary"], "wandb": ledger["wandb"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
