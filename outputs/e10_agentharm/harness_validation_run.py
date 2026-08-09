#!/usr/bin/env python3
"""E10 AgentHarm harness validation — PUBLIC split, zero paid model calls.

This is NOT the AgentHarm score and the public split is NOT the held-out split.
It proves the plumbing: dataset load -> tool injection -> agent scaffold ->
grading functions -> refusal judge -> metrics -> eval log.

Every model in the run is `mockllm/model`, a local no-cost provider bundled with
inspect_ai.  The process runs with HF_HUB_OFFLINE=1 so a cache miss fails loudly
instead of silently downloading.

Usage:
    outputs/_setup/venvs/e10/bin/python outputs/e10_agentharm/harness_validation_run.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Fail loudly rather than silently reaching the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
# mockllm needs no credentials; make sure no real provider key is picked up.
for _var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
    os.environ.pop(_var, None)

from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput, get_model  # noqa: E402
from inspect_ai.tool import ToolChoice, ToolInfo  # noqa: E402
from inspect_evals.agentharm import agentharm  # noqa: E402
from inspect_evals.agentharm.utils import AGENTHARM_DATASET_REVISION  # noqa: E402

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs" / "harness_validation"

REFUSAL_JUDGE_OUTPUT = (
    "The response declines the request outright and offers no assistance.\n"
    "Answer: Yes"
)
SEMANTIC_JUDGE_OUTPUT = "fail"


def _fixed_output(content: str) -> Any:
    def _generate(
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        return ModelOutput.from_content(model="mockllm", content=content)

    return _generate


def build_judges() -> tuple[Any, Any]:
    """Two independent deterministic mock judges (memoize=False keeps them distinct)."""
    refusal_judge = get_model(
        "mockllm/model",
        memoize=False,
        custom_outputs=_fixed_output(REFUSAL_JUDGE_OUTPUT),
    )
    semantic_judge = get_model(
        "mockllm/model",
        memoize=False,
        custom_outputs=_fixed_output(SEMANTIC_JUDGE_OUTPUT),
    )
    return refusal_judge, semantic_judge


def run_config(name: str, agent: str, limit: int | None) -> dict[str, Any]:
    refusal_judge, semantic_judge = build_judges()
    task = agentharm(
        agent=agent,
        split="test_public",
        refusal_judge=refusal_judge,
        semantic_judge=semantic_judge,
        force_redownload=False,
    )
    dataset_size = len(task.dataset)

    logs = inspect_eval(
        task,
        model="mockllm/model",
        limit=limit,
        log_dir=str(LOG_DIR / name),
        display="none",
        fail_on_error=False,
        max_connections=8,
    )
    log = logs[0]

    metrics: dict[str, Any] = {}
    if log.results is not None:
        for score in log.results.scores:
            for metric_name, metric in score.metrics.items():
                metrics[f"{score.name}/{metric_name}"] = metric.value

    return {
        "config": name,
        "agent": agent,
        "split": "test_public",
        "dataset": "harmful",
        "dataset_revision": AGENTHARM_DATASET_REVISION,
        "dataset_size": dataset_size,
        "samples_run": None if log.results is None else log.results.total_samples,
        "samples_completed": None
        if log.results is None
        else log.results.completed_samples,
        "eval_status": log.status,
        "error": None if log.error is None else str(log.error.message),
        "eval_model": "mockllm/model",
        "refusal_judge_model": "mockllm/model",
        "semantic_judge_model": "mockllm/model",
        "metrics": metrics,
        "log_location": log.location,
        "task_version": log.eval.task_version,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit samples (default: whole public split)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "evidence" / "harness_validation_result.json",
    )
    args = parser.parse_args()

    results = [
        run_config("refusal_agent", "refusal", args.limit),
        run_config("default_agent_mockllm", "default", args.limit),
    ]

    payload = {
        "label": "harness_validation",
        "is_model_score": False,
        "score": None,
        "note": (
            "Harness validation only. Every model is mockllm/model (no paid calls). "
            "Run on the PUBLIC split. This is NOT the AgentHarm score and the public "
            "split is NOT the held-out split."
        ),
        "limit": args.limit,
        "configs": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(r["eval_status"] == "success" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
