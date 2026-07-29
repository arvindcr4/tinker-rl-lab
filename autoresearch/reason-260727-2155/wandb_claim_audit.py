#!/usr/bin/env python3
"""Read-only W&B run inspector for the NeurIPS 36320 evidence audit.

Authentication is read only from WANDB_API_KEY. The script never prints the
credential and never writes W&B state or repository files.
"""

from __future__ import annotations

import argparse
import json
import re

import wandb


SUMMARY_TERMS = ("reward", "held", "acc", "zvf", "last", "eval", "delta")
CONFIG_TERMS = (
    "model",
    "method",
    "estimator",
    "platform",
    "task",
    "seed",
    "group",
    "step",
    "source",
    "run_id",
)


def compact_run(run: object) -> dict[str, object]:
    summary = dict(run.summary)
    config = dict(run.config or {})
    selected_summary = {
        key: summary[key]
        for key in sorted(summary)
        if any(term in key.lower() for term in SUMMARY_TERMS)
    }
    return {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": run.created_at,
        "group": run.group,
        "job_type": run.job_type,
        "tags": run.tags,
        "config": {
            key: config[key]
            for key in sorted(config)
            if any(term in key.lower() for term in CONFIG_TERMS)
        },
        "summary": selected_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project")
    parser.add_argument("--entity", default="arvindcr4-pes-university")
    parser.add_argument("--pattern", default=".")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    pattern = re.compile(args.pattern, re.IGNORECASE)
    api = wandb.Api(timeout=90)
    runs = list(api.runs(f"{args.entity}/{args.project}", per_page=args.limit))
    print(json.dumps({"project": args.project, "run_count": len(runs)}))
    for run in runs:
        record = compact_run(run)
        haystack = json.dumps(record, sort_keys=True, default=str)
        if pattern.search(haystack):
            print(json.dumps(record, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
