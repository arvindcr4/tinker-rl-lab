#!/usr/bin/env python3
"""Read-only, credential-redacting W&B detail inspector for selected runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import tempfile

import wandb


SAFE_CONFIG_KEYS = {
    "batch",
    "batch_size",
    "estimator",
    "eval_n",
    "group",
    "group_size",
    "lr",
    "method",
    "model",
    "model_short",
    "num_steps",
    "platform",
    "seed",
    "steps",
    "task",
}
SAFE_METADATA_KEYS = {
    "codePath",
    "codePathLocal",
    "git",
    "host",
    "program",
    "python",
    "root",
    "startedAt",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project")
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--entity", default="arvindcr4-pes-university")
    parser.add_argument("--log-excerpts", action="store_true")
    args = parser.parse_args()

    api = wandb.Api(timeout=90)
    for run_id in args.run_ids:
        run = api.run(f"{args.entity}/{args.project}/{run_id}")
        metadata = run.metadata or {}
        history = run.history(pandas=False)
        record = {
            "id": run.id,
            "name": run.name,
            "created_at": run.created_at,
            "state": run.state,
            "group": run.group,
            "tags": run.tags,
            "config": {
                key: value
                for key, value in sorted(dict(run.config or {}).items())
                if key in SAFE_CONFIG_KEYS
            },
            "summary": {
                key: value
                for key, value in sorted(dict(run.summary).items())
                if key.startswith(("eval/", "final/", "train/"))
                or key in {"runtime", "_runtime"}
            },
            "metadata": {
                key: metadata[key]
                for key in sorted(metadata)
                if key in SAFE_METADATA_KEYS
            },
            "files": sorted(file.name for file in run.files()),
            "history_rows": len(history),
            "history_keys": sorted({key for row in history for key in row}),
        }
        if args.log_excerpts and "output.log" in record["files"]:
            with tempfile.TemporaryDirectory(prefix="wandb-audit-") as tmpdir:
                downloaded = run.file("output.log").download(root=tmpdir, replace=True)
                log_text = Path(downloaded.name).read_text(errors="replace")
            matching = [
                line
                for line in log_text.splitlines()
                if re.search(
                    r"tinker|run.?id|checkpoint|held.?out|eval|save|warning|error",
                    line,
                    flags=re.IGNORECASE,
                )
            ]
            redacted = [
                re.sub(r"wandb_v1_[A-Za-z0-9_-]+", "<REDACTED>", line)
                for line in matching
            ]
            record["log_excerpts"] = redacted[:80]
        print(json.dumps(record, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
