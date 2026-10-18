#!/usr/bin/env python3
"""Fetch only manifest-indexed experiment results; never relabel broad run lists."""

from __future__ import annotations
import argparse
import itertools
import json
import subprocess
from pathlib import Path

INDEX = Path(__file__).with_name("results_index.json")


def fetch_hf(repo_id, filename="run_manifest.json", revision=None):
    if not revision:
        return {"source": "hf", "error": "pinned_revision_required"}
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        return {"source": "hf", "data": json.loads(Path(path).read_text())}
    except Exception:
        return {"source": "hf", "error": "source_unavailable"}


def fetch_wandb(entity="arvindcr4", project="tinker-rl-lab", n=5, *, framework, task, run_ids):
    if not run_ids or not 0 < n <= 100:
        return {"source": "wandb", "error": "bounded_run_ids_required"}
    try:
        import wandb

        runs = itertools.islice(
            wandb.Api().runs(
                f"{entity}/{project}",
                filters={
                    "config.framework": framework,
                    "config.task": task,
                    "id": {"$in": list(run_ids)},
                },
            ),
            n,
        )
        rows = []
        for run in runs:
            if (
                run.id not in run_ids
                or run.config.get("framework") != framework
                or run.config.get("task") != task
            ):
                return {"source": "wandb", "error": "identity_mismatch"}
            rows.append(
                {
                    "id": run.id,
                    "name": run.name,
                    "state": run.state,
                    "url": run.url,
                    "sampled_history": run.history(samples=5).to_dict("records"),
                }
            )
        return {"source": "wandb", "runs": rows}
    except Exception:
        return {"source": "wandb", "error": "source_unavailable"}


def fetch_gcs(uri):
    if not uri.startswith("gs://") or any(c in uri for c in "*?["):
        return {"source": "gcs", "error": "exact_receipt_uri_required"}
    try:
        proc = subprocess.run(
            ["gcloud", "storage", "cat", uri], capture_output=True, text=True, timeout=30
        )
        if proc.returncode:
            raise ValueError("unavailable")
        return {"source": "gcs", "data": json.loads(proc.stdout)}
    except Exception:
        return {"source": "gcs", "error": "source_unavailable"}


def fetch_all(framework="trl", task="gsm8k", *, index=None):
    result = {"framework": framework, "task": task, "status": "unavailable", "sources": []}
    try:
        entries = index if index is not None else json.loads(INDEX.read_text())["entries"]
        matches = [
            entry for entry in entries if entry["framework"] == framework and entry["task"] == task
        ]
    except (OSError, ValueError, KeyError, TypeError):
        result["error"] = "reviewed_results_index_required"
        return result
    for entry in matches:
        identity = {
            k: entry[k] for k in ("framework", "task", "model_revision", "seed", "protocol_hash")
        }
        if "hf" in entry:
            source = fetch_hf(**entry["hf"])
            if "data" in source and any(source["data"].get(k) != v for k, v in identity.items()):
                source = {"source": "hf", "error": "identity_mismatch"}
            result["sources"].append(source)
        if "wandb" in entry:
            result["sources"].append(fetch_wandb(framework=framework, task=task, **entry["wandb"]))
        if "gcs" in entry:
            source = fetch_gcs(entry["gcs"])
            if "data" in source and any(source["data"].get(k) != v for k, v in identity.items()):
                source = {"source": "gcs", "error": "identity_mismatch"}
            result["sources"].append(source)
    sources = result["sources"]
    good = sum("error" not in s and bool(s.get("data") or s.get("runs")) for s in sources)
    result["status"] = (
        "complete"
        if good and good == len(sources)
        else "partial"
        if good
        else "unavailable"
        if any("error" in s for s in sources)
        else "empty"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", default="trl")
    parser.add_argument("--task", default="gsm8k")
    args = parser.parse_args()
    result = fetch_all(args.framework, args.task)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
