#!/usr/bin/env python3
"""Fetch canonical-experiment results into the HF Spaces dashboard (read-only).

Loads from the three stores the experiment writes to — HuggingFace Hub (adapter +
run_manifest.json), Weights & Biases (run histories), and the GCS receipt bucket.
Each source is optional: missing credentials / optional deps degrade gracefully,
so the Space always renders whatever it can reach. Consumed by app.py's
"Live Results" tab.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def fetch_hf(repo_id: str, filename: str = "run_manifest.json") -> dict:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:  # pragma: no cover - optional dep
        return {"source": "hf", "repo": repo_id, "error": f"huggingface_hub unavailable: {e}"}
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        data = json.loads(Path(path).read_text())
        return {"source": "hf", "repo": repo_id, "filename": filename, "data": data}
    except Exception as e:  # pragma: no cover - network/auth
        return {"source": "hf", "repo": repo_id, "error": str(e)}


def fetch_wandb(entity: str = "arvindcr4", project: str = "tinker-rl-lab", n: int = 5) -> dict:
    try:
        import wandb
    except Exception as e:  # pragma: no cover - optional dep
        return {"source": "wandb", "error": f"wandb unavailable: {e}"}
    try:
        api = wandb.Api()
        runs = list(api.runs(f"{entity}/{project}"))[:n]
        out = [
            {
                "id": r.id,
                "name": r.name,
                "state": r.state,
                "url": r.url,
                "history_tail": r.history(samples=5).to_dict("records"),
            }
            for r in runs
        ]
        return {"source": "wandb", "entity": entity, "project": project, "runs": out}
    except Exception as e:  # pragma: no cover - network/auth
        return {"source": "wandb", "error": str(e)}


def fetch_gcs(prefix: str = "gs://arvindcr-tinker-rl-preflight-358208640342/preflight/") -> dict:
    try:
        out = subprocess.run(
            ["gcloud", "storage", "ls", prefix], capture_output=True, text=True, timeout=30
        )
        if out.returncode != 0:
            return {"source": "gcs", "prefix": prefix, "error": out.stderr.strip() or "gcloud unavailable"}
        return {"source": "gcs", "prefix": prefix, "objects": out.stdout.splitlines()}
    except Exception as e:  # pragma: no cover - no gcloud
        return {"source": "gcs", "error": str(e)}


def fetch_all(framework: str = "trl", task: str = "gsm8k") -> dict:
    """Collect results for one (framework, task) cell from every reachable store."""
    return {
        "framework": framework,
        "task": task,
        "hf": fetch_hf(f"arvindcr4/{task}-grpo-results"),
        "wandb": fetch_wandb(),
        "gcs": fetch_gcs(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch canonical-experiment results for the Space.")
    ap.add_argument("--framework", default="trl")
    ap.add_argument("--task", default="gsm8k")
    args = ap.parse_args()
    print(json.dumps(fetch_all(args.framework, args.task), indent=2, default=str))


if __name__ == "__main__":
    main()
