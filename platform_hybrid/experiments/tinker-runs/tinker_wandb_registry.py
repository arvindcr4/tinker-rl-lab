#!/usr/bin/env python3
"""Build and optionally publish a complete Tinker/W&B experiment registry.

The Tinker API is authoritative for training-run and checkpoint identity. W&B
is authoritative for experiment configuration and reported metrics. Historical
Tinker runs generally have no user metadata, so this script keeps exact ID
matches separate from time/model candidates instead of presenting guesses as
confirmed joins.

Examples:
    .venv/bin/python platform_hybrid/experiments/tinker-runs/tinker_wandb_registry.py
    .venv/bin/python platform_hybrid/experiments/tinker-runs/tinker_wandb_registry.py --publish
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import netrc
import os
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ENTITY = "arvindcr4-pes-university"
REGISTRY_PROJECT = "tinker-rl-lab-world-class"
GRAPHQL_URL = "https://api.wandb.ai/graphql"
TINKER_ID_RE = re.compile(
    r"(?P<id>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}:train:\d+)",
    re.IGNORECASE,
)
UUID_RE = re.compile(
    r"(?P<id>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)
SENSITIVE_KEY_RE = re.compile(
    r"(^|_)(api_?key|access_?token|token|auth|credential|password|secret)($|_)",
    re.IGNORECASE,
)
SENSITIVE_VALUE_RES = (
    re.compile(r"wandb_v1_[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)(?:tml|tinker)[_-]?(?:key|token)[=:][^\s,;]+"),
)
WANDB_RUNS_QUERY = """
query($e:String!,$p:String!,$c:String) {
  project(name:$p,entityName:$e) {
    runs(first:250,after:$c) {
      edges { node {
        name displayName state createdAt updatedAt config summaryMetrics
        group jobType tags
      } }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    default_output = (
        root
        / "platform_hybrid"
        / "experiments"
        / "results"
        / "tinker_wandb_registry"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--registry-project", default=REGISTRY_PROJECT)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--candidate-window-minutes",
        type=float,
        default=30.0,
        help="Maximum end/start delta for non-exact model/time candidates.",
    )
    return parser.parse_args()


def load_tinker_key(repo_root: Path) -> None:
    if os.environ.get("TINKER_API_KEY"):
        return
    env_path = repo_root / ".env"
    if not env_path.exists():
        raise RuntimeError("TINKER_API_KEY is unset and repository .env is missing")
    for line in env_path.read_text().splitlines():
        if line.startswith("TINKER_API_KEY="):
            os.environ["TINKER_API_KEY"] = (
                line.split("=", 1)[1].strip().strip('"').strip("'")
            )
            return
    raise RuntimeError("TINKER_API_KEY is unset and not present in repository .env")


def wandb_key() -> str:
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]
    auth = netrc.netrc().authenticators("api.wandb.ai")
    if not auth or not auth[2]:
        raise RuntimeError("W&B authentication not found in environment or ~/.netrc")
    return auth[2]


def sanitize(value: Any, key: str = "") -> Any:
    """Remove credentials while retaining useful run config and summaries."""
    if SENSITIVE_KEY_RE.search(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            str(k): sanitize(v, str(k))
            for k, v in value.items()
            if str(k) != "_wandb"
        }
    if isinstance(value, list):
        return [sanitize(item, key) for item in value]
    if isinstance(value, str):
        out = value
        for pattern in SENSITIVE_VALUE_RES:
            out = pattern.sub("<redacted>", out)
        return out
    return value


def unwrap_config(config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict) and "value" in value:
            value = value["value"]
        result[key] = sanitize(value, key)
    return result


def iso_to_epoch(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def normalize_model(value: Any) -> str:
    if value is None:
        return ""
    model = str(value).strip().lower().split("/")[-1]
    substitutions = {
        "-instruct-2507": "",
        "-instruct": "",
        "-it": "",
        "_": "-",
    }
    for old, new in substitutions.items():
        model = model.replace(old, new)
    return re.sub(r"[^a-z0-9]", "", model)


def first_value(mapping: dict[str, Any], names: Iterable[str]) -> Any:
    lower = {str(key).lower(): value for key, value in mapping.items()}
    for name in names:
        value = lower.get(name.lower())
        if value not in (None, ""):
            return value
    return None


def fetch_tinker(repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    load_tinker_key(repo_root)
    import tinker

    rest = tinker.ServiceClient().create_rest_client()

    runs: list[dict[str, Any]] = []
    offset = 0
    while True:
        response = rest.list_training_runs(
            limit=100, offset=offset, access_scope="owned"
        ).result(timeout=120)
        page = [item.model_dump(mode="json", exclude_none=True) for item in response.training_runs]
        runs.extend(page)
        total = response.cursor.total_count
        if not page or len(runs) >= total:
            break
        offset += len(page)

    checkpoints: list[dict[str, Any]] = []
    offset = 0
    while True:
        response = rest.list_user_checkpoints(limit=100, offset=offset).result(timeout=120)
        page = [item.model_dump(mode="json", exclude_none=True) for item in response.checkpoints]
        checkpoints.extend(page)
        total = response.cursor.total_count
        if not page or len(checkpoints) >= total:
            break
        offset += len(page)

    checkpoint_counts: Counter[str] = Counter()
    for checkpoint in checkpoints:
        match = TINKER_ID_RE.search(str(checkpoint.get("tinker_path", "")))
        if match:
            run_id = match.group("id").lower()
            checkpoint["training_run_id"] = run_id
            checkpoint_counts[run_id] += 1

    for run in runs:
        run_id = str(run["training_run_id"]).lower()
        run["training_run_id"] = run_id
        run["checkpoint_count"] = checkpoint_counts[run_id]
        run["base_uuid"] = run_id.split(":train:", 1)[0]
        run["normalized_model"] = normalize_model(run.get("base_model"))
    return runs, checkpoints


def graphql(key: str, query: str, variables: dict[str, Any]) -> dict[str, Any]:
    auth = base64.b64encode(f"api:{key}".encode()).decode()
    body = json.dumps({"query": query, "variables": variables}).encode()
    request = urllib.request.Request(
        GRAPHQL_URL,
        body,
        {"Authorization": "Basic " + auth, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    if result.get("errors"):
        raise RuntimeError(f"W&B GraphQL error: {result['errors']}")
    return result["data"]


def fetch_wandb(entity: str) -> tuple[list[str], list[dict[str, Any]]]:
    key = wandb_key()
    os.environ.setdefault("WANDB_API_KEY", key)
    import wandb

    projects = sorted(project.name for project in wandb.Api(timeout=60).projects(entity))
    runs: list[dict[str, Any]] = []
    for project in projects:
        cursor: str | None = None
        while True:
            data = graphql(
                key,
                WANDB_RUNS_QUERY,
                {"e": entity, "p": project, "c": cursor},
            )
            connection = data["project"]["runs"]
            for edge in connection["edges"]:
                node = edge["node"]
                config = unwrap_config(json.loads(node.get("config") or "{}"))
                summary = sanitize(json.loads(node.get("summaryMetrics") or "{}"))
                display_name = node.get("displayName") or node["name"]
                url = f"https://wandb.ai/{entity}/{project}/runs/{node['name']}"
                combined = json.dumps(
                    {
                        "config": config,
                        "summary": summary,
                        "display_name": display_name,
                        "tags": node.get("tags") or [],
                    },
                    sort_keys=True,
                )
                exact_ids = sorted(
                    {match.group("id").lower() for match in TINKER_ID_RE.finditer(combined)}
                )
                uuids = sorted(
                    {match.group("id").lower() for match in UUID_RE.finditer(combined)}
                )
                model = first_value(config, ("model", "base_model", "model_name", "model_short"))
                run = {
                    "entity": entity,
                    "project": project,
                    "run_id": node["name"],
                    "display_name": display_name,
                    "state": node.get("state"),
                    "created_at": node.get("createdAt"),
                    "updated_at": node.get("updatedAt"),
                    "group": node.get("group"),
                    "job_type": node.get("jobType"),
                    "tags": node.get("tags") or [],
                    "url": url,
                    "config": config,
                    "summary": summary,
                    "referenced_tinker_ids": exact_ids,
                    "referenced_uuids": uuids,
                    "model": model,
                    "normalized_model": normalize_model(model),
                }
                runs.append(run)
            page_info = connection["pageInfo"]
            if not page_info["hasNextPage"]:
                break
            cursor = page_info["endCursor"]
    return projects, runs


def correlate(
    tinker_runs: list[dict[str, Any]],
    wandb_runs: list[dict[str, Any]],
    candidate_window_minutes: float,
) -> list[dict[str, Any]]:
    exact_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    uuid_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in wandb_runs:
        for run_id in run["referenced_tinker_ids"]:
            exact_index[run_id].append(run)
        for uuid in run["referenced_uuids"]:
            uuid_index[uuid].append(run)

    tinker_uuid_counts = Counter(run["base_uuid"] for run in tinker_runs)
    match_info: dict[str, dict[str, Any]] = {}
    for tinker_run in tinker_runs:
        run_id = tinker_run["training_run_id"]
        base_uuid = tinker_run["base_uuid"]
        matches = exact_index.get(run_id, [])
        if matches:
            match_info[run_id] = {
                "matches": matches,
                "method": "exact_tinker_id",
                "confidence": "exact",
                "delta_minutes": None,
            }
        elif tinker_uuid_counts[base_uuid] == 1:
            matches = uuid_index.get(base_uuid, [])
            if matches:
                match_info[run_id] = {
                    "matches": matches,
                    "method": "unique_base_uuid",
                    "confidence": "strong",
                    "delta_minutes": None,
                }

    # For non-exact rows, accept a time/model candidate only when the pair is
    # mutually nearest and clearly separated from each side's runner-up. This
    # makes the candidate layer one-to-one and avoids recycling a W&B run as the
    # explanation for several Tinker jobs launched around the same time.
    used_wandb_urls = {
        match["url"]
        for info in match_info.values()
        for match in info["matches"]
    }
    unmatched_tinker = [
        run for run in tinker_runs if run["training_run_id"] not in match_info
    ]
    unmatched_wandb = [
        run
        for run in wandb_runs
        if run["url"] not in used_wandb_urls and not run["referenced_tinker_ids"]
    ]
    tinker_candidates: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    wandb_candidates: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    wandb_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in unmatched_wandb:
        if run["normalized_model"]:
            wandb_by_model[run["normalized_model"]].append(run)
    for tinker_run in unmatched_tinker:
        model = tinker_run["normalized_model"]
        end_epoch = iso_to_epoch(tinker_run.get("last_request_time"))
        if not model or end_epoch is None:
            continue
        for wandb_run in wandb_by_model[model]:
            start_epoch = iso_to_epoch(wandb_run.get("created_at"))
            if start_epoch is None:
                continue
            delta = abs(end_epoch - start_epoch) / 60.0
            if delta <= candidate_window_minutes:
                tinker_candidates[tinker_run["training_run_id"]].append(
                    (delta, wandb_run)
                )
                wandb_candidates[wandb_run["url"]].append((delta, tinker_run))
    for candidates in tinker_candidates.values():
        candidates.sort(key=lambda item: item[0])
    for candidates in wandb_candidates.values():
        candidates.sort(key=lambda item: item[0])

    for tinker_run in unmatched_tinker:
        run_id = tinker_run["training_run_id"]
        choices = tinker_candidates.get(run_id, [])
        if not choices:
            continue
        delta, wandb_run = choices[0]
        reverse_choices = wandb_candidates[wandb_run["url"]]
        tinker_separated = len(choices) == 1 or choices[1][0] - delta >= 5.0
        wandb_separated = (
            len(reverse_choices) == 1 or reverse_choices[1][0] - delta >= 5.0
        )
        reciprocal = reverse_choices[0][1]["training_run_id"] == run_id
        if reciprocal and tinker_separated and wandb_separated:
            match_info[run_id] = {
                "matches": [wandb_run],
                "method": "mutual_model_time_candidate",
                "confidence": "candidate",
                "delta_minutes": delta,
            }

    rows: list[dict[str, Any]] = []
    for tinker_run in tinker_runs:
        run_id = tinker_run["training_run_id"]
        info = match_info.get(
            run_id,
            {
                "matches": [],
                "method": "unmatched",
                "confidence": "unmatched",
                "delta_minutes": None,
            },
        )
        matches = info["matches"]
        method = info["method"]
        confidence = info["confidence"]
        delta_minutes = info["delta_minutes"]

        matches = sorted(
            matches,
            key=lambda run: run.get("updated_at") or run.get("created_at") or "",
            reverse=True,
        )
        primary = matches[0] if matches else {}
        config = primary.get("config", {})
        summary = primary.get("summary", {})
        final_metric = first_value(
            summary,
            (
                "final_reward",
                "eval_reward",
                "reward",
                "final_accuracy",
                "eval_accuracy",
                "frac_solved",
                "pass@1",
            ),
        )
        rows.append(
            {
                "training_run_id": run_id,
                "base_model": tinker_run.get("base_model"),
                "is_lora": tinker_run.get("is_lora"),
                "lora_rank": tinker_run.get("lora_rank"),
                "corrupted": tinker_run.get("corrupted"),
                "last_request_time": tinker_run.get("last_request_time"),
                "checkpoint_count": tinker_run.get("checkpoint_count", 0),
                "last_checkpoint_path": (
                    (tinker_run.get("last_sampler_checkpoint") or {}).get("tinker_path")
                    or (tinker_run.get("last_checkpoint") or {}).get("tinker_path")
                ),
                "match_confidence": confidence,
                "match_method": method,
                "matched_wandb_runs": len(matches),
                "wandb_project": primary.get("project"),
                "wandb_run_id": primary.get("run_id"),
                "wandb_name": primary.get("display_name"),
                "wandb_state": primary.get("state"),
                "wandb_url": primary.get("url"),
                "time_delta_minutes": (
                    round(delta_minutes, 3) if delta_minutes is not None else None
                ),
                "algorithm": first_value(config, ("algorithm", "algo", "method")),
                "dataset": first_value(config, ("dataset", "task", "environment", "env")),
                "group_size": first_value(config, ("group_size", "group", "G")),
                "learning_rate": first_value(config, ("learning_rate", "lr")),
                "steps": first_value(config, ("steps", "max_steps", "total_steps")),
                "seed": first_value(config, ("seed", "random_seed")),
                "final_metric": final_metric,
                "all_wandb_urls": [match["url"] for match in matches],
            }
        )
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def build_report(
    projects: list[str],
    tinker_runs: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
    wandb_runs: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    confidence = Counter(row["match_confidence"] for row in correlations)
    tinker_models = Counter(run.get("base_model") or "UNKNOWN" for run in tinker_runs)
    wandb_projects = Counter(run["project"] for run in wandb_runs)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tinker_run_count": len(tinker_runs),
        "tinker_checkpoint_count": len(checkpoints),
        "wandb_project_count": len(projects),
        "wandb_run_count": len(wandb_runs),
        "match_confidence_counts": dict(sorted(confidence.items())),
        "tinker_model_counts": dict(tinker_models.most_common()),
        "wandb_project_counts": dict(sorted(wandb_projects.items())),
        "credential_fields_redacted": True,
    }
    lines = [
        "# Tinker and W&B experiment registry",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Coverage",
        "",
        f"- Tinker training runs: **{len(tinker_runs)}**",
        f"- Tinker checkpoints: **{len(checkpoints)}**",
        f"- W&B projects: **{len(projects)}**",
        f"- W&B runs: **{len(wandb_runs)}**",
        f"- Exact ID matches: **{confidence.get('exact', 0)}**",
        f"- Unique UUID matches: **{confidence.get('strong', 0)}**",
        f"- Model/time candidates: **{confidence.get('candidate', 0)}**",
        f"- Unmatched Tinker runs: **{confidence.get('unmatched', 0)}**",
        "",
        "`candidate` is intentionally not treated as a confirmed join. Historical",
        "Tinker runs generally lack dataset, algorithm, learning-rate, and metric",
        "metadata, so exact correlation is possible only when a W&B run records the",
        "Tinker training ID (or its unique UUID).",
        "",
        "## W&B projects",
        "",
        "| project | runs |",
        "|---|---:|",
    ]
    lines.extend(f"| {project} | {wandb_projects[project]} |" for project in projects)
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `tinker_runs.jsonl`: all Tinker run metadata.",
            "- `tinker_checkpoints.jsonl`: every checkpoint returned by Tinker.",
            "- `wandb_runs.jsonl`: sanitized W&B metadata, config, and summary.",
            "- `tinker_wandb_correlation.csv`: one row per Tinker run.",
            "- `manifest.json`: counts and distributions.",
            "",
        ]
    )
    return manifest, "\n".join(lines)


def publish(
    entity: str,
    project: str,
    output_dir: Path,
    manifest: dict[str, Any],
    correlations: list[dict[str, Any]],
) -> str:
    os.environ.setdefault("WANDB_API_KEY", wandb_key())
    import wandb

    stamp = datetime.now().astimezone().strftime("%Y%m%d")
    run = wandb.init(
        entity=entity,
        project=project,
        name=f"tinker-experiment-registry-{stamp}",
        job_type="experiment-registry",
        group="review-readiness",
        tags=["tinker", "registry", "correlation", "review"],
        config={
            "registry_schema_version": 1,
            "source": "tinker-api+wandb-api",
            **{key: value for key, value in manifest.items() if key.endswith("_count")},
        },
        reinit="finish_previous",
    )
    columns = [
        "training_run_id",
        "base_model",
        "lora_rank",
        "corrupted",
        "last_request_time",
        "checkpoint_count",
        "last_checkpoint_path",
        "match_confidence",
        "match_method",
        "wandb_project",
        "wandb_name",
        "wandb_state",
        "wandb_url",
        "algorithm",
        "dataset",
        "group_size",
        "learning_rate",
        "steps",
        "seed",
        "final_metric",
    ]
    table = wandb.Table(columns=columns)
    for row in correlations:
        table.add_data(*(row.get(column) for column in columns))
    run.log({"registry/tinker_experiments": table})

    artifact = wandb.Artifact(
        name="tinker-experiment-registry",
        type="dataset",
        description="Complete sanitized Tinker inventory and W&B correlation.",
        metadata=manifest,
    )
    artifact.add_dir(str(output_dir))
    run.log_artifact(artifact, aliases=["latest", stamp])
    run.summary.update(manifest)
    run_url = run.url
    run.finish()
    return run_url


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching Tinker runs and checkpoints...", flush=True)
    tinker_runs, checkpoints = fetch_tinker(repo_root)
    print(
        f"Fetched {len(tinker_runs)} Tinker runs and {len(checkpoints)} checkpoints.",
        flush=True,
    )

    print("Fetching W&B projects and runs...", flush=True)
    projects, wandb_runs = fetch_wandb(args.entity)
    print(
        f"Fetched {len(wandb_runs)} W&B runs across {len(projects)} projects.",
        flush=True,
    )

    correlations = correlate(
        tinker_runs, wandb_runs, args.candidate_window_minutes
    )
    manifest, report = build_report(
        projects, tinker_runs, checkpoints, wandb_runs, correlations
    )

    write_jsonl(args.output_dir / "tinker_runs.jsonl", tinker_runs)
    write_jsonl(args.output_dir / "tinker_checkpoints.jsonl", checkpoints)
    write_jsonl(args.output_dir / "wandb_runs.jsonl", wandb_runs)
    write_csv(args.output_dir / "tinker_wandb_correlation.csv", correlations)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "README.md").write_text(report + "\n")

    if args.publish:
        print("Publishing registry table and artifact to W&B...", flush=True)
        run_url = publish(
            args.entity,
            args.registry_project,
            args.output_dir,
            manifest,
            correlations,
        )
        manifest["wandb_registry_url"] = run_url
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        print(f"W&B registry: {run_url}")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
