"""Read-only audit aggregation with explicit completion and metric contracts.

No network or credentials are touched at import. Missing/failed evidence is
counted as excluded, never pooled as a zero or an equivalent estimand.
"""

from __future__ import annotations
import base64
import collections
import json
import math
import os
from pathlib import Path
import statistics
import urllib.request

ENT, PROJ = "arvindcr4-pes-university", "zvf-audit"
METRICS = {
    "zvf": ("mean_zvf", "zvf"),
    "gu": ("gu", "mean_gu"),
    "reward": ("final_reward", "reward"),
    "frac_solved": ("frac_solved",),
}


def aggregate(runs, dimension="model"):
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    excluded = collections.Counter()
    for state, config, summary, _run_id in runs:
        if state != "finished" or summary.get("completion_contract_satisfied") is not True:
            excluded["completion_unverified"] += 1
            continue
        spec = config.get("comparison_hash")
        definitions = config.get("metric_definitions", {})
        if not spec or not isinstance(definitions, dict):
            excluded["comparison_or_metric_contract_missing"] += 1
            continue
        dimension_value = config.get(dimension)
        if dimension_value is None:
            excluded["dimension_missing"] += 1
            continue
        for metric, names in METRICS.items():
            definition = definitions.get(metric)
            value = next((summary[n] for n in names if summary.get(n) is not None), None)
            if (
                not definition
                or not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                excluded[f"{metric}_missing"] += 1
                continue
            # Separate estimands and specifications even under the same visible label.
            key = json.dumps([dimension_value, spec, metric, definition])
            groups[key][metric].append(float(value))
    return {
        "groups": {
            k: {metric: {"mean": statistics.mean(v), "n": len(v)} for metric, v in values.items()}
            for k, values in groups.items()
        },
        "excluded": dict(excluded),
        "input_runs": len(runs),
    }


def fetch_runs():
    key = os.environ["WANDB_API_KEY"]
    auth = base64.b64encode(f"api:{key}".encode()).decode()
    query = "query($e:String!,$p:String!,$c:String){project(name:$p,entityName:$e){runs(first:250,after:$c){edges{node{name state config summaryMetrics}}pageInfo{hasNextPage endCursor}}}}"
    runs, cursor = [], None
    while True:
        body = json.dumps(
            {"query": query, "variables": {"e": ENT, "p": PROJ, "c": cursor}}
        ).encode()
        request = urllib.request.Request(
            "https://api.wandb.ai/graphql",
            body,
            {"Authorization": "Basic " + auth, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.load(response)
        if payload.get("errors"):
            raise RuntimeError("W&B inventory incomplete")
        connection = payload["data"]["project"]["runs"]
        for edge in connection["edges"]:
            node = edge["node"]
            config = {
                k: v.get("value", v) if isinstance(v, dict) else v
                for k, v in json.loads(node.get("config") or "{}").items()
            }
            runs.append(
                (
                    node["state"],
                    config,
                    json.loads(node.get("summaryMetrics") or "{}"),
                    node["name"],
                )
            )
        if not connection["pageInfo"]["hasNextPage"]:
            return runs
        cursor = connection["pageInfo"]["endCursor"]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(aggregate(fetch_runs()), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
