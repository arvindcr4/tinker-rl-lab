"""Candidate v2 limited W&B history export; never claim complete pagination.

Credentials are read only inside explicit main(). Config, summary and sampled
history come from one response; this is not an atomic immutable W&B snapshot.
Missing/duplicate expected steps or failed downloads make the CLI exit nonzero.
A new output path is required; historical exports are never overwritten.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import netrc
import pathlib
import urllib.request

REG = (
    pathlib.Path(__file__).resolve().parents[1]
    / "experiments/results/tinker_wandb_registry/wandb_runs.jsonl"
)
VERSION = "zvf-history-limited-export-v2-candidate"
SPEC = json.dumps(
    {
        "keys": ["step", "zvf", "reward", "gu", "n_groups", "n_datums", "frac_solved"],
        "samples": 1000,
    }
)
QUERY = 'query($s:[JSONString!]!,$n:String!){ project(name:"zvf-audit",entityName:"arvindcr4-pes-university"){ run(name:$n){ name state config summaryMetrics sampledHistory(specs:$s) } } }'


def validate_history(run, history):
    expected = run["config"].get("steps")
    if type(expected) is not int or expected <= 0:
        raise ValueError("positive expected steps required")
    steps = [x.get("step") for x in history]
    if (
        len(steps) != expected
        or any(type(step) is not int for step in steps)
        or set(steps) != set(range(expected))
    ):
        raise ValueError("incomplete or duplicate expected-step history")
    if not all(
        all(type(row.get(k)) in (int, float) and math.isfinite(row[k]) for k in ("reward", "zvf"))
        for row in history
    ):
        raise ValueError("nonfinite/missing history outcome")


def get(record, authorization, opener=urllib.request.urlopen):
    request = urllib.request.Request(
        "https://api.wandb.ai/graphql",
        json.dumps({"query": QUERY, "variables": {"s": [SPEC], "n": record["run_id"]}}).encode(),
        {"Authorization": "Basic " + authorization, "Content-Type": "application/json"},
    )
    for _ in range(3):
        try:
            with opener(request, timeout=60) as response:
                data = json.load(response)
            if data.get("errors"):
                raise ValueError("GraphQL returned errors")
            run = data["data"]["project"]["run"]
            if run["name"] != record["run_id"]:
                raise ValueError("run identity mismatch")
            config = json.loads(run["config"]) if isinstance(run["config"], str) else run["config"]
            config = {
                k: (v["value"] if isinstance(v, dict) and "value" in v else v)
                for k, v in config.items()
                if not k.startswith("_")
            }
            expected_config = {k: v for k, v in record["config"].items() if not k.startswith("_")}
            if config != expected_config:
                raise ValueError("configuration changed since run registry snapshot")
            summary = (
                json.loads(run["summaryMetrics"])
                if isinstance(run["summaryMetrics"], str)
                else run["summaryMetrics"]
            )
            output = {
                "run_id": run["name"],
                "name": record["display_name"],
                "model": record["model"],
                "state": run["state"],
                "config": config,
                "summary": summary,
                "history": run["sampledHistory"][0],
                "export_version": VERSION,
                "retrieval_status": "limited_sampled_history",
                "sample_limit": 1000,
                "snapshot_status": "single_response_not_immutable_atomic_snapshot",
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            validate_history(output, output["history"])
            output["expected_step_coverage"] = "all_expected_step_ids_present"
            return output
        except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
            last = str(error)
    return {
        "run_id": record["run_id"],
        "export_version": VERSION,
        "error": last,
        "retrieval_status": "incomplete",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("refuse to overwrite history artifact")
    records = [json.loads(line) for line in REG.read_text().splitlines() if line.strip()]
    records = [r for r in records if r["project"] == "zvf-audit"]
    if not records or len({r["run_id"] for r in records}) != len(records):
        raise ValueError("empty/duplicate registry cohort")
    credentials = netrc.netrc().authenticators("api.wandb.ai")
    if credentials is None:
        raise ValueError("W&B credentials unavailable")
    authorization = base64.b64encode(("api:" + credentials[2]).encode()).decode()
    with ThreadPoolExecutor(12) as executor:
        result = list(executor.map(lambda record: get(record, authorization), records))
    registry_hash = hashlib.sha256(REG.read_bytes()).hexdigest()
    for item in result:
        item["registry_sha256"] = registry_hash
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    errors = sum("error" in item for item in result)
    print(
        json.dumps(
            {
                "schema_version": VERSION,
                "runs": len(result),
                "errors": errors,
                "complete_paginated_export": False,
            }
        )
    )
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
