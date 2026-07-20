#!/usr/bin/env python3
"""Compute root-macro PPO/SAO signal-survival metrics from token JSONL."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

try:  # package execution
    from .metrics import ppo_gate, root_metrics, sao_gate
except ImportError:  # direct script execution
    from metrics import ppo_gate, root_metrics, sao_gate


def load_records(path: Path, algorithm: str, epsilon: float, epsilon_low: float, epsilon_high: float):
    records = []
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        for field in ("root_trajectory_id", "ratio", "advantage"):
            if field not in record:
                raise ValueError(f"{path}:{line_number}: missing {field}")
        if algorithm == "ppo":
            computed = ppo_gate(record["ratio"], record["advantage"], epsilon)
        elif algorithm == "sao":
            computed = sao_gate(record["ratio"], epsilon_low, epsilon_high)
        else:
            computed = int(record.get("gate", 1))
        if "gate" in record and int(record["gate"]) != computed:
            raise ValueError(
                f"{path}:{line_number}: logged gate {record['gate']} != computed gate {computed}"
            )
        record["gate"] = computed
        records.append(record)
    return records


def summarize(roots, threshold: float):
    values = list(roots.values())
    if not values:
        raise ValueError("trace contains no roots")
    return {
        "n_roots": len(values),
        "root_macro_pam": statistics.fmean(item["pam"] for item in values),
        "root_macro_gsr": statistics.fmean(item["gsr"] for item in values),
        "root_macro_egm": statistics.fmean(item["egm"] for item in values),
        "zuf": statistics.fmean(float(item["egm"] <= threshold) for item in values),
        "zuf_threshold": threshold,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--algorithm", choices=("grpo", "ppo", "sao"), required=True)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon-low", type=float, default=0.2)
    parser.add_argument("--epsilon-high", type=float, default=0.2)
    parser.add_argument("--zuf-threshold", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    records = load_records(
        args.trace, args.algorithm, args.epsilon, args.epsilon_low, args.epsilon_high
    )
    roots = root_metrics(records)
    report = {
        "algorithm": args.algorithm,
        "summary": summarize(roots, args.zuf_threshold),
        "roots": roots,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
