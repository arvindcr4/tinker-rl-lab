"""Aggregate per-shard JSONL outputs from react_eval.py into a benchmark score."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="JSONL files or glob (e.g. results_shard_*.jsonl)")
    p.add_argument("--out", required=True, help="Where to write aggregated JSON")
    p.add_argument("--success-threshold", type=float, default=1.0,
                   help="Score >= this counts as success (WebArena uses 1.0)")
    args = p.parse_args()

    paths: list[Path] = []
    for pat in args.inputs:
        matched = list(Path().glob(pat)) if "*" in pat else [Path(pat)]
        paths.extend(m for m in matched if m.exists())
    if not paths:
        raise SystemExit("no input files matched")

    rows: list[dict] = []
    seen_ids: set[str] = set()
    for p_in in paths:
        for line in p_in.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["env_id"] in seen_ids:
                continue                       # first-writer-wins on dedup
            seen_ids.add(r["env_id"])
            rows.append(r)

    by_prefix: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        # browsergym/webarena.310 -> webarena
        prefix = r["env_id"].split("/", 1)[-1].split(".", 1)[0]
        by_prefix[prefix].append(r)

    def summarize(items: list[dict]) -> dict:
        scores = [r["score"] for r in items]
        succ = sum(1 for s in scores if s >= args.success_threshold)
        errs = sum(1 for r in items if r.get("error"))
        steps = [r["num_steps"] for r in items]
        actions = [r["valid_action_count"] for r in items]
        walltimes = [r["wall_time_sec"] for r in items]
        return {
            "n": len(items),
            "success_rate": succ / len(items) if items else 0.0,
            "mean_reward": statistics.fmean(scores) if scores else 0.0,
            "episodes_with_error": errs,
            "mean_num_steps": statistics.fmean(steps) if steps else 0.0,
            "mean_valid_actions": statistics.fmean(actions) if actions else 0.0,
            "total_wall_time_sec": sum(walltimes),
            "max_wall_time_sec": max(walltimes) if walltimes else 0.0,
        }

    out = {
        "overall": summarize(rows),
        "by_benchmark": {k: summarize(v) for k, v in by_prefix.items()},
        "num_shards": len(paths),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
