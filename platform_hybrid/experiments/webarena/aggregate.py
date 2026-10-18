"""Aggregate per-shard JSONL outputs from react_eval.py into a benchmark score."""
from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    env_id: str
    score: float
    num_steps: int
    valid_action_count: int
    wall_time_sec: float
    error: str | None = None
    seed: int | None = None
    evaluation_id: str | None = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EpisodeResult:
        return cls(
            env_id=data["env_id"],
            score=data["score"],
            num_steps=data["num_steps"],
            valid_action_count=data["valid_action_count"],
            wall_time_sec=data.get("wall_time_sec", 0.0),
            error=data.get("error"),
            seed=data["seed"],
            evaluation_id=data["evaluation_id"],
        )


@dataclass
class AggregationResult:
    n: int
    success_rate: float
    mean_reward: float
    episodes_with_error: int
    mean_num_steps: float
    mean_valid_actions: float
    total_wall_time_sec: float
    max_wall_time_sec: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "success_rate": self.success_rate,
            "mean_reward": self.mean_reward,
            "episodes_with_error": self.episodes_with_error,
            "mean_num_steps": self.mean_num_steps,
            "mean_valid_actions": self.mean_valid_actions,
            "total_wall_time_sec": self.total_wall_time_sec,
            "max_wall_time_sec": self.max_wall_time_sec,
        }


def parse_results(paths: Sequence[Path]) -> List[EpisodeResult]:
    """Fail closed on corrupt, unversioned, or conflicting episode records."""
    records = {}
    for path in paths:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                item = EpisodeResult.from_dict(data)
                if not item.evaluation_id or type(item.seed) is not int:
                    raise ValueError("missing episode identity")
                if type(item.score) not in (int, float) or not math.isfinite(item.score) or not 0 <= item.score <= 1:
                    raise ValueError("invalid score")
                if "total_reward" in data:
                    total = data["total_reward"]
                    if type(total) not in (int, float) or not math.isfinite(total) or not 0 <= total <= item.num_steps:
                        raise ValueError("invalid total reward")
                if item.num_steps < 0 or not 0 <= item.valid_action_count <= item.num_steps:
                    raise ValueError("invalid action counts")
                key = (item.evaluation_id, item.env_id, item.seed)
                if key in records and records[key] != item:
                    raise ValueError("conflicting duplicate episode")
                records[key] = item
            except Exception as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    if len({r.evaluation_id for r in records.values()}) > 1:
        raise ValueError("cannot pool different evaluation configurations")
    return list(records.values())


def summarize(items: List[EpisodeResult], success_threshold: float = 1.0) -> AggregationResult:
    """Summarize a list of EpisodeResult objects."""
    n = len(items)
    if n == 0:
        return AggregationResult(
            n=0, success_rate=0.0, mean_reward=0.0, episodes_with_error=0,
            mean_num_steps=0.0, mean_valid_actions=0.0, total_wall_time_sec=0.0,
            max_wall_time_sec=0.0,
        )

    scores = [r.score for r in items]
    succ = sum(1 for s in scores if s >= success_threshold)
    errs = sum(1 for r in items if r.error)
    steps = [r.num_steps for r in items]
    actions = [r.valid_action_count for r in items]
    walltimes = [r.wall_time_sec for r in items]
    
    return AggregationResult(
        n=n,
        success_rate=succ / n,
        mean_reward=statistics.fmean(scores),
        episodes_with_error=errs,
        mean_num_steps=statistics.fmean(steps),
        mean_valid_actions=statistics.fmean(actions),
        total_wall_time_sec=sum(walltimes),
        max_wall_time_sec=max(walltimes) if walltimes else 0.0,
    )


def aggregate_results(results: List[EpisodeResult], success_threshold: float = 1.0, num_shards: int = 1) -> Dict[str, Any]:
    """Aggregate total results and by-benchmark results."""
    by_prefix: Dict[str, List[EpisodeResult]] = defaultdict(list)
    for r in results:
        # browsergym/webarena.310 -> webarena
        prefix = r.env_id.split("/", 1)[-1].split(".", 1)[0]
        by_prefix[prefix].append(r)

    out = {
        "completion_status": "coverage_unverified",
        "overall": summarize(results, success_threshold).to_dict(),
        "by_benchmark": {k: summarize(v, success_threshold).to_dict() for k, v in by_prefix.items()},
        "num_shards": num_shards,
    }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="JSONL files or glob (e.g. results_shard_*.jsonl)")
    p.add_argument("--receipts", nargs="+", help="Matching per-shard evaluation receipt JSON files")
    p.add_argument("--manifest", help="JSON list of expected env_id/seed/evaluation_id records")
    p.add_argument("--out", required=True, help="Where to write aggregated JSON")
    p.add_argument("--success-threshold", type=float, default=1.0,
                   help="Score >= this counts as success (WebArena uses 1.0)")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    paths: List[Path] = []
    for pat in args.inputs:
        matched = list(Path().glob(pat)) if "*" in pat else [Path(pat)]
        paths.extend(m for m in matched if m.exists())
    if not paths:
        logger.error("no input files matched")
        return 1

    results = parse_results(paths)
    out = aggregate_results(results, args.success_threshold, len(paths))
    
    if args.receipts:
        receipts = [json.loads(Path(path).read_text()) for path in args.receipts]
        identities = {r.evaluation_id for r in results}
        if {r["evaluation_id"] for r in receipts} != identities or len(identities) != 1:
            raise ValueError("receipt identity mismatch")
        if any(r["effective_config"] != receipts[0]["effective_config"] for r in receipts):
            raise ValueError("receipt configuration mismatch")
        out["evaluation_receipt"] = receipts[0]
    if args.manifest:
        expected = {(r["evaluation_id"], r["env_id"], r["seed"]) for r in json.loads(Path(args.manifest).read_text())}
        observed = {(r.evaluation_id, r.env_id, r.seed) for r in results}
        if expected != observed:
            raise ValueError("episode coverage does not match expected manifest")
        out["completion_status"] = "complete_with_errors" if any(r.error for r in results) else "complete"
    Path(args.out).write_text(json.dumps(out, indent=2))
    logger.info("Aggregated %d results into %s", len(results), args.out)
    logger.info("Result:\n%s", json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
