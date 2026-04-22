#!/usr/bin/env python3
"""Import AI Scientist-v2 idea JSON into research_loop/queue.jsonl.

The importer is deliberately non-executing. It extracts hypotheses from an idea
file and appends queue entries consumed by research_loop/coordinator.py.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT / "research_loop" / "queue.jsonl"

SECRET_PATTERNS = [
    re.compile(rb"wandb_v1_[A-Za-z0-9_\-]{20,}"),
    re.compile(rb"tml-[A-Za-z0-9_\-]{20,}"),
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
]

TITLE_KEYS = ["Title", "title", "Name", "name", "idea_name", "Idea Name"]
HYPOTHESIS_KEYS = ["Hypothesis", "hypothesis", "Idea", "idea", "Motivation", "motivation"]
EXPERIMENT_KEYS = [
    "Experiment",
    "experiment",
    "Experiments",
    "experiments",
    "Proposed Experiment",
    "proposed_experiment",
    "Plan",
    "plan",
]
RATIONALE_KEYS = [
    "Rationale",
    "rationale",
    "Expected Results",
    "expected_results",
    "Interestingness",
    "interestingness",
    "Novelty",
    "novelty",
]


def assert_no_token_shapes(data: bytes, source: Path) -> None:
    for pattern in SECRET_PATTERNS:
        if pattern.search(data):
            raise RuntimeError(f"refusing to import token-shaped content from {source}")


def load_json(path: Path) -> Any:
    data = path.read_bytes()
    assert_no_token_shapes(data, path)
    return json.loads(data.decode("utf-8"))


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "; ".join(as_text(v) for v in value if as_text(v))
    if isinstance(value, dict):
        parts = []
        for key, val in value.items():
            text = as_text(val)
            if text:
                parts.append(f"{key}: {text}")
        return "; ".join(parts)
    return str(value).strip()


def first_present(obj: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key in obj:
            text = as_text(obj[key])
            if text:
                return text
    return ""


def extract_ideas(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ["ideas", "Ideas", "generated_ideas", "research_ideas", "Research Ideas"]:
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        # Some generators return a dict keyed by idea names.
        dict_values = [v for v in payload.values() if isinstance(v, dict)]
        if dict_values:
            return dict_values
        return [payload]
    return []


def compact(text: str, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def slug(text: str, fallback: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return (s or fallback)[:48]


def infer_category(text: str, default: str) -> str:
    lower = text.lower()
    if any(term in lower for term in ["browsergym", "webarena", "miniwob", "workarena"]):
        return "browser"
    if any(term in lower for term in ["zvf", "zero-variance", "zero variance"]):
        return "zvf"
    if any(term in lower for term in ["reward", "partial credit", "verifier"]):
        return "reward"
    if any(term in lower for term in ["curriculum", "easy_first", "hard_first"]):
        return "curriculum"
    if any(term in lower for term in ["group size", "group_size", "batch diversity"]):
        return "group"
    return default


def infer_knobs(text: str) -> str:
    lower = text.lower()
    knobs: list[str] = []

    for field in ["group_size", "batch", "rank", "steps", "max_tokens", "eval_subset"]:
        for match in re.finditer(rf"{field}\s*[:=]\s*([0-9]+)", text):
            knobs.append(f"{field}: {match.group(1)}")

    for field in ["lr", "temperature"]:
        for match in re.finditer(rf"{field}\s*[:=]\s*([0-9.eE+-]+)", text):
            value = match.group(1).rstrip(".,;)")
            knobs.append(f"{field}: {value}")

    if "easy_first" in lower or "easy-first" in lower:
        knobs.append("curriculum: easy_first")
    if "hard_first" in lower or "hard-first" in lower:
        knobs.append("curriculum: hard_first")
    if "rank-based" in lower or "adv_norm=rank" in lower or "adv_norm: rank" in lower:
        knobs.append("adv_norm: rank")
    if "adv_norm=none" in lower or "adv_norm: none" in lower:
        knobs.append("adv_norm: none")
    if "graded" in lower or "partial credit" in lower:
        knobs.append("reward_shape: graded")
    if "partial" in lower and "reward" in lower:
        knobs.append("reward_shape: partial")
    if "browsergym" in lower or "webarena" in lower or "miniwob" in lower:
        knobs.append("route: browsergym_config_proposal")

    deduped: list[str] = []
    for knob in knobs:
        if knob not in deduped:
            deduped.append(knob)
    return ", ".join(deduped) if deduped else "AI Scientist-v2 proposal; agent must map to safe YAML/config"


def make_queue_record(
    idea: dict[str, Any],
    *,
    index: int,
    ideas_path: Path,
    default_category: str,
    run_id: str,
) -> dict[str, Any]:
    title = first_present(idea, TITLE_KEYS) or f"AI Scientist-v2 idea {index:03d}"
    hypothesis = first_present(idea, HYPOTHESIS_KEYS)
    experiment = first_present(idea, EXPERIMENT_KEYS)
    rationale = first_present(idea, RATIONALE_KEYS)
    all_text = "\n".join([title, hypothesis, experiment, rationale, as_text(idea)])

    oneline_source = hypothesis or experiment or title
    oneline = compact(f"{title}: {oneline_source}")
    category = infer_category(all_text, default_category)
    knobs = infer_knobs(all_text)
    rationale_text = compact(
        " | ".join(part for part in [hypothesis, experiment, rationale] if part),
        limit=900,
    )

    return {
        "id": f"ASV2-{run_id}-{index:03d}-{slug(title, str(index))}",
        "category": category,
        "oneline": oneline,
        "knobs": knobs,
        "rationale": rationale_text or "Imported from AI Scientist-v2 idea JSON.",
        "source": {
            "tool": "ai_scientist_v2",
            "ideas_file": str(ideas_path),
            "idea_index": index,
            "title": title,
            "imported_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def load_existing_queue(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_queue(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ideas", required=True, type=Path, help="AI Scientist-v2 idea JSON")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--category", default="ai_scientist_v2")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-duplicates", action="store_true")
    args = parser.parse_args()

    ideas_path = args.ideas if args.ideas.is_absolute() else ROOT / args.ideas
    queue_path = args.queue if args.queue.is_absolute() else ROOT / args.queue
    payload = load_json(ideas_path)
    ideas = extract_ideas(payload)
    if args.limit is not None:
        ideas = ideas[: args.limit]
    if not ideas:
        raise SystemExit(f"no idea objects found in {ideas_path}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    records = [
        make_queue_record(
            idea,
            index=i,
            ideas_path=ideas_path,
            default_category=args.category,
            run_id=run_id,
        )
        for i, idea in enumerate(ideas, 1)
    ]

    existing = load_existing_queue(queue_path)
    existing_onelines = {compact(row.get("oneline", "")).lower() for row in existing}
    if not args.allow_duplicates:
        records = [
            row for row in records
            if compact(row.get("oneline", "")).lower() not in existing_onelines
        ]

    if args.dry_run:
        print(json.dumps({"queue": str(queue_path), "records": records}, indent=2))
        return 0

    write_queue(queue_path, existing + records)
    print(json.dumps({"queue": str(queue_path), "imported": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
