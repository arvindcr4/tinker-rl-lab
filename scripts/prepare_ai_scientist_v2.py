#!/usr/bin/env python3
"""Prepare a redacted AI Scientist-v2 run pack for this repository.

This script does not call an LLM. It builds a local context bundle that can be
passed to Sakana AI Scientist-v2 for ideation, then imported back through
scripts/import_ai_scientist_v2_ideas.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATIC_TOPIC = ROOT / "ai-scientist-v2" / "tinker_rl_lab_topic.md"
CONTRACT = ROOT / "ai-scientist-v2" / "experiment_contract.md"

SECRET_PATTERNS = [
    re.compile(rb"wandb_v1_[A-Za-z0-9_\-]{20,}"),
    re.compile(rb"tml-[A-Za-z0-9_\-]{20,}"),
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
]

CONTEXT_SOURCES = [
    ("AI Scientist-v2 contract", "ai-scientist-v2/experiment_contract.md", 12_000),
    ("Research loop plan", "research_loop/README.md", 14_000),
    ("Current best recipe", "research_loop/best_recipe.yaml", 8_000),
    ("Research-loop learnings", "research_loop/learnings.md", 16_000),
    ("Seed hypotheses", "research_loop/hypotheses_seed.md", 12_000),
    ("BrowserGym Tinker plan", "docs/research/browsergym_tinker_plan.md", 18_000),
    ("Result ledger", "reports/final/result_ledger.md", 22_000),
    ("Statistical analysis", "experiments/statistical_analysis.md", 12_000),
    ("ZVF validation", "experiments/zvf_predictive_validation.md", 10_000),
    ("NIA sources", "nia-sources.md", 12_000),
]


def resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def token_exact_values() -> list[bytes]:
    names = [
        "WANDB_API_KEY",
        "TINKER_API_KEY",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "S2_API_KEY",
    ]
    values: list[bytes] = []
    for name in names:
        value = os.environ.get(name)
        if value:
            values.append(value.encode("utf-8"))
    return values


def redact(data: bytes, exact_values: list[bytes]) -> bytes:
    out = data
    for pattern in SECRET_PATTERNS:
        out = pattern.sub(b"[REDACTED_SECRET]", out)
    for value in exact_values:
        out = out.replace(value, b"[REDACTED_SECRET]")
    return out


def text_excerpt(path: Path, limit: int, exact_values: list[bytes]) -> tuple[str, dict[str, Any]]:
    if not path.exists():
        return "", {"path": str(path.relative_to(ROOT)), "exists": False}

    raw = path.read_bytes()
    redacted = redact(raw, exact_values)
    truncated = len(redacted) > limit
    excerpt = redacted[:limit].decode("utf-8", errors="replace")
    rel = path.relative_to(ROOT).as_posix()
    meta = {
        "path": rel,
        "exists": True,
        "bytes": len(raw),
        "redacted_sha256": hashlib.sha256(redacted).hexdigest(),
        "excerpt_bytes": min(len(redacted), limit),
        "truncated": truncated,
    }
    if truncated:
        excerpt += "\n\n[TRUNCATED]\n"
    return excerpt, meta


def build_context(max_context_chars: int, exact_values: list[bytes]) -> tuple[str, list[dict[str, Any]]]:
    sections: list[str] = []
    manifest: list[dict[str, Any]] = []
    total = 0

    for title, rel_path, limit in CONTEXT_SOURCES:
        path = resolve_repo_path(rel_path)
        excerpt, meta = text_excerpt(path, limit, exact_values)
        manifest.append({"title": title, **meta})
        if not excerpt:
            continue
        remaining = max_context_chars - total
        if remaining <= 0:
            break
        if len(excerpt) > remaining:
            excerpt = excerpt[:remaining] + "\n\n[CONTEXT LIMIT REACHED]\n"
        sections.append(f"## {title}\n\nSource: `{rel_path}`\n\n```text\n{excerpt}\n```\n")
        total += len(excerpt)

    return "\n".join(sections), manifest


def write_run_script(dest: Path) -> Path:
    script = dest / "run_ideation.sh"
    ideas_json = dest / "tinker_rl_lab_topic.json"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        'REPO_ROOT="${REPO_ROOT:-/Users/arvind/tinker-rl-lab}"\n'
        'AI_SCIENTIST_V2_DIR="${AI_SCIENTIST_V2_DIR:-$HOME/AI-Scientist-v2}"\n'
        'MODEL="${AI_SCIENTIST_MODEL:-gpt-4o-2024-11-20}"\n'
        'MAX_IDEAS="${AI_SCIENTIST_MAX_IDEAS:-12}"\n'
        'REFLECTIONS="${AI_SCIENTIST_REFLECTIONS:-3}"\n\n'
        'if [[ ! -d "$AI_SCIENTIST_V2_DIR" ]]; then\n'
        '  echo "AI_SCIENTIST_V2_DIR does not exist: $AI_SCIENTIST_V2_DIR" >&2\n'
        "  exit 1\n"
        "fi\n\n"
        'cd "$AI_SCIENTIST_V2_DIR"\n'
        "python ai_scientist/perform_ideation_temp_free.py \\\n"
        f'  --workshop-file "{dest / "tinker_rl_lab_topic.md"}" \\\n'
        '  --model "$MODEL" \\\n'
        '  --max-num-generations "$MAX_IDEAS" \\\n'
        '  --num-reflections "$REFLECTIONS"\n\n'
        'echo "If ideation wrote the JSON elsewhere, move or pass that path manually."\n'
        "python \"$REPO_ROOT/scripts/import_ai_scientist_v2_ideas.py\" \\\n"
        f'  --ideas "{ideas_json}" \\\n'
        "  --dry-run\n",
        encoding="utf-8",
    )
    mode = script.stat().st_mode
    script.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / ".ai_scientist_v2_runs" / "latest",
        help="where to write the generated run pack",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=80_000,
        help="maximum characters of local context appended to the topic file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace output directory if it already exists",
    )
    args = parser.parse_args()

    dest = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if dest.exists():
        generated_latest = dest.name == "latest" and dest.parent.name == ".ai_scientist_v2_runs"
        if args.force or generated_latest:
            shutil.rmtree(dest)
        else:
            raise SystemExit(f"output directory already exists; pass --force: {dest}")
    dest.mkdir(parents=True, exist_ok=True)

    exact_values = token_exact_values()
    context, source_manifest = build_context(args.max_context_chars, exact_values)
    static_topic = redact(STATIC_TOPIC.read_bytes(), exact_values).decode("utf-8", errors="replace")

    topic = (
        static_topic
        + "\n\n# Local Context Digest\n\n"
        + "The following redacted excerpts are provided so AI Scientist-v2 can "
        + "propose experiments that fit this repository's current evidence and "
        + "execution surfaces.\n\n"
        + context
    )
    topic_path = dest / "tinker_rl_lab_topic.md"
    topic_path.write_text(topic, encoding="utf-8")

    context_path = dest / "context.md"
    context_path.write_text("# Local Context Digest\n\n" + context, encoding="utf-8")
    shutil.copy2(CONTRACT, dest / "experiment_contract.md")
    run_script = write_run_script(dest)

    # Fail closed if a token-shaped value slipped through after redaction.
    for path in [topic_path, context_path, dest / "experiment_contract.md", run_script]:
        data = path.read_bytes()
        for pattern in SECRET_PATTERNS:
            if pattern.search(data):
                raise RuntimeError(f"token-shaped content remained in {path}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(ROOT),
        "output_dir": str(dest),
        "topic_file": str(topic_path),
        "context_file": str(context_path),
        "run_script": str(run_script),
        "max_context_chars": args.max_context_chars,
        "sources": source_manifest,
        "next_steps": [
            f"bash {run_script}",
            "python3 scripts/import_ai_scientist_v2_ideas.py --ideas <generated-ideas.json> --dry-run",
            "python3 scripts/import_ai_scientist_v2_ideas.py --ideas <generated-ideas.json>",
            "python3 research_loop/coordinator.py wave new --size 4 --phase 1",
        ],
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
