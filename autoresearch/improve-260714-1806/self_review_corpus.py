#!/usr/bin/env python3
"""Deterministic file-level review ledger for the canonical paper corpus.

This script does not call an LLM.  It enumerates the include closure produced by
``inventory_papers.py`` and records structural and claim-review hotspots so the
human/self-review pass has an explicit, reproducible coverage boundary.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from audit_papers import PAPERS


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
INCLUDE_MAP = OUT / "include_map.json"

CITE_RE = re.compile(r"\\cite\w*(?:\[[^]]*\])*\{([^}]+)\}")
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
SECTION_RE = re.compile(r"\\(?:part|chapter|section|subsection|subsubsection)\*?\{")
BIB_KEY_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)", re.IGNORECASE)
BIBITEM_RE = re.compile(r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}")
TODO_RE = re.compile(r"\b(?:TODO|FIXME|TBD|XXX)\b|\\todo\b", re.IGNORECASE)
PLACEHOLDER_RE = re.compile(
    r"placeholder|pending regeneration|pending build|to be filled|citation needed",
    re.IGNORECASE,
)
RISK_RE = re.compile(
    r"state[- ]of[- ]the[- ]art|\bnovel\b|\bfirst\b|caus(?:al|ally|ality)|"
    r"outperform|\boptimal\b|statistically significant|\bprove[sd]?\b|"
    r"\bguarantee[sd]?\b|\bsuperior(?:ity)?\b|\bdominant\b|\b17\s*[x×]\b|"
    r"G\s*[={}]\s*32|closed[- ]loop|PPO|SAO",
    re.IGNORECASE,
)


def active_lines(text: str) -> list[tuple[int, str]]:
    """Return source lines with non-escaped TeX comments removed."""
    rows: list[tuple[int, str]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        rows.append((number, re.sub(r"(?<!\\)%.*$", "", raw)))
    return rows


def bib_keys() -> set[str]:
    keys: set[str] = set()
    for base in (ROOT / "platform_hybrid" / "paper", ROOT / "zvf-program"):
        for path in base.rglob("*.bib"):
            keys.update(BIB_KEY_RE.findall(path.read_text(errors="replace")))
        for path in base.rglob("*.tex"):
            keys.update(BIBITEM_RE.findall(path.read_text(errors="replace")))
    return keys


def write_flags(rows: list[dict[str, object]], snippets: dict[str, list[str]]) -> None:
    lines = [
        "# Self-review claim and submission flags",
        "",
        "Generated deterministically by `self_review_corpus.py`. Every included",
        "source file appears in `FILE_REVIEW.tsv`; this document expands only files",
        "with active claim-risk, TODO, placeholder, or unresolved-citation flags.",
        "A flag is a review location, not automatically a defect.",
        "",
    ]
    for row in rows:
        path = str(row["path"])
        if path not in snippets:
            continue
        lines.extend(
            [
                f"## `{path}`",
                "",
                f"Consumers: {row['consumers']}",
                "",
                *[f"- {snippet}" for snippet in snippets[path]],
                "",
            ]
        )
    (OUT / "SELF_REVIEW_FLAGS.md").write_text("\n".join(lines))


def main() -> None:
    include_map = json.loads(INCLUDE_MAP.read_text())
    consumers: dict[str, list[str]] = defaultdict(list)
    roots = {entry["root"] for entry in include_map.values()}
    missing_by_file: Counter[str] = Counter()
    for paper, entry in include_map.items():
        for path in entry["files"]:
            consumers[path].append(paper)
        for missing in entry["missing"]:
            source = missing.split(" -> ", 1)[0]
            missing_by_file[source] += 1

    known_bib_keys = bib_keys()
    rows: list[dict[str, object]] = []
    snippets: dict[str, list[str]] = {}
    labels_by_paper: dict[str, Counter[str]] = {}

    for paper, entry in include_map.items():
        counts: Counter[str] = Counter()
        for rel in entry["files"]:
            source = (ROOT / rel).read_text(errors="replace")
            active_source = "\n".join(line for _, line in active_lines(source))
            counts.update(LABEL_RE.findall(active_source))
        labels_by_paper[paper] = counts

    for rel in sorted(consumers):
        path = ROOT / rel
        text = path.read_text(errors="replace")
        active = active_lines(text)
        active_text = "\n".join(line for _, line in active)
        citations = {
            key.strip()
            for group in CITE_RE.findall(active_text)
            for key in group.split(",")
            if key.strip()
        }
        unresolved = sorted(
            key for key in citations - known_bib_keys if not key.startswith("#")
        )
        todo_lines = [number for number, line in active if TODO_RE.search(line)]
        placeholder_lines = [number for number, line in active if PLACEHOLDER_RE.search(line)]
        risk_lines = [number for number, line in active if RISK_RE.search(line)]

        flagged: list[str] = []
        for number, line in active:
            kinds: list[str] = []
            if TODO_RE.search(line):
                kinds.append("TODO")
            if PLACEHOLDER_RE.search(line):
                kinds.append("placeholder")
            if RISK_RE.search(line):
                kinds.append("claim")
            if kinds:
                compact = re.sub(r"\s+", " ", line).strip()
                flagged.append(f"L{number} [{'/'.join(kinds)}] {compact[:280]}")
        if unresolved:
            flagged.append("[citations] unresolved keys: " + ", ".join(unresolved))
        if flagged:
            snippets[rel] = flagged

        kind = "root" if rel in roots else "shared" if len(consumers[rel]) > 1 else "paper-specific"
        rows.append(
            {
                "path": rel,
                "kind": kind,
                "consumers": ",".join(sorted(consumers[rel])),
                "consumer_count": len(consumers[rel]),
                "lines": len(text.splitlines()),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sections": len(SECTION_RE.findall(active_text)),
                "labels": len(LABEL_RE.findall(active_text)),
                "citation_keys": len(citations),
                "unresolved_citations": ",".join(unresolved),
                "todo_lines": ",".join(map(str, todo_lines)),
                "placeholder_lines": ",".join(map(str, placeholder_lines)),
                "claim_risk_lines": ",".join(map(str, risk_lines)),
                "missing_input_hooks": missing_by_file[rel],
                "mechanical_status": "attention" if unresolved or todo_lines or placeholder_lines or missing_by_file[rel] else "screened",
            }
        )

    with (OUT / "FILE_REVIEW.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    write_flags(rows, snippets)

    duplicate_labels = {
        paper: sorted(label for label, count in counts.items() if count > 1)
        for paper, counts in labels_by_paper.items()
    }
    rendered_fallbacks: dict[str, int] = {}
    rendered_markers = (
        "figure placeholder",
        "pending regeneration",
        "pending build",
        "missing figure artifact",
    )
    for paper, _, _, pdf_rel in PAPERS:
        result = subprocess.run(
            ["pdftotext", str(ROOT / pdf_rel), "-"],
            check=True,
            text=True,
            capture_output=True,
        )
        lower = result.stdout.lower()
        rendered_fallbacks[paper] = sum(lower.count(marker) for marker in rendered_markers)
    summary = {
        "manuscripts": len(include_map),
        "closure_mentions": sum(len(entry["files"]) for entry in include_map.values()),
        "unique_source_files": len(rows),
        "source_lines": sum(int(row["lines"]) for row in rows),
        "source_bytes": sum(int(row["bytes"]) for row in rows),
        "files_with_unresolved_citations": sum(bool(row["unresolved_citations"]) for row in rows),
        "files_with_todos": sum(bool(row["todo_lines"]) for row in rows),
        "files_with_placeholders": sum(bool(row["placeholder_lines"]) for row in rows),
        "active_rendered_figure_fallbacks": sum(rendered_fallbacks.values()),
        "rendered_figure_fallbacks_by_manuscript": rendered_fallbacks,
        "files_with_claim_hotspots": sum(bool(row["claim_risk_lines"]) for row in rows),
        "missing_input_hooks": sum(int(row["missing_input_hooks"]) for row in rows),
        "duplicate_labels_by_manuscript": duplicate_labels,
    }
    (OUT / "self_review_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
