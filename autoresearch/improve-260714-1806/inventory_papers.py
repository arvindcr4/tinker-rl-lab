#!/usr/bin/env python3
"""Deterministic inventory and overlap audit for the 18-paper program."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

from audit_papers import OUT, PAPERS, ROOT, extract_pdf, pdf_pages, sha256


INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
WORD_RE = re.compile(r"[a-z][a-z0-9-]+")


def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def resolve_input(parent: Path, root_dir: Path, raw: str) -> Path | None:
    candidates = [parent / raw, root_dir / raw]
    if not raw.endswith(".tex"):
        candidates.extend([parent / f"{raw}.tex", root_dir / f"{raw}.tex"])
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def include_closure(root_tex: Path) -> tuple[list[Path], list[str]]:
    root_dir = root_tex.parent
    visited: set[Path] = set()
    missing: list[str] = []

    def visit(path: Path) -> None:
        path = path.resolve()
        if path in visited:
            return
        visited.add(path)
        text = strip_comments(path.read_text(errors="replace"))
        for raw in INPUT_RE.findall(text):
            child = resolve_input(path.parent, root_dir, raw.strip())
            if child is None:
                missing.append(f"{path.relative_to(ROOT)} -> {raw}")
            else:
                visit(child)

    visit(root_tex)
    return sorted(visited), sorted(set(missing))


def expanded_source(root_tex: Path) -> tuple[str, str | None]:
    result = subprocess.run(
        ["latexpand", root_tex.name],
        cwd=root_tex.parent,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return root_tex.read_text(errors="replace"), result.stderr.strip()
    return result.stdout, result.stderr.strip() or None


def shingles(text: str, size: int = 7) -> set[tuple[str, ...]]:
    before_refs = re.split(r"\n\s*References\s*\n", text, maxsplit=1)[0]
    words = WORD_RE.findall(before_refs.lower())
    return {tuple(words[i : i + size]) for i in range(max(0, len(words) - size + 1))}


def jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def key_pages(text: str) -> list[str]:
    selected: list[str] = []
    for page in re.split(r"(?=\n\[PAGE \d+\]\n)", text):
        lower = page.lower()
        if any(token in lower for token in ("abstract", "limitations", "conclusion", "claim status")):
            selected.append(page.strip())
    return selected


def main() -> None:
    (OUT / "source").mkdir(exist_ok=True)
    (OUT / "text").mkdir(exist_ok=True)
    inventory: list[dict[str, object]] = []
    include_map: dict[str, object] = {}
    texts: dict[str, str] = {}
    packets: list[str] = []

    for index, (identifier, label, tex_rel, pdf_rel) in enumerate(PAPERS, start=1):
        tex_path, pdf_path = ROOT / tex_rel, ROOT / pdf_rel
        text = extract_pdf(pdf_path)
        texts[identifier] = text
        (OUT / "text" / f"{identifier}.txt").write_text(text)

        closure, missing = include_closure(tex_path)
        expanded, expand_warning = expanded_source(tex_path)
        (OUT / "source" / f"{identifier}.tex").write_text(expanded)
        latest_source_mtime = max(path.stat().st_mtime for path in closure)
        row = {
            "id": identifier,
            "label": label,
            "tex": tex_rel,
            "pdf": pdf_rel,
            "pages": pdf_pages(pdf_path),
            "words": len(text.split()),
            "characters": len(text),
            "source_files": len(closure),
            "missing_inputs": len(missing),
            "pdf_current": pdf_path.stat().st_mtime >= latest_source_mtime,
            "pdf_sha256": sha256(pdf_path),
            "tex_sha256": sha256(tex_path),
        }
        inventory.append(row)
        include_map[identifier] = {
            "root": tex_rel,
            "files": [str(path.relative_to(ROOT)) for path in closure],
            "missing": missing,
            "latexpand_warning": expand_warning,
        }
        packets.extend(
            [
                f"\n# {identifier}: {label}\n",
                f"Root: `{tex_rel}`  Pages: {row['pages']}  Words: {row['words']}\n",
                *key_pages(text),
            ]
        )
        print(
            f"[{index:02d}/{len(PAPERS)}] {identifier}: {row['pages']} pages, "
            f"{row['source_files']} source files, current={row['pdf_current']}",
            flush=True,
        )

    with (OUT / "inventory.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(inventory)
    (OUT / "include_map.json").write_text(json.dumps(include_map, indent=2) + "\n")
    (OUT / "reading_packet.md").write_text("\n\n".join(packets) + "\n")

    shingle_map = {identifier: shingles(text) for identifier, text in texts.items()}
    pairs: list[dict[str, object]] = []
    identifiers = list(shingle_map)
    for left_index, left in enumerate(identifiers):
        for right in identifiers[left_index + 1 :]:
            pairs.append(
                {"left": left, "right": right, "jaccard_7gram": jaccard(shingle_map[left], shingle_map[right])}
            )
    pairs.sort(key=lambda row: float(row["jaccard_7gram"]), reverse=True)
    with (OUT / "similarity.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(pairs)

    corpus_digest = hashlib.sha256()
    for row in inventory:
        corpus_digest.update(str(row["pdf_sha256"]).encode())
    (OUT / "corpus.sha256").write_text(corpus_digest.hexdigest() + "\n")


if __name__ == "__main__":
    main()
