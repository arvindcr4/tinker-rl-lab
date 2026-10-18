"""Authored papers must not have broken citations (2026-10-18 citation check).

Runs the structural citation audit once and asserts every authored-tier paper
resolves all of its citations, bibliographies, and ``\\input`` files. Frozen
copies (snapshots, exports, archive, scratch) are out of scope by design.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "reports/citation_check_2026-10-18/citation_audit.py"


@pytest.fixture(scope="module")
def audit(tmp_path_factory):
    out = tmp_path_factory.mktemp("citation") / "citation_audit.json"
    argv = sys.argv
    sys.argv = ["citation_audit.py", str(REPO), str(out)]
    try:
        spec = importlib.util.spec_from_file_location("citation_audit", AUDIT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
    finally:
        sys.argv = argv
    return json.loads(out.read_text())


def _unguarded(paper):
    return [m for m in paper["missing_inputs"] if not m.get("guarded")]


def test_authored_papers_have_no_broken_citations(audit):
    broken = {}
    for paper in audit["papers"]:
        if paper["tier"] != "authored":
            continue
        problems = {
            "orphan_intext": [o["key"] for o in paper["orphan_intext"]],
            "missing_bibs": [m["declared"] for m in paper["missing_bibs"]],
            "dup_keys": [d["key"] for d in paper["dup_keys"]],
            "orphan_bibitems": paper["orphan_bibitems"],
            "duplicate_works": [d["keys"] for d in paper["dup_titles"] + paper["dup_dois"]],
            "missing_inputs": [m["input"] for m in _unguarded(paper)],
            "no_bibliography": paper["no_bibliography"],
        }
        problems = {k: v for k, v in problems.items() if v}
        if problems:
            broken[paper["root"]] = problems
    assert not broken, f"broken authored papers: {json.dumps(broken, indent=1)}"


@pytest.mark.parametrize(
    ("link", "target"),
    [
        ("paper/references.bib", "zvf-program/flagship/paper/flagship.bib"),
        (
            "papers/spectral_entropy_grpo/references.bib",
            "zvf-program/flagship/paper/flagship.bib",
        ),
        (
            "papers/spectral_entropy_grpo/spectral_entropy_paper.tex",
            "zvf-program/flagship/paper/spectral_entropy_paper.tex",
        ),
    ],
)
def test_paper_alias_is_portable_and_resolves(link, target):
    path = REPO / link
    assert path.is_symlink(), f"{link} must alias its canonical source"
    assert not path.readlink().is_absolute(), f"{link} must be checkout-relative"
    assert path.is_file(), f"{link} is missing or a dangling symlink"
    assert path.resolve() == REPO / target, f"{link} points to the wrong canonical source"
