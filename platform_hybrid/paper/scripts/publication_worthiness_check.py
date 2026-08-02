#!/usr/bin/env python3
"""Structural publication-worthiness checks for the active manuscript roster.

Independent venue candidates only (12). Former companions R01/R02/R06/R07/U01/P08
are archived under platform_hybrid/paper/archive/absorbed/ — not live roots.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Repo root: platform_hybrid/paper/scripts/ -> ../../../
REPO_ROOT = Path(__file__).resolve().parents[3]

# Live independent venue-candidate roots only
ACTIVE_ROSTER: dict[str, str] = {
    "P01": "platform_hybrid/paper/paper_P1_scaling.tex",
    "P02": "platform_hybrid/paper/paper_P2_zvf.tex",
    "P03": "platform_hybrid/paper/paper_P3_group_size.tex",
    "P04": "platform_hybrid/paper/paper_P4_length_bias.tex",
    "P05": "platform_hybrid/paper/paper_P5_minreport.tex",
    "P06": "platform_hybrid/paper/paper_P6_registry.tex",
    "P07": "platform_hybrid/paper/paper_P7_zvf_controller.tex",
    "R03": "platform_hybrid/paper/neurips_2026_variants/main_workshop.tex",
    "R04": "platform_hybrid/paper/neurips_2026_variants/main_dnb.tex",
    "R05": "zvf-program/theory/zvf_theory.tex",
    "R08": "zvf-program/audit/reproducibility_audit.tex",
    "N01": "platform_hybrid/paper/unified_signal_starvation/main.tex",
}

# Former independent IDs → parent + archive directory (relative to repo)
ABSORBED_ARCHIVED: dict[str, tuple[str, str]] = {
    "R01": ("R04", "platform_hybrid/paper/archive/absorbed/R01_acm"),
    "R02": ("P02", "platform_hybrid/paper/archive/absorbed/R02_main_zvf"),
    "R06": ("P05", "platform_hybrid/paper/archive/absorbed/R06_min_report"),
    "R07": ("P06", "platform_hybrid/paper/archive/absorbed/R07_grpo_registry"),
    "U01": ("thesis+R04", "platform_hybrid/paper/archive/absorbed/U01_main_compendium"),
    "P08": ("thesis", "platform_hybrid/paper/archive/absorbed/P08_fraud"),
}

# Former live paths that must NOT exist (independent versions removed)
FORBIDDEN_LIVE_PATHS: tuple[str, ...] = (
    "platform_hybrid/paper/acm_main.tex",
    "platform_hybrid/paper/neurips_2026_variants/main_zvf.tex",
    "zvf-program/position/min_report_rl.tex",
    "zvf-program/registry/grpo_registry.tex",
    "platform_hybrid/paper/main.tex",
    "platform_hybrid/paper/paper_P8_fraud.tex",
)

# Back-compat aliases
ROSTER = ACTIVE_ROSTER
INDEPENDENT_SET: frozenset[str] = frozenset(ACTIVE_ROSTER)
OUT_OF_SET: dict[str, str] = {
    k: f"ABSORBED → {parent} (archived at {path})"
    for k, (parent, path) in ABSORBED_ARCHIVED.items()
}
ABSORPTION: dict[str, str] = {k: v[0] for k, v in ABSORBED_ARCHIVED.items()}

# Parent papers must declare retired absorbed satellites
PARENT_ABSORPTION_MARKERS: dict[str, list[tuple[str, str]]] = {
    "P02": [
        (
            "platform_hybrid/paper/sections/p2_abstract.tex",
            r"Absorbed satellite \(retired\)|archive/absorbed/R02",
        ),
    ],
    "P05": [
        (
            "platform_hybrid/paper/sections/p5_abstract.tex",
            r"Absorbed satellite \(retired\)|archive/absorbed/R06",
        ),
    ],
    "P06": [
        (
            "platform_hybrid/paper/sections/p6_abstract.tex",
            r"Absorbed satellite \(retired\)|archive/absorbed/R07",
        ),
    ],
    "R04": [
        (
            "platform_hybrid/paper/neurips_2026_variants/sections/abstract_dnb.tex",
            r"Absorbed satellites \(retired\)|archive/absorbed/R01",
        ),
    ],
}

CLAIM_BOUNDARY_MARKERS: dict[str, list[tuple[str, str]]] = {
    "P01": [
        (
            "platform_hybrid/paper/sections/p1_abstract.tex",
            r"Venue claim boundary \(workshop-short / limits audit\)",
        ),
        (
            "platform_hybrid/paper/sections/p1_conclusion.tex",
            r"Submission scope",
        ),
    ],
    "P06": [
        (
            "platform_hybrid/paper/sections/p6_abstract.tex",
            r"Venue claim boundary \(position-artifact / resource\)",
        ),
        (
            "platform_hybrid/paper/sections/p6_conclusion.tex",
            r"Vehicle packaging",
        ),
    ],
    "P07": [
        (
            "platform_hybrid/paper/sections/p7_abstract.tex",
            r"controller \\emph\{not\} promoted|controller.*not.*promoted",
        ),
        (
            "platform_hybrid/paper/sections/p7_limitations.tex",
            r"non-claim\} of this workshop-short submission",
        ),
    ],
    "R04": [
        (
            "platform_hybrid/paper/neurips_2026_variants/sections/abstract_dnb.tex",
            r"Venue claim boundary \(position-artifact / datasets-and-benchmarks\)",
        ),
        (
            "platform_hybrid/paper/neurips_2026_variants/main_dnb.tex",
            r"Scope relative to companion ZVF vehicles",
        ),
    ],
    "N01": [
        (
            "platform_hybrid/paper/unified_signal_starvation/main.tex",
            r"Venue claim boundary \(workshop-short methods / proposal\)",
        ),
        (
            "platform_hybrid/paper/unified_signal_starvation/README.md",
            r"workshop-short methods/proposal",
        ),
    ],
}

FABRICATED_RUN_PATTERNS = [
    r"\brun[_-]?id\s*[:=]\s*fake",
    r"\bfabricated\s+run\b",
    r"\bsynthetic[_-]?run[_-]?9999\b",
]


def independent_set() -> frozenset[str]:
    return INDEPENDENT_SET


def out_of_set() -> dict[str, str]:
    return dict(OUT_OF_SET)


def check_roster_partition() -> list[str]:
    errors: list[str] = []
    if len(ACTIVE_ROSTER) != 12:
        errors.append(f"active roster size {len(ACTIVE_ROSTER)} != 12")
    if len(ABSORBED_ARCHIVED) != 6:
        errors.append(f"absorbed count {len(ABSORBED_ARCHIVED)} != 6")
    if INDEPENDENT_SET & set(ABSORBED_ARCHIVED):
        errors.append(f"overlap: {INDEPENDENT_SET & set(ABSORBED_ARCHIVED)}")
    return errors


def check_primary_sources_exist(root: Path = REPO_ROOT) -> list[str]:
    errors: list[str] = []
    for id_, rel in ACTIVE_ROSTER.items():
        path = root / rel
        if not path.is_file():
            errors.append(f"{id_}: missing active primary source {rel}")
    return errors


def check_live_paths_removed(root: Path = REPO_ROOT) -> list[str]:
    """Independent versions of absorbed papers must not remain at live paths."""
    errors: list[str] = []
    for rel in FORBIDDEN_LIVE_PATHS:
        if (root / rel).exists():
            errors.append(f"absorbed independent root still live: {rel}")
    return errors


def check_archives_present(root: Path = REPO_ROOT) -> list[str]:
    errors: list[str] = []
    readme = root / "platform_hybrid/paper/archive/absorbed/README.md"
    if not readme.is_file():
        errors.append("missing archive/absorbed/README.md")
    for id_, (parent, rel) in ABSORBED_ARCHIVED.items():
        d = root / rel
        if not d.is_dir():
            errors.append(f"{id_}: missing archive dir {rel}")
            continue
        tex_files = list(d.rglob("*.tex"))
        if not tex_files:
            errors.append(f"{id_}: archive dir has no .tex: {rel}")
    return errors


def _check_marker_table(
    table: dict[str, list[tuple[str, str]]],
    root: Path,
    kind: str,
) -> list[str]:
    errors: list[str] = []
    for id_, markers in table.items():
        for rel, pattern in markers:
            path = root / rel
            if not path.is_file():
                errors.append(f"{id_}: missing {kind} file {rel}")
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if not re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
                errors.append(f"{id_}: {kind} pattern not found in {rel}: {pattern}")
    return errors


def check_claim_boundaries(root: Path = REPO_ROOT) -> list[str]:
    errors: list[str] = []
    for id_ in CLAIM_BOUNDARY_MARKERS:
        if id_ not in INDEPENDENT_SET:
            errors.append(f"{id_}: claim marker listed but not in independent set")
    errors.extend(_check_marker_table(CLAIM_BOUNDARY_MARKERS, root, "claim-boundary"))
    return errors


def check_absorption_markers(root: Path = REPO_ROOT) -> list[str]:
    errors: list[str] = []
    errors.extend(
        _check_marker_table(PARENT_ABSORPTION_MARKERS, root, "parent-absorption")
    )
    readme = root / "platform_hybrid/paper/PAPERS_README.md"
    if not readme.is_file():
        errors.append("missing PAPERS_README.md")
    else:
        text = readme.read_text(encoding="utf-8", errors="replace")
        if "Absorption map" not in text:
            errors.append("PAPERS_README.md missing Absorption map section")
        if "archive/absorbed" not in text:
            errors.append("PAPERS_README.md must point at archive/absorbed")
        for child in ABSORBED_ARCHIVED:
            if child not in text:
                errors.append(f"PAPERS_README missing absorbed ID {child}")
    abs_md = root / "platform_hybrid/paper/ABSORPTION.md"
    if not abs_md.is_file():
        errors.append("missing platform_hybrid/paper/ABSORPTION.md")
    return errors


def parse_scores_after(path: Path) -> dict[str, str]:
    """Map ID -> tier string from paper_scores_after.md table rows."""
    text = path.read_text(encoding="utf-8", errors="replace")
    tiers: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 8:
            continue
        id_cell = cells[0].replace("*", "").strip()
        if id_cell not in ACTIVE_ROSTER and id_cell not in ABSORBED_ARCHIVED:
            continue
        tier = ""
        for c in cells:
            cl = c.replace("*", "").lower()
            if "publication-worthy now" in cl:
                tier = "publication-worthy now"
                break
            if "conditionally worthy" in cl:
                tier = "conditionally worthy"
                break
            if "not worthy" in cl or "archived" in cl or "absorbed" in cl:
                tier = "not worthy as standalone"
                break
        if tier:
            tiers[id_cell] = tier
    return tiers


def parse_rank_after_count(path: Path) -> int | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"In-set rated publication-worthy now:\D*?(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(
        r"Independent set · \*\*publication-worthy now\*\*\s*\|\s*\*\*(\d+)\*\*",
        text,
    )
    if m:
        return int(m.group(1))
    m = re.search(
        r"count of independent-set members rated publication-worthy now[^\d]*(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return int(m.group(1))
    return None


def check_score_artifacts(
    scores_path: Path | None,
    rank_path: Path | None,
) -> list[str]:
    errors: list[str] = []
    if scores_path is None or rank_path is None:
        return errors
    if not scores_path.is_file():
        return [f"missing scores file: {scores_path}"]
    if not rank_path.is_file():
        return [f"missing rank file: {rank_path}"]

    tiers = parse_scores_after(scores_path)
    for id_ in ACTIVE_ROSTER:
        if id_ not in tiers:
            errors.append(f"scores_after missing active ID {id_}")
        elif tiers[id_] != "publication-worthy now":
            errors.append(
                f"active {id_} tier is {tiers.get(id_)!r}, expected publication-worthy now"
            )
    # Absorbed IDs, if present, must not be scored as independent "now"
    for id_ in ABSORBED_ARCHIVED:
        if tiers.get(id_) == "publication-worthy now":
            errors.append(f"archived {id_} must not be publication-worthy now")

    blob = scores_path.read_text(encoding="utf-8", errors="replace") + rank_path.read_text(
        encoding="utf-8", errors="replace"
    )
    for pat in FABRICATED_RUN_PATTERNS:
        if re.search(pat, blob, flags=re.IGNORECASE):
            errors.append(f"fabricated-run pattern matched: {pat}")

    count = parse_rank_after_count(rank_path)
    if count is None:
        errors.append("rank_after missing in-set 'now' integer count")
    elif count != len(INDEPENDENT_SET):
        errors.append(
            f"rank_after count {count} != independent-set size {len(INDEPENDENT_SET)}"
        )

    rank_text = rank_path.read_text(encoding="utf-8", errors="replace")
    if "Double-count" not in rank_text and "double-count" not in rank_text:
        if "absorb" not in rank_text.lower() and "archive" not in rank_text.lower():
            errors.append("rank_after missing double-count/absorption section")
    return errors


def run_all_checks(
    root: Path = REPO_ROOT,
    scores_path: Path | None = None,
    rank_path: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    errors.extend(check_roster_partition())
    errors.extend(check_primary_sources_exist(root))
    errors.extend(check_live_paths_removed(root))
    errors.extend(check_archives_present(root))
    errors.extend(check_claim_boundaries(root))
    errors.extend(check_absorption_markers(root))
    errors.extend(check_score_artifacts(scores_path, rank_path))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=None)
    parser.add_argument("--rank", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    errors = run_all_checks(args.root, args.scores, args.rank)
    if errors:
        print("FAIL")
        for e in errors:
            print(" -", e)
        return 1
    print("PASS")
    print(f"ACTIVE_ROSTER_SIZE={len(ACTIVE_ROSTER)}")
    print(f"ABSORBED_ARCHIVED_SIZE={len(ABSORBED_ARCHIVED)}")
    print("ACTIVE_ROSTER=" + ",".join(sorted(ACTIVE_ROSTER)))
    print(
        "ABSORPTION="
        + ",".join(f"{k}->{v[0]}@{v[1]}" for k, v in sorted(ABSORBED_ARCHIVED.items()))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
