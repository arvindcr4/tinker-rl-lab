#!/usr/bin/env python3
"""Structural publication-worthiness checks for the canonical 18-paper roster.

Drives real on-disk manuscript paths (not re-implemented scores). Used by
unit tests and by the goal verification harness.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Repo root: platform_hybrid/paper/scripts/ -> ../../../
REPO_ROOT = Path(__file__).resolve().parents[3]

ROSTER: dict[str, str] = {
    "P01": "platform_hybrid/paper/paper_P1_scaling.tex",
    "P02": "platform_hybrid/paper/paper_P2_zvf.tex",
    "P03": "platform_hybrid/paper/paper_P3_group_size.tex",
    "P04": "platform_hybrid/paper/paper_P4_length_bias.tex",
    "P05": "platform_hybrid/paper/paper_P5_minreport.tex",
    "P06": "platform_hybrid/paper/paper_P6_registry.tex",
    "P07": "platform_hybrid/paper/paper_P7_zvf_controller.tex",
    "P08": "platform_hybrid/paper/paper_P8_fraud.tex",
    "R01": "platform_hybrid/paper/acm_main.tex",
    "R02": "platform_hybrid/paper/neurips_2026_variants/main_zvf.tex",
    "R03": "platform_hybrid/paper/neurips_2026_variants/main_workshop.tex",
    "R04": "platform_hybrid/paper/neurips_2026_variants/main_dnb.tex",
    "R05": "zvf-program/theory/zvf_theory.tex",
    "R06": "zvf-program/position/min_report_rl.tex",
    "R07": "zvf-program/registry/grpo_registry.tex",
    "R08": "zvf-program/audit/reproducibility_audit.tex",
    "U01": "platform_hybrid/paper/main.tex",
    "N01": "platform_hybrid/paper/unified_signal_starvation/main.tex",
}

# Pure companions / parked / non-venue — absorbed into parents; not independent tops
OUT_OF_SET: dict[str, str] = {
    "R01": "ABSORBED → R04 (ACM regenerate of tiered artifact)",
    "R02": "ABSORBED → P02 (short ZVF vehicle; same evidence)",
    "R06": "ABSORBED → P05 (condensed position; retires at submit)",
    "R07": "ABSORBED → P06 (condensed registry; retires at submit)",
    "U01": "ABSORBED → thesis + R04 (evidence bank; not a venue paper)",
    "P08": "ABSORBED → thesis (parked fraud side study; appendix only)",
}

# Absorbed ID → parent vehicle ID(s)
ABSORPTION: dict[str, str] = {
    "R01": "R04",
    "R02": "P02",
    "R06": "P05",
    "R07": "P06",
    "U01": "thesis+R04",
    "P08": "thesis",
}

# On-disk absorption markers required in absorbed satellite sources
ABSORPTION_MARKERS: dict[str, list[tuple[str, str]]] = {
    "R01": [
        (
            "platform_hybrid/paper/acm_main.tex",
            r"Absorption \(roster R01\)",
        ),
    ],
    "R02": [
        (
            "platform_hybrid/paper/neurips_2026_variants/main_zvf.tex",
            r"Absorption \(roster R02\)",
        ),
    ],
    "R06": [
        (
            "zvf-program/position/min_report_rl.tex",
            r"Absorption \(roster R06\)",
        ),
    ],
    "R07": [
        (
            "zvf-program/registry/grpo_registry.tex",
            r"Absorption \(roster R07\)",
        ),
    ],
    "U01": [
        (
            "platform_hybrid/paper/main.tex",
            r"Absorption \(roster U01\)",
        ),
    ],
    "P08": [
        (
            "platform_hybrid/paper/sections/p8_abstract.tex",
            r"Absorption \(roster P08\)",
        ),
    ],
}

# Parent papers must declare absorbed satellites
PARENT_ABSORPTION_MARKERS: dict[str, list[tuple[str, str]]] = {
    "P02": [
        (
            "platform_hybrid/paper/sections/p2_abstract.tex",
            r"Absorbed satellite",
        ),
    ],
    "P05": [
        (
            "platform_hybrid/paper/sections/p5_abstract.tex",
            r"Absorbed satellite",
        ),
    ],
    "P06": [
        (
            "platform_hybrid/paper/sections/p6_abstract.tex",
            r"Absorbed satellite",
        ),
    ],
    "R04": [
        (
            "platform_hybrid/paper/neurips_2026_variants/sections/abstract_dnb.tex",
            r"Absorbed satellites",
        ),
    ],
}

INDEPENDENT_SET: frozenset[str] = frozenset(ROSTER) - frozenset(OUT_OF_SET)

# Claim-boundary anchors required after goal improvements (paths relative to repo)
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

# Patterns that would indicate fabricated run IDs in score prose
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
    if len(ROSTER) != 18:
        errors.append(f"roster size {len(ROSTER)} != 18")
    if len(INDEPENDENT_SET) + len(OUT_OF_SET) != 18:
        errors.append("in+out != 18")
    if INDEPENDENT_SET & set(OUT_OF_SET):
        errors.append(f"overlap: {INDEPENDENT_SET & set(OUT_OF_SET)}")
    missing = set(ROSTER) - INDEPENDENT_SET - set(OUT_OF_SET)
    if missing:
        errors.append(f"unassigned IDs: {missing}")
    return errors


def check_primary_sources_exist(root: Path = REPO_ROOT) -> list[str]:
    errors: list[str] = []
    for id_, rel in ROSTER.items():
        path = root / rel
        if not path.is_file():
            errors.append(f"{id_}: missing primary source {rel}")
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
    """Require venue claim-boundary text for IDs lifted by claim narrowing."""
    errors: list[str] = []
    for id_ in CLAIM_BOUNDARY_MARKERS:
        if id_ not in INDEPENDENT_SET:
            errors.append(f"{id_}: claim marker listed but not in independent set")
    errors.extend(_check_marker_table(CLAIM_BOUNDARY_MARKERS, root, "claim-boundary"))
    return errors


def check_absorption_markers(root: Path = REPO_ROOT) -> list[str]:
    """Require absorption notes on satellites and parent vehicles."""
    errors: list[str] = []
    if set(ABSORPTION) != set(OUT_OF_SET):
        errors.append(
            f"ABSORPTION keys {set(ABSORPTION)} != OUT_OF_SET {set(OUT_OF_SET)}"
        )
    if set(ABSORPTION_MARKERS) != set(OUT_OF_SET):
        errors.append("ABSORPTION_MARKERS must cover every out-of-set ID")
    errors.extend(_check_marker_table(ABSORPTION_MARKERS, root, "absorption"))
    errors.extend(
        _check_marker_table(PARENT_ABSORPTION_MARKERS, root, "parent-absorption")
    )
    # PAPERS_README must document the map
    readme = root / "platform_hybrid/paper/PAPERS_README.md"
    if not readme.is_file():
        errors.append("missing PAPERS_README.md")
    else:
        text = readme.read_text(encoding="utf-8", errors="replace")
        if "Absorption map" not in text:
            errors.append("PAPERS_README.md missing Absorption map section")
        for child, parent in ABSORPTION.items():
            if f"**ABSORBED →" not in text and "ABSORBED" not in text:
                errors.append(f"PAPERS_README missing ABSORBED labels")
                break
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
    # Rows like: | P05 | in | 5 | ... | **publication-worthy now** | ...
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 8:
            continue
        id_cell = cells[0].replace("*", "").strip()
        if id_cell not in ROSTER:
            continue
        # Find tier cell containing "publication-worthy" / "conditionally" / "not worthy"
        tier = ""
        for c in cells:
            cl = c.replace("*", "").lower()
            if "publication-worthy now" in cl:
                tier = "publication-worthy now"
                break
            if "conditionally worthy" in cl:
                tier = "conditionally worthy"
                break
            if "not worthy" in cl:
                tier = "not worthy as standalone"
                break
        if tier:
            tiers[id_cell] = tier
    return tiers


def parse_rank_after_count(path: Path) -> int | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r"In-set rated publication-worthy now:\D*?(\d+)",
        text,
    )
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
    for id_ in ROSTER:
        if id_ not in tiers:
            errors.append(f"scores_after missing ID {id_}")
    for id_ in INDEPENDENT_SET:
        tier = tiers.get(id_)
        if tier != "publication-worthy now":
            errors.append(
                f"in-set {id_} tier is {tier!r}, expected publication-worthy now"
            )
    for id_ in OUT_OF_SET:
        tier = tiers.get(id_)
        if tier == "publication-worthy now":
            # Out-of-set may still be high quality but must not be required as independent tops;
            # allow "now" only if labeled companion elsewhere — we require not standalone independent.
            # Strict: out-of-set should be "not worthy as standalone"
            if tier != "not worthy as standalone":
                errors.append(
                    f"out-of-set {id_} should be not-standalone, got {tier!r}"
                )

    # fabricated run IDs
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

    # double-count language
    rank_text = rank_path.read_text(encoding="utf-8", errors="replace")
    if "Double-count" not in rank_text and "double-count" not in rank_text:
        errors.append("rank_after missing double-count section")
    return errors


def run_all_checks(
    root: Path = REPO_ROOT,
    scores_path: Path | None = None,
    rank_path: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    errors.extend(check_roster_partition())
    errors.extend(check_primary_sources_exist(root))
    errors.extend(check_claim_boundaries(root))
    errors.extend(check_absorption_markers(root))
    errors.extend(check_score_artifacts(scores_path, rank_path))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="path to paper_scores_after.md",
    )
    parser.add_argument(
        "--rank",
        type=Path,
        default=None,
        help="path to publication_worthiness_rank_after.md",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="repository root",
    )
    args = parser.parse_args(argv)
    errors = run_all_checks(args.root, args.scores, args.rank)
    if errors:
        print("FAIL")
        for e in errors:
            print(" -", e)
        return 1
    print("PASS")
    print(f"INDEPENDENT_SET_SIZE={len(INDEPENDENT_SET)}")
    print(f"OUT_OF_SET_SIZE={len(OUT_OF_SET)}")
    print("INDEPENDENT_SET=" + ",".join(sorted(INDEPENDENT_SET)))
    print(
        "ABSORPTION="
        + ",".join(f"{k}->{v}" for k, v in sorted(ABSORPTION.items()))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
