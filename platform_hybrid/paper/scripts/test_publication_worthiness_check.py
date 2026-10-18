#!/usr/bin/env python3
"""Unit tests for publication_worthiness_check — P1–P12 roster."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from publication_worthiness_check import (
    ABSORBED_ARCHIVED,
    ACTIVE_ROSTER,
    FORBIDDEN_LIVE_PATHS,
    INDEPENDENT_SET,
    VENUE_TIER,
    check_archives_present,
    check_claim_boundaries,
    check_live_paths_removed,
    check_primary_sources_exist,
    check_roster_partition,
    main,
    parse_rank_after_count,
    parse_scores_after,
    run_all_checks,
)


class TestPublicationWorthinessCheck(unittest.TestCase):
    def test_active_roster_is_p1_through_p12(self):
        assert len(ACTIVE_ROSTER) == 12
        assert set(ACTIVE_ROSTER) == {f"P{i}" for i in range(1, 13)}
        assert check_roster_partition() == []

    def test_all_active_primary_sources_exist(self):
        assert check_primary_sources_exist() == []

    def test_every_paper_has_venue_tier(self):
        for pid in ACTIVE_ROSTER:
            assert pid in VENUE_TIER
            assert VENUE_TIER[pid] in {
                "workshop-short",
                "workshop-artifact",
                "position-artifact",
                "main-track",
            }

    def test_absorbed_independent_roots_removed_from_live_paths(self):
        assert check_live_paths_removed() == []
        root = Path(__file__).resolve().parents[3]
        for rel in FORBIDDEN_LIVE_PATHS:
            assert not (root / rel).exists(), f"still live: {rel}"

    def test_archives_present_for_all_absorbed(self):
        assert len(ABSORBED_ARCHIVED) == 6
        assert check_archives_present() == []

    def test_claim_boundaries_present(self):
        errors = check_claim_boundaries()
        assert errors == [], errors

    def test_p8_is_workshop_not_fraud(self):
        assert "workshop" in ACTIVE_ROSTER["P8"]
        assert "fraud" not in ACTIVE_ROSTER["P8"]
        assert "P08_fraud" in ABSORBED_ARCHIVED

    def test_parse_scores_and_rank_roundtrip(self):
        scores = """# scores
| ID | Set | E | H | D | M | Total | Tier | Venue | cites |
|---|---|---|---|---|---|---|---|---|---|
"""
        for id_ in [f"P{i}" for i in range(1, 13)]:
            scores += (
                f"| {id_} | in | 1 | 1 | 1 | 1 | 4 | "
                f"**publication-worthy now** | {VENUE_TIER[id_]} | path |\n"
            )
        for id_ in sorted(ABSORBED_ARCHIVED):
            scores += (
                f"| {id_} | archived | 1 | 1 | 1 | 1 | 4 | "
                f"not worthy as standalone | — | archive |\n"
            )
        rank = f"""# rank
**Independent set size:** {len(INDEPENDENT_SET)}
**In-set rated publication-worthy now:** **{len(INDEPENDENT_SET)}**

## Double-count / absorption
Archived satellites must not be double-counted.
"""
        with tempfile.TemporaryDirectory() as td:
            sp = Path(td) / "paper_scores_after.md"
            rp = Path(td) / "publication_worthiness_rank_after.md"
            sp.write_text(scores)
            rp.write_text(rank)
            tiers = parse_scores_after(sp)
            for id_ in ACTIVE_ROSTER:
                assert tiers[id_] == "publication-worthy now"
            assert parse_rank_after_count(rp) == 12
            errors = run_all_checks(scores_path=sp, rank_path=rp)
            assert errors == [], errors

    def test_main_cli_sources_only_pass(self):
        assert main([]) == 0


if __name__ == "__main__":
    unittest.main()
