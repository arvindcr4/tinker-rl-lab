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
        self.assertEqual(len(ACTIVE_ROSTER), 12)
        self.assertEqual(set(ACTIVE_ROSTER), {f"P{i}" for i in range(1, 13)})
        self.assertEqual(check_roster_partition(), [])

    def test_all_active_primary_sources_exist(self):
        self.assertEqual(check_primary_sources_exist(), [])

    def test_every_paper_has_venue_tier(self):
        for pid in ACTIVE_ROSTER:
            self.assertIn(pid, VENUE_TIER)
            self.assertIn(
                VENUE_TIER[pid],
                {"workshop-short", "workshop-artifact", "position-artifact", "main-track"},
            )

    def test_absorbed_independent_roots_removed_from_live_paths(self):
        self.assertEqual(check_live_paths_removed(), [])
        root = Path(__file__).resolve().parents[3]
        for rel in FORBIDDEN_LIVE_PATHS:
            self.assertFalse((root / rel).exists(), msg=f"still live: {rel}")

    def test_archives_present_for_all_absorbed(self):
        self.assertEqual(len(ABSORBED_ARCHIVED), 6)
        self.assertEqual(check_archives_present(), [])

    def test_claim_boundaries_present(self):
        errors = check_claim_boundaries()
        self.assertEqual(errors, [], msg=errors)

    def test_p8_is_workshop_not_fraud(self):
        self.assertIn("workshop", ACTIVE_ROSTER["P8"])
        self.assertNotIn("fraud", ACTIVE_ROSTER["P8"])
        self.assertIn("P08_fraud", ABSORBED_ARCHIVED)

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
                self.assertEqual(tiers[id_], "publication-worthy now")
            self.assertEqual(parse_rank_after_count(rp), 12)
            errors = run_all_checks(scores_path=sp, rank_path=rp)
            self.assertEqual(errors, [], msg=errors)

    def test_main_cli_sources_only_pass(self):
        self.assertEqual(main([]), 0)


if __name__ == "__main__":
    unittest.main()
