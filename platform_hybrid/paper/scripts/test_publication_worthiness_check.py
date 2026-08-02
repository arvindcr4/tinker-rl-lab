#!/usr/bin/env python3
"""Unit tests for publication_worthiness_check — drives the real shipped module."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from publication_worthiness_check import (
    ABSORPTION,
    INDEPENDENT_SET,
    OUT_OF_SET,
    ROSTER,
    check_absorption_markers,
    check_claim_boundaries,
    check_primary_sources_exist,
    check_roster_partition,
    main,
    parse_rank_after_count,
    parse_scores_after,
    run_all_checks,
)


class TestPublicationWorthinessCheck(unittest.TestCase):
    def test_roster_partition_complete(self):
        self.assertEqual(len(ROSTER), 18)
        self.assertEqual(len(INDEPENDENT_SET) + len(OUT_OF_SET), 18)
        self.assertFalse(INDEPENDENT_SET & set(OUT_OF_SET))
        self.assertEqual(check_roster_partition(), [])

    def test_all_primary_sources_exist(self):
        self.assertEqual(check_primary_sources_exist(), [])

    def test_claim_boundaries_present_for_lifted_ids(self):
        """Real source files must contain venue claim-boundary text."""
        errors = check_claim_boundaries()
        self.assertEqual(errors, [], msg=errors)

    def test_lifted_ids_are_in_independent_set(self):
        for id_ in ("P01", "P06", "P07", "R04", "N01"):
            self.assertIn(id_, INDEPENDENT_SET)

    def test_out_of_set_includes_parked_and_companions(self):
        for id_ in ("P08", "U01", "R01", "R02", "R06", "R07"):
            self.assertIn(id_, OUT_OF_SET)

    def test_absorption_map_covers_all_out_of_set(self):
        self.assertEqual(set(ABSORPTION), set(OUT_OF_SET))
        self.assertEqual(ABSORPTION["R02"], "P02")
        self.assertEqual(ABSORPTION["R06"], "P05")
        self.assertEqual(ABSORPTION["R07"], "P06")
        self.assertEqual(ABSORPTION["R01"], "R04")
        self.assertIn("thesis", ABSORPTION["U01"])
        self.assertEqual(ABSORPTION["P08"], "thesis")

    def test_absorption_markers_present_in_sources(self):
        errors = check_absorption_markers()
        self.assertEqual(errors, [], msg=errors)

    def test_parse_scores_and_rank_roundtrip(self):
        scores = """# scores
| ID | Set | E | H | D | M | Total | Tier | Venue | cites |
|---|---|---|---|---|---|---|---|---|---|
"""
        for id_ in sorted(ROSTER):
            if id_ in INDEPENDENT_SET:
                tier = "**publication-worthy now**"
            else:
                tier = "not worthy as standalone"
            scores += f"| {id_} | x | 1 | 1 | 1 | 1 | 4 | {tier} | workshop-short | path |\n"

        rank = f"""# rank
**Independent set size:** {len(INDEPENDENT_SET)}
**In-set rated publication-worthy now:** **{len(INDEPENDENT_SET)}**

## Double-count risk
P05 vs R06
"""
        with tempfile.TemporaryDirectory() as td:
            sp = Path(td) / "paper_scores_after.md"
            rp = Path(td) / "publication_worthiness_rank_after.md"
            sp.write_text(scores)
            rp.write_text(rank)
            tiers = parse_scores_after(sp)
            self.assertEqual(len(tiers), 18)
            for id_ in INDEPENDENT_SET:
                self.assertEqual(tiers[id_], "publication-worthy now")
            self.assertEqual(parse_rank_after_count(rp), len(INDEPENDENT_SET))
            errors = run_all_checks(scores_path=sp, rank_path=rp)
            self.assertEqual(errors, [], msg=errors)

    def test_main_cli_sources_only_pass(self):
        self.assertEqual(main([]), 0)


if __name__ == "__main__":
    unittest.main()
