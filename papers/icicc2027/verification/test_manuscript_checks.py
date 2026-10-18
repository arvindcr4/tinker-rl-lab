"""Offline regression tests for editorial checks; no benchmark execution."""

# ruff: noqa: PT009, PT027 -- Keep standalone source-package tests stdlib-only.

from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_manuscript", ROOT / "verify_manuscript.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class EditorialChecks(unittest.TestCase):
    def test_current_source_arithmetic(self):
        self.assertGreater(
            VERIFY.check_printed_arithmetic((ROOT / "paper_camera.tex").read_text()), 0
        )

    def test_changed_printed_percentage_fails(self):
        with self.assertRaises(AssertionError):
            VERIFY.check_printed_arithmetic("450/(1,967-1,259) = 450/708 = 63.57\\%")

    def test_changed_conditional_denominator_fails(self):
        with self.assertRaises(AssertionError):
            VERIFY.check_printed_arithmetic("450/(1,967-1,258) = 450/708 = 63.56\\%")

    def test_missing_arithmetic_fails(self):
        with self.assertRaises(AssertionError):
            VERIFY.check_printed_arithmetic("No numbers")

    def test_contrasts_come_from_inventory(self):
        inventory = json.loads((ROOT / "verification/paired-outcomes-inventory.json").read_text())
        self.assertEqual(VERIFY.count_primary_contrasts(inventory), 17)
        next(row for row in inventory if row["summary"]["lane"] == "E9")["summary"].pop(
            "above_median"
        )
        self.assertEqual(VERIFY.count_primary_contrasts(inventory), 16)

    def test_rate_and_count_signflip_differ(self):
        rate = [
            Fraction(-1, 24),
            Fraction(1, 33),
            Fraction(0),
            Fraction(-4, 25),
            Fraction(0),
            Fraction(-1, 24),
        ]
        count = list(map(Fraction, [-1, 1, 0, -4, 0, -1]))
        self.assertEqual(VERIFY.exact_sign_flip(rate), Fraction(1, 4))
        self.assertEqual(VERIFY.exact_sign_flip(count), Fraction(1, 2))

    def test_empty_pdf_text_fails(self):
        with self.assertRaises(AssertionError):
            VERIFY.check_rendered_text(r"\title{Example manuscript}", "")

    def test_wrong_title_fails(self):
        with self.assertRaises(AssertionError):
            VERIFY.check_rendered_text(r"\title{Example manuscript}", "Different paper " * 1000)

    def test_missing_section_fails(self):
        with self.assertRaises(AssertionError):
            VERIFY.check_rendered_text(
                r"\title{Example manuscript}", "Example manuscript Introduction " * 1000
            )

    def test_substantive_text_passes(self):
        VERIFY.check_rendered_text(
            r"\title{Example manuscript}",
            "Example manuscript Introduction Campaign Design References " * 1000,
        )


if __name__ == "__main__":
    unittest.main()
