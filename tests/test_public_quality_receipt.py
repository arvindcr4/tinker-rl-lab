"""Receipt regressions use synthetic PDFs, never modify frozen review evidence."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tools import check_public_quality_receipt as quality


class QualityReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pdf = self.root / "reviewed.pdf"
        self.pdf.write_bytes(b"%PDF-synthetic fixture")
        self.receipt = {
            "schema": "public-document-quality-v1",
            "pdf_filename": self.pdf.name,
            "sha256": hashlib.sha256(self.pdf.read_bytes()).hexdigest(),
            "bytes": self.pdf.stat().st_size,
            "pages": 2,
            "rendered_review_pages": [1, 2],
            "visual_review": "pass: synthetic fixture review",
            "independent_review": "pass: synthetic fixture independent review",
            "all_page_text_bounds_failures": [],
            "replacement_character_pages": [],
            "embedded_file_count": 0,
            "privacy_pattern_failure_count": 0,
            "scientific_case_ledger_unchanged": True,
        }
        self.save()

    def save(self):
        (self.root / "receipt.json").write_text(json.dumps(self.receipt), encoding="utf-8")

    def check(self):
        return quality.verify_receipt(self.root, "receipt.json", "reviewed.pdf")

    def test_valid_binding(self):
        self.assertEqual(self.check()["recorded_pages"], 2)

    def test_changed_equal_length_bytes(self):
        self.pdf.write_bytes(b"%PDF-SYNTHETIC fixture")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.check()

    def test_changed_size(self):
        self.pdf.write_bytes(b"%PDF-longer synthetic fixture")
        with self.assertRaisesRegex(ValueError, "byte count mismatch"):
            self.check()

    def test_receipt_cannot_select_alternate_pdf(self):
        self.receipt["pdf_filename"] = "old.pdf"
        (self.root / "old.pdf").write_bytes(self.pdf.read_bytes())
        self.save()
        with self.assertRaisesRegex(ValueError, "filename mismatch"):
            self.check()

    def test_missing_and_unsafe_paths(self):
        for path in ("missing.json", "../receipt.json", "/receipt.json"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                quality.verify_receipt(self.root, path, "reviewed.pdf")
        self.pdf.unlink()
        with self.assertRaisesRegex(ValueError, "missing publication file"):
            self.check()

    def test_symlink_pdf_rejected(self):
        original = self.pdf.read_bytes()
        self.pdf.unlink()
        (self.root / "other.pdf").write_bytes(original)
        self.pdf.symlink_to(self.root / "other.pdf")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.check()

    def test_invalid_fields(self):
        fields = {
            "schema": [None, "unknown"],
            "bytes": [True, 0, "21"],
            "sha256": [None, "bad"],
            "pages": [True, 0, 2.5],
            "rendered_review_pages": [None, [], [True], [0], [3], [1, 1]],
            "visual_review": [None, "pending", "pass:", "fail: bad"],
            "independent_review": [None, "unreviewed", "pass:   "],
            "status": [None, "draft", "stale", "pending"],
            "all_page_text_bounds_failures": [None, [1]],
            "replacement_character_pages": [None, [2]],
            "embedded_file_count": [False, 1],
            "privacy_pattern_failure_count": [False, 1],
            "scientific_case_ledger_unchanged": [False, 1, None],
        }
        original = dict(self.receipt)
        for field, values in fields.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    self.receipt = {**original, field: value}
                    self.save()
                    with self.assertRaises(ValueError):
                        self.check()

    def test_missing_required_fields(self):
        original = dict(self.receipt)
        for field in original:
            with self.subTest(field=field):
                self.receipt = dict(original)
                del self.receipt[field]
                self.save()
                with self.assertRaises(ValueError):
                    self.check()

    def test_malformed_receipts(self):
        for text in ("[]", '{"schema": 1, "schema": 2}', '{"bytes": NaN}', "{"):
            with self.subTest(text=text):
                (self.root / "receipt.json").write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.check()

    def test_non_pdf_rejected_even_with_matching_digest(self):
        self.pdf.write_bytes(b"not a PDF")
        self.receipt.update(bytes=9, sha256=hashlib.sha256(self.pdf.read_bytes()).hexdigest())
        self.save()
        with self.assertRaisesRegex(ValueError, "PDF header"):
            self.check()
        self.pdf.rename(self.root / "reviewed.txt")
        with self.assertRaisesRegex(ValueError, "must be a PDF"):
            quality.verify_receipt(self.root, "receipt.json", "reviewed.txt")

    def test_cli_reports_success_and_failure(self):
        args = ["--root", str(self.root), "--receipt", "receipt.json", "--pdf", "reviewed.pdf"]
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(quality.main(args), 0)
            self.pdf.write_bytes(b"changed")
            self.assertEqual(quality.main(args), 1)
        self.assertIn("PASS:", output.getvalue())
        self.assertIn("FAIL:", output.getvalue())
        self.assertIn("no independent visual", output.getvalue())
