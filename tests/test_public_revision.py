"""Synthetic versioned exports exercise metadata binding without claiming review."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tools import check_public_revision as revision


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "revision"
        self.thesis = self.base / "thesis"
        self.thesis.mkdir(parents=True)
        (self.thesis / "final.pdf").write_bytes(b"%PDF-synthetic")
        (self.thesis / "chapter.md").write_text("Synthetic chapter")
        self.receipt = {
            "schema": "public-document-quality-v1",
            "pdf_filename": "final.pdf",
            **self.entry(self.thesis / "final.pdf", self.thesis),
            "pages": 1,
            "rendered_review_pages": [1],
            "visual_review": "pass: synthetic",
            "independent_review": "pass: synthetic",
            "all_page_text_bounds_failures": [],
            "replacement_character_pages": [],
            "embedded_file_count": 0,
            "privacy_pattern_failure_count": 0,
            "scientific_case_ledger_unchanged": True,
        }
        self.receipt.pop("path")
        self.write(self.thesis / "public_quality_receipt.json", self.receipt)
        self.availability = {
            "schema": "public-source-availability-v1",
            "title": "Synthetic",
            "date": "2026-10-04",
            "scope": "Synthetic fixture only",
            "withheld_categories": ["No execution payloads"],
            "residual_limits": ["Does not establish scientific validity"],
            "included_files": [
                self.entry(self.thesis / name, self.thesis) for name in ("chapter.md", "final.pdf")
            ],
        }
        self.save_availability()
        self.refresh_manifest()

    def entry(self, path, base):
        return {
            "path": path.relative_to(base).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def write(self, path, data):
        path.write_text(json.dumps(data), encoding="utf-8")

    def save_availability(self):
        self.write(self.thesis / "public_source_availability.json", self.availability)

    def refresh_manifest(self):
        self.manifest = {
            "schema": "public-scientific-release-v1",
            "documents": {"thesis_pages": 1},
            "files": [
                self.entry(path, self.root) for path in self.thesis.iterdir() if path.is_file()
            ],
        }
        self.write(self.base / "PUBLICATION_MANIFEST.json", self.manifest)

    def check(self):
        return revision.check_revision(self.root, "revision", "thesis/final.pdf")

    def test_valid_complete_revision_and_cli(self):
        self.assertEqual(self.check()["source_files_verified"], 2)
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(
                revision.main(
                    [
                        "--root",
                        str(self.root),
                        "--revision",
                        "revision",
                        "--pdf",
                        "thesis/final.pdf",
                    ]
                ),
                0,
            )
        self.assertIn('"status": "PASS"', output.getvalue())

    def test_unlisted_file_fails_but_build_scratch_is_excluded(self):
        scratch = self.thesis / "build"
        scratch.mkdir()
        (scratch / "log.txt").write_text("scratch")
        self.check()
        (self.thesis / "unlisted.md").write_text("not approved")
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            self.check()

    def test_manifest_cannot_export_other_revision(self):
        (self.root / "other.txt").write_text("outside")
        self.manifest["files"].append(self.entry(self.root / "other.txt", self.root))
        self.write(self.base / "PUBLICATION_MANIFEST.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            self.check()

    def test_stale_receipt_is_not_fixed_by_new_manifest(self):
        (self.thesis / "final.pdf").write_bytes(b"%PDF-synthetic updated")
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "stale quality receipt"):
            self.check()

    def test_conflicting_page_count_fails(self):
        self.manifest["documents"]["thesis_pages"] = 2
        self.write(self.base / "PUBLICATION_MANIFEST.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "page counts disagree"):
            self.check()

    def test_availability_malformed_stale_or_omitted_binding_fails(self):
        original = json.loads(json.dumps(self.availability))
        mutations = [
            {"schema": "unknown"},
            {"scope": ""},
            {"residual_limits": []},
            {"included_files": []},
            {"included_files": [None]},
            {"included_files": original["included_files"][:1]},
            {"included_files": original["included_files"] * 2},
        ]
        for field, value in [
            ("bytes", True),
            ("bytes", 999),
            ("sha256", "bad"),
            ("sha256", "0" * 64),
            ("path", "../escape"),
        ]:
            entries = json.loads(json.dumps(original["included_files"]))
            entries[0][field] = value
            mutations.append({"included_files": entries})
        for update in mutations:
            with self.subTest(update=update):
                self.availability = {**original, **update}
                self.save_availability()
                self.refresh_manifest()
                with self.assertRaises(ValueError):
                    self.check()

    def test_missing_receipt_and_invalid_revision_fail_cli(self):
        (self.thesis / "public_quality_receipt.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing publication metadata"):
            self.check()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                revision.main(
                    ["--root", str(self.root), "--revision", "../bad", "--pdf", "thesis/final.pdf"]
                ),
                1,
            )
