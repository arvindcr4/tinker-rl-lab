"""Public package contents, reproducibility and fail-closed validation."""

import json
from pathlib import Path
import zipfile

import unittest
import tempfile
import io
from contextlib import redirect_stdout

from tools import build_public_review_package as package


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def entries(root, names, relative_to=None):
    return [
        {
            "path": str(Path(name).relative_to(relative_to)) if relative_to else name,
            "bytes": (root / name).stat().st_size,
            "sha256": package.digest((root / name).read_bytes()),
        }
        for name in sorted(names)
    ]


def make_release(tmp_path):
    root = tmp_path / "repo"
    for name in package.REQUIRED:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("public source\n")
    pdf = root / package.PDF
    pdf.write_bytes(b"%PDF-1.7\nfixture")
    write_json(
        root / package.RECEIPT,
        {
            "schema": "public-document-quality-v1",
            "pdf_filename": pdf.name,
            "bytes": pdf.stat().st_size,
            "sha256": package.digest(pdf.read_bytes()),
            "pages": 1,
            "rendered_review_pages": [1],
            "visual_review": "pass: fixture only",
            "independent_review": "pass: fixture only",
            "all_page_text_bounds_failures": [],
            "replacement_character_pages": [],
            "embedded_file_count": 0,
            "privacy_pattern_failure_count": 0,
            "scientific_case_ledger_unchanged": True,
        },
    )
    write_json(
        root / package.AVAILABILITY,
        {
            "schema": "public-source-availability-v1",
            "scope": "test",
            "title": "test",
            "date": "2026-10-04",
            "withheld_categories": ["private exports"],
            "residual_limits": ["not training reproduction"],
            "included_files": entries(
                root, package.REQUIRED - {package.AVAILABILITY, package.RECEIPT}, package.THESIS
            ),
        },
    )
    write_json(
        root / package.PUBLICATION,
        {
            "schema": "public-scientific-release-v1",
            "documents": {"thesis_pages": 1},
            "files": entries(root, package.REQUIRED),
        },
    )
    return root


class TestPublicReviewPackage(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)
        self.release = make_release(self.tmp_path)
        self.stdout = io.StringIO()
        redirect = redirect_stdout(self.stdout)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def test_build_is_deterministic_and_verifies(self):
        first, second = self.tmp_path / "one.zip", self.tmp_path / "two.zip"
        result = package.build_package(self.release, first)
        package.build_package(self.release, second)
        assert first.read_bytes() == second.read_bytes()
        assert result["status"] == "PACKAGED"
        assert package.verify_package(first)["sha256"] == result["sha256"]
        with zipfile.ZipFile(first) as archive:
            assert set(archive.namelist()) == package.REQUIRED | {
                package.PUBLICATION,
                package.PACKAGE,
            }
        with self.assertRaisesRegex(ValueError, "already exists"):
            package.build_package(self.release, first)

    def test_stale_receipt_and_missing_inputs_leave_no_archive(self):
        output = self.tmp_path / "bad.zip"
        (self.release / package.PDF).write_bytes(b"%PDF-changed")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            package.build_package(self.release, output)
        assert not output.exists()

    def test_unlisted_file_is_not_silently_packaged(self):
        (self.release / package.THESIS / "operational.txt").write_text("not reviewed")
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            package.build_package(self.release, self.tmp_path / "bad.zip")

    def test_symlink_rejected(self):
        target = self.release / package.THESIS / "ch01_introduction.md"
        original = target.read_bytes()
        target.unlink()
        elsewhere = self.tmp_path / "source"
        elsewhere.write_bytes(original)
        target.symlink_to(elsewhere)
        with self.assertRaisesRegex(ValueError, "symlink"):
            package.build_package(self.release, self.tmp_path / "bad.zip")

    def test_private_or_noncanonical_paths_rejected(self):
        for name in [
            "../private",
            "/etc/passwd",
            "outputs/export.json",
            f"{package.THESIS}/../private",
            f"{package.THESIS}/.env",
        ]:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "package|scratch|hidden"):
                    package.public_name(name)

    def test_tampered_and_extra_archive_members_rejected(self):
        original = self.tmp_path / "one.zip"
        package.build_package(self.release, original)
        with zipfile.ZipFile(original) as archive:
            contents = {name: archive.read(name) for name in archive.namelist()}
        contents[package.PDF] = b"%PDF-tampered"
        changed = self.tmp_path / "changed.zip"
        with zipfile.ZipFile(changed, "w") as archive:
            for name, data in contents.items():
                archive.writestr(name, data)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            package.verify_package(changed)
        with zipfile.ZipFile(original, "a") as archive:
            archive.writestr("private/account.json", "{}")
        with self.assertRaisesRegex(ValueError, "outside public"):
            package.verify_package(original)

    def test_cli_success_and_error(self):
        path = self.tmp_path / "public.zip"
        assert package.main(["--root", str(self.release), "--output", str(path)]) == 0
        assert package.main(["--verify", str(path)]) == 0
        assert package.main(["--root", str(self.release), "--output", str(path)]) == 1
        assert '"status": "FAIL"' in self.stdout.getvalue()
