"""Offline integrity regressions; corrupted files exist only in temporary fixtures."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools import check_public_release as release


ROOT = Path(__file__).resolve().parents[1]


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "data.txt").write_bytes(b"synthetic public content")
        self.manifest = {
            "schema": "public-scientific-release-v1",
            "files": [
                {
                    "path": "data.txt",
                    "bytes": len(b"synthetic public content"),
                    "sha256": hashlib.sha256(b"synthetic public content").hexdigest(),
                }
            ],
        }
        self.save()

    def save(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def check(self, expected_paths=None):
        return release.verify_manifest(self.root, "manifest.json", expected_paths=expected_paths)

    def test_valid_manifest_and_inventory(self):
        self.assertEqual(
            self.check({"data.txt"}),
            {"files_verified": 1, "bytes_verified": len(b"synthetic public content")},
        )

    def test_equal_length_content_change_fails_digest(self):
        (self.root / "data.txt").write_bytes(b"synthetic public CONTENT")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.check()

    def test_truncation_fails_byte_count(self):
        (self.root / "data.txt").write_bytes(b"short")
        with self.assertRaisesRegex(ValueError, "byte count mismatch"):
            self.check()

    def test_missing_file_fails(self):
        (self.root / "data.txt").unlink()
        with self.assertRaisesRegex(ValueError, "missing publication file"):
            self.check()

    def test_duplicate_file_entry_fails(self):
        self.manifest["files"] *= 2
        self.save()
        with self.assertRaisesRegex(ValueError, "duplicate publication path"):
            self.check()

    def test_unknown_schema_or_empty_inventory_fails(self):
        for key, value in [("schema", "unknown"), ("files", []), ("files", {})]:
            with self.subTest(key=key, value=value):
                original = deepcopy(self.manifest)
                self.manifest[key] = value
                self.save()
                with self.assertRaises(ValueError):
                    self.check()
                self.manifest = original

    def test_malformed_entry_fields_fail(self):
        for entry in [
            None,
            "data.txt",
            {"path": "data.txt"},
            {**self.manifest["files"][0], "extra": 1},
        ]:
            with self.subTest(entry=entry):
                self.manifest["files"] = [entry]
                self.save()
                with self.assertRaises(ValueError):
                    self.check()

    def test_invalid_size_and_digest_fail(self):
        base = deepcopy(self.manifest["files"][0])
        for key, value in [
            ("bytes", True),
            ("bytes", -1),
            ("bytes", 24.0),
            ("sha256", "0" * 63),
            ("sha256", "G" * 64),
            ("sha256", None),
        ]:
            with self.subTest(key=key, value=value):
                self.manifest["files"] = [{**base, key: value}]
                self.save()
                with self.assertRaises(ValueError):
                    self.check()

    def test_unsafe_and_noncanonical_paths_fail_before_reading(self):
        for path in [
            "../data.txt",
            "/data.txt",
            "./data.txt",
            "x/../data.txt",
            "x//data.txt",
            "x\\data.txt",
            "",
            None,
            "bad\x00path",
        ]:
            with self.subTest(path=path):
                self.manifest["files"][0]["path"] = path
                self.save()
                with self.assertRaises(ValueError):
                    self.check()

    def test_file_symlink_is_rejected_even_within_root(self):
        (self.root / "alias.txt").symlink_to(self.root / "data.txt")
        self.manifest["files"][0]["path"] = "alias.txt"
        self.save()
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.check()

    def test_directory_symlink_is_rejected(self):
        (self.root / "real").mkdir()
        (self.root / "real/data.txt").write_bytes(b"synthetic public content")
        (self.root / "alias").symlink_to(self.root / "real", target_is_directory=True)
        self.manifest["files"][0]["path"] = "alias/data.txt"
        self.save()
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.check()

    def test_manifest_cannot_include_itself(self):
        self.manifest["files"][0]["path"] = "manifest.json"
        self.save()
        with self.assertRaisesRegex(ValueError, "hash itself"):
            self.check()

    def test_inventory_omission_or_unexpected_entry_fails(self):
        for expected in [set(), {"data.txt", "omitted.txt"}]:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                    self.check(expected)

    def test_duplicate_json_keys_and_nonfinite_fail(self):
        for content in [
            '{"schema":"one","schema":"two"}',
            '{"bad":NaN}',
            '{"bad":Infinity}',
            '{"bad":1e999}',
        ]:
            with self.subTest(content=content):
                (self.root / "manifest.json").write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.check()


class SourceDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / "research/public_analysis"
        self.package.mkdir(parents=True)
        self.source = "def example():\n    return 7\n"
        (self.package / "example.py").write_text(self.source, encoding="utf-8")
        self.provenance = {
            "modules": {
                "example.py": {
                    "selected_definition_source_sha256": {
                        "example": hashlib.sha256(self.source.rstrip().encode()).hexdigest(),
                    },
                }
            }
        }
        self.save()

    def save(self):
        (self.package / "PROVENANCE.json").write_text(json.dumps(self.provenance), encoding="utf-8")

    def test_exact_definition_is_checked(self):
        self.assertEqual(
            release.verify_source_definitions(self.root),
            {
                "source_modules_verified": 1,
                "copied_definitions_verified": 1,
            },
        )

    def test_whitespace_change_inside_definition_is_detected(self):
        (self.package / "example.py").write_text(self.source.replace("return 7", "return  7"))
        with self.assertRaisesRegex(ValueError, "definition changed"):
            release.verify_source_definitions(self.root)

    def test_new_header_does_not_change_definition_identity(self):
        (self.package / "example.py").write_text('"""New public header."""\n' + self.source)
        self.assertEqual(
            release.verify_source_definitions(self.root)["copied_definitions_verified"], 1
        )

    def test_duplicate_definition_is_not_silently_overwritten(self):
        (self.package / "example.py").write_text(self.source * 2)
        with self.assertRaisesRegex(ValueError, "duplicate source definition"):
            release.verify_source_definitions(self.root)

    def test_missing_definition_is_detected(self):
        (self.package / "example.py").write_text("pass\n")
        with self.assertRaisesRegex(ValueError, "missing copied source definition"):
            release.verify_source_definitions(self.root)

    def test_source_path_traversal_rejected(self):
        self.provenance["modules"]["../example.py"] = self.provenance["modules"].pop("example.py")
        self.save()
        with self.assertRaisesRegex(ValueError, "module basename"):
            release.verify_source_definitions(self.root)


class PublishedReleaseTests(unittest.TestCase):
    def test_actual_frozen_publication(self):
        result = release.check_public_release(ROOT)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["files_verified"], 108)
        self.assertEqual(result["copied_definitions_verified"], 58)
        self.assertGreater(result["local_navigation_links_verified"], 0)

    def test_cli_from_unrelated_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "tools/check_public_release.py")],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(json.loads(result.stdout)["status"], "PASS")

    def test_cli_invalid_root_has_structured_failure_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "tools/check_public_release.py"),
                    "--root",
                    directory,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)["status"], "FAIL")
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
