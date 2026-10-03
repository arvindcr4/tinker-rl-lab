"""Local bundle tests use synthetic resource bytes, without a native TeX install."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

from tools import prepare_public_tex_bundle as bundle


class PublicTexBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.cache = self.base / "cached resources"
        self.cache.mkdir()
        (self.cache / "plain.tex").write_bytes(b"plain resource\n")
        (self.cache / "pgf.sty").write_bytes(b"old pgf\n")
        self.resources = self.base / "installed resource's root"
        (self.resources / "nested").mkdir(parents=True)
        (self.resources / "nested/pgf.sty").write_bytes(b"updated pgf\n")
        (self.resources / "plot.tex").write_bytes(b"plot resource\n")
        (self.resources / "README.txt").write_bytes(b"not runtime\n")
        self.output = self.base / "prepared bundle"
        self.licenses = {
            "cache": "Package-specific fixture licenses",
            "pgf": "Fixture LPPL evidence",
        }

    def prepare(self, **kwargs):
        return bundle.prepare_bundle(
            self.cache, [("pgf", self.resources)], self.output, self.licenses, **kwargs
        )

    def snapshot(self):
        return {
            str(path): path.read_bytes()
            for root in (self.cache, self.resources)
            for path in root.rglob("*")
            if path.is_file()
        }

    def rewrite_zip(self, modify):
        path = self.output / "public-tex-bundle.zip"
        with zipfile.ZipFile(path) as archive:
            contents = {name: archive.read(name) for name in archive.namelist()}
        modify(contents)
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in contents.items():
                archive.writestr(name, data)
        return path

    def test_deterministic_bytes_and_explicit_overlay_provenance(self):
        before = self.snapshot()
        self.prepare(cache_origin="https://example.invalid/provenance-only")
        first = {path.name: path.read_bytes() for path in self.output.iterdir()}
        self.output = self.base / "second output"
        self.prepare(cache_origin="https://example.invalid/provenance-only")
        self.assertEqual(first, {path.name: path.read_bytes() for path in self.output.iterdir()})
        self.assertEqual(before, self.snapshot())
        provenance = json.loads(first["PROVENANCE.json"])
        self.assertEqual(provenance["resource_files"], 3)
        self.assertEqual(provenance["zip_sha256"], bundle.sha256(first["public-tex-bundle.zip"]))
        self.assertEqual(len(provenance["overlays"]), 1)
        overlay = provenance["overlays"][0]
        self.assertEqual(overlay["previous"]["sha256"], bundle.sha256(b"old pgf\n"))
        self.assertEqual(overlay["replacement"]["license_evidence"], self.licenses["pgf"])
        with zipfile.ZipFile(self.output / "public-tex-bundle.zip") as archive:
            self.assertEqual(archive.read("pgf.sty"), b"updated pgf\n")
            self.assertNotIn("README.txt", archive.namelist())
            for info in archive.infolist():
                self.assertEqual(info.date_time, (1980, 1, 1, 0, 0, 0))
                self.assertEqual(info.compress_type, zipfile.ZIP_STORED)
        self.assertEqual(
            bundle.validate_bundle(self.output / "public-tex-bundle.zip")["resource_files"], 3
        )

    def test_cache_only_supported_with_explicit_evidence(self):
        result = bundle.prepare_bundle(self.cache, [], self.output, {"cache": "fixture terms"})
        self.assertEqual(result["resource_files"], 2)

    def test_duplicate_overlay_basename_fails_even_for_identical_bytes(self):
        (self.resources / "pgf.sty").write_bytes(b"updated pgf\n")
        with self.assertRaisesRegex(ValueError, "duplicate overlay basename"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_duplicate_roots_and_missing_license_notes_fail(self):
        with self.assertRaisesRegex(ValueError, "duplicate resource root"):
            bundle.prepare_bundle(
                self.cache, [("pgf", self.resources)] * 2, self.output, self.licenses
            )
        self.licenses.pop("pgf")
        with self.assertRaisesRegex(ValueError, "license evidence"):
            self.prepare()

    def test_unsafe_and_reserved_cache_names_fail(self):
        for name in ("bad name.tex", "..tex", "bad\\path.tex", "SHA256SUM", "FILELIST"):
            with self.subTest(name=name):
                path = self.cache / name
                path.write_bytes(b"bad")
                with self.assertRaisesRegex(ValueError, "resource name"):
                    self.prepare()
                path.unlink()
        self.assertFalse(self.output.exists())

    def test_cache_directories_are_not_silently_skipped(self):
        (self.cache / "unexpected").mkdir()
        with self.assertRaisesRegex(ValueError, "regular resource"):
            self.prepare()

    def test_symlink_files_directories_roots_and_ancestors_fail(self):
        for name, target in (("ignored.txt", self.cache / "plain.tex"), ("linked-dir", self.cache)):
            with self.subTest(name=name):
                link = self.resources / name
                link.symlink_to(target)
                with self.assertRaisesRegex(ValueError, "symlink"):
                    self.prepare()
                link.unlink()
        alias = self.base / "alias"
        alias.symlink_to(self.resources, target_is_directory=True)
        for root in (alias, alias / "nested"):
            with self.assertRaisesRegex(ValueError, "symlink"):
                bundle.prepare_bundle(self.cache, [("pgf", root)], self.output, self.licenses)
        (self.cache / "linked.tex").symlink_to(self.cache / "plain.tex")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.prepare()

    def test_special_file_and_oversized_resource_fail_without_reading(self):
        if hasattr(os, "mkfifo"):
            fifo = self.resources / "ignored.fifo"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(ValueError, "nonregular"):
                self.prepare()
            fifo.unlink()
        with patch.object(bundle, "MAX_FILE_BYTES", 2):
            with self.assertRaisesRegex(ValueError, "size limit"):
                self.prepare()

    def test_directory_traversal_errors_fail_closed(self):
        original = bundle.os.walk

        def unreadable(root, **kwargs):
            yield str(root), [], ["plot.tex"]
            kwargs["onerror"](PermissionError("unreadable nested resource directory"))
            yield from original(root, **kwargs)

        with patch.object(bundle.os, "walk", side_effect=unreadable):
            with self.assertRaisesRegex(PermissionError, "unreadable nested"):
                self.prepare()
        self.assertFalse(self.output.exists())

    def test_bad_crc_and_deflate_data_raise_actionable_errors(self):
        self.prepare()
        original = self.output / "public-tex-bundle.zip"
        with zipfile.ZipFile(original) as archive:
            contents = {name: archive.read(name) for name in archive.namelist()}
        for compression in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            path = self.base / f"corrupt-{compression}.zip"
            with zipfile.ZipFile(path, "w", compression=compression) as archive:
                for name, data in contents.items():
                    archive.writestr(name, data)
            with zipfile.ZipFile(path) as archive:
                entry = archive.getinfo("FILELIST")
                offset = entry.header_offset + 30 + len(entry.filename) + len(entry.extra)
            data = bytearray(path.read_bytes())
            # BTYPE=3 is invalid DEFLATE; for stored data this breaks the CRC.
            if compression == zipfile.ZIP_DEFLATED:
                data[offset] = (data[offset] & ~6) | 6
            else:
                data[offset] ^= 1
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "invalid local TeX bundle"):
                bundle.validate_bundle(path)

    def test_parent_traversal_is_not_normalized_past_symlink_checks(self):
        target = self.base / "target"
        (target / "nested").mkdir(parents=True)
        (target / self.cache.name).mkdir()
        (target / self.cache.name / "plain.tex").write_bytes(b"different resource")
        alias = self.base / "alias"
        alias.symlink_to(target / "nested", target_is_directory=True)
        supplied = alias / ".." / self.cache.name / "plain.tex"
        self.assertNotEqual(supplied.read_bytes(), (self.cache / "plain.tex").read_bytes())
        with self.assertRaisesRegex(ValueError, "canonical local path"):
            bundle.local_path(supplied)

    def test_existing_outputs_are_never_clobbered(self):
        for kind in ("directory", "file", "symlink"):
            self.output = self.base / kind
            if kind == "directory":
                self.output.mkdir()
            elif kind == "file":
                self.output.write_bytes(b"preserve")
            else:
                self.output.symlink_to(self.base / "absent")
            with self.assertRaisesRegex(ValueError, "already exists"):
                self.prepare()
            self.assertTrue(os.path.lexists(self.output))
        self.assertEqual((self.base / "file").read_bytes(), b"preserve")

    def test_output_inside_inputs_or_checkout_rejected(self):
        for output in (self.cache / "new", self.resources / "new", bundle.ROOT / "new-bundle"):
            self.output = output
            with self.assertRaisesRegex(ValueError, "outside"):
                self.prepare()

    def test_output_parent_traversal_cannot_select_the_wrong_directory(self):
        target = self.base / "target"
        (target / "nested").mkdir(parents=True)
        alias = self.base / "output alias"
        alias.symlink_to(target / "nested", target_is_directory=True)
        self.output = alias / ".." / "prepared"
        with self.assertRaisesRegex(ValueError, "canonical output path"):
            self.prepare()
        self.assertFalse((target / "prepared").exists())
        self.assertFalse((self.base / "prepared").exists())

    def test_corrupt_resource_or_identity_rejected(self):
        for name in ("pgf.sty", "SHA256SUM", "FILELIST"):
            self.output = self.base / name
            self.prepare()
            path = self.rewrite_zip(lambda contents: contents.update({name: b"corrupt"}))
            with self.assertRaises(ValueError):
                bundle.validate_bundle(path)
        invalid = self.base / "invalid.zip"
        invalid.write_bytes(b"not a ZIP")
        with self.assertRaisesRegex(ValueError, "invalid local"):
            bundle.validate_bundle(invalid)

    def test_duplicate_unsafe_and_symlink_zip_members_rejected(self):
        for kind in ("duplicate", "unsafe", "symlink", "extra"):
            self.output = self.base / kind
            self.prepare()
            path = self.output / "public-tex-bundle.zip"
            with zipfile.ZipFile(path, "a") as archive, warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                if kind == "symlink":
                    info = zipfile.ZipInfo("linked.tex")
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                    archive.writestr(info, b"plain.tex")
                else:
                    name = {
                        "duplicate": "plain.tex",
                        "unsafe": "../escape.tex",
                        "extra": "extra.tex",
                    }[kind]
                    archive.writestr(name, b"unexpected")
            with self.assertRaises(ValueError):
                bundle.validate_bundle(path)

    def test_embedded_nul_zip_name_is_not_silently_truncated(self):
        self.prepare()
        path = self.output / "public-tex-bundle.zip"
        data = path.read_bytes().replace(b"plain.tex", b"plain\x00tex")
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "noncanonical ZIP member name"):
            bundle.validate_bundle(path)

    def test_changed_source_before_publication_fails(self):
        original = bundle.validate_bundle

        def modify_after_validation(path):
            result = original(path)
            (self.cache / "plain.tex").write_bytes(b"changed")
            return result

        with patch.object(bundle, "validate_bundle", side_effect=modify_after_validation):
            with self.assertRaisesRegex(ValueError, "changed during preparation"):
                self.prepare()
        self.assertFalse(self.output.exists())

    def test_partial_output_has_no_completion_marker(self):
        with patch.object(bundle.os, "link", side_effect=OSError("fixture failure")):
            with self.assertRaises(OSError):
                self.prepare()
        self.assertTrue(self.output.is_dir())
        self.assertFalse((self.output / "PROVENANCE.json").exists())

    def test_urls_are_rejected_without_network_calls(self):
        for value in (
            "https://example.invalid/tool.zip",
            "file:///tmp/tool.zip",
            "ftp://host/tool.zip",
        ):
            with self.assertRaisesRegex(ValueError, "local path"):
                bundle.local_path(value)

    def test_cli_paths_with_spaces_and_duplicate_evidence(self):
        command = [
            sys.executable,
            "-B",
            str(bundle.ROOT / "tools/prepare_public_tex_bundle.py"),
            "--cache-dir",
            str(self.cache),
            "--resource-root",
            f"pgf={self.resources}",
            "--license-evidence",
            "cache=fixture terms",
            "--license-evidence",
            "pgf=fixture terms",
            "--output-dir",
            str(self.output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "PREPARED")
        completed = subprocess.run(
            command + ["--license-evidence", "pgf=duplicate"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertIn("duplicate license", json.loads(completed.stderr)["error"])


if __name__ == "__main__":
    unittest.main()
