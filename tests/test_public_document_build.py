"""Safe-public-build regressions use temporary fixtures and fabricated native tools."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools import build_public_documents as build
from tools import check_public_release as release
from tools import prepare_public_tex_bundle as tex_bundle


ROOT = Path(__file__).resolve().parents[1]
PDF = b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< >>\n%%EOF\n"


@unittest.skipUnless(os.name == "posix", "requires POSIX process groups")
class PublicDocumentBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "checkout"
        self.root.mkdir()
        self.output = self.base / "rebuilt"
        for directory in release.PUBLIC_ROOTS:
            (self.root / directory).mkdir(parents=True)
        for name in release.SNAPSHOT_EXTRAS:
            self.write(name, b"synthetic public input\n")
        self.addendum = self.root / release.REVISION / "addendum"
        self.addendum.mkdir()
        spec = build.DOCUMENTS["addendum"]
        self.write(
            f"{release.REVISION}/addendum/{spec['builder']}",
            (ROOT / release.REVISION / "addendum" / spec["builder"]).read_bytes(),
        )
        self.write(
            f"{release.REVISION}/addendum/{Path(spec['pdf']).stem}.md",
            b"# Public synthetic fixture\n\n3 October 2026\n\n## Frozen endpoint\n\nFixture body.\n",
        )
        self.write(f"{release.REVISION}/addendum/{spec['pdf']}", PDF + b"old reviewed output\n")
        self.write(
            f"{release.REVISION}/addendum/{Path(spec['pdf']).stem}.tex", b"frozen generated tex\n"
        )
        # Ignored legacy build products must never be copied into the fresh stage.
        self.write(f"{release.REVISION}/addendum/build/{spec['pdf']}", PDF + b"stale build\n")
        self.write(f"{release.REVISION}/addendum/build/sentinel", b"must not be staged")
        self.bin = self.base / "bin"
        self.bin.mkdir()
        self.native(
            "pandoc",
            "import sys\nsys.stdin.read()\nprint(r'\\section{Synthetic test}')\n",
        )
        self.native(
            "tectonic",
            "import os, pathlib, sys, time\n"
            "args = sys.argv[1:]\n"
            "assert '--only-cached' in args and '--untrusted' in args\n"
            "out = pathlib.Path(args[args.index('--outdir') + 1])\n"
            "assert not (out / 'sentinel').exists(), 'old build directory was staged'\n"
            "assert not list(pathlib.Path.cwd().glob('*.pdf')), 'old final PDF was staged'\n"
            "mode = os.environ.get('TEST_NATIVE_MODE', 'success')\n"
            "if mode == 'timeout':\n    time.sleep(20)\n"
            "if mode == 'failure':\n    print('fabricated native failure')\n    sys.exit(9)\n"
            "if mode == 'missing':\n    sys.exit(0)\n"
            "if mode == 'signal':\n    os.kill(os.getpid(), 15)\n"
            "name = pathlib.Path(args[2]).stem + '.pdf'\n"
            f"data = {PDF!r}\n"
            "if mode == 'empty':\n    data = b''\n"
            "if mode == 'invalid':\n    data = b'not a PDF' * 20\n"
            "if mode == 'truncated':\n    data = b'%PDF-1.7\\n' + b'x' * 80\n"
            "if mode == 'symlink':\n"
            "    target = out / 'payload.pdf'\n"
            "    target.write_bytes(data)\n"
            "    (out / name).symlink_to(target)\n"
            "else:\n    (out / name).write_bytes(data)\n",
        )
        self.environment = patch.dict(
            os.environ,
            {
                "PATH": str(self.bin) + os.pathsep + os.environ.get("PATH", ""),
                "TEST_NATIVE_MODE": "success",
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.freeze()

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def native(self, name, source):
        path = self.bin / name
        path.write_text(f"#!{sys.executable}\n" + source, encoding="utf-8")
        path.chmod(0o700)

    def freeze(self):
        entries = []
        for name in sorted(release.snapshot_paths(self.root)):
            path = self.root / name
            entries.append(
                {"path": name, "bytes": path.stat().st_size, "sha256": release.sha256(path)}
            )
        self.write(
            release.MANIFEST,
            json.dumps({"schema": "public-scientific-release-v1", "files": entries}).encode(),
        )

    def snapshot(self):
        return {
            path.relative_to(self.root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.root.rglob("*")
            if path.is_file()
        }

    def run_build(self, **kwargs):
        return build.build_documents(self.root, self.output, ["addendum"], **kwargs)

    def make_bundle(self):
        cache = self.base / "local cache"
        cache.mkdir()
        (cache / "plain.tex").write_bytes(b"synthetic TeX resource")
        destination = self.base / "toolchain's bundle with spaces"
        tex_bundle.prepare_bundle(cache, [], destination, {"cache": "Synthetic fixture"})
        return destination / "public-tex-bundle.zip"

    def test_local_bundle_is_staged_hash_bound_and_preserves_safety_flags(self):
        supplied = self.make_bundle()
        before = self.snapshot()
        native = self.bin / "tectonic"
        native.write_text(
            native.read_text() + "\n"
            "bundle = pathlib.Path(args[args.index('--bundle') + 1])\n"
            f"assert bundle != pathlib.Path({str(supplied)!r})\n"
            "assert bundle.name == 'local-tex-bundle.zip'\n"
            f"assert bundle.read_bytes() == {supplied.read_bytes()!r}\n"
        )
        result = self.run_build(tectonic_bundle=supplied)
        self.assertEqual(result["status"], "BUILT")
        self.assertEqual(before, self.snapshot())
        receipt = json.loads((self.output / "BUILD_RECEIPT.json").read_text())
        self.assertEqual(receipt["tectonic_flags"], ["--only-cached", "--untrusted"])
        self.assertEqual(receipt["tectonic_bundle"]["sha256"], release.sha256(supplied))
        self.assertEqual(receipt["tectonic_bundle"]["bytes"], supplied.stat().st_size)
        self.assertEqual(receipt["tectonic_bundle"]["option"], "--bundle")
        self.assertEqual(receipt["tectonic_bundle"]["resource_files"], 1)
        self.assertFalse((self.output / "local-tex-bundle.zip").exists())

    def test_bad_or_missing_bundle_fails_before_native_execution(self):
        corrupt = self.base / "corrupt.zip"
        corrupt.write_bytes(b"not a bundle")
        for supplied in (
            self.base / "missing.zip",
            corrupt,
            "https://example.invalid/tex.zip",
            "file:///tmp/tex.zip",
        ):
            with (
                self.subTest(supplied=supplied),
                patch.object(build, "build_environment") as native,
            ):
                with self.assertRaises((ValueError, OSError)):
                    self.run_build(tectonic_bundle=supplied)
                native.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_symlink_bundle_and_parent_are_rejected(self):
        supplied = self.make_bundle()
        link = self.base / "linked.zip"
        link.symlink_to(supplied)
        parent = self.base / "bundle alias"
        parent.symlink_to(supplied.parent, target_is_directory=True)
        for candidate in (link, parent / supplied.name):
            with self.assertRaisesRegex(ValueError, "symlink"):
                self.run_build(tectonic_bundle=candidate)
        self.assertFalse(self.output.exists())

    def test_corrupt_deflate_bundle_cli_returns_json_failure(self):
        supplied = self.make_bundle()
        with zipfile.ZipFile(supplied) as archive:
            contents = {name: archive.read(name) for name in archive.namelist()}
        with zipfile.ZipFile(supplied, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in contents.items():
                archive.writestr(name, data)
        with zipfile.ZipFile(supplied) as archive:
            entry = archive.getinfo("FILELIST")
            offset = entry.header_offset + 30 + len(entry.filename) + len(entry.extra)
        data = bytearray(supplied.read_bytes())
        data[offset] = (data[offset] & ~6) | 6  # Invalid DEFLATE BTYPE=3.
        supplied.write_bytes(data)
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "tools/build_public_documents.py"),
                "--root",
                str(self.root),
                "--document",
                "addendum",
                "--output-dir",
                str(self.output),
                "--tectonic-bundle",
                str(supplied),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(json.loads(completed.stderr)["status"], "FAIL")
        self.assertIn("invalid local TeX bundle", json.loads(completed.stderr)["error"])
        self.assertEqual(completed.stdout, "")
        self.assertFalse(self.output.exists())

    def test_changed_or_missing_bundle_before_publication_is_rejected(self):
        supplied = self.make_bundle()
        contents = supplied.read_bytes()
        original = build.run_builder
        for mode in ("source", "staged", "missing"):
            supplied.write_bytes(contents)

            def change_bundle(directory, document, env, timeout, log, mode=mode):
                original(directory, document, env, timeout, log)
                if mode == "missing":
                    supplied.unlink()
                else:
                    target = supplied if mode == "source" else log.parent / "local-tex-bundle.zip"
                    target.write_bytes(b"changed bundle")

            with (
                self.subTest(mode=mode),
                patch.object(build, "run_builder", side_effect=change_bundle),
            ):
                with self.assertRaisesRegex(ValueError, "bundle changed"):
                    self.run_build(tectonic_bundle=supplied)
                self.assertFalse(self.output.exists())

    def test_bundle_changed_during_stage_copy_is_rejected(self):
        supplied = self.make_bundle()
        original = build.shutil.copyfileobj

        def corrupt_copy(source, destination, *args, **kwargs):
            original(source, destination, *args, **kwargs)
            if destination.name.endswith("local-tex-bundle.zip"):
                destination.write(b"unexpected")

        with patch.object(build.shutil, "copyfileobj", side_effect=corrupt_copy):
            with self.assertRaisesRegex(ValueError, "changed while staging"):
                self.run_build(tectonic_bundle=supplied)
        self.assertFalse(self.output.exists())

    def test_bundle_shim_cannot_fall_back_to_unrestricted_tool(self):
        supplied = self.make_bundle()
        marker = self.base / "unsafe-tool-ran"
        self.native(
            "tectonic", f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n"
        )
        original = build.build_environment

        def disable_shim(stage, bundle_path=None):
            env = original(stage, bundle_path)
            self.assertEqual(env["PATH"], str(stage / "native-tools"))
            (stage / "native-tools/tectonic").chmod(0o600)
            return env

        with patch.object(build, "build_environment", side_effect=disable_shim):
            with self.assertRaisesRegex(build.BuildError, "builder failed"):
                self.run_build(tectonic_bundle=supplied)
        self.assertFalse(marker.exists())
        self.assertFalse(self.output.exists())

    def add_synthetic_thesis(self):
        spec = build.DOCUMENTS["thesis"]
        prefix = f"{release.REVISION}/thesis"
        builder = (
            "from pathlib import Path\nimport shutil, subprocess\n"
            "here = Path(__file__).resolve().parent\n"
            "assert (here / 'figures/fig_test.pdf').is_file()\n"
            "assert not (here / 'thesis_master.tex').exists()\n"
            "(here / 'thesis_master.tex').write_text('synthetic master')\n"
            "(here / 'build').mkdir()\n"
            "subprocess.run(['tectonic', '-X', 'compile', 'thesis_master.tex', "
            "'--outdir', str(here / 'build')], check=True)\n"
            f"shutil.copyfile(here / 'build/thesis_master.pdf', here / {spec['pdf']!r})\n"
        )
        self.write(f"{prefix}/{spec['builder']}", builder.encode())
        self.write(f"{prefix}/figures/fig_test.pdf", PDF)
        self.write(f"{prefix}/{spec['pdf']}", PDF + b"old thesis")
        self.write(f"{prefix}/thesis_master.tex", b"generated old master")
        self.freeze()

    def test_success_keeps_frozen_sources_and_publishes_receipt_last(self):
        before = self.snapshot()
        result = self.run_build()
        self.assertEqual(result["status"], "BUILT")
        self.assertEqual(before, self.snapshot())
        self.assertEqual((self.output / build.DOCUMENTS["addendum"]["pdf"]).read_bytes(), PDF)
        receipt = json.loads((self.output / "BUILD_RECEIPT.json").read_text())
        inputs = receipt["documents"]["addendum"]["staged_inputs"]
        self.assertEqual(len(inputs), 2)  # Only the frozen builder and Markdown.
        self.assertEqual(receipt["tectonic_flags"], ["--only-cached", "--untrusted"])
        self.assertFalse(receipt["network_isolation"])
        self.assertTrue((self.output / "addendum_build.log").is_file())

    def test_both_documents_are_fresh_before_any_publication(self):
        self.add_synthetic_thesis()
        before = self.snapshot()
        result = build.build_documents(self.root, self.output, ["thesis", "addendum"])
        self.assertEqual(set(result["documents"]), {"thesis", "addendum"})
        self.assertEqual(before, self.snapshot())
        for spec in build.DOCUMENTS.values():
            self.assertEqual((self.output / spec["pdf"]).read_bytes(), PDF)

    def test_second_document_failure_does_not_publish_first(self):
        self.add_synthetic_thesis()
        original = build.run_builder

        def fail_second(directory, document, env, timeout, log):
            if document == "thesis":
                raise build.BuildError("synthetic second build failure")
            return original(directory, document, env, timeout, log)

        with patch.object(build, "run_builder", side_effect=fail_second):
            with self.assertRaisesRegex(build.BuildError, "second build failure"):
                build.build_documents(self.root, self.output, ["addendum", "thesis"])
        self.assertFalse(self.output.exists())

    def test_concurrent_source_edit_blocks_publication(self):
        original = build.run_builder

        def edit_after_build(*args):
            original(*args)
            self.write("README.md", b"concurrent source edit")

        with patch.object(build, "run_builder", side_effect=edit_after_build):
            with self.assertRaisesRegex(ValueError, "mismatch"):
                self.run_build()
        self.assertFalse(self.output.exists())

    def test_failure_modes_do_not_publish_or_modify_source(self):
        before = self.snapshot()
        for mode in ("failure", "signal", "missing", "empty", "invalid", "truncated"):
            with self.subTest(mode=mode), patch.dict(os.environ, {"TEST_NATIVE_MODE": mode}):
                with self.assertRaises(build.BuildError):
                    self.run_build()
                self.assertFalse(self.output.exists())
                self.assertEqual(before, self.snapshot())

    def test_missing_output_cannot_reuse_old_final_or_old_build_pdf(self):
        with patch.dict(os.environ, {"TEST_NATIVE_MODE": "missing"}):
            with self.assertRaisesRegex(build.BuildError, "builder failed"):
                self.run_build()
        self.assertFalse(self.output.exists())

    def test_timeout_does_not_publish_and_leaves_no_stage(self):
        before = self.snapshot()
        with patch.dict(os.environ, {"TEST_NATIVE_MODE": "timeout"}):
            with self.assertRaisesRegex(build.BuildError, "timed out"):
                self.run_build(timeout=0.3)
        self.assertEqual(before, self.snapshot())
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.base.glob(".public-documents-*")))

    def test_existing_destination_is_never_replaced(self):
        for kind in ("empty directory", "file", "dangling symlink"):
            with self.subTest(kind=kind):
                self.output = self.base / kind
                if kind == "empty directory":
                    self.output.mkdir()
                elif kind == "file":
                    self.output.write_text("keep me")
                else:
                    self.output.symlink_to(self.base / "absent")
                with self.assertRaisesRegex(build.BuildError, "already exists"):
                    self.run_build()
                self.assertTrue(os.path.lexists(self.output))
        self.assertEqual((self.base / "file").read_text(), "keep me")

    def test_destination_must_be_outside_checkout_with_existing_parent(self):
        for destination in (self.root / "new-output", self.base / "absent-parent" / "output"):
            with self.subTest(destination=destination), self.assertRaises(build.BuildError):
                build.build_documents(self.root, destination, ["addendum"])
        alias = self.base / "checkout-alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(build.BuildError, "outside"):
            build.build_documents(self.root, alias / "output", ["addendum"])

    def test_output_parent_traversal_cannot_select_the_wrong_directory(self):
        target = self.base / "target"
        (target / "nested").mkdir(parents=True)
        alias = self.base / "output alias"
        alias.symlink_to(target / "nested", target_is_directory=True)
        with self.assertRaisesRegex(build.BuildError, "canonical output path"):
            build.build_documents(self.root, alias / ".." / "built", ["addendum"])
        self.assertFalse((target / "built").exists())
        self.assertFalse((self.base / "built").exists())

    def test_integrity_failure_happens_before_native_execution(self):
        self.write(
            f"{release.REVISION}/addendum/{build.DOCUMENTS['addendum']['builder']}", b"changed"
        )
        with patch.object(build, "build_environment") as environment:
            with self.assertRaisesRegex(ValueError, "mismatch"):
                self.run_build()
            environment.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_unlisted_public_input_fails_closed(self):
        self.write(f"{release.REVISION}/addendum/unlisted.md", b"unlisted")
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            self.run_build()

    def test_missing_native_tools_has_actionable_error(self):
        with patch.dict(os.environ, {"PATH": ""}):
            with self.assertRaisesRegex(build.BuildError, "missing native tool.*pandoc.*tectonic"):
                self.run_build()
        self.assertFalse(self.output.exists())

    def test_interpreter_path_with_spaces_preserves_native_safety_flags(self):
        spaced = self.base / "python environment with spaces"
        spaced.mkdir()
        interpreter = spaced / "python"
        interpreter.symlink_to(sys.executable)
        with patch.object(build.sys, "executable", str(interpreter)):
            result = self.run_build()
        self.assertEqual(result["status"], "BUILT")

    def test_relative_native_path_keeps_meaning_after_staged_chdir(self):
        relative = os.path.relpath(self.bin, Path.cwd())
        with patch.dict(os.environ, {"PATH": relative}):
            result = self.run_build()
        self.assertEqual(result["status"], "BUILT")

    def test_native_paths_with_spaces_and_quotes_are_literal(self):
        native_dir = self.base / "native tool's directory"
        self.bin.rename(native_dir)
        with patch.dict(os.environ, {"PATH": str(native_dir)}):
            result = self.run_build()
        self.assertEqual(result["status"], "BUILT")

    def test_native_env_interpreter_uses_original_path_after_safe_shim(self):
        native = self.bin / "tectonic"
        _, body = native.read_text().split("\n", 1)
        native.write_text("#!/usr/bin/env python3\n" + body)
        (self.bin / "python3").symlink_to(sys.executable)
        relative = os.path.relpath(self.bin, Path.cwd())
        with patch.dict(os.environ, {"PATH": relative}):
            result = self.run_build()
        self.assertEqual(result["status"], "BUILT")

    def test_unexecutable_shim_cannot_fall_back_to_unrestricted_native_tool(self):
        marker = self.base / "unrestricted-tool-ran"
        self.native(
            "tectonic", f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n"
        )
        original = build.build_environment

        def disable_shim(stage):
            env = original(stage)
            (stage / "native-tools/tectonic").chmod(0o600)
            return env

        with patch.object(build, "build_environment", side_effect=disable_shim):
            with self.assertRaisesRegex(build.BuildError, "builder failed"):
                self.run_build()
        self.assertFalse(marker.exists())
        self.assertFalse(self.output.exists())

    def test_bad_timeout_is_rejected(self):
        for timeout in (0, -1, float("nan"), float("inf"), 3601):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(build.BuildError, "timeout"):
                self.run_build(timeout=timeout)

    def test_stage_preserves_declared_figure_pdfs_only(self):
        prefix = f"{release.REVISION}/thesis"
        self.write(f"{prefix}/build_thesis.py", b"# synthetic builder\n")
        self.write(f"{prefix}/figures/fig_test.pdf", PDF)
        self.write(f"{prefix}/figures/fig_test.tex", b"figure source")
        self.write(f"{prefix}/{build.DOCUMENTS['thesis']['pdf']}", PDF)
        self.write(f"{prefix}/thesis_master.tex", b"generated master")
        self.write(f"{prefix}/ignored.log", b"old log")
        self.freeze()
        manifest, _ = build.verified_manifest(self.root)
        stage = self.base / "stage"
        stage.mkdir()
        directory, copied = build.stage_inputs(self.root, stage, manifest, "thesis")
        self.assertEqual((directory / "figures/fig_test.pdf").read_bytes(), PDF)
        self.assertEqual(len(copied), 3)
        self.assertFalse((directory / build.DOCUMENTS["thesis"]["pdf"]).exists())
        self.assertFalse((directory / "thesis_master.tex").exists())
        self.assertFalse((directory / "ignored.log").exists())
        (directory / "figures/fig_test.pdf").write_bytes(b"changed stage")
        self.assertEqual((self.root / prefix / "figures/fig_test.pdf").read_bytes(), PDF)

    def test_changed_input_during_staging_is_rejected(self):
        manifest, _ = build.verified_manifest(self.root)
        self.write(
            f"{release.REVISION}/addendum/{build.DOCUMENTS['addendum']['builder']}", b"changed"
        )
        stage = self.base / "stage"
        stage.mkdir()
        with self.assertRaisesRegex(build.BuildError, "changed while staging"):
            build.stage_inputs(self.root, stage, manifest, "addendum")

    def test_symlinked_final_output_is_rejected(self):
        stage = self.base / "stage"
        stage.mkdir()
        payload = stage / "payload.pdf"
        payload.write_bytes(PDF)
        (stage / build.DOCUMENTS["addendum"]["pdf"]).symlink_to(payload)
        with self.assertRaisesRegex(ValueError, "symlink"):
            build.plausible_pdf(stage, stage, "addendum")

    def test_publication_race_does_not_clobber_existing_directory(self):
        payload = self.base / "new.pdf"
        payload.write_bytes(PDF)
        self.output.mkdir()
        sentinel = self.output / "new.pdf"
        sentinel.write_bytes(b"preserve")
        with self.assertRaisesRegex(build.BuildError, "already exists"):
            build.publish(self.output, [payload], {})
        self.assertEqual(sentinel.read_bytes(), b"preserve")
        self.assertFalse((self.output / "BUILD_RECEIPT.json").exists())

    def test_partial_publication_is_retained_without_completion_receipt(self):
        payload = self.base / "new.pdf"
        payload.write_bytes(PDF)
        with patch.object(build.os, "link", side_effect=OSError("test disk error")):
            with self.assertRaisesRegex(build.BuildError, "publication incomplete"):
                build.publish(self.output, [payload], {})
        self.assertTrue(self.output.is_dir())
        self.assertFalse((self.output / "BUILD_RECEIPT.json").exists())

    def test_cli_failure_is_json_and_nonzero(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "tools/build_public_documents.py"),
                "--root",
                str(self.root),
                "--document",
                "addendum",
                "--output-dir",
                str(self.root / "bad"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(json.loads(completed.stderr)["status"], "FAIL")
        self.assertEqual(completed.stdout, "")


if __name__ == "__main__":
    unittest.main()
