from __future__ import annotations

import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pilot.bootstrap import BootstrapError, load_secrets, main, safe_extract


class BootstrapTests(unittest.TestCase):
    def test_safe_extract_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "bad.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                info = tarfile.TarInfo("../escape")
                info.size = 1
                archive.addfile(info, io.BytesIO(b"x"))
            with self.assertRaisesRegex(BootstrapError, "escapes destination"):
                safe_extract(archive_path, Path(temporary) / "output")

    def test_secrets_are_loaded_then_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "secrets.json"
            path.write_text(json.dumps({"HF_TOKEN": "hf_test", "WANDB_API_KEY": "wb_test"}))
            load_secrets(path)
            self.assertFalse(path.exists())

    def test_secrets_fail_closed_on_extra_fields_and_are_still_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "secrets.json"
            path.write_text(
                json.dumps(
                    {"HF_TOKEN": "hf_test", "WANDB_API_KEY": "wb_test", "EXTRA": "bad"}
                )
            )
            with self.assertRaisesRegex(BootstrapError, "unexpected"):
                load_secrets(path)
            self.assertFalse(path.exists())

    def test_main_sets_cublas_workspace_config_before_any_remote_work(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(BootstrapError, "missing"):
                main()
            self.assertEqual(os.environ.get("CUBLAS_WORKSPACE_CONFIG"), ":4096:8")


if __name__ == "__main__":
    unittest.main()
