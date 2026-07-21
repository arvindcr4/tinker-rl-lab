from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pilot.remote_training import _verify_files
from pilot.remote_core import RemoteContractError


class RemoteTrainingSafetyTests(unittest.TestCase):
    def test_bound_file_verifier_rejects_missing_or_tampered_corpus_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "group.pt"
            path.write_bytes(b"payload")
            from pilot.protocol import sha256_file

            digest = sha256_file(path)
            _verify_files(root, {"group.pt": digest}, label="corpus")
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(RemoteContractError, "hash mismatch"):
                _verify_files(root, {"group.pt": digest}, label="corpus")
            with self.assertRaisesRegex(RemoteContractError, "hash mismatch"):
                _verify_files(root, {"missing.pt": digest}, label="corpus")


if __name__ == "__main__":
    unittest.main()
