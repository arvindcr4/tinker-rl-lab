from __future__ import annotations

import unittest

from .modal_e1_swe_bench_pro import _extract_diff, _validate_unified_diff


VALID_PATCH = """diff --git a/a.txt b/a.txt
index 7898192..6178079 100644
--- a/a.txt
+++ b/a.txt
@@ -1 +1 @@
-old
+new
"""


class E1PatchValidationTests(unittest.TestCase):
    def test_accepts_concrete_unified_diff(self) -> None:
        self.assertEqual(_validate_unified_diff(VALID_PATCH), (True, "valid unified diff"))
        self.assertEqual(_extract_diff(f"preface\n{VALID_PATCH}"), VALID_PATCH)

    def test_rejects_placeholder_hunk(self) -> None:
        invalid = VALID_PATCH.replace("@@ -1 +1 @@", "@@ ... @@")
        valid, reason = _validate_unified_diff(invalid)
        self.assertFalse(valid)
        self.assertIn("concrete hunk", reason)
        self.assertEqual(_extract_diff(invalid), "")

    def test_rejects_prose_after_diff(self) -> None:
        invalid = VALID_PATCH + "I will now modify another file.\n"
        valid, reason = _validate_unified_diff(invalid)
        self.assertFalse(valid)
        self.assertIn("invalid patch line", reason)


if __name__ == "__main__":
    unittest.main()
