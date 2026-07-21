from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .combine_receipts import combine


class ReceiptCombinationTests(unittest.TestCase):
    def test_combiner_fails_closed_on_cross_stack_drift(self) -> None:
        base = {
            "stack": "trl",
            "status": "PASS",
            "tolerances": {"rtol": 1e-6, "atol": 1e-8, "dtype": "float64"},
            "fixture_digest": "same",
            "controller_matrix": [{"case": "same"}],
            "controller_action_ontology": {"keep": "keep"},
            "intended_cases": [{"verdict": "PASS"}],
            "native_cases": [{"verdict": "MATERIAL_DIFFERENCE"}],
            "source_hashes": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            trl_path = Path(directory) / "trl.json"
            verl_path = Path(directory) / "verl.json"
            trl_path.write_text(json.dumps(base))
            verl = dict(base, stack="verl", fixture_digest="different")
            verl_path.write_text(json.dumps(verl))
            manifest = combine(trl_path, verl_path)
        self.assertEqual(manifest["status"], "S1_FAIL")
        self.assertIn("cross-stack field differs: fixture_digest", manifest["errors"])

    def test_combiner_accepts_two_matching_passing_receipts(self) -> None:
        base = {
            "stack": "trl",
            "status": "PASS",
            "tolerances": {"rtol": 1e-6, "atol": 1e-8, "dtype": "float64"},
            "fixture_digest": "same",
            "controller_matrix": [{"case": "same"}],
            "controller_action_ontology": {"keep": "keep"},
            "intended_cases": [{"verdict": "PASS"}],
            "native_cases": [{"verdict": "MATERIAL_DIFFERENCE"}],
            "source_hashes": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            trl_path = Path(directory) / "trl.json"
            verl_path = Path(directory) / "verl.json"
            trl_path.write_text(json.dumps(base))
            verl_path.write_text(json.dumps(dict(base, stack="verl")))
            manifest = combine(trl_path, verl_path)
        self.assertEqual(manifest["status"], "S1_PASS")
        self.assertEqual(manifest["errors"], [])


if __name__ == "__main__":
    unittest.main()
