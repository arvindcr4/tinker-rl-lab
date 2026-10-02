"""Regression tests for the stored-group exact/tolerance ZVF distinction.

These tests use the standard library and also run under pytest.
"""

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "platform_modal/scripts/pcd_vs_zvf.py"
RESULTS = ROOT / "platform_hybrid/experiments/results"
SPEC = importlib.util.spec_from_file_location("pcd_vs_zvf", SCRIPT)
PCD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PCD)


class TestZvfDefinitions(unittest.TestCase):
    def test_population_and_sample_variance_are_distinct(self):
        self.assertEqual(PCD.pvar([0.0, 1.0]), 0.25)
        self.assertEqual(PCD.svar([0.0, 1.0]), 0.5)
        self.assertEqual(PCD.pcd([0.0, 1.0]), 0.25)

    def test_tolerance_uses_sample_variance_and_inclusive_boundary(self):
        self.assertEqual(PCD.zvf_tolerance_ind([0.0, 1.0], 0.3), 0.0)
        self.assertEqual(PCD.zvf_tolerance_ind([0.0, 1.0], 0.5), 1.0)
        self.assertEqual(PCD.zvf_tolerance_ind([0.0, 1.0], math.nextafter(0.5, 0)), 0.0)

    def test_historical_indicator_remains_exact_zero(self):
        for group in ([0.0] * 8, [1.0] * 8):
            self.assertEqual(PCD.zvf_ind(group), 1.0)
            self.assertEqual(PCD.zvf_exact_ind(group), 1.0)
            self.assertEqual(PCD.zvf_tolerance_ind(group), 1.0)
        tiny_contrast = [0.0, 1e-4]
        self.assertEqual(PCD.zvf_ind(tiny_contrast), 0.0)
        self.assertEqual(PCD.zvf_exact_ind(tiny_contrast), 0.0)
        self.assertEqual(PCD.zvf_tolerance_ind(tiny_contrast), 1.0)

    def test_invalid_thresholds_are_rejected(self):
        for threshold in (-1.0, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                PCD.zvf_tolerance_ind([0.0, 1.0], threshold)

    def test_degenerate_groups_are_rejected(self):
        with self.assertRaises(ValueError):
            PCD.pvar([])
        for group in ([], [0.0]):
            with self.subTest(group=group), self.assertRaises(ValueError):
                PCD.svar(group)

    def test_invalid_analysis_inputs_are_rejected(self):
        for groups in ([], [[1.0]], [[0.0, float("nan")]], [[0.0, float("inf")]]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                PCD.analyze_jitter(groups)
        for kwargs in ({"amplitude": -1}, {"epsilon": -1}, {"epsilon_grid": [float("nan")]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PCD.analyze_jitter([[0.0, 1.0]], **kwargs)

    def test_zero_amplitude_has_no_effect(self):
        analysis, details = PCD.analyze_jitter([[0.0] * 8, [0.0, 1.0] * 4], amplitude=0)
        self.assertEqual(analysis["exact_zero_before_count"], analysis["exact_zero_after_count"])
        self.assertEqual(analysis["pcd_delta"], 0.0)
        self.assertTrue(all(row["changed_count"] == 0 for row in analysis["epsilon_sensitivity"]))
        self.assertTrue(
            all(row["sample_variance_before"] == row["sample_variance_after"] for row in details)
        )

    def test_seed_is_repeatable_and_does_not_touch_global_rng(self):
        state = random.getstate()
        groups = [[0.0] * 8, [1.0] * 8]
        first = PCD.analyze_jitter(groups, seed=0)
        self.assertEqual(first, PCD.analyze_jitter(groups, seed=0))
        self.assertNotEqual(first, PCD.analyze_jitter(groups, seed=1))
        self.assertEqual(state, random.getstate())

    def test_custom_epsilon_is_included_in_sweep(self):
        analysis, _ = PCD.analyze_jitter([[0.0, 1.0]], epsilon=0.5, epsilon_grid=[0.0])
        self.assertEqual([row["epsilon"] for row in analysis["epsilon_sensitivity"]], [0.0, 0.5])
        self.assertEqual(analysis["tolerance_before_count"], 1)

    def test_import_does_not_read_inputs_write_artifacts_or_print(self):
        code = (
            "import importlib.util, random; state = random.getstate(); "
            f"spec = importlib.util.spec_from_file_location('check', {str(SCRIPT)!r}); "
            "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); "
            "assert random.getstate() == state"
        )
        with tempfile.TemporaryDirectory() as workdir:
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=workdir,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertEqual(result.stdout, "")
            self.assertEqual(list(Path(workdir).iterdir()), [])


class TestStoredGroupRecomputation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups, cls.sources = PCD.load_groups(RESULTS)
        cls.analysis, cls.details = PCD.analyze_jitter(cls.groups)

    def test_all_600_groups_and_historical_source_order_are_used(self):
        self.assertEqual(len(self.groups), 600)
        self.assertEqual({len(group) for group in self.groups}, {8})
        self.assertEqual([source["seed"] for source in self.sources], [123, 42, 456])
        self.assertEqual([source["n_groups"] for source in self.sources], [200, 200, 200])
        self.assertEqual(sum(all(reward == 1 for reward in group) for group in self.groups), 76)
        self.assertEqual(sum(all(reward == 0 for reward in group) for group in self.groups), 19)

    def test_exact_ties_collapse_but_canonical_tolerance_does_not(self):
        self.assertEqual(self.analysis["exact_zero_before_count"], 95)
        self.assertEqual(self.analysis["exact_zero_after_count"], 0)
        self.assertEqual(self.analysis["tolerance_before_count"], 95)
        self.assertEqual(self.analysis["tolerance_after_count"], 95)
        self.assertEqual(self.analysis["tolerance_changed_count"], 0)
        self.assertEqual(self.analysis["tolerance_before_fraction"], 95 / 600)
        self.assertEqual(self.analysis["tolerance_after_fraction"], 95 / 600)

    def test_seeded_epsilon_sensitivity(self):
        rows = self.analysis["epsilon_sensitivity"]
        self.assertEqual([row["before_count"] for row in rows], [95] * 8)
        self.assertEqual([row["after_count"] for row in rows], [0, 0, 0, 71, 95, 95, 95, 95])
        self.assertEqual([row["changed_count"] for row in rows], [95, 95, 95, 24, 0, 0, 0, 0])
        self.assertTrue(all(row["variance_ddof"] == 1 for row in rows))

    def test_pcd_is_small_change_not_literal_invariance(self):
        self.assertAlmostEqual(self.analysis["pcd_before"], 0.15380208333333334, places=15)
        self.assertAlmostEqual(self.analysis["pcd_after"], 0.1538023817497187, places=15)
        self.assertGreater(self.analysis["pcd_delta"], 0)
        self.assertLess(self.analysis["pcd_delta"], 1e-6)
        self.assertEqual(f"{self.analysis['pcd_before']:.6f}", f"{self.analysis['pcd_after']:.6f}")

    def test_group_ledger_matches_summaries_and_ddof(self):
        for row in self.details:
            self.assertAlmostEqual(
                row["sample_variance_after"], row["population_variance_after"] * 8 / 7, places=15
            )
        for phase in ("before", "after"):
            self.assertEqual(
                sum(row[f"exact_zero_{phase}"] for row in self.details),
                self.analysis[f"exact_zero_{phase}_count"],
            )
            self.assertEqual(
                sum(row[f"tolerance_{phase}"] for row in self.details),
                self.analysis[f"tolerance_{phase}_count"],
            )

    def test_source_hashes_match_raw_tensors(self):
        for source in self.sources:
            self.assertEqual(
                source["sha256"],
                hashlib.sha256((RESULTS / source["file"]).read_bytes()).hexdigest(),
            )

    def test_missing_or_nonbinary_source_data_fails_closed(self):
        with tempfile.TemporaryDirectory() as workdir:
            with self.assertRaises(ValueError):
                PCD.load_groups(workdir)
            path = Path(workdir) / "tinker_gsm8k_zvf_s1.json"
            path.write_text(json.dumps({"per_problem": [{"rewards": [0.0, 0.5]}]}))
            with self.assertRaises(ValueError):
                PCD.load_groups(workdir)

    def test_rerun_matches_committed_outputs_and_preserves_inputs(self):
        names = [
            "pcd_vs_zvf_shape.tsv",
            "pcd_vs_zvf_recomputed_summary.tsv",
            "pcd_vs_zvf_epsilon_sensitivity.tsv",
            "pcd_vs_zvf_jitter_groups.tsv",
            "pcd_vs_zvf_tolerance_analysis.json",
        ]
        before_hashes = [source["sha256"] for source in self.sources]
        historical_bytes = (RESULTS / "pcd_vs_zvf_summary.tsv").read_bytes()
        with tempfile.TemporaryDirectory() as workdir, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(PCD.main(["--results-dir", str(RESULTS), "--output-dir", workdir]), 0)
            for name in names:
                with self.subTest(name=name):
                    self.assertEqual(
                        (Path(workdir) / name).read_bytes(), (RESULTS / name).read_bytes()
                    )
            self.assertFalse((Path(workdir) / "pcd_vs_zvf_summary.tsv").exists())
            with (Path(workdir) / "pcd_vs_zvf_recomputed_summary.tsv").open() as handle:
                summary = {
                    row["metric"]: row["value"] for row in csv.DictReader(handle, delimiter="\t")
                }
            self.assertEqual(summary["zvf_batch_before_jitter"], "0.1583")
            self.assertEqual(summary["zvf_batch_after_jitter"], "0.0000")
            self.assertEqual(float(summary["zvf_tolerance_batch_after_jitter"]), 95 / 600)
        self.assertEqual(
            before_hashes, [source["sha256"] for source in PCD.load_groups(RESULTS)[1]]
        )
        self.assertEqual(historical_bytes, (RESULTS / "pcd_vs_zvf_summary.tsv").read_bytes())


if __name__ == "__main__":
    unittest.main()
