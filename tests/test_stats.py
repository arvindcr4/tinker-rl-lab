"""Tests for the REAL implementations in ``utils/stats.py`` and
``utils/verify_results.py``.

Small synthetic inputs with real assertions; bootstrap resampling uses a
seeded RNG and few reps to stay fast. Heavyweight paths (rliable install,
model downloads) are not exercised here.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.stats import (
    bootstrap_ci,
    compute_bootstrap_ci,
    compute_iqm,
    generate_results_table,
    load_multi_seed_results,
    mann_whitney_u,
    welch_ttest,
)
from utils.verify_results import _match_key, verify


class TestComputeBootstrapCI(unittest.TestCase):
    def test_mean_and_ci_ordering(self):
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        mean, lower, upper = compute_bootstrap_ci(
            scores, n_bootstrap=200, rng=np.random.default_rng(0)
        )
        self.assertAlmostEqual(mean, 0.5, places=7)
        self.assertLessEqual(lower, mean)
        self.assertLessEqual(mean, upper)
        self.assertLess(lower, upper)

    def test_reproducible_with_rng(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        first = compute_bootstrap_ci(scores, n_bootstrap=200, rng=np.random.default_rng(7))
        second = compute_bootstrap_ci(scores, n_bootstrap=200, rng=np.random.default_rng(7))
        self.assertEqual(first, second)

    def test_alias_matches(self):
        self.assertIs(bootstrap_ci, compute_bootstrap_ci)


class TestStatisticalTests(unittest.TestCase):
    def test_welch_separated_groups_significant(self):
        a = np.array([0.80, 0.82, 0.85, 0.83, 0.81, 0.84])
        b = np.array([0.50, 0.52, 0.48, 0.51, 0.49, 0.53])
        out = welch_ttest(a, b)
        self.assertTrue(out["significant_at_005"])
        self.assertGreater(out["effect_size_cohens_d"], 1.0)
        self.assertAlmostEqual(out["mean_a"], float(np.mean(a)), places=7)
        self.assertEqual(out["n_a"], 6)
        self.assertEqual(out["n_b"], 6)

    def test_welch_identical_groups_not_significant(self):
        a = np.array([0.5, 0.6, 0.55, 0.65, 0.52, 0.58])
        out = welch_ttest(a, a.copy())
        self.assertFalse(out["significant_at_005"])
        self.assertAlmostEqual(out["effect_size_cohens_d"], 0.0, places=7)

    def test_mann_whitney_separated(self):
        a = np.array([0.80, 0.82, 0.85, 0.83, 0.81, 0.84])
        b = np.array([0.50, 0.52, 0.48, 0.51, 0.49, 0.53])
        out = mann_whitney_u(a, b)
        self.assertTrue(out["significant_at_005"])
        self.assertGreater(out["median_a"], out["median_b"])


class TestComputeIQM(unittest.TestCase):
    def test_iqm_central_mass(self):
        scores = np.array([0.0, 0.5, 0.5, 0.5, 1.0])
        self.assertAlmostEqual(compute_iqm(scores), 0.5, places=7)

    def test_iqm_empty_is_nan(self):
        self.assertTrue(np.isnan(compute_iqm(np.array([]))))


class TestLoadMultiSeedResults(unittest.TestCase):
    def test_loads_seed_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp) / "demo"
            for seed in (0, 1):
                seed_dir = exp_dir / f"seed_{seed}"
                seed_dir.mkdir(parents=True)
                with open(seed_dir / "metrics.jsonl", "w") as f:
                    for step in range(3):
                        f.write(json.dumps({"reward/mean": 0.1 * (seed + 1) + step}) + "\n")
            got = load_multi_seed_results(tmp, "demo")
            self.assertEqual(sorted(got.keys()), [0, 1])
            self.assertEqual(len(got[0]), 3)
            self.assertAlmostEqual(got[1][2]["reward/mean"], 0.1 * 2 + 2, places=7)


class TestGenerateResultsTable(unittest.TestCase):
    def test_writes_tex_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tex = str(Path(tmp) / "results_table.tex")
            df = generate_results_table(
                {
                    "algo_a": np.array([0.8, 0.82, 0.85]),
                    "algo_b": np.array([0.5, 0.52, 0.48]),
                },
                output_path=tex,
            )
            self.assertTrue(Path(tex).exists())
            self.assertTrue(Path(tex.replace(".tex", ".csv")).exists())
            self.assertEqual(set(df["Algorithm"]), {"algo_a", "algo_b"})


class TestVerifyResults(unittest.TestCase):
    def _write_result(self, directory: Path, name: str, last10: float, peak: float):
        payload = {
            "experiment": name,
            "last10_avg": last10,
            "peak": peak,
        }
        (directory / f"{name}.json").write_text(json.dumps(payload))

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.expected = {"gsm8k_qwen3_8b": {"last10": 0.344, "peak": 0.625}}

    def test_verify_within_tolerance(self):
        self._write_result(self.root, "gsm8k_qwen3_8b_s42", 0.35, 0.63)
        rows, failed = verify(self.root, self.expected, last10_tol=0.05, peak_tol=0.10)
        self.assertEqual(failed, 0)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0][6])

    def test_verify_outside_tolerance(self):
        self._write_result(self.root, "gsm8k_qwen3_8b_s42", 0.90, 0.95)
        _, failed = verify(self.root, self.expected, last10_tol=0.05, peak_tol=0.10)
        self.assertEqual(failed, 1)

    def test_match_key_prefers_longest(self):
        expected = {
            "gsm8k_qwen3_8b": {"last10": 0.0, "peak": 0.0},
            "gsm8k_qwen3_8b_base": {"last10": 0.0, "peak": 0.0},
        }
        self.assertEqual(_match_key("gsm8k_qwen3_8b_base_s1", expected), "gsm8k_qwen3_8b_base")
        self.assertEqual(_match_key("gsm8k_qwen3_8b_s42", expected), "gsm8k_qwen3_8b")


if __name__ == "__main__":
    unittest.main()
