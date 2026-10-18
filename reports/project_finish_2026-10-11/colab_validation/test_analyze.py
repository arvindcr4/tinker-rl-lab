"""Acceptance tests for frozen provenance and fail-closed behavior; unittest only."""

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import pytest

BASE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("colab_analyze", BASE / "analyze.py")
analyze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze)


class ReanalysisAcceptance(unittest.TestCase):
    def test_fixed_cohort_and_historical_reproduction(self):
        result = analyze.analyze(BASE / "inputs")
        historical = result["historical_lineage"]
        assert (
            historical["source_rows"],
            historical["candidate_cells"],
            historical["effective_pairs"],
        ) == (80, 23, 17)
        assert historical["synthetic_effective_cells"] == 9
        assert len(historical["rejected_cells"]) == 6
        self.assertAlmostEqual(
            historical["last10_spearman"]["coefficient"], 0.268711656441718, places=10
        )
        proof = result["synthetic_source_proof"]
        assert (
            proof["exact_generator_matches"],
            proof["unique_method_seed_step_keys"],
            proof["excess_duplicate_rows"],
        ) == (5540, 4500, 1040)
        assert [s["n_runs"] for s in result["measured_studies"]] == [12, 10]
        assert [
            s["n"]
            for study in result["measured_studies"]
            for s in study["stratified_mean_zvf_vs_recorded_posttraining_evaluation"]
        ] == [3, 3, 3, 3, 5, 5]
        json.dumps(result, allow_nan=False)

    def test_altered_input_rejected_before_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(BASE / "inputs", root / "inputs")
            shutil.copy(BASE / "PROTOCOL.md", root / "PROTOCOL.md")
            with (root / "inputs/zvf_summary.tsv").open("a") as fh:
                fh.write("\n")
            with pytest.raises(ValueError, match="input hash mismatch"):
                analyze.analyze(root / "inputs")

    def test_duplicate_measured_run_rejected_even_without_hash_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("groupsize_zvf_sweep", "drgrpo_vs_grpo"):
                shutil.copy(BASE / "inputs" / (name + "_selected.json"), root)
            path = root / "groupsize_zvf_sweep_selected.json"
            data = json.loads(path.read_text())
            data["runs"][-1] = data["runs"][0]
            path.write_text(json.dumps(data))
            with pytest.raises(ValueError, match="duplicate or unexpected measured run"):
                analyze.measured(root)

    def test_constant_evaluation_is_undefined(self):
        result = analyze.association([0.1, 0.2, 0.3], [0.99, 0.99, 0.99])
        assert result["coefficient"] is None
        assert result["reason"] == "constant_axis"


if __name__ == "__main__":
    unittest.main()
