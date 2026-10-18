"""Unit tests for the automated spectral benchmark harness."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


from pilot.benchmark_spectral_harness import (
    compute_theoretical_flops,
    evaluate_algorithm_trial,
    generate_benchmark_fixture,
    run_spectral_benchmark_harness,
)


class SpectralBenchmarkHarnessTests(unittest.TestCase):
    def test_fixture_generation(self) -> None:
        G, L = 4, 128
        normal_data, zvf_data = generate_benchmark_fixture(G, L)

        assert normal_data["rewards"].shape == (G,)
        assert zvf_data["rewards"].shape == (G,)
        assert normal_data["logps"].shape == (G, L)
        assert zvf_data["logps"].shape == (G, L)

        # Normal rewards have non-zero variance
        assert float(normal_data["rewards"].std()) > 0.0
        # ZVF rewards have exactly zero variance
        assert float(zvf_data["rewards"].std()) == 0.0

    def test_theoretical_flops_scaling(self) -> None:
        flops_std_4_512 = compute_theoretical_flops("standard_grpo", G=4, L=512)
        flops_std_8_512 = compute_theoretical_flops("standard_grpo", G=8, L=512)
        flops_spec_4_512 = compute_theoretical_flops("spectral_legendre_grpo", G=4, L=512)
        flops_givens_4_512 = compute_theoretical_flops("entropic_givens_grpo", G=4, L=512)

        # Standard FLOPs scale linearly with group size
        assert flops_std_8_512 == 2 * flops_std_4_512
        # Spectral and Givens FLOPs exceed standard FLOPs
        assert flops_spec_4_512 > flops_std_4_512
        assert flops_givens_4_512 > flops_spec_4_512

    def test_trial_evaluation_zvf_starvation_recovery(self) -> None:
        G, L = 4, 256
        normal_data, zvf_data = generate_benchmark_fixture(G, L)

        std_metrics = evaluate_algorithm_trial(
            "standard_grpo", normal_data, zvf_data, G, L, num_warmup=1, num_runs=2
        )
        spec_metrics = evaluate_algorithm_trial(
            "spectral_legendre_grpo", normal_data, zvf_data, G, L, num_warmup=1, num_runs=2
        )
        givens_metrics = evaluate_algorithm_trial(
            "entropic_givens_grpo", normal_data, zvf_data, G, L, num_warmup=1, num_runs=2
        )

        # Standard GRPO under ZVF collapses to 0 advantage std and 0 gradient norm
        assert std_metrics.std_advantages_zvf == 0.0
        assert std_metrics.gradient_norm_zvf == 0.0
        assert std_metrics.gradient_norm_retention == 0.0

        # Spectral and Givens GRPO eliminate ZVF starvation
        assert spec_metrics.std_advantages_zvf > 0.0
        assert spec_metrics.gradient_norm_zvf > 0.0
        assert spec_metrics.gradient_norm_retention > 0.0

        assert givens_metrics.std_advantages_zvf > 0.0
        assert givens_metrics.gradient_norm_zvf > 0.0
        assert givens_metrics.gradient_norm_retention > 0.0

    def test_full_benchmark_harness_output_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "test_spectral_results.json"
            result = run_spectral_benchmark_harness(
                group_sizes=(4,),
                sequence_lengths=(512,),
                output_path=out_file,
                num_warmup=1,
                num_runs=2,
            )

            assert out_file.exists()
            with out_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            assert "metadata" in data
            assert "summary" in data
            assert "scaling_trials" in data
            assert len(data["scaling_trials"]) == 1

            trial = data["scaling_trials"][0]
            assert trial["group_size"] == 4
            assert trial["sequence_length"] == 512
            assert "standard_grpo" in trial["results"]
            assert "spectral_legendre_grpo" in trial["results"]
            assert "entropic_givens_grpo" in trial["results"]


if __name__ == "__main__":
    unittest.main()
