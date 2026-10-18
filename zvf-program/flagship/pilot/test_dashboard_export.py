from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pilot.dashboard_export import (
    DashboardExportError,
    EntropicGatingData,
    GradientRecoveryData,
    SpectralTrajectoryData,
    export_comparative_dashboard_html,
    export_gating_density_heatmap_html,
    export_gradient_recovery_html,
    export_spectral_trajectory_html,
    prepare_entropic_gating_data,
    prepare_gradient_recovery_data,
    prepare_spectral_trajectory_data,
)
import pytest


class DashboardExportTests(unittest.TestCase):
    def test_spectral_trajectory_preparation_2d(self) -> None:
        coeffs = torch.randn(4, 8)
        data = prepare_spectral_trajectory_data(coeffs, labels=["Mode0", "Mode1", "Mode2", "Mode3"])
        assert isinstance(data, SpectralTrajectoryData)
        assert len(data.distances) == 4
        assert len(data.distances[0]) == 4
        assert data.labels == ["Mode0", "Mode1", "Mode2", "Mode3"]
        assert data.metadata["n_modes"] == 4
        assert data.metadata["d_model"] == 8

    def test_spectral_trajectory_preparation_3d(self) -> None:
        coeffs = torch.randn(3, 4, 16)
        data = prepare_spectral_trajectory_data(coeffs)
        assert len(data.distances) == 3
        assert len(data.labels) == 3
        assert data.metadata["batch_size"] == 3
        assert data.metadata["n_modes"] == 4

    def test_spectral_trajectory_invalid_inputs(self) -> None:
        with pytest.raises(DashboardExportError):
            prepare_spectral_trajectory_data(torch.tensor([1.0, 2.0, 3.0]))  # 1D invalid

        with pytest.raises(DashboardExportError):
            prepare_spectral_trajectory_data(np.array([[1.0, np.nan], [2.0, 3.0]]))  # NaN

        with pytest.raises(DashboardExportError):
            prepare_spectral_trajectory_data(
                torch.randn(3, 4, 8), labels=["L1", "L2"]
            )  # label count mismatch

    def test_entropic_gating_preparation(self) -> None:
        logits = torch.randn(6, 10)
        data = prepare_entropic_gating_data(logits, n_noise_dims=2)
        assert isinstance(data, EntropicGatingData)
        assert len(data.pre_gating_entropy) == 6
        assert len(data.post_gating_entropy) == 6
        assert data.metadata["n_noise_dims"] == 2
        assert data.metadata["sequence_length"] == 6

    def test_entropic_gating_invalid_inputs(self) -> None:
        with pytest.raises(DashboardExportError):
            prepare_entropic_gating_data(
                torch.randn(5, 5), n_noise_dims=5
            )  # invalid n_noise_dims >= cols

        with pytest.raises(DashboardExportError):
            prepare_entropic_gating_data(
                torch.randn(5, 5), n_noise_dims=0
            )  # invalid n_noise_dims <= 0

    def test_gradient_recovery_preparation(self) -> None:
        receipts = {
            "intended_full": [
                {
                    "step": 1,
                    "gradient_norm": 0.95,
                    "gradient_cosine": 0.99,
                    "gradient_relative_l2": 0.01,
                },
                {
                    "step": 2,
                    "gradient_norm": 0.98,
                    "gradient_cosine": 0.995,
                    "gradient_relative_l2": 0.008,
                },
            ],
            "native_trl": [
                {
                    "step": 1,
                    "gradient_norm": 0.85,
                    "gradient_cosine": 0.92,
                    "gradient_relative_l2": 0.05,
                },
                {
                    "step": 2,
                    "gradient_norm": 0.88,
                    "gradient_cosine": 0.94,
                    "gradient_relative_l2": 0.04,
                },
            ],
        }
        data = prepare_gradient_recovery_data(receipts)
        assert isinstance(data, GradientRecoveryData)
        assert data.steps == [1, 2]
        assert "intended_full" in data.curves
        assert data.curves["intended_full"] == [0.95, 0.98]
        assert data.metadata["num_conditions"] == 2

    def test_gradient_recovery_invalid_inputs(self) -> None:
        with pytest.raises(DashboardExportError):
            prepare_gradient_recovery_data({})  # Empty receipts dict

        with pytest.raises(DashboardExportError):
            prepare_gradient_recovery_data({"cond1": []})  # Empty receipt list

    def test_export_spectral_trajectory_html(self) -> None:
        coeffs = torch.randn(3, 4, 8)
        data = prepare_spectral_trajectory_data(coeffs)
        html_str = export_spectral_trajectory_html(data, title="Test Spectral Distances")

        assert html_str.startswith("<!DOCTYPE html>")
        assert "<title>Test Spectral Distances</title>" in html_str
        assert "spectral-matrix-svg" in html_str
        assert "spectralData =" in html_str

    def test_export_gating_density_heatmap_html(self) -> None:
        logits = torch.randn(4, 8)
        data = prepare_entropic_gating_data(logits)
        html_str = export_gating_density_heatmap_html(data)

        assert html_str.startswith("<!DOCTYPE html>")
        assert "gating-heatmap-svg" in html_str
        assert "gatingData =" in html_str

    def test_export_gradient_recovery_html(self) -> None:
        receipts = {
            "intended_full": [{"step": 10, "gradient_norm": 0.95}],
            "spectral_legendre": [{"step": 10, "gradient_norm": 0.99}],
        }
        data = prepare_gradient_recovery_data(receipts)
        html_str = export_gradient_recovery_html(data)

        assert html_str.startswith("<!DOCTYPE html>")
        assert "gradient-curves-svg" in html_str
        assert "gradientData =" in html_str

    def test_export_comparative_dashboard_html_and_file(self) -> None:
        spec_data = prepare_spectral_trajectory_data(torch.randn(3, 4, 8))
        gate_data = prepare_entropic_gating_data(torch.randn(4, 8))
        grad_data = prepare_gradient_recovery_data(
            {
                "intended_full": [{"step": 1, "gradient_norm": 1.0}],
                "entropic_givens": [{"step": 1, "gradient_norm": 0.98}],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "dashboard.html"
            html_str = export_comparative_dashboard_html(
                spectral_data=spec_data,
                gating_data=gate_data,
                gradient_data=grad_data,
                output_path=out_file,
                title="ZVF Flagship Pilot Comparative Dashboard",
            )

            assert out_file.exists()
            file_content = out_file.read_text(encoding="utf-8")
            assert html_str == file_content
            assert "ZVF Flagship Pilot Comparative Dashboard" in file_content
            assert "Max Spectral Distance" in file_content
            assert "Mean Entropy Reduction" in file_content
            assert "spectral-matrix-svg" in file_content
            assert "gating-heatmap-svg" in file_content
            assert "gradient-curves-svg" in file_content


if __name__ == "__main__":
    unittest.main()
