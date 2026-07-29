from __future__ import annotations

import unittest
import torch

from pilot.spectral_attention import (
    SpectralAttentionError,
    LegendreSpectralRouting,
    compute_legendre_polynomials,
    legendre_basis,
    legendre_grid,
    legendre_spectral_projection,
    spectral_pairwise_distance,
)


class SpectralAttentionTests(unittest.TestCase):
    def test_legendre_grid_bounds(self) -> None:
        grid = legendre_grid(5)
        self.assertEqual(grid.shape, (5,))
        self.assertAlmostEqual(float(grid[0]), -1.0)
        self.assertAlmostEqual(float(grid[-1]), 1.0)
        self.assertAlmostEqual(float(grid[2]), 0.0)

    def test_legendre_grid_single_token(self) -> None:
        grid = legendre_grid(1)
        self.assertEqual(grid.shape, (1,))
        self.assertEqual(float(grid[0]), 0.0)

    def test_legendre_polynomials_orthogonality_and_values(self) -> None:
        t = torch.tensor([-1.0, 0.0, 0.5, 1.0], dtype=torch.float64)
        polys = compute_legendre_polynomials(4, t)
        self.assertEqual(polys.shape, (4, 4))
        # P0(x) = 1
        self.assertTrue(torch.allclose(polys[:, 0], torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)))
        # P1(x) = x
        self.assertTrue(torch.allclose(polys[:, 1], t))
        # P2(x) = 0.5 * (3*x^2 - 1)
        expected_p2 = 0.5 * (3 * t**2 - 1)
        self.assertTrue(torch.allclose(polys[:, 2], expected_p2))

    def test_legendre_spectral_projection_shapes(self) -> None:
        x = torch.randn(8, 16, 32, dtype=torch.float64)
        coeffs = legendre_spectral_projection(x, n_modes=6)
        self.assertEqual(coeffs.shape, (8, 6, 32))

    def test_spectral_pairwise_distance_positive(self) -> None:
        c1 = torch.randn(4, 32, dtype=torch.float64)
        c2 = torch.randn(4, 32, dtype=torch.float64)
        dist = spectral_pairwise_distance(c1, c2)
        self.assertGreaterEqual(float(dist), 0.0)
        zero_dist = spectral_pairwise_distance(c1, c1)
        self.assertAlmostEqual(float(zero_dist), 0.0)

    def test_legendre_spectral_routing_forward(self) -> None:
        router = LegendreSpectralRouting(d_model=16, n_modes=4, n_cut=2)
        x = torch.randn(4, 10, 16)
        output = router(x)
        self.assertEqual(output.shape, (4, 10, 16))
        self.assertTrue(torch.isfinite(output).all())

    def test_invalid_inputs_raise_spectral_attention_error(self) -> None:
        with self.assertRaises(SpectralAttentionError):
            legendre_grid(0)
        with self.assertRaises(SpectralAttentionError):
            compute_legendre_polynomials(0, torch.tensor([0.0]))
        with self.assertRaises(SpectralAttentionError):
            LegendreSpectralRouting(d_model=16, n_modes=4, n_cut=5)


if __name__ == "__main__":
    unittest.main()
