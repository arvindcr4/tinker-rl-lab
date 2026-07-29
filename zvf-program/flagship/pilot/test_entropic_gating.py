from __future__ import annotations

import unittest
import torch

from pilot.entropic_gating import (
    EntropicGatingError,
    GivensEntropyGate,
    apply_givens_rotation_pair,
    compute_attention_entropy,
    compute_entropy_density,
    eliminate_noise_components,
    givens_rotation_angle,
)


class EntropicGatingTests(unittest.TestCase):
    def test_compute_entropy_density_uniform(self) -> None:
        # Uniform distribution p = [0.25, 0.25, 0.25, 0.25], H = log(4)
        p = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
        h = compute_entropy_density(p)
        self.assertAlmostEqual(float(h), torch.log(torch.tensor(4.0, dtype=torch.float64)).item())

    def test_givens_rotation_norm_preservation(self) -> None:
        x = torch.tensor([[3.0, 4.0, 5.0]], dtype=torch.float64)
        theta = givens_rotation_angle(x[0, 0], x[0, 1])
        x_rot = apply_givens_rotation_pair(x, 0, 1, theta)
        
        orig_norm = torch.linalg.vector_norm(x)
        rot_norm = torch.linalg.vector_norm(x_rot)
        self.assertAlmostEqual(float(orig_norm), float(rot_norm), places=12)
        # Entry 1 should be zeroed out
        self.assertAlmostEqual(float(x_rot[0, 1]), 0.0, places=12)
        # Entry 0 should equal sqrt(3^2 + 4^2) = 5.0
        self.assertAlmostEqual(float(x_rot[0, 0]), 5.0, places=12)

    def test_eliminate_noise_components(self) -> None:
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
        x_proj, x_rot = eliminate_noise_components(x, n_noise_dims=1)
        self.assertEqual(x_proj.shape, (1, 4))
        self.assertEqual(float(x_proj[0, 3]), 0.0)
        # Norm of x_rot must equal orig norm
        self.assertAlmostEqual(float(torch.linalg.vector_norm(x)), float(torch.linalg.vector_norm(x_rot)), places=12)

    def test_givens_entropy_gate_forward(self) -> None:
        gate = GivensEntropyGate(tau_entropy=10.0, n_noise_dims=1)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
        # Entropy of softmax(x) is small (< 10), so noise elimination should trigger
        x_out = gate(x)
        self.assertEqual(float(x_out[0, 3]), 0.0)

    def test_invalid_givens_params_raise_error(self) -> None:
        x = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        with self.assertRaises(EntropicGatingError):
            apply_givens_rotation_pair(x, i=0, j=5, theta=torch.tensor(0.0))
        with self.assertRaises(EntropicGatingError):
            eliminate_noise_components(x, n_noise_dims=2)


if __name__ == "__main__":
    unittest.main()
