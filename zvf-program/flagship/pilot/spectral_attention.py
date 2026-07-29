from __future__ import annotations

import math
import torch
import torch.nn as nn


class SpectralAttentionError(RuntimeError):
    """Raised when Legendre spectral attention operations violate invariants."""


def legendre_grid(
    seq_len: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Map discrete sequence indices l in {0, ..., seq_len - 1} to t in [-1, 1]."""
    if seq_len <= 0:
        raise SpectralAttentionError("seq_len must be positive")
    if seq_len == 1:
        return torch.tensor([0.0], device=device, dtype=dtype or torch.float32)
    return torch.linspace(-1.0, 1.0, steps=seq_len, device=device, dtype=dtype or torch.float32)


def compute_legendre_polynomials(n_modes: int, t: torch.Tensor) -> torch.Tensor:
    """Evaluate Legendre polynomials P_0(t), ..., P_{n_modes-1}(t) using Rodrigues recurrence.

    Returns tensor of shape (*t.shape, n_modes).
    """
    if n_modes <= 0:
        raise SpectralAttentionError("n_modes must be positive")
    if not torch.isfinite(t).all():
        raise SpectralAttentionError("t tensor contains non-finite values")

    p0 = torch.ones_like(t)
    if n_modes == 1:
        return p0.unsqueeze(-1)
    p1 = t.clone()
    polys = [p0, p1]
    for n in range(1, n_modes - 1):
        pn_plus_1 = ((2 * n + 1) * t * polys[n] - n * polys[n - 1]) / (n + 1)
        polys.append(pn_plus_1)
    return torch.stack(polys[:n_modes], dim=-1)


def legendre_basis(
    seq_len: int,
    n_modes: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute the Legendre polynomial basis matrix of shape [seq_len, n_modes]."""
    t = legendre_grid(seq_len, device=device, dtype=dtype)
    return compute_legendre_polynomials(n_modes, t)


def legendre_spectral_projection(
    x: torch.Tensor,
    n_modes: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project sequence tensor x onto truncated Legendre polynomial basis.

    x: shape [B, L, D] or [L, D]
    mask: completion mask of shape [B, L] or [L]
    returns: spectral coefficients c of shape [B, n_modes, D] (or [n_modes, D])
    """
    if x.ndim not in {2, 3}:
        raise SpectralAttentionError(f"x must be 2D or 3D tensor, got ndim={x.ndim}")
    if not torch.isfinite(x).all():
        raise SpectralAttentionError("input tensor x contains non-finite values")

    is_2d = x.ndim == 2
    if is_2d:
        x_3d = x.unsqueeze(0)
        mask_2d = mask.unsqueeze(0) if mask is not None else None
    else:
        x_3d = x
        mask_2d = mask

    batch_size, seq_len, d_model = x_3d.shape
    basis = legendre_basis(seq_len, n_modes, device=x_3d.device, dtype=x_3d.dtype)  # [L, N]

    n_idx = torch.arange(n_modes, device=x_3d.device, dtype=x_3d.dtype)
    scale_n = 2 * n_idx + 1  # [N]

    if mask_2d is not None:
        if mask_2d.shape != (batch_size, seq_len):
            raise SpectralAttentionError("mask shape must match x batch and sequence dimensions")
        x_eff = x_3d * mask_2d.unsqueeze(-1)
        l_eff = mask_2d.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, 1]
    else:
        x_eff = x_3d
        l_eff = torch.full((batch_size, 1), float(seq_len), device=x_3d.device, dtype=x_3d.dtype)

    proj = torch.einsum("bld,ln->bnd", x_eff, basis)
    scale_factor = scale_n.unsqueeze(0).unsqueeze(-1) / l_eff.unsqueeze(-1)  # [B, N, 1]
    c = proj * scale_factor

    if is_2d:
        return c.squeeze(0)
    return c


def spectral_pairwise_distance(c_i: torch.Tensor, c_j: torch.Tensor) -> torch.Tensor:
    """Compute continuous L^2 weighted distance between two spectral coefficient sets.

    d_spec(y_i, y_j) = sum_{n=0}^{N-1} (2 / (2n + 1)) ||c_{i,n} - c_{j,n}||_2^2
    """
    if c_i.shape != c_j.shape:
        raise SpectralAttentionError("spectral coefficient shapes must match")
    if c_i.ndim < 2:
        raise SpectralAttentionError("spectral coefficients must have at least 2 dimensions [..., N, D]")

    n_modes = c_i.shape[-2]
    n_idx = torch.arange(n_modes, device=c_i.device, dtype=c_i.dtype)
    weights = 2.0 / (2.0 * n_idx + 1.0)  # [N]

    diff_sq = (c_i - c_j).pow(2).sum(dim=-1)  # [..., N]
    dist = (diff_sq * weights).sum(dim=-1)
    return dist


class LegendreSpectralRouting(nn.Module):
    """Hierarchical spectral routing module using Legendre polynomial mode splitting."""

    def __init__(self, d_model: int, n_modes: int = 8, n_cut: int | None = None) -> None:
        super().__init__()
        if d_model <= 0:
            raise SpectralAttentionError("d_model must be positive")
        if n_modes <= 0:
            raise SpectralAttentionError("n_modes must be positive")

        self.d_model = d_model
        self.n_modes = n_modes
        self.n_cut = n_cut if n_cut is not None else max(1, n_modes // 2)
        if self.n_cut <= 0 or self.n_cut > n_modes:
            raise SpectralAttentionError("n_cut must be in [1, n_modes]")

        self.w_low = nn.Linear(d_model, d_model, bias=False)
        self.w_high = nn.Linear(d_model, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise SpectralAttentionError(f"LegendreSpectralRouting expects 3D tensor [B, L, D], got ndim={x.ndim}")
        c = legendre_spectral_projection(x, self.n_modes, mask=mask)  # [B, N, D]

        seq_len = x.shape[1]
        basis = legendre_basis(seq_len, self.n_modes, device=x.device, dtype=x.dtype)  # [L, N]

        c_low = c[:, : self.n_cut, :]
        c_high = c[:, self.n_cut :, :]
        p_low = basis[:, : self.n_cut]
        p_high = basis[:, self.n_cut :]

        x_low = torch.einsum("bnd,ln->bld", c_low, p_low)
        x_high = torch.einsum("bnd,ln->bld", c_high, p_high)

        s_route = self.act(self.w_low(x_low) + self.w_high(x_high))
        return s_route
