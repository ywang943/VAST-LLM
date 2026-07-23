"""
SIGReg: Sketched Isotropic Gaussian Regularization.
Source: LeJEPA (rbalestr-lab/lejepa), adapted to be device-agnostic.

Pushes continuous embeddings toward an isotropic Gaussian distribution.
Applied to z_cont BEFORE VQ quantization.
"""

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization.

    Uses random projections (slices) to test whether z_cont matches
    a standard Gaussian in every 1D projected direction.
    Uses the Epps-Pulley characteristic-function statistic.

    Args:
        knots: number of quadrature points for the integral (17 is from the paper)
        n_slices: number of random projection directions (256 is from the paper)
    """

    def __init__(self, knots: int = 17, n_slices: int = 256):
        super().__init__()
        self.n_slices = n_slices

        # Quadrature grid t ∈ [0, 3] (where Gaussian CF is numerically significant)
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)

        # Trapezoidal weights, window = exp(-t²/2)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = torch.exp(-t.square() / 2.0)       # target CF: real part of N(0,1)

        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: embeddings (..., D) — works for (B,D), (B,L,D), etc.
        Returns:
            scalar SIGReg loss
        """
        z = z.reshape(-1, z.shape[-1])   # (N, D)
        z = z - z.mean(0, keepdim=True)  # center

        # Random projection matrix: (D, n_slices), columns normalized
        A = torch.randn(z.size(-1), self.n_slices, device=z.device, dtype=z.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True)

        # Project: (N, n_slices); then add quadrature dim: (N, n_slices, knots)
        proj = z @ A                                      # (N, n_slices)
        x_t = proj.unsqueeze(-1) * self.t                # (N, n_slices, knots)

        # Empirical CF vs N(0,1) CF = exp(-t²/2) (real) + 0j (imaginary)
        err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()

        # Weighted integral over t, averaged over slices; scale by N for consistency
        statistic = (err @ self.weights) * z.size(0)
        return statistic.mean()
