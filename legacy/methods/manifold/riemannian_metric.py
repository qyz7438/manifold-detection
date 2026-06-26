r"""Adaptive Riemannian metric on the complex latent manifold.

The metric is parameterized as :math:`M(z) = U(z)^{\top} U(z) + \epsilon I`
so that it is symmetric positive definite for every latent point :math:`z`.
At initialization the network producing :math:`U(z)` is zero, so the metric
reduces to a scaled Euclidean metric.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveRiemannianMetric(nn.Module):
    r"""Learnable local metric on a complex latent space.

    Args:
        latent_dim: dimensionality of the latent manifold.
        eps: small positive constant ensuring positive definiteness.
    """

    def __init__(self, latent_dim: int, eps: float = 1e-4):
        super().__init__()
        self.latent_dim = latent_dim
        self.eps = eps

        # Operate on the real concatenation [Re(z), Im(z)].
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * latent_dim),
        )
        # Initialize to zero so that U(z)=0 and M(z)=eps*I.
        with torch.no_grad():
            nn.init.zeros_(self.net[0].weight)
            nn.init.zeros_(self.net[0].bias)

    def metric(self, z: torch.Tensor) -> torch.Tensor:
        r"""Compute the metric matrix :math:`M(z)`.

        Args:
            z: complex tensor of shape ``(..., latent_dim)``.

        Returns:
            Real symmetric positive definite tensor of shape
            ``(..., latent_dim, latent_dim)``.
        """
        if not torch.is_complex(z):
            raise ValueError("AdaptiveRiemannianMetric expects a complex-valued input")

        # Concatenate real and imaginary parts: (... , 2*latent_dim).
        real_input = torch.cat([z.real, z.imag], dim=-1)
        U = self.net(real_input)
        U = U.reshape(*z.shape[:-1], self.latent_dim, self.latent_dim)

        # M = U^T U + eps * I is symmetric positive definite.
        M = torch.matmul(U.transpose(-2, -1), U)
        eye = torch.eye(self.latent_dim, dtype=M.dtype, device=M.device)
        M = M + self.eps * eye
        return M

    def local_distance(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        r"""Compute the squared Riemannian distance between two latent points.

        Implements

        .. math::
            d(z_1, z_2)^2 = (z_1 - z_2)^{*} \, M(z) \, (z_1 - z_2),

        where :math:`M(z)` is evaluated at the midpoint
        :math:`z = (z_1 + z_2) / 2`.

        Args:
            z1, z2: complex tensors of shape ``(..., latent_dim)``.

        Returns:
            Real tensor of shape ``(...)`` containing the squared distances.
        """
        if not (torch.is_complex(z1) and torch.is_complex(z2)):
            raise ValueError("local_distance expects complex-valued inputs")
        diff = z1 - z2
        z_mid = 0.5 * (z1 + z2)
        M = self.metric(z_mid).to(diff.dtype)

        # diff^* M diff, keeping arbitrary leading batch dimensions.
        diff_col = diff.unsqueeze(-1)              # (..., latent_dim, 1)
        diff_row = diff.conj().unsqueeze(-2)       # (..., 1, latent_dim)
        energy = torch.matmul(diff_row, torch.matmul(M, diff_col))
        return energy.real.squeeze(-1).squeeze(-1)
