r"""Differentiable Sinkhorn optimal transport distance.

Implements the entropically regularized Sinkhorn algorithm in pure PyTorch
so that it can be used as a loss term inside larger training loops.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SinkhornOT(nn.Module):
    r"""Entropic optimal transport distance via Sinkhorn iterations.

    Args:
        eps: entropic regularization strength (must be > 0).
        max_iter: number of Sinkhorn fixed-point iterations.
        p: power used to build pairwise cost matrices (``|x_i - y_j| ** p``).
        stable: if ``True``, perform the division in log-space for numerical
            stability when ``eps`` is very small.
    """

    def __init__(self, eps: float = 0.01, max_iter: int = 100, p: int = 2, stable: bool = False):
        super().__init__()
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        self.eps = eps
        self.max_iter = max_iter
        self.p = p
        self.stable = stable

    @staticmethod
    def pairwise_cost(x: torch.Tensor, y: torch.Tensor, p: int = 2) -> torch.Tensor:
        """Build the pairwise cost matrix ``C[i, j] = |x_i - y_j| ** p``.

        Args:
            x: tensor of shape ``(n,)``.
            y: tensor of shape ``(m,)``.
            p: power of the Euclidean distance.

        Returns:
            Cost matrix of shape ``(n, m)``.
        """
        return (x.unsqueeze(1) - y.unsqueeze(0)).abs().pow(p)

    def _sinkhorn_standard(
        self, mu: torch.Tensor, nu: torch.Tensor, K: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        """Standard Sinkhorn iteration in probability space."""
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)
        eps_stab = 1e-20

        for _ in range(self.max_iter):
            u = mu / (K @ v + eps_stab)
            v = nu / (K.T @ u + eps_stab)

        pi = u.unsqueeze(1) * K * v.unsqueeze(0)
        return (pi * C).sum()

    def _sinkhorn_log(
        self, mu: torch.Tensor, nu: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        """Log-space Sinkhorn iteration for improved numerical stability."""
        log_mu = torch.log(mu + 1e-20)
        log_nu = torch.log(nu + 1e-20)
        log_K = -C / self.eps

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        for _ in range(self.max_iter):
            u = log_mu - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)
            v = log_nu - torch.logsumexp(log_K + u.unsqueeze(1), dim=0)

        log_pi = u.unsqueeze(1) + log_K + v.unsqueeze(0)
        pi = torch.exp(log_pi)
        return (pi * C).sum()

    def forward(
        self,
        mu: torch.Tensor,
        nu: torch.Tensor,
        cost_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Sinkhorn distance between ``mu`` and ``nu``.

        Args:
            mu: source probability vector of shape ``(n,)``. Must sum to 1.
            nu: target probability vector of shape ``(m,)``. Must sum to 1.
            cost_matrix: pairwise cost matrix of shape ``(n, m)``.

        Returns:
            Scalar Sinkhorn distance with gradients w.r.t. ``cost_matrix``.
        """
        if not torch.allclose(mu.sum(), torch.tensor(1.0, dtype=mu.dtype, device=mu.device), atol=1e-5):
            raise ValueError("mu must be a probability distribution")
        if not torch.allclose(nu.sum(), torch.tensor(1.0, dtype=nu.dtype, device=nu.device), atol=1e-5):
            raise ValueError("nu must be a probability distribution")

        C = cost_matrix
        if self.stable:
            return self._sinkhorn_log(mu, nu, C)

        K = torch.exp(-C / self.eps)
        return self._sinkhorn_standard(mu, nu, K, C)
