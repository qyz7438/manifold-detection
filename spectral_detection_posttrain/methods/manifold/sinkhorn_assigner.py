r"""Differentiable Sinkhorn soft-assignment for prototype-based manifolds.

Given a cost matrix (e.g. squared Euclidean distances to prototypes), the
assigner returns a doubly-stochastic soft assignment matrix.  Row sums are
normalized to 1 (each sample is distributed over prototypes) and column sums
are normalized to ``batch_size / num_prototypes`` so that every prototype
receives equal total mass on average.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SinkhornAssigner(nn.Module):
    r"""Entropically regularized Sinkhorn assignment.

    Solves

    .. math::
        \min_Q \langle Q, C \rangle - \varepsilon H(Q)

    subject to

    .. math::
        Q \mathbf{1} = \mathbf{1}, \quad
        Q^\top \mathbf{1} = \frac{B}{K} \mathbf{1}

    where :math:`B` is the batch size and :math:`K` is the number of
    prototypes.

    Args:
        eps: entropic regularization strength (must be > 0).
        max_iter: number of Sinkhorn fixed-point iterations.
        stable: if ``True`` (recommended) use log-space iterations for
            numerical stability.
    """

    def __init__(self, eps: float = 0.05, max_iter: int = 50, stable: bool = True):
        super().__init__()
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        self.eps = eps
        self.max_iter = max_iter
        self.stable = stable

    def _sinkhorn_standard(self, cost: torch.Tensor) -> torch.Tensor:
        """Standard probability-space iteration."""
        batch_size, num_prototypes = cost.shape
        device = cost.device
        dtype = cost.dtype

        row_target = torch.ones(batch_size, device=device, dtype=dtype)
        col_target = torch.full(
            (num_prototypes,), batch_size / num_prototypes, device=device, dtype=dtype
        )

        K = torch.exp(-cost / self.eps)
        u = torch.ones_like(row_target)
        v = torch.ones_like(col_target)
        eps_stab = 1e-20

        for _ in range(self.max_iter):
            u = row_target / (K @ v + eps_stab)
            v = col_target / (K.T @ u + eps_stab)

        return u.unsqueeze(1) * K * v.unsqueeze(0)

    def _sinkhorn_log(self, cost: torch.Tensor) -> torch.Tensor:
        """Log-space iteration for better numerical stability."""
        batch_size, num_prototypes = cost.shape
        device = cost.device
        dtype = cost.dtype

        log_row_target = torch.zeros(batch_size, device=device, dtype=dtype)
        log_col_target = torch.full(
            (num_prototypes,),
            torch.log(torch.tensor(batch_size / num_prototypes, dtype=dtype, device=device)),
            device=device,
            dtype=dtype,
        )

        log_K = -cost / self.eps
        u = torch.zeros_like(log_row_target)
        v = torch.zeros_like(log_col_target)

        for _ in range(self.max_iter):
            u = log_row_target - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)
            v = log_col_target - torch.logsumexp(log_K + u.unsqueeze(1), dim=0)

        log_Q = u.unsqueeze(1) + log_K + v.unsqueeze(0)
        return torch.exp(log_Q)

    def forward(self, cost: torch.Tensor) -> torch.Tensor:
        """Compute soft assignment matrix.

        Args:
            cost: pairwise cost matrix of shape ``(B, K)``.

        Returns:
            Soft assignment matrix ``Q`` of shape ``(B, K)`` with row sums
            equal to 1 and column sums equal to ``B / K``.
        """
        if cost.ndim != 2:
            raise ValueError(f"cost must be 2D, got shape {cost.shape}")

        if self.stable:
            return self._sinkhorn_log(cost)
        return self._sinkhorn_standard(cost)

    def extra_repr(self) -> str:
        return f"eps={self.eps}, max_iter={self.max_iter}, stable={self.stable}"
