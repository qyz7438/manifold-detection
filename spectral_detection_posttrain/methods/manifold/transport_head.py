r"""ChordEdit-style low-energy transport head on a prototype manifold.

Given a feature :math:`z` and its squared distances to the class prototypes,
the head predicts a transport field :math:`u(z)` that moves :math:`z` toward
the prototype sub-manifold.  The field is a prototype-weighted sum of
per-prototype residuals, and its energy is regularized to be small and smooth.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransportHead(nn.Module):
    r"""Predict a low-energy transport field in feature space.

    The transport is defined as

    .. math::
        u(z) = \sum_{k=1}^K w_k(z) \, \phi_k(z)

    where :math:`w_k(z) = \text{softmax}_k(-d_k(z) / \tau)` are weights from
    the prototype distances and :math:`\phi_k(z)` are per-prototype residual
    vectors produced by a shared MLP.

    Args:
        feature_dim: dimensionality of the input feature :math:`z`.
        num_prototypes: number of prototypes per class.
        hidden_dim: hidden width of the residual MLP. Defaults to
            ``feature_dim``.
        tau: temperature for the distance-based softmax weights.
        residual_scale: initialize the final MLP layer with this small std so
            that :math:`u(z) \approx 0` at the start of training.
    """

    def __init__(
        self,
        feature_dim: int,
        num_prototypes: int,
        hidden_dim: int | None = None,
        tau: float = 0.1,
        residual_scale: float = 1e-3,
    ):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if num_prototypes <= 0:
            raise ValueError("num_prototypes must be positive")
        if tau <= 0.0:
            raise ValueError("tau must be positive")

        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
        self.hidden_dim = hidden_dim or feature_dim
        self.tau = tau

        # Shared MLP predicts all K residual vectors at once: (B, K * D).
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, num_prototypes * feature_dim),
        )

        # Initialize the last layer near zero so u(z) starts small.
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=residual_scale)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self, features: torch.Tensor, squared_distances: torch.Tensor
    ) -> torch.Tensor:
        """Predict the transport field.

        Args:
            features: tensor of shape ``(B, D)``.
            squared_distances: squared distances to prototypes, shape
                ``(B, K)``.

        Returns:
            Transport field ``u(z)`` of shape ``(B, D)``.
        """
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must have shape (B, {self.feature_dim}), got {features.shape}"
            )
        if squared_distances.shape != (features.shape[0], self.num_prototypes):
            raise ValueError(
                f"squared_distances must have shape (B, {self.num_prototypes}), "
                f"got {squared_distances.shape}"
            )

        weights = F.softmax(-squared_distances / self.tau, dim=-1)  # (B, K)
        residuals = self.mlp(features).view(
            -1, self.num_prototypes, self.feature_dim
        )  # (B, K, D)

        transport = torch.einsum("bk,bkd->bd", weights, residuals)  # (B, D)
        return transport

    def transport_energy(
        self,
        features: torch.Tensor,
        squared_distances: torch.Tensor,
        smoothness_weight: float = 0.0,
    ) -> torch.Tensor:
        """Compute the squared :math:`L^2` energy of the transport field.

        Optionally adds a Hutchinson-estimated squared Frobenius norm of the
        Jacobian :math:`\nabla_z u(z)` for smoothness regularization.

        Args:
            features: tensor of shape ``(B, D)``. Requires ``requires_grad=True``
                if ``smoothness_weight > 0``.
            squared_distances: squared distances to prototypes, shape ``(B, K)``.
            smoothness_weight: coefficient :math:`\alpha` for the Jacobian
                smoothness term. Set to 0 to disable.

        Returns:
            Scalar energy tensor.
        """
        if smoothness_weight < 0.0:
            raise ValueError("smoothness_weight must be non-negative")

        transport = self.forward(features, squared_distances)
        energy = (transport ** 2).sum(dim=-1).mean()

        if smoothness_weight > 0.0:
            if not features.requires_grad:
                raise ValueError(
                    "features must require gradients to compute the smoothness penalty"
                )
            energy = energy + smoothness_weight * self._hutchinson_jacobian_norm(
                features, squared_distances
            )

        return energy

    def _hutchinson_jacobian_norm(
        self, features: torch.Tensor, squared_distances: torch.Tensor
    ) -> torch.Tensor:
        r"""Unbiased single-sample Hutchinson estimate of
        :math:`\|\nabla_z u(z)\|_F^2`.

        Uses the identity

        .. math::
            \mathbb{E}_{v \sim \mathcal{N}(0, I)}
            \| \nabla_z (v^{\top} u(z)) \|^2
            = \| \nabla_z u(z) \|_F^2.
        """
        transport = self.forward(features, squared_distances)  # (B, D)
        v = torch.randn_like(transport)
        vtu = (v * transport).sum()
        grad_vtu = torch.autograd.grad(
            vtu, features, create_graph=True, retain_graph=True
        )[0]
        return (grad_vtu ** 2).sum(dim=-1).mean()

    def extra_repr(self) -> str:
        return (
            f"feature_dim={self.feature_dim}, num_prototypes={self.num_prototypes}, "
            f"hidden_dim={self.hidden_dim}, tau={self.tau}"
        )
