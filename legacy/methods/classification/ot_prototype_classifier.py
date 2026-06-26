"""OT-based prototype classifier using Sinkhorn distances."""

from __future__ import annotations

import torch
import torch.nn as nn

from spectral_detection_posttrain.methods.manifold.sinkhorn_ot import SinkhornOT


class OTPrototypeClassifier(nn.Module):
    """Prototype classifier with Sinkhorn optimal-transport distances.

    Each class is represented by a set of ``n_prototypes`` points in feature
    space.  For a sample feature vector the distance to a class is the entropic
    optimal-transport distance between the sample point mass and the uniformly
    weighted prototype points.  Classification logits are the negative
    distances.

    Args:
        feature_dim: dimensionality of the input feature vectors.
        num_classes: number of classes.
        n_prototypes: number of prototype points per class.
        eps: entropic regularization for Sinkhorn iterations.
        max_iter: number of Sinkhorn fixed-point iterations.
        p: power of the Euclidean distance used in the cost matrix.
        momentum: EMA momentum used by :meth:`update_prototypes`.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        n_prototypes: int = 4,
        eps: float = 0.05,
        max_iter: int = 50,
        p: int = 2,
        momentum: float = 0.9,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.n_prototypes = n_prototypes
        self.p = p
        self.momentum = momentum

        self.prototypes = nn.Parameter(
            torch.randn(num_classes, n_prototypes, feature_dim)
            / (feature_dim ** 0.5)
        )
        self.sinkhorn = SinkhornOT(eps=eps, max_iter=max_iter, p=p, stable=True)

    def update_prototypes(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> None:
        """Online EMA update of prototypes via nearest-prototype assignment.

        For each class, the provided features are assigned to their nearest
        prototype and each prototype is moved toward the mean of its assigned
        features with momentum ``1 - momentum``.

        Args:
            features: tensor of shape ``(N, feature_dim)``.
            labels: tensor of shape ``(N,)`` containing class indices.
        """
        if features.dim() != 2:
            raise ValueError("features must be a 2D tensor")
        if labels.dim() != 1:
            raise ValueError("labels must be a 1D tensor")
        if features.size(0) != labels.size(0):
            raise ValueError("features and labels must have the same batch size")

        with torch.no_grad():
            for c in range(self.num_classes):
                mask = labels == c
                if not mask.any():
                    continue
                cls_feats = features[mask]
                dists = torch.cdist(cls_feats, self.prototypes[c], p=self.p)
                assignments = dists.argmin(dim=1)
                for k in range(self.n_prototypes):
                    assigned = cls_feats[assignments == k]
                    if assigned.numel() == 0:
                        continue
                    self.prototypes.data[c, k] = (
                        self.momentum * self.prototypes.data[c, k]
                        + (1 - self.momentum) * assigned.mean(dim=0)
                    )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute classification logits for a batch of features.

        Args:
            features: tensor of shape ``(B, feature_dim)``.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        if features.dim() != 2:
            raise ValueError("features must be a 2D tensor")

        batch_size = features.size(0)
        logits = features.new_empty(batch_size, self.num_classes)
        for i in range(batch_size):
            for c in range(self.num_classes):
                logits[i, c] = -self._distance_to_class(features[i], c)
        return logits

    def _distance_to_class(self, feature: torch.Tensor, c: int) -> torch.Tensor:
        """Sinkhorn distance between a sample and class ``c`` prototypes."""
        prototypes = self.prototypes[c]  # (K, D)
        cost = (
            torch.cdist(feature.unsqueeze(0), prototypes, p=self.p)
            .squeeze(0)
            .pow(self.p)
        )  # (K,)

        mu = torch.ones(1, device=feature.device, dtype=feature.dtype)
        nu = (
            torch.ones(self.n_prototypes, device=feature.device, dtype=feature.dtype)
            / self.n_prototypes
        )
        return self.sinkhorn(mu, nu, cost.unsqueeze(0))
