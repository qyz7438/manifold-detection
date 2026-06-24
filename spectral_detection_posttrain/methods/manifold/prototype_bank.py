r"""Online prototype bank with EMA updates for class-wise sub-manifolds.

The bank maintains a set of prototypes per class.  Prototypes are updated
online via soft assignments (e.g. from Sinkhorn) and an exponential moving
average.  They are stored as buffers, not parameters, so they are not
optimized directly by the outer optimizer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeBank(nn.Module):
    r"""Class-wise prototype sub-manifold bank.

    For each class :math:`c` we maintain :math:`K` prototypes
    :math:`p_{c,k} \in \mathbb{R}^D`.  The prototypes are updated with a
    soft-assignment EMA:

    .. math::
        S_{c,k} \leftarrow \lambda S_{c,k} + (1 - \lambda)
            \sum_{b: y_b=c} q_{b,k} z_b \\
        N_{c,k} \leftarrow \lambda N_{c,k} + (1 - \lambda)
            \sum_{b: y_b=c} q_{b,k} \\
        p_{c,k} = S_{c,k} / N_{c,k}

    Args:
        num_classes: number of object categories.
        num_prototypes_per_class: number of sub-manifold anchors per class.
        feature_dim: dimensionality of the input feature vectors.
        ema_decay: EMA decay :math:`\lambda` in :math:`[0, 1)`.
        init_scale: standard deviation for random prototype initialization.
    """

    def __init__(
        self,
        num_classes: int,
        num_prototypes_per_class: int,
        feature_dim: int,
        ema_decay: float = 0.99,
        init_scale: float = 0.1,
    ):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if num_prototypes_per_class <= 0:
            raise ValueError("num_prototypes_per_class must be positive")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")

        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.feature_dim = feature_dim
        self.ema_decay = ema_decay

        prototypes = torch.randn(
            num_classes, num_prototypes_per_class, feature_dim
        ) * init_scale
        self.register_buffer("prototypes", prototypes.clone())
        self.register_buffer("ema_sums", prototypes.clone())
        self.register_buffer(
            "ema_counts",
            torch.full(
                (num_classes, num_prototypes_per_class),
                1.0,
                dtype=prototypes.dtype,
                device=prototypes.device,
            ),
        )

    def compute_distances(
        self, features: torch.Tensor, class_ids: torch.Tensor
    ) -> torch.Tensor:
        """Squared Euclidean distance from each feature to its class prototypes.

        Args:
            features: tensor of shape ``(B, D)``.
            class_ids: integer tensor of shape ``(B,)`` with values in
                ``[0, num_classes)``.

        Returns:
            Distance matrix of shape ``(B, K)`` where ``K`` is the number of
            prototypes per class.
        """
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must have shape (B, {self.feature_dim}), got {features.shape}"
            )
        if class_ids.shape != features.shape[:1]:
            raise ValueError(
                f"class_ids shape {class_ids.shape} does not match features batch dim"
            )
        class_prototypes = self.prototypes[class_ids]  # (B, K, D)
        diff = features.unsqueeze(1) - class_prototypes  # (B, K, D)
        return (diff ** 2).sum(dim=-1)  # (B, K)

    def update(
        self,
        features: torch.Tensor,
        class_ids: torch.Tensor,
        assignments: torch.Tensor,
    ) -> None:
        """EMA-update prototypes using soft assignments.

        Args:
            features: tensor of shape ``(B, D)``.
            class_ids: integer tensor of shape ``(B,)``.
            assignments: soft weights of shape ``(B, K)``.  Typically row sums
                are close to 1 (each sample is distributed over prototypes).
        """
        if assignments.shape != (features.shape[0], self.num_prototypes_per_class):
            raise ValueError(
                f"assignments must have shape (B, {self.num_prototypes_per_class}), "
                f"got {assignments.shape}"
            )

        with torch.no_grad():
            for c in range(self.num_classes):
                mask = class_ids == c
                count = int(mask.sum().item())
                if count == 0:
                    continue

                feat_c = features[mask]          # (B_c, D)
                assign_c = assignments[mask]     # (B_c, K)

                # Weighted contributions per prototype: (K, D) and (K,)
                weighted_sum = torch.einsum("bk,bd->kd", assign_c, feat_c)
                weighted_count = assign_c.sum(dim=0)

                self.ema_sums[c] = (
                    self.ema_decay * self.ema_sums[c]
                    + (1.0 - self.ema_decay) * weighted_sum
                )
                self.ema_counts[c] = (
                    self.ema_decay * self.ema_counts[c]
                    + (1.0 - self.ema_decay) * weighted_count
                )

                denom = self.ema_counts[c].unsqueeze(-1).clamp_min(1e-12)
                self.prototypes[c] = self.ema_sums[c] / denom

    def initialize_from_centers(
        self, class_centers: torch.Tensor, noise_scale: float = 0.01
    ) -> None:
        """Warm-start prototypes around provided class centers.

        Useful for offline prototype warmup: run k-means/Sinkhorn on a frozen
        detector's box features to obtain initial centers, then spread
        ``num_prototypes_per_class`` prototypes around each center with small
        Gaussian noise.

        Args:
            class_centers: tensor of shape ``(num_classes, feature_dim)``.
            noise_scale: standard deviation of the additive noise.
        """
        if class_centers.shape != (self.num_classes, self.feature_dim):
            raise ValueError(
                f"class_centers must have shape ({self.num_classes}, {self.feature_dim}), "
                f"got {class_centers.shape}"
            )

        with torch.no_grad():
            base = class_centers.unsqueeze(1)  # (C, 1, D)
            noise = torch.randn_like(self.prototypes) * noise_scale
            init = base + noise
            self.prototypes.copy_(init)
            self.ema_sums.copy_(init * self.ema_counts.unsqueeze(-1))

    def get_prototypes(self, class_id: int | None = None) -> torch.Tensor:
        """Return prototypes for one class or all classes.

        Args:
            class_id: if ``None``, returns all prototypes; otherwise the class
                index.

        Returns:
            Prototype tensor of shape ``(num_classes, K, D)`` or ``(K, D)``.
        """
        if class_id is None:
            return self.prototypes
        return self.prototypes[class_id]

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"num_prototypes_per_class={self.num_prototypes_per_class}, "
            f"feature_dim={self.feature_dim}, ema_decay={self.ema_decay}"
        )
