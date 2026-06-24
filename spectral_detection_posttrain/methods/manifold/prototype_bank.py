r"""Online prototype bank with EMA updates for class-wise sub-manifolds.

The bank maintains a set of prototypes per class.  Prototypes are updated
online via soft assignments (e.g. from Sinkhorn) and an exponential moving
average.  They are stored as buffers, not parameters, so they are not
optimized directly by the outer optimizer.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_frequency_weights(
    counts: torch.Tensor,
    mode: Literal["none", "inv_sqrt", "effective_num"] = "inv_sqrt",
    beta: float = 0.999,
) -> torch.Tensor:
    """Compute per-class reweighting factors from sample counts.

    Args:
        counts: integer tensor of shape ``(num_classes,)``.  Background (index 0)
            is kept at weight 1.0.
        mode:
            - ``none``: returns all ones.
            - ``inv_sqrt``: ``1 / sqrt(count)``.
            - ``effective_num``: ``(1 - beta) / (1 - beta^count)``.
        beta: smoothing parameter for ``effective_num`` mode.

    Returns:
        Float tensor of shape ``(num_classes,)`` with weights >= 0.
    """
    counts = counts.float().clamp_min(1.0)
    if mode == "none":
        weights = torch.ones_like(counts)
    elif mode == "inv_sqrt":
        weights = 1.0 / torch.sqrt(counts)
    elif mode == "effective_num":
        weights = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
    else:
        raise ValueError(f"Unknown class reweight mode: {mode}")

    # Background class should not be up-weighted by the reweighting scheme.
    weights[0] = 1.0
    return weights


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
        class_weights: torch.Tensor | None = None,
    ) -> None:
        """EMA-update prototypes using soft assignments.

        Args:
            features: tensor of shape ``(B, D)``.
            class_ids: integer tensor of shape ``(B,)``.
            assignments: soft weights of shape ``(B, K)``.  Typically row sums
                are close to 1 (each sample is distributed over prototypes).
            class_weights: optional per-sample weights of shape ``(B,)`` used to
                up-weight rare classes during EMA updates.
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

                if class_weights is not None:
                    w = class_weights[mask].unsqueeze(-1)  # (B_c, 1)
                    weighted_sum = torch.einsum("bk,bd->kd", assign_c * w, feat_c)
                    weighted_count = (assign_c * w).sum(dim=0)
                else:
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


class RemoteSensingPrototypeBank(PrototypeBank):
    r"""Class-wise prototype bank with explicit orientation/scale bins.

    Remote-sensing objects (e.g. NWPU VHR-10) live on a product manifold
    ``SO(2) x scale x aspect-ratio``.  This bank maintains separate sub-manifold
    anchors for discrete orientation and scale bins derived from each bounding
    box, so that an airplane at 0 degrees and an airplane at 90 degrees are not
    forced to collapse to the same prototype.

    Args:
        num_classes: number of object categories.
        num_prototypes_per_class: number of sub-manifold anchors per
            (class, orient, scale) cell.
        feature_dim: dimensionality of the input feature vectors.
        n_orient_bins: number of orientation bins.  ``1`` disables orientation
            conditioning and falls back to class-level prototypes.
        n_scale_bins: number of scale bins.  ``1`` disables scale conditioning.
        ema_decay: EMA decay :math:`\lambda` in ``[0, 1)``.
        init_scale: standard deviation for random prototype initialization.
        orient_min_bin_size: minimum number of samples required to use the
            orientation-specific prototype cell.  Below this the cell falls back
            to the class-level average.
        scale_min_bin_size: analogous for scale.
    """

    def __init__(
        self,
        num_classes: int,
        num_prototypes_per_class: int,
        feature_dim: int,
        n_orient_bins: int = 4,
        n_scale_bins: int = 4,
        ema_decay: float = 0.99,
        init_scale: float = 0.1,
        orient_min_bin_size: int = 2,
        scale_min_bin_size: int = 2,
    ):
        # Bypass PrototypeBank.__init__ because we need a 5-D prototype tensor.
        nn.Module.__init__(self)
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if num_prototypes_per_class <= 0:
            raise ValueError("num_prototypes_per_class must be positive")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if n_orient_bins < 1 or n_scale_bins < 1:
            raise ValueError("n_orient_bins and n_scale_bins must be >= 1")

        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.feature_dim = feature_dim
        self.n_orient_bins = n_orient_bins
        self.n_scale_bins = n_scale_bins
        self.ema_decay = ema_decay
        self.orient_min_bin_size = max(1, orient_min_bin_size)
        self.scale_min_bin_size = max(1, scale_min_bin_size)

        prototypes = torch.randn(
            num_classes,
            n_orient_bins,
            n_scale_bins,
            num_prototypes_per_class,
            feature_dim,
        ) * init_scale
        self.register_buffer("prototypes", prototypes.clone())
        self.register_buffer("ema_sums", prototypes.clone())
        self.register_buffer(
            "ema_counts",
            torch.full(
                (num_classes, n_orient_bins, n_scale_bins, num_prototypes_per_class),
                1.0,
                dtype=prototypes.dtype,
                device=prototypes.device,
            ),
        )
        # Track how many samples each cell has seen for fallback decisions.
        self.register_buffer(
            "sample_counts",
            torch.zeros(
                num_classes,
                n_orient_bins,
                n_scale_bins,
                dtype=torch.long,
                device=prototypes.device,
            ),
        )

    def compute_distances(
        self,
        features: torch.Tensor,
        class_ids: torch.Tensor,
        orient_idx: torch.Tensor | None = None,
        scale_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Squared Euclidean distance to the (class, orient, scale) prototypes.

        Args:
            features: tensor of shape ``(B, D)``.
            class_ids: integer tensor of shape ``(B,)``.
            orient_idx: optional integer tensor of shape ``(B,)`` in
                ``[0, n_orient_bins)``.
            scale_idx: optional integer tensor of shape ``(B,)`` in
                ``[0, n_scale_bins)``.

        Returns:
            Distance matrix of shape ``(B, K)``.
        """
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must have shape (B, {self.feature_dim}), got {features.shape}"
            )
        if class_ids.shape != features.shape[:1]:
            raise ValueError(
                f"class_ids shape {class_ids.shape} does not match features batch dim"
            )

        orient_idx, scale_idx = self._resolve_transform_indices(
            features, class_ids, orient_idx, scale_idx
        )
        class_prototypes = self.prototypes[class_ids, orient_idx, scale_idx]  # (B, K, D)
        diff = features.unsqueeze(1) - class_prototypes  # (B, K, D)
        return (diff ** 2).sum(dim=-1)  # (B, K)

    def update(
        self,
        features: torch.Tensor,
        class_ids: torch.Tensor,
        assignments: torch.Tensor,
        orient_idx: torch.Tensor | None = None,
        scale_idx: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        """EMA-update prototypes using soft assignments and optional class weights.

        Args:
            features: tensor of shape ``(B, D)``.
            class_ids: integer tensor of shape ``(B,)``.
            assignments: soft weights of shape ``(B, K)``.
            orient_idx: optional integer tensor of shape ``(B,)``.
            scale_idx: optional integer tensor of shape ``(B,)``.
            class_weights: optional per-sample weights of shape ``(B,)``.
        """
        if assignments.shape != (features.shape[0], self.num_prototypes_per_class):
            raise ValueError(
                f"assignments must have shape (B, {self.num_prototypes_per_class}), "
                f"got {assignments.shape}"
            )

        orient_idx, scale_idx = self._resolve_transform_indices(
            features, class_ids, orient_idx, scale_idx
        )

        with torch.no_grad():
            for c in range(self.num_classes):
                class_mask = class_ids == c
                if not class_mask.any():
                    continue

                # Optional class reweighting.
                sample_weights = None
                if class_weights is not None:
                    sample_weights = class_weights[class_mask]

                feat_c = features[class_mask]          # (B_c, D)
                assign_c = assignments[class_mask]     # (B_c, K)
                orient_c = orient_idx[class_mask]      # (B_c,)
                scale_c = scale_idx[class_mask]        # (B_c,)

                bin_pairs = torch.stack([orient_c, scale_c], dim=1).unique(dim=0)
                for pair in bin_pairs.tolist():
                    o, s = pair
                    bin_mask = (orient_c == o) & (scale_c == s)
                    if not bin_mask.any():
                        continue
                    feat_bin = feat_c[bin_mask]
                    assign_bin = assign_c[bin_mask]
                    if sample_weights is not None:
                        w = sample_weights[bin_mask].unsqueeze(-1)
                        weighted_sum = torch.einsum("bk,bd->kd", assign_bin * w, feat_bin)
                        weighted_count = (assign_bin * w).sum(dim=0)
                    else:
                        weighted_sum = torch.einsum("bk,bd->kd", assign_bin, feat_bin)
                        weighted_count = assign_bin.sum(dim=0)

                    self.ema_sums[c, o, s] = (
                        self.ema_decay * self.ema_sums[c, o, s]
                        + (1.0 - self.ema_decay) * weighted_sum
                    )
                    self.ema_counts[c, o, s] = (
                        self.ema_decay * self.ema_counts[c, o, s]
                        + (1.0 - self.ema_decay) * weighted_count
                    )

                    denom = self.ema_counts[c, o, s].unsqueeze(-1).clamp_min(1e-12)
                    self.prototypes[c, o, s] = self.ema_sums[c, o, s] / denom
                    self.sample_counts[c, o, s] += int(bin_mask.sum().item())

    def initialize_from_centers(
        self, class_centers: torch.Tensor, noise_scale: float = 0.01
    ) -> None:
        """Warm-start all (class, orient, scale) cells around class centers."""
        if class_centers.shape != (self.num_classes, self.feature_dim):
            raise ValueError(
                f"class_centers must have shape ({self.num_classes}, {self.feature_dim}), "
                f"got {class_centers.shape}"
            )

        with torch.no_grad():
            base = class_centers[:, None, None, None, :]  # (C, 1, 1, 1, D)
            noise = torch.randn_like(self.prototypes) * noise_scale
            init = base + noise
            self.prototypes.copy_(init)
            self.ema_sums.copy_(init * self.ema_counts.unsqueeze(-1))

    def get_prototypes(
        self,
        class_id: int | None = None,
        orient_idx: int | None = None,
        scale_idx: int | None = None,
    ) -> torch.Tensor:
        """Return prototypes for one or all (class, orient, scale) cells."""
        if class_id is None:
            return self.prototypes
        if orient_idx is None and scale_idx is None:
            return self.prototypes[class_id]
        if orient_idx is not None and scale_idx is not None:
            return self.prototypes[class_id, orient_idx, scale_idx]
        raise ValueError("orient_idx and scale_idx must be provided together")

    @staticmethod
    def orient_idx_from_boxes(
        boxes: torch.Tensor, n_bins: int, degrees: bool = False
    ) -> torch.Tensor:
        """Map axis-aligned boxes to orientation bins via aspect-ratio angle.

        Args:
            boxes: ``(B, 4)`` tensor of ``(x1, y1, x2, y2)``.
            n_bins: number of orientation bins.  ``1`` returns all zeros.
            degrees: if True return angles in degrees (not used for binning).

        Returns:
            Integer tensor of shape ``(B,)`` with values in ``[0, n_bins)``.
        """
        if n_bins <= 1:
            return torch.zeros(boxes.shape[0], dtype=torch.long, device=boxes.device)
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        # Angle in [0, pi/2] because boxes are axis-aligned.
        angle = torch.atan2(h.clamp_min(1e-6), w.clamp_min(1e-6))
        # Map [0, pi/2] -> [0, n_bins)
        idx = (angle / (math.pi / 2.0) * n_bins).long().clamp_min(0).clamp_max(n_bins - 1)
        return idx

    @staticmethod
    def scale_idx_from_boxes(
        boxes: torch.Tensor,
        n_bins: int,
        log_scale: bool = True,
        min_area: float = 1.0,
    ) -> torch.Tensor:
        """Map boxes to scale bins via sqrt(area).

        Args:
            boxes: ``(B, 4)`` tensor.
            n_bins: number of scale bins.  ``1`` returns all zeros.
            log_scale: if True use log(area) for binning (more stable across
                large scale ranges).
            min_area: clamp area before log to avoid negative infinities.

        Returns:
            Integer tensor of shape ``(B,)`` with values in ``[0, n_bins)``.
        """
        if n_bins <= 1:
            return torch.zeros(boxes.shape[0], dtype=torch.long, device=boxes.device)
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        area = (w * h).clamp_min(min_area)
        scale = torch.sqrt(area)
        if log_scale:
            scale = torch.log(scale)
        # Robust linear binning over the observed batch range.
        scale_min = scale.min()
        scale_max = scale.max()
        if scale_max - scale_min < 1e-6:
            return torch.zeros(boxes.shape[0], dtype=torch.long, device=boxes.device)
        norm = (scale - scale_min) / (scale_max - scale_min + 1e-12)
        idx = (norm * n_bins).long().clamp_min(0).clamp_max(n_bins - 1)
        return idx

    def _resolve_transform_indices(
        self,
        features: torch.Tensor,
        class_ids: torch.Tensor,
        orient_idx: torch.Tensor | None,
        scale_idx: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return valid orient/scale indices, falling back to class-level cells."""
        device = features.device
        if orient_idx is None:
            orient_idx = torch.zeros(class_ids.shape[0], dtype=torch.long, device=device)
        if scale_idx is None:
            scale_idx = torch.zeros(class_ids.shape[0], dtype=torch.long, device=device)

        # Count how many samples in the current batch fall into each cell.
        batch_counts = torch.zeros(
            self.num_classes, self.n_orient_bins, self.n_scale_bins,
            dtype=torch.long, device=device
        )
        for b in range(class_ids.shape[0]):
            c = int(class_ids[b].item())
            o = int(orient_idx[b].item())
            s = int(scale_idx[b].item())
            batch_counts[c, o, s] += 1

        # If a requested cell is still empty, fall back to the most populated
        # (class, orient, scale) cell for that class to avoid random prototypes.
        fallback_orient = torch.zeros(self.num_classes, dtype=torch.long, device=device)
        fallback_scale = torch.zeros(self.num_classes, dtype=torch.long, device=device)
        for c in range(self.num_classes):
            counts_c = self.sample_counts[c] + batch_counts[c]
            if counts_c.numel() > 0 and counts_c.any():
                flat_idx = int(counts_c.argmax().item())
                fallback_orient[c] = flat_idx // self.n_scale_bins
                fallback_scale[c] = flat_idx % self.n_scale_bins

        for b in range(class_ids.shape[0]):
            c = int(class_ids[b].item())
            o = int(orient_idx[b].item())
            s = int(scale_idx[b].item())
            total_count = int(self.sample_counts[c, o, s].item()) + int(batch_counts[c, o, s].item())
            if total_count < min(self.orient_min_bin_size, self.scale_min_bin_size):
                orient_idx[b] = fallback_orient[c]
                scale_idx[b] = fallback_scale[c]

        return orient_idx, scale_idx

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"num_prototypes_per_class={self.num_prototypes_per_class}, "
            f"n_orient_bins={self.n_orient_bins}, "
            f"n_scale_bins={self.n_scale_bins}, "
            f"feature_dim={self.feature_dim}, ema_decay={self.ema_decay}"
        )
