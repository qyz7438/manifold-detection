r"""Sinkhorn optimal-transport loss for semantic segmentation.

The loss treats predicted class probabilities and ground-truth labels as
spatial distributions and computes a differentiable entropic-transport
distance.  To keep the computation tractable, only a subset of pixels is
sampled per image.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_detection_posttrain.methods.manifold.sinkhorn_ot import SinkhornOT


class OTSegmentationLoss(nn.Module):
    r"""Sampled spatial-class optimal-transport loss for segmentation.

    For each image, ``sample_count`` pixels are drawn.  Each sampled pixel is a
    point with a class distribution (predicted probabilities) and a spatial
    coordinate.  The loss is the Sinkhorn distance between the predicted point
    cloud and the ground-truth point cloud, using a cost that combines spatial
    distance and class mismatch.

    Args:
        num_classes: number of semantic classes (including background).
        eps: entropic regularization for Sinkhorn iterations.
        max_iter: number of Sinkhorn fixed-point iterations.
        sample_count: number of pixels sampled per image.
        sample_mode: ``"random"`` or ``"uncertain"``.  Uncertain sampling
            preferentially selects pixels with high prediction entropy.
        spatial_weight: weight of the spatial distance term in the cost.
        class_weight: weight of the class mismatch term in the cost.
        p: power used for spatial distances (``1`` or ``2``).
        ignore_index: optional class index to ignore in the target.

    Shape:
        - pred: :math:`(B, C, H, W)` logits.
        - target: :math:`(B, H, W)` long labels.
        - output: scalar loss.
    """

    def __init__(
        self,
        num_classes: int,
        eps: float = 0.01,
        max_iter: int = 100,
        sample_count: int = 128,
        sample_mode: str = "random",
        spatial_weight: float = 1.0,
        class_weight: float = 1.0,
        p: int = 2,
        ignore_index: int | None = None,
    ):
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be > 1")
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if sample_mode not in {"random", "uncertain"}:
            raise ValueError(f"Unknown sample_mode: {sample_mode}")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")

        self.num_classes = num_classes
        self.sample_count = sample_count
        self.sample_mode = sample_mode
        self.spatial_weight = spatial_weight
        self.class_weight = class_weight
        self.p = p
        self.ignore_index = ignore_index

        self.sinkhorn = SinkhornOT(eps=eps, max_iter=max_iter, p=p, stable=True)

    def _sample_pixels(
        self, probs: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample pixel indices and return coordinates/probs/labels.

        Args:
            probs: predicted probabilities ``(B, C, H, W)``.
            target: ground-truth labels ``(B, H, W)``.

        Returns:
            ``(coords, sampled_probs, sampled_target)`` where ``coords`` has
            shape ``(B, K, 2)`` (normalized y, x), ``sampled_probs`` has shape
            ``(B, K, C)``, and ``sampled_target`` has shape ``(B, K,)``.
        """
        B, C, H, W = probs.shape
        K = min(self.sample_count, H * W)
        device = probs.device

        if self.sample_mode == "uncertain":
            # Entropy per pixel: high entropy => uncertain.
            entropy = -(probs * (probs + 1e-12).log()).sum(dim=1)  # (B, H, W)
            # Ignore pixels matching ignore_index.
            if self.ignore_index is not None:
                entropy = entropy.masked_fill(target == self.ignore_index, -1e9)
            flat_entropy = entropy.reshape(B, -1).clamp_min(0.0)
            # Guard against all-zero weights (e.g. perfect predictions).
            flat_entropy = flat_entropy + flat_entropy.sum(dim=1, keepdim=True).clamp_min(1e-12) * 1e-6
            sample_indices = torch.multinomial(flat_entropy, K, replacement=False)
        else:
            sample_indices = torch.stack([
                torch.randperm(H * W, device=device)[:K]
                for _ in range(B)
            ], dim=0)

        # Convert flat indices to 2D normalized coordinates.
        ys = (sample_indices // W).float() / max(H - 1, 1)
        xs = (sample_indices % W).float() / max(W - 1, 1)
        coords = torch.stack([ys, xs], dim=-1)  # (B, K, 2)

        # Gather probabilities and targets at sampled positions.
        flat_probs = probs.reshape(B, C, -1)  # (B, C, H*W)
        sampled_probs = flat_probs.gather(
            2, sample_indices.unsqueeze(1).expand(-1, C, -1)
        ).permute(0, 2, 1)  # (B, K, C)

        flat_target = target.reshape(B, -1)
        sampled_target = flat_target.gather(1, sample_indices)  # (B, K)

        # Filter ignore_index samples by replacing them with a valid random
        # pixel.  This keeps the sampled set valid while preserving gradients.
        if self.ignore_index is not None:
            mask = sampled_target == self.ignore_index
            if mask.any():
                replacement = torch.randint(
                    0, H * W, sample_indices.shape, device=device
                )
                sample_indices = torch.where(mask, replacement, sample_indices)
                ys = (sample_indices // W).float() / max(H - 1, 1)
                xs = (sample_indices % W).float() / max(W - 1, 1)
                coords = torch.stack([ys, xs], dim=-1)
                sampled_probs = flat_probs.gather(
                    2, sample_indices.unsqueeze(1).expand(-1, C, -1)
                ).permute(0, 2, 1)
                sampled_target = flat_target.gather(1, sample_indices)

        return coords, sampled_probs, sampled_target

    @staticmethod
    def _pairwise_spatial_cost(coords: torch.Tensor, p: int) -> torch.Tensor:
        r"""Compute pairwise :math:`\ell_p` distance between coordinates.

        Args:
            coords: tensor of shape ``(B, K, 2)``.
            p: power of the distance.

        Returns:
            Cost matrix of shape ``(B, K, K)``.
        """
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, K, K, 2)
        return diff.norm(dim=-1, p=p)

    def _image_cost_matrix(
        self,
        coords: torch.Tensor,
        sampled_probs: torch.Tensor,
        sampled_target: torch.Tensor,
    ) -> torch.Tensor:
        """Build the combined spatial + class cost matrix for one image.

        Args:
            coords: ``(K, 2)``.
            sampled_probs: ``(K, C)``.
            sampled_target: ``(K,)``.

        Returns:
            Cost matrix of shape ``(K, K)``.
        """
        K = coords.shape[0]
        spatial_cost = self._pairwise_spatial_cost(coords.unsqueeze(0), self.p).squeeze(0)
        # Class mismatch cost: 1 - predicted probability of the target class.
        pred_target_prob = sampled_probs.gather(1, sampled_target.unsqueeze(1)).squeeze(1)
        class_cost = 1.0 - pred_target_prob  # (K,)
        # Broadcast to (K, K): cost[i, j] depends on source i and target j.
        class_cost_matrix = class_cost.unsqueeze(1).expand(K, K)
        return self.spatial_weight * spatial_cost + self.class_weight * class_cost_matrix

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the OT segmentation loss.

        Args:
            pred: logits of shape ``(B, C, H, W)``.
            target: ground-truth labels of shape ``(B, H, W)``.

        Returns:
            Scalar tensor containing the averaged Sinkhorn distance.
        """
        if pred.shape[1] != self.num_classes:
            raise ValueError(
                f"pred has {pred.shape[1]} channels but num_classes={self.num_classes}"
            )
        if pred.shape[0] != target.shape[0]:
            raise ValueError("pred and target must have the same batch size")

        B = pred.shape[0]
        probs = F.softmax(pred, dim=1)

        coords, sampled_probs, sampled_target = self._sample_pixels(probs, target)

        # Uniform weights over sampled points.
        K = coords.shape[1]
        mu = torch.ones(K, device=pred.device, dtype=pred.dtype) / K
        nu = mu.clone()

        loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        valid_images = 0
        for b in range(B):
            cost = self._image_cost_matrix(
                coords[b], sampled_probs[b], sampled_target[b]
            )
            dist = self.sinkhorn(mu, nu, cost)
            if torch.isfinite(dist):
                loss = loss + dist
                valid_images += 1

        if valid_images == 0:
            return loss  # zero
        return loss / valid_images
