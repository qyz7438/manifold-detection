r"""OT-based image-text alignment losses.

Instead of relying solely on pairwise contrastive objectives, these losses
measure the structured discrepancy between the image feature distribution and
the text feature distribution inside a mini-batch using a differentiable
Sinkhorn optimal-transport distance.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_detection_posttrain.methods.manifold.sinkhorn_ot import SinkhornOT


class OTImageTextAlignment(nn.Module):
    r"""Sinkhorn distance between batch-level image and text feature distributions.

    The module L2-normalises both modalities, builds a pairwise cost matrix
    from one minus the cosine similarity, discounts the diagonal entries so
    that paired samples are cheaper to align, and finally computes the
    entropic optimal-transport distance via :class:`SinkhornOT`.

    Args:
        feature_dim: dimensionality of image and text features. Kept for API
            consistency and downstream projection layers.
        eps: entropic regularisation strength passed to Sinkhorn.
        max_iter: number of Sinkhorn fixed-point iterations.
        diag_discount: factor by which diagonal costs are scaled
            (``0`` means no discount, ``1`` means free diagonal alignment).
    """

    def __init__(
        self,
        feature_dim: int,
        eps: float = 0.01,
        max_iter: int = 50,
        diag_discount: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.eps = eps
        self.max_iter = max_iter
        self.diag_discount = diag_discount
        self.sinkhorn = SinkhornOT(eps=eps, max_iter=max_iter, p=2, stable=False)

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        r"""Compute the OT distance between image and text feature distributions.

        Args:
            image_features: tensor of shape ``(B, D)``.
            text_features: tensor of shape ``(B, D)``.

        Returns:
            Scalar Sinkhorn distance with gradients w.r.t. both inputs.
        """
        if image_features.shape != text_features.shape:
            raise ValueError("image_features and text_features must have the same shape")
        if image_features.ndim != 2:
            raise ValueError("expected 2-D input tensors")

        # L2-normalise so that cosine similarity equals the dot product.
        image_norm = F.normalize(image_features, p=2, dim=-1)
        text_norm = F.normalize(text_features, p=2, dim=-1)

        # Pairwise cost: C[i, j] = 1 - cosine_similarity(image_i, text_j).
        cos_sim = torch.matmul(image_norm, text_norm.t())
        cost = 1.0 - cos_sim

        # Make paired samples cheaper to align while keeping costs non-negative.
        batch_size = cost.size(0)
        diag_mask = torch.eye(batch_size, device=cost.device, dtype=cost.dtype)
        cost = cost * (1.0 - self.diag_discount * diag_mask)

        mu = torch.ones(batch_size, device=cost.device, dtype=cost.dtype) / batch_size
        nu = torch.ones(batch_size, device=cost.device, dtype=cost.dtype) / batch_size

        return self.sinkhorn(mu, nu, cost)
