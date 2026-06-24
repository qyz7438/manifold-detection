r"""Equiangular Tight Frame (ETF) classifier for detector box predictors.

An ETF classifier fixes the classification weight matrix so that every class
vector has unit norm and pairwise cosine similarity equal to
``-1 / (num_classes - 1)``.  This is the geometry predicted by Neural Collapse
for optimal classifiers.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_etf_weight(num_classes: int, feature_dim: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    r"""Build an ETF weight matrix of shape ``(num_classes, feature_dim)``.

    The construction follows Pythagoras et al. (Neural Collapse):

    1. Start with a standard basis in ``num_classes`` dimensions.
    2. Shift by the all-ones vector to remove the trivial direction.
    3. Normalize so each row has unit length.

    The resulting rows satisfy ``W_i \cdot W_j = -1 / (num_classes - 1)`` for
    ``i != j`` and ``||W_i|| = 1``.

    Args:
        num_classes: number of classes (including background if applicable).
        feature_dim: dimensionality of the input features.  Must be >= num_classes.
        dtype: torch dtype for the returned tensor.

    Returns:
        Float tensor of shape ``(num_classes, feature_dim)``.
    """
    if feature_dim < num_classes:
        raise ValueError(
            f"feature_dim ({feature_dim}) must be >= num_classes ({num_classes}) for ETF"
        )

    # Standard basis in R^C.
    e = torch.eye(num_classes, dtype=dtype)
    # Center: remove the all-ones direction.
    centered = e - (1.0 / num_classes) * torch.ones_like(e)
    # Normalize rows.
    norms = centered.norm(dim=1, keepdim=True).clamp_min(1e-12)
    etf_2d = centered / norms

    # Embed into R^D by padding zeros if D > C.
    if feature_dim > num_classes:
        padding = torch.zeros(num_classes, feature_dim - num_classes, dtype=dtype)
        etf_2d = torch.cat([etf_2d, padding], dim=1)

    return etf_2d


class ETFClassifier(nn.Module):
    """Fixed ETF classification layer.

    The weight matrix is computed once and frozen (``requires_grad=False``).
    An optional learnable bias can be enabled, but the standard ETF predictor
    uses no bias.

    Args:
        feature_dim: dimensionality of input features.
        num_classes: number of output classes.
        use_bias: if True, add a learnable bias term.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        weight = build_etf_weight(num_classes, feature_dim)
        self.register_buffer("weight", weight)
        self.bias: nn.Parameter | None = None
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute logits as ``features @ W^T`` (plus optional bias)."""
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features must have dim {self.feature_dim}, got {features.shape[-1]}"
            )
        logits = F.linear(features, self.weight, self.bias)
        return logits

    def extra_repr(self) -> str:
        return f"feature_dim={self.feature_dim}, num_classes={self.num_classes}, use_bias={self.bias is not None}"


def replace_cls_score_with_etf(
    box_predictor: nn.Module,
    num_classes: int | None = None,
) -> nn.Module:
    """Replace the ``cls_score`` layer of a box predictor with an ETF classifier.

    Args:
        box_predictor: a module with attributes ``cls_score`` and ``bbox_pred``.
        num_classes: number of output classes.  If None, inferred from
            ``cls_score.out_features``.

    Returns:
        The modified ``box_predictor``.
    """
    cls_score = getattr(box_predictor, "cls_score", None)
    if cls_score is None:
        raise ValueError("box_predictor must have a cls_score attribute")

    in_features = cls_score.in_features
    if num_classes is None:
        num_classes = cls_score.out_features

    etf_cls = ETFClassifier(
        feature_dim=in_features,
        num_classes=num_classes,
        use_bias=False,
    )
    box_predictor.cls_score = etf_cls
    return box_predictor
