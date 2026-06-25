r"""Equiangular Tight Frame (ETF) classifier for detector box predictors.

An ETF classifier fixes the foreground classification weight matrix so that
every foreground class vector has equal norm and pairwise cosine similarity
equal to ``-1 / (num_foreground_classes - 1)``.  This is the geometry
predicted by Neural Collapse for optimal classifiers.

In detection, class index 0 is background and is *not* part of the ETF
simplex.  The background weight is initialized separately and the whole
matrix can be rescaled to preserve the logit magnitude of the pretrained
classifier being replaced.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_etf_weight(
    num_foreground_classes: int,
    feature_dim: int,
    background_mode: str = "neg_mean",
    original_weight: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""Build a foreground ETF weight matrix of shape ``(num_foreground_classes, feature_dim)``.

    The ``background_mode`` and ``original_weight`` arguments are accepted for
    public-interface compatibility; background construction is handled by
    :func:`build_background_weight`.
    """
    if feature_dim < num_foreground_classes:
        raise ValueError(
            f"feature_dim ({feature_dim}) must be >= num_foreground_classes ({num_foreground_classes}) for ETF"
        )

    e = torch.eye(num_foreground_classes, dtype=dtype)
    centered = e - (1.0 / num_foreground_classes) * torch.ones_like(e)
    norms = centered.norm(dim=1, keepdim=True).clamp_min(1e-12)
    etf = centered / norms

    if feature_dim > num_foreground_classes:
        padding = torch.zeros(num_foreground_classes, feature_dim - num_foreground_classes, dtype=dtype)
        etf = torch.cat([etf, padding], dim=1)

    return etf


def build_background_weight(
    foreground_etf: torch.Tensor,
    mode: str = "neg_mean",
    original_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Build a single background weight row of shape ``(1, feature_dim)``."""
    if mode == "neg_mean":
        bg = -foreground_etf.mean(dim=0, keepdim=True)
        bg = bg / bg.norm(dim=1, keepdim=True).clamp_min(1e-12)
    elif mode == "original":
        if original_weight is None:
            raise ValueError("original_weight is required when background_mode='original'")
        bg = original_weight[0:1].to(dtype=foreground_etf.dtype, device=foreground_etf.device)
        bg = bg / bg.norm(dim=1, keepdim=True).clamp_min(1e-12)
    else:
        raise ValueError(f"Unsupported background_mode: {mode}")
    return bg


class ETFClassifier(nn.Module):
    """Fixed ETF classification layer with detection-aware background handling.

    The foreground weight matrix is frozen and forms a simplex ETF.  The
    background row is initialized separately.  An optional learnable projector
    can be inserted before the fixed classifier, and the whole matrix can be
    rescaled to match the logit magnitude of the pretrained classifier.

    Args:
        feature_dim: dimensionality of input features.
        num_classes: number of output classes (including background as index 0).
        use_bias: if True, add a learnable bias term.
        use_projector: if True, add LayerNorm + Linear projector before ETF.
        preserve_logit_scale: if True, rescale ETF rows to the mean original weight norm.
        background_mode: ``"neg_mean"`` or ``"original"``.
        original_weight: pretrained ``cls_score.weight`` used for scale/background init.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        use_bias: bool = False,
        use_projector: bool = False,
        preserve_logit_scale: bool = True,
        background_mode: str = "neg_mean",
        original_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_foreground_classes = max(1, num_classes - 1)
        self.use_projector = use_projector
        self.preserve_logit_scale = preserve_logit_scale
        self.background_mode = background_mode

        foreground_etf = build_etf_weight(self.num_foreground_classes, feature_dim)

        scale = 1.0
        if preserve_logit_scale and original_weight is not None:
            scale = float(original_weight.norm(dim=1).mean().item())
            if math.isfinite(scale) and scale > 0:
                foreground_etf = foreground_etf * scale
        self.register_buffer("logit_scale", torch.tensor(scale, dtype=torch.float32))

        background_weight = build_background_weight(
            foreground_etf, mode=background_mode, original_weight=original_weight
        )
        if preserve_logit_scale and scale > 0:
            background_weight = background_weight * scale
        elif background_mode == "original" and original_weight is not None:
            # When not preserving the global logit scale, keep the original
            # background row's magnitude so the pretrained background geometry
            # is reproduced exactly.
            bg_norm = original_weight[0:1].norm(dim=1, keepdim=True).clamp_min(1e-12)
            background_weight = background_weight * bg_norm

        weight = torch.cat([background_weight, foreground_etf], dim=0)
        self.register_buffer("weight", weight)

        self.bias: nn.Parameter | None = None
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(num_classes))

        self.projector: nn.Module | None = None
        if use_projector:
            self.projector = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, feature_dim),
            )
            # Initialize the linear layer as an identity so the projector starts close to a no-op.
            with torch.no_grad():
                self.projector[1].weight.copy_(torch.eye(feature_dim))
                nn.init.zeros_(self.projector[1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute logits as ``projector(features) @ W^T`` (plus optional bias)."""
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features must have dim {self.feature_dim}, got {features.shape[-1]}"
            )
        x = features
        if self.projector is not None:
            x = self.projector(x)
        logits = F.linear(x, self.weight, self.bias)
        return logits

    def extra_repr(self) -> str:
        return (
            f"feature_dim={self.feature_dim}, num_classes={self.num_classes}, "
            f"use_bias={self.bias is not None}, use_projector={self.use_projector}, "
            f"preserve_logit_scale={self.preserve_logit_scale}, "
            f"background_mode={self.background_mode}"
        )


def replace_cls_score_with_etf(
    box_predictor: nn.Module,
    num_classes: int | None = None,
    use_projector: bool = False,
    preserve_logit_scale: bool = True,
    background_mode: str = "neg_mean",
) -> nn.Module:
    """Replace the ``cls_score`` layer of a box predictor with an ETF classifier.

    Args:
        box_predictor: a module with attributes ``cls_score`` and ``bbox_pred``.
        num_classes: number of output classes (including background).  If None,
            inferred from ``cls_score.out_features``.
        use_projector: add a trainable projector before the frozen ETF.
        preserve_logit_scale: rescale ETF rows to the original weight norm.
        background_mode: how to initialize the background row.
    """
    cls_score = getattr(box_predictor, "cls_score", None)
    if cls_score is None:
        raise ValueError("box_predictor must have a cls_score attribute")

    in_features = cls_score.in_features
    if num_classes is None:
        num_classes = cls_score.out_features

    original_weight = None
    if hasattr(cls_score, "weight"):
        original_weight = cls_score.weight.detach().clone()

    etf_cls = ETFClassifier(
        feature_dim=in_features,
        num_classes=num_classes,
        use_bias=False,
        use_projector=use_projector,
        preserve_logit_scale=preserve_logit_scale,
        background_mode=background_mode,
        original_weight=original_weight,
    )
    box_predictor.cls_score = etf_cls
    return box_predictor
