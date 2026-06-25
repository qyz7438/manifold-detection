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
    num_classes: int,
    feature_dim: int,
    background_mode: str = "neg_mean",
    original_weight: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""Build a foreground ETF weight matrix of shape ``(num_classes, feature_dim)``.

    ``num_classes`` here refers to the number of foreground classes.  The
    ``background_mode`` and ``original_weight`` arguments are accepted for
    public-interface compatibility; background construction is handled by
    :func:`build_background_weight`.
    """
    if feature_dim < num_classes:
        raise ValueError(
            f"feature_dim ({feature_dim}) must be >= num_classes ({num_classes}) for ETF"
        )

    e = torch.eye(num_classes, dtype=dtype)
    centered = e - (1.0 / num_classes) * torch.ones_like(e)
    norms = centered.norm(dim=1, keepdim=True).clamp_min(1e-12)
    etf = centered / norms

    if feature_dim > num_classes:
        padding = torch.zeros(num_classes, feature_dim - num_classes, dtype=dtype)
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

    When a projector is requested and the original pretrained classifier
    weights are available, the projector is initialized so that
    ``W_etf @ P = W_orig`` (and the bias is copied).  This makes the ETF
    classifier produce *exactly* the same logits as the baseline at the start
    of post-training, avoiding the AP collapse that occurs when a randomly
    oriented ETF weight matrix is dropped into a feature space tuned to the
    original classifier.

    Args:
        feature_dim: dimensionality of input features.
        num_classes: number of output classes (including background as index 0).
        use_bias: if True, add a learnable bias term.
        use_projector: if True, add a Linear projector before the ETF.
        preserve_logit_scale: if True, rescale ETF rows to the mean original weight norm.
        background_mode: ``"neg_mean"`` or ``"original"``.
        original_weight: pretrained ``cls_score.weight`` used for scale/background init.
        original_bias: pretrained ``cls_score.bias`` used when ``use_bias=True``.
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
        original_bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(
                f"num_classes ({num_classes}) must be >= 2 for ETFClassifier "
                "(background + at least one foreground class)"
            )
        if original_weight is not None and original_weight.shape != (num_classes, feature_dim):
            raise ValueError(
                f"original_weight must have shape {(num_classes, feature_dim)}, "
                f"got {original_weight.shape}"
            )
        if original_bias is not None and original_bias.shape != (num_classes,):
            raise ValueError(
                f"original_bias must have shape {(num_classes,)}, got {original_bias.shape}"
            )

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_foreground_classes = num_classes - 1
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
            if original_bias is not None:
                self.bias = nn.Parameter(original_bias.clone().to(dtype=weight.dtype))
            else:
                self.bias = nn.Parameter(torch.zeros(num_classes, dtype=weight.dtype))

        self.projector: nn.Module | None = None
        if use_projector:
            # A single linear layer.  When the original classifier weights are
            # available we initialize it so that W_etf @ P = W_orig, which makes
            # the ETF head an exact replica of the baseline at initialization.
            self.projector = nn.Linear(feature_dim, feature_dim, bias=False)
            with torch.no_grad():
                if original_weight is not None:
                    w = self.weight  # (num_classes, feature_dim)
                    # (num_classes, num_classes)
                    gram = w @ w.T
                    # Regularize the Gram matrix slightly for numerical stability.
                    eye = torch.eye(num_classes, device=w.device, dtype=w.dtype)
                    inv_gram = torch.linalg.inv(gram + eye * 1e-6)
                    # P = w^T @ inv(w @ w^T) @ W_orig  -> (feature_dim, feature_dim)
                    P = w.T @ inv_gram @ original_weight.to(dtype=w.dtype, device=w.device)
                    self.projector.weight.copy_(P)
                else:
                    self.projector.weight.copy_(torch.eye(feature_dim))

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
    original_bias = None
    if hasattr(cls_score, "weight"):
        original_weight = cls_score.weight.detach().clone()
    if hasattr(cls_score, "bias") and cls_score.bias is not None:
        original_bias = cls_score.bias.detach().clone()

    etf_cls = ETFClassifier(
        feature_dim=in_features,
        num_classes=num_classes,
        use_bias=original_bias is not None,
        use_projector=use_projector,
        preserve_logit_scale=preserve_logit_scale,
        background_mode=background_mode,
        original_weight=original_weight,
        original_bias=original_bias,
    )
    box_predictor.cls_score = etf_cls
    return box_predictor
