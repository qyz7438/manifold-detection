r"""Data-driven adaptive ETF classifier for imbalanced remote-sensing detection.

The standard simplex Equiangular Tight Frame (ETF) assumes balanced classes and
assigns every foreground class the same angular sector.  In practice, remote
sensing datasets are heavily imbalanced and some class pairs are much easier to
confuse than others.  This module replaces the fixed ETF foreground prototypes
with **learnable prototypes** that are initialized from the actual training-data
class centroids and regularized toward a data-dependent target geometry:

* per-class norms can be stretched/shrunk according to class frequency (GOF-like);
* pairwise angles can be enlarged between frequently confused classes.

The background row is kept as a fixed buffer, following the same convention as
:mod:`etf_predictor`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_detection_posttrain.core.models.etf_predictor import (
    build_background_weight,
    build_etf_weight,
)


class AdaptiveETFClassifier(nn.Module):
    """Learnable foreground prototypes with data-driven geometric regularization.

    Args:
        feature_dim: dimensionality of the input ROI features.
        num_classes: number of output classes (background is index 0).
        use_bias: whether to include a learnable bias term.
        background_mode: how to initialize the background row.
        original_weight: pretrained ``cls_score.weight`` used for scale/bias init.
        original_bias: pretrained ``cls_score.bias`` used when ``use_bias=True``.
        preserve_logit_scale: rescale ETF-style init to the mean original norm.
        init_mode: ``"etf"`` for standard simplex ETF initialization, or
            ``"centroids"`` to initialize foreground prototypes from class centers.
        foreground_centers: ``(num_foreground_classes, feature_dim)`` class centers
            collected from the training set.  Required when ``init_mode="centroids"``.
        target_confusion: optional ``(C, C)`` normalized confusion matrix among the
            ``C = num_classes - 1`` foreground classes.  Higher values request larger
            angular separation between the corresponding prototypes.
        lambda_geo: weight of the prototype-geometry regularizer.
        geo_margin_delta: how much to push the cosine bound below the ETF baseline
            for a fully-confused class pair (value 1.0).
        geo_norm_weight: weight of the prototype norm regularizer inside ``geo_loss``.
        scale_mode: ``"none"`` keeps unit norms, ``"freq"`` scales by ``sqrt(count)``,
            and ``"effective"`` scales by effective number (beta=0.999).
        class_counts: ``(num_classes,)`` per-class sample counts; required for
            frequency/effective scaling and used to fill missing centroids.
        use_projector: if True, add a learnable LayerNorm + Linear projector
            between the input features and the prototype classifier.
        projector_dim: hidden dimension of the projector. If None, defaults to
            ``feature_dim``.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        use_bias: bool = False,
        background_mode: str = "neg_mean",
        original_weight: torch.Tensor | None = None,
        original_bias: torch.Tensor | None = None,
        preserve_logit_scale: bool = True,
        init_mode: str = "etf",
        foreground_centers: torch.Tensor | None = None,
        target_confusion: torch.Tensor | None = None,
        lambda_geo: float = 0.0,
        geo_margin_delta: float = 0.2,
        geo_norm_weight: float = 0.01,
        scale_mode: str = "none",
        class_counts: torch.Tensor | None = None,
        use_projector: bool = False,
        projector_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2 for AdaptiveETFClassifier")
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
        self.background_mode = background_mode
        self.preserve_logit_scale = preserve_logit_scale
        self.lambda_geo = float(lambda_geo)
        self.geo_margin_delta = float(geo_margin_delta)
        self.geo_norm_weight = float(geo_norm_weight)
        self.scale_mode = scale_mode

        # Foreground prototype initialization.
        if init_mode == "centroids":
            if foreground_centers is None:
                raise ValueError("foreground_centers is required when init_mode='centroids'")
            if foreground_centers.shape != (self.num_foreground_classes, feature_dim):
                raise ValueError(
                    f"foreground_centers must have shape "
                    f"{(self.num_foreground_classes, feature_dim)}, "
                    f"got {foreground_centers.shape}"
                )
            fg = foreground_centers.detach().clone().to(dtype=torch.float32)
            # Normalize rows; missing classes (all-zero centroids) fall back to ETF rows.
            norms = fg.norm(dim=1, keepdim=True).clamp_min(1e-12)
            fg = fg / norms
            missing = (norms.squeeze(1) < 1e-8).to(fg.device)
            if missing.any():
                etf_fallback = build_etf_weight(self.num_foreground_classes, feature_dim).to(fg.device)
                fg[missing] = etf_fallback[missing]
        elif init_mode == "etf":
            fg = build_etf_weight(self.num_foreground_classes, feature_dim)
            if preserve_logit_scale and original_weight is not None:
                scale = float(original_weight.norm(dim=1).mean().item())
                if math.isfinite(scale) and scale > 0:
                    fg = fg * scale
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

        self.foreground_prototypes = nn.Parameter(fg)

        # Per-class norm scales.
        if scale_mode in ("freq", "effective"):
            if class_counts is None:
                raise ValueError(f"class_counts is required for scale_mode='{scale_mode}'")
            scales = self._init_scales(class_counts, scale_mode)
            self.class_scales = nn.Parameter(scales)
        else:
            self.register_buffer("class_scales", torch.ones(self.num_foreground_classes))

        # Background row.  Initialize from the *initial* foreground geometry and keep fixed.
        with torch.no_grad():
            bg = build_background_weight(
                self.foreground_prototypes * self.class_scales.view(-1, 1),
                mode=background_mode,
                original_weight=original_weight,
            )
            if preserve_logit_scale and init_mode == "etf" and original_weight is not None:
                scale = float(original_weight.norm(dim=1).mean().item())
                if math.isfinite(scale) and scale > 0:
                    bg = bg * scale
            elif background_mode == "original" and original_weight is not None:
                bg_norm = original_weight[0:1].norm(dim=1, keepdim=True).clamp_min(1e-12)
                bg = bg * bg_norm
        self.register_buffer("background_weight", bg)

        # Optional bias.
        self.bias: nn.Parameter | None = None
        if use_bias:
            if original_bias is not None:
                self.bias = nn.Parameter(original_bias.clone().to(dtype=bg.dtype))
            else:
                self.bias = nn.Parameter(torch.zeros(num_classes, dtype=bg.dtype))

        # Target-geometry regularization matrix.
        self.register_buffer(
            "beta_matrix",
            self._build_beta_matrix(target_confusion),
        )

        # Optional learnable projector (feature -> feature).
        self.projector: nn.Module | None = None
        if use_projector:
            pdim = projector_dim if projector_dim is not None else feature_dim
            if pdim != feature_dim:
                raise ValueError(
                    f"projector_dim ({pdim}) must equal feature_dim ({feature_dim}) "
                    "in the current implementation; the prototypes live in the "
                    "projector output space and the background weight must match it."
                )
            self.projector = nn.Sequential(
                nn.LayerNorm(feature_dim, elementwise_affine=False),
                nn.Linear(feature_dim, pdim, bias=True),
                nn.LayerNorm(pdim, elementwise_affine=False),
            )

    def _init_scales(self, class_counts: torch.Tensor, mode: str) -> torch.Tensor:
        fg_counts = class_counts[1:].float().clamp_min(1.0)
        if mode == "freq":
            scales = torch.sqrt(fg_counts / fg_counts.max())
        elif mode == "effective":
            beta = 0.999
            # Effective number; clamp exponent to avoid nan for tiny counts.
            effective = (1.0 - beta) / (1.0 - beta ** fg_counts)
            scales = effective / effective.max()
        else:
            scales = torch.ones_like(fg_counts)
        return scales.to(dtype=torch.float32)

    def _build_beta_matrix(self, target_confusion: torch.Tensor | None) -> torch.Tensor:
        C = self.num_foreground_classes
        device = target_confusion.device if target_confusion is not None else torch.device("cpu")
        beta = torch.full((C, C), -1.0 / max(1, C - 1), dtype=torch.float32, device=device)
        if target_confusion is not None and self.lambda_geo > 0.0:
            if target_confusion.shape != (C, C):
                raise ValueError(
                    f"target_confusion must have shape {(C, C)}, got {target_confusion.shape}"
                )
            conf = target_confusion.detach().clone().to(dtype=torch.float32, device=device)
            off_mask = ~torch.eye(C, dtype=torch.bool, device=device)
            # Higher confusion -> smaller cosine bound (larger enforced angle).
            beta_off = beta[off_mask] - self.geo_margin_delta * conf[off_mask]
            beta[off_mask] = beta_off.clamp(min=-0.9999)
        return beta

    def get_weight(self) -> torch.Tensor:
        """Return the full ``(num_classes, feature_dim)`` classification weight matrix."""
        weight_fg = self.foreground_prototypes * self.class_scales.view(-1, 1)
        return torch.cat([self.background_weight, weight_fg], dim=0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute logits with the current (possibly stretched) prototypes."""
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features must have dim {self.feature_dim}, got {features.shape[-1]}"
            )
        if self.projector is not None:
            features = self.projector(features)
        logits = F.linear(features, self.get_weight(), self.bias)
        return logits

    def geometry_loss(self) -> torch.Tensor | None:
        """Regularize foreground prototype geometry toward the data-driven target.

        Returns ``None`` when ``lambda_geo == 0`` so the caller can skip it.
        """
        if self.lambda_geo <= 0.0:
            return None
        W = self.foreground_prototypes
        Wn = F.normalize(W, dim=1)
        gram = Wn @ Wn.T
        C = self.num_foreground_classes
        off_mask = ~torch.eye(C, dtype=torch.bool, device=W.device)
        loss_angle = F.relu(gram[off_mask] - self.beta_matrix[off_mask]).square().mean()
        loss_norm = (W.norm(dim=1) - 1.0).square().mean()
        return self.lambda_geo * (loss_angle + self.geo_norm_weight * loss_norm)

    def extra_repr(self) -> str:
        return (
            f"feature_dim={self.feature_dim}, num_classes={self.num_classes}, "
            f"lambda_geo={self.lambda_geo}, geo_margin_delta={self.geo_margin_delta}, "
            f"scale_mode={self.scale_mode}, use_bias={self.bias is not None}, "
            f"use_projector={self.projector is not None}"
        )


def replace_cls_score_with_adaptive_etf(
    box_predictor: nn.Module,
    num_classes: int | None = None,
    **kwargs,
) -> nn.Module:
    """Replace ``box_predictor.cls_score`` with an :class:`AdaptiveETFClassifier`.

    If ``box_predictor`` is a :class:`ManifoldCorrectionPredictor` wrapper, the
    replacement is performed on the inner ``base_predictor`` so that active
    correction continues to operate before the classifier.

    All keyword arguments are forwarded to :class:`AdaptiveETFClassifier`.
    """
    from spectral_detection_posttrain.methods.manifold import ManifoldCorrectionPredictor

    target_predictor = box_predictor
    if isinstance(box_predictor, ManifoldCorrectionPredictor):
        target_predictor = box_predictor.base_predictor

    cls_score = getattr(target_predictor, "cls_score", None)
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

    adaptive_cls = AdaptiveETFClassifier(
        feature_dim=in_features,
        num_classes=num_classes,
        use_bias=original_bias is not None,
        original_weight=original_weight,
        original_bias=original_bias,
        **kwargs,
    )
    target_predictor.cls_score = adaptive_cls
    return box_predictor


def get_adaptive_etf_module(box_predictor: nn.Module) -> AdaptiveETFClassifier | None:
    """Return the :class:`AdaptiveETFClassifier` if one is installed."""
    from spectral_detection_posttrain.methods.manifold import ManifoldCorrectionPredictor

    target = box_predictor
    if isinstance(box_predictor, ManifoldCorrectionPredictor):
        target = box_predictor.base_predictor
    cls_score = getattr(target, "cls_score", None)
    return cls_score if isinstance(cls_score, AdaptiveETFClassifier) else None
