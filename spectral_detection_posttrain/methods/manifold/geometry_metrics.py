r"""Geometry diagnostics for manifold-guided detector features.

This module measures whether the manifold regularization / correction is actually
making the feature space lower-dimensional, more compact, and more class-separated.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from spectral_detection_posttrain.methods.manifold.intrinsic_dim import IntrinsicDimEstimator


def _safe_tensor(x: torch.Tensor | None) -> torch.Tensor:
    if x is None:
        return torch.zeros(())
    return x


def estimate_intrinsic_dimension(
    features: torch.Tensor,
    method: str = "pca",
    variance_threshold: float = 0.95,
) -> torch.Tensor:
    """Estimate intrinsic dimension of a feature cloud.

    Args:
        features: tensor of shape ``(N, D)``.
        method: ``"pca"`` (number of components explaining ``variance_threshold``
            variance) or ``"twonn"``.
        variance_threshold: explained-variance threshold for PCA mode.

    Returns:
        Scalar tensor with the estimated intrinsic dimension.
    """
    estimator = IntrinsicDimEstimator(method=method, pca_variance_threshold=variance_threshold)
    return estimator.estimate_id(features)


def compute_class_centroids(
    features: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-class feature centroids and counts.

    Args:
        features: ``(N, D)``.
        labels: ``(N,)`` class indices (0 = background).
        num_classes: total number of classes including background.

    Returns:
        ``(centroids, counts)`` where centroids has shape ``(num_classes, D)`` and
        counts has shape ``(num_classes,)``.  Empty classes have centroid 0 and
        count 0.
    """
    dim = features.shape[-1]
    centroids = features.new_zeros(num_classes, dim)
    counts = features.new_zeros(num_classes, dtype=torch.long)

    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            centroids[c] = features[mask].mean(dim=0)
            counts[c] = mask.sum().item()
    return centroids, counts


def compute_intra_class_compactness(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    centroids: torch.Tensor | None = None,
    normalize: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute mean intra-class distance to centroid.

    Args:
        features: ``(N, D)``.
        labels: ``(N,)`` class indices.
        num_classes: total number of classes including background.
        centroids: optional pre-computed centroids ``(num_classes, D)``.
        normalize: if True, L2-normalize features before distance computation.

    Returns:
        Dict with ``mean``, ``median``, ``max`` over classes, plus per-class values.
    """
    if normalize:
        features = F.normalize(features, dim=-1)
        if centroids is not None:
            centroids = F.normalize(centroids, dim=-1)

    if centroids is None:
        centroids, _ = compute_class_centroids(features, labels, num_classes)

    per_class_mean = features.new_zeros(num_classes)
    per_class_count = features.new_zeros(num_classes, dtype=torch.long)

    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            feats_c = features[mask]
            dists = (feats_c - centroids[c].unsqueeze(0)).norm(dim=-1)
            per_class_mean[c] = dists.mean()
            per_class_count[c] = mask.sum().item()

    valid = per_class_count > 0
    if valid.any():
        return {
            "intra_mean": per_class_mean[valid].mean(),
            "intra_median": per_class_mean[valid].median(),
            "intra_max": per_class_mean[valid].max(),
            "per_class_intra_mean": per_class_mean.detach(),
            "per_class_count": per_class_count,
        }
    return {
        "intra_mean": features.new_tensor(0.0),
        "intra_median": features.new_tensor(0.0),
        "intra_max": features.new_tensor(0.0),
        "per_class_intra_mean": per_class_mean,
        "per_class_count": per_class_count,
    }


def compute_inter_class_separation(
    centroids: torch.Tensor, labels: torch.Tensor | None = None, normalize: bool = True
) -> dict[str, torch.Tensor]:
    """Compute pairwise distances between class centroids.

    Args:
        centroids: ``(num_classes, D)``; background (class 0) is ignored if
            ``labels`` is not provided.
        labels: optional ``(N,)`` used to decide which classes are present.
        normalize: if True, L2-normalize centroids before distance computation.

    Returns:
        Dict with ``min``, ``mean``, ``max`` pairwise distance over foreground
        classes, plus the full pairwise distance matrix.
    """
    if labels is not None:
        present_classes = sorted({int(c.item()) for c in labels.unique()})
        # Drop background if it is the only class without foreground.
        if 0 in present_classes and len(present_classes) > 1:
            present_classes = [c for c in present_classes if c != 0]
        centroids = centroids[present_classes]
    else:
        centroids = centroids[1:]  # skip background

    if centroids.shape[0] < 2:
        return {
            "inter_min": centroids.new_tensor(0.0),
            "inter_mean": centroids.new_tensor(0.0),
            "inter_max": centroids.new_tensor(0.0),
        }

    if normalize:
        centroids = F.normalize(centroids, dim=-1)

    dists = torch.cdist(centroids, centroids, p=2)
    # Take upper triangle excluding diagonal.
    triu = torch.triu(dists, diagonal=1)
    mask = triu > 0
    values = triu[mask]
    if values.numel() == 0:
        return {
            "inter_min": dists.new_tensor(0.0),
            "inter_mean": dists.new_tensor(0.0),
            "inter_max": dists.new_tensor(0.0),
        }
    return {
        "inter_min": values.min(),
        "inter_mean": values.mean(),
        "inter_max": values.max(),
    }


def compute_manifold_geometry(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    corrected_features: torch.Tensor | None = None,
    method: str = "pca",
    normalize: bool = True,
) -> dict[str, Any]:
    """Compute a full geometry report for a batch of box features.

    Args:
        features: raw box features ``(N, D)``.
        labels: class labels ``(N,)`` (0 = background).
        num_classes: total number of classes including background.
        corrected_features: optional corrected features ``(N, D)`` (e.g. from
            active manifold correction).  When provided, the report includes
            before/after ID and compactness comparisons.
        method: intrinsic-dimension estimator (``"pca"`` or ``"twonn"``).
        normalize: normalize features for compactness/separation computation.

    Returns:
        Nested dict of scalar tensors suitable for JSON logging after ``float()``
        conversion.
    """
    result: dict[str, Any] = {}

    # Overall intrinsic dimension.
    fg_mask = labels >= 1
    if fg_mask.any() and features.shape[0] >= 2:
        result["id_overall"] = estimate_intrinsic_dimension(features, method=method)
        result["id_foreground"] = estimate_intrinsic_dimension(features[fg_mask], method=method)
    else:
        result["id_overall"] = features.new_tensor(float("nan"))
        result["id_foreground"] = features.new_tensor(float("nan"))

    # Per-class intrinsic dimension (foreground only).
    per_class_id = {}
    for c in range(1, num_classes):
        mask = labels == c
        if mask.sum() >= 2:
            per_class_id[str(c)] = float(estimate_intrinsic_dimension(features[mask], method=method).item())
        else:
            per_class_id[str(c)] = float("nan")
    result["per_class_id"] = per_class_id

    # Centroids and compactness for raw features.
    centroids, counts = compute_class_centroids(features, labels, num_classes)
    result["per_class_count"] = {str(c): int(counts[c].item()) for c in range(num_classes)}
    intra = compute_intra_class_compactness(features, labels, num_classes, centroids, normalize=normalize)
    result.update({k: v for k, v in intra.items() if not k.startswith("per_class_")})
    result["per_class_intra_mean"] = intra["per_class_intra_mean"].detach().tolist()

    inter = compute_inter_class_separation(centroids, labels=labels, normalize=normalize)
    result.update(inter)

    # Before/after comparison when correction is provided.
    if corrected_features is not None and corrected_features.shape[0] == features.shape[0]:
        corr_centroids, _ = compute_class_centroids(corrected_features, labels, num_classes)
        corr_intra = compute_intra_class_compactness(
            corrected_features, labels, num_classes, corr_centroids, normalize=normalize
        )
        corr_inter = compute_inter_class_separation(corr_centroids, labels=labels, normalize=normalize)

        result["id_corrected_overall"] = estimate_intrinsic_dimension(corrected_features, method=method)
        result["id_corrected_foreground"] = estimate_intrinsic_dimension(corrected_features[fg_mask], method=method)
        result["id_delta_foreground"] = result["id_corrected_foreground"] - result["id_foreground"]
        result["intra_mean_corrected"] = corr_intra["intra_mean"]
        result["intra_delta"] = corr_intra["intra_mean"] - intra["intra_mean"]
        result["inter_mean_corrected"] = corr_inter["inter_mean"]
        result["inter_delta"] = corr_inter["inter_mean"] - inter["inter_mean"]

    return result


def scalar_geometry_report(geometry: dict[str, Any]) -> dict[str, float]:
    """Convert a geometry report to JSON-serializable floats."""
    report: dict[str, float] = {}
    for key, value in geometry.items():
        if isinstance(value, torch.Tensor):
            report[key] = float(value.detach().cpu().item())
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float)):
                    report[f"{key}_{sub_key}"] = sub_value
                elif isinstance(sub_value, torch.Tensor):
                    report[f"{key}_{sub_key}"] = float(sub_value.detach().cpu().item())
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, (int, float)):
                    report[f"{key}_{idx}"] = item
    return report
