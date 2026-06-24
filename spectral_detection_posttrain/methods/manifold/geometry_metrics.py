r"""Geometry diagnostics for manifold-guided detector features.

This module measures whether the manifold regularization / correction is actually
making the feature space lower-dimensional, more compact, and more class-separated.

New metrics (added alongside the original ID / compactness / separation reports):

* **Effective rank** ``effective_rank_*``: the ``exp(entropy)`` of the normalized
  singular-value spectrum.  It is a soft measure of the number of "active"
  dimensions and can distinguish a healthy low-rank structure from a collapsed
  degenerate spectrum.

* **Spectral decay** ``spectral_top{k}_ratio``: cumulative energy captured by the
  top ``k`` singular values.  Fast decay indicates that most variance lives on a
  small number of directions.

* **NC1 (Neural Collapse)** ``nc1_*``: ratio of within-class covariance to
  between-class covariance on foreground features,
  ``Tr(Sigma_W) / Tr(Sigma_B)``.  Lower values mean tighter class clusters that
  are well separated from each other.

* **TP/FP separability AUC** ``separability_*``: AUC of a binary classifier that
  scores foreground samples by their distance to their own class centroid and
  background samples by their distance to the nearest foreground centroid.
  Values above 0.5 mean foreground and background are separable.
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


def compute_effective_rank(features: torch.Tensor) -> torch.Tensor:
    """Compute the effective rank of a feature matrix.

    Effective rank is ``exp(-sum(p_i * log(p_i)))`` where ``p_i`` are the
    singular values normalized to sum to one.  It provides a soft count of the
    number of ``active`` dimensions in the data.

    Args:
        features: tensor of shape ``(N, D)``.

    Returns:
        Scalar tensor with the effective rank.  Returns ``NaN`` for empty or
        single-sample inputs and ``0`` for a zero matrix.
    """
    if features.ndim != 2 or features.shape[0] < 2:
        return torch.tensor(float("nan"), device=features.device, dtype=features.dtype)

    with torch.no_grad():
        _, s, _ = torch.linalg.svd(features, full_matrices=False)
        s = s.clamp_min(0.0)
        total = s.sum()
        if total <= 0:
            return torch.tensor(0.0, device=features.device, dtype=features.dtype)
        p = s / total
        p = p.clamp_min(1e-12)
        entropy = -(p * torch.log(p)).sum()
        return torch.exp(entropy)


def compute_spectral_decay(
    features: torch.Tensor, topk: tuple[int, ...] = (1, 5, 10)
) -> dict[str, torch.Tensor]:
    """Compute cumulative spectral energy ratios for the top singular values.

    Args:
        features: tensor of shape ``(N, D)``.
        topk: which top-k ratios to report.

    Returns:
        Dict with keys ``spectral_top{k}_ratio``.  If ``k`` exceeds the number
        of singular values the ratio is clamped to ``1.0``.
    """
    if features.ndim != 2 or features.shape[0] < 1:
        return {
            f"spectral_top{k}_ratio": torch.tensor(
                float("nan"), device=features.device, dtype=features.dtype
            )
            for k in topk
        }

    with torch.no_grad():
        _, s, _ = torch.linalg.svd(features, full_matrices=False)
        s = s.clamp_min(0.0)
        total = s.sum()
        if total <= 0:
            return {
                f"spectral_top{k}_ratio": torch.tensor(
                    0.0, device=features.device, dtype=features.dtype
                )
                for k in topk
            }
        cumulative = s.cumsum(dim=0)
        result: dict[str, torch.Tensor] = {}
        n = s.numel()
        for k in topk:
            kk = min(k, n) - 1
            result[f"spectral_top{k}_ratio"] = cumulative[kk] / total
        return result


def compute_nc1(
    features: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Compute the Neural Collapse NC1 metric on foreground features.

    NC1 is defined as ``Tr(Sigma_W) / Tr(Sigma_B)`` where:

    * ``Sigma_W`` is the within-class covariance (average squared distance of
      each sample to its class centroid).
    * ``Sigma_B`` is the between-class covariance (weighted squared distance of
      each class centroid to the global foreground mean).

    A low NC1 value indicates that features collapse to tight class clusters
    that are well separated from each other.

    Args:
        features: ``(N, D)``.
        labels: ``(N,)`` class indices (0 = background).
        num_classes: total number of classes including background.

    Returns:
        Scalar tensor with the NC1 value.  Returns ``NaN`` when there are fewer
        than two foreground samples, fewer than two foreground classes, or when
        the between-class covariance is zero.
    """
    fg_mask = labels >= 1
    if not fg_mask.any():
        return torch.tensor(float("nan"), device=features.device, dtype=features.dtype)

    fg_features = features[fg_mask]
    fg_labels = labels[fg_mask]
    n_fg = fg_features.shape[0]
    if n_fg < 2:
        return torch.tensor(float("nan"), device=features.device, dtype=features.dtype)

    dim = features.shape[-1]
    centroids = features.new_zeros(num_classes, dim)
    counts = features.new_zeros(num_classes, dtype=torch.long)
    for c in range(1, num_classes):
        mask = fg_labels == c
        if mask.any():
            centroids[c] = fg_features[mask].mean(dim=0)
            counts[c] = mask.sum().item()

    valid_classes = (counts[1:] > 0).nonzero(as_tuple=False).squeeze(-1) + 1
    if valid_classes.numel() < 2:
        return torch.tensor(float("nan"), device=features.device, dtype=features.dtype)

    global_mean = fg_features.mean(dim=0)

    # Trace of within-class covariance.
    sigma_w = features.new_tensor(0.0)
    for c in valid_classes.tolist():
        mask = fg_labels == c
        diff = fg_features[mask] - centroids[c].unsqueeze(0)
        sigma_w += (diff ** 2).sum()
    sigma_w = sigma_w / n_fg

    # Trace of between-class covariance.
    sigma_b = features.new_tensor(0.0)
    for c in valid_classes.tolist():
        diff = centroids[c] - global_mean
        sigma_b += counts[c].float() * (diff ** 2).sum()
    sigma_b = sigma_b / n_fg

    if sigma_b <= 0:
        return torch.tensor(float("nan"), device=features.device, dtype=features.dtype)
    return sigma_w / sigma_b


def compute_nc1_per_class(
    features: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> dict[str, float]:
    """Compute NC1 per foreground class.

    For each class ``c`` compute ``Tr(Sigma_W,c) / Tr(Sigma_B,c)`` where
    ``Sigma_W,c`` is the within-class scatter and ``Sigma_B,c`` is the
    between-class scatter relative to the global foreground mean.

    Returns:
        Dict mapping class index string to NC1 value (NaN for empty classes).
    """
    fg_mask = labels >= 1
    result: dict[str, float] = {}
    if not fg_mask.any():
        return result

    fg_features = features[fg_mask]
    fg_labels = labels[fg_mask]
    global_mean = fg_features.mean(dim=0)

    for c in range(1, num_classes):
        class_mask = fg_labels == c
        n_c = int(class_mask.sum().item())
        if n_c < 2:
            result[str(c)] = float("nan")
            continue
        class_feats = fg_features[class_mask]
        class_mean = class_feats.mean(dim=0)
        sigma_w = ((class_feats - class_mean) ** 2).sum() / n_c
        sigma_b = ((class_mean - global_mean) ** 2).sum()
        if sigma_b <= 0:
            result[str(c)] = float("nan")
        else:
            result[str(c)] = float((sigma_w / sigma_b).item())
    return result


def compute_nc2(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    etf_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute NC2: mean cosine similarity between class centroids and ETF template.

    The ETF template rows are already unit norm.  We L2-normalize the class
    centroids and compute the average cosine similarity to the corresponding
    ETF row.  Higher values mean the learned centroids are better aligned with
    the ETF geometry.

    Args:
        features: ``(N, D)``.
        labels: ``(N,)`` class indices (0 = background).
        num_classes: total number of classes including background.
        etf_weight: ``(num_classes, D)`` ETF weight matrix.

    Returns:
        Scalar tensor with mean cosine similarity.  Returns NaN when no valid
        foreground classes exist.
    """
    fg_mask = labels >= 1
    if not fg_mask.any():
        return torch.tensor(float("nan"), device=features.device, dtype=features.dtype)

    centroids, counts = compute_class_centroids(features, labels, num_classes)
    valid_classes = (counts[1:] > 0).nonzero(as_tuple=False).squeeze(-1) + 1
    if valid_classes.numel() == 0:
        return torch.tensor(float("nan"), device=features.device, dtype=features.dtype)

    centroids_norm = F.normalize(centroids[valid_classes], dim=-1)
    etf_norm = F.normalize(etf_weight[valid_classes].to(features.device, features.dtype), dim=-1)
    cos_sims = (centroids_norm * etf_norm).sum(dim=-1)
    return cos_sims.mean()


def compute_per_class_effective_rank(
    features: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> dict[str, float]:
    """Compute effective rank per foreground class."""
    result: dict[str, float] = {}
    for c in range(1, num_classes):
        mask = labels == c
        if int(mask.sum().item()) < 2:
            result[str(c)] = float("nan")
        else:
            result[str(c)] = float(compute_effective_rank(features[mask]).item())
    return result


def compute_per_class_spectral_decay(
    features: torch.Tensor, labels: torch.Tensor, num_classes: int, topk: tuple[int, ...] = (5,)
) -> dict[str, dict[str, float]]:
    """Compute spectral top-k energy ratio per foreground class."""
    result: dict[str, dict[str, float]] = {}
    for c in range(1, num_classes):
        mask = labels == c
        if int(mask.sum().item()) < 1:
            result[str(c)] = {f"spectral_top{k}_ratio": float("nan") for k in topk}
        else:
            decay = compute_spectral_decay(features[mask], topk=topk)
            result[str(c)] = {k: float(v.item()) for k, v in decay.items()}
    return result


def compute_separability_auc(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    centroids: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | dict[str, float]]:
    """Compute TP/FP separability based on distances to class centroids.

    For each foreground sample the score is its Euclidean distance to its own
    class centroid; for each background sample the score is its distance to the
    nearest foreground centroid.  The AUC is then the probability that a random
    foreground sample has a smaller distance than a random background sample
    (Mann-Whitney U).

    Args:
        features: ``(N, D)``.
        labels: ``(N,)`` class indices (0 = background, >= 1 foreground).
        num_classes: total number of classes including background.
        centroids: optional pre-computed centroids ``(num_classes, D)``.

    Returns:
        Dict with ``separability_overall`` (tensor) and
        ``per_class_separability`` (dict of floats).
    """
    fg_mask = labels >= 1
    bg_mask = labels == 0
    n_pos = int(fg_mask.sum().item())
    n_neg = int(bg_mask.sum().item())

    result: dict[str, torch.Tensor | dict[str, float]] = {
        "separability_overall": torch.tensor(
            float("nan"), device=features.device, dtype=features.dtype
        ),
        "per_class_separability": {},
    }

    if n_pos == 0 or n_neg == 0:
        return result

    if centroids is None:
        centroids, _ = compute_class_centroids(features, labels, num_classes)

    fg_centroids = centroids[1:]  # (C-1, D)

    # Foreground: distance to own class centroid.
    pos_labels = labels[fg_mask]
    pos_dists = (features[fg_mask] - centroids[pos_labels]).norm(dim=-1)

    # Background: distance to nearest foreground centroid.
    bg_to_fg = torch.cdist(features[bg_mask], fg_centroids)
    neg_dists = bg_to_fg.min(dim=-1).values

    result["separability_overall"] = IntrinsicDimEstimator._auc_from_sorted_distances(
        pos_dists, neg_dists
    )

    # Per-class one-vs-rest AUC using distance to that class centroid.
    per_class: dict[str, float] = {}
    for c in range(1, num_classes):
        class_mask = labels == c
        rest_mask = labels != c
        n_class = int(class_mask.sum().item())
        n_rest = int(rest_mask.sum().item())
        if n_class == 0 or n_rest == 0:
            continue
        pos_c = (features[class_mask] - centroids[c]).norm(dim=-1)
        neg_c = (features[rest_mask] - centroids[c]).norm(dim=-1)
        auc_c = IntrinsicDimEstimator._auc_from_sorted_distances(pos_c, neg_c)
        per_class[str(c)] = float(auc_c.item())
    result["per_class_separability"] = per_class

    return result


def compute_manifold_geometry(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    corrected_features: torch.Tensor | None = None,
    method: str = "pca",
    normalize: bool = True,
    etf_weight: torch.Tensor | None = None,
    class_frequency_weights: torch.Tensor | None = None,
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
        etf_weight: optional ``(num_classes, D)`` ETF weight matrix for NC2.
        class_frequency_weights: optional ``(num_classes,)`` per-class weights.

    Returns:
        Nested dict of scalar tensors suitable for JSON logging after ``float()``
        conversion.

    The returned dict now also contains:

    * ``effective_rank_overall`` / ``effective_rank_foreground``.
    * ``spectral_top1_ratio``, ``spectral_top5_ratio``, ``spectral_top10_ratio``
      (overall and foreground).
    * ``nc1_overall`` (foreground only) and ``per_class_nc1``.
    * ``nc2_overall`` (when ``etf_weight`` is provided).
    * ``per_class_effective_rank`` and ``per_class_spectral_top5_ratio``.
    * ``separability_overall`` and ``per_class_separability``.
    * ``class_frequency_weights`` (when provided).

    When ``corrected_features`` is provided, corrected variants of the above are
    included (e.g. ``effective_rank_corrected_overall``,
    ``nc1_corrected`` / ``nc1_delta``).
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

    # Effective rank (overall and foreground).
    if features.shape[0] >= 2:
        result["effective_rank_overall"] = compute_effective_rank(features)
    else:
        result["effective_rank_overall"] = features.new_tensor(float("nan"))
    if fg_mask.any() and features[fg_mask].shape[0] >= 2:
        result["effective_rank_foreground"] = compute_effective_rank(features[fg_mask])
    else:
        result["effective_rank_foreground"] = features.new_tensor(float("nan"))

    # Spectral decay (overall and foreground).
    spectral_overall = compute_spectral_decay(features)
    spectral_fg = compute_spectral_decay(features[fg_mask]) if fg_mask.any() else {}
    result.update(spectral_overall)
    for key, value in spectral_fg.items():
        result[f"{key}_foreground"] = value

    # NC1 (foreground only).
    result["nc1_overall"] = compute_nc1(features, labels, num_classes)
    result["per_class_nc1"] = compute_nc1_per_class(features, labels, num_classes)

    # NC2: alignment between class centroids and ETF template.
    if etf_weight is not None:
        result["nc2_overall"] = compute_nc2(features, labels, num_classes, etf_weight)

    # Per-class effective rank and spectral decay.
    result["per_class_effective_rank"] = compute_per_class_effective_rank(
        features, labels, num_classes
    )
    result["per_class_spectral_top5_ratio"] = compute_per_class_spectral_decay(
        features, labels, num_classes, topk=(5,)
    )

    # Class frequency weights (for logging context).
    if class_frequency_weights is not None:
        result["class_frequency_weights"] = {
            str(c): float(class_frequency_weights[c].item()) for c in range(num_classes)
        }

    # Centroids and compactness for raw features.
    centroids, counts = compute_class_centroids(features, labels, num_classes)
    result["per_class_count"] = {str(c): int(counts[c].item()) for c in range(num_classes)}
    intra = compute_intra_class_compactness(features, labels, num_classes, centroids, normalize=normalize)
    result.update({k: v for k, v in intra.items() if not k.startswith("per_class_")})
    result["per_class_intra_mean"] = intra["per_class_intra_mean"].detach().tolist()

    inter = compute_inter_class_separation(centroids, labels=labels, normalize=normalize)
    result.update(inter)

    # TP/FP separability AUC.
    sep = compute_separability_auc(features, labels, num_classes, centroids=centroids)
    result["separability_overall"] = sep["separability_overall"]
    result["per_class_separability"] = sep["per_class_separability"]

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

        # Corrected effective rank.
        result["effective_rank_corrected_overall"] = compute_effective_rank(corrected_features)
        if fg_mask.any() and corrected_features[fg_mask].shape[0] >= 2:
            result["effective_rank_corrected_foreground"] = compute_effective_rank(
                corrected_features[fg_mask]
            )
        else:
            result["effective_rank_corrected_foreground"] = corrected_features.new_tensor(float("nan"))

        # Corrected spectral decay.
        corr_spectral_overall = compute_spectral_decay(corrected_features)
        corr_spectral_fg = (
            compute_spectral_decay(corrected_features[fg_mask]) if fg_mask.any() else {}
        )
        for key, value in corr_spectral_overall.items():
            result[f"{key}_corrected"] = value
        for key, value in corr_spectral_fg.items():
            result[f"{key}_corrected_foreground"] = value

        # Corrected NC1.
        result["nc1_corrected"] = compute_nc1(corrected_features, labels, num_classes)
        if torch.isnan(result["nc1_overall"]) or torch.isnan(result["nc1_corrected"]):
            result["nc1_delta"] = corrected_features.new_tensor(float("nan"))
        else:
            result["nc1_delta"] = result["nc1_corrected"] - result["nc1_overall"]

        # Corrected separability.
        corr_sep = compute_separability_auc(
            corrected_features, labels, num_classes, centroids=corr_centroids
        )
        result["separability_corrected"] = corr_sep["separability_overall"]
        if torch.isnan(result["separability_overall"]) or torch.isnan(
            result["separability_corrected"]
        ):
            result["separability_delta"] = corrected_features.new_tensor(float("nan"))
        else:
            result["separability_delta"] = (
                result["separability_corrected"] - result["separability_overall"]
            )

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
