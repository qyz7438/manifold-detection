from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def _as_2d_float(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("features must be a 2D array")
    return array


def pca_dimensionality_summary(features: np.ndarray, *, max_components: int = 128) -> dict[str, object]:
    features = _as_2d_float(features)
    sample_count, feature_dim = features.shape
    n_components = min(int(max_components), sample_count - 1, feature_dim)
    if n_components <= 0:
        return {
            "sample_count": int(sample_count),
            "feature_dim": int(feature_dim),
            "components": 0,
            "explained_variance_ratio": [],
            "cumulative_explained_variance": [],
            "components_for_80pct": 0,
            "components_for_90pct": 0,
            "components_for_95pct": 0,
            "components_for_99pct": 0,
            "participation_ratio": 0.0,
        }

    scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(scaled)
    ratios = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    cumulative = np.cumsum(ratios)
    eigenvalues = np.asarray(pca.explained_variance_, dtype=np.float64)
    participation = float((eigenvalues.sum() ** 2) / np.square(eigenvalues).sum()) if eigenvalues.size else 0.0

    def dims_for(threshold: float) -> int:
        if cumulative.size == 0:
            return 0
        return int(np.searchsorted(cumulative, float(threshold), side="left") + 1)

    return {
        "sample_count": int(sample_count),
        "feature_dim": int(feature_dim),
        "components": int(n_components),
        "explained_variance_ratio": ratios.tolist(),
        "cumulative_explained_variance": cumulative.tolist(),
        "components_for_80pct": dims_for(0.80),
        "components_for_90pct": dims_for(0.90),
        "components_for_95pct": dims_for(0.95),
        "components_for_99pct": dims_for(0.99),
        "participation_ratio": participation,
    }


def binary_ranking_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    precision_targets: tuple[float, ...] = (0.7, 0.8, 0.9),
) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if scores.shape[0] != labels.shape[0]:
        raise ValueError("scores and labels must have the same length")
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    result: dict[str, float] = {
        "count": int(scores.shape[0]),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "auc": 0.0,
        "average_precision": 0.0,
    }
    if positive_count > 0 and negative_count > 0:
        result["auc"] = float(roc_auc_score(labels.astype(np.int32), scores))
        result["average_precision"] = float(average_precision_score(labels.astype(np.int32), scores))

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp_cum = np.cumsum(sorted_labels.astype(np.float64))
    rank = np.arange(1, sorted_labels.shape[0] + 1, dtype=np.float64)
    precision = tp_cum / rank
    recall = tp_cum / max(1, positive_count)
    for target in precision_targets:
        valid = precision >= float(target)
        key = f"recall_at_precision_{target:g}"
        result[key] = float(recall[valid].max()) if valid.any() else 0.0
    return result


def center_margin_scores(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    query_features: np.ndarray,
) -> np.ndarray:
    train_features = _as_2d_float(train_features)
    query_features = _as_2d_float(query_features)
    train_labels = np.asarray(train_labels, dtype=bool)
    if train_features.shape[0] != train_labels.shape[0]:
        raise ValueError("train_features and train_labels must have the same row count")
    if train_features.shape[1] != query_features.shape[1]:
        raise ValueError("train_features and query_features must have the same feature dimension")
    positive = train_features[train_labels]
    negative = train_features[~train_labels]
    if positive.size == 0 or negative.size == 0:
        return np.zeros((query_features.shape[0],), dtype=np.float64)
    positive_center = positive.mean(axis=0, keepdims=True)
    negative_center = negative.mean(axis=0, keepdims=True)
    positive_distance = np.linalg.norm(query_features - positive_center, axis=1)
    negative_distance = np.linalg.norm(query_features - negative_center, axis=1)
    return negative_distance - positive_distance
