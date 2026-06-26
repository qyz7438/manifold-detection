from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


LEGACY_IFFT_FEATURE_NAMES = [
    "raw_edge",
    "phase_edge",
    "hp015_edge",
    "fft_edge_truncation",
    "low_edge",
    "high_edge",
    "high_minus_low_edge",
    "low_energy_ratio",
    "mid_energy_ratio",
    "high_energy_ratio",
    "high_low_energy_ratio",
    "phase_abs_low",
    "phase_abs_mid",
    "phase_abs_high",
    "negative_phase_abs_low",
    "negative_phase_abs_high",
    "energy_times_negative_phase_high",
    "entropy",
    "center_surround",
    "laplacian",
    "autocorr_peak",
    "phase_std",
    "phase_abs_diff",
]


@dataclass(frozen=True)
class TrainEffectScorer:
    method: str
    scaler: StandardScaler
    weights: np.ndarray
    train_signed: np.ndarray

    def score(self, features: np.ndarray) -> np.ndarray:
        features_z = self.scaler.transform(np.asarray(features, dtype=np.float64))
        signed = features_z * self.weights.reshape(1, -1)
        if self.method == "train_effect_sum":
            return signed.sum(axis=1)
        if self.method == "rank_sum":
            output = np.zeros((signed.shape[0],), dtype=np.float64)
            for column in range(signed.shape[1]):
                reference = self.train_signed[:, column]
                output += np.searchsorted(np.sort(reference), signed[:, column], side="right") / max(1, reference.shape[0])
            return output
        raise ValueError(f"Unknown scorer method: {self.method}")


@dataclass(frozen=True)
class CalibratedThreshold:
    target_precision: float
    threshold: float
    margin: float
    selected_prefix: int
    tp_prefix: int
    fp_prefix: int
    precision_prefix: float
    recall_prefix: float
    reason: str = "ok"


def fit_train_effect_scorer(features: np.ndarray, labels: np.ndarray, *, method: str = "train_effect_sum") -> TrainEffectScorer:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if features.ndim != 2:
        raise ValueError("features must be 2D")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have the same row count")
    if method not in {"train_effect_sum", "rank_sum"}:
        raise ValueError(f"Unknown scorer method: {method}")
    scaler = StandardScaler().fit(features)
    features_z = scaler.transform(features)
    weights = features_z[labels].mean(axis=0) - features_z[~labels].mean(axis=0)
    if method == "rank_sum":
        weights = np.sign(weights)
        weights[weights == 0.0] = 1.0
    train_signed = features_z * weights.reshape(1, -1)
    return TrainEffectScorer(method=method, scaler=scaler, weights=weights.astype(np.float64), train_signed=train_signed)


def parse_legacy_ifft_feature_specs(feature_specs: list[str]) -> list[tuple[int, int, str]]:
    parsed = []
    for spec in feature_specs:
        try:
            feature_name, crop_text = spec.split("@", maxsplit=1)
            crop_size = int(crop_text)
        except ValueError as exc:
            raise ValueError(f"Invalid raw-iFFT feature spec '{spec}', expected name@crop_size") from exc
        if feature_name not in LEGACY_IFFT_FEATURE_NAMES:
            raise ValueError(f"Unknown raw-iFFT feature '{feature_name}' in '{spec}'")
        parsed.append((LEGACY_IFFT_FEATURE_NAMES.index(feature_name), crop_size, spec))
    return parsed


def score_legacy_ifft_metric_bank(
    metric_bank_by_crop: dict[int, torch.Tensor],
    parsed_specs: list[tuple[int, int, str]],
    *,
    mean: torch.Tensor,
    scale: torch.Tensor,
    weights: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    if not parsed_specs:
        first = next(iter(metric_bank_by_crop.values()))
        return first.new_empty((first.shape[0],))
    columns = []
    for feature_index, crop_size, spec in parsed_specs:
        if int(crop_size) not in metric_bank_by_crop:
            raise ValueError(f"Missing metric bank for raw-iFFT feature '{spec}'")
        bank = metric_bank_by_crop[int(crop_size)]
        columns.append(bank[:, int(feature_index)].float())
    features = torch.stack(columns, dim=1)
    device = features.device
    mean = mean.to(device=device, dtype=features.dtype)
    scale = scale.to(device=device, dtype=features.dtype).clamp_min(1e-6)
    weights = weights.to(device=device, dtype=features.dtype)
    return ((features - mean) / scale * weights).sum(dim=1) - float(threshold)


def score_scene_legacy_ifft_metric_bank(
    metric_bank_by_crop: dict[int, torch.Tensor],
    labels: torch.Tensor,
    scene_groups: list[dict],
    *,
    fallback_score: float = -1.0e6,
) -> torch.Tensor:
    if not metric_bank_by_crop:
        raise ValueError("metric_bank_by_crop must not be empty")
    first = next(iter(metric_bank_by_crop.values()))
    scores = first.new_full((first.shape[0],), float(fallback_score))
    labels = labels.to(device=first.device, dtype=torch.long)
    for group in scene_groups:
        if not bool(group.get("enabled", True)):
            continue
        classes = [int(value) for value in group.get("classes", [])]
        if not classes:
            continue
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for class_id in classes:
            mask |= labels == int(class_id)
        if not bool(mask.any()):
            continue
        parsed_specs = [
            (int(item["feature_index"]), int(item["crop_size"]), str(item["spec"]))
            for item in group.get("parsed_features", [])
        ]
        if not parsed_specs:
            continue
        group_scores = score_legacy_ifft_metric_bank(
            metric_bank_by_crop,
            parsed_specs,
            mean=torch.tensor(group["scaler_mean"], dtype=first.dtype, device=first.device),
            scale=torch.tensor(group["scaler_scale"], dtype=first.dtype, device=first.device),
            weights=torch.tensor(group["weights"], dtype=first.dtype, device=first.device),
            threshold=float(group["threshold"]),
        )
        scores[mask] = group_scores[mask]
    return scores


def calibrate_precision_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    target_precision: float,
    margin: float = 0.0,
) -> CalibratedThreshold:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if scores.shape[0] != labels.shape[0]:
        raise ValueError("scores and labels must have the same length")
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels.astype(np.int64))
    rank = np.arange(1, sorted_labels.shape[0] + 1)
    precision = tp / rank
    recall = tp / max(1, int(labels.sum()))
    valid = np.flatnonzero(precision >= float(target_precision))
    if valid.shape[0] == 0:
        return CalibratedThreshold(
            target_precision=float(target_precision),
            threshold=float("inf"),
            margin=float(margin),
            selected_prefix=0,
            tp_prefix=0,
            fp_prefix=0,
            precision_prefix=0.0,
            recall_prefix=0.0,
            reason="no_prefix_reaches_target_precision",
        )
    best_recall = recall[valid].max()
    best_candidates = valid[recall[valid] == best_recall]
    best_index = int(best_candidates[-1])
    return CalibratedThreshold(
        target_precision=float(target_precision),
        threshold=float(sorted_scores[best_index] + float(margin)),
        margin=float(margin),
        selected_prefix=int(rank[best_index]),
        tp_prefix=int(tp[best_index]),
        fp_prefix=int(rank[best_index] - tp[best_index]),
        precision_prefix=float(precision[best_index]),
        recall_prefix=float(recall[best_index]),
    )


def apply_selection_policy(
    scores: np.ndarray,
    *,
    threshold: float,
    primary_scores: np.ndarray | None = None,
    primary_threshold: float | None = None,
    image_ids: np.ndarray | None = None,
    top_k_per_image: int | None = None,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    selected = scores >= float(threshold)
    if primary_scores is not None and primary_threshold is not None:
        selected &= np.asarray(primary_scores, dtype=np.float64) >= float(primary_threshold)
    if image_ids is not None and top_k_per_image is not None and int(top_k_per_image) > 0:
        image_ids = np.asarray(image_ids)
        limited = np.zeros_like(selected, dtype=bool)
        for image_id in np.unique(image_ids[selected]):
            indices = np.flatnonzero(selected & (image_ids == image_id))
            if indices.shape[0] == 0:
                continue
            order = indices[np.argsort(-scores[indices])]
            limited[order[: int(top_k_per_image)]] = True
        selected = limited
    return selected


def threshold_metrics(selected: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    selected = np.asarray(selected, dtype=bool)
    labels = np.asarray(labels, dtype=bool)
    if selected.shape[0] != labels.shape[0]:
        raise ValueError("selected and labels must have the same length")
    selected_count = int(selected.sum())
    true_positive = int((selected & labels).sum())
    false_positive = int((selected & (~labels)).sum())
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    return {
        "selected": selected_count,
        "tp": true_positive,
        "fp": false_positive,
        "precision": float(true_positive / selected_count) if selected_count else 0.0,
        "recall": float(true_positive / positive_count) if positive_count else 0.0,
        "false_positive_rate": float(false_positive / negative_count) if negative_count else 0.0,
    }
