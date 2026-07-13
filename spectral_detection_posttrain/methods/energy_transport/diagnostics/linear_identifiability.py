"""Train-cache-only linear diagnostics for local candidate Delta-U."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LocalCandidateRows:
    """Sparse candidate rows without proposal-set or target-derived features."""

    roi_features: torch.Tensor
    detector_features: torch.Tensor
    action_features: torch.Tensor
    target: torch.Tensor
    image_ids: torch.Tensor


@dataclass(frozen=True)
class FrozenLocalFeatureMap:
    """Train-only PCA and standardization state for local candidate rows."""

    roi_mean: torch.Tensor
    roi_scale: torch.Tensor
    roi_components: torch.Tensor
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor


@dataclass(frozen=True)
class RidgeRegressor:
    """Closed-form ridge model with an unregularized intercept."""

    weights: torch.Tensor
    bias: torch.Tensor
    l2: float


def extract_local_candidate_rows(
    records: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    *,
    num_classes: int,
) -> LocalCandidateRows:
    if not records or num_classes <= 0:
        raise ValueError("records and num_classes must be non-empty")
    roi_rows: list[torch.Tensor] = []
    detector_rows: list[torch.Tensor] = []
    action_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    image_rows: list[torch.Tensor] = []
    for detector, trace in records:
        proposal_indices = torch.as_tensor(
            trace["proposal_indices"], dtype=torch.long
        ).flatten()
        count = int(proposal_indices.numel())
        if count == 0:
            continue
        spatial = torch.as_tensor(detector["spatial_features"], dtype=torch.float32)
        logits = torch.as_tensor(detector["class_logits"], dtype=torch.float32)
        labels = torch.as_tensor(detector["predicted_labels"], dtype=torch.long)
        scores = torch.as_tensor(detector["scores"], dtype=torch.float32)
        boxes = torch.as_tensor(detector["boxes"], dtype=torch.float32)
        if proposal_indices.min() < 0 or proposal_indices.max() >= spatial.shape[0]:
            raise ValueError("candidate proposal index is outside detector rows")
        selected_labels = labels[proposal_indices]
        if selected_labels.min() < 0 or selected_labels.max() >= num_classes:
            raise ValueError("predicted label is outside num_classes")
        image_h, image_w = (float(value) for value in detector["image_size"])
        if min(image_h, image_w) <= 0.0:
            raise ValueError("image dimensions must be positive")
        selected_boxes = boxes[proposal_indices]
        widths = (selected_boxes[:, 2] - selected_boxes[:, 0]).clamp_min(1e-6)
        heights = (selected_boxes[:, 3] - selected_boxes[:, 1]).clamp_min(1e-6)
        geometry = torch.stack(
            (
                selected_boxes[:, 0] / image_w,
                selected_boxes[:, 1] / image_h,
                selected_boxes[:, 2] / image_w,
                selected_boxes[:, 3] / image_h,
                widths / image_w,
                heights / image_h,
                widths * heights / (image_w * image_h),
                torch.log(widths / heights),
            ),
            dim=1,
        )
        one_hot = F.one_hot(selected_labels, num_classes=num_classes).to(torch.float32)
        detector_features = torch.cat(
            (
                logits[proposal_indices],
                one_hot,
                scores[proposal_indices, None],
                geometry,
            ),
            dim=1,
        )
        box_deltas = torch.as_tensor(trace["box_deltas"], dtype=torch.float32)
        energies = torch.as_tensor(
            trace["action_energies"], dtype=torch.float32
        ).flatten()
        target = torch.as_tensor(
            trace["singleton_delta_u"], dtype=torch.float32
        ).flatten()
        if box_deltas.shape != (count, 4) or energies.shape != (count,):
            raise ValueError("candidate action feature shape mismatch")
        if target.shape != (count,):
            raise ValueError("candidate target shape mismatch")
        roi_rows.append(spatial[proposal_indices].mean(dim=(-2, -1)))
        detector_rows.append(detector_features)
        action_rows.append(torch.cat((box_deltas, energies[:, None]), dim=1))
        target_rows.append(target)
        image_rows.append(
            torch.full((count,), int(detector["image_id"]), dtype=torch.long)
        )
    if not target_rows:
        raise ValueError("records contain no sparse candidate rows")
    result = LocalCandidateRows(
        roi_features=torch.cat(roi_rows),
        detector_features=torch.cat(detector_rows),
        action_features=torch.cat(action_rows),
        target=torch.cat(target_rows),
        image_ids=torch.cat(image_rows),
    )
    _validate_rows(result)
    return result


def move_local_candidate_rows(
    rows: LocalCandidateRows, device: torch.device | str
) -> LocalCandidateRows:
    return LocalCandidateRows(
        roi_features=rows.roi_features.to(device),
        detector_features=rows.detector_features.to(device),
        action_features=rows.action_features.to(device),
        target=rows.target.to(device),
        image_ids=rows.image_ids.to(device),
    )


def fit_local_feature_map(
    rows: LocalCandidateRows, *, pca_dim: int
) -> FrozenLocalFeatureMap:
    _validate_rows(rows)
    if pca_dim <= 0:
        raise ValueError("pca_dim must be positive")
    roi_mean = rows.roi_features.mean(dim=0)
    roi_scale = rows.roi_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized_roi = (rows.roi_features - roi_mean) / roi_scale
    max_dim = min(normalized_roi.shape)
    dimension = min(int(pca_dim), int(max_dim))
    _, _, right = torch.linalg.svd(normalized_roi, full_matrices=False)
    components = right[:dimension].transpose(0, 1).contiguous()
    raw_features = _compose_features(rows, normalized_roi @ components)
    feature_mean = raw_features.mean(dim=0)
    feature_scale = raw_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    return FrozenLocalFeatureMap(
        roi_mean=roi_mean,
        roi_scale=roi_scale,
        roi_components=components,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )


def transform_local_candidate_rows(
    rows: LocalCandidateRows,
    feature_map: FrozenLocalFeatureMap,
    *,
    roi_features: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_rows(rows)
    roi = rows.roi_features if roi_features is None else torch.as_tensor(
        roi_features, dtype=rows.roi_features.dtype, device=rows.roi_features.device
    )
    if roi.shape != rows.roi_features.shape:
        raise ValueError("replacement ROI features must preserve shape")
    latent = (
        (roi - feature_map.roi_mean) / feature_map.roi_scale
    ) @ feature_map.roi_components
    raw_features = _compose_features(rows, latent)
    if raw_features.shape[1] != feature_map.feature_mean.numel():
        raise ValueError("feature map dimension does not match candidate rows")
    return (raw_features - feature_map.feature_mean) / feature_map.feature_scale


def within_image_shuffle_order(
    image_ids: torch.Tensor, *, seed: int
) -> tuple[torch.Tensor, float]:
    ids = torch.as_tensor(image_ids, dtype=torch.long).flatten().cpu()
    if ids.numel() == 0:
        raise ValueError("image_ids must be non-empty")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.arange(ids.numel())
    changed = 0
    for image_id in torch.unique(ids, sorted=True).tolist():
        rows = torch.nonzero(ids.eq(int(image_id)), as_tuple=False).flatten()
        if rows.numel() < 2:
            continue
        shift = int(
            torch.randint(1, rows.numel(), (1,), generator=generator).item()
        )
        order[rows] = rows.roll(shifts=shift)
        changed += int(rows.numel())
    return order, changed / max(1, ids.numel())


def image_balanced_weights(image_ids: torch.Tensor) -> torch.Tensor:
    ids = torch.as_tensor(image_ids, dtype=torch.long).flatten()
    if ids.numel() == 0:
        raise ValueError("image_ids must be non-empty")
    weights = torch.empty(ids.numel(), dtype=torch.float32, device=ids.device)
    for image_id in torch.unique(ids, sorted=True).tolist():
        mask = ids.eq(int(image_id))
        weights[mask] = 1.0 / float(mask.sum().item())
    return weights


def fit_ridge_regression(
    features: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    l2: float,
) -> RidgeRegressor:
    x = torch.as_tensor(features)
    y = torch.as_tensor(target, dtype=x.dtype, device=x.device).flatten()
    if x.ndim != 2 or x.shape[0] != y.numel() or x.shape[0] == 0:
        raise ValueError("features and target must share non-empty rows")
    if l2 < 0.0 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("ridge inputs and l2 must be finite and non-negative")
    weights_input = (
        torch.ones(y.numel(), dtype=x.dtype, device=x.device)
        if sample_weights is None
        else torch.as_tensor(sample_weights, dtype=x.dtype, device=x.device).flatten()
    )
    if weights_input.shape != y.shape or (weights_input <= 0.0).any():
        raise ValueError("sample weights must be positive and match target rows")
    solve_dtype = torch.float64
    x_solve = x.to(solve_dtype)
    y_solve = y.to(solve_dtype)
    normalized_weights = weights_input.to(solve_dtype)
    normalized_weights = normalized_weights / normalized_weights.sum()
    x_mean = (normalized_weights[:, None] * x_solve).sum(dim=0)
    y_mean = (normalized_weights * y_solve).sum()
    centered_x = x_solve - x_mean
    centered_y = y_solve - y_mean
    weighted_x = normalized_weights[:, None] * centered_x
    gram = centered_x.transpose(0, 1) @ weighted_x
    gram = gram + float(l2) * torch.eye(
        x.shape[1], dtype=solve_dtype, device=x.device
    )
    right = centered_x.transpose(0, 1) @ (normalized_weights * centered_y)
    weights = torch.linalg.solve(gram, right)
    bias = y_mean - x_mean @ weights
    return RidgeRegressor(
        weights=weights.to(x.dtype),
        bias=bias.to(x.dtype),
        l2=float(l2),
    )


def predict_ridge_regression(
    model: RidgeRegressor, features: torch.Tensor
) -> torch.Tensor:
    x = torch.as_tensor(
        features, dtype=model.weights.dtype, device=model.weights.device
    )
    if x.ndim != 2 or x.shape[1] != model.weights.numel():
        raise ValueError("ridge feature dimension mismatch")
    return x @ model.weights + model.bias


def evaluate_identifiability_gates(
    payload: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    gates = config["gates"]
    heldout = payload["heldout"]
    support = heldout["support"]
    arms = heldout["arms"]
    full = arms["local_full"]
    full_raw = full["raw"]
    safety = full["frozen_threshold"]
    selection = payload["selection"]["local_full"]
    pairwise_keys = (
        "pairwise_accuracy",
        "pairwise_accuracy_candidate_weighted",
    )
    full_pairwise = [float(full_raw[key]) for key in pairwise_keys]
    fit_pairwise = [float(selection["fit_raw"][key]) for key in pairwise_keys]
    tune_pairwise = [float(selection["tune_raw"][key]) for key in pairwise_keys]
    control_gain = min(
        full_pairwise[index] - float(arms[control]["raw"][key])
        for index, key in enumerate(pairwise_keys)
        for control in (
            "within_image_feature_shuffle",
            "within_image_label_shuffle",
        )
    )
    gate_values = {
        "G0_heldout_support": (
            int(support["candidate_count"]) >= int(gates["min_heldout_candidates"])
            and int(support["image_count"]) >= int(gates["min_heldout_images"])
        ),
        "G1_heldout_ranking": min(full_pairwise)
        >= float(gates["min_heldout_pairwise"]),
        "G2_frozen_threshold_safety": (
            selection["used_identity_fallback"] is False
            and int(safety["selected_count"]) >= int(gates["min_selected_actions"])
            and float(safety["mean_delta_u_per_image"])
            >= float(gates["min_mean_delta_u"])
            and float(safety["mean_delta_u_lcb"])
            > float(gates["min_mean_delta_u_lcb"])
            and float(safety["positive_precision_lift"])
            >= float(gates["min_positive_precision_lift"])
            and float(safety["action_image_rate"])
            <= float(gates["max_action_image_rate"])
        ),
        "G3_generalization_gap": (
            max(fit - final for fit, final in zip(fit_pairwise, full_pairwise))
            <= float(gates["max_fit_to_heldout_pairwise_gap"])
            and max(tune - final for tune, final in zip(tune_pairwise, full_pairwise))
            <= float(gates["max_tune_to_heldout_pairwise_gap"])
        ),
        "G4_control_gain": control_gain >= float(gates["min_gain_over_controls"]),
    }
    return {
        "all_passed": all(gate_values.values()),
        "gates": gate_values,
        "diagnostics": {
            "minimum_control_pairwise_gain": control_gain,
            "fit_to_heldout_pairwise_gaps": [
                fit - final for fit, final in zip(fit_pairwise, full_pairwise)
            ],
            "tune_to_heldout_pairwise_gaps": [
                tune - final for tune, final in zip(tune_pairwise, full_pairwise)
            ],
        },
    }


def _compose_features(rows: LocalCandidateRows, latent: torch.Tensor) -> torch.Tensor:
    delta = rows.action_features[:, :4]
    interactions = (latent.unsqueeze(2) * delta.unsqueeze(1)).flatten(start_dim=1)
    return torch.cat(
        (latent, interactions, rows.detector_features, rows.action_features), dim=1
    )


def _validate_rows(rows: LocalCandidateRows) -> None:
    count = rows.target.numel()
    tensors = (
        rows.roi_features,
        rows.detector_features,
        rows.action_features,
    )
    if count == 0 or any(value.ndim != 2 or value.shape[0] != count for value in tensors):
        raise ValueError("local candidate feature tensors must share non-empty rows")
    if rows.target.shape != (count,) or rows.image_ids.shape != (count,):
        raise ValueError("local candidate targets and image IDs must be flat")
    if rows.action_features.shape[1] < 5:
        raise ValueError("action features must contain box delta and energy")
    if not all(torch.isfinite(value).all() for value in (*tensors, rows.target)):
        raise ValueError("local candidate rows must be finite")
