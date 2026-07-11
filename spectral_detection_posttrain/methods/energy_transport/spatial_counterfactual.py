"""Action-aligned spatial ROI features for Delta-U diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from spectral_detection_posttrain.methods.energy_transport.linear_identifiability import (
    extract_local_candidate_rows,
)


@dataclass(frozen=True)
class SpatialCandidateRows:
    spatial_features: torch.Tensor
    detector_features: torch.Tensor
    action_features: torch.Tensor
    target: torch.Tensor
    image_ids: torch.Tensor


@dataclass(frozen=True)
class FrozenSpatialFeatureMap:
    channel_mean: torch.Tensor
    channel_scale: torch.Tensor
    channel_components: torch.Tensor
    source_mean: torch.Tensor
    source_scale: torch.Tensor
    source_components: torch.Tensor
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    block_count: int


def extract_spatial_candidate_rows(
    records: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    *,
    num_classes: int,
) -> SpatialCandidateRows:
    local = extract_local_candidate_rows(records, num_classes=num_classes)
    spatial_rows: list[torch.Tensor] = []
    for detector, trace in records:
        indices = torch.as_tensor(trace["proposal_indices"], dtype=torch.long).flatten()
        if indices.numel():
            spatial = torch.as_tensor(detector["spatial_features"], dtype=torch.float32)
            spatial_rows.append(spatial[indices])
    if not spatial_rows:
        raise ValueError("records contain no spatial candidate rows")
    rows = SpatialCandidateRows(
        spatial_features=torch.cat(spatial_rows),
        detector_features=local.detector_features,
        action_features=local.action_features,
        target=local.target,
        image_ids=local.image_ids,
    )
    _validate_rows(rows)
    return rows


def move_spatial_candidate_rows(
    rows: SpatialCandidateRows, device: torch.device | str
) -> SpatialCandidateRows:
    return SpatialCandidateRows(
        spatial_features=rows.spatial_features.to(device),
        detector_features=rows.detector_features.to(device),
        action_features=rows.action_features.to(device),
        target=rows.target.to(device),
        image_ids=rows.image_ids.to(device),
    )


def replace_action_features(
    rows: SpatialCandidateRows, action_features: torch.Tensor
) -> SpatialCandidateRows:
    replacement = torch.as_tensor(
        action_features,
        dtype=rows.action_features.dtype,
        device=rows.action_features.device,
    )
    if replacement.shape != rows.action_features.shape:
        raise ValueError("replacement action features must preserve shape")
    return SpatialCandidateRows(
        spatial_features=rows.spatial_features,
        detector_features=rows.detector_features,
        action_features=replacement,
        target=rows.target,
        image_ids=rows.image_ids,
    )


def spatial_counterfactual_blocks(
    spatial_features: torch.Tensor,
    action_features: torch.Tensor,
    *,
    alignment_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    spatial = torch.as_tensor(spatial_features)
    action = torch.as_tensor(
        action_features, dtype=spatial.dtype, device=spatial.device
    )
    if spatial.ndim != 4 or spatial.shape[-2:] != (7, 7):
        raise ValueError("spatial features must have shape (N,C,7,7)")
    if action.ndim != 2 or action.shape[0] != spatial.shape[0] or action.shape[1] < 4:
        raise ValueError("action features must align with spatial rows")
    height, width = spatial.shape[-2:]
    global_pool = spatial.mean(dim=(-2, -1))
    left = spatial[:, :, :, 0].mean(dim=2)
    right = spatial[:, :, :, -1].mean(dim=2)
    top = spatial[:, :, 0, :].mean(dim=2)
    bottom = spatial[:, :, -1, :].mean(dim=2)
    center = spatial[
        :, :, height // 3 : height - height // 3, width // 3 : width - width // 3
    ].mean(dim=(-2, -1))
    ring = torch.cat(
        (
            spatial[:, :, 0, :],
            spatial[:, :, -1, :],
            spatial[:, :, 1:-1, 0],
            spatial[:, :, 1:-1, -1],
        ),
        dim=2,
    ).mean(dim=2)
    middle_h, middle_w = height // 2, width // 2
    top_left = spatial[:, :, : middle_h + 1, : middle_w + 1].mean(dim=(-2, -1))
    bottom_right = spatial[:, :, middle_h:, middle_w:].mean(dim=(-2, -1))
    top_right = spatial[:, :, : middle_h + 1, middle_w:].mean(dim=(-2, -1))
    bottom_left = spatial[:, :, middle_h:, : middle_w + 1].mean(dim=(-2, -1))
    left_right = right - left
    top_bottom = bottom - top
    ring_center = ring - center
    horizontal_context = 0.5 * (left + right) - center
    vertical_context = 0.5 * (top + bottom) - center
    diagonal_main = bottom_right - top_left
    diagonal_cross = bottom_left - top_right
    delta = (
        action[:, :4]
        if alignment_delta is None
        else torch.as_tensor(
            alignment_delta, dtype=spatial.dtype, device=spatial.device
        )
    )
    if delta.shape != (spatial.shape[0], 4):
        raise ValueError("alignment delta must have shape (N,4)")
    aligned = (
        delta[:, 0, None] * left_right
        + delta[:, 1, None] * top_bottom
        + delta[:, 2, None] * horizontal_context
        + delta[:, 3, None] * vertical_context
    )
    return torch.stack(
        (
            global_pool,
            left_right,
            top_bottom,
            ring_center,
            horizontal_context,
            vertical_context,
            diagonal_main,
            diagonal_cross,
            aligned,
        ),
        dim=1,
    )


def spatial_layout_shuffle(
    spatial_features: torch.Tensor, *, seed: int
) -> tuple[torch.Tensor, float]:
    spatial = torch.as_tensor(spatial_features)
    if spatial.ndim != 4 or spatial.shape[0] == 0:
        raise ValueError("spatial features must have non-empty (N,C,H,W) shape")
    count, channels, height, width = spatial.shape
    flat = spatial.reshape(count, channels, height * width)
    shuffled = torch.empty_like(flat)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    changed = 0
    for row in range(count):
        order = torch.randperm(height * width, generator=generator)
        changed += int(order.ne(torch.arange(height * width)).sum().item())
        shuffled[row] = flat[row, :, order.to(flat.device)]
    return shuffled.reshape_as(spatial), changed / (count * height * width)


def fit_spatial_feature_map(
    rows: SpatialCandidateRows,
    blocks: torch.Tensor,
    *,
    channel_pca_dim: int,
    final_pca_dim: int,
) -> FrozenSpatialFeatureMap:
    _validate_rows(rows)
    source = _validate_blocks(rows, blocks)
    if min(channel_pca_dim, final_pca_dim) <= 0:
        raise ValueError("PCA dimensions must be positive")
    global_block = source[:, 0]
    channel_values = rows.spatial_features.permute(0, 2, 3, 1).reshape(
        -1, rows.spatial_features.shape[1]
    )
    channel_mean = channel_values.mean(dim=0)
    channel_scale = channel_values.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized_channel_values = (channel_values - channel_mean) / channel_scale
    normalized_blocks = source / channel_scale[None, None, :]
    normalized_blocks[:, 0] = (global_block - channel_mean) / channel_scale
    _, _, channel_right = torch.linalg.svd(
        normalized_channel_values, full_matrices=False
    )
    channel_dim = min(int(channel_pca_dim), int(channel_right.shape[0]))
    channel_components = channel_right[:channel_dim].transpose(0, 1).contiguous()
    projected = (normalized_blocks @ channel_components).flatten(start_dim=1)
    source_mean = projected.mean(dim=0)
    source_scale = projected.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized_source = (projected - source_mean) / source_scale
    _, _, source_right = torch.linalg.svd(normalized_source, full_matrices=False)
    final_dim = min(int(final_pca_dim), int(source_right.shape[0]))
    source_components = source_right[:final_dim].transpose(0, 1).contiguous()
    latent = normalized_source @ source_components
    raw_features = _compose_features(rows, latent)
    return FrozenSpatialFeatureMap(
        channel_mean=channel_mean,
        channel_scale=channel_scale,
        channel_components=channel_components,
        source_mean=source_mean,
        source_scale=source_scale,
        source_components=source_components,
        feature_mean=raw_features.mean(dim=0),
        feature_scale=raw_features.std(dim=0, unbiased=False).clamp_min(1e-6),
        block_count=int(source.shape[1]),
    )


def transform_spatial_features(
    rows: SpatialCandidateRows,
    blocks: torch.Tensor,
    feature_map: FrozenSpatialFeatureMap,
) -> torch.Tensor:
    _validate_rows(rows)
    source = _validate_blocks(rows, blocks)
    if source.shape[1] != feature_map.block_count:
        raise ValueError("spatial block count differs from fitted map")
    normalized_blocks = source / feature_map.channel_scale[None, None, :]
    normalized_blocks[:, 0] = (
        source[:, 0] - feature_map.channel_mean
    ) / feature_map.channel_scale
    projected = (normalized_blocks @ feature_map.channel_components).flatten(start_dim=1)
    latent = (
        (projected - feature_map.source_mean) / feature_map.source_scale
    ) @ feature_map.source_components
    raw_features = _compose_features(rows, latent)
    return (raw_features - feature_map.feature_mean) / feature_map.feature_scale


def evaluate_spatial_counterfactual_gates(
    payload: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    gates = config["gates"]
    heldout = payload["heldout"]
    arms = heldout["arms"]
    full = arms["structured_full"]
    raw = full["raw"]
    safety = full["frozen_threshold"]
    selection = payload["selection"]["structured_full"]
    keys = ("pairwise_accuracy", "pairwise_accuracy_candidate_weighted")
    final_values = [float(raw[key]) for key in keys]
    fit_values = [float(selection["fit_raw"][key]) for key in keys]
    tune_values = [float(selection["tune_raw"][key]) for key in keys]
    controls = ("global_pooled", "spatial_layout_shuffle", "delta_alignment_shuffle")
    control_gain = min(
        final_values[index] - float(arms[control]["raw"][key])
        for index, key in enumerate(keys)
        for control in controls
    )
    gate_values = {
        "G0_heldout_support": (
            int(heldout["support"]["candidate_count"])
            >= int(gates["min_heldout_candidates"])
            and int(heldout["support"]["image_count"])
            >= int(gates["min_heldout_images"])
        ),
        "G1_heldout_ranking": min(final_values)
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
            max(fit - final for fit, final in zip(fit_values, final_values))
            <= float(gates["max_fit_to_heldout_pairwise_gap"])
            and max(tune - final for tune, final in zip(tune_values, final_values))
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
                fit - final for fit, final in zip(fit_values, final_values)
            ],
            "tune_to_heldout_pairwise_gaps": [
                tune - final for tune, final in zip(tune_values, final_values)
            ],
        },
    }


def _compose_features(rows: SpatialCandidateRows, latent: torch.Tensor) -> torch.Tensor:
    return torch.cat((latent, rows.detector_features, rows.action_features), dim=1)


def _validate_blocks(rows: SpatialCandidateRows, blocks: torch.Tensor) -> torch.Tensor:
    source = torch.as_tensor(
        blocks,
        dtype=rows.spatial_features.dtype,
        device=rows.spatial_features.device,
    )
    if (
        source.ndim != 3
        or source.shape[0] != rows.target.numel()
        or source.shape[2] != rows.spatial_features.shape[1]
        or not torch.isfinite(source).all()
    ):
        raise ValueError("spatial blocks must have finite (N,B,C) shape")
    return source


def _validate_rows(rows: SpatialCandidateRows) -> None:
    count = rows.target.numel()
    if (
        count == 0
        or rows.spatial_features.ndim != 4
        or rows.spatial_features.shape[0] != count
        or rows.spatial_features.shape[-2:] != (7, 7)
        or rows.detector_features.ndim != 2
        or rows.detector_features.shape[0] != count
        or rows.action_features.ndim != 2
        or rows.action_features.shape[0] != count
        or rows.action_features.shape[1] < 5
        or rows.target.shape != (count,)
        or rows.image_ids.shape != (count,)
    ):
        raise ValueError("spatial candidate rows must share non-empty rows")
    if not all(
        torch.isfinite(value).all()
        for value in (
            rows.spatial_features,
            rows.detector_features,
            rows.action_features,
            rows.target,
        )
    ):
        raise ValueError("spatial candidate rows must be finite")
