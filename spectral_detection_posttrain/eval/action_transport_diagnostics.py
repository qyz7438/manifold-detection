"""Proposal-level diagnostics for bounded ROI box actions."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from spectral_detection_posttrain.methods.energy_transport import (
    ROIActionState,
    ROITransportActions,
    apply_box_delta,
)


def box_only_actions(
    actions: ROITransportActions,
    *,
    scale: float = 1.0,
) -> ROITransportActions:
    """Keep a scaled box action while making every other action an exact no-op."""

    if scale < 0.0:
        raise ValueError("scale must be non-negative")
    return ROITransportActions(
        feature_delta=torch.zeros_like(actions.feature_delta),
        score_delta=torch.zeros_like(actions.score_delta),
        box_delta=actions.box_delta * float(scale),
        keep_logit=torch.zeros_like(actions.keep_logit),
    )


def permute_box_actions_within_images(
    actions: ROITransportActions,
    image_indices: torch.Tensor,
    *,
    generator: torch.Generator,
) -> ROITransportActions:
    """Break proposal/action alignment without changing each image's action set."""

    if image_indices.shape != actions.score_delta.shape:
        raise ValueError("image_indices must contain one value per action")
    permuted = actions.box_delta.clone()
    for image_idx in torch.unique(image_indices.detach().cpu(), sorted=True).tolist():
        rows = torch.nonzero(image_indices == int(image_idx), as_tuple=False).flatten()
        if rows.numel() <= 1:
            continue
        order = torch.randperm(int(rows.numel()), generator=generator, device="cpu")
        order = order.to(device=rows.device)
        permuted[rows] = actions.box_delta[rows[order]]
    return ROITransportActions(
        feature_delta=torch.zeros_like(actions.feature_delta),
        score_delta=torch.zeros_like(actions.score_delta),
        box_delta=permuted,
        keep_logit=torch.zeros_like(actions.keep_logit),
    )


def oracle_accept_improving_box_actions(
    state: ROIActionState,
    actions: ROITransportActions,
    *,
    matched_gt_boxes: torch.Tensor,
    matched_gt_labels: torch.Tensor,
    image_sizes: list[tuple[int, int]],
    min_iou_gain: float = 0.0,
) -> ROITransportActions:
    """Diagnostic upper bound that keeps an action only when true IoU improves."""

    if min_iou_gain < 0.0:
        raise ValueError("min_iou_gain must be non-negative")
    if state.matched_gt_indices is None:
        raise ValueError("state must include matched_gt_indices")
    post_boxes = _apply_box_actions_by_image(
        state.boxes,
        actions.box_delta,
        state.image_indices,
        image_sizes,
    )
    base_iou = _elementwise_iou(state.boxes, matched_gt_boxes)
    post_iou = _elementwise_iou(post_boxes, matched_gt_boxes)
    accept = (
        state.matched_gt_indices.ge(0)
        & state.labels.eq(matched_gt_labels)
        & ((post_iou - base_iou) > float(min_iou_gain))
    )
    delta = torch.where(accept[:, None], actions.box_delta, torch.zeros_like(actions.box_delta))
    return ROITransportActions(
        feature_delta=torch.zeros_like(actions.feature_delta),
        score_delta=torch.zeros_like(actions.score_delta),
        box_delta=delta,
        keep_logit=torch.zeros_like(actions.keep_logit),
    )


def proposal_transition_tensors(
    state: ROIActionState,
    *,
    matched_gt_boxes: torch.Tensor,
    matched_gt_labels: torch.Tensor,
    box_delta: torch.Tensor,
    image_sizes: list[tuple[int, int]],
) -> dict[str, torch.Tensor]:
    """Measure proposal geometry before and after one decoded box action."""

    count = state.batch_size
    if matched_gt_boxes.shape != (count, 4):
        raise ValueError("matched_gt_boxes must have shape (B, 4)")
    if matched_gt_labels.shape != (count,):
        raise ValueError("matched_gt_labels must have shape (B,)")
    if box_delta.shape != (count, 4):
        raise ValueError("box_delta must have shape (B, 4)")
    if state.matched_gt_indices is None:
        raise ValueError("state must include matched_gt_indices")

    post_boxes = _apply_box_actions_by_image(
        state.boxes,
        box_delta,
        state.image_indices,
        image_sizes,
    )

    matched = state.matched_gt_indices >= 0
    base_iou = _elementwise_iou(state.boxes, matched_gt_boxes)
    post_iou = _elementwise_iou(post_boxes, matched_gt_boxes)
    base_iou = torch.where(matched, base_iou, torch.zeros_like(base_iou))
    post_iou = torch.where(matched, post_iou, torch.zeros_like(post_iou))

    base_center_error, base_size_error = _normalized_geometry_error(
        state.boxes,
        matched_gt_boxes,
    )
    post_center_error, post_size_error = _normalized_geometry_error(
        post_boxes,
        matched_gt_boxes,
    )
    nan = torch.full_like(base_iou, float("nan"))
    base_center_error = torch.where(matched, base_center_error, nan)
    post_center_error = torch.where(matched, post_center_error, nan)
    base_size_error = torch.where(matched, base_size_error, nan)
    post_size_error = torch.where(matched, post_size_error, nan)

    target_delta = _encode_target_delta(state.boxes, matched_gt_boxes)
    action_norm = box_delta.norm(dim=1)
    target_norm = target_delta.norm(dim=1)
    cosine = (box_delta * target_delta).sum(dim=1) / (
        action_norm * target_norm
    ).clamp_min(1e-8)
    cosine_valid = matched & (action_norm > 1e-8) & (target_norm > 1e-8)
    cosine = torch.where(cosine_valid, cosine, nan)

    if state.logits is None:
        background_dominant = torch.zeros_like(matched)
    else:
        probabilities = torch.softmax(state.logits, dim=-1)
        if probabilities.shape[1] <= 1:
            background_dominant = torch.zeros_like(matched)
        else:
            background_dominant = probabilities[:, 0] >= probabilities[:, 1:].amax(dim=1)

    return {
        "matched": matched.detach().cpu(),
        "class_correct": (matched & state.labels.eq(matched_gt_labels)).detach().cpu(),
        "background_dominant": background_dominant.detach().cpu(),
        "base_iou": base_iou.detach().cpu(),
        "post_iou": post_iou.detach().cpu(),
        "delta_iou": (post_iou - base_iou).detach().cpu(),
        "action_l2": box_delta.norm(dim=1).detach().cpu(),
        "action_linf": box_delta.abs().amax(dim=1).detach().cpu(),
        "target_delta_l2": target_norm.detach().cpu(),
        "action_target_cosine": cosine.detach().cpu(),
        "base_center_error": base_center_error.detach().cpu(),
        "post_center_error": post_center_error.detach().cpu(),
        "base_size_error": base_size_error.detach().cpu(),
        "post_size_error": post_size_error.detach().cpu(),
    }


def concatenate_transition_batches(
    batches: list[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not batches:
        return {}
    keys = tuple(batches[0].keys())
    if any(tuple(batch.keys()) != keys for batch in batches):
        raise ValueError("transition batches must share keys")
    return {key: torch.cat([batch[key] for batch in batches], dim=0) for key in keys}


def summarize_proposal_transitions(
    transitions: Mapping[str, torch.Tensor],
    *,
    threshold: float = 0.75,
) -> dict[str, dict[str, float | int | None]]:
    """Summarize action magnitude and localization changes by proposal group."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if not transitions:
        return {}

    matched = transitions["matched"].bool()
    class_correct = transitions["class_correct"].bool()
    background = transitions["background_dominant"].bool()
    base_iou = transitions["base_iou"]
    all_rows = torch.ones_like(matched)

    groups: dict[str, torch.Tensor] = {
        "all": all_rows,
        "matched": matched,
        "unmatched": ~matched,
        "class_correct": class_correct,
        "class_wrong": matched & ~class_correct,
        "background_dominant": background,
        "foreground_dominant": ~background,
    }
    intervals = (
        ("iou_0_0.3", 0.0, 0.3, False),
        ("iou_0.3_0.5", 0.3, 0.5, False),
        ("iou_0.5_0.75", 0.5, 0.75, False),
        ("iou_0.75_1.01", 0.75, 1.01, True),
    )
    for name, low, high, include_high in intervals:
        upper = base_iou <= high if include_high else base_iou < high
        mask = matched & (base_iou >= low) & upper
        groups[name] = mask
        groups[f"class_correct_{name}"] = mask & class_correct
        groups[f"class_wrong_{name}"] = mask & ~class_correct

    return {
        name: _summarize_group(transitions, mask, threshold=float(threshold))
        for name, mask in groups.items()
    }


def _summarize_group(
    transitions: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, float | int | None]:
    mask = mask.bool()
    matched_mask = mask & transitions["matched"].bool()
    base_iou = transitions["base_iou"]
    post_iou = transitions["post_iou"]
    delta_iou = transitions["delta_iou"]
    improved = matched_mask & (delta_iou > 1e-6)
    degraded = matched_mask & (delta_iou < -1e-6)
    promoted = matched_mask & (base_iou < threshold) & (post_iou >= threshold)
    demoted = matched_mask & (base_iou >= threshold) & (post_iou < threshold)

    result: dict[str, float | int | None] = {
        "count": int(mask.sum().item()),
        "matched_count": int(matched_mask.sum().item()),
        "improved_count": int(improved.sum().item()),
        "degraded_count": int(degraded.sum().item()),
        "promoted_75": int(promoted.sum().item()),
        "demoted_75": int(demoted.sum().item()),
        "action_l2_mean": _mean(transitions["action_l2"], mask),
        "action_l2_median": _quantile(transitions["action_l2"], mask, 0.5),
        "action_l2_p95": _quantile(transitions["action_l2"], mask, 0.95),
        "action_linf_mean": _mean(transitions["action_linf"], mask),
        "target_delta_l2_mean": _mean(transitions["target_delta_l2"], matched_mask),
        "action_target_cosine_mean": _mean(
            transitions["action_target_cosine"],
            matched_mask,
        ),
        "base_iou_mean": _mean(base_iou, matched_mask),
        "post_iou_mean": _mean(post_iou, matched_mask),
        "delta_iou_mean": _mean(delta_iou, matched_mask),
        "delta_iou_median": _quantile(delta_iou, matched_mask, 0.5),
        "base_center_error_mean": _mean(transitions["base_center_error"], matched_mask),
        "post_center_error_mean": _mean(transitions["post_center_error"], matched_mask),
        "base_size_error_mean": _mean(transitions["base_size_error"], matched_mask),
        "post_size_error_mean": _mean(transitions["post_size_error"], matched_mask),
    }
    matched_count = max(1, int(matched_mask.sum().item()))
    result["improved_rate"] = float(improved.sum().item()) / matched_count
    result["degraded_rate"] = float(degraded.sum().item()) / matched_count
    return result


def _elementwise_iou(boxes: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(boxes[:, :2], targets[:, :2])
    bottom_right = torch.minimum(boxes[:, 2:], targets[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0.0).prod(dim=1)
    box_area = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0.0).prod(dim=1)
    target_area = (targets[:, 2:] - targets[:, :2]).clamp_min(0.0).prod(dim=1)
    union = box_area + target_area - intersection
    return intersection / union.clamp_min(1e-8)


def _apply_box_actions_by_image(
    boxes: torch.Tensor,
    box_delta: torch.Tensor,
    image_indices: torch.Tensor,
    image_sizes: list[tuple[int, int]],
) -> torch.Tensor:
    post_boxes = boxes.clone()
    for image_idx, image_size in enumerate(image_sizes):
        mask = image_indices == image_idx
        if mask.any():
            post_boxes[mask] = apply_box_delta(
                boxes[mask],
                box_delta[mask],
                image_size=image_size,
            )
    return post_boxes


def _normalized_geometry_error(
    boxes: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    box_size = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-6)
    target_size = (targets[:, 2:] - targets[:, :2]).clamp_min(1e-6)
    box_center = 0.5 * (boxes[:, :2] + boxes[:, 2:])
    target_center = 0.5 * (targets[:, :2] + targets[:, 2:])
    center_error = ((box_center - target_center) / target_size).norm(dim=1)
    size_error = (torch.log(box_size / target_size)).abs().mean(dim=1)
    return center_error, size_error


def _encode_target_delta(boxes: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    box_size = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-6)
    target_size = (targets[:, 2:] - targets[:, :2]).clamp_min(1e-6)
    box_center = 0.5 * (boxes[:, :2] + boxes[:, 2:])
    target_center = 0.5 * (targets[:, :2] + targets[:, 2:])
    offset = (target_center - box_center) / box_size
    scale = torch.log(target_size / box_size)
    return torch.cat((offset, scale), dim=1)


def _mean(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    selected = values[mask]
    selected = selected[torch.isfinite(selected)]
    if selected.numel() == 0:
        return None
    return float(selected.float().mean().item())


def _quantile(values: torch.Tensor, mask: torch.Tensor, q: float) -> float | None:
    selected = values[mask]
    selected = selected[torch.isfinite(selected)]
    if selected.numel() == 0:
        return None
    return float(torch.quantile(selected.float(), q).item())
