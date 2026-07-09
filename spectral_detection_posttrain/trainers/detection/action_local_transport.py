"""Trainer helpers for action-local energy-guided ROI transport."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms

from spectral_detection_posttrain.core.matching.box_iou import box_iou
from spectral_detection_posttrain.methods.energy_transport import (
    HighWaterMarkLossConfig,
    ROIActionState,
    ROITransportActions,
    ap75_boundary_weights,
    apply_box_delta,
    high_water_mark_action_loss,
    transport_action_energy,
)
from spectral_detection_posttrain.trainers.detection.roi_state import (
    build_roi_action_state_from_predictions,
)


@dataclass(frozen=True)
class ProposalActionBatch:
    """Proposal-aligned action state plus training/eval side information."""

    state: ROIActionState
    matched_gt_boxes: torch.Tensor
    matched_gt_labels: torch.Tensor
    image_sizes: list[tuple[int, int]]


@dataclass(frozen=True)
class SupervisedActionLossConfig:
    """Loss weights for supervised local box transport."""

    target_iou: float = 0.75
    positive_iou_floor: float = 0.50
    positive_iou_ceiling: float = 0.75
    boundary_band: float = 0.10
    min_boundary_weight: float = 0.05
    max_box_delta: float = 0.20

    box_weight: float = 1.0
    high_iou_preserve_weight: float = 0.20
    energy_weight: float = 0.01
    gate_loss_weight: float = 0.0
    gate_actions: bool = False

    hwm_weight: float = 0.0
    hwm_epsilon: float = 0.05
    hwm_box_weight: float = 1.0
    hwm_keep_weight: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "target_iou",
            "positive_iou_floor",
            "positive_iou_ceiling",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.positive_iou_floor > self.positive_iou_ceiling:
            raise ValueError("positive_iou_floor must be <= positive_iou_ceiling")
        if self.boundary_band <= 0.0:
            raise ValueError("boundary_band must be positive")
        if self.max_box_delta <= 0.0:
            raise ValueError("max_box_delta must be positive")
        for name in (
            "min_boundary_weight",
            "box_weight",
            "high_iou_preserve_weight",
            "energy_weight",
            "gate_loss_weight",
            "hwm_weight",
            "hwm_epsilon",
            "hwm_box_weight",
            "hwm_keep_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")


def encode_box_delta(
    boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    *,
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Encode target boxes as standard center-size deltas from proposal boxes."""

    if boxes.shape != target_boxes.shape or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes and target_boxes must share shape (B, 4)")
    if min(weights) <= 0.0:
        raise ValueError("weights must be positive")

    wx, wy, ww, wh = weights
    widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(1e-6)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(1e-6)
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    target_widths = (target_boxes[:, 2] - target_boxes[:, 0]).clamp_min(1e-6)
    target_heights = (target_boxes[:, 3] - target_boxes[:, 1]).clamp_min(1e-6)
    target_ctr_x = target_boxes[:, 0] + 0.5 * target_widths
    target_ctr_y = target_boxes[:, 1] + 0.5 * target_heights

    dx = wx * (target_ctr_x - ctr_x) / widths
    dy = wy * (target_ctr_y - ctr_y) / heights
    dw = ww * torch.log(target_widths / widths)
    dh = wh * torch.log(target_heights / heights)
    return torch.stack((dx, dy, dw, dh), dim=-1)


@torch.no_grad()
def extract_proposal_action_batch(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    targets: list[dict[str, torch.Tensor]] | None = None,
    *,
    match_mode: str = "class_aware",
    box_base: str = "decoded",
) -> ProposalActionBatch:
    """Extract proposal ROI features and matched GT boxes from a detector."""

    if box_base not in {"decoded", "proposal"}:
        raise ValueError("box_base must be 'decoded' or 'proposal'")

    transformed, transformed_targets = model.transform(images, targets)
    features = model.backbone(transformed.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    proposals, _ = model.rpn(transformed, features, transformed_targets)
    roi_features = model.roi_heads.box_roi_pool(
        features, proposals, transformed.image_sizes
    )
    roi_features = model.roi_heads.box_head(roi_features)
    logits, box_regression = model.roi_heads.box_predictor(roi_features)
    scores, labels = _foreground_scores_and_labels(logits)
    base_boxes = (
        _select_predicted_boxes(model, proposals, box_regression, labels)
        if box_base == "decoded"
        else torch.cat(proposals, dim=0)
    )

    predictions: list[dict[str, torch.Tensor]] = []
    offset = 0
    for proposal in proposals:
        count = int(proposal.shape[0])
        predictions.append(
            {
                "boxes": base_boxes[offset : offset + count],
                "scores": scores[offset : offset + count],
                "labels": labels[offset : offset + count],
            }
        )
        offset += count

    state = build_roi_action_state_from_predictions(
        predictions,
        roi_features.detach(),
        transformed_targets,
        logits=logits.detach(),
    )
    if transformed_targets is not None:
        state = rematch_roi_action_state(
            state,
            transformed_targets,
            match_mode=match_mode,
        )
    matched_boxes, matched_labels = _gather_matched_gt(state, transformed_targets)
    return ProposalActionBatch(
        state=state,
        matched_gt_boxes=matched_boxes,
        matched_gt_labels=matched_labels,
        image_sizes=list(transformed.image_sizes),
    )


def rematch_roi_action_state(
    state: ROIActionState,
    targets: list[dict[str, torch.Tensor]],
    *,
    match_mode: str = "class_aware",
) -> ROIActionState:
    """Recompute proposal-to-GT matches with class-aware or class-agnostic IoU."""

    if match_mode not in {"class_aware", "class_agnostic"}:
        raise ValueError("match_mode must be 'class_aware' or 'class_agnostic'")
    matched_gt = torch.full_like(state.labels, -1)
    matched_iou = state.scores.new_zeros(state.scores.shape)

    for image_idx, target in enumerate(targets):
        mask = state.image_indices == image_idx
        if not mask.any():
            continue
        boxes = state.boxes[mask]
        labels = state.labels[mask]
        gt_boxes = target.get("boxes", torch.empty(0, 4)).to(
            device=state.boxes.device,
            dtype=state.boxes.dtype,
        )
        gt_labels = target.get("labels", torch.empty(0, dtype=torch.long)).to(
            device=state.labels.device,
            dtype=state.labels.dtype,
        )
        if boxes.numel() == 0 or gt_boxes.numel() == 0:
            continue

        ious = box_iou(boxes, gt_boxes)
        if match_mode == "class_aware":
            valid = labels[:, None] == gt_labels[None, :]
        else:
            valid = torch.ones_like(ious, dtype=torch.bool)
        masked = ious.masked_fill(~valid, -1.0)
        best_iou, best_gt = masked.max(dim=1)
        valid_match = best_iou > 0.0
        rows = torch.nonzero(mask, as_tuple=False).flatten()
        matched_gt[rows[valid_match]] = best_gt[valid_match].long()
        matched_iou[rows[valid_match]] = best_iou[valid_match].to(matched_iou.dtype)

    return ROIActionState(
        features=state.features,
        boxes=state.boxes,
        scores=state.scores,
        labels=state.labels,
        image_indices=state.image_indices,
        proposal_indices=state.proposal_indices,
        logits=state.logits,
        matched_gt_indices=matched_gt,
        ious=matched_iou,
        verifier_scores=state.verifier_scores,
    )


def effective_actions(
    actions: ROITransportActions,
    *,
    gate_actions: bool,
) -> ROITransportActions:
    """Apply keep-logit as action strength when gate mode is enabled."""

    if not gate_actions:
        return actions
    strength = torch.sigmoid(actions.keep_logit)
    return ROITransportActions(
        feature_delta=actions.feature_delta * strength[:, None],
        score_delta=actions.score_delta * strength,
        box_delta=actions.box_delta * strength[:, None],
        keep_logit=actions.keep_logit,
    )


def supervised_action_transport_loss(
    batch: ProposalActionBatch,
    actions: ROITransportActions,
    *,
    config: SupervisedActionLossConfig | None = None,
    teacher_actions: ROITransportActions | None = None,
) -> dict[str, torch.Tensor]:
    """Compute supervised local transport loss for proposal-to-GT box actions."""

    cfg = config or SupervisedActionLossConfig()
    state = batch.state
    zero = actions.box_delta.sum() * 0.0
    if state.batch_size == 0 or state.ious is None or state.matched_gt_indices is None:
        return _loss_dict(zero, zero, zero, zero, zero, zero, zero)

    eff = effective_actions(actions, gate_actions=cfg.gate_actions)
    matched = state.matched_gt_indices >= 0
    fg = state.labels >= 1
    ious = state.ious
    local_positive = (
        matched
        & fg
        & (ious >= float(cfg.positive_iou_floor))
        & (ious < float(cfg.positive_iou_ceiling))
    )

    boundary_weights = ap75_boundary_weights(
        ious,
        center=float(cfg.target_iou),
        band=float(cfg.boundary_band),
        min_weight=float(cfg.min_boundary_weight),
    )

    loss_box = zero
    if local_positive.any() and cfg.box_weight > 0.0:
        target_delta = encode_box_delta(
            state.boxes[local_positive],
            batch.matched_gt_boxes[local_positive],
        ).clamp(-float(cfg.max_box_delta), float(cfg.max_box_delta))
        per_coord = F.smooth_l1_loss(
            eff.box_delta[local_positive],
            target_delta.to(eff.box_delta.dtype),
            reduction="none",
            beta=0.05,
        )
        per_sample = per_coord.mean(dim=-1)
        weights = boundary_weights[local_positive].to(per_sample.dtype)
        loss_box = float(cfg.box_weight) * _weighted_mean(per_sample, weights)

    high_iou = matched & fg & (ious >= float(cfg.target_iou))
    loss_preserve = zero
    if high_iou.any() and cfg.high_iou_preserve_weight > 0.0:
        per_sample = eff.box_delta[high_iou].pow(2).mean(dim=-1)
        weights = ious[high_iou].to(per_sample.dtype)
        loss_preserve = float(cfg.high_iou_preserve_weight) * _weighted_mean(
            per_sample,
            weights,
        )

    loss_gate = zero
    if cfg.gate_loss_weight > 0.0:
        gate_mask = matched & fg & (ious >= float(cfg.positive_iou_floor))
        if gate_mask.any():
            target_gate = (
                ious[gate_mask] < float(cfg.target_iou)
            ).to(actions.keep_logit.dtype)
            per_sample = F.binary_cross_entropy_with_logits(
                actions.keep_logit[gate_mask],
                target_gate,
                reduction="none",
            )
            weights = boundary_weights[gate_mask].to(per_sample.dtype)
            loss_gate = float(cfg.gate_loss_weight) * _weighted_mean(
                per_sample,
                weights,
            )

    loss_energy = zero
    if cfg.energy_weight > 0.0:
        loss_energy = float(cfg.energy_weight) * transport_action_energy(
            actions.feature_delta,
            score_delta=actions.score_delta,
            box_delta=actions.box_delta,
        )

    loss_hwm = zero
    if teacher_actions is not None and cfg.hwm_weight > 0.0:
        hwm = high_water_mark_action_loss(
            state,
            actions,
            teacher_actions,
            HighWaterMarkLossConfig(
                lambda_anchor=float(cfg.hwm_weight),
                epsilon=float(cfg.hwm_epsilon),
                iou_center=float(cfg.target_iou),
                iou_band=float(cfg.boundary_band),
                min_boundary_weight=float(cfg.min_boundary_weight),
                feature_weight=0.0,
                score_weight=0.0,
                box_weight=float(cfg.hwm_box_weight),
                keep_weight=float(cfg.hwm_keep_weight),
            ),
        )
        loss_hwm = hwm["loss_hwm_total"]

    total = loss_box + loss_preserve + loss_gate + loss_energy + loss_hwm
    return _loss_dict(
        total,
        loss_box,
        loss_preserve,
        loss_gate,
        loss_energy,
        loss_hwm,
        boundary_weights.detach().mean(),
    )


@torch.no_grad()
def action_batch_to_predictions(
    batch: ProposalActionBatch,
    actions: ROITransportActions | None = None,
    *,
    gate_actions: bool = False,
    score_threshold: float = 0.001,
    nms_threshold: float = 0.50,
    detections_per_img: int = 300,
) -> list[dict[str, torch.Tensor]]:
    """Convert proposal state and optional actions into detection predictions."""

    state = batch.state
    if actions is None:
        actions = _zero_actions(state)
    actions = effective_actions(actions, gate_actions=gate_actions)
    outputs: list[dict[str, torch.Tensor]] = []

    for image_idx, image_size in enumerate(batch.image_sizes):
        mask = state.image_indices == image_idx
        if not mask.any():
            outputs.append(_empty_prediction(state.boxes.device, state.boxes.dtype))
            continue
        boxes = apply_box_delta(
            state.boxes[mask],
            actions.box_delta[mask],
            image_size=image_size,
        )
        scores = (state.scores[mask] + actions.score_delta[mask]).clamp(0.0, 1.0)
        labels = state.labels[mask]

        keep = scores >= float(score_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        if boxes.numel() > 0:
            keep_idx = batched_nms(boxes, scores, labels, float(nms_threshold))
            keep_idx = keep_idx[: int(detections_per_img)]
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]
            labels = labels[keep_idx]

        outputs.append(
            {
                "boxes": boxes.detach().cpu(),
                "scores": scores.detach().cpu(),
                "labels": labels.detach().cpu(),
            }
        )
    return outputs


def _foreground_scores_and_labels(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits, dim=-1)
    if probs.shape[1] <= 1:
        scores, labels = probs.max(dim=1)
        return scores, labels.long()
    fg_scores, fg_labels = probs[:, 1:].max(dim=1)
    return fg_scores, (fg_labels + 1).long()


def _select_predicted_boxes(
    model: torch.nn.Module,
    proposals: list[torch.Tensor],
    box_regression: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Decode detector bbox regression for each proposal's predicted label."""

    proposal_boxes = torch.cat(proposals, dim=0)
    if proposal_boxes.numel() == 0:
        return proposal_boxes
    if box_regression.ndim != 2:
        raise ValueError("box_regression must have shape (N, C*4) or (N, 4)")
    if box_regression.shape[0] != proposal_boxes.shape[0]:
        raise ValueError("box_regression rows must match proposals")

    if box_regression.shape[1] == 4:
        deltas = box_regression
    else:
        if box_regression.shape[1] % 4 != 0:
            raise ValueError("box_regression second dim must be divisible by 4")
        num_classes = box_regression.shape[1] // 4
        labels_clamped = labels.clamp(0, num_classes - 1).long()
        start = labels_clamped * 4
        coord_idx = start[:, None] + torch.arange(4, device=labels.device)[None, :]
        deltas = box_regression.gather(1, coord_idx)

    box_coder = getattr(getattr(model, "roi_heads", None), "box_coder", None)
    weights = getattr(box_coder, "weights", (1.0, 1.0, 1.0, 1.0))
    return apply_box_delta(proposal_boxes, deltas, weights=tuple(float(v) for v in weights))


def _gather_matched_gt(
    state: ROIActionState,
    targets: list[dict[str, torch.Tensor]] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    matched_boxes = state.boxes.new_zeros(state.boxes.shape)
    matched_labels = state.labels.new_zeros(state.labels.shape)
    if (
        targets is None
        or state.matched_gt_indices is None
        or state.matched_gt_indices.numel() == 0
    ):
        return matched_boxes, matched_labels

    for image_idx, target in enumerate(targets):
        mask = state.image_indices == image_idx
        if not mask.any():
            continue
        indices = state.matched_gt_indices[mask]
        valid = indices >= 0
        if not valid.any():
            continue
        local_rows = torch.nonzero(mask, as_tuple=False).flatten()[valid]
        gt_boxes = target["boxes"].to(device=state.boxes.device, dtype=state.boxes.dtype)
        gt_labels = target["labels"].to(device=state.labels.device, dtype=state.labels.dtype)
        matched_boxes[local_rows] = gt_boxes[indices[valid].long()]
        matched_labels[local_rows] = gt_labels[indices[valid].long()]
    return matched_boxes, matched_labels


def _zero_actions(state: ROIActionState) -> ROITransportActions:
    return ROITransportActions(
        feature_delta=state.features.new_zeros(state.features.shape),
        score_delta=state.scores.new_zeros(state.scores.shape),
        box_delta=state.boxes.new_zeros(state.boxes.shape),
        keep_logit=state.scores.new_zeros(state.scores.shape),
    )


def _empty_prediction(device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.empty((0, 4), device=device, dtype=dtype).cpu(),
        "scores": torch.empty((0,), device=device, dtype=dtype).cpu(),
        "labels": torch.empty((0,), device=device, dtype=torch.long).cpu(),
    }


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def _loss_dict(
    total: torch.Tensor,
    box: torch.Tensor,
    preserve: torch.Tensor,
    gate: torch.Tensor,
    energy: torch.Tensor,
    hwm: torch.Tensor,
    boundary_weight: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "loss_total": total,
        "loss_box": box.detach(),
        "loss_high_iou_preserve": preserve.detach(),
        "loss_gate": gate.detach(),
        "loss_energy": energy.detach(),
        "loss_hwm": hwm.detach(),
        "mean_boundary_weight": boundary_weight.detach(),
    }
