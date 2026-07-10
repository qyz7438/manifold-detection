"""NMS-aware set policy primitives using detector-visible proposal signals only."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SetPolicyOutput:
    """Per-proposal action and movement predictions for a candidate action set."""

    action_logits: torch.Tensor
    move_logits: torch.Tensor
    conflict_stats: torch.Tensor


@dataclass(frozen=True)
class SetPolicySelection:
    """The bounded actions selected from a set policy output."""

    box_delta: torch.Tensor
    selected_mask: torch.Tensor
    candidate_indices: torch.Tensor
    selection_scores: torch.Tensor


@dataclass(frozen=True)
class SetPolicyLossConfig:
    """Weights for identity-aware action CE and positive-aware movement BCE."""

    identity_weight: float = 1.0
    move_positive_weight: float = 1.0
    action_weight: float = 1.0
    move_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.identity_weight < 0.0:
            raise ValueError("identity_weight must be non-negative")
        if self.move_positive_weight <= 0.0:
            raise ValueError("move_positive_weight must be positive")
        if self.action_weight < 0.0 or self.move_weight < 0.0:
            raise ValueError("loss weights must be non-negative")


class NMSAwareSetPolicyHead(nn.Module):
    """Score local ROI actions using spatial evidence and observable NMS context.

    The head accepts detector outputs only: ROI features, predicted logits and
    labels, confidence scores, proposal boxes, and the image extent.  It never
    takes ground-truth boxes, labels, or matching assignments.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        candidate_deltas: torch.Tensor,
        hidden_dim: int = 96,
        spatial_size: int = 7,
        energy_weight: float = 0.05,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or num_classes <= 0 or hidden_dim <= 0 or spatial_size <= 0:
            raise ValueError("in_channels, num_classes, hidden_dim, and spatial_size must be positive")
        _validate_candidate_deltas(candidate_deltas)
        if energy_weight < 0.0:
            raise ValueError("energy_weight must be non-negative")

        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.spatial_size = int(spatial_size)
        self.energy_weight = float(energy_weight)
        self.register_buffer("candidate_deltas", candidate_deltas.detach().clone())

        self.spatial_pool = nn.AdaptiveAvgPool2d((self.spatial_size, self.spatial_size))
        self.spatial_encoder = nn.Sequential(
            nn.Linear(self.in_channels * self.spatial_size * self.spatial_size, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        context_dim = 2 * self.num_classes + 1 + 4 + 4
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.action_head = nn.Linear(self.hidden_dim, candidate_deltas.shape[0])
        self.move_head = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        nn.init.zeros_(self.move_head.weight)
        nn.init.zeros_(self.move_head.bias)

    def forward(
        self,
        spatial_features: torch.Tensor,
        class_logits: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        boxes: torch.Tensor,
        image_size: tuple[int, int] | torch.Tensor,
        energy_weight: float | None = None,
    ) -> SetPolicyOutput:
        count = _validate_head_inputs(
            spatial_features, class_logits, labels, scores, boxes, self.in_channels, self.num_classes
        )
        if energy_weight is None:
            energy_weight = self.energy_weight
        if energy_weight < 0.0:
            raise ValueError("energy_weight must be non-negative")

        conflict_stats = class_aware_conflict_statistics(boxes, labels, scores, image_size)
        normalized_boxes = _normalize_boxes(boxes, image_size)
        label_code = F.one_hot(labels.long(), num_classes=self.num_classes).to(dtype=class_logits.dtype)
        context = torch.cat(
            (class_logits, label_code, scores[:, None].to(dtype=class_logits.dtype), normalized_boxes, conflict_stats),
            dim=1,
        )
        spatial = self.spatial_encoder(self.spatial_pool(spatial_features).reshape(count, -1))
        hidden = self.trunk(torch.cat((spatial, self.context_encoder(context)), dim=1))
        action_logits = self.action_head(hidden)
        candidate_energy = self.candidate_deltas.to(dtype=action_logits.dtype).square().sum(dim=1) / 0.04
        action_logits = action_logits - float(energy_weight) * candidate_energy[None, :]
        return SetPolicyOutput(
            action_logits=action_logits,
            move_logits=self.move_head(hidden).squeeze(-1),
            conflict_stats=conflict_stats,
        )


def class_aware_conflict_statistics(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    image_size: tuple[int, int] | torch.Tensor,
    nms_threshold: float = 0.5,
) -> torch.Tensor:
    """Return max/mean/thresholded/higher-score same-class IoU statistics."""
    count = _validate_detection_vectors(boxes, labels, scores)
    if not 0.0 <= nms_threshold <= 1.0:
        raise ValueError("nms_threshold must be in [0, 1]")
    normalized_boxes = _normalize_boxes(boxes, image_size)
    if count == 0:
        return normalized_boxes.new_zeros((0, 4))

    ious = _pairwise_iou(normalized_boxes)
    same_class = labels[:, None].eq(labels[None, :])
    peer_mask = same_class & ~torch.eye(count, device=boxes.device, dtype=torch.bool)
    peer_count = peer_mask.sum(dim=1)
    peer_iou = ious * peer_mask.to(dtype=ious.dtype)
    max_iou = peer_iou.max(dim=1).values
    mean_iou = peer_iou.sum(dim=1) / peer_count.clamp_min(1).to(dtype=ious.dtype)
    over_threshold = (peer_iou >= float(nms_threshold)) & peer_mask
    over_fraction = over_threshold.sum(dim=1).to(dtype=ious.dtype) / peer_count.clamp_min(1).to(dtype=ious.dtype)
    higher_score = scores[None, :] > scores[:, None]
    higher_iou = peer_iou * higher_score.to(dtype=ious.dtype)
    higher_strength = higher_iou.max(dim=1).values
    return torch.stack((max_iou, mean_iou, over_fraction, higher_strength), dim=1)


def set_policy_loss(
    output: SetPolicyOutput,
    action_targets: torch.Tensor,
    move_targets: torch.Tensor,
    supervision_mask: torch.Tensor,
    config: SetPolicyLossConfig,
) -> dict[str, torch.Tensor]:
    """Compute action classification and move-gate losses on supervised rows."""
    count, candidates = _validate_output(output)
    _validate_targets(action_targets, move_targets, supervision_mask, count, candidates)
    supervised = supervision_mask.bool()
    zero = output.action_logits.sum() * 0.0 + output.move_logits.sum() * 0.0
    if not supervised.any():
        return {
            "loss_total": zero,
            "loss_action": zero,
            "loss_move": zero,
            "action_accuracy": zero.detach() + 1.0,
            "move_accuracy": zero.detach() + 1.0,
            "action_positive_accuracy": zero.detach() + 1.0,
        }

    per_action = F.cross_entropy(output.action_logits, action_targets.long(), reduction="none")
    action_weights = torch.where(
        action_targets.eq(0),
        per_action.new_full((count,), float(config.identity_weight)),
        per_action.new_ones((count,)),
    )
    masked_action_weights = action_weights[supervised]
    loss_action = (per_action[supervised] * masked_action_weights).sum() / masked_action_weights.sum().clamp_min(1e-8)
    loss_move = F.binary_cross_entropy_with_logits(
        output.move_logits[supervised],
        move_targets[supervised].to(dtype=output.move_logits.dtype),
        pos_weight=output.move_logits.new_tensor(float(config.move_positive_weight)),
    )
    loss_total = float(config.action_weight) * loss_action + float(config.move_weight) * loss_move
    with torch.no_grad():
        action_predictions = output.action_logits.argmax(dim=1)
        move_predictions = output.move_logits.ge(0.0)
        positive = supervised & move_targets.bool()
        action_positive_accuracy = (
            action_predictions[positive].eq(action_targets[positive]).float().mean()
            if positive.any()
            else loss_total.new_tensor(1.0)
        )
    return {
        "loss_total": loss_total,
        "loss_action": loss_action,
        "loss_move": loss_move,
        "action_accuracy": action_predictions[supervised].eq(action_targets[supervised]).float().mean(),
        "move_accuracy": move_predictions[supervised].eq(move_targets[supervised].bool()).float().mean(),
        "action_positive_accuracy": action_positive_accuracy,
    }


def select_set_policy_actions(
    output: SetPolicyOutput,
    candidate_deltas: torch.Tensor,
    image_indices: torch.Tensor,
    observable_mask: torch.Tensor,
    max_actions_per_image: int = 4,
    move_threshold: float = 0.0,
    require_move_gate: bool = True,
) -> SetPolicySelection:
    """Select at most one positive-margin action per proposal under image budgets."""
    count, candidates = _validate_output(output)
    _validate_candidate_deltas(candidate_deltas)
    if candidate_deltas.shape[0] != candidates:
        raise ValueError("candidate_deltas must match output.action_logits width")
    if image_indices.shape != (count,) or observable_mask.shape != (count,):
        raise ValueError("image_indices and observable_mask must have shape (N,)")
    if max_actions_per_image <= 0:
        raise ValueError("max_actions_per_image must be positive")
    if candidates == 1:
        zero_indices = torch.zeros(count, dtype=torch.long, device=output.action_logits.device)
        return SetPolicySelection(
            box_delta=candidate_deltas.to(device=output.action_logits.device)[zero_indices],
            selected_mask=torch.zeros(count, dtype=torch.bool, device=output.action_logits.device),
            candidate_indices=zero_indices,
            selection_scores=output.move_logits + output.action_logits[:, 0] * 0.0,
        )

    best_non_identity_logits, best_non_identity = output.action_logits[:, 1:].max(dim=1)
    best_non_identity = best_non_identity + 1
    action_margin = best_non_identity_logits - output.action_logits[:, 0]
    selection_scores = output.move_logits + action_margin
    move_eligible = output.move_logits.gt(float(move_threshold))
    if not require_move_gate:
        move_eligible = torch.ones_like(move_eligible)
    eligible = observable_mask.bool() & move_eligible & action_margin.gt(0.0)
    selected_mask = torch.zeros(count, dtype=torch.bool, device=output.action_logits.device)
    image_indices = image_indices.to(device=output.action_logits.device)
    for image_index in torch.unique(image_indices[eligible], sorted=True).tolist():
        rows = torch.nonzero(eligible & image_indices.eq(image_index), as_tuple=False).flatten()
        order = torch.argsort(selection_scores[rows], descending=True)
        selected_mask[rows[order[:max_actions_per_image]]] = True
    candidate_indices = torch.where(selected_mask, best_non_identity, torch.zeros_like(best_non_identity))
    deltas = candidate_deltas.to(device=output.action_logits.device, dtype=output.action_logits.dtype)
    return SetPolicySelection(
        box_delta=deltas[candidate_indices],
        selected_mask=selected_mask,
        candidate_indices=candidate_indices,
        selection_scores=selection_scores,
    )


def _validate_candidate_deltas(candidate_deltas: torch.Tensor) -> None:
    if candidate_deltas.ndim != 2 or candidate_deltas.shape[1] != 4 or candidate_deltas.shape[0] == 0:
        raise ValueError("candidate_deltas must have shape (K, 4) with K >= 1")
    if not torch.isfinite(candidate_deltas).all():
        raise ValueError("candidate_deltas must be finite")
    if candidate_deltas[0].count_nonzero().item() != 0:
        raise ValueError("candidate_deltas[0] must be the all-zero identity")


def _validate_head_inputs(
    spatial_features: torch.Tensor,
    class_logits: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    boxes: torch.Tensor,
    in_channels: int,
    num_classes: int,
) -> int:
    if spatial_features.ndim != 4 or spatial_features.shape[1] != in_channels:
        raise ValueError(f"spatial_features must have shape (N, {in_channels}, H, W)")
    count = spatial_features.shape[0]
    if class_logits.shape != (count, num_classes):
        raise ValueError(f"class_logits must have shape ({count}, {num_classes})")
    _validate_detection_vectors(boxes, labels, scores, count)
    if labels.numel() and (labels.lt(0).any() or labels.ge(num_classes).any()):
        raise ValueError("labels must be in [0, num_classes)")
    return count


def _validate_detection_vectors(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    count: int | None = None,
) -> int:
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    expected_count = boxes.shape[0] if count is None else count
    if boxes.shape[0] != expected_count or labels.shape != (expected_count,) or scores.shape != (expected_count,):
        raise ValueError("boxes, labels, and scores must share proposal count N")
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise ValueError("boxes and scores must be finite")
    return expected_count


def _validate_output(output: SetPolicyOutput) -> tuple[int, int]:
    if output.action_logits.ndim != 2:
        raise ValueError("output.action_logits must have shape (N, K)")
    count, candidates = output.action_logits.shape
    if candidates == 0 or output.move_logits.shape != (count,) or output.conflict_stats.shape != (count, 4):
        raise ValueError("invalid SetPolicyOutput tensor shapes")
    return count, candidates


def _validate_targets(
    action_targets: torch.Tensor,
    move_targets: torch.Tensor,
    supervision_mask: torch.Tensor,
    count: int,
    candidates: int,
) -> None:
    if action_targets.shape != (count,) or move_targets.shape != (count,) or supervision_mask.shape != (count,):
        raise ValueError("action_targets, move_targets, and supervision_mask must have shape (N,)")
    if action_targets.dtype.is_floating_point or action_targets.dtype == torch.bool:
        raise ValueError("action_targets must be integer candidate indices")
    if action_targets.numel() and (action_targets.lt(0).any() or action_targets.ge(candidates).any()):
        raise ValueError("action_targets are outside the candidate set")
    if not torch.isfinite(move_targets).all() or not ((move_targets == 0) | (move_targets == 1)).all():
        raise ValueError("move_targets must be binary")
    selected_identity = supervision_mask.bool() & move_targets.bool() & action_targets.eq(0)
    if selected_identity.any():
        raise ValueError("selected positive action targets must be non-identity")


def _normalize_boxes(boxes: torch.Tensor, image_size: tuple[int, int] | torch.Tensor) -> torch.Tensor:
    if isinstance(image_size, torch.Tensor):
        if image_size.numel() != 2:
            raise ValueError("image_size tensor must contain (height, width)")
        height, width = image_size.reshape(-1).to(device=boxes.device, dtype=boxes.dtype)
    else:
        if len(image_size) != 2:
            raise ValueError("image_size must be (height, width)")
        height, width = image_size
        height = boxes.new_tensor(height)
        width = boxes.new_tensor(width)
    if height.item() <= 0.0 or width.item() <= 0.0:
        raise ValueError("image_size dimensions must be positive")
    scale = torch.stack((width, height, width, height))
    return boxes / scale


def _pairwise_iou(boxes: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(boxes[:, None, :2], boxes[None, :, :2])
    bottom_right = torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0.0).prod(dim=-1)
    area = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0.0).prod(dim=-1)
    return intersection / (area[:, None] + area[None, :] - intersection).clamp_min(1e-8)
