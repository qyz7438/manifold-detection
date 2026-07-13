"""Joint Delta-U probing, supervision, and bounded proposal action selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ProposalSetEdges:
    """Directed within-image proposal edges with fixed eight-value features."""

    edge_index: torch.Tensor
    edge_features: torch.Tensor
    node_count: int


@dataclass(frozen=True)
class GroupHeldoutSplit:
    """A group-level split projected back onto proposal rows."""

    train_mask: torch.Tensor
    heldout_mask: torch.Tensor
    train_groups: torch.Tensor
    heldout_groups: torch.Tensor


@dataclass(frozen=True)
class JointDeltaUSelection:
    """One bounded candidate action per proposal, subject to image budgets."""

    candidate_indices: torch.Tensor
    selected_mask: torch.Tensor
    box_delta: torch.Tensor
    delta_u: torch.Tensor


def build_proposal_set_edges(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    image_indices: torch.Tensor,
    image_sizes: torch.Tensor,
    max_neighbors: int = 32,
    same_class_only: bool = True,
) -> ProposalSetEdges:
    """Connect each proposal to its highest-IoU neighbors in the same image.

    Edge rows are ``[IoU, dx, dy, log_w_ratio, log_h_ratio, score_diff,
    source_higher, same_class]``.  ``dx`` and ``dy`` are target-minus-source
    center offsets normalized by the source image width and height.
    """
    count = _validate_proposal_tensors(boxes, labels, scores, image_indices)
    if max_neighbors < 0:
        raise ValueError("max_neighbors must be non-negative")
    if count == 0 or max_neighbors == 0:
        return ProposalSetEdges(
            edge_index=torch.empty((2, 0), dtype=torch.long, device=boxes.device),
            edge_features=boxes.new_zeros((0, 8)),
            node_count=count,
        )

    sizes = _image_sizes_per_row(image_sizes, image_indices, count, boxes)
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for image_index in torch.unique(image_indices, sorted=True):
        rows = torch.nonzero(image_indices.eq(image_index), as_tuple=False).flatten()
        if rows.numel() < 2:
            continue
        local_boxes = boxes[rows]
        local_iou = _pairwise_iou(local_boxes, local_boxes)
        for local_source, source in enumerate(rows):
            allowed = torch.ones(rows.numel(), dtype=torch.bool, device=boxes.device)
            allowed[local_source] = False
            if same_class_only:
                allowed &= labels[rows].eq(labels[source])
            local_targets = torch.nonzero(allowed, as_tuple=False).flatten()
            if local_targets.numel() == 0:
                continue
            ranking = torch.argsort(local_iou[local_source, local_targets], descending=True)
            chosen = local_targets[ranking[:max_neighbors]]
            sources.append(source.expand(chosen.numel()))
            targets.append(rows[chosen])

    if not sources:
        return ProposalSetEdges(
            edge_index=torch.empty((2, 0), dtype=torch.long, device=boxes.device),
            edge_features=boxes.new_zeros((0, 8)),
            node_count=count,
        )

    source = torch.cat(sources)
    target = torch.cat(targets)
    source_boxes = boxes[source]
    target_boxes = boxes[target]
    source_widths, source_heights, source_centers = _box_geometry(source_boxes)
    target_widths, target_heights, target_centers = _box_geometry(target_boxes)
    eps = torch.finfo(boxes.dtype).eps if boxes.is_floating_point() else 1e-6
    source_sizes = sizes[source]
    features = torch.stack(
        (
            _elementwise_iou(source_boxes, target_boxes),
            (target_centers[:, 0] - source_centers[:, 0]) / source_sizes[:, 1].clamp_min(eps),
            (target_centers[:, 1] - source_centers[:, 1]) / source_sizes[:, 0].clamp_min(eps),
            torch.log(target_widths / source_widths.clamp_min(eps)),
            torch.log(target_heights / source_heights.clamp_min(eps)),
            scores[source].to(boxes.dtype) - scores[target].to(boxes.dtype),
            scores[source].gt(scores[target]).to(boxes.dtype),
            labels[source].eq(labels[target]).to(boxes.dtype),
        ),
        dim=1,
    )
    return ProposalSetEdges(
        edge_index=torch.stack((source, target)),
        edge_features=features,
        node_count=count,
    )


def group_heldout_split(
    group_ids: torch.Tensor,
    heldout_fraction: float,
    seed: int,
) -> GroupHeldoutSplit:
    """Deterministically assign complete image groups to train or heldout."""
    if group_ids.ndim != 1:
        raise ValueError("group_ids must have shape (N,)")
    if not 0.0 <= heldout_fraction <= 1.0:
        raise ValueError("heldout_fraction must be in [0, 1]")

    groups = torch.unique(group_ids, sorted=True)
    group_count = groups.numel()
    heldout_count = int(round(group_count * float(heldout_fraction)))
    if 0.0 < heldout_fraction < 1.0 and group_count > 1:
        heldout_count = min(max(heldout_count, 1), group_count - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.randperm(group_count, generator=generator, device="cpu").to(groups.device)
    heldout_groups = groups[order[:heldout_count]]
    train_groups = groups[order[heldout_count:]]
    heldout_mask = torch.isin(group_ids, heldout_groups)
    return GroupHeldoutSplit(
        train_mask=~heldout_mask,
        heldout_mask=heldout_mask,
        train_groups=train_groups,
        heldout_groups=heldout_groups,
    )


class JointDeltaUProbe(nn.Module):
    """Score candidate action utility from ROI evidence and proposal-set context."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        candidate_deltas: torch.Tensor,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        _validate_candidate_deltas(candidate_deltas)
        if min(in_channels, num_classes, hidden_dim) <= 0:
            raise ValueError("probe dimensions must be positive")
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.num_candidates = int(candidate_deltas.shape[0])
        self.hidden_dim = int(hidden_dim)
        self.register_buffer("candidate_deltas", candidate_deltas.detach().clone().float())
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(2 * self.num_classes + 5, self.hidden_dim),
            nn.SiLU(),
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim + 8, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.SiLU(),
        )
        self.utility_head = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        spatial_features: torch.Tensor,
        class_logits: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        boxes: torch.Tensor,
        image_indices: torch.Tensor,
        image_sizes: torch.Tensor,
        edges: ProposalSetEdges | None = None,
    ) -> torch.Tensor:
        """Return identity-relative Delta-U scores of shape ``(N, K)``."""
        count = _validate_probe_inputs(
            spatial_features,
            class_logits,
            labels,
            scores,
            boxes,
            image_indices,
            self.in_channels,
            self.num_classes,
        )
        if count == 0:
            return spatial_features.new_zeros((0, self.num_candidates))

        sizes = _image_sizes_per_row(image_sizes, image_indices, count, boxes)
        spatial_code = self.spatial_encoder(spatial_features)
        normalized_boxes = _normalize_boxes(boxes, sizes).to(dtype=spatial_code.dtype)
        probabilities = torch.softmax(class_logits, dim=1).to(dtype=spatial_code.dtype)
        label_one_hot = F.one_hot(
            labels.long().clamp(0, self.num_classes - 1), num_classes=self.num_classes
        ).to(dtype=spatial_code.dtype)
        context = torch.cat(
            (probabilities, label_one_hot, scores[:, None].to(spatial_code.dtype), normalized_boxes),
            dim=1,
        )
        node_code = self.node_encoder(torch.cat((spatial_code, self.context_encoder(context)), dim=1))
        aggregate = self._aggregate_edges(node_code, edges)
        state_code = self.update(torch.cat((node_code, aggregate), dim=1))
        action_code = self.action_encoder(
            self.candidate_deltas.to(device=state_code.device, dtype=state_code.dtype)
        )
        state_grid = state_code[:, None, :].expand(-1, self.num_candidates, -1)
        action_grid = action_code[None, :, :].expand(count, -1, -1)
        raw = self.utility_head(torch.cat((state_grid, action_grid), dim=2)).squeeze(2)
        return raw - raw[:, 0:1]

    def _aggregate_edges(
        self,
        node_code: torch.Tensor,
        edges: ProposalSetEdges | None,
    ) -> torch.Tensor:
        count = node_code.shape[0]
        aggregate = node_code.new_zeros((count, self.hidden_dim))
        if edges is None or edges.edge_index.numel() == 0:
            return aggregate
        if edges.node_count != count:
            raise ValueError("edges.node_count must match the probe batch size")
        if edges.edge_index.ndim != 2 or edges.edge_index.shape[0] != 2:
            raise ValueError("edges.edge_index must have shape (2, E)")
        if edges.edge_features.shape != (edges.edge_index.shape[1], 8):
            raise ValueError("edges.edge_features must have shape (E, 8)")

        source = edges.edge_index[0].to(device=node_code.device, dtype=torch.long)
        target = edges.edge_index[1].to(device=node_code.device, dtype=torch.long)
        if source.numel() and (source.min() < 0 or target.min() < 0 or source.max() >= count or target.max() >= count):
            raise ValueError("edge indices must be valid node indices")
        edge_features = edges.edge_features.to(device=node_code.device, dtype=node_code.dtype)
        messages = self.edge_encoder(torch.cat((node_code[source], edge_features), dim=1))
        aggregate.index_add_(0, target, messages)
        degree = node_code.new_zeros((count, 1))
        degree.index_add_(0, target, node_code.new_ones((target.numel(), 1)))
        return aggregate / degree.clamp_min(1.0)


@dataclass(frozen=True)
class JointDeltaULossConfig:
    """Weights and tolerances for direct Delta-U supervision."""

    smooth_l1_weight: float = 1.0
    sign_bce_weight: float = 1.0
    pairwise_ranking_weight: float = 1.0
    pairwise_epsilon: float = 1e-3
    smooth_l1_beta: float = 1.0

    def __post_init__(self) -> None:
        for name in ("smooth_l1_weight", "sign_bce_weight", "pairwise_ranking_weight"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.pairwise_epsilon < 0.0:
            raise ValueError("pairwise_epsilon must be non-negative")
        if self.smooth_l1_beta <= 0.0:
            raise ValueError("smooth_l1_beta must be positive")


def joint_delta_u_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    group_ids: torch.Tensor | None = None,
    config: JointDeltaULossConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Return masked regression, sign, and within-proposal ranking supervision."""
    valid = _validate_delta_u_tensors(prediction, target, mask)
    config = config or JointDeltaULossConfig()
    zero = prediction.sum() * 0.0
    if not valid.any():
        return {
            "loss": zero,
            "smooth_l1": zero,
            "sign_bce": zero,
            "pairwise_ranking": zero,
            "valid_count": zero.detach(),
            "pair_count": zero.detach(),
        }

    smooth_l1 = F.smooth_l1_loss(
        prediction[valid], target[valid], beta=config.smooth_l1_beta, reduction="mean"
    )
    sign_target = target[valid].gt(float(config.pairwise_epsilon)).to(dtype=prediction.dtype)
    sign_bce = F.binary_cross_entropy_with_logits(prediction[valid], sign_target)
    if group_ids is None:
        target_difference = target[:, :, None] - target[:, None, :]
        prediction_difference = prediction[:, :, None] - prediction[:, None, :]
        pair_mask = (
            valid[:, :, None]
            & valid[:, None, :]
            & target_difference.gt(float(config.pairwise_epsilon))
        )
    else:
        if group_ids.shape != (prediction.shape[0],):
            raise ValueError("group_ids must have shape (N,)")
        row_ids = torch.arange(prediction.shape[0], device=prediction.device)[:, None]
        row_ids = row_ids.expand_as(valid)[valid]
        flat_groups = group_ids.to(device=prediction.device)[row_ids]
        flat_target = target[valid]
        flat_prediction = prediction[valid]
        target_difference = flat_target[:, None] - flat_target[None, :]
        prediction_difference = flat_prediction[:, None] - flat_prediction[None, :]
        pair_mask = flat_groups[:, None].eq(flat_groups[None, :]) & target_difference.gt(
            float(config.pairwise_epsilon)
        )
    if pair_mask.any():
        pairwise_ranking = F.softplus(-prediction_difference[pair_mask]).mean()
    else:
        pairwise_ranking = zero
    loss = (
        float(config.smooth_l1_weight) * smooth_l1
        + float(config.sign_bce_weight) * sign_bce
        + float(config.pairwise_ranking_weight) * pairwise_ranking
    )
    return {
        "loss": loss,
        "smooth_l1": smooth_l1,
        "sign_bce": sign_bce,
        "pairwise_ranking": pairwise_ranking,
        "valid_count": valid.sum().detach().to(dtype=prediction.dtype),
        "pair_count": pair_mask.sum().detach().to(dtype=prediction.dtype),
    }


def select_joint_delta_u_actions(
    candidate_deltas: torch.Tensor,
    delta_u: torch.Tensor,
    image_indices: torch.Tensor,
    max_actions_per_image: int = 4,
) -> JointDeltaUSelection:
    """Choose the best positive non-identity Delta-U action under image budgets."""
    _validate_candidate_deltas(candidate_deltas)
    if delta_u.ndim != 2 or delta_u.shape[1] != candidate_deltas.shape[0]:
        raise ValueError("delta_u must have shape (N, K) matching candidate_deltas")
    count = delta_u.shape[0]
    if image_indices.shape != (count,):
        raise ValueError("image_indices must have shape (N,)")
    if max_actions_per_image <= 0:
        raise ValueError("max_actions_per_image must be positive")
    if not torch.isfinite(delta_u).all():
        raise ValueError("delta_u must be finite")

    selected_mask = torch.zeros(count, dtype=torch.bool, device=delta_u.device)
    if candidate_deltas.shape[0] == 1:
        candidate_indices = torch.zeros(count, dtype=torch.long, device=delta_u.device)
        best_delta_u = delta_u.new_zeros((count,))
    else:
        best_delta_u, candidate_indices = delta_u[:, 1:].max(dim=1)
        candidate_indices = candidate_indices + 1
        eligible = best_delta_u.gt(0.0)
        local_images = image_indices.to(delta_u.device)
        for image_index in torch.unique(local_images[eligible], sorted=True):
            rows = torch.nonzero(eligible & local_images.eq(image_index), as_tuple=False).flatten()
            order = torch.argsort(best_delta_u[rows], descending=True)
            selected_mask[rows[order[:max_actions_per_image]]] = True
        candidate_indices = torch.where(
            selected_mask, candidate_indices, torch.zeros_like(candidate_indices)
        )
        best_delta_u = torch.where(selected_mask, best_delta_u, torch.zeros_like(best_delta_u))

    deltas = candidate_deltas.to(device=delta_u.device, dtype=delta_u.dtype)
    return JointDeltaUSelection(
        candidate_indices=candidate_indices,
        selected_mask=selected_mask,
        box_delta=deltas[candidate_indices],
        delta_u=best_delta_u,
    )


def joint_delta_u_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    epsilon: float = 1e-3,
    selected_mask: torch.Tensor | None = None,
    candidate_indices: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute masked Delta-U quality and action-selection diagnostics."""
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative")
    valid = _validate_delta_u_tensors(prediction, target, mask)
    if not valid.any():
        return {
            "masked_mae": 0.0,
            "sign_accuracy": 0.0,
            "pairwise_accuracy": 0.0,
            "oracle_regret": 0.0,
            "positive_precision": 0.0,
            "positive_recall": 0.0,
            "selected_true_delta": 0.0,
        }

    with torch.no_grad():
        predicted = prediction.detach()
        expected = target.detach()
        masked_mae = (predicted[valid] - expected[valid]).abs().mean()
        sign_accuracy = predicted[valid].gt(0.0).eq(expected[valid].gt(float(epsilon))).float().mean()
        target_difference = expected[:, :, None] - expected[:, None, :]
        prediction_difference = predicted[:, :, None] - predicted[:, None, :]
        pair_mask = valid[:, :, None] & valid[:, None, :] & target_difference.gt(float(epsilon))
        pairwise_accuracy = (
            prediction_difference[pair_mask].gt(0.0).float().mean()
            if pair_mask.any()
            else predicted.new_zeros(())
        )
        predicted_positive = predicted.gt(0.0) & valid
        target_positive = expected.gt(float(epsilon)) & valid
        true_positive = (predicted_positive & target_positive).sum()
        precision = _safe_ratio(true_positive, predicted_positive.sum(), predicted)
        recall = _safe_ratio(true_positive, target_positive.sum(), predicted)

        best_prediction, best_indices = predicted.masked_fill(~valid, float("-inf")).max(dim=1)
        row_valid = valid.any(dim=1)
        row_ids = torch.arange(predicted.shape[0], device=predicted.device)
        selected_target = expected[row_ids, best_indices]
        oracle_target = expected.masked_fill(~valid, float("-inf")).max(dim=1).values
        oracle_regret = (oracle_target[row_valid] - selected_target[row_valid]).mean()

        if candidate_indices is not None:
            if candidate_indices.shape != (predicted.shape[0],):
                raise ValueError("candidate_indices must have shape (N,)")
            action_indices = candidate_indices.to(device=predicted.device, dtype=torch.long)
            if action_indices.numel() and (action_indices.min() < 0 or action_indices.max() >= predicted.shape[1]):
                raise ValueError("candidate_indices must reference valid candidates")
            action_selected = (
                selected_mask.to(device=predicted.device, dtype=torch.bool)
                if selected_mask is not None
                else action_indices.ne(0)
            )
        else:
            action_indices = best_indices
            action_selected = best_prediction.gt(0.0) & row_valid
            if selected_mask is not None:
                if selected_mask.shape != (predicted.shape[0],):
                    raise ValueError("selected_mask must have shape (N,)")
                action_selected &= selected_mask.to(device=predicted.device, dtype=torch.bool)
        action_valid = action_selected & valid[row_ids, action_indices]
        selected_true_delta = (
            expected[row_ids[action_valid], action_indices[action_valid]].mean()
            if action_valid.any()
            else predicted.new_zeros(())
        )
        return {
            "masked_mae": float(masked_mae.item()),
            "sign_accuracy": float(sign_accuracy.item()),
            "pairwise_accuracy": float(pairwise_accuracy.item()),
            "oracle_regret": float(oracle_regret.item()),
            "positive_precision": float(precision.item()),
            "positive_recall": float(recall.item()),
            "selected_true_delta": float(selected_true_delta.item()),
        }


def _validate_proposal_tensors(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    image_indices: torch.Tensor,
) -> int:
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    count = boxes.shape[0]
    if labels.shape != (count,) or scores.shape != (count,) or image_indices.shape != (count,):
        raise ValueError("labels, scores, and image_indices must have shape (N,)")
    if not boxes.is_floating_point():
        raise ValueError("boxes must have a floating point dtype")
    return count


def _validate_probe_inputs(
    spatial_features: torch.Tensor,
    class_logits: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    boxes: torch.Tensor,
    image_indices: torch.Tensor,
    in_channels: int,
    num_classes: int,
) -> int:
    if spatial_features.ndim != 4 or spatial_features.shape[1] != in_channels:
        raise ValueError(f"spatial_features must have shape (N, {in_channels}, H, W)")
    count = _validate_proposal_tensors(boxes, labels, scores, image_indices)
    if spatial_features.shape[0] != count or class_logits.shape != (count, num_classes):
        raise ValueError("spatial_features and class_logits must align with proposal rows")
    return count


def _validate_delta_u_tensors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if prediction.ndim != 2 or target.shape != prediction.shape:
        raise ValueError("prediction and target must share shape (N, K)")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError("prediction and target must have floating point dtypes")
    if mask is None:
        return torch.ones_like(prediction, dtype=torch.bool)
    if mask.shape != prediction.shape:
        raise ValueError("mask must have shape (N, K)")
    return mask.to(device=prediction.device, dtype=torch.bool)


def _validate_candidate_deltas(candidate_deltas: torch.Tensor) -> None:
    if candidate_deltas.ndim != 2 or candidate_deltas.shape[0] == 0 or candidate_deltas.shape[1] != 4:
        raise ValueError("candidate_deltas must have shape (K, 4) with K >= 1")
    if not torch.isfinite(candidate_deltas).all():
        raise ValueError("candidate_deltas must be finite")
    if candidate_deltas[0].count_nonzero().item() != 0:
        raise ValueError("candidate_deltas[0] must be the all-zero identity")


def _image_sizes_per_row(
    image_sizes: torch.Tensor,
    image_indices: torch.Tensor,
    count: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    sizes = torch.as_tensor(image_sizes, device=reference.device, dtype=reference.dtype)
    if sizes.ndim == 1 and sizes.shape == (2,):
        sizes = sizes.expand(count, 2)
    elif sizes.ndim == 2 and sizes.shape[1] == 2:
        if sizes.shape[0] == count:
            pass
        elif count == 0:
            sizes = sizes.new_zeros((0, 2))
        elif image_indices.min() >= 0 and image_indices.max() < sizes.shape[0]:
            sizes = sizes[image_indices.long().to(sizes.device)]
        else:
            raise ValueError("image_sizes must provide one (height, width) pair per image or proposal")
    else:
        raise ValueError("image_sizes must have shape (2,), (B, 2), or (N, 2)")
    if not torch.isfinite(sizes).all() or (sizes <= 0.0).any():
        raise ValueError("image_sizes must be finite and positive")
    return sizes


def _box_geometry(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(torch.finfo(boxes.dtype).eps)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(torch.finfo(boxes.dtype).eps)
    centers = torch.stack(((boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5), dim=1)
    return widths, heights, centers


def _normalize_boxes(boxes: torch.Tensor, image_sizes: torch.Tensor) -> torch.Tensor:
    scale = torch.stack(
        (image_sizes[:, 1], image_sizes[:, 0], image_sizes[:, 1], image_sizes[:, 0]), dim=1
    )
    return boxes / scale.clamp_min(torch.finfo(boxes.dtype).eps)


def _pairwise_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0.0).prod(dim=2)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).clamp_min(0.0).prod(dim=1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).clamp_min(0.0).prod(dim=1)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection).clamp_min(1e-8)


def _elementwise_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(boxes_a[:, :2], boxes_b[:, :2])
    bottom_right = torch.minimum(boxes_a[:, 2:], boxes_b[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0.0).prod(dim=1)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).clamp_min(0.0).prod(dim=1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).clamp_min(0.0).prod(dim=1)
    return intersection / (area_a + area_b - intersection).clamp_min(1e-8)


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if denominator.item() == 0:
        return reference.new_zeros(())
    return numerator.to(dtype=reference.dtype) / denominator.to(dtype=reference.dtype)
