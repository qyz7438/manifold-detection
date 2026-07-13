"""Image-level listwise no-op versus one bounded detector action."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_detection_posttrain.methods.energy_transport.policy.set_policy import (
    NMSAwareSetPolicyHead,
    class_aware_conflict_statistics,
)
from spectral_detection_posttrain.methods.energy_transport.operators import apply_box_delta


@dataclass(frozen=True)
class GlobalTop1Output:
    action_logits: torch.Tensor
    noop_logit: torch.Tensor
    conflict_stats: torch.Tensor


@dataclass(frozen=True)
class FlattenedGlobalLogits:
    logits: torch.Tensor
    proposal_indices: torch.Tensor
    candidate_indices: torch.Tensor


@dataclass(frozen=True)
class GlobalTop1Target:
    is_noop: bool
    proposal_index: int
    candidate_index: int
    delta_u: float


@dataclass(frozen=True)
class GlobalTop1Selection:
    is_noop: bool
    proposal_index: int
    candidate_index: int
    selection_logit: float
    box_delta: torch.Tensor


class GlobalTop1PolicyHead(nn.Module):
    """Score one global no-op against every observable proposal-action pair."""

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
        self.proposal_policy = NMSAwareSetPolicyHead(
            in_channels=in_channels,
            num_classes=num_classes,
            candidate_deltas=candidate_deltas,
            hidden_dim=hidden_dim,
            spatial_size=spatial_size,
            energy_weight=energy_weight,
        )
        for parameter in self.proposal_policy.move_head.parameters():
            parameter.requires_grad_(False)
        self.noop_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        spatial_features: torch.Tensor,
        class_logits: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        boxes: torch.Tensor,
        image_size: tuple[int, int] | torch.Tensor,
        observable_mask: torch.Tensor,
    ) -> GlobalTop1Output:
        output = self.proposal_policy(
            spatial_features,
            class_logits,
            labels,
            scores,
            boxes,
            image_size,
        )
        if observable_mask.shape != (output.action_logits.shape[0],):
            raise ValueError("observable_mask must have shape (N,)")
        observable_noop = output.action_logits[
            observable_mask.to(device=output.action_logits.device).bool(), 0
        ]
        if observable_noop.numel():
            noop_logit = torch.logsumexp(observable_noop, dim=0) - math.log(observable_noop.numel())
            noop_logit = noop_logit + self.noop_bias
        else:
            noop_logit = self.noop_bias
        return GlobalTop1Output(
            action_logits=output.action_logits,
            noop_logit=noop_logit,
            conflict_stats=output.conflict_stats,
        )


class AdaptiveConsensusGlobalTop1PolicyHead(nn.Module):
    """Global top-1/no-op policy for one detector-derived delta per proposal."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 96,
        spatial_size: int = 7,
        energy_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.energy_weight = float(energy_weight)
        self.base = GlobalTop1PolicyHead(
            in_channels=in_channels,
            num_classes=num_classes,
            candidate_deltas=torch.zeros((2, 4)),
            hidden_dim=hidden_dim,
            spatial_size=spatial_size,
            energy_weight=0.0,
        )
        self.delta_head = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(
        self,
        spatial_features: torch.Tensor,
        class_logits: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        boxes: torch.Tensor,
        image_size: tuple[int, int] | torch.Tensor,
        observable_mask: torch.Tensor,
        *,
        adaptive_deltas: torch.Tensor,
    ) -> GlobalTop1Output:
        if adaptive_deltas.shape != (boxes.shape[0], 4):
            raise ValueError("adaptive_deltas must have shape (N, 4)")
        output = self.base(
            spatial_features,
            class_logits,
            labels,
            scores,
            boxes,
            image_size,
            observable_mask,
        )
        deltas = adaptive_deltas.to(device=boxes.device, dtype=output.action_logits.dtype)
        action_logits = output.action_logits.clone()
        adaptive_score = self.delta_head(deltas).squeeze(1)
        adaptive_score = adaptive_score - self.energy_weight * deltas.square().sum(dim=1) / 0.04
        available = deltas.abs().amax(dim=1).gt(0.0) & observable_mask.to(device=boxes.device).bool()
        action_logits[:, 1] = torch.where(
            available,
            action_logits[:, 1] + adaptive_score,
            action_logits.new_full((boxes.shape[0],), -1e9),
        )
        return GlobalTop1Output(action_logits, output.noop_logit, output.conflict_stats)


def select_adaptive_consensus_action(
    output: GlobalTop1Output,
    adaptive_deltas: torch.Tensor,
    observable_mask: torch.Tensor,
    *,
    allow_noop: bool = True,
) -> GlobalTop1Selection:
    count = output.action_logits.shape[0]
    if output.action_logits.shape != (count, 2):
        raise ValueError("adaptive output action_logits must have shape (N, 2)")
    if adaptive_deltas.shape != (count, 4) or observable_mask.shape != (count,):
        raise ValueError("adaptive_deltas and observable_mask must align with proposals")
    available = observable_mask.to(device=output.action_logits.device).bool()
    available = available & adaptive_deltas.to(device=output.action_logits.device).abs().amax(dim=1).gt(0.0)
    rows = torch.nonzero(available, as_tuple=False).flatten()
    if rows.numel() == 0:
        return GlobalTop1Selection(True, -1, 0, float(output.noop_logit.item()), adaptive_deltas.new_zeros(adaptive_deltas.shape))
    scores = output.action_logits[rows, 1]
    best_position = int(scores.argmax().item())
    best_row = int(rows[best_position].item())
    best_logit = float(scores[best_position].item())
    if allow_noop and float(output.noop_logit.item()) >= best_logit:
        return GlobalTop1Selection(True, -1, 0, float(output.noop_logit.item()), adaptive_deltas.new_zeros(adaptive_deltas.shape))
    box_delta = adaptive_deltas.new_zeros(adaptive_deltas.shape)
    box_delta[best_row] = adaptive_deltas[best_row]
    return GlobalTop1Selection(False, best_row, 1, best_logit, box_delta)


class SetContextGlobalTop1PolicyHead(nn.Module):
    """Permutation-equivariant proposal scorer with detector-visible set context."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        candidate_deltas: torch.Tensor,
        hidden_dim: int = 96,
        spatial_size: int = 2,
        energy_weight: float = 0.05,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or num_classes <= 1 or hidden_dim <= 0 or spatial_size <= 0:
            raise ValueError("model dimensions must be positive and num_classes must include foreground")
        if candidate_deltas.ndim != 2 or candidate_deltas.shape[1] != 4 or candidate_deltas.shape[0] < 2:
            raise ValueError("candidate_deltas must have shape (K, 4) with K >= 2")
        if candidate_deltas[0].count_nonzero().item() != 0:
            raise ValueError("candidate index 0 must be no-op")
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
        self.local_trunk = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.set_trunk = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.action_head = nn.Linear(self.hidden_dim, candidate_deltas.shape[0])
        self.noop_head = nn.Linear(2 * self.hidden_dim, 1)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        nn.init.zeros_(self.noop_head.weight)
        nn.init.zeros_(self.noop_head.bias)

    def forward(
        self,
        spatial_features: torch.Tensor,
        class_logits: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        boxes: torch.Tensor,
        image_size: tuple[int, int] | torch.Tensor,
        observable_mask: torch.Tensor,
    ) -> GlobalTop1Output:
        if spatial_features.ndim != 4 or spatial_features.shape[1] != self.in_channels:
            raise ValueError("spatial_features shape mismatch")
        count = spatial_features.shape[0]
        if class_logits.shape != (count, self.num_classes):
            raise ValueError("class_logits shape mismatch")
        if labels.shape != (count,) or scores.shape != (count,) or boxes.shape != (count, 4):
            raise ValueError("detector vector shape mismatch")
        if observable_mask.shape != (count,):
            raise ValueError("observable_mask must have shape (N,)")
        conflict_stats = class_aware_conflict_statistics(boxes, labels, scores, image_size)
        normalized_boxes = _normalize_set_boxes(boxes, image_size)
        label_code = F.one_hot(labels.long(), num_classes=self.num_classes).to(dtype=class_logits.dtype)
        context = torch.cat(
            (class_logits, label_code, scores[:, None].to(class_logits.dtype), normalized_boxes, conflict_stats),
            dim=1,
        )
        spatial = self.spatial_encoder(self.spatial_pool(spatial_features).reshape(count, -1))
        local = self.local_trunk(torch.cat((spatial, self.context_encoder(context)), dim=1))
        observable = observable_mask.to(device=local.device).bool()
        if observable.any():
            visible = local[observable]
            set_mean = visible.mean(dim=0)
            set_max = visible.max(dim=0).values
        else:
            set_mean = local.new_zeros((self.hidden_dim,))
            set_max = local.new_zeros((self.hidden_dim,))
        set_mean_rows = set_mean[None, :].expand(count, -1)
        set_max_rows = set_max[None, :].expand(count, -1)
        action_hidden = self.set_trunk(torch.cat((local, set_mean_rows, set_max_rows), dim=1))
        action_logits = self.action_head(action_hidden)
        action_energy = self.candidate_deltas.to(dtype=action_logits.dtype).square().sum(dim=1) / 0.04
        action_logits = action_logits - self.energy_weight * action_energy[None, :]
        noop_logit = self.noop_head(torch.cat((set_mean, set_max), dim=0)).squeeze(-1)
        return GlobalTop1Output(action_logits, noop_logit, conflict_stats)


def _normalize_set_boxes(
    boxes: torch.Tensor,
    image_size: tuple[int, int] | torch.Tensor,
) -> torch.Tensor:
    size = torch.as_tensor(image_size, dtype=boxes.dtype, device=boxes.device).flatten()
    if size.numel() != 2 or (size <= 0).any():
        raise ValueError("image_size must contain positive height and width")
    height, width = size[0], size[1]
    scale = torch.stack((width, height, width, height))
    return boxes / scale.clamp_min(1.0)


def action_conditioned_nms_topology(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    image_size: tuple[int, int] | torch.Tensor,
    candidate_deltas: torch.Tensor,
    nms_threshold: float = 0.5,
) -> torch.Tensor:
    """Return detector-visible NMS topology for every proposal-action pair."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    count = boxes.shape[0]
    if labels.shape != (count,) or scores.shape != (count,):
        raise ValueError("labels and scores must have shape (N,)")
    if candidate_deltas.ndim != 2 or candidate_deltas.shape[1] != 4:
        raise ValueError("candidate_deltas must have shape (K, 4)")
    if not 0.0 <= float(nms_threshold) <= 1.0:
        raise ValueError("nms_threshold must be in [0, 1]")
    size = torch.as_tensor(image_size).flatten()
    if size.numel() != 2:
        raise ValueError("image_size must contain height and width")
    height, width = int(size[0].item()), int(size[1].item())
    candidates = candidate_deltas.shape[0]
    moved = []
    for candidate_index in range(candidates):
        deltas = candidate_deltas[candidate_index].to(device=boxes.device, dtype=boxes.dtype)[None, :].expand(count, -1)
        moved.append(apply_box_delta(boxes, deltas, image_size=(height, width)))
    moved_boxes = torch.stack(moved, dim=1)
    ious = _cross_box_iou(moved_boxes.reshape(-1, 4), boxes).reshape(count, candidates, count)
    higher_same = labels[:, None].eq(labels[None, :]) & (scores[None, :] > scores[:, None])
    post_max = (ious * higher_same[:, None, :].to(dtype=ious.dtype)).max(dim=2).values
    base_max = post_max[:, :1]
    survival_margin = float(nms_threshold) - post_max
    topology_change = post_max - base_max
    peer_count = torch.log1p(higher_same.sum(dim=1).to(dtype=ious.dtype))[:, None].expand(-1, candidates)
    return torch.stack((post_max, survival_margin, topology_change, peer_count), dim=2)


class ActionTopologyGlobalTop1PolicyHead(nn.Module):
    """Set-context policy augmented by action-conditioned NMS topology."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        candidate_deltas: torch.Tensor,
        hidden_dim: int = 96,
        spatial_size: int = 2,
        energy_weight: float = 0.05,
        nms_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.base = SetContextGlobalTop1PolicyHead(
            in_channels=in_channels,
            num_classes=num_classes,
            candidate_deltas=candidate_deltas,
            hidden_dim=hidden_dim,
            spatial_size=spatial_size,
            energy_weight=energy_weight,
        )
        self.nms_threshold = float(nms_threshold)
        self.topology_head = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.topology_head[-1].weight)
        nn.init.zeros_(self.topology_head[-1].bias)

    def forward(
        self,
        spatial_features: torch.Tensor,
        class_logits: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        boxes: torch.Tensor,
        image_size: tuple[int, int] | torch.Tensor,
        observable_mask: torch.Tensor,
        *,
        topology_shuffle_seed: int | None = None,
    ) -> GlobalTop1Output:
        output = self.base(
            spatial_features,
            class_logits,
            labels,
            scores,
            boxes,
            image_size,
            observable_mask,
        )
        topology = action_conditioned_nms_topology(
            boxes,
            labels,
            scores,
            image_size,
            self.base.candidate_deltas,
            self.nms_threshold,
        )
        if topology_shuffle_seed is not None:
            topology = _shuffle_observable_action_topology(
                topology,
                observable_mask,
                int(topology_shuffle_seed),
            )
        topology_logits = self.topology_head(topology).squeeze(-1)
        return GlobalTop1Output(
            action_logits=output.action_logits + topology_logits,
            noop_logit=output.noop_logit,
            conflict_stats=output.conflict_stats,
        )


class NativeActionTopologyGlobalTop1PolicyHead(nn.Module):
    """Set-context policy consuming class-expanded native-NMS topology."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        candidate_deltas: torch.Tensor,
        hidden_dim: int = 96,
        spatial_size: int = 2,
        energy_weight: float = 0.05,
        topology_dim: int = 8,
    ) -> None:
        super().__init__()
        if topology_dim <= 0:
            raise ValueError("topology_dim must be positive")
        self.base = SetContextGlobalTop1PolicyHead(
            in_channels=in_channels,
            num_classes=num_classes,
            candidate_deltas=candidate_deltas,
            hidden_dim=hidden_dim,
            spatial_size=spatial_size,
            energy_weight=energy_weight,
        )
        self.topology_dim = int(topology_dim)
        self.topology_head = nn.Sequential(
            nn.Linear(self.topology_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.topology_head[-1].weight)
        nn.init.zeros_(self.topology_head[-1].bias)

    def forward(
        self,
        spatial_features: torch.Tensor,
        class_logits: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        boxes: torch.Tensor,
        image_size: tuple[int, int] | torch.Tensor,
        observable_mask: torch.Tensor,
        *,
        native_topology: torch.Tensor,
        topology_shuffle_seed: int | None = None,
    ) -> GlobalTop1Output:
        output = self.base(
            spatial_features,
            class_logits,
            labels,
            scores,
            boxes,
            image_size,
            observable_mask,
        )
        expected = (boxes.shape[0], self.base.candidate_deltas.shape[0], self.topology_dim)
        if native_topology.shape != expected:
            raise ValueError(f"native_topology must have shape {expected}")
        topology = native_topology.to(device=boxes.device, dtype=spatial_features.dtype)
        if topology_shuffle_seed is not None:
            topology = _shuffle_observable_action_topology(topology, observable_mask, topology_shuffle_seed)
        topology_logits = self.topology_head(topology).squeeze(-1)
        return GlobalTop1Output(
            action_logits=output.action_logits + topology_logits,
            noop_logit=output.noop_logit,
            conflict_stats=output.conflict_stats,
        )


def _shuffle_observable_action_topology(
    topology: torch.Tensor,
    observable_mask: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    shuffled = topology.clone()
    rows = torch.nonzero(observable_mask.to(device=topology.device).bool(), as_tuple=False).flatten()
    if rows.numel() == 0 or topology.shape[1] <= 1:
        return shuffled
    values = topology[rows, 1:, :].reshape(-1, topology.shape[2])
    if values.shape[0] < 2:
        return shuffled
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(values.shape[0], generator=generator).to(device=topology.device)
    identity = torch.arange(values.shape[0], device=topology.device)
    if torch.equal(permutation, identity):
        permutation = permutation.roll(1)
    shuffled[rows, 1:, :] = values[permutation].reshape(rows.numel(), topology.shape[1] - 1, topology.shape[2])
    return shuffled


def _cross_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.ndim != 2 or boxes2.ndim != 2 or boxes1.shape[1] != 4 or boxes2.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4) and (M, 4)")
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0.0).prod(dim=2)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0.0).prod(dim=1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0.0).prod(dim=1)
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp_min(1e-8)


def flatten_observable_action_logits(
    output: GlobalTop1Output,
    observable_mask: torch.Tensor,
) -> FlattenedGlobalLogits:
    if output.action_logits.ndim != 2 or output.action_logits.shape[1] < 2:
        raise ValueError("action_logits must have shape (N, K) with K >= 2")
    count, candidates = output.action_logits.shape
    if observable_mask.shape != (count,):
        raise ValueError("observable_mask must have shape (N,)")
    rows = torch.nonzero(
        observable_mask.to(device=output.action_logits.device).bool(), as_tuple=False
    ).flatten()
    proposal_indices = torch.cat(
        (
            torch.full((1,), -1, dtype=torch.long, device=rows.device),
            rows.repeat_interleave(candidates - 1),
        )
    )
    candidate_indices = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=rows.device),
            torch.arange(1, candidates, device=rows.device).repeat(rows.numel()),
        )
    )
    logits = torch.cat((output.noop_logit.reshape(1), output.action_logits[rows, 1:].reshape(-1)))
    return FlattenedGlobalLogits(logits, proposal_indices, candidate_indices)


def build_global_top1_target(
    delta_u: torch.Tensor,
    observable_mask: torch.Tensor,
    min_delta_u: float = 0.0,
) -> GlobalTop1Target:
    if delta_u.ndim != 2 or delta_u.shape[1] < 2:
        raise ValueError("delta_u must have shape (N, K) with K >= 2")
    if observable_mask.shape != (delta_u.shape[0],):
        raise ValueError("observable_mask must have shape (N,)")
    if not math.isfinite(float(min_delta_u)):
        raise ValueError("min_delta_u must be finite")
    rows = torch.nonzero(observable_mask.to(device=delta_u.device).bool(), as_tuple=False).flatten()
    if rows.numel() == 0:
        return GlobalTop1Target(True, -1, 0, 0.0)
    utilities = delta_u[rows, 1:]
    if not torch.isfinite(utilities).all():
        raise ValueError("observable non-noop delta_u values must be finite")
    flat_index = int(utilities.reshape(-1).argmax().item())
    candidate_count = utilities.shape[1]
    row_offset, candidate_offset = divmod(flat_index, candidate_count)
    best_delta = float(utilities[row_offset, candidate_offset].item())
    if best_delta <= float(min_delta_u):
        return GlobalTop1Target(True, -1, 0, 0.0)
    return GlobalTop1Target(
        False,
        int(rows[row_offset].item()),
        int(candidate_offset + 1),
        best_delta,
    )


def global_top1_loss(
    output: GlobalTop1Output,
    observable_mask: torch.Tensor,
    target: GlobalTop1Target,
) -> dict[str, torch.Tensor]:
    flattened = flatten_observable_action_logits(output, observable_mask)
    if target.is_noop:
        target_index = 0
    else:
        matches = flattened.proposal_indices.eq(target.proposal_index) & flattened.candidate_indices.eq(
            target.candidate_index
        )
        if int(matches.sum().item()) != 1:
            raise ValueError("target action is not in the detector-observable candidate set")
        target_index = int(torch.nonzero(matches, as_tuple=False).flatten()[0].item())
    target_tensor = torch.tensor([target_index], dtype=torch.long, device=flattened.logits.device)
    loss = F.cross_entropy(flattened.logits[None, :], target_tensor)
    prediction = int(flattened.logits.argmax().item())
    return {
        "loss_total": loss,
        "accuracy": loss.detach().new_tensor(float(prediction == target_index)),
        "predicted_noop": loss.detach().new_tensor(float(prediction == 0)),
    }


def global_top1_balanced_margin_loss(
    output: GlobalTop1Output,
    observable_mask: torch.Tensor,
    target: GlobalTop1Target,
    *,
    action_margin: float = 0.2,
    rank_margin: float = 0.2,
    actionability_weight: float = 1.0,
    rank_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Balance no-op/action decisions and conditionally rank action candidates."""
    if action_margin < 0.0 or rank_margin < 0.0:
        raise ValueError("margins must be non-negative")
    if actionability_weight < 0.0 or rank_weight < 0.0:
        raise ValueError("loss weights must be non-negative")
    flattened = flatten_observable_action_logits(output, observable_mask)
    noop_logit = flattened.logits[0]
    action_logits = flattened.logits[1:]
    if target.is_noop:
        target_index = 0
        if action_logits.numel():
            loss_actionability = F.softplus(action_logits.max() - noop_logit + float(action_margin))
        else:
            loss_actionability = noop_logit * 0.0
        loss_rank = noop_logit * 0.0
    else:
        matches = flattened.proposal_indices.eq(target.proposal_index) & flattened.candidate_indices.eq(
            target.candidate_index
        )
        if int(matches.sum().item()) != 1:
            raise ValueError("target action is not in the detector-observable candidate set")
        target_index = int(torch.nonzero(matches, as_tuple=False).flatten()[0].item())
        target_logit = flattened.logits[target_index]
        loss_actionability = F.softplus(noop_logit - target_logit + float(action_margin))
        action_target_index = target_index - 1
        if action_logits.numel() > 1:
            negative_mask = torch.ones_like(action_logits, dtype=torch.bool)
            negative_mask[action_target_index] = False
            hard_negative = action_logits[negative_mask].max()
            loss_rank = F.softplus(hard_negative - target_logit + float(rank_margin))
        else:
            loss_rank = target_logit * 0.0
    loss_total = float(actionability_weight) * loss_actionability + float(rank_weight) * loss_rank
    prediction = int(flattened.logits.argmax().item())
    return {
        "loss_total": loss_total,
        "loss_actionability": loss_actionability,
        "loss_rank": loss_rank,
        "accuracy": loss_total.detach().new_tensor(float(prediction == target_index)),
        "predicted_noop": loss_total.detach().new_tensor(float(prediction == 0)),
    }


def select_global_top1_action(
    output: GlobalTop1Output,
    candidate_deltas: torch.Tensor,
    observable_mask: torch.Tensor,
    *,
    allow_noop: bool = True,
) -> GlobalTop1Selection:
    if candidate_deltas.ndim != 2 or candidate_deltas.shape[1] != 4:
        raise ValueError("candidate_deltas must have shape (K, 4)")
    if candidate_deltas.shape[0] != output.action_logits.shape[1]:
        raise ValueError("candidate_deltas must match action_logits")
    if candidate_deltas[0].count_nonzero().item() != 0:
        raise ValueError("candidate index 0 must be no-op")
    flattened = flatten_observable_action_logits(output, observable_mask)
    if not allow_noop and flattened.logits.numel() > 1:
        selected_index = int(flattened.logits[1:].argmax().item()) + 1
    else:
        selected_index = int(flattened.logits.argmax().item())
    proposal_index = int(flattened.proposal_indices[selected_index].item())
    candidate_index = int(flattened.candidate_indices[selected_index].item())
    box_delta = output.action_logits.new_zeros((output.action_logits.shape[0], 4))
    is_noop = selected_index == 0
    if not is_noop:
        box_delta[proposal_index] = candidate_deltas[candidate_index].to(
            device=box_delta.device, dtype=box_delta.dtype
        )
    return GlobalTop1Selection(
        is_noop=is_noop,
        proposal_index=proposal_index,
        candidate_index=candidate_index,
        selection_logit=float(flattened.logits[selected_index].detach().item()),
        box_delta=box_delta,
    )
