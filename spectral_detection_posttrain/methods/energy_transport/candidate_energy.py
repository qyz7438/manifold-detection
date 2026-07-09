"""Discrete local box candidates and identity-relative energy selection."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch
from torch import nn
import torch.nn.functional as F

from spectral_detection_posttrain.methods.energy_transport.actions import ROITransportActions
from spectral_detection_posttrain.methods.energy_transport.contracts import ROIActionState
from spectral_detection_posttrain.methods.energy_transport.operators import apply_box_delta


@dataclass(frozen=True)
class CandidateEnergyLossConfig:
    temperature: float = 0.10
    energy_weight: float = 1e-3

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.energy_weight < 0.0:
            raise ValueError("energy_weight must be non-negative")


@dataclass(frozen=True)
class CandidateQualityTargets:
    candidate_quality: torch.Tensor
    target_indices: torch.Tensor
    base_iou: torch.Tensor
    oracle_gain: torch.Tensor
    matched: torch.Tensor
    class_correct: torch.Tensor


class SpatialCandidateEnergyHead(nn.Module):
    """Score box actions from spatial ROI evidence before the detector box head."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 256,
        spatial_size: int = 7,
    ) -> None:
        super().__init__()
        if min(in_channels, num_classes, hidden_dim, spatial_size) <= 0 or num_classes <= 1:
            raise ValueError("head dimensions must be positive and num_classes must exceed one")
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.spatial_size = int(spatial_size)

        self.feature_encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(16 * self.spatial_size * self.spatial_size, self.hidden_dim),
            nn.SiLU(),
        )
        context_dim = 2 * self.num_classes + 1 + 4
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.energy_head = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        output = self.energy_head[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def encode_features(self, features: torch.Tensor) -> torch.Tensor:
        expected = (self.in_channels, self.spatial_size, self.spatial_size)
        if features.ndim != 4 or tuple(features.shape[1:]) != expected:
            raise ValueError(f"features must have shape (B, {expected[0]}, {expected[1]}, {expected[2]})")
        return self.feature_encoder(features)

    def energy_from_code(
        self,
        feature_code: torch.Tensor,
        class_logits: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        box_delta: torch.Tensor,
    ) -> torch.Tensor:
        batch = feature_code.shape[0]
        if feature_code.shape != (batch, self.hidden_dim):
            raise ValueError(f"feature_code must have shape (B, {self.hidden_dim})")
        if class_logits.shape != (batch, self.num_classes):
            raise ValueError(f"class_logits must have shape (B, {self.num_classes})")
        if labels.shape != (batch,) or scores.shape != (batch,):
            raise ValueError("labels and scores must have shape (B,)")
        if box_delta.shape != (batch, 4):
            raise ValueError("box_delta must have shape (B, 4)")

        probabilities = torch.softmax(class_logits, dim=-1)
        one_hot = F.one_hot(
            labels.long().clamp(0, self.num_classes - 1),
            num_classes=self.num_classes,
        ).to(dtype=feature_code.dtype)
        context = torch.cat(
            (
                probabilities.to(dtype=feature_code.dtype),
                one_hot,
                scores[:, None].to(dtype=feature_code.dtype),
                box_delta.to(dtype=feature_code.dtype),
            ),
            dim=1,
        )
        context_code = self.context_encoder(context)
        return self.energy_head(torch.cat((feature_code, context_code), dim=1)).squeeze(1)

    def forward(
        self,
        features: torch.Tensor,
        class_logits: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        box_delta: torch.Tensor,
    ) -> torch.Tensor:
        feature_code = self.encode_features(features)
        return self.energy_from_code(feature_code, class_logits, labels, scores, box_delta)


def build_symmetric_box_candidates(
    step_sizes: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> torch.Tensor:
    """Build identity plus symmetric translation, size, and joint directions."""

    if not step_sizes or any(step <= 0.0 for step in step_sizes):
        raise ValueError("step_sizes must contain positive values")
    directions: list[tuple[float, float, float, float]] = []
    compass = [pair for pair in product((-1.0, 0.0, 1.0), repeat=2) if pair != (0.0, 0.0)]
    directions.extend((dx, dy, 0.0, 0.0) for dx, dy in compass)
    directions.extend((0.0, 0.0, dw, dh) for dw, dh in compass)
    hadamard = (
        (1.0, 1.0, 1.0, 1.0),
        (1.0, -1.0, 1.0, -1.0),
        (1.0, 1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0, 1.0),
    )
    directions.extend(hadamard)
    directions.extend(tuple(-value for value in row) for row in hadamard)

    rows = [(0.0, 0.0, 0.0, 0.0)]
    for step in sorted(set(float(value) for value in step_sizes)):
        rows.extend(tuple(step * value for value in direction) for direction in directions)
    candidates = torch.tensor(rows, dtype=torch.float32)
    if torch.unique(candidates, dim=0).shape[0] != candidates.shape[0]:
        raise RuntimeError("candidate construction produced duplicate actions")
    return candidates


@torch.no_grad()
def build_candidate_quality_targets(
    state: ROIActionState,
    candidate_deltas: torch.Tensor,
    *,
    matched_gt_boxes: torch.Tensor,
    matched_gt_labels: torch.Tensor,
    image_sizes: list[tuple[int, int]],
    min_iou_gain: float = 0.002,
) -> CandidateQualityTargets:
    count = state.batch_size
    if candidate_deltas.ndim != 2 or candidate_deltas.shape[1] != 4:
        raise ValueError("candidate_deltas must have shape (K, 4)")
    if not torch.equal(candidate_deltas[0], torch.zeros_like(candidate_deltas[0])):
        raise ValueError("candidate index 0 must be identity")
    if matched_gt_boxes.shape != (count, 4) or matched_gt_labels.shape != (count,):
        raise ValueError("matched GT tensors do not match state batch")
    if state.matched_gt_indices is None:
        raise ValueError("state must include matched_gt_indices")
    if min_iou_gain < 0.0:
        raise ValueError("min_iou_gain must be non-negative")

    candidate_deltas = candidate_deltas.to(device=state.boxes.device, dtype=state.boxes.dtype)
    candidate_boxes = _apply_candidate_boxes(
        state.boxes,
        candidate_deltas,
        state.image_indices,
        image_sizes,
    )
    target_boxes = matched_gt_boxes[:, None, :].expand_as(candidate_boxes)
    quality = _elementwise_iou(
        candidate_boxes.reshape(-1, 4),
        target_boxes.reshape(-1, 4),
    ).reshape(count, candidate_deltas.shape[0])
    matched = state.matched_gt_indices >= 0
    class_correct = matched & state.labels.eq(matched_gt_labels)
    quality = torch.where(class_correct[:, None], quality, torch.zeros_like(quality))

    base_iou = quality[:, 0]
    best_quality, best_indices = quality.max(dim=1)
    oracle_gain = best_quality - base_iou
    move = class_correct & (oracle_gain > float(min_iou_gain))
    target_indices = torch.where(move, best_indices, torch.zeros_like(best_indices))
    return CandidateQualityTargets(
        candidate_quality=quality,
        target_indices=target_indices,
        base_iou=base_iou,
        oracle_gain=oracle_gain,
        matched=matched,
        class_correct=class_correct,
    )


def candidate_action_energy_loss(
    energies: torch.Tensor,
    *,
    candidate_quality: torch.Tensor,
    target_indices: torch.Tensor,
    scores: torch.Tensor,
    config: CandidateEnergyLossConfig | None = None,
) -> dict[str, torch.Tensor]:
    cfg = config or CandidateEnergyLossConfig()
    if energies.ndim != 2:
        raise ValueError("energies must have shape (B, K)")
    if candidate_quality.shape != energies.shape:
        raise ValueError("candidate_quality must share energies shape")
    if target_indices.shape != (energies.shape[0],) or scores.shape != target_indices.shape:
        raise ValueError("target_indices and scores must have shape (B,)")

    per_sample = F.cross_entropy(
        -energies / float(cfg.temperature),
        target_indices.long(),
        reduction="none",
    )
    oracle_gain = candidate_quality.max(dim=1).values - candidate_quality[:, 0]
    weights = (0.5 + scores.detach().clamp(0.0, 1.0)) * (1.0 + 4.0 * oracle_gain.detach())
    ranking = (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)
    energy = energies.pow(2).mean()
    total = ranking + float(cfg.energy_weight) * energy
    return {
        "loss_total": total,
        "loss_ranking": ranking.detach(),
        "loss_energy": energy.detach(),
        "oracle_move_rate": target_indices.ne(0).float().mean().detach(),
        "oracle_gain_mean": oracle_gain.mean().detach(),
    }


def select_min_energy_box_actions(
    state: ROIActionState,
    candidate_deltas: torch.Tensor,
    energies: torch.Tensor,
    *,
    min_energy_drop: float = 0.0,
    min_score: float = 0.05,
    require_foreground_dominant: bool = True,
    max_actions_per_image: int | None = None,
) -> tuple[ROITransportActions, torch.Tensor, torch.Tensor]:
    if candidate_deltas.ndim != 2 or candidate_deltas.shape[1] != 4:
        raise ValueError("candidate_deltas must have shape (K, 4)")
    if energies.shape != (state.batch_size, candidate_deltas.shape[0]):
        raise ValueError("energies must have shape (B, K)")
    if min_energy_drop < 0.0:
        raise ValueError("min_energy_drop must be non-negative")
    if max_actions_per_image is not None and max_actions_per_image <= 0:
        raise ValueError("max_actions_per_image must be positive")

    best_energy, selected = energies.min(dim=1)
    drop = energies[:, 0] - best_energy
    move = selected.ne(0) & (drop > float(min_energy_drop)) & state.scores.ge(float(min_score))
    if require_foreground_dominant and state.logits is not None and state.logits.shape[1] > 1:
        probabilities = torch.softmax(state.logits, dim=-1)
        move = move & (probabilities[:, 1:].amax(dim=1) > probabilities[:, 0])
    if max_actions_per_image is not None:
        limited = torch.zeros_like(move)
        for image_idx in torch.unique(state.image_indices, sorted=True).tolist():
            rows = torch.nonzero(move & state.image_indices.eq(image_idx), as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            order = torch.argsort(drop[rows], descending=True)
            limited[rows[order[: int(max_actions_per_image)]]] = True
        move = limited
    selected = torch.where(move, selected, torch.zeros_like(selected))
    candidates = candidate_deltas.to(device=energies.device, dtype=energies.dtype)
    box_delta = candidates[selected]
    actions = ROITransportActions(
        feature_delta=state.features.new_zeros(state.features.shape),
        score_delta=state.scores.new_zeros(state.scores.shape),
        box_delta=box_delta.to(dtype=state.boxes.dtype),
        keep_logit=state.scores.new_zeros(state.scores.shape),
    )
    return actions, selected, move


def _apply_candidate_boxes(
    boxes: torch.Tensor,
    candidate_deltas: torch.Tensor,
    image_indices: torch.Tensor,
    image_sizes: list[tuple[int, int]],
) -> torch.Tensor:
    count = boxes.shape[0]
    candidates = candidate_deltas.shape[0]
    expanded_boxes = boxes[:, None, :].expand(count, candidates, 4).reshape(-1, 4)
    expanded_delta = candidate_deltas[None, :, :].expand(count, candidates, 4).reshape(-1, 4)
    output = expanded_boxes.clone()
    expanded_images = image_indices[:, None].expand(count, candidates).reshape(-1)
    for image_idx, image_size in enumerate(image_sizes):
        mask = expanded_images == image_idx
        if mask.any():
            output[mask] = apply_box_delta(
                expanded_boxes[mask],
                expanded_delta[mask],
                image_size=image_size,
            )
    return output.reshape(count, candidates, 4)


def _elementwise_iou(boxes: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(boxes[:, :2], targets[:, :2])
    bottom_right = torch.minimum(boxes[:, 2:], targets[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0.0).prod(dim=1)
    area = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0.0).prod(dim=1)
    target_area = (targets[:, 2:] - targets[:, :2]).clamp_min(0.0).prod(dim=1)
    return intersection / (area + target_area - intersection).clamp_min(1e-8)
