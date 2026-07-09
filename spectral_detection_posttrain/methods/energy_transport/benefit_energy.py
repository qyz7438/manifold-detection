"""Pairwise energy model for choosing a local box action over identity."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from spectral_detection_posttrain.methods.energy_transport.actions import ROITransportActions
from spectral_detection_posttrain.methods.energy_transport.contracts import ROIActionState
from spectral_detection_posttrain.methods.energy_transport.operators import apply_box_delta


@dataclass(frozen=True)
class BenefitEnergyLossConfig:
    min_positive_gain: float = 0.005
    temperature: float = 0.05
    ranking_weight: float = 1.0
    regression_weight: float = 1.0
    boundary_iou: float = 0.75
    boundary_band: float = 0.10
    boundary_boost: float = 2.0
    foreground_boost: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "min_positive_gain",
            "ranking_weight",
            "regression_weight",
            "boundary_boost",
            "foreground_boost",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.boundary_band <= 0.0:
            raise ValueError("boundary_band must be positive")
        if not 0.0 <= self.boundary_iou <= 1.0:
            raise ValueError("boundary_iou must be in [0, 1]")


@dataclass(frozen=True)
class ActionBenefitTargets:
    true_gain: torch.Tensor
    base_iou: torch.Tensor
    post_iou: torch.Tensor
    matched: torch.Tensor
    class_correct: torch.Tensor
    foreground_dominant: torch.Tensor


class ActionBenefitEnergyHead(nn.Module):
    """Score identity and a proposed action with one shared conditional energy."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or num_classes <= 1 or hidden_dim <= 0:
            raise ValueError("feature_dim, num_classes, and hidden_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)

        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.hidden_dim),
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

    def forward(
        self,
        features: torch.Tensor,
        class_logits: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        box_delta: torch.Tensor,
    ) -> torch.Tensor:
        batch = features.shape[0]
        if features.shape != (batch, self.feature_dim):
            raise ValueError(f"features must have shape (B, {self.feature_dim})")
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
        ).to(dtype=features.dtype)
        context = torch.cat(
            (
                probabilities.to(dtype=features.dtype),
                one_hot,
                scores[:, None].to(dtype=features.dtype),
                box_delta.to(dtype=features.dtype),
            ),
            dim=1,
        )
        feature_code = self.feature_encoder(features)
        context_code = self.context_encoder(context)
        return self.energy_head(torch.cat((feature_code, context_code), dim=1)).squeeze(1)

    def energy_gap(
        self,
        features: torch.Tensor,
        class_logits: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        box_delta: torch.Tensor,
    ) -> torch.Tensor:
        """Return E(identity) - E(action), positive when the action is preferred."""

        identity = torch.zeros_like(box_delta)
        base_energy = self(features, class_logits, labels, scores, identity)
        action_energy = self(features, class_logits, labels, scores, box_delta)
        return base_energy - action_energy


@torch.no_grad()
def build_action_benefit_targets(
    state: ROIActionState,
    actions: ROITransportActions,
    *,
    matched_gt_boxes: torch.Tensor,
    matched_gt_labels: torch.Tensor,
    image_sizes: list[tuple[int, int]],
) -> ActionBenefitTargets:
    count = state.batch_size
    if matched_gt_boxes.shape != (count, 4):
        raise ValueError("matched_gt_boxes must have shape (B, 4)")
    if matched_gt_labels.shape != (count,):
        raise ValueError("matched_gt_labels must have shape (B,)")
    if state.matched_gt_indices is None:
        raise ValueError("state must include matched_gt_indices")

    post_boxes = state.boxes.clone()
    for image_idx, image_size in enumerate(image_sizes):
        mask = state.image_indices == image_idx
        if mask.any():
            post_boxes[mask] = apply_box_delta(
                state.boxes[mask],
                actions.box_delta[mask],
                image_size=image_size,
            )
    base_iou = _elementwise_iou(state.boxes, matched_gt_boxes)
    post_iou = _elementwise_iou(post_boxes, matched_gt_boxes)
    matched = state.matched_gt_indices >= 0
    base_iou = torch.where(matched, base_iou, torch.zeros_like(base_iou))
    post_iou = torch.where(matched, post_iou, torch.zeros_like(post_iou))
    class_correct = matched & state.labels.eq(matched_gt_labels)

    if state.logits is None or state.logits.shape[1] <= 1:
        foreground_dominant = torch.ones_like(matched)
    else:
        probabilities = torch.softmax(state.logits, dim=-1)
        foreground_dominant = probabilities[:, 1:].amax(dim=1) > probabilities[:, 0]
    return ActionBenefitTargets(
        true_gain=post_iou - base_iou,
        base_iou=base_iou,
        post_iou=post_iou,
        matched=matched,
        class_correct=class_correct,
        foreground_dominant=foreground_dominant,
    )


def action_benefit_energy_loss(
    predicted_gain: torch.Tensor,
    *,
    true_gain: torch.Tensor,
    base_iou: torch.Tensor,
    class_correct: torch.Tensor,
    foreground_dominant: torch.Tensor,
    scores: torch.Tensor,
    config: BenefitEnergyLossConfig | None = None,
) -> dict[str, torch.Tensor]:
    cfg = config or BenefitEnergyLossConfig()
    shape = predicted_gain.shape
    for name, value in (
        ("true_gain", true_gain),
        ("base_iou", base_iou),
        ("class_correct", class_correct),
        ("foreground_dominant", foreground_dominant),
        ("scores", scores),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} must share predicted_gain shape")

    class_correct = class_correct.bool()
    foreground_dominant = foreground_dominant.bool()
    positive = class_correct & (true_gain > float(cfg.min_positive_gain))
    target = positive.to(dtype=predicted_gain.dtype)
    logits = predicted_gain / float(cfg.temperature)

    boundary = torch.exp(
        -(base_iou - float(cfg.boundary_iou)).abs() / float(cfg.boundary_band)
    )
    relevance = (
        (0.5 + scores.detach().clamp(0.0, 1.0))
        * (1.0 + float(cfg.boundary_boost) * boundary.detach())
        * (1.0 + float(cfg.foreground_boost) * foreground_dominant.float())
    )
    positive_count = positive.sum().clamp_min(1).to(dtype=predicted_gain.dtype)
    negative_count = (~positive).sum().clamp_min(1).to(dtype=predicted_gain.dtype)
    balance = torch.where(
        positive,
        0.5 * float(predicted_gain.numel()) / positive_count,
        0.5 * float(predicted_gain.numel()) / negative_count,
    )
    ranking_per_sample = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    ranking = _weighted_mean(ranking_per_sample, relevance * balance)

    if class_correct.any():
        regression_per_sample = F.smooth_l1_loss(
            predicted_gain[class_correct],
            true_gain[class_correct].detach(),
            reduction="none",
            beta=0.02,
        )
        regression = _weighted_mean(regression_per_sample, relevance[class_correct])
    else:
        regression = predicted_gain.sum() * 0.0
    total = float(cfg.ranking_weight) * ranking + float(cfg.regression_weight) * regression
    return {
        "loss_total": total,
        "loss_ranking": ranking.detach(),
        "loss_regression": regression.detach(),
        "positive_rate": positive.float().mean().detach(),
        "predicted_gain_mean": predicted_gain.mean().detach(),
        "true_gain_mean": true_gain[class_correct].mean().detach()
        if class_correct.any()
        else true_gain.sum().detach() * 0.0,
    }


def apply_action_benefit_gate(
    state: ROIActionState,
    actions: ROITransportActions,
    predicted_gain: torch.Tensor,
    *,
    min_predicted_gain: float = 0.0,
    min_score: float = 0.05,
    require_foreground_dominant: bool = True,
    max_actions_per_image: int | None = None,
) -> tuple[ROITransportActions, torch.Tensor]:
    if predicted_gain.shape != state.scores.shape:
        raise ValueError("predicted_gain must contain one value per proposal")
    if min_score < 0.0 or min_score > 1.0:
        raise ValueError("min_score must be in [0, 1]")
    if max_actions_per_image is not None and max_actions_per_image <= 0:
        raise ValueError("max_actions_per_image must be positive")

    mask = (predicted_gain > float(min_predicted_gain)) & (state.scores >= float(min_score))
    if require_foreground_dominant and state.logits is not None and state.logits.shape[1] > 1:
        probabilities = torch.softmax(state.logits, dim=-1)
        mask = mask & (probabilities[:, 1:].amax(dim=1) > probabilities[:, 0])
    if max_actions_per_image is not None:
        limited = torch.zeros_like(mask)
        for image_idx in torch.unique(state.image_indices, sorted=True).tolist():
            rows = torch.nonzero(mask & (state.image_indices == image_idx), as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            order = torch.argsort(predicted_gain[rows], descending=True)
            keep = rows[order[: int(max_actions_per_image)]]
            limited[keep] = True
        mask = limited

    gated_delta = torch.where(mask[:, None], actions.box_delta, torch.zeros_like(actions.box_delta))
    gated = ROITransportActions(
        feature_delta=torch.zeros_like(actions.feature_delta),
        score_delta=torch.zeros_like(actions.score_delta),
        box_delta=gated_delta,
        keep_logit=torch.zeros_like(actions.keep_logit),
    )
    return gated, mask


def _elementwise_iou(boxes: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(boxes[:, :2], targets[:, :2])
    bottom_right = torch.minimum(boxes[:, 2:], targets[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0.0).prod(dim=1)
    area = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0.0).prod(dim=1)
    target_area = (targets[:, 2:] - targets[:, :2]).clamp_min(0.0).prod(dim=1)
    return intersection / (area + target_area - intersection).clamp_min(1e-8)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)
