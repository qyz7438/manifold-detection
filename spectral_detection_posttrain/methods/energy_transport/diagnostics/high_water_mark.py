"""High-water-mark teacher losses for action-local ROI transport.

The detector can already discover useful local actions near the AP75 boundary,
but later epochs may drift away from those actions.  This module provides a
small fixed-point primitive: compare the current action head against a frozen
high-water-mark teacher on the same ROI state, with optional emphasis on
proposals whose matched IoU is near the AP75 threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from spectral_detection_posttrain.methods.energy_transport.action.actions import (
    ROITransportActions,
)
from spectral_detection_posttrain.methods.energy_transport.action.contracts import (
    ROIActionState,
)


@dataclass(frozen=True)
class HighWaterMarkLossConfig:
    """Weights and margins for anchoring actions to a frozen best-epoch policy."""

    lambda_anchor: float = 1.0
    epsilon: float = 0.05
    iou_center: float = 0.75
    iou_band: float = 0.10
    boundary_power: float = 1.0
    min_boundary_weight: float = 0.0

    feature_weight: float = 0.0
    score_weight: float = 0.0
    box_weight: float = 1.0
    keep_weight: float = 1.0
    detach_teacher: bool = True

    def __post_init__(self) -> None:
        if self.lambda_anchor < 0.0:
            raise ValueError("lambda_anchor must be non-negative")
        if self.epsilon < 0.0:
            raise ValueError("epsilon must be non-negative")
        if not 0.0 <= self.iou_center <= 1.0:
            raise ValueError("iou_center must be in [0, 1]")
        if self.iou_band <= 0.0:
            raise ValueError("iou_band must be positive")
        if self.boundary_power <= 0.0:
            raise ValueError("boundary_power must be positive")
        if self.min_boundary_weight < 0.0:
            raise ValueError("min_boundary_weight must be non-negative")
        for name in (
            "feature_weight",
            "score_weight",
            "box_weight",
            "keep_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass
class HighWaterMarkModuleSnapshot:
    """Frozen copy of the action head that achieved the best validation metric."""

    metric_value: float | None = None
    epoch: int | None = None
    state_dict: dict[str, torch.Tensor] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_available(self) -> bool:
        return self.state_dict is not None


def should_update_high_water_mark(
    metric_value: float,
    snapshot: HighWaterMarkModuleSnapshot | None,
    *,
    min_delta: float = 0.0,
    higher_is_better: bool = True,
) -> bool:
    """Return whether a validation metric should replace the current snapshot."""

    if min_delta < 0.0:
        raise ValueError("min_delta must be non-negative")
    if snapshot is None or snapshot.metric_value is None:
        return True
    if higher_is_better:
        return float(metric_value) > float(snapshot.metric_value) + float(min_delta)
    return float(metric_value) < float(snapshot.metric_value) - float(min_delta)


def capture_high_water_mark_module(
    module: torch.nn.Module,
    *,
    metric_value: float,
    epoch: int,
    metadata: dict[str, Any] | None = None,
    to_cpu: bool = True,
) -> HighWaterMarkModuleSnapshot:
    """Clone a module state dict for later teacher restoration."""

    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    state: dict[str, torch.Tensor] = {}
    for key, value in module.state_dict().items():
        tensor = value.detach().clone()
        state[key] = tensor.cpu() if to_cpu else tensor
    return HighWaterMarkModuleSnapshot(
        metric_value=float(metric_value),
        epoch=int(epoch),
        state_dict=state,
        metadata=dict(metadata or {}),
    )


def load_high_water_mark_module(
    module: torch.nn.Module,
    snapshot: HighWaterMarkModuleSnapshot,
    *,
    strict: bool = True,
) -> torch.nn.Module:
    """Load a frozen high-water-mark snapshot into a compatible module."""

    if not snapshot.is_available:
        raise ValueError("snapshot has no state_dict")
    module.load_state_dict(snapshot.state_dict, strict=strict)
    return module


def ap75_boundary_weights(
    ious: torch.Tensor,
    *,
    center: float = 0.75,
    band: float = 0.10,
    power: float = 1.0,
    min_weight: float = 0.0,
) -> torch.Tensor:
    """Triangular proposal weights centered on the AP75 decision boundary."""

    if ious.ndim != 1:
        raise ValueError(f"ious must have shape (B,), got {ious.shape}")
    if not 0.0 <= center <= 1.0:
        raise ValueError("center must be in [0, 1]")
    if band <= 0.0:
        raise ValueError("band must be positive")
    if power <= 0.0:
        raise ValueError("power must be positive")
    if min_weight < 0.0:
        raise ValueError("min_weight must be non-negative")

    distance = (ious - float(center)).abs()
    weights = (1.0 - distance / float(band)).clamp_min(0.0)
    weights = weights.pow(float(power))
    if min_weight > 0.0:
        weights = weights.clamp_min(float(min_weight))
    return weights


def high_water_mark_action_loss(
    state: ROIActionState,
    actions: ROITransportActions,
    teacher_actions: ROITransportActions,
    config: HighWaterMarkLossConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Anchor current actions to a frozen high-water-mark teacher.

    The loss is a boundary-weighted margin on the per-proposal action distance.
    A proposal inside the margin pays no cost; outside the margin it is pulled
    back toward the teacher action.  ``state.ious`` is used to emphasize the
    AP75 boundary.  If IoUs are unavailable, all proposals receive equal weight.
    """

    cfg = config or HighWaterMarkLossConfig()
    _validate_action_shapes(actions, teacher_actions)
    zero = actions.feature_delta.sum() * 0.0
    if cfg.lambda_anchor <= 0.0 or _num_active_components(cfg) == 0:
        return _loss_dict(zero, zero, zero, zero, zero, zero, zero)

    teacher = _detach_actions(teacher_actions) if cfg.detach_teacher else teacher_actions

    feature_dist = _component_distance(
        actions.feature_delta,
        teacher.feature_delta,
        cfg.feature_weight,
    )
    score_dist = _component_distance(
        actions.score_delta,
        teacher.score_delta,
        cfg.score_weight,
    )
    box_dist = _component_distance(actions.box_delta, teacher.box_delta, cfg.box_weight)
    keep_dist = _component_distance(
        torch.sigmoid(actions.keep_logit),
        torch.sigmoid(teacher.keep_logit),
        cfg.keep_weight,
    )

    action_dist = feature_dist + score_dist + box_dist + keep_dist
    margin_dist = torch.sqrt(action_dist.clamp_min(1e-12))
    per_sample = F.relu(margin_dist - float(cfg.epsilon)).pow(2)

    if state.ious is None:
        weights = torch.ones_like(per_sample)
    else:
        weights = ap75_boundary_weights(
            state.ious.to(device=per_sample.device, dtype=per_sample.dtype),
            center=cfg.iou_center,
            band=cfg.iou_band,
            power=cfg.boundary_power,
            min_weight=cfg.min_boundary_weight,
        )

    denom = weights.sum().clamp_min(1.0)
    anchor = float(cfg.lambda_anchor) * (weights * per_sample).sum() / denom
    return _loss_dict(
        anchor,
        feature_dist.detach().mean(),
        score_dist.detach().mean(),
        box_dist.detach().mean(),
        keep_dist.detach().mean(),
        weights.detach().mean(),
        margin_dist.detach().mean(),
    )


def stop_high_water_mark_loss(
    state: ROIActionState,
    actions: ROITransportActions,
    teacher_actions: ROITransportActions,
    *,
    lambda_stop: float = 0.02,
    epsilon: float = 0.05,
    iou_center: float = 0.75,
    iou_band: float = 0.10,
) -> dict[str, torch.Tensor]:
    """Convenience wrapper for the stop/gate-only high-water-mark loss."""

    return high_water_mark_action_loss(
        state,
        actions,
        teacher_actions,
        HighWaterMarkLossConfig(
            lambda_anchor=lambda_stop,
            epsilon=epsilon,
            iou_center=iou_center,
            iou_band=iou_band,
            feature_weight=0.0,
            score_weight=0.0,
            box_weight=0.0,
            keep_weight=1.0,
        ),
    )


def _component_distance(
    value: torch.Tensor,
    target: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    batch = value.shape[0]
    if weight <= 0.0:
        return value.new_zeros(batch)
    diff = value - target.to(device=value.device, dtype=value.dtype)
    if diff.ndim == 1:
        return float(weight) * diff.pow(2)
    return float(weight) * diff.pow(2).flatten(1).mean(dim=1)


def _validate_action_shapes(
    actions: ROITransportActions,
    teacher_actions: ROITransportActions,
) -> None:
    for name in ("feature_delta", "score_delta", "box_delta", "keep_logit"):
        value = getattr(actions, name)
        target = getattr(teacher_actions, name)
        if value.shape != target.shape:
            raise ValueError(
                f"{name} shape mismatch: {value.shape} vs {target.shape}"
            )


def _detach_actions(actions: ROITransportActions) -> ROITransportActions:
    return ROITransportActions(
        feature_delta=actions.feature_delta.detach(),
        score_delta=actions.score_delta.detach(),
        box_delta=actions.box_delta.detach(),
        keep_logit=actions.keep_logit.detach(),
    )


def _num_active_components(config: HighWaterMarkLossConfig) -> int:
    return sum(
        weight > 0.0
        for weight in (
            config.feature_weight,
            config.score_weight,
            config.box_weight,
            config.keep_weight,
        )
    )


def _loss_dict(
    total: torch.Tensor,
    feature_dist: torch.Tensor,
    score_dist: torch.Tensor,
    box_dist: torch.Tensor,
    keep_dist: torch.Tensor,
    boundary_weight: torch.Tensor,
    action_distance: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "loss_hwm_total": total,
        "loss_hwm_feature_dist": feature_dist,
        "loss_hwm_score_dist": score_dist,
        "loss_hwm_box_dist": box_dist,
        "loss_hwm_keep_dist": keep_dist,
        "loss_hwm_boundary_weight": boundary_weight,
        "loss_hwm_action_distance": action_distance,
    }
