"""Typed contracts for action-local detector transport."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ConstraintConfig:
    """Safety constraints shared by score, box, and feature actions."""

    score_threshold: float = 0.05
    max_score_delta: float = 0.2
    max_box_delta: float = 0.2
    max_rescues_per_image: int = 1
    threshold_margin: float = 0.0
    energy_weight: float = 1.0
    threshold_weight: float = 1.0
    budget_weight: float = 1.0
    min_quality_margin: float = 0.0
    min_iou_floor: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0, 1]")
        if self.max_score_delta <= 0.0:
            raise ValueError("max_score_delta must be positive")
        if self.max_box_delta <= 0.0:
            raise ValueError("max_box_delta must be positive")
        if self.max_rescues_per_image < 0:
            raise ValueError("max_rescues_per_image must be non-negative")
        for name in (
            "threshold_margin",
            "energy_weight",
            "threshold_weight",
            "budget_weight",
            "min_quality_margin",
            "min_iou_floor",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class ROIActionState:
    """Proposal-aligned ROI state before taking a transport action."""

    features: torch.Tensor
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor
    image_indices: torch.Tensor
    proposal_indices: torch.Tensor
    logits: torch.Tensor | None = None
    matched_gt_indices: torch.Tensor | None = None
    ious: torch.Tensor | None = None
    verifier_scores: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError(
                f"features must have shape (B, D), got {self.features.shape}"
            )
        batch = self.features.shape[0]
        _check_shape("boxes", self.boxes, (batch, 4))
        _check_shape("scores", self.scores, (batch,))
        _check_shape("labels", self.labels, (batch,))
        _check_shape("image_indices", self.image_indices, (batch,))
        _check_shape("proposal_indices", self.proposal_indices, (batch,))
        if self.logits is not None and (
            self.logits.ndim != 2 or self.logits.shape[0] != batch
        ):
            raise ValueError("logits must have shape (B, C)")
        if self.matched_gt_indices is not None:
            _check_shape("matched_gt_indices", self.matched_gt_indices, (batch,))
        if self.ious is not None:
            _check_shape("ious", self.ious, (batch,))
        if self.verifier_scores is not None:
            _check_shape("verifier_scores", self.verifier_scores, (batch,))

    @property
    def batch_size(self) -> int:
        return int(self.features.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])


@dataclass(frozen=True)
class ActionOutcome:
    """Post-action proposal state for diagnostics and rewards."""

    boxes: torch.Tensor
    scores: torch.Tensor
    old_scores: torch.Tensor
    labels: torch.Tensor
    threshold: float
    features: torch.Tensor | None = None
    ious: torch.Tensor | None = None
    keep_prob: torch.Tensor | None = None

    def __post_init__(self) -> None:
        batch = self.scores.shape[0]
        _check_shape("boxes", self.boxes, (batch, 4))
        _check_shape("old_scores", self.old_scores, (batch,))
        _check_shape("labels", self.labels, (batch,))
        if self.features is not None and self.features.ndim != 2:
            raise ValueError("features must have shape (B, D)")
        if self.features is not None and self.features.shape[0] != batch:
            raise ValueError("features batch dim must match scores")
        if self.ious is not None:
            _check_shape("ious", self.ious, (batch,))
        if self.keep_prob is not None:
            _check_shape("keep_prob", self.keep_prob, (batch,))
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")

    @property
    def threshold_crossings(self) -> torch.Tensor:
        return (self.old_scores < self.threshold) & (self.scores >= self.threshold)


@dataclass(frozen=True)
class PreferenceBatch:
    """Pairwise preference contract for action-local DPO style objectives."""

    chosen_indices: torch.Tensor
    rejected_indices: torch.Tensor
    quality_gap: torch.Tensor
    iou_gap: torch.Tensor
    valid_mask: torch.Tensor
    reference_margin: torch.Tensor | None = None

    def __post_init__(self) -> None:
        shape = self.chosen_indices.shape
        for name, value in (
            ("rejected_indices", self.rejected_indices),
            ("quality_gap", self.quality_gap),
            ("iou_gap", self.iou_gap),
            ("valid_mask", self.valid_mask),
        ):
            if value.shape != shape:
                raise ValueError(
                    "preference tensors must share shape; "
                    f"{name} has {value.shape}, expected {shape}"
                )
        if self.reference_margin is not None and self.reference_margin.shape != shape:
            raise ValueError("reference_margin must share shape with preference tensors")

    @property
    def num_valid(self) -> int:
        return int(self.valid_mask.bool().sum().item())


def _check_shape(name: str, value: torch.Tensor, expected: tuple[int, ...]) -> None:
    if value.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
