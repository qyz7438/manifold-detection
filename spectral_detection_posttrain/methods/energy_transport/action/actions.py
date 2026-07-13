"""Small action and constraint primitives for energy-guided ROI transport."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ROITransportActions:
    """Action outputs for one set of ROI features.

    The action space is intentionally local: it permits small feature, score,
    and box updates plus an optional keep/reject logit.  Downstream trainers can
    choose which parts to activate while sharing the same diagnostics.
    """

    feature_delta: torch.Tensor
    score_delta: torch.Tensor
    box_delta: torch.Tensor
    keep_logit: torch.Tensor


class ActionLocalTransportHead(nn.Module):
    """Predict bounded local correction actions from ROI features.

    The head is initialized as an identity/no-op policy.  This mirrors the safe
    ChordEdit-style transport intuition: learn a low-energy displacement only
    where the verifier or detection objective supports taking action.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int | None = None,
        max_score_delta: float = 0.2,
        max_box_delta: float = 0.2,
        residual_scale: float = 0.0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if max_score_delta <= 0.0:
            raise ValueError("max_score_delta must be positive")
        if max_box_delta <= 0.0:
            raise ValueError("max_box_delta must be positive")

        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim or feature_dim)
        self.max_score_delta = float(max_score_delta)
        self.max_box_delta = float(max_box_delta)

        self.trunk = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_head = nn.Linear(self.hidden_dim, self.feature_dim)
        self.score_head = nn.Linear(self.hidden_dim, 1)
        self.box_head = nn.Linear(self.hidden_dim, 4)
        self.keep_head = nn.Linear(self.hidden_dim, 1)
        self._init_near_identity(residual_scale)

    def forward(self, features: torch.Tensor) -> ROITransportActions:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must have shape (B, {self.feature_dim}), got {features.shape}"
            )

        hidden = self.trunk(features)
        feature_delta = self.feature_head(hidden)
        score_delta = self.max_score_delta * torch.tanh(
            self.score_head(hidden)
        ).squeeze(-1)
        box_delta = self.max_box_delta * torch.tanh(self.box_head(hidden))
        keep_logit = self.keep_head(hidden).squeeze(-1)
        return ROITransportActions(
            feature_delta=feature_delta,
            score_delta=score_delta,
            box_delta=box_delta,
            keep_logit=keep_logit,
        )

    def _init_near_identity(self, residual_scale: float) -> None:
        for head in (self.feature_head, self.score_head, self.box_head, self.keep_head):
            nn.init.normal_(head.weight, mean=0.0, std=residual_scale)
            nn.init.zeros_(head.bias)


def apply_bounded_score_delta(
    scores: torch.Tensor,
    raw_delta: torch.Tensor,
    max_delta: float,
) -> torch.Tensor:
    """Apply a bounded residual score action and clamp to valid probabilities."""
    if max_delta <= 0.0:
        raise ValueError("max_delta must be positive")
    if scores.shape != raw_delta.shape:
        raise ValueError(
            "scores and raw_delta must share shape, "
            f"got {scores.shape} and {raw_delta.shape}"
        )
    delta = float(max_delta) * torch.tanh(raw_delta)
    return (scores + delta).clamp(0.0, 1.0)


def threshold_preservation_loss(
    old_scores: torch.Tensor,
    score_delta: torch.Tensor,
    *,
    low_quality_mask: torch.Tensor,
    threshold: float,
    margin: float = 0.0,
) -> torch.Tensor:
    """Penalize low-quality candidates that cross the eval score threshold."""
    if (
        old_scores.shape != score_delta.shape
        or old_scores.shape != low_quality_mask.shape
    ):
        raise ValueError("old_scores, score_delta, and low_quality_mask must share shape")
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")

    new_scores = old_scores + score_delta
    crossing = F.relu(new_scores - (float(threshold) - float(margin)))
    selected = crossing[low_quality_mask.bool()]
    if selected.numel() == 0:
        return score_delta.sum() * 0.0
    return (selected ** 2).mean()


def rescue_budget_loss(
    old_scores: torch.Tensor,
    score_delta: torch.Tensor,
    *,
    candidate_mask: torch.Tensor,
    threshold: float,
    max_rescues: float,
    temperature: float = 0.02,
) -> torch.Tensor:
    """Softly cap how many candidate scores may cross a threshold.

    This encodes the main lesson from oracle score rescue: raising many boxes can
    improve AP75 while flooding evaluation with false positives.
    """
    if (
        old_scores.shape != score_delta.shape
        or old_scores.shape != candidate_mask.shape
    ):
        raise ValueError("old_scores, score_delta, and candidate_mask must share shape")
    if max_rescues < 0.0:
        raise ValueError("max_rescues must be non-negative")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    candidate_mask = candidate_mask.bool()
    if candidate_mask.sum().item() == 0:
        return score_delta.sum() * 0.0

    new_scores = old_scores + score_delta
    soft_crossings = torch.sigmoid(
        (new_scores[candidate_mask] - float(threshold)) / float(temperature)
    )
    overflow = F.relu(soft_crossings.sum() - float(max_rescues))
    return overflow ** 2


def transport_action_energy(
    feature_delta: torch.Tensor,
    *,
    score_delta: torch.Tensor | None = None,
    box_delta: torch.Tensor | None = None,
    score_weight: float = 1.0,
    box_weight: float = 1.0,
) -> torch.Tensor:
    """Squared low-energy penalty for a local transport action."""
    if feature_delta.ndim != 2:
        raise ValueError(
            f"feature_delta must have shape (B, D), got {feature_delta.shape}"
        )
    if score_weight < 0.0 or box_weight < 0.0:
        raise ValueError("energy weights must be non-negative")

    per_sample = (feature_delta ** 2).sum(dim=-1)
    batch = feature_delta.shape[0]
    if score_delta is not None:
        if score_delta.shape != (batch,):
            raise ValueError(
                f"score_delta must have shape ({batch},), got {score_delta.shape}"
            )
        per_sample = per_sample + float(score_weight) * (score_delta ** 2)
    if box_delta is not None:
        if box_delta.shape != (batch, 4):
            raise ValueError(
                f"box_delta must have shape ({batch}, 4), got {box_delta.shape}"
            )
        per_sample = per_sample + float(box_weight) * (box_delta ** 2).sum(dim=-1)
    return per_sample.mean()


def summarize_score_actions(
    old_scores: torch.Tensor,
    score_delta: torch.Tensor,
    *,
    threshold: float,
    low_quality_mask: torch.Tensor | None = None,
    candidate_mask: torch.Tensor | None = None,
) -> dict[str, float | int]:
    """Return scalar diagnostics for score-rescue pressure.

    The summary is detached by design.  It is meant for experiment metadata and
    safety gates, especially the failure mode where pairwise preferences improve
    while too many low-quality proposals cross the eval threshold.
    """
    if old_scores.shape != score_delta.shape:
        raise ValueError("old_scores and score_delta must share shape")
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if low_quality_mask is None:
        low_quality_mask = torch.zeros_like(old_scores, dtype=torch.bool)
    if candidate_mask is None:
        candidate_mask = torch.ones_like(old_scores, dtype=torch.bool)
    if (
        old_scores.shape != low_quality_mask.shape
        or old_scores.shape != candidate_mask.shape
    ):
        raise ValueError("masks must share shape with old_scores")

    with torch.no_grad():
        old_scores_det = old_scores.detach()
        score_delta_det = score_delta.detach()
        low_quality = low_quality_mask.detach().bool()
        candidates = candidate_mask.detach().bool()
        new_scores = old_scores_det + score_delta_det
        crossings = (
            candidates
            & (old_scores_det < float(threshold))
            & (new_scores >= float(threshold))
        )
        low_quality_crossings = crossings & low_quality

        if low_quality.any():
            mean_low_quality = float(score_delta_det[low_quality].mean().item())
        else:
            mean_low_quality = 0.0
        if candidates.any():
            mean_candidate = float(score_delta_det[candidates].mean().item())
        else:
            mean_candidate = 0.0

        return {
            "num_candidates": int(candidates.sum().item()),
            "num_threshold_crossings": int(crossings.sum().item()),
            "num_low_quality_crossings": int(low_quality_crossings.sum().item()),
            "mean_score_delta": (
                float(score_delta_det.mean().item()) if score_delta_det.numel() else 0.0
            ),
            "mean_candidate_delta": mean_candidate,
            "mean_low_quality_delta": mean_low_quality,
        }
