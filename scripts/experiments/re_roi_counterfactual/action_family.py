"""Frozen action family for the re-ROI counterfactual evidence protocol."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch


ActionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class ActionSpec:
    """A single frozen action in the re-ROI protocol."""

    family: str
    apply: ActionFn
    energy: float


def _identity(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return boxes.clone(), scores.clone()


def _score_down(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return boxes.clone(), (scores - 0.02).clamp(0.0, 1.0)


def _score_up(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return boxes.clone(), (scores + 0.02).clamp(0.0, 1.0)


def _translate(dx: float, dy: float) -> ActionFn:
    def _apply(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        shifted = boxes.clone()
        shifted[:, 0] += dx * widths
        shifted[:, 2] += dx * widths
        shifted[:, 1] += dy * heights
        shifted[:, 3] += dy * heights
        return shifted, scores.clone()

    return _apply


def _scale(factor: float) -> ActionFn:
    def _apply(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        new_widths = widths * factor
        new_heights = heights * factor
        scaled = boxes.new_zeros(boxes.shape)
        scaled[:, 0] = ctr_x - 0.5 * new_widths
        scaled[:, 1] = ctr_y - 0.5 * new_heights
        scaled[:, 2] = ctr_x + 0.5 * new_widths
        scaled[:, 3] = ctr_y + 0.5 * new_heights
        return scaled, scores.clone()

    return _apply


def _drop(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop returns empty tensors; the caller must handle removed detections."""
    return boxes.new_zeros((0, 4)), scores.new_zeros((0,))


ACTION_FAMILY: tuple[ActionSpec, ...] = (
    ActionSpec("identity_permutation", _identity, energy=0.0),
    ActionSpec("score_down", _score_down, energy=0.02),
    ActionSpec("score_up", _score_up, energy=0.02),
    ActionSpec("translate_left", _translate(-0.02, 0.0), energy=0.02),
    ActionSpec("translate_right", _translate(0.02, 0.0), energy=0.02),
    ActionSpec("translate_up", _translate(0.0, -0.02), energy=0.02),
    ActionSpec("translate_down", _translate(0.0, 0.02), energy=0.02),
    # A +/-0.02 log-area step corresponds to exp(+/-0.01) per side.
    ActionSpec("scale_down", _scale(math.exp(-0.01)), energy=0.02),
    ActionSpec("scale_up", _scale(math.exp(0.01)), energy=0.02),
    ActionSpec("drop", _drop, energy=0.05),
)


def get_action_family() -> tuple[ActionSpec, ...]:
    """Return the frozen action family."""
    return ACTION_FAMILY


def action_family_hash() -> str:
    """Return a stable string identifier for the frozen action family."""
    return "re_roi_action_family_v2_9_non_identity_002_step"
