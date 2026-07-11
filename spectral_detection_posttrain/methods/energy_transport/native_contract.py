"""Locked detector-native candidate and parity contracts for C1 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class DetectorNativeCandidates:
    """Proposal rows selected using detector-visible values only."""

    spatial_features: torch.Tensor
    class_logits: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor
    boxes: torch.Tensor
    image_indices: torch.Tensor
    proposal_indices: torch.Tensor


def build_native_c1_deltas(step: float = 0.05) -> torch.Tensor:
    """Return no-op followed by one-coordinate translation/scale actions."""
    if not 0.0 < float(step) <= 1.0:
        raise ValueError("step must be in (0, 1]")
    deltas = torch.zeros((9, 4), dtype=torch.float32)
    row = 1
    for coordinate in range(4):
        deltas[row, coordinate] = float(step)
        deltas[row + 1, coordinate] = -float(step)
        row += 2
    return deltas


def build_detector_native_candidates(
    spatial_features: torch.Tensor,
    class_logits: torch.Tensor,
    scores: torch.Tensor,
    boxes: torch.Tensor,
    image_indices: torch.Tensor,
    score_threshold: float = 0.05,
) -> DetectorNativeCandidates:
    """Filter proposal rows without ground-truth or oracle utility inputs."""
    if spatial_features.ndim != 4:
        raise ValueError("spatial_features must have shape [N, C, H, W]")
    if class_logits.ndim != 2:
        raise ValueError("class_logits must have shape [N, K]")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [N, 4]")
    count = spatial_features.shape[0]
    tensors = (class_logits, scores, boxes, image_indices)
    if any(tensor.shape[0] != count for tensor in tensors):
        raise ValueError("all proposal tensors must have the same leading dimension")
    if scores.ndim != 1 or image_indices.ndim != 1:
        raise ValueError("scores and image_indices must be vectors")
    if class_logits.shape[1] < 2:
        raise ValueError("class_logits must include background and foreground classes")
    if not 0.0 <= float(score_threshold) <= 1.0:
        raise ValueError("score_threshold must be in [0, 1]")

    devices = {tensor.device for tensor in (spatial_features, class_logits, scores, boxes, image_indices)}
    if len(devices) != 1:
        raise ValueError("all proposal tensors must be on the same device")
    predicted = class_logits.argmax(dim=1)
    observable = predicted.gt(0) & scores.ge(float(score_threshold))
    proposal_indices = torch.arange(count, device=spatial_features.device)[observable]
    return DetectorNativeCandidates(
        spatial_features=spatial_features[observable],
        class_logits=class_logits[observable],
        labels=predicted[observable],
        scores=scores[observable],
        boxes=boxes[observable],
        image_indices=image_indices[observable],
        proposal_indices=proposal_indices,
    )


def validate_strict_parity_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the aggregate and per-image exact-zero native parity gates."""
    aggregate = artifact.get("aggregate_zero_action_parity", {})
    strict = artifact.get("strict_zero_action_parity", {})
    images = strict.get("images")
    mismatched = strict.get("mismatched_images")
    box_error = strict.get("max_box_abs_error")
    score_error = strict.get("max_score_abs_error")
    checks = {
        "completed": artifact.get("completed") is True,
        "aggregate_passed": aggregate.get("passed") is True,
        "strict_passed": strict.get("passed") is True,
        "positive_image_coverage": type(images) is int and images > 0,
        "mismatched_images_zero": type(mismatched) is int and mismatched == 0,
        "actions_are_exact_zero": strict.get("actions_are_exact_zero") is True,
        "max_box_abs_error_zero": type(box_error) in (int, float) and box_error == 0.0,
        "max_score_abs_error_zero": type(score_error) in (int, float) and score_error == 0.0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def evaluate_native_contract_gates(
    payload: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate locked parity, action-space, budget, and provenance gates."""
    parity = payload.get("parity", {})
    contract = payload.get("candidate_contract", {})
    required = config.get("gates", {})
    checks = {
        "baseline_parity": parity.get("baseline", {}).get("passed") is True,
        "fullft_parity": parity.get("fullft", {}).get("passed") is True,
        "candidate_count": contract.get("candidate_count")
        == required.get("required_candidate_count"),
        "noop_is_first": contract.get("noop_is_first") is True,
        "max_abs_delta": contract.get("max_abs_delta")
        == required.get("required_max_abs_delta"),
        "budget_one": contract.get("max_actions_per_image")
        == required.get("required_max_actions_per_image"),
        "budget_enforced": contract.get("budget_enforced") is True,
        "detector_only": contract.get("detector_only") is True,
        "score_threshold": contract.get("score_threshold")
        == required.get("required_score_threshold"),
        "candidate_source": contract.get("candidate_source")
        == required.get("required_candidate_source")
        == "detector_visible_proposals_only",
        "training_gt_scope": contract.get("training_gt_scope")
        == required.get("required_training_gt_scope")
        == "utility_labels_only_never_candidate_filter",
    }
    return {"all_passed": all(checks.values()), "checks": checks}
