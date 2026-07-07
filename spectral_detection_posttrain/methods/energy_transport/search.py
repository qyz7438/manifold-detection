"""Oracle-bounded action search for energy-guided detector transport."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActionSearchConfig:
    """Constraints for selecting a low-energy next action in score space.

    This search is intentionally local and conservative: a verified high-IoU
    candidate receives the smallest score increase that reaches the target
    state, and at most ``max_rescues_per_image`` such actions are selected.
    """

    score_threshold: float = 0.05
    target_iou: float = 0.75
    low_quality_iou: float = 0.30
    max_score_delta: float = 0.20
    threshold_margin: float = 0.01
    max_rescues_per_image: int = 1
    positive_target_score: float | None = None
    demote_low_quality: bool = False
    demote_target_score: float | None = None

    def __post_init__(self) -> None:
        for name in ("score_threshold", "target_iou", "low_quality_iou"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.max_score_delta <= 0.0:
            raise ValueError("max_score_delta must be positive")
        if self.threshold_margin < 0.0:
            raise ValueError("threshold_margin must be non-negative")
        if self.max_rescues_per_image < 0:
            raise ValueError("max_rescues_per_image must be non-negative")
        for name in ("positive_target_score", "demote_target_score"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class ScoreActionSearchResult:
    """Selected residual score actions and detached diagnostics."""

    score_delta: torch.Tensor
    rescue_mask: torch.Tensor
    demote_mask: torch.Tensor
    summary: dict[str, float | int]


def select_min_energy_score_actions(
    scores: torch.Tensor,
    ious: torch.Tensor,
    image_indices: torch.Tensor,
    *,
    gt_indices: torch.Tensor | None = None,
    rescue_candidate_mask: torch.Tensor | None = None,
    low_quality_mask: torch.Tensor | None = None,
    config: ActionSearchConfig | None = None,
) -> ScoreActionSearchResult:
    """Select low-energy residual score actions for proposal candidates.

    The endpoint is not a class prototype.  It is the nearest feasible detector
    state where a verified high-IoU candidate crosses the evaluation threshold.
    ``ious`` and ``rescue_candidate_mask`` may come from an oracle in analysis
    runs or from a learned verifier in deployable runs.
    """
    cfg = config or ActionSearchConfig()
    _check_vector("scores", scores)
    _check_vector("ious", ious)
    _check_vector("image_indices", image_indices)
    if scores.shape != ious.shape or scores.shape != image_indices.shape:
        raise ValueError("scores, ious, and image_indices must share shape")

    if gt_indices is None:
        gt_indices = torch.arange(scores.numel(), device=scores.device)
    _check_vector("gt_indices", gt_indices)
    if gt_indices.shape != scores.shape:
        raise ValueError("gt_indices must share shape with scores")

    if rescue_candidate_mask is None:
        rescue_candidate_mask = torch.ones_like(scores, dtype=torch.bool)
    if low_quality_mask is None:
        low_quality_mask = ious <= float(cfg.low_quality_iou)
    if rescue_candidate_mask.shape != scores.shape or low_quality_mask.shape != scores.shape:
        raise ValueError("masks must share shape with scores")

    score_delta = torch.zeros_like(scores)
    rescue_mask = torch.zeros_like(scores, dtype=torch.bool)
    demote_mask = torch.zeros_like(scores, dtype=torch.bool)

    target_score = _positive_target_score(cfg)
    can_rescue = (
        rescue_candidate_mask.bool()
        & (ious >= float(cfg.target_iou))
        & (gt_indices >= 0)
        & (scores < target_score)
    )
    required_delta = (target_score - scores).clamp(min=0.0)
    can_rescue = can_rescue & (required_delta > 0.0) & (required_delta <= float(cfg.max_score_delta))

    with torch.no_grad():
        for image_idx in torch.unique(image_indices.detach(), sorted=True):
            in_image = image_indices == image_idx
            image_candidates = torch.nonzero(in_image & can_rescue, as_tuple=False).flatten()
            if image_candidates.numel() == 0 or cfg.max_rescues_per_image == 0:
                continue

            per_gt = []
            for gt_idx in torch.unique(gt_indices[image_candidates].detach(), sorted=True):
                if int(gt_idx.item()) < 0:
                    continue
                gt_members = image_candidates[gt_indices[image_candidates] == gt_idx]
                if gt_members.numel() == 0:
                    continue
                member_energy = required_delta[gt_members] ** 2
                min_energy = torch.min(member_energy)
                tied = gt_members[member_energy == min_energy]
                if tied.numel() > 1:
                    best = tied[torch.argmax(ious[tied])]
                else:
                    best = tied[0]
                per_gt.append(best)

            if not per_gt:
                continue
            selected = torch.stack(per_gt)
            order_energy = required_delta[selected] ** 2
            order_iou = ious[selected]
            order_score = scores[selected]
            order = sorted(
                range(selected.numel()),
                key=lambda idx: (
                    float(order_energy[idx].item()),
                    -float(order_iou[idx].item()),
                    -float(order_score[idx].item()),
                ),
            )
            keep = [int(selected[idx].item()) for idx in order[: int(cfg.max_rescues_per_image)]]
            if keep:
                keep_tensor = torch.tensor(keep, dtype=torch.long, device=scores.device)
                rescue_mask[keep_tensor] = True

    score_delta[rescue_mask] = required_delta[rescue_mask]

    if cfg.demote_low_quality:
        demote_target = _demote_target_score(cfg)
        demote_required = (demote_target - scores).clamp(max=0.0)
        can_demote = (
            low_quality_mask.bool()
            & (scores >= float(cfg.score_threshold))
            & (demote_required.abs() <= float(cfg.max_score_delta))
        )
        demote_mask = can_demote & (~rescue_mask)
        score_delta[demote_mask] = demote_required[demote_mask]

    new_scores = (scores + score_delta).clamp(0.0, 1.0)
    energy = score_delta.square()
    summary = {
        "num_candidates": int(scores.numel()),
        "num_rescue_candidates": int(can_rescue.sum().item()),
        "num_rescued": int(rescue_mask.sum().item()),
        "num_demoted": int(demote_mask.sum().item()),
        "num_threshold_crossings": int(
            ((scores < float(cfg.score_threshold)) & (new_scores >= float(cfg.score_threshold))).sum().item()
        ),
        "num_low_quality_crossings": int(
            (
                low_quality_mask.bool()
                & (scores < float(cfg.score_threshold))
                & (new_scores >= float(cfg.score_threshold))
            ).sum().item()
        ),
        "mean_abs_score_delta": float(score_delta.abs().mean().item()) if score_delta.numel() else 0.0,
        "mean_action_energy": float(energy.mean().item()) if energy.numel() else 0.0,
        "total_action_energy": float(energy.sum().item()) if energy.numel() else 0.0,
        "target_score": float(target_score),
    }
    return ScoreActionSearchResult(
        score_delta=score_delta,
        rescue_mask=rescue_mask,
        demote_mask=demote_mask,
        summary=summary,
    )


def apply_score_action_to_prediction(
    prediction: dict,
    score_delta: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    relabel_mask: torch.Tensor | None = None,
) -> dict:
    """Apply residual score actions to a detection prediction dict."""
    scores = prediction.get("scores", torch.empty((0,), dtype=torch.float32))
    if score_delta.shape != scores.shape:
        raise ValueError("score_delta must share shape with prediction scores")

    output = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in prediction.items()
    }
    output["scores"] = (scores + score_delta.to(scores.device, dtype=scores.dtype)).clamp(0.0, 1.0)
    if labels is not None:
        if relabel_mask is None:
            raise ValueError("relabel_mask is required when labels are provided")
        if labels.shape != scores.shape or relabel_mask.shape != scores.shape:
            raise ValueError("labels and relabel_mask must share shape with scores")
        output["labels"][relabel_mask.bool()] = labels.to(output["labels"].device, dtype=output["labels"].dtype)[
            relabel_mask.bool()
        ]
    return output


def _positive_target_score(config: ActionSearchConfig) -> float:
    if config.positive_target_score is not None:
        return max(float(config.score_threshold), float(config.positive_target_score))
    return min(1.0, float(config.score_threshold) + float(config.threshold_margin))


def _demote_target_score(config: ActionSearchConfig) -> float:
    if config.demote_target_score is not None:
        return float(config.demote_target_score)
    return max(0.0, float(config.score_threshold) - float(config.threshold_margin))


def _check_vector(name: str, value: torch.Tensor) -> None:
    if value.ndim != 1:
        raise ValueError(f"{name} must be a 1-D tensor")
