from __future__ import annotations

import torch

from spectral_detection_posttrain.core.matching.pred_gt_matcher import match_predictions_to_gt


def matched_gt_indices(
    prediction: dict,
    target: dict,
    *,
    iou_threshold: float = 0.75,
    score_threshold: float = 0.05,
) -> set[int]:
    matched = match_predictions_to_gt(
        prediction,
        target,
        iou_threshold=float(iou_threshold),
        score_threshold=float(score_threshold),
    )
    return {int(item["gt_index"]) for item in matched["matches"]}


def unmatched_gt_candidate_mask(
    prediction: dict,
    target: dict,
    candidate_gt_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    iou_threshold: float = 0.75,
    score_threshold: float = 0.05,
) -> torch.Tensor:
    matched_gt = matched_gt_indices(
        prediction,
        target,
        iou_threshold=float(iou_threshold),
        score_threshold=float(score_threshold),
    )
    candidate_gt_indices = candidate_gt_indices.long()
    candidate_mask = candidate_mask.to(candidate_gt_indices.device).bool()
    if not matched_gt:
        return candidate_mask.clone()
    matched_tensor = torch.tensor(sorted(matched_gt), dtype=torch.long, device=candidate_gt_indices.device)
    already_matched = (candidate_gt_indices.unsqueeze(1) == matched_tensor.unsqueeze(0)).any(dim=1)
    return candidate_mask & (~already_matched)


def apply_detection_score_oracle(
    prediction: dict,
    *,
    indices: torch.Tensor,
    labels: torch.Tensor,
    score: float,
) -> dict:
    out = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in prediction.items()
    }
    if indices.numel() == 0:
        return out
    indices = indices.detach().cpu().long()
    labels = labels.detach().cpu().long()
    out["scores"][indices] = float(score)
    out["labels"][indices] = labels
    return out
