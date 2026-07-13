"""Preference construction helpers for action-local detector transport."""

from __future__ import annotations

import torch

from spectral_detection_posttrain.methods.energy_transport.contracts import PreferenceBatch


def build_top_bottom_preferences(
    quality: torch.Tensor,
    ious: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    min_quality_margin: float = 0.0,
    min_iou_floor: float = 0.0,
    min_iou_gap: float = 0.0,
) -> PreferenceBatch:
    """Build one top-vs-bottom preference pair per group.

    Groups with fewer than two candidates are skipped.  Pairs that do not pass
    the margin/floor checks are returned with ``valid_mask=False`` so trainers
    can log why preference coverage is low instead of silently training on weak
    or tied pairs.
    """
    if quality.ndim != 1 or ious.ndim != 1 or group_ids.ndim != 1:
        raise ValueError("quality, ious, and group_ids must be 1-D tensors")
    if quality.shape != ious.shape or quality.shape != group_ids.shape:
        raise ValueError("quality, ious, and group_ids must share shape")
    if min_quality_margin < 0.0:
        raise ValueError("min_quality_margin must be non-negative")
    if min_iou_floor < 0.0:
        raise ValueError("min_iou_floor must be non-negative")
    if min_iou_gap < 0.0:
        raise ValueError("min_iou_gap must be non-negative")

    chosen: list[torch.Tensor] = []
    rejected: list[torch.Tensor] = []
    quality_gaps: list[torch.Tensor] = []
    iou_gaps: list[torch.Tensor] = []
    valid: list[torch.Tensor] = []

    with torch.no_grad():
        for group in torch.unique(group_ids.detach(), sorted=True):
            member_indices = torch.nonzero(group_ids == group, as_tuple=False).flatten()
            if member_indices.numel() < 2:
                continue

            member_quality = quality[member_indices]
            local_chosen = int(torch.argmax(member_quality).item())
            local_rejected = int(torch.argmin(member_quality).item())
            chosen_idx = member_indices[local_chosen]
            rejected_idx = member_indices[local_rejected]

            q_gap = quality[chosen_idx] - quality[rejected_idx]
            i_gap = ious[chosen_idx] - ious[rejected_idx]
            is_valid = (
                (q_gap >= float(min_quality_margin))
                & (ious[chosen_idx] >= float(min_iou_floor))
                & (i_gap >= float(min_iou_gap))
            )

            chosen.append(chosen_idx)
            rejected.append(rejected_idx)
            quality_gaps.append(q_gap)
            iou_gaps.append(i_gap)
            valid.append(is_valid)

    if not chosen:
        empty_long = torch.empty(0, dtype=torch.long, device=group_ids.device)
        empty_float = torch.empty(0, dtype=quality.dtype, device=quality.device)
        empty_bool = torch.empty(0, dtype=torch.bool, device=group_ids.device)
        return PreferenceBatch(
            chosen_indices=empty_long,
            rejected_indices=empty_long.clone(),
            quality_gap=empty_float,
            iou_gap=empty_float.clone(),
            valid_mask=empty_bool,
        )

    return PreferenceBatch(
        chosen_indices=torch.stack(chosen).long(),
        rejected_indices=torch.stack(rejected).long(),
        quality_gap=torch.stack(quality_gaps).to(dtype=quality.dtype),
        iou_gap=torch.stack(iou_gaps).to(dtype=ious.dtype),
        valid_mask=torch.stack(valid).bool(),
    )
