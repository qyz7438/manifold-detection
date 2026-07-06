from __future__ import annotations

import pytest
import torch

from spectral_detection_posttrain.methods.energy_transport import (
    ActionOutcome,
    ConstraintConfig,
    PreferenceBatch,
    ROIActionState,
    apply_box_delta,
    build_top_bottom_preferences,
    clip_boxes_to_image,
)


def test_roi_action_state_validates_aligned_batch_dimensions() -> None:
    features = torch.randn(3, 8)
    state = ROIActionState(
        features=features,
        boxes=torch.zeros(3, 4),
        scores=torch.tensor([0.1, 0.2, 0.3]),
        labels=torch.tensor([1, 1, 2]),
        image_indices=torch.tensor([0, 0, 1]),
        proposal_indices=torch.tensor([0, 1, 0]),
    )

    assert state.batch_size == 3
    assert state.feature_dim == 8

    with pytest.raises(ValueError, match="scores"):
        ROIActionState(
            features=features,
            boxes=torch.zeros(3, 4),
            scores=torch.tensor([0.1, 0.2]),
            labels=torch.tensor([1, 1, 2]),
            image_indices=torch.tensor([0, 0, 1]),
            proposal_indices=torch.tensor([0, 1, 0]),
        )


def test_constraint_config_rejects_invalid_thresholds_and_budgets() -> None:
    cfg = ConstraintConfig(score_threshold=0.05, max_rescues_per_image=2)
    assert cfg.score_threshold == 0.05

    with pytest.raises(ValueError, match="score_threshold"):
        ConstraintConfig(score_threshold=1.2)
    with pytest.raises(ValueError, match="max_rescues_per_image"):
        ConstraintConfig(max_rescues_per_image=-1)


def test_clip_boxes_to_image_clamps_and_preserves_valid_order() -> None:
    boxes = torch.tensor([
        [-5.0, -1.0, 12.0, 20.0],
        [8.0, 8.0, 2.0, 2.0],
    ])

    clipped = clip_boxes_to_image(boxes, image_size=(10, 15))

    assert torch.all(clipped[:, 0::2] >= 0.0)
    assert torch.all(clipped[:, 1::2] >= 0.0)
    assert torch.all(clipped[:, 0::2] <= 15.0)
    assert torch.all(clipped[:, 1::2] <= 10.0)
    assert torch.all(clipped[:, 2] >= clipped[:, 0])
    assert torch.all(clipped[:, 3] >= clipped[:, 1])


def test_apply_box_delta_decodes_standard_center_size_action() -> None:
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    deltas = torch.tensor([[0.1, 0.0, 0.0, 0.0]])

    decoded = apply_box_delta(boxes, deltas, image_size=(20, 20))

    assert torch.allclose(decoded, torch.tensor([[1.0, 0.0, 11.0, 10.0]]), atol=1e-5)


def test_action_outcome_records_threshold_crossings() -> None:
    outcome = ActionOutcome(
        boxes=torch.zeros(3, 4),
        scores=torch.tensor([0.04, 0.08, 0.03]),
        old_scores=torch.tensor([0.02, 0.04, 0.10]),
        labels=torch.tensor([1, 1, 2]),
        threshold=0.05,
    )

    assert outcome.threshold_crossings.tolist() == [False, True, False]


def test_preference_batch_requires_valid_margin_and_matching_shapes() -> None:
    batch = PreferenceBatch(
        chosen_indices=torch.tensor([0, 2]),
        rejected_indices=torch.tensor([1, 3]),
        quality_gap=torch.tensor([0.2, 0.3]),
        iou_gap=torch.tensor([0.1, 0.2]),
        valid_mask=torch.tensor([True, False]),
    )
    assert batch.num_valid == 1

    with pytest.raises(ValueError, match="share shape"):
        PreferenceBatch(
            chosen_indices=torch.tensor([0]),
            rejected_indices=torch.tensor([1, 2]),
            quality_gap=torch.tensor([0.2]),
            iou_gap=torch.tensor([0.1]),
            valid_mask=torch.tensor([True]),
        )


def test_build_top_bottom_preferences_uses_margin_and_iou_floor() -> None:
    quality = torch.tensor([0.9, 0.2, 0.70, 0.69, 0.4])
    ious = torch.tensor([0.8, 0.1, 0.6, 0.5, 0.9])
    group_ids = torch.tensor([0, 0, 1, 1, 2])

    batch = build_top_bottom_preferences(
        quality,
        ious,
        group_ids,
        min_quality_margin=0.1,
        min_iou_floor=0.5,
    )

    assert batch.chosen_indices.tolist() == [0, 2]
    assert batch.rejected_indices.tolist() == [1, 3]
    assert batch.valid_mask.tolist() == [True, False]
    assert batch.num_valid == 1


def test_build_top_bottom_preferences_returns_empty_when_no_group_has_pair() -> None:
    quality = torch.tensor([0.9, 0.8])
    ious = torch.tensor([0.7, 0.6])
    group_ids = torch.tensor([0, 1])

    batch = build_top_bottom_preferences(quality, ious, group_ids)

    assert batch.chosen_indices.numel() == 0
    assert batch.num_valid == 0
