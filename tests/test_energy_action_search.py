from __future__ import annotations

import torch

from spectral_detection_posttrain.methods.energy_transport import (
    ActionSearchConfig,
    apply_score_action_to_prediction,
    select_min_energy_score_actions,
)
from scripts.eval_energy_action_search import replace_or_append_selected_actions_to_base_prediction


def test_min_energy_search_rescues_smallest_delta_per_gt() -> None:
    scores = torch.tensor([0.01, 0.04, 0.03])
    ious = torch.tensor([0.80, 0.82, 0.20])
    image_indices = torch.zeros(3, dtype=torch.long)
    gt_indices = torch.tensor([0, 0, 1])
    cfg = ActionSearchConfig(
        score_threshold=0.05,
        threshold_margin=0.01,
        target_iou=0.75,
        max_score_delta=0.10,
        max_rescues_per_image=1,
    )

    result = select_min_energy_score_actions(
        scores,
        ious,
        image_indices,
        gt_indices=gt_indices,
        config=cfg,
    )

    assert result.rescue_mask.tolist() == [False, True, False]
    assert torch.allclose(result.score_delta, torch.tensor([0.0, 0.02, 0.0]))
    assert result.summary["num_rescued"] == 1
    assert result.summary["num_threshold_crossings"] == 1


def test_search_budget_is_applied_per_image() -> None:
    scores = torch.tensor([0.04, 0.04, 0.04, 0.04])
    ious = torch.tensor([0.80, 0.81, 0.82, 0.83])
    image_indices = torch.tensor([0, 0, 1, 1])
    gt_indices = torch.tensor([0, 1, 0, 1])
    cfg = ActionSearchConfig(max_rescues_per_image=1, threshold_margin=0.01)

    result = select_min_energy_score_actions(
        scores,
        ious,
        image_indices,
        gt_indices=gt_indices,
        config=cfg,
    )

    assert result.rescue_mask[:2].sum().item() == 1
    assert result.rescue_mask[2:].sum().item() == 1
    assert result.summary["num_rescued"] == 2


def test_search_does_not_cross_low_quality_candidates() -> None:
    scores = torch.tensor([0.04, 0.04])
    ious = torch.tensor([0.20, 0.90])
    image_indices = torch.zeros(2, dtype=torch.long)
    cfg = ActionSearchConfig(score_threshold=0.05, target_iou=0.75, threshold_margin=0.01)

    result = select_min_energy_score_actions(scores, ious, image_indices, config=cfg)

    assert result.rescue_mask.tolist() == [False, True]
    assert result.summary["num_low_quality_crossings"] == 0


def test_search_can_use_verifier_quality_instead_of_oracle_iou() -> None:
    scores = torch.tensor([0.04, 0.04])
    ious = torch.tensor([0.90, 0.90])
    verifier_quality = torch.tensor([0.20, 0.90])
    image_indices = torch.zeros(2, dtype=torch.long)
    cfg = ActionSearchConfig(score_threshold=0.05, target_iou=0.75, threshold_margin=0.01)

    result = select_min_energy_score_actions(
        scores,
        ious,
        image_indices,
        selection_quality=verifier_quality,
        config=cfg,
    )

    assert result.rescue_mask.tolist() == [False, True]
    assert result.summary["num_rescue_candidates"] == 1


def test_search_can_demote_low_quality_predictions() -> None:
    scores = torch.tensor([0.08, 0.04])
    ious = torch.tensor([0.10, 0.90])
    image_indices = torch.zeros(2, dtype=torch.long)
    cfg = ActionSearchConfig(
        score_threshold=0.05,
        threshold_margin=0.01,
        demote_low_quality=True,
        max_score_delta=0.10,
    )

    result = select_min_energy_score_actions(scores, ious, image_indices, config=cfg)

    assert result.demote_mask.tolist() == [True, False]
    assert torch.allclose(result.score_delta, torch.tensor([-0.04, 0.02]))
    assert result.summary["num_demoted"] == 1


def test_apply_score_action_to_prediction_clamps_and_optionally_relabels() -> None:
    prediction = {
        "boxes": torch.zeros(2, 4),
        "scores": torch.tensor([0.95, 0.03]),
        "labels": torch.tensor([1, 1]),
    }
    updated = apply_score_action_to_prediction(
        prediction,
        torch.tensor([0.10, 0.04]),
        labels=torch.tensor([2, 3]),
        relabel_mask=torch.tensor([False, True]),
    )

    assert torch.allclose(updated["scores"], torch.tensor([1.0, 0.07]))
    assert updated["labels"].tolist() == [1, 3]
    assert prediction["labels"].tolist() == [1, 1]


def test_replace_or_append_selected_actions_replaces_same_gt_low_iou_base() -> None:
    base = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "scores": torch.tensor([0.80]),
        "labels": torch.tensor([1]),
    }
    source = {
        "boxes": torch.tensor([[0.0, 0.0, 20.0, 20.0]]),
        "scores": torch.tensor([0.04]),
        "labels": torch.tensor([1]),
    }
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 20.0, 20.0]]),
        "labels": torch.tensor([1]),
    }

    updated = replace_or_append_selected_actions_to_base_prediction(
        base,
        source,
        target,
        score_delta=torch.tensor([0.02]),
        rescue_mask=torch.tensor([True]),
        source_gt_indices=torch.tensor([0]),
        target_iou=0.75,
    )

    assert updated["boxes"].shape[0] == 1
    assert torch.allclose(updated["boxes"][0], source["boxes"][0])
    assert torch.allclose(updated["scores"], torch.tensor([0.80]))


def test_replace_or_append_selected_actions_appends_when_no_base_match() -> None:
    base = {
        "boxes": torch.tensor([[40.0, 40.0, 60.0, 60.0]]),
        "scores": torch.tensor([0.70]),
        "labels": torch.tensor([1]),
    }
    source = {
        "boxes": torch.tensor([[0.0, 0.0, 20.0, 20.0]]),
        "scores": torch.tensor([0.04]),
        "labels": torch.tensor([1]),
    }
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 20.0, 20.0]]),
        "labels": torch.tensor([1]),
    }

    updated = replace_or_append_selected_actions_to_base_prediction(
        base,
        source,
        target,
        score_delta=torch.tensor([0.02]),
        rescue_mask=torch.tensor([True]),
        source_gt_indices=torch.tensor([0]),
        target_iou=0.75,
    )

    assert updated["boxes"].shape[0] == 2
    assert torch.allclose(updated["scores"], torch.tensor([0.70, 0.06]))
