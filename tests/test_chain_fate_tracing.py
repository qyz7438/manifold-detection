import pytest
import torch

from scripts.trace_chain_candidate_fate import (
    candidate_fate_masks,
    classify_candidate_suppressors,
    summarize_candidate_fate,
)


def test_candidate_fate_masks_expose_per_candidate_transition_state():
    candidate_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    candidate_labels = torch.tensor([1, 1])
    candidate_scores = torch.tensor([0.9, 0.8])
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [21.0, 21.0, 31.0, 31.0]]),
        "labels": torch.tensor([1, 1]),
    }
    final_prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        "labels": torch.tensor([1, 1]),
        "scores": torch.tensor([0.9, 0.8]),
    }

    masks = candidate_fate_masks(
        candidate_boxes,
        candidate_labels,
        candidate_scores,
        target,
        final_prediction,
        score_threshold=0.05,
        candidate_to_final_iou=0.9,
        tp_iou_threshold=0.75,
    )

    assert masks["score_ge"].tolist() == [True, True]
    assert masks["entered"].tolist() == [True, True]
    assert masks["ap75_tp"].tolist() == [True, False]


def test_summarize_candidate_fate_counts_entered_and_ap75_tp():
    candidate_boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
            [40.0, 40.0, 50.0, 50.0],
        ]
    )
    candidate_labels = torch.tensor([1, 1, 1])
    candidate_scores = torch.tensor([0.9, 0.8, 0.7])
    target = {
        "boxes": torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [21.0, 21.0, 31.0, 31.0],
                [40.0, 40.0, 50.0, 50.0],
            ]
        ),
        "labels": torch.tensor([1, 1, 1]),
    }
    final_prediction = {
        "boxes": torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
                [80.0, 80.0, 90.0, 90.0],
            ]
        ),
        "labels": torch.tensor([1, 1, 1]),
        "scores": torch.tensor([0.9, 0.8, 0.7]),
    }

    report = summarize_candidate_fate(
        candidate_boxes,
        candidate_labels,
        candidate_scores,
        target,
        final_prediction,
        score_threshold=0.05,
        candidate_to_final_iou=0.9,
        tp_iou_threshold=0.75,
    )

    assert report["candidate_count"] == 3
    assert report["score_ge_threshold_count"] == 3
    assert report["entered_final_count"] == 2
    assert report["ap75_tp_count"] == 1
    assert report["entered_but_not_tp_count"] == 1
    assert report["score_ge_threshold_but_not_entered_count"] == 1
    assert report["entered_final_rate"] == pytest.approx(2 / 3)


def test_classify_candidate_suppressors_finds_worse_same_gt_duplicate():
    candidate_boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [30.0, 30.0, 36.0, 36.0],
        ]
    )
    candidate_labels = torch.tensor([1, 1])
    candidate_scores = torch.tensor([0.8, 0.8])
    candidate_gt_indices = torch.tensor([0, 1])
    target = {
        "boxes": torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [30.0, 30.0, 40.0, 40.0],
            ]
        ),
        "labels": torch.tensor([1, 1]),
    }
    final_prediction = {
        "boxes": torch.tensor([[1.0, 1.0, 11.0, 11.0]]),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.9]),
    }

    report = classify_candidate_suppressors(
        candidate_boxes,
        candidate_labels,
        candidate_scores,
        candidate_gt_indices,
        target,
        final_prediction,
        score_threshold=0.05,
        candidate_to_final_iou=0.9,
        tp_iou_threshold=0.75,
        nms_iou_threshold=0.5,
    )

    assert report["blocked_candidate_count"] == 2
    assert report["same_gt_worse_duplicate_count"] == 1
    assert report["not_decoded_close_enough_count"] == 1
    assert report["no_same_class_overlap_count"] == 1
