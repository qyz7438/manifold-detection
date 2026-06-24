import torch

from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.eval.rescue_oracle import (
    apply_detection_score_oracle,
    matched_gt_indices,
    unmatched_gt_candidate_mask,
)


def test_unmatched_gt_candidate_mask_filters_duplicate_candidates():
    prediction = {
        "boxes": torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
            ]
        ),
        "labels": torch.tensor([1, 2, 2]),
        "scores": torch.tensor([0.9, 0.01, 0.01]),
    }
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        "labels": torch.tensor([1, 2]),
    }
    candidate_gt = torch.tensor([0, 0, 1])
    candidates = torch.tensor([False, True, True])

    mask = unmatched_gt_candidate_mask(
        prediction,
        target,
        candidate_gt,
        candidates,
        iou_threshold=0.75,
        score_threshold=0.05,
    )

    assert matched_gt_indices(prediction, target, iou_threshold=0.75, score_threshold=0.05) == {0}
    assert mask.tolist() == [False, False, True]


def test_detection_score_oracle_can_turn_low_score_candidate_into_ap75_tp():
    prediction = {
        "boxes": torch.tensor([[20.0, 20.0, 30.0, 30.0]]),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.01]),
    }
    target = {
        "boxes": torch.tensor([[20.0, 20.0, 30.0, 30.0]]),
        "labels": torch.tensor([2]),
    }

    before = evaluate_detection_predictions([prediction], [target], iou_threshold=0.5, score_threshold=0.05)
    oracle = apply_detection_score_oracle(
        prediction,
        indices=torch.tensor([0]),
        labels=torch.tensor([2]),
        score=0.2,
    )
    after = evaluate_detection_predictions([oracle], [target], iou_threshold=0.5, score_threshold=0.05)

    assert before["ap75"] == 0.0
    assert after["ap75"] == 1.0
