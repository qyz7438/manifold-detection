import torch

from spectral_detection_posttrain.matching.box_iou import box_iou
from spectral_detection_posttrain.matching.pred_gt_matcher import match_predictions_to_gt


def test_box_iou_identity_and_no_overlap():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    same = box_iou(boxes, boxes)
    other = box_iou(boxes, torch.tensor([[20.0, 20.0, 30.0, 30.0]]))
    assert same.item() == 1.0
    assert other.item() == 0.0


def test_match_predictions_to_gt_matches_same_class_high_iou():
    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        "labels": torch.tensor([1, 1]),
        "scores": torch.tensor([0.9, 0.8]),
    }
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 11.0, 11.0]]),
        "labels": torch.tensor([1]),
    }
    result = match_predictions_to_gt(prediction, target, iou_threshold=0.5, score_threshold=0.05)
    assert len(result["matches"]) == 1
    assert result["matches"][0]["pred_index"] == 0
    assert result["unmatched_predictions"] == [1]
    assert result["unmatched_gt"] == []
