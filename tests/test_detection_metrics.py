from __future__ import annotations

import pytest
import torch

from spectral_detection_posttrain.eval.detection_metrics import detection_ece, evaluate_detection_predictions, precision_at_recall


def test_precision_at_recall_uses_ranked_detections() -> None:
    scored = [(0.9, True), (0.8, False), (0.7, True), (0.6, True)]
    assert precision_at_recall(scored, total_gt=3, target_recall=2 / 3) == 3 / 4
    assert precision_at_recall(scored, total_gt=3, target_recall=1.0) == 3 / 4


def test_precision_at_recall_returns_none_when_unreachable() -> None:
    scored = [(0.9, True), (0.8, False)]
    assert precision_at_recall(scored, total_gt=3, target_recall=0.85) is None


def test_detection_ece_detects_overconfident_errors() -> None:
    calibrated = [(0.9, True), (0.8, True)]
    overconfident_wrong = [(0.9, False), (0.8, False)]
    assert detection_ece(overconfident_wrong) > detection_ece(calibrated)


def test_evaluate_detection_predictions_reports_calibration_metrics() -> None:
    predictions = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
            "labels": torch.tensor([1, 1]),
            "scores": torch.tensor([0.9, 0.8]),
        }
    ]
    targets = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])}]
    metrics = evaluate_detection_predictions(predictions, targets, iou_threshold=0.5, fixed_recall=0.85)
    assert metrics["precision_at_recall_0_85"] == 1.0
    assert metrics["high_conf_fp_count"] == 1
    assert metrics["ece"] is not None


def test_summarize_iou_diagnostics_reports_ap75_related_fields():
    from spectral_detection_posttrain.eval.detection_metrics import summarize_iou_diagnostics

    summary = summarize_iou_diagnostics(
        matched_ious=[0.9, 0.76, 0.4],
        matched_scores=[0.95, 0.7, 0.8],
    )

    assert summary["tp_iou_mean"] > 0.0
    assert summary["tp_iou_ge_075_rate"] == pytest.approx(2 / 3)
    assert "score_iou_corr" in summary
