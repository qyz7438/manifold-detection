import torch

from spectral_detection_posttrain.analysis.raw_ifft_features import (
    crop_and_resize_boxes,
    penn_fudan_legacy_ifft_metric_bank,
    raw_ifft_feature_summary,
)


def test_crop_and_resize_boxes_returns_fixed_size_crops_for_each_box():
    image = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8)
    boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0], [2.0, 2.0, 7.0, 7.0]])

    crops = crop_and_resize_boxes(image, boxes, crop_size=16)

    assert crops.shape == (2, 3, 16, 16)
    assert torch.isfinite(crops).all()


def test_raw_ifft_feature_summary_reports_expected_scalar_groups():
    crop = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    crop[:, :, 8:24, 8:24] = 1.0

    summary = raw_ifft_feature_summary(crop)

    assert summary.shape == (1, 12)
    assert torch.isfinite(summary).all()
    assert summary[0, 0] > 0.0


def test_penn_fudan_legacy_ifft_metric_bank_keeps_legacy_scalar_count():
    crop = torch.zeros((2, 3, 32, 32), dtype=torch.float32)
    crop[:, :, 8:24, 8:24] = 1.0

    metrics = penn_fudan_legacy_ifft_metric_bank(crop)

    assert metrics.shape == (2, 23)
    assert torch.isfinite(metrics).all()
